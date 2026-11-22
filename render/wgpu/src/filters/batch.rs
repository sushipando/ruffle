//! Filters of many cached bitmaps, applied together, whatever their parameters.
//!
//! `layer_atlas::apply_filters_batched` applies *identical* filters to many bitmaps at
//! once, by applying them to an atlas. Here, every bitmap has its own filters: the
//! bitmaps are placed in an atlas (with enough transparent space around them) — level-0
//! simple cached bitmaps are drawn directly into it — and each step of their filter
//! lists is done for all of them in a few passes, one quad per bitmap with its own
//! parameters. Downscales are shared across all blur groups, and all blurs of a step
//! share a single tightly-packed horizontal/vertical blur atlas.
//!
//! Only Blur, Glow (as drop shadows, without an offset) and ColorMatrix are supported,
//! and only while rendering for display, since the results aren't bit-exact (see
//! `BlurFilter::apply_merged`).

use crate::buffer_pool::{AlwaysCompatible, PoolEntry, PooledTexture, TexturePool};
use crate::descriptors::Descriptors;
use crate::dynamic_transforms::DynamicTransforms;
use crate::filters::blur::{KERNEL_UNIFORM_SIZE, downscale_factors, merged_kernel};
use crate::filters::glow::GlowUniform;
use crate::filters::{FilterVertex, FilterVertexWithBlur};
use crate::mesh::Mesh;
use crate::surface::layer_atlas::{self, LAYER_TEXTURE_USAGE};
use ruffle_render::backend::CacheVolatility;
use ruffle_render::bitmap::BitmapHandle;
use ruffle_render::commands::CommandList;
use ruffle_render::filters::Filter;
use ruffle_render::quality::StageQuality;
use std::collections::{BTreeMap, HashMap};
use swf::{BlurFilter as BlurFilterArgs, Rectangle, Twips};
use wgpu::util::StagingBelt;
use wgpu_profiler::Scope;

/// Region positions and sizes are multiples of this, so that they scale down exactly
/// (blurs are scaled down by at most 8 in each direction).
const ALIGN: u32 = 8;

/// An entry to be filtered by `apply_together`.
pub struct FilterTogetherEntry {
    pub texture: wgpu::Texture,
    /// Used sub-region `(width, height)` inside `texture` (excluding `CacheSlack` padding).
    pub used_size: (u32, u32),
    pub filters: Vec<Filter>,
    /// If `Some`, the entry hasn't been drawn into `texture` yet and can be drawn
    /// directly into the filter atlas, avoiding a separate layer atlas pass and two
    /// texture copies.
    pub direct_commands: Option<CommandList>,
    /// Kept alive in tick-scoped `FilterAtlasResult` so `handle_ptr` can never dangle.
    pub handle: BitmapHandle,
    /// Raw pointer identity of the entry's `BitmapHandle` (`Arc::as_ptr(&handle.0) as usize`).
    pub handle_ptr: usize,
    /// Persistence policy for this filtered entry.
    pub volatility: CacheVolatility,
}

/// Filtered CAB atlases kept alive for direct sampling during the current frame (and, for
/// `CacheVolatility::Tick`, the remaining sub-frames of the current movie tick), avoiding
/// per-CAB copy-out blits and per-CAB bind groups for animated CABs.
#[derive(Default)]
pub struct FilterAtlasResult {
    textures: Vec<PoolEntry<PooledTexture, AlwaysCompatible>>,
    pub bind_groups: Vec<wgpu::BindGroup>,
    /// Maps `BitmapHandle` pointer -> `(atlas_bind_group_index, slot_x, slot_y, atlas_w, atlas_h)`.
    pub slots: fnv::FnvHashMap<usize, (usize, u32, u32, u32, u32)>,
    handles: fnv::FnvHashMap<usize, BitmapHandle>,
    subframe_keys: Vec<usize>,
}

impl FilterAtlasResult {
    pub fn clear(&mut self) {
        self.slots.clear();
        self.handles.clear();
        self.subframe_keys.clear();
        self.bind_groups.clear();
        self.textures.clear();
    }

    pub fn remove_slot(&mut self, handle_ptr: usize) {
        self.slots.remove(&handle_ptr);
        self.handles.remove(&handle_ptr);
    }

    /// Drops single-subframe slots and any atlas textures no longer referenced by live
    /// tick-scoped slots, returning those textures to the pool immediately.
    pub fn end_frame(&mut self) {
        for key in self.subframe_keys.drain(..) {
            self.slots.remove(&key);
            self.handles.remove(&key);
        }
        if self.slots.is_empty() {
            self.clear();
            return;
        }
        let mut used = vec![false; self.bind_groups.len()];
        for &(bg_idx, _, _, _, _) in self.slots.values() {
            if bg_idx < used.len() {
                used[bg_idx] = true;
            }
        }
        if used.iter().all(|&u| u) {
            return;
        }
        let mut remap = vec![0usize; used.len()];
        let mut new_textures = Vec::with_capacity(self.textures.len());
        let mut new_bind_groups = Vec::with_capacity(self.bind_groups.len());
        for (old_idx, (tex, bg)) in self
            .textures
            .drain(..)
            .zip(self.bind_groups.drain(..))
            .enumerate()
        {
            if used[old_idx] {
                remap[old_idx] = new_bind_groups.len();
                new_textures.push(tex);
                new_bind_groups.push(bg);
            }
        }
        self.textures = new_textures;
        self.bind_groups = new_bind_groups;
        for slot in self.slots.values_mut() {
            slot.0 = remap[slot.0];
        }
    }
}

/// The kernels and scale of a blur (see `BlurFilter::apply`), if it can be merged.
struct BlurPlan {
    factors: (u32, u32),
    /// The kernels at the lower resolution, horizontal and vertical.
    kernels: [Vec<[f32; 2]>; 2],
}

fn blur_plan(filter: &BlurFilterArgs) -> Option<BlurPlan> {
    let factors = downscale_factors(filter);
    let kernel = |strength: f32, factor: u32| {
        let strength = strength / factor as f32;
        if strength.min(255.0) <= 1.0 || filter.num_passes() == 0 {
            // A no-op in this direction.
            Some(vec![[0.0, 1.0]])
        } else {
            merged_kernel(strength, filter.num_passes())
        }
    };
    Some(BlurPlan {
        factors,
        kernels: [
            kernel(filter.blur_x.to_f32(), factors.0)?,
            kernel(filter.blur_y.to_f32(), factors.1)?,
        ],
    })
}

/// Whether `apply_together` can apply this filter.
pub fn can_apply_together(filter: &Filter) -> bool {
    match filter {
        Filter::BlurFilter(filter) => blur_plan(filter).is_some(),
        Filter::GlowFilter(filter) => blur_plan(&filter.inner_blur_filter()).is_some(),
        Filter::ColorMatrixFilter(_) => true,
        _ => false,
    }
}

/// The bitmaps (by index, with their regions and gutters) in one atlas, and its used size.
type Batch = (Vec<(usize, Region, u32)>, u32, u32);

/// Where a bitmap is in an atlas: `(x, y, width, height)`.
#[derive(Clone, Copy)]
struct Region {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

impl Region {
    /// The region's corners in a texture of the given size, normalized (0 to 1).
    fn normalized(self, size: (u32, u32)) -> [[f32; 2]; 4] {
        let (left, top) = (self.x as f32 / size.0 as f32, self.y as f32 / size.1 as f32);
        let right = (self.x + self.width) as f32 / size.0 as f32;
        let bottom = (self.y + self.height) as f32 / size.1 as f32;
        [[left, top], [right, top], [right, bottom], [left, bottom]]
    }
}

fn quad(target: [[f32; 2]; 4], source: [[f32; 2]; 4]) -> [FilterVertex; 4] {
    std::array::from_fn(|i| FilterVertex {
        position: target[i],
        uv: source[i],
    })
}

/// Parent downscale factor in the shared pyramid: halves whichever axis is larger (both
/// when equal), so asymmetric factors like `(4, 2)` reuse the shared `(2, 2)` level.
fn parent_scale(scale: (u32, u32)) -> (u32, u32) {
    let (sx, sy) = scale;
    let nsx = if sx >= sy { (sx / 2).max(1) } else { sx };
    let nsy = if sy >= sx { (sy / 2).max(1) } else { sy };
    (nsx, nsy)
}

/// Collects per-quad uniforms at aligned offsets of one buffer, deduplicating identical
/// payloads so consecutive quads sharing the same uniform can be drawn in one call.
struct Uniforms {
    bytes: Vec<u8>,
    alignment: usize,
    dedup: HashMap<Vec<u8>, u32>,
}

impl Uniforms {
    fn new(descriptors: &Descriptors) -> Self {
        Self {
            bytes: vec![],
            alignment: descriptors.limits.min_uniform_buffer_offset_alignment as usize,
            dedup: HashMap::new(),
        }
    }

    /// Adds `data`, reserving at least `size` bytes, and returns its offset (reusing an
    /// existing offset if identical bytes were already pushed).
    fn push(&mut self, data: &[u8], size: usize) -> u32 {
        if let Some(&offset) = self.dedup.get(data) {
            return offset;
        }
        let offset = self.bytes.len().next_multiple_of(self.alignment);
        self.bytes.resize(offset + size.max(data.len()), 0);
        self.bytes[offset..offset + data.len()].copy_from_slice(data);
        let offset = offset as u32;
        self.dedup.insert(data.to_vec(), offset);
        offset
    }

    fn buffer(
        &self,
        scratch: &mut Scratch<'_>,
        encoder: &mut wgpu::CommandEncoder,
        staging_belt: &mut StagingBelt,
    ) -> wgpu::Buffer {
        scratch.upload(encoder, staging_belt, &self.bytes)
    }
}

/// The bytes of `struct Kernel` in `blur.wgsl`, up to the last pair of taps.
fn kernel_uniform(direction: [f32; 2], taps: &[[f32; 2]]) -> Vec<u8> {
    let mut uniform = vec![0.0f32; 4 + taps.len().div_ceil(2) * 4];
    uniform[0] = direction[0];
    uniform[1] = direction[1];
    uniform[2] = f32::from_bits(taps.len() as u32);
    for (index, tap) in taps.iter().enumerate() {
        uniform[4 + index * 2..][..2].copy_from_slice(tap);
    }
    bytemuck::cast_slice(&uniform).to_vec()
}

/// Uploads uniforms and vertices into buffers reused across passes and frames (creating
/// buffers for every pass was ~13% of render time when drawing 144 frames per second).
/// The uploads are copies recorded in the command encoder, so a buffer can be reused as
/// soon as the passes that use it are recorded.
pub struct ScratchBuffers {
    free: std::sync::Mutex<Vec<wgpu::Buffer>>,
}

impl ScratchBuffers {
    pub fn new() -> Self {
        Self {
            free: Default::default(),
        }
    }
}

/// The scratch buffers used by one `apply_together`, returned to the pool when dropped.
struct Scratch<'a> {
    descriptors: &'a Descriptors,
    taken: Vec<wgpu::Buffer>,
}

impl Scratch<'_> {
    fn upload(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        staging_belt: &mut StagingBelt,
        bytes: &[u8],
    ) -> wgpu::Buffer {
        let size = (bytes.len().max(16) as u64).next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT);
        let buffer = {
            let mut free = self
                .descriptors
                .filters
                .scratch_buffers
                .free
                .lock()
                .expect("scratch buffers lock");
            let best = free
                .iter()
                .enumerate()
                .filter(|(_, buffer)| buffer.size() >= size)
                .min_by_key(|(_, buffer)| buffer.size())
                .map(|(index, _)| index);
            match best {
                Some(index) => free.swap_remove(index),
                None => self
                    .descriptors
                    .device
                    .create_buffer(&wgpu::BufferDescriptor {
                        label: create_debug_label!("Filter batch scratch").as_deref(),
                        size: size.next_power_of_two().max(4096),
                        usage: wgpu::BufferUsages::UNIFORM
                            | wgpu::BufferUsages::VERTEX
                            | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }),
            }
        };
        let mut padded = bytes.to_vec();
        padded.resize(size as usize, 0);
        staging_belt
            .write_buffer(
                encoder,
                &buffer,
                0,
                wgpu::BufferSize::new(size).expect("not zero"),
            )
            .copy_from_slice(&padded);
        self.taken.push(buffer.clone());
        buffer
    }

    fn vertices<T: bytemuck::Pod>(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        staging_belt: &mut StagingBelt,
        vertices: &[T],
    ) -> wgpu::Buffer {
        self.upload(encoder, staging_belt, bytemuck::cast_slice(vertices))
    }
}

impl Drop for Scratch<'_> {
    fn drop(&mut self) {
        self.descriptors
            .filters
            .scratch_buffers
            .free
            .lock()
            .expect("scratch buffers lock")
            .append(&mut self.taken);
    }
}

fn clear_pass<'a>(
    encoder: &'a mut wgpu::CommandEncoder,
    view: &'a wgpu::TextureView,
) -> wgpu::RenderPass<'a> {
    crate::perf_counters::count(crate::perf_counters::Counter::RenderPass);
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: create_debug_label!("Filter batch").as_deref(),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            },
            depth_slice: None,
        })],
        ..Default::default()
    })
}

/// Draws consecutive runs of quads sharing the same dynamic uniform offset in a single
/// `draw_indexed` call using `descriptors.quad.batch_quad_indices`.
fn draw_batched_quads(
    render_pass: &mut wgpu::RenderPass<'_>,
    bind_group: &wgpu::BindGroup,
    offsets: &[u32],
) {
    let mut i = 0;
    while i < offsets.len() {
        let offset = offsets[i];
        let mut end = i + 1;
        while end < offsets.len()
            && offsets[end] == offset
            && end - i < crate::descriptors::MAX_BATCH_QUADS
        {
            end += 1;
        }
        let count = (end - i) as u32;
        render_pass.set_bind_group(0, bind_group, &[offset]);
        render_pass.draw_indexed(0..(count * 6), i as i32 * 4, 0..1);
        i = end;
    }
}

/// Draws quads (four vertices each, from `vertices`) with the dynamic kernel pipeline,
/// each with its own kernel at the given offset of `uniforms`.
fn draw_kernel_quads(
    descriptors: &Descriptors,
    render_pass: &mut wgpu::RenderPass<'_>,
    source: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    uniforms: &wgpu::Buffer,
    vertices: &wgpu::Buffer,
    offsets: &[u32],
) {
    let blur = &descriptors.filters.blur;
    let bind_group = descriptors
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: create_debug_label!("Filter batch kernel binds").as_deref(),
            layout: &blur.dynamic_kernel_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(source),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: uniforms,
                        offset: 0,
                        size: wgpu::BufferSize::new(KERNEL_UNIFORM_SIZE),
                    }),
                },
            ],
        });
    render_pass.set_pipeline(blur.dynamic_kernel_pipeline(descriptors));
    render_pass.set_vertex_buffer(0, vertices.slice(..));
    render_pass.set_index_buffer(
        descriptors.quad.batch_quad_indices.slice(..),
        wgpu::IndexFormat::Uint32,
    );
    draw_batched_quads(render_pass, &bind_group, offsets);
}

/// A texture that holds a whole atlas (or a scaled-down version of it).
struct AtlasTexture {
    texture: crate::buffer_pool::PoolEntry<
        crate::buffer_pool::PooledTexture,
        crate::buffer_pool::AlwaysCompatible,
    >,
    size: (u32, u32),
}

fn atlas_texture(
    descriptors: &Descriptors,
    texture_pool: &mut TexturePool,
    size: (u32, u32),
) -> AtlasTexture {
    AtlasTexture {
        texture: texture_pool.get_texture(
            descriptors,
            wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            LAYER_TEXTURE_USAGE,
            wgpu::TextureFormat::Rgba8Unorm,
            1,
        ),
        size,
    }
}

/// A bitmap blurred in one step of `apply_together`.
struct BlurItem {
    batch_index: usize,
    region: Region,
    gutter: u32,
    plan: BlurPlan,
}

/// Applies each entry's filters to its texture, in place (all of which must be supported,
/// see `can_apply_together`). Entries with `direct_commands` are drawn straight into the
/// filter atlas instead of copied in. Returns the indices of entries that were too large,
/// which the caller must draw (if `direct_commands` is still `Some`) and filter itself.
#[expect(clippy::too_many_arguments)]
pub fn apply_together(
    entries: &mut [FilterTogetherEntry],
    descriptors: &Descriptors,
    meshes: &Vec<Mesh>,
    encoder: &mut Scope<'_, wgpu::CommandEncoder>,
    staging_belt: &mut StagingBelt,
    dynamic_transforms: &DynamicTransforms,
    texture_pool: &mut TexturePool,
    quality: StageQuality,
    max_size: u32,
    direct_out: &mut FilterAtlasResult,
) -> Vec<usize> {
    let mut scratch = Scratch {
        descriptors,
        taken: vec![],
    };
    // Shelf-pack the textures by their used sub-region (excluding `CacheSlack` dead
    // padding), each with transparent space around it as wide as its filters can spread.
    let used_extent = |entry: &FilterTogetherEntry| -> (u32, u32) {
        (
            entry.used_size.0.min(entry.texture.width()).max(1),
            entry.used_size.1.min(entry.texture.height()).max(1),
        )
    };
    let mut order: Vec<usize> = (0..entries.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(used_extent(&entries[i]).1));
    let mut batches: Vec<Batch> = vec![];
    let mut batch = vec![];
    let mut skipped = vec![];
    let (mut x, mut y, mut row_height, mut used_width) = (0, 0, 0, 0);
    for index in order {
        let entry = &entries[index];
        let growth = entry
            .filters
            .iter()
            .fold(Rectangle::<Twips>::ZERO, |rect, filter| {
                filter.calculate_dest_rect(rect)
            });
        let reach = [
            -growth.x_min.to_pixels(),
            -growth.y_min.to_pixels(),
            growth.x_max.to_pixels(),
            growth.y_max.to_pixels(),
        ]
        .into_iter()
        .fold(0.0f64, f64::max)
        .ceil() as u32;
        let gutter = (reach + 2).next_multiple_of(ALIGN);
        let (uw, uh) = used_extent(entry);
        let width = uw.next_multiple_of(ALIGN);
        let height = uh.next_multiple_of(ALIGN);
        let (w, h) = (width + gutter * 2, height + gutter * 2);
        if w > max_size || h > max_size {
            skipped.push(index);
            continue;
        }
        if x + w > max_size {
            (x, y, row_height) = (0, y + row_height, 0);
        }
        if y + h > max_size {
            batches.push((std::mem::take(&mut batch), used_width, y + row_height));
            (x, y, row_height, used_width) = (0, 0, 0, 0);
        }
        let region = Region {
            x: x + gutter,
            y: y + gutter,
            width,
            height,
        };
        batch.push((index, region, gutter));
        x += w;
        used_width = used_width.max(x);
        row_height = row_height.max(h);
    }
    if !batch.is_empty() {
        batches.push((batch, used_width, y + row_height));
    }

    let nearest = descriptors.bitmap_samplers.get_sampler(false, false);
    let linear = descriptors.bitmap_samplers.get_sampler(false, true);
    for (batch, width, height) in batches {
        let batch_len = batch.len() as u64;
        let size = (
            width.next_multiple_of(256).min(max_size),
            height.next_multiple_of(256).min(max_size),
        );
        let current_init = atlas_texture(descriptors, texture_pool, size);

        let mut direct_layers = vec![];
        let mut copy_in = vec![];
        for &(index, region, _) in &batch {
            let (uw, uh) = used_extent(&entries[index]);
            if let Some(commands) = entries[index].direct_commands.take() {
                direct_layers.push((
                    commands,
                    (region.x as i32, region.y as i32),
                    (region.x, region.y, uw, uh),
                ));
            } else {
                copy_in.push((index, region, uw, uh));
            }
        }
        if !direct_layers.is_empty() {
            let prior_atlas = (!direct_out.slots.is_empty()).then_some(&*direct_out);
            layer_atlas::render_into_atlas(
                direct_layers.into_iter(),
                &current_init.texture.0,
                descriptors,
                meshes,
                staging_belt,
                dynamic_transforms,
                encoder,
                texture_pool,
                quality,
                prior_atlas,
            );
        } else {
            drop(clear_pass(encoder, &current_init.texture.1));
        }
        for (index, region, uw, uh) in copy_in {
            let texture = &entries[index].texture;
            encoder.copy_texture_to_texture(
                texture.as_image_copy(),
                wgpu::TexelCopyTextureInfo {
                    texture: &current_init.texture.0,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: region.x,
                        y: region.y,
                        z: 0,
                    },
                    aspect: Default::default(),
                },
                wgpu::Extent3d {
                    width: uw,
                    height: uh,
                    depth_or_array_layers: 1,
                },
            );
        }

        let steps = batch
            .iter()
            .map(|&(index, _, _)| entries[index].filters.len())
            .max()
            .unwrap_or(0);
        let mut current_owned: Option<AtlasTexture> = Some(current_init);
        let mut current_direct: Option<(usize, (u32, u32))> = None;
        let mut active = batch;
        for step in 0..steps {
            let (current_view, current_size) = match (&current_owned, current_direct) {
                (Some(cur), _) => (&cur.texture.1, cur.size),
                (None, Some((idx, sz))) => (&direct_out.textures[idx].1, sz),
                _ => unreachable!("current atlas is always live"),
            };

            // Collect all blurs (including those of glows) for this step.
            let mut blur_items: Vec<BlurItem> = vec![];
            for (batch_index, &(index, region, gutter)) in active.iter().enumerate() {
                let blur = match entries[index].filters.get(step) {
                    Some(Filter::BlurFilter(filter)) => filter.clone(),
                    Some(Filter::GlowFilter(filter)) => filter.inner_blur_filter(),
                    _ => continue,
                };
                let plan = blur_plan(&blur).expect("checked by can_apply_together");
                blur_items.push(BlurItem {
                    batch_index,
                    region,
                    gutter,
                    plan,
                });
            }

            // The blurred (scaled-down) result of each bitmap blurred in this step:
            // index into `blurred_textures`, and its slot in that tightly-packed atlas.
            let mut blurred: Vec<Option<(usize, Region)>> = vec![None; active.len()];
            let mut blurred_textures: Vec<AtlasTexture> = vec![];

            if !blur_items.is_empty() {
                // 1. Tightly-packed shared downscale pyramid: only allocate space for the
                //    bitmaps that actually need each intermediate scale, instead of scaling
                //    down the entire full-resolution atlas.
                let mut scale_regions: HashMap<(u32, u32), Vec<(usize, Region, u32)>> =
                    HashMap::new();
                for item in &blur_items {
                    let mut s = item.plan.factors;
                    while s != (1, 1) {
                        scale_regions.entry(s).or_default().push((
                            item.batch_index,
                            item.region,
                            item.gutter,
                        ));
                        s = parent_scale(s);
                    }
                }
                let mut scales: Vec<(u32, u32)> = scale_regions.keys().copied().collect();
                scales.sort_by_key(|&(sx, sy)| (sx.max(sy), sx * sy, sx, sy));

                let mut downscaled: HashMap<(u32, u32), (AtlasTexture, HashMap<usize, Region>)> =
                    HashMap::new();
                for scale in scales {
                    let parent = parent_scale(scale);
                    let mut regions = scale_regions.remove(&scale).unwrap_or_default();
                    regions.sort_by_key(|&(bi, _, _)| bi);
                    regions.dedup_by_key(|(bi, _, _)| *bi);
                    regions.sort_by_key(|&(_, region, gutter)| {
                        let pad_y = (gutter / scale.1).max(2);
                        std::cmp::Reverse(region.height / scale.1 + pad_y * 2)
                    });

                    let mut packed: Vec<(usize, Region, Region)> =
                        Vec::with_capacity(regions.len());
                    let mut packed_map: HashMap<usize, Region> =
                        HashMap::with_capacity(regions.len());
                    let (mut dx, mut dy, mut d_row_h, mut d_used_w) = (0u32, 0u32, 0u32, 0u32);
                    for (bi, region, gutter) in regions {
                        let sw = (region.width / scale.0).max(1);
                        let sh = (region.height / scale.1).max(1);
                        let pad_x = (gutter / scale.0).max(2);
                        let pad_y = (gutter / scale.1).max(2);
                        let (pw, ph) = (sw + pad_x * 2, sh + pad_y * 2);
                        if dx + pw > max_size {
                            (dx, dy, d_row_h) = (0, dy + d_row_h, 0);
                        }
                        let slot = Region {
                            x: dx + pad_x,
                            y: dy + pad_y,
                            width: sw,
                            height: sh,
                        };
                        packed.push((bi, region, slot));
                        packed_map.insert(bi, slot);
                        dx += pw;
                        d_used_w = d_used_w.max(dx);
                        d_row_h = d_row_h.max(ph);
                    }
                    let ds_size = (
                        d_used_w.next_multiple_of(128).max(128).min(max_size),
                        (dy + d_row_h).next_multiple_of(128).max(128).min(max_size),
                    );
                    let target = atlas_texture(descriptors, texture_pool, ds_size);
                    let (source_view, source_size, parent_map) = if parent == (1, 1) {
                        (current_view, current_size, None)
                    } else {
                        let (src, map) = &downscaled[&parent];
                        (&src.texture.1, src.size, Some(map))
                    };
                    let mut uniforms = Uniforms::new(descriptors);
                    let copy = uniforms.push(
                        &kernel_uniform([0.0, 0.0], &[[0.0, 1.0]]),
                        KERNEL_UNIFORM_SIZE as usize,
                    );
                    let vertices: Vec<_> = packed
                        .iter()
                        .flat_map(|&(bi, orig_region, dst_slot)| {
                            let src_region = match parent_map {
                                Some(map) => map[&bi],
                                None => orig_region,
                            };
                            quad(
                                dst_slot.normalized(target.size),
                                src_region.normalized(source_size),
                            )
                        })
                        .collect();
                    let uniforms = uniforms.buffer(&mut scratch, encoder, staging_belt);
                    let vertices = scratch.vertices(encoder, staging_belt, &vertices);
                    let mut render_pass = clear_pass(encoder, &target.texture.1);
                    draw_kernel_quads(
                        descriptors,
                        &mut render_pass,
                        source_view,
                        linear,
                        &uniforms,
                        &vertices,
                        &vec![copy; packed.len()],
                    );
                    drop(render_pass);
                    downscaled.insert(scale, (target, packed_map));
                }

                // 2. Pack all blurred items across all downscale factors into tightly-packed
                //    blur atlases (normally just one), so H-blur and V-blur each take 1 pass
                //    for the whole step and touch much smaller textures.
                let mut blur_order: Vec<usize> = (0..blur_items.len()).collect();
                blur_order.sort_by_key(|&i| {
                    let item = &blur_items[i];
                    let pad_y = (item.gutter / item.plan.factors.1).max(2);
                    std::cmp::Reverse(item.region.height / item.plan.factors.1 + pad_y * 2)
                });
                type BlurBatch = (Vec<(usize, Region)>, u32, u32);
                let mut blur_batches: Vec<BlurBatch> = vec![];
                let mut b_batch = vec![];
                let (mut bx, mut by, mut b_row_h, mut b_used_w) = (0u32, 0u32, 0u32, 0u32);
                for item_idx in blur_order {
                    let item = &blur_items[item_idx];
                    let (fx, fy) = item.plan.factors;
                    let sw = (item.region.width / fx).max(1);
                    let sh = (item.region.height / fy).max(1);
                    let pad_x = 2u32;
                    let pad_y = (item.gutter / fy).max(2);
                    let (pw, ph) = (sw + pad_x * 2, sh + pad_y * 2);
                    if bx + pw > max_size {
                        (bx, by, b_row_h) = (0, by + b_row_h, 0);
                    }
                    if by + ph > max_size {
                        blur_batches.push((std::mem::take(&mut b_batch), b_used_w, by + b_row_h));
                        (bx, by, b_row_h, b_used_w) = (0, 0, 0, 0);
                    }
                    let slot = Region {
                        x: bx + pad_x,
                        y: by + pad_y,
                        width: sw,
                        height: sh,
                    };
                    b_batch.push((item_idx, slot));
                    bx += pw;
                    b_used_w = b_used_w.max(bx);
                    b_row_h = b_row_h.max(ph);
                }
                if !b_batch.is_empty() {
                    blur_batches.push((b_batch, b_used_w, by + b_row_h));
                }

                for (b_batch, bw, bh) in blur_batches {
                    let blur_size = (
                        bw.next_multiple_of(128).max(128).min(max_size),
                        bh.next_multiple_of(128).max(128).min(max_size),
                    );
                    // Horizontal pass: group by source scale and sort by kernel offset so
                    // identical blur kernels draw in a single `draw_indexed` call.
                    let mut h_by_scale: BTreeMap<(u32, u32), Vec<(usize, Region)>> = BTreeMap::new();
                    for &(item_idx, slot) in &b_batch {
                        h_by_scale
                            .entry(blur_items[item_idx].plan.factors)
                            .or_default()
                            .push((item_idx, slot));
                    }
                    let mut h_draws = vec![];
                    for (factors, items) in h_by_scale {
                        let (source_size, ds_map) = if factors == (1, 1) {
                            (current_size, None)
                        } else {
                            let (src, map) = &downscaled[&factors];
                            (src.size, Some(map))
                        };
                        let texel = [1.0 / source_size.0 as f32, 0.0];
                        let mut uniforms = Uniforms::new(descriptors);
                        let mut sorted: Vec<(u32, usize, Region)> = items
                            .iter()
                            .map(|&(item_idx, slot)| {
                                let off = uniforms.push(
                                    &kernel_uniform(texel, &blur_items[item_idx].plan.kernels[0]),
                                    KERNEL_UNIFORM_SIZE as usize,
                                );
                                (off, item_idx, slot)
                            })
                            .collect();
                        sorted.sort_by_key(|&(off, _, _)| off);
                        let offsets: Vec<u32> = sorted.iter().map(|&(off, _, _)| off).collect();
                        let vertices: Vec<_> = sorted
                            .iter()
                            .flat_map(|&(_, item_idx, slot)| {
                                let src_region = match ds_map {
                                    Some(map) => map[&blur_items[item_idx].batch_index],
                                    None => blur_items[item_idx].region,
                                };
                                quad(
                                    slot.normalized(blur_size),
                                    src_region.normalized(source_size),
                                )
                            })
                            .collect();
                        let u_buf = uniforms.buffer(&mut scratch, encoder, staging_belt);
                        let v_buf = scratch.vertices(encoder, staging_belt, &vertices);
                        h_draws.push((factors, u_buf, v_buf, offsets));
                    }

                    let blur_h = atlas_texture(descriptors, texture_pool, blur_size);
                    {
                        let mut render_pass = clear_pass(encoder, &blur_h.texture.1);
                        for (factors, u_buf, v_buf, offsets) in &h_draws {
                            let source_view = if *factors == (1, 1) {
                                current_view
                            } else {
                                &downscaled[factors].0.texture.1
                            };
                            draw_kernel_quads(
                                descriptors,
                                &mut render_pass,
                                source_view,
                                linear,
                                u_buf,
                                v_buf,
                                offsets,
                            );
                        }
                    }

                    // Vertical pass: all items live in `blur_h`, sorted by kernel offset.
                    let blur_v = atlas_texture(descriptors, texture_pool, blur_size);
                    let texel = [0.0, 1.0 / blur_size.1 as f32];
                    let mut v_uniforms = Uniforms::new(descriptors);
                    let mut v_sorted: Vec<(u32, Region)> = b_batch
                        .iter()
                        .map(|&(item_idx, slot)| {
                            let off = v_uniforms.push(
                                &kernel_uniform(texel, &blur_items[item_idx].plan.kernels[1]),
                                KERNEL_UNIFORM_SIZE as usize,
                            );
                            (off, slot)
                        })
                        .collect();
                    v_sorted.sort_by_key(|&(off, _)| off);
                    let v_offsets: Vec<u32> = v_sorted.iter().map(|&(off, _)| off).collect();
                    let v_vertices: Vec<_> = v_sorted
                        .iter()
                        .flat_map(|&(_, slot)| {
                            let norm = slot.normalized(blur_size);
                            quad(norm, norm)
                        })
                        .collect();
                    let v_u_buf = v_uniforms.buffer(&mut scratch, encoder, staging_belt);
                    let v_v_buf = scratch.vertices(encoder, staging_belt, &v_vertices);
                    {
                        let mut render_pass = clear_pass(encoder, &blur_v.texture.1);
                        draw_kernel_quads(
                            descriptors,
                            &mut render_pass,
                            &blur_h.texture.1,
                            linear,
                            &v_u_buf,
                            &v_v_buf,
                            &v_offsets,
                        );
                    }

                    let tex_idx = blurred_textures.len();
                    for &(item_idx, slot) in &b_batch {
                        blurred[blur_items[item_idx].batch_index] = Some((tex_idx, slot));
                    }
                    blurred_textures.push(blur_v);
                }
            }

            // On step 0 all entries are still active and reuse the initial layout. On
            // step > 0, repack only the remaining multi-filter entries tightly so we never
            // allocate a giant atlas or copy-through already-finished 1-filter CABs.
            let (dst_regions, target_size): (Vec<Region>, (u32, u32)) = if step == 0 {
                (active.iter().map(|&(_, r, _)| r).collect(), current_size)
            } else {
                let mut order: Vec<usize> = (0..active.len()).collect();
                order.sort_by_key(|&i| std::cmp::Reverse(active[i].1.height + active[i].2 * 2));
                let mut regions_out = vec![
                    Region {
                        x: 0,
                        y: 0,
                        width: 0,
                        height: 0,
                    };
                    active.len()
                ];
                let (mut rx, mut ry, mut r_row_h, mut r_used_w) = (0u32, 0u32, 0u32, 0u32);
                for ai in order {
                    let (_, src_region, gutter) = active[ai];
                    let (w, h) = (src_region.width + gutter * 2, src_region.height + gutter * 2);
                    if rx + w > max_size {
                        (rx, ry, r_row_h) = (0, ry + r_row_h, 0);
                    }
                    regions_out[ai] = Region {
                        x: rx + gutter,
                        y: ry + gutter,
                        width: src_region.width,
                        height: src_region.height,
                    };
                    rx += w;
                    r_used_w = r_used_w.max(rx);
                    r_row_h = r_row_h.max(h);
                }
                let sz = (
                    r_used_w.next_multiple_of(128).max(128).min(max_size),
                    (ry + r_row_h).next_multiple_of(128).max(128).min(max_size),
                );
                (regions_out, sz)
            };

            // The step's results: blurs scaled back up, glows, and color matrices.
            let target = atlas_texture(descriptors, texture_pool, target_size);
            let mut kernel_uniforms = Uniforms::new(descriptors);
            let mut other_uniforms = Uniforms::new(descriptors);
            let mut kernel_quads: BTreeMap<usize, Vec<[FilterVertex; 4]>> = BTreeMap::new();
            let mut glow_by_tex: BTreeMap<usize, Vec<(u32, [FilterVertexWithBlur; 4])>> =
                BTreeMap::new();
            let mut color_matrix_items: Vec<(u32, [FilterVertex; 4])> = vec![];
            for (batch_index, &(index, src_region, _)) in active.iter().enumerate() {
                let dst_corners = dst_regions[batch_index].normalized(target_size);
                let src_corners = src_region.normalized(current_size);
                match (entries[index].filters.get(step), blurred[batch_index]) {
                    (Some(Filter::BlurFilter(_)), Some((texture, small))) => {
                        let small_size = blurred_textures[texture].size;
                        kernel_quads
                            .entry(texture)
                            .or_default()
                            .push(quad(dst_corners, small.normalized(small_size)));
                    }
                    (Some(Filter::GlowFilter(filter)), Some((texture, small))) => {
                        let small_corners = small.normalized(blurred_textures[texture].size);
                        let uniform = GlowUniform {
                            color: [
                                f32::from(filter.color.r) / 255.0,
                                f32::from(filter.color.g) / 255.0,
                                f32::from(filter.color.b) / 255.0,
                                f32::from(filter.color.a) / 255.0,
                            ],
                            strength: filter.strength.to_f32(),
                            inner: filter.is_inner().into(),
                            knockout: filter.is_knockout().into(),
                            composite_source: filter.composite_source().into(),
                        };
                        let offset = other_uniforms
                            .push(bytemuck::bytes_of(&uniform), size_of::<GlowUniform>());
                        glow_by_tex.entry(texture).or_default().push((
                            offset,
                            std::array::from_fn(|i| FilterVertexWithBlur {
                                position: dst_corners[i],
                                source_uv: src_corners[i],
                                blur_uv: small_corners[i],
                            }),
                        ));
                    }
                    (Some(Filter::ColorMatrixFilter(filter)), _) => {
                        let offset = other_uniforms
                            .push(bytemuck::cast_slice(&filter.matrix), size_of::<[f32; 20]>());
                        color_matrix_items.push((offset, quad(dst_corners, src_corners)));
                    }
                    _ => unreachable!("active entries always have a supported filter at step"),
                }
            }

            let kernel_draws = if !kernel_quads.is_empty() {
                let copy = kernel_uniforms.push(
                    &kernel_uniform([0.0, 0.0], &[[0.0, 1.0]]),
                    KERNEL_UNIFORM_SIZE as usize,
                );
                let u_buf = kernel_uniforms.buffer(&mut scratch, encoder, staging_belt);
                let verts: Vec<_> = kernel_quads
                    .into_iter()
                    .map(|(texture, quads)| {
                        let count = quads.len();
                        (
                            texture,
                            count,
                            scratch.vertices(encoder, staging_belt, &quads),
                        )
                    })
                    .collect();
                Some((copy, u_buf, verts))
            } else {
                None
            };

            let other_uniforms_buf = (!other_uniforms.bytes.is_empty())
                .then(|| other_uniforms.buffer(&mut scratch, encoder, staging_belt));

            let glow = &descriptors.filters.glow;
            let glow_draws: Vec<_> = if let Some(ref u_buf) = other_uniforms_buf {
                glow_by_tex
                    .into_iter()
                    .map(|(texture, mut items)| {
                        items.sort_by_key(|&(off, _)| off);
                        let offsets: Vec<u32> = items.iter().map(|&(off, _)| off).collect();
                        let quads: Vec<[FilterVertexWithBlur; 4]> =
                            items.into_iter().map(|(_, q)| q).collect();
                        let v_buf = scratch.vertices(encoder, staging_belt, &quads);
                        let bind_group =
                            descriptors
                                .device
                                .create_bind_group(&wgpu::BindGroupDescriptor {
                                    label: create_debug_label!("Filter batch glow binds")
                                        .as_deref(),
                                    layout: &glow.linear_bind_group_layout,
                                    entries: &[
                                        wgpu::BindGroupEntry {
                                            binding: 0,
                                            resource: wgpu::BindingResource::TextureView(
                                                current_view,
                                            ),
                                        },
                                        wgpu::BindGroupEntry {
                                            binding: 1,
                                            resource: wgpu::BindingResource::Sampler(linear),
                                        },
                                        wgpu::BindGroupEntry {
                                            binding: 2,
                                            resource: wgpu::BindingResource::Buffer(
                                                wgpu::BufferBinding {
                                                    buffer: u_buf,
                                                    offset: 0,
                                                    size: wgpu::BufferSize::new(
                                                        size_of::<GlowUniform>() as u64,
                                                    ),
                                                },
                                            ),
                                        },
                                        wgpu::BindGroupEntry {
                                            binding: 3,
                                            resource: wgpu::BindingResource::TextureView(
                                                &blurred_textures[texture].texture.1,
                                            ),
                                        },
                                    ],
                                });
                        (bind_group, v_buf, offsets)
                    })
                    .collect()
            } else {
                vec![]
            };

            let color_matrix = &descriptors.filters.color_matrix;
            let cm_draw = if !color_matrix_items.is_empty()
                && let Some(ref u_buf) = other_uniforms_buf
            {
                color_matrix_items.sort_by_key(|&(off, _)| off);
                let offsets: Vec<u32> = color_matrix_items.iter().map(|&(off, _)| off).collect();
                let quads: Vec<[FilterVertex; 4]> =
                    color_matrix_items.into_iter().map(|(_, q)| q).collect();
                let v_buf = scratch.vertices(encoder, staging_belt, &quads);
                let bind_group = descriptors
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: create_debug_label!("Filter batch color matrix binds").as_deref(),
                        layout: &color_matrix.dynamic_bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(current_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(nearest),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                    buffer: u_buf,
                                    offset: 0,
                                    size: wgpu::BufferSize::new(size_of::<[f32; 20]>() as u64),
                                }),
                            },
                        ],
                    });
                Some((bind_group, v_buf, offsets))
            } else {
                None
            };

            {
                let mut render_pass = clear_pass(encoder, &target.texture.1);
                if let Some((copy, ref u_buf, ref verts)) = kernel_draws {
                    for (texture, count, vertices) in verts {
                        draw_kernel_quads(
                            descriptors,
                            &mut render_pass,
                            &blurred_textures[*texture].texture.1,
                            linear,
                            u_buf,
                            vertices,
                            &vec![copy; *count],
                        );
                    }
                }
                render_pass.set_index_buffer(
                    descriptors.quad.batch_quad_indices.slice(..),
                    wgpu::IndexFormat::Uint32,
                );
                if !glow_draws.is_empty() {
                    render_pass.set_pipeline(glow.linear_pipeline(descriptors));
                    for (bind_group, v_buf, offsets) in &glow_draws {
                        render_pass.set_vertex_buffer(0, v_buf.slice(..));
                        draw_batched_quads(&mut render_pass, bind_group, offsets);
                    }
                }
                if let Some((bind_group, v_buf, offsets)) = &cm_draw {
                    render_pass.set_pipeline(color_matrix.dynamic_pipeline(descriptors));
                    render_pass.set_vertex_buffer(0, v_buf.slice(..));
                    draw_batched_quads(&mut render_pass, bind_group, offsets);
                }
            }

            // Finalize entries whose last filter was `step`, and carry only multi-filter
            // entries into `step + 1`.
            let mut next_active = vec![];
            let mut direct_count = 0u64;
            let atlas_idx = direct_out.bind_groups.len();
            for (batch_index, &(index, _, gutter)) in active.iter().enumerate() {
                let dst_region = dst_regions[batch_index];
                if entries[index].filters.len() == step + 1 {
                    match entries[index].volatility {
                        CacheVolatility::Subframe | CacheVolatility::Tick(_) => {
                            let ptr = entries[index].handle_ptr;
                            direct_out.slots.insert(
                                ptr,
                                (
                                    atlas_idx,
                                    dst_region.x,
                                    dst_region.y,
                                    target_size.0,
                                    target_size.1,
                                ),
                            );
                            direct_out.handles.insert(ptr, entries[index].handle.clone());
                            if matches!(entries[index].volatility, CacheVolatility::Subframe) {
                                direct_out.subframe_keys.push(ptr);
                            }
                            direct_count += 1;
                        }
                        CacheVolatility::Persistent => {
                            let (uw, uh) = used_extent(&entries[index]);
                            encoder.copy_texture_to_texture(
                                wgpu::TexelCopyTextureInfo {
                                    texture: &target.texture.0,
                                    mip_level: 0,
                                    origin: wgpu::Origin3d {
                                        x: dst_region.x,
                                        y: dst_region.y,
                                        z: 0,
                                    },
                                    aspect: Default::default(),
                                },
                                entries[index].texture.as_image_copy(),
                                wgpu::Extent3d {
                                    width: uw,
                                    height: uh,
                                    depth_or_array_layers: 1,
                                },
                            );
                        }
                    }
                } else {
                    next_active.push((index, dst_region, gutter));
                }
            }

            if direct_count > 0 {
                let bind_group =
                    descriptors
                        .device
                        .create_bind_group(&wgpu::BindGroupDescriptor {
                            layout: &descriptors.bind_layouts.bitmap,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: wgpu::BindingResource::TextureView(&target.texture.1),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::Sampler(linear),
                                },
                            ],
                            label: create_debug_label!("Filter atlas direct bind group").as_deref(),
                        });
                direct_out.textures.push(target.texture);
                direct_out.bind_groups.push(bind_group);
                crate::perf_counters::count_named_n("filter-atlas-direct", direct_count);
                if next_active.is_empty() {
                    break;
                }
                current_owned = None;
                current_direct = Some((atlas_idx, target_size));
                active = next_active;
            } else {
                if next_active.is_empty() {
                    break;
                }
                current_owned = Some(target);
                current_direct = None;
                active = next_active;
            }
        }
        crate::perf_counters::count_named_n("filter-together", batch_len);
    }
    skipped
}
