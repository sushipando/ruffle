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
use crate::filters::blur::{
    KERNEL_UNIFORM_SIZE, downscale_factors, merged_kernel, unpaired_kernel,
};
use crate::filters::glow::{
    GlowColorMatrixUniform, GlowUniform, GlowVBlurUniform, MAX_FUSED_V_TAPS,
};
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
    /// Optional dedicated LRU-cache texture to copy the finished filtered result into
    /// (for 2-hit promotion into `FilteredCabCache`).
    pub promote_copy: Option<wgpu::Texture>,
}

/// Filtered CAB atlases kept alive for direct sampling during the current frame (and, for
/// `CacheVolatility::Tick`, the remaining sub-frames of the current movie tick), avoiding
/// per-CAB copy-out blits and per-CAB bind groups for animated CABs.
#[derive(Default)]
pub struct FilterAtlasResult {
    textures: Vec<Option<PoolEntry<PooledTexture, AlwaysCompatible>>>,
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

    /// Exposes an externally owned cached texture (e.g. from `FilteredCabCache`) for
    /// zero-copy direct sampling in the main pass.
    pub fn insert_external_slot(
        &mut self,
        handle: BitmapHandle,
        handle_ptr: usize,
        bind_group: wgpu::BindGroup,
        used_size: (u32, u32),
        volatility: CacheVolatility,
    ) {
        let atlas_idx = self.bind_groups.len();
        self.textures.push(None);
        self.bind_groups.push(bind_group);
        self.slots
            .insert(handle_ptr, (atlas_idx, 0, 0, used_size.0, used_size.1));
        self.handles.insert(handle_ptr, handle);
        if matches!(volatility, CacheVolatility::Subframe) {
            self.subframe_keys.push(handle_ptr);
        }
    }

    /// Points a same-frame duplicate CAB (`to_ptr`) at an already-filtered leader's slot
    /// (`from_ptr`) in this atlas.
    pub fn alias_slot(
        &mut self,
        from_ptr: usize,
        handle: BitmapHandle,
        to_ptr: usize,
        volatility: CacheVolatility,
    ) -> bool {
        let Some(&slot) = self.slots.get(&from_ptr) else {
            return false;
        };
        self.slots.insert(to_ptr, slot);
        self.handles.insert(to_ptr, handle);
        if matches!(volatility, CacheVolatility::Subframe) {
            self.subframe_keys.push(to_ptr);
        }
        true
    }

    /// Copies a leader's finished atlas slot (`from_ptr`) into a `Persistent` follower's
    /// dedicated CAB texture so cross-volatility same-frame duplicates share one filter pass.
    pub fn copy_slot_to_texture(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        from_ptr: usize,
        dst_texture: &wgpu::Texture,
        used_size: (u32, u32),
    ) -> bool {
        let Some(&(atlas_idx, slot_x, slot_y, _, _)) = self.slots.get(&from_ptr) else {
            return false;
        };
        let Some(Some(src_tex)) = self.textures.get(atlas_idx) else {
            return false;
        };
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &src_tex.0,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: slot_x,
                    y: slot_y,
                    z: 0,
                },
                aspect: Default::default(),
            },
            dst_texture.as_image_copy(),
            wgpu::Extent3d {
                width: used_size.0,
                height: used_size.1,
                depth_or_array_layers: 1,
            },
        );
        true
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

/// Maximum unpaired tap count for skipping the `(2, 2)` downscale pass via direct 2×2
/// bilinear H-blur from `current`.
const MAX_DIRECT_2X2_TAPS: usize = 48;

/// The kernels and scale of a blur (see `BlurFilter::apply`), if it can be merged.
struct BlurPlan {
    factors: (u32, u32),
    /// The kernels at the lower resolution, horizontal and vertical.
    kernels: [Vec<[f32; 2]>; 2],
    /// When `factors == (2, 2)`, an exact integer-offset horizontal kernel so H-blur can
    /// sample `current` directly with hardware 2×2 bilinear averaging, skipping the
    /// `(2, 2)` downscale pass.
    direct_2x2_h: Option<Vec<[f32; 2]>>,
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
    let direct_2x2_h = if factors == (2, 2) {
        let strength = filter.blur_x.to_f32() / 2.0;
        if strength.min(255.0) <= 1.0 || filter.num_passes() == 0 {
            Some(vec![[0.0, 1.0]])
        } else {
            unpaired_kernel(strength, filter.num_passes(), MAX_DIRECT_2X2_TAPS)
        }
    } else {
        None
    };
    Some(BlurPlan {
        factors,
        kernels: [
            kernel(filter.blur_x.to_f32(), factors.0)?,
            kernel(filter.blur_y.to_f32(), factors.1)?,
        ],
        direct_2x2_h,
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

#[derive(Clone, Copy)]
enum KernelPipeline {
    Rgba,
    R8FromRgba,
    R8FromR8,
}

fn glow_vblur_uniform(
    glow: GlowUniform,
    direction: [f32; 2],
    taps: &[[f32; 2]],
) -> GlowVBlurUniform {
    let mut packed_taps = [[0.0f32; 4]; MAX_FUSED_V_TAPS / 2];
    for (i, pair) in taps.chunks(2).enumerate() {
        packed_taps[i][0] = pair[0][0];
        packed_taps[i][1] = pair[0][1];
        if let Some(second) = pair.get(1) {
            packed_taps[i][2] = second[0];
            packed_taps[i][3] = second[1];
        }
    }
    GlowVBlurUniform {
        glow,
        dir_x: direction[0],
        dir_y: direction[1],
        count: taps.len() as u32,
        _padding: 0,
        taps: packed_taps,
    }
}

/// Draws quads (four vertices each, from `vertices`) with the dynamic kernel pipeline,
/// each with its own kernel at the given offset of `uniforms`.
#[expect(clippy::too_many_arguments)]
fn draw_kernel_quads(
    descriptors: &Descriptors,
    render_pass: &mut wgpu::RenderPass<'_>,
    source: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    uniforms: &wgpu::Buffer,
    vertices: &wgpu::Buffer,
    offsets: &[u32],
    pipeline: KernelPipeline,
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
    render_pass.set_pipeline(match pipeline {
        KernelPipeline::Rgba => blur.dynamic_kernel_pipeline(descriptors),
        KernelPipeline::R8FromRgba => blur.dynamic_kernel_r8_from_rgba_pipeline(descriptors),
        KernelPipeline::R8FromR8 => blur.dynamic_kernel_r8_from_r_pipeline(descriptors),
    });
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
    atlas_texture_with_format(
        descriptors,
        texture_pool,
        size,
        wgpu::TextureFormat::Rgba8Unorm,
    )
}

fn atlas_texture_with_format(
    descriptors: &Descriptors,
    texture_pool: &mut TexturePool,
    size: (u32, u32),
    format: wgpu::TextureFormat,
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
            format,
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
    is_glow: bool,
    has_fused_cm: bool,
}

#[derive(Clone, Copy)]
enum BlurredSlot {
    /// Both H and V blurs have completed; ready to sample in the composite pass.
    Ready { texture: usize, slot: Region },
    /// Only H-blur has completed (in `blurred_textures[texture]`); V-blur will be fused
    /// into the Glow composite shader using `blur_items[blur_item_idx].plan.kernels[1]`.
    FusedV {
        texture: usize,
        slot: Region,
        blur_item_idx: usize,
    },
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

        let mut current_owned: Option<AtlasTexture> = Some(current_init);
        let mut current_direct: Option<(usize, (u32, u32))> = None;
        // Each active entry tracks `(entry_index, region, gutter, next_filter_idx)`.
        let mut active: Vec<(usize, Region, u32, usize)> = batch
            .into_iter()
            .map(|(index, region, gutter)| (index, region, gutter, 0))
            .collect();
        let mut step_num = 0usize;
        while !active.is_empty() {
            let (current_view, current_size) = match (&current_owned, current_direct) {
                (Some(cur), _) => (&cur.texture.1, cur.size),
                (None, Some((idx, sz))) => (
                    &direct_out.textures[idx]
                        .as_ref()
                        .expect("multi-step direct atlas is owned")
                        .1,
                    sz,
                ),
                _ => unreachable!("current atlas is always live"),
            };

            // Collect all blurs (including those of glows) for this step, and check
            // whether a GlowFilter can fuse a trailing ColorMatrixFilter into the same pass.
            let mut blur_items: Vec<BlurItem> = vec![];
            let mut fused_cm_at: Vec<bool> = vec![false; active.len()];
            for (batch_index, &(index, region, gutter, filter_idx)) in active.iter().enumerate() {
                let (blur, is_glow, has_fused_cm) = match entries[index].filters.get(filter_idx) {
                    Some(Filter::BlurFilter(filter)) => (filter.clone(), false, false),
                    Some(Filter::GlowFilter(filter)) => {
                        let next_is_cm = matches!(
                            entries[index].filters.get(filter_idx + 1),
                            Some(Filter::ColorMatrixFilter(_))
                        );
                        (filter.inner_blur_filter(), true, next_is_cm)
                    }
                    _ => continue,
                };
                fused_cm_at[batch_index] = has_fused_cm;
                let plan = blur_plan(&blur).expect("checked by can_apply_together");
                blur_items.push(BlurItem {
                    batch_index,
                    region,
                    gutter,
                    plan,
                    is_glow,
                    has_fused_cm,
                });
            }

            // GlowFilter only reads the alpha channel of the blurred texture. When every
            // blur in this step belongs to a GlowFilter, run the downscale + H/V blur
            // passes in `R8Unorm` (1 byte/px instead of 4 bytes/px — 4× less bandwidth).
            let alpha_only = !blur_items.is_empty() && blur_items.iter().all(|item| item.is_glow);
            let blur_format = if alpha_only {
                wgpu::TextureFormat::R8Unorm
            } else {
                wgpu::TextureFormat::Rgba8Unorm
            };
            if alpha_only {
                crate::perf_counters::count_named("filter-r8");
            }

            // The blurred (scaled-down) result of each bitmap blurred in this step.
            let mut blurred: Vec<Option<BlurredSlot>> = vec![None; active.len()];
            let mut blurred_textures: Vec<AtlasTexture> = vec![];

            if !blur_items.is_empty() {
                // 1. Tightly-packed shared downscale pyramid: skip `(2, 2)` items whose
                //    H-blur can sample `current` directly with hardware 2×2 bilinear averaging.
                let mut scale_regions: HashMap<(u32, u32), Vec<(usize, Region, u32)>> =
                    HashMap::new();
                for item in &blur_items {
                    if item.plan.direct_2x2_h.is_some() {
                        continue;
                    }
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
                    let target =
                        atlas_texture_with_format(descriptors, texture_pool, ds_size, blur_format);
                    let (source_view, source_size, parent_map, ds_pipeline) = if parent == (1, 1) {
                        (
                            current_view,
                            current_size,
                            None,
                            if alpha_only {
                                KernelPipeline::R8FromRgba
                            } else {
                                KernelPipeline::Rgba
                            },
                        )
                    } else {
                        let (src, map) = &downscaled[&parent];
                        (
                            &src.texture.1,
                            src.size,
                            Some(map),
                            if alpha_only {
                                KernelPipeline::R8FromR8
                            } else {
                                KernelPipeline::Rgba
                            },
                        )
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
                        ds_pipeline,
                    );
                    drop(render_pass);
                    downscaled.insert(scale, (target, packed_map));
                }

                // 2. Pack all blurred items across all downscale factors into tightly-packed
                //    blur atlases (normally just one), so H-blur and V-blur each take at most
                //    1 pass for the whole step and touch much smaller textures.
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
                    // Horizontal pass: group items reading `current` directly (both `(1,1)`
                    // and direct-2x2) together, and downscaled items by their pyramid level.
                    let mut h_from_current: Vec<(usize, Region)> = vec![];
                    let mut h_from_ds: BTreeMap<(u32, u32), Vec<(usize, Region)>> = BTreeMap::new();
                    let mut direct_2x2_count = 0u64;
                    for &(item_idx, slot) in &b_batch {
                        let item = &blur_items[item_idx];
                        if item.plan.factors == (1, 1) {
                            h_from_current.push((item_idx, slot));
                        } else if item.plan.direct_2x2_h.is_some() {
                            h_from_current.push((item_idx, slot));
                            direct_2x2_count += 1;
                        } else {
                            h_from_ds
                                .entry(item.plan.factors)
                                .or_default()
                                .push((item_idx, slot));
                        }
                    }
                    if direct_2x2_count > 0 {
                        crate::perf_counters::count_named_n("filter-direct-2x2", direct_2x2_count);
                    }

                    let current_h_draw = if !h_from_current.is_empty() {
                        let mut uniforms = Uniforms::new(descriptors);
                        let mut sorted: Vec<(u32, usize, Region)> = h_from_current
                            .iter()
                            .map(|&(item_idx, slot)| {
                                let item = &blur_items[item_idx];
                                let (texel_x, taps) =
                                    if let Some(ref direct) = item.plan.direct_2x2_h {
                                        (2.0 / current_size.0 as f32, direct.as_slice())
                                    } else {
                                        (
                                            1.0 / current_size.0 as f32,
                                            item.plan.kernels[0].as_slice(),
                                        )
                                    };
                                let off = uniforms.push(
                                    &kernel_uniform([texel_x, 0.0], taps),
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
                                quad(
                                    slot.normalized(blur_size),
                                    blur_items[item_idx].region.normalized(current_size),
                                )
                            })
                            .collect();
                        let u_buf = uniforms.buffer(&mut scratch, encoder, staging_belt);
                        let v_buf = scratch.vertices(encoder, staging_belt, &vertices);
                        Some((u_buf, v_buf, offsets))
                    } else {
                        None
                    };

                    let mut h_ds_draws = vec![];
                    for (factors, items) in h_from_ds {
                        let (src, ds_map) = &downscaled[&factors];
                        let source_size = src.size;
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
                                let src_region = ds_map[&blur_items[item_idx].batch_index];
                                quad(
                                    slot.normalized(blur_size),
                                    src_region.normalized(source_size),
                                )
                            })
                            .collect();
                        let u_buf = uniforms.buffer(&mut scratch, encoder, staging_belt);
                        let v_buf = scratch.vertices(encoder, staging_belt, &vertices);
                        h_ds_draws.push((factors, u_buf, v_buf, offsets));
                    }

                    let blur_h = atlas_texture_with_format(
                        descriptors,
                        texture_pool,
                        blur_size,
                        blur_format,
                    );
                    {
                        let mut render_pass = clear_pass(encoder, &blur_h.texture.1);
                        if let Some((ref u_buf, ref v_buf, ref offsets)) = current_h_draw {
                            draw_kernel_quads(
                                descriptors,
                                &mut render_pass,
                                current_view,
                                linear,
                                u_buf,
                                v_buf,
                                offsets,
                                if alpha_only {
                                    KernelPipeline::R8FromRgba
                                } else {
                                    KernelPipeline::Rgba
                                },
                            );
                        }
                        for (factors, u_buf, v_buf, offsets) in &h_ds_draws {
                            draw_kernel_quads(
                                descriptors,
                                &mut render_pass,
                                &downscaled[factors].0.texture.1,
                                linear,
                                u_buf,
                                v_buf,
                                offsets,
                                if alpha_only {
                                    KernelPipeline::R8FromR8
                                } else {
                                    KernelPipeline::Rgba
                                },
                            );
                        }
                    }

                    // Partition items: unscaled `(1, 1)` GlowFilters (without fused CM and
                    // with <= MAX_FUSED_V_TAPS vertical taps) fuse V-blur directly into the
                    // Glow composite pass, skipping the `blur_v` write/read completely.
                    let mut fused_v_items: Vec<(usize, Region)> = vec![];
                    let mut normal_v_items: Vec<(usize, Region)> = vec![];
                    for &(item_idx, slot) in &b_batch {
                        let item = &blur_items[item_idx];
                        if item.is_glow
                            && !item.has_fused_cm
                            && item.plan.factors == (1, 1)
                            && item.plan.kernels[1].len() <= MAX_FUSED_V_TAPS
                        {
                            fused_v_items.push((item_idx, slot));
                        } else {
                            normal_v_items.push((item_idx, slot));
                        }
                    }

                    // Vertical pass for items that cannot fuse V-blur (downscaled blurs,
                    // BlurFilters, or Glow+ColorMatrix).
                    if !normal_v_items.is_empty() {
                        // When some `(1,1)` items fused out, tightly repack `normal_v_items`
                        // into a smaller `blur_v` atlas so we don't allocate/clear dead space.
                        let (v_slots, v_size): (Vec<(usize, Region, Region)>, (u32, u32)) =
                            if fused_v_items.is_empty() {
                                (
                                    normal_v_items
                                        .iter()
                                        .map(|&(item_idx, slot)| (item_idx, slot, slot))
                                        .collect(),
                                    blur_size,
                                )
                            } else {
                                let mut v_packed = Vec::with_capacity(normal_v_items.len());
                                let (mut vx, mut vy, mut v_row_h, mut v_used_w) =
                                    (0u32, 0u32, 0u32, 0u32);
                                for &(item_idx, h_slot) in &normal_v_items {
                                    let pad = 2u32;
                                    let (pw, ph) = (h_slot.width + pad * 2, h_slot.height + pad * 2);
                                    if vx + pw > max_size {
                                        (vx, vy, v_row_h) = (0, vy + v_row_h, 0);
                                    }
                                    let dst_slot = Region {
                                        x: vx + pad,
                                        y: vy + pad,
                                        width: h_slot.width,
                                        height: h_slot.height,
                                    };
                                    v_packed.push((item_idx, h_slot, dst_slot));
                                    vx += pw;
                                    v_used_w = v_used_w.max(vx);
                                    v_row_h = v_row_h.max(ph);
                                }
                                let sz = (
                                    v_used_w.next_multiple_of(128).max(128).min(max_size),
                                    (vy + v_row_h).next_multiple_of(128).max(128).min(max_size),
                                );
                                (v_packed, sz)
                            };

                        let blur_v = atlas_texture_with_format(
                            descriptors,
                            texture_pool,
                            v_size,
                            blur_format,
                        );
                        let texel = [0.0, 1.0 / blur_size.1 as f32];
                        let mut v_uniforms = Uniforms::new(descriptors);
                        let mut v_sorted: Vec<(u32, Region, Region)> = v_slots
                            .iter()
                            .map(|&(item_idx, h_slot, dst_slot)| {
                                let off = v_uniforms.push(
                                    &kernel_uniform(texel, &blur_items[item_idx].plan.kernels[1]),
                                    KERNEL_UNIFORM_SIZE as usize,
                                );
                                (off, h_slot, dst_slot)
                            })
                            .collect();
                        v_sorted.sort_by_key(|&(off, _, _)| off);
                        let v_offsets: Vec<u32> = v_sorted.iter().map(|&(off, _, _)| off).collect();
                        let v_vertices: Vec<_> = v_sorted
                            .iter()
                            .flat_map(|&(_, h_slot, dst_slot)| {
                                quad(dst_slot.normalized(v_size), h_slot.normalized(blur_size))
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
                                if alpha_only {
                                    KernelPipeline::R8FromR8
                                } else {
                                    KernelPipeline::Rgba
                                },
                            );
                        }

                        let tex_idx = blurred_textures.len();
                        for &(item_idx, _, dst_slot) in &v_slots {
                            blurred[blur_items[item_idx].batch_index] = Some(BlurredSlot::Ready {
                                texture: tex_idx,
                                slot: dst_slot,
                            });
                        }
                        blurred_textures.push(blur_v);
                    }

                    if !fused_v_items.is_empty() {
                        let h_tex_idx = blurred_textures.len();
                        for &(item_idx, slot) in &fused_v_items {
                            blurred[blur_items[item_idx].batch_index] = Some(BlurredSlot::FusedV {
                                texture: h_tex_idx,
                                slot,
                                blur_item_idx: item_idx,
                            });
                        }
                        crate::perf_counters::count_named_n(
                            "filter-fused-vblur",
                            fused_v_items.len() as u64,
                        );
                        blurred_textures.push(blur_h);
                    }
                }
            }

            // On step 0 all entries are still active and reuse the initial layout. On
            // step > 0, repack only the remaining multi-filter entries tightly so we never
            // allocate a giant atlas or copy-through already-finished 1-filter CABs.
            let (dst_regions, target_size): (Vec<Region>, (u32, u32)) = if step_num == 0 {
                (active.iter().map(|&(_, r, _, _)| r).collect(), current_size)
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
                    let (_, src_region, gutter, _) = active[ai];
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

            // The step's results: blurs scaled back up, glows (normal, fused V-blur, or
            // fused Glow+ColorMatrix), and standalone color matrices.
            let target = atlas_texture(descriptors, texture_pool, target_size);
            let mut kernel_uniforms = Uniforms::new(descriptors);
            let mut other_uniforms = Uniforms::new(descriptors);
            let mut kernel_quads: BTreeMap<usize, Vec<[FilterVertex; 4]>> = BTreeMap::new();
            let mut glow_by_tex: BTreeMap<usize, Vec<(u32, [FilterVertexWithBlur; 4])>> =
                BTreeMap::new();
            let mut glow_vblur_by_tex: BTreeMap<usize, Vec<(u32, [FilterVertexWithBlur; 4])>> =
                BTreeMap::new();
            let mut glow_cm_by_tex: BTreeMap<usize, Vec<(u32, [FilterVertexWithBlur; 4])>> =
                BTreeMap::new();
            let mut color_matrix_items: Vec<(u32, [FilterVertex; 4])> = vec![];
            let mut fused_cm_count = 0u64;
            for (batch_index, &(index, src_region, _, filter_idx)) in active.iter().enumerate() {
                let dst_corners = dst_regions[batch_index].normalized(target_size);
                let src_corners = src_region.normalized(current_size);
                match (entries[index].filters.get(filter_idx), blurred[batch_index]) {
                    (
                        Some(Filter::BlurFilter(_)),
                        Some(BlurredSlot::Ready { texture, slot }),
                    ) => {
                        let small_size = blurred_textures[texture].size;
                        kernel_quads
                            .entry(texture)
                            .or_default()
                            .push(quad(dst_corners, slot.normalized(small_size)));
                    }
                    (Some(Filter::GlowFilter(filter)), Some(slot_kind)) => {
                        let glow_uniform = GlowUniform {
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
                        match slot_kind {
                            BlurredSlot::Ready { texture, slot } => {
                                let small_corners =
                                    slot.normalized(blurred_textures[texture].size);
                                let verts = std::array::from_fn(|i| FilterVertexWithBlur {
                                    position: dst_corners[i],
                                    source_uv: src_corners[i],
                                    blur_uv: small_corners[i],
                                });
                                if fused_cm_at[batch_index]
                                    && let Some(Filter::ColorMatrixFilter(cm)) =
                                        entries[index].filters.get(filter_idx + 1)
                                {
                                    let cm_uniform = GlowColorMatrixUniform::from_glow_and_matrix(
                                        glow_uniform,
                                        &cm.matrix,
                                    );
                                    let offset = other_uniforms.push(
                                        bytemuck::bytes_of(&cm_uniform),
                                        size_of::<GlowColorMatrixUniform>(),
                                    );
                                    glow_cm_by_tex
                                        .entry(texture)
                                        .or_default()
                                        .push((offset, verts));
                                    fused_cm_count += 1;
                                } else {
                                    let offset = other_uniforms.push(
                                        bytemuck::bytes_of(&glow_uniform),
                                        size_of::<GlowUniform>(),
                                    );
                                    glow_by_tex
                                        .entry(texture)
                                        .or_default()
                                        .push((offset, verts));
                                }
                            }
                            BlurredSlot::FusedV {
                                texture,
                                slot,
                                blur_item_idx,
                            } => {
                                let h_size = blurred_textures[texture].size;
                                let small_corners = slot.normalized(h_size);
                                let v_dir = [0.0, 1.0 / h_size.1 as f32];
                                let vb_uniform = glow_vblur_uniform(
                                    glow_uniform,
                                    v_dir,
                                    &blur_items[blur_item_idx].plan.kernels[1],
                                );
                                let offset = other_uniforms.push(
                                    bytemuck::bytes_of(&vb_uniform),
                                    size_of::<GlowVBlurUniform>(),
                                );
                                glow_vblur_by_tex.entry(texture).or_default().push((
                                    offset,
                                    std::array::from_fn(|i| FilterVertexWithBlur {
                                        position: dst_corners[i],
                                        source_uv: src_corners[i],
                                        blur_uv: small_corners[i],
                                    }),
                                ));
                            }
                        }
                    }
                    (Some(Filter::ColorMatrixFilter(filter)), _) => {
                        let offset = other_uniforms
                            .push(bytemuck::cast_slice(&filter.matrix), size_of::<[f32; 20]>());
                        color_matrix_items.push((offset, quad(dst_corners, src_corners)));
                    }
                    _ => unreachable!("active entries always have a supported filter at step"),
                }
            }
            if fused_cm_count > 0 {
                crate::perf_counters::count_named_n("filter-fused-cm", fused_cm_count);
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
            let build_glow_draws =
                |by_tex: BTreeMap<usize, Vec<(u32, [FilterVertexWithBlur; 4])>>,
                 layout: &wgpu::BindGroupLayout,
                 uniform_binding: u32,
                 uniform_size: u64,
                 scratch: &mut Scratch<'_>,
                 encoder: &mut Scope<'_, wgpu::CommandEncoder>,
                 staging_belt: &mut StagingBelt| {
                    let Some(ref u_buf) = other_uniforms_buf else {
                        return vec![];
                    };
                    by_tex
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
                                        layout,
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
                                                binding: uniform_binding,
                                                resource: wgpu::BindingResource::Buffer(
                                                    wgpu::BufferBinding {
                                                        buffer: u_buf,
                                                        offset: 0,
                                                        size: wgpu::BufferSize::new(uniform_size),
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
                };

            let glow_draws = build_glow_draws(
                glow_by_tex,
                &glow.linear_bind_group_layout,
                2,
                size_of::<GlowUniform>() as u64,
                &mut scratch,
                encoder,
                staging_belt,
            );
            let glow_vblur_draws = build_glow_draws(
                glow_vblur_by_tex,
                &glow.vblur_bind_group_layout,
                4,
                size_of::<GlowVBlurUniform>() as u64,
                &mut scratch,
                encoder,
                staging_belt,
            );
            let glow_cm_draws = build_glow_draws(
                glow_cm_by_tex,
                &glow.cm_bind_group_layout,
                5,
                size_of::<GlowColorMatrixUniform>() as u64,
                &mut scratch,
                encoder,
                staging_belt,
            );

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
                            KernelPipeline::Rgba,
                        );
                    }
                }
                render_pass.set_index_buffer(
                    descriptors.quad.batch_quad_indices.slice(..),
                    wgpu::IndexFormat::Uint32,
                );
                if !glow_draws.is_empty() {
                    render_pass.set_pipeline(if alpha_only {
                        glow.linear_r8_pipeline(descriptors)
                    } else {
                        glow.linear_pipeline(descriptors)
                    });
                    for (bind_group, v_buf, offsets) in &glow_draws {
                        render_pass.set_vertex_buffer(0, v_buf.slice(..));
                        draw_batched_quads(&mut render_pass, bind_group, offsets);
                    }
                }
                if !glow_vblur_draws.is_empty() {
                    render_pass.set_pipeline(if alpha_only {
                        glow.vblur_r8_pipeline(descriptors)
                    } else {
                        glow.vblur_rgba_pipeline(descriptors)
                    });
                    for (bind_group, v_buf, offsets) in &glow_vblur_draws {
                        render_pass.set_vertex_buffer(0, v_buf.slice(..));
                        draw_batched_quads(&mut render_pass, bind_group, offsets);
                    }
                }
                if !glow_cm_draws.is_empty() {
                    render_pass.set_pipeline(if alpha_only {
                        glow.cm_r8_pipeline(descriptors)
                    } else {
                        glow.cm_rgba_pipeline(descriptors)
                    });
                    for (bind_group, v_buf, offsets) in &glow_cm_draws {
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

            // Finalize entries whose last filter was completed in this step, and carry only
            // remaining multi-filter entries into the next iteration.
            let mut next_active = vec![];
            let mut direct_count = 0u64;
            let atlas_idx = direct_out.bind_groups.len();
            for (batch_index, &(index, _, gutter, filter_idx)) in active.iter().enumerate() {
                let dst_region = dst_regions[batch_index];
                let next_filter_idx = filter_idx + if fused_cm_at[batch_index] { 2 } else { 1 };
                if next_filter_idx >= entries[index].filters.len() {
                    let (uw, uh) = used_extent(&entries[index]);
                    if let Some(ref promote_tex) = entries[index].promote_copy {
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
                            promote_tex.as_image_copy(),
                            wgpu::Extent3d {
                                width: uw,
                                height: uh,
                                depth_or_array_layers: 1,
                            },
                        );
                    }
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
                    next_active.push((index, dst_region, gutter, next_filter_idx));
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
                direct_out.textures.push(Some(target.texture));
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
            step_num += 1;
        }
        crate::perf_counters::count_named_n("filter-together", batch_len);
    }
    skipped
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filters::FilterSource;
    use ruffle_render::bitmap::BitmapHandleImpl;
    use std::sync::Arc;
    use swf::{
        BlurFilterFlags, Color, ColorMatrixFilter as SwfColorMatrixFilter, Fixed8, Fixed16,
        GlowFilter as SwfGlowFilter, GlowFilterFlags,
    };

    #[derive(Debug)]
    struct DummyHandle;
    impl BitmapHandleImpl for DummyHandle {}

    #[test]
    fn uniform_sizes_match_wgsl_alignment() {
        assert_eq!(std::mem::size_of::<GlowUniform>(), 32);
        assert_eq!(std::mem::size_of::<GlowVBlurUniform>(), 304);
        assert_eq!(std::mem::size_of::<GlowColorMatrixUniform>(), 112);
    }

    #[test]
    fn unpaired_kernel_matches_merged_weight_sum_and_has_integer_offsets() {
        for strength in [4.0f32, 6.5, 12.0, 15.0] {
            for passes in 1..=3 {
                let merged = merged_kernel(strength, passes).expect("merged");
                let taps = unpaired_kernel(strength, passes, MAX_DIRECT_2X2_TAPS)
                    .expect("fits within max taps");
                let merged_sum: f32 = merged.iter().map(|t| t[1]).sum();
                let sum: f32 = taps.iter().map(|t| t[1]).sum();
                assert!(
                    (sum - merged_sum).abs() < 1e-5,
                    "unpaired sum={sum} != merged_sum={merged_sum}"
                );
                for tap in &taps {
                    assert_eq!(tap[0].fract(), 0.0, "non-integer offset {}", tap[0]);
                }
            }
        }
        // 3 passes of width 16 has 49 taps (> MAX_DIRECT_2X2_TAPS) and falls back cleanly.
        assert!(unpaired_kernel(16.0, 3, MAX_DIRECT_2X2_TAPS).is_none());
    }

    #[test]
    fn apply_together_r8_and_fusions_match_reference() {
        let instance = wgpu::Instance::default();
        let Some(adapter) =
            futures::executor::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
                ..Default::default()
            }))
            .ok()
        else {
            return;
        };
        let (device, queue) = futures::executor::block_on(adapter.request_device(&Default::default()))
            .expect("create wgpu device");
        let descriptors = Descriptors::new(instance, adapter, device, queue);

        // Enable `BlurDownscale` at 2.5x scale so `(2, 2)` direct H-blur is exercised too.
        ruffle_render::filters::set_view_scale(Some(2.5));

        let width = 64u32;
        let height = 64u32;
        let mut src_rgba = vec![0u8; (width * height * 4) as usize];
        // Draw a crisp premultiplied colored rectangle + diagonal inside the center.
        for y in 20..44 {
            for x in 20..44 {
                let idx = ((y * width + x) * 4) as usize;
                let alpha = if x == 20 || y == 20 { 180u8 } else { 255u8 };
                src_rgba[idx] = ((200u32 * alpha as u32) / 255) as u8;
                src_rgba[idx + 1] = ((120u32 * alpha as u32) / 255) as u8;
                src_rgba[idx + 2] = ((40u32 * alpha as u32) / 255) as u8;
                src_rgba[idx + 3] = alpha;
            }
        }

        let make_src_tex = |label: &str| {
            let tex = descriptors.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            descriptors.queue.write_texture(
                tex.as_image_copy(),
                &src_rgba,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(width * 4),
                    rows_per_image: Some(height),
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
            tex
        };

        let read_tex = |tex: &wgpu::Texture| -> Vec<u8> {
            let padded_bpr = (width * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
            let buf = descriptors.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (padded_bpr * height) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut enc = descriptors.device.create_command_encoder(&Default::default());
            enc.copy_texture_to_buffer(
                tex.as_image_copy(),
                wgpu::TexelCopyBufferInfo {
                    buffer: &buf,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(padded_bpr),
                        rows_per_image: Some(height),
                    },
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
            descriptors.queue.submit([enc.finish()]);
            let slice = buf.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            let _ = descriptors.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            });
            rx.recv().unwrap().unwrap();
            let data = slice.get_mapped_range().unwrap();
            let mut out = Vec::with_capacity((width * height * 4) as usize);
            for y in 0..height as usize {
                let start = y * padded_bpr as usize;
                out.extend_from_slice(&data[start..start + (width * 4) as usize]);
            }
            drop(data);
            buf.unmap();
            out
        };

        let glow_1x1 = Filter::GlowFilter(SwfGlowFilter {
            color: Color::from_rgba(0xFF40CCFF),
            blur_x: Fixed16::from_f32(5.0),
            blur_y: Fixed16::from_f32(5.0),
            strength: Fixed8::from_f32(2.0),
            flags: GlowFilterFlags::COMPOSITE_SOURCE | GlowFilterFlags::from_passes(2),
        });
        let glow_2x2 = Filter::GlowFilter(SwfGlowFilter {
            color: Color::from_rgba(0xFFFF8020),
            blur_x: Fixed16::from_f32(12.0),
            blur_y: Fixed16::from_f32(12.0),
            strength: Fixed8::from_f32(1.5),
            flags: GlowFilterFlags::COMPOSITE_SOURCE | GlowFilterFlags::from_passes(1),
        });
        let color_matrix = Filter::ColorMatrixFilter(SwfColorMatrixFilter {
            matrix: [
                0.8, 0.2, 0.0, 0.0, 10.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.9, 0.0, 5.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
            ],
        });
        let blur_1x1 = Filter::BlurFilter(BlurFilterArgs {
            blur_x: Fixed16::from_f32(4.0),
            blur_y: Fixed16::from_f32(4.0),
            flags: BlurFilterFlags::from_passes(1),
        });

        let cases: &[(&str, Vec<Vec<Filter>>)] = &[
            // Pure glow batch (exercises R8Unorm + fused V-blur + direct 2x2 H-blur + Glow+CM)
            (
                "r8_all_glow",
                vec![
                    vec![glow_1x1.clone()],
                    vec![glow_2x2.clone()],
                    vec![glow_2x2.clone(), color_matrix.clone()],
                    vec![glow_1x1.clone(), color_matrix.clone()],
                ],
            ),
            // Mixed Blur + Glow batch (exercises RGBA fallback + fused V-blur RGBA + Glow+CM RGBA)
            (
                "rgba_mixed",
                vec![
                    vec![blur_1x1.clone()],
                    vec![glow_1x1.clone()],
                    vec![glow_2x2.clone(), color_matrix.clone()],
                ],
            ),
        ];

        let profiler = wgpu_profiler::GpuProfiler::new(
            &descriptors.device,
            wgpu_profiler::GpuProfilerSettings {
                enable_timer_queries: false,
                enable_debug_groups: false,
                max_num_pending_frames: 1,
            },
        )
        .unwrap();

        for (case_name, filter_lists) in cases {
            let mut entries = Vec::new();
            let mut ref_outputs = Vec::new();
            let mut pool = TexturePool::new();
            let mut belt = StagingBelt::new(descriptors.device.clone(), 64 * 1024);
            let mut dyn_transforms = DynamicTransforms::new(&descriptors);

            // Compute reference output for each filter list via sequential `descriptors.filters.apply`.
            for (i, filters) in filter_lists.iter().enumerate() {
                let ref_src = make_src_tex("ref_src");
                let mut enc = descriptors.device.create_command_encoder(&Default::default());
                let mut cur_target = None;
                for filter in filters {
                    let source = match &cur_target {
                        Some(t) => FilterSource::for_entire_texture(
                            crate::surface::target::CommandTarget::color_texture(t),
                        ),
                        None => FilterSource::for_entire_texture(&ref_src),
                    };
                    let mut scope = profiler.scope("ref", &mut enc);
                    cur_target = Some(descriptors.filters.apply(
                        &descriptors,
                        &mut scope,
                        &mut pool,
                        &mut belt,
                        source,
                        filter.clone(),
                    ));
                }
                let final_tex = cur_target.as_ref().unwrap().color_texture();
                let ref_out = make_src_tex("ref_out");
                enc.copy_texture_to_texture(
                    final_tex.as_image_copy(),
                    ref_out.as_image_copy(),
                    final_tex.size(),
                );
                belt.finish();
                descriptors.queue.submit([enc.finish()]);
                belt.recall();
                ref_outputs.push(read_tex(&ref_out));

                let handle = BitmapHandle(Arc::new(DummyHandle));
                let handle_ptr = Arc::as_ptr(&handle.0) as *const () as usize;
                entries.push(FilterTogetherEntry {
                    texture: make_src_tex(&format!("{case_name}_{i}")),
                    used_size: (width, height),
                    filters: filters.clone(),
                    direct_commands: None,
                    handle,
                    handle_ptr,
                    volatility: CacheVolatility::Persistent,
                    promote_copy: None,
                });
            }

            // Run `apply_together` for the whole batch.
            let mut direct_out = FilterAtlasResult::default();
            let meshes = Vec::new();
            let mut enc = descriptors.device.create_command_encoder(&Default::default());
            {
                let mut scope = profiler.scope("batch", &mut enc);
                let skipped = apply_together(
                    &mut entries,
                    &descriptors,
                    &meshes,
                    &mut scope,
                    &mut belt,
                    &dyn_transforms,
                    &mut pool,
                    StageQuality::High,
                    2048,
                    &mut direct_out,
                );
                assert!(skipped.is_empty(), "{case_name}: unexpected skipped entries");
            }
            belt.finish();
            descriptors.queue.submit([enc.finish()]);
            belt.recall();
            let _ = &mut dyn_transforms;

            for (i, entry) in entries.iter().enumerate() {
                let got = read_tex(&entry.texture);
                let expected = &ref_outputs[i];
                let mut max_diff = 0u8;
                let mut diff_count = 0usize;
                for (&a, &b) in got.iter().zip(expected.iter()) {
                    let d = a.abs_diff(b);
                    if d > max_diff {
                        max_diff = d;
                    }
                    if d > 2 {
                        diff_count += 1;
                    }
                }
                assert!(
                    max_diff <= 2,
                    "{case_name}[{i}] diverged from reference: max_diff={max_diff}, >2 count={diff_count}"
                );
            }
        }
        ruffle_render::filters::set_view_scale(None);
    }
}
