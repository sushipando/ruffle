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

use crate::buffer_pool::TexturePool;
use crate::descriptors::Descriptors;
use crate::dynamic_transforms::DynamicTransforms;
use crate::filters::blur::{KERNEL_UNIFORM_SIZE, downscale_factors, merged_kernel};
use crate::filters::glow::GlowUniform;
use crate::filters::{FilterVertex, FilterVertexWithBlur};
use crate::mesh::Mesh;
use crate::surface::layer_atlas::{self, LAYER_TEXTURE_USAGE};
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
    pub filters: Vec<Filter>,
    /// If `Some`, the entry hasn't been drawn into `texture` yet and can be drawn
    /// directly into the filter atlas, avoiding a separate layer atlas pass and two
    /// texture copies.
    pub direct_commands: Option<CommandList>,
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
    fn scaled(self, factors: (u32, u32)) -> Self {
        Self {
            x: self.x / factors.0,
            y: self.y / factors.1,
            width: self.width / factors.0,
            height: self.height / factors.1,
        }
    }

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

/// Collects per-quad uniforms at aligned offsets of one buffer.
struct Uniforms {
    bytes: Vec<u8>,
    alignment: usize,
}

impl Uniforms {
    fn new(descriptors: &Descriptors) -> Self {
        Self {
            bytes: vec![],
            alignment: descriptors.limits.min_uniform_buffer_offset_alignment as usize,
        }
    }

    /// Adds `data`, reserving at least `size` bytes, and returns its offset.
    fn push(&mut self, data: &[u8], size: usize) -> u32 {
        let offset = self.bytes.len().next_multiple_of(self.alignment);
        self.bytes.resize(offset + size.max(data.len()), 0);
        self.bytes[offset..offset + data.len()].copy_from_slice(data);
        offset as u32
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
        descriptors.quad.indices.slice(..),
        wgpu::IndexFormat::Uint32,
    );
    for (index, offset) in offsets.iter().enumerate() {
        render_pass.set_bind_group(0, &bind_group, &[*offset]);
        render_pass.draw_indexed(0..6, index as i32 * 4, 0..1);
    }
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
) -> Vec<usize> {
    let mut scratch = Scratch {
        descriptors,
        taken: vec![],
    };
    // Shelf-pack the textures, each with transparent space around it as wide as its
    // filters can spread (like `apply_filters_batched`).
    let mut order: Vec<usize> = (0..entries.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(entries[i].texture.height()));
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
        let width = entry.texture.width().next_multiple_of(ALIGN);
        let height = entry.texture.height().next_multiple_of(ALIGN);
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
        let size = (
            width.next_multiple_of(256).min(max_size),
            height.next_multiple_of(256).min(max_size),
        );
        let mut current = atlas_texture(descriptors, texture_pool, size);

        let mut direct_layers = vec![];
        let mut copy_in = vec![];
        for &(index, region, _) in &batch {
            if let Some(commands) = entries[index].direct_commands.take() {
                let texture = &entries[index].texture;
                direct_layers.push((
                    commands,
                    (region.x as i32, region.y as i32),
                    (region.x, region.y, texture.width(), texture.height()),
                ));
            } else {
                copy_in.push((index, region));
            }
        }
        if !direct_layers.is_empty() {
            layer_atlas::render_into_atlas(
                direct_layers.into_iter(),
                &current.texture.0,
                descriptors,
                meshes,
                staging_belt,
                dynamic_transforms,
                encoder,
                texture_pool,
                quality,
            );
        } else {
            drop(clear_pass(encoder, &current.texture.1));
        }
        for (index, region) in copy_in {
            let texture = &entries[index].texture;
            encoder.copy_texture_to_texture(
                texture.as_image_copy(),
                wgpu::TexelCopyTextureInfo {
                    texture: &current.texture.0,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: region.x,
                        y: region.y,
                        z: 0,
                    },
                    aspect: Default::default(),
                },
                texture.size(),
            );
        }

        let steps = batch
            .iter()
            .map(|&(index, _, _)| entries[index].filters.len())
            .max()
            .unwrap_or(0);
        for step in 0..steps {
            // Collect all blurs (including those of glows) for this step.
            let mut blur_items: Vec<BlurItem> = vec![];
            for (batch_index, &(index, region, gutter)) in batch.iter().enumerate() {
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
            let mut blurred: Vec<Option<(usize, Region)>> = vec![None; batch.len()];
            let mut blurred_textures: Vec<AtlasTexture> = vec![];

            if !blur_items.is_empty() {
                // 1. Shared downscale pyramid: compute each needed intermediate scale once
                //    for all bitmaps that use it (or a deeper scale descended from it).
                let mut scale_regions: HashMap<(u32, u32), Vec<(usize, Region)>> = HashMap::new();
                for item in &blur_items {
                    let mut s = item.plan.factors;
                    while s != (1, 1) {
                        scale_regions
                            .entry(s)
                            .or_default()
                            .push((item.batch_index, item.region));
                        s = parent_scale(s);
                    }
                }
                let mut scales: Vec<(u32, u32)> = scale_regions.keys().copied().collect();
                scales.sort_by_key(|&(sx, sy)| (sx.max(sy), sx * sy, sx, sy));

                let mut downscaled: HashMap<(u32, u32), AtlasTexture> = HashMap::new();
                for scale in scales {
                    let parent = parent_scale(scale);
                    let mut regions = scale_regions.remove(&scale).unwrap_or_default();
                    regions.sort_by_key(|&(bi, _)| bi);
                    regions.dedup_by_key(|(bi, _)| *bi);

                    let target = atlas_texture(
                        descriptors,
                        texture_pool,
                        (size.0 / scale.0, size.1 / scale.1),
                    );
                    let (source_view, source_size) = if parent == (1, 1) {
                        (&current.texture.1, current.size)
                    } else {
                        let src = &downscaled[&parent];
                        (&src.texture.1, src.size)
                    };
                    let mut uniforms = Uniforms::new(descriptors);
                    let copy = uniforms.push(
                        &kernel_uniform([0.0, 0.0], &[[0.0, 1.0]]),
                        KERNEL_UNIFORM_SIZE as usize,
                    );
                    let vertices: Vec<_> = regions
                        .iter()
                        .flat_map(|&(_, region)| {
                            quad(
                                region.scaled(scale).normalized(target.size),
                                region.scaled(parent).normalized(source_size),
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
                        &vec![copy; regions.len()],
                    );
                    drop(render_pass);
                    downscaled.insert(scale, target);
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
                    let sw = item.region.width / fx;
                    let sh = item.region.height / fy;
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
                    // Horizontal pass: group by source scale so all scales share 1 pass.
                    let mut h_by_scale: BTreeMap<(u32, u32), Vec<(usize, Region)>> = BTreeMap::new();
                    for &(item_idx, slot) in &b_batch {
                        h_by_scale
                            .entry(blur_items[item_idx].plan.factors)
                            .or_default()
                            .push((item_idx, slot));
                    }
                    let mut h_draws = vec![];
                    for (factors, items) in h_by_scale {
                        let source_size = if factors == (1, 1) {
                            current.size
                        } else {
                            downscaled[&factors].size
                        };
                        let texel = [1.0 / source_size.0 as f32, 0.0];
                        let mut uniforms = Uniforms::new(descriptors);
                        let offsets: Vec<u32> = items
                            .iter()
                            .map(|&(item_idx, _)| {
                                uniforms.push(
                                    &kernel_uniform(texel, &blur_items[item_idx].plan.kernels[0]),
                                    KERNEL_UNIFORM_SIZE as usize,
                                )
                            })
                            .collect();
                        let vertices: Vec<_> = items
                            .iter()
                            .flat_map(|&(item_idx, slot)| {
                                let src = blur_items[item_idx]
                                    .region
                                    .scaled(factors)
                                    .normalized(source_size);
                                quad(slot.normalized(blur_size), src)
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
                                &current.texture.1
                            } else {
                                &downscaled[factors].texture.1
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

                    // Vertical pass: all items live in `blur_h`, so 1 bind group & 1 pass.
                    let blur_v = atlas_texture(descriptors, texture_pool, blur_size);
                    let texel = [0.0, 1.0 / blur_size.1 as f32];
                    let mut v_uniforms = Uniforms::new(descriptors);
                    let v_offsets: Vec<u32> = b_batch
                        .iter()
                        .map(|&(item_idx, _)| {
                            v_uniforms.push(
                                &kernel_uniform(texel, &blur_items[item_idx].plan.kernels[1]),
                                KERNEL_UNIFORM_SIZE as usize,
                            )
                        })
                        .collect();
                    let v_vertices: Vec<_> = b_batch
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

            // The step's results: blurs scaled back up, glows, color matrices, and
            // bitmaps without a filter in this step copied.
            let target = atlas_texture(descriptors, texture_pool, size);
            let mut kernel_uniforms = Uniforms::new(descriptors);
            let mut other_uniforms = Uniforms::new(descriptors);
            // Quads drawn with the kernel pipeline, by source texture (None: `current`).
            let mut kernel_quads: BTreeMap<Option<usize>, Vec<[FilterVertex; 4]>> = BTreeMap::new();
            // Glow quads grouped by `blurred_textures` index (usually just 0), with dynamic offset.
            let mut glow_by_tex: BTreeMap<usize, (Vec<u32>, Vec<[FilterVertexWithBlur; 4]>)> =
                BTreeMap::new();
            let mut color_matrix_offsets: Vec<u32> = vec![];
            let mut color_matrix_quads: Vec<[FilterVertex; 4]> = vec![];
            for (batch_index, &(index, region, _)) in batch.iter().enumerate() {
                let corners = region.normalized(size);
                match (entries[index].filters.get(step), blurred[batch_index]) {
                    (Some(Filter::BlurFilter(_)), Some((texture, small))) => {
                        let small_size = blurred_textures[texture].size;
                        kernel_quads
                            .entry(Some(texture))
                            .or_default()
                            .push(quad(corners, small.normalized(small_size)));
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
                        let entry = glow_by_tex.entry(texture).or_default();
                        entry.0.push(offset);
                        entry.1.push(std::array::from_fn(|i| FilterVertexWithBlur {
                            position: corners[i],
                            source_uv: corners[i],
                            blur_uv: small_corners[i],
                        }));
                    }
                    (Some(Filter::ColorMatrixFilter(filter)), _) => {
                        let offset = other_uniforms
                            .push(bytemuck::cast_slice(&filter.matrix), size_of::<[f32; 20]>());
                        color_matrix_offsets.push(offset);
                        color_matrix_quads.push(quad(corners, corners));
                    }
                    (None, _) => {
                        kernel_quads
                            .entry(None)
                            .or_default()
                            .push(quad(corners, corners));
                    }
                    _ => unreachable!("only supported filters are batched"),
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
                    .map(|(texture, (offsets, quads))| {
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
                                                &current.texture.1,
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
            let cm_draw = if !color_matrix_quads.is_empty()
                && let Some(ref u_buf) = other_uniforms_buf
            {
                let v_buf = scratch.vertices(encoder, staging_belt, &color_matrix_quads);
                let bind_group = descriptors
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: create_debug_label!("Filter batch color matrix binds").as_deref(),
                        layout: &color_matrix.dynamic_bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(&current.texture.1),
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
                Some((bind_group, v_buf, color_matrix_offsets))
            } else {
                None
            };

            {
                let mut render_pass = clear_pass(encoder, &target.texture.1);
                if let Some((copy, ref u_buf, ref verts)) = kernel_draws {
                    for (texture, count, vertices) in verts {
                        let (source, sampler) = match texture {
                            Some(texture) => (&blurred_textures[*texture].texture.1, linear),
                            None => (&current.texture.1, nearest),
                        };
                        draw_kernel_quads(
                            descriptors,
                            &mut render_pass,
                            source,
                            sampler,
                            u_buf,
                            vertices,
                            &vec![copy; *count],
                        );
                    }
                }
                render_pass.set_index_buffer(
                    descriptors.quad.indices.slice(..),
                    wgpu::IndexFormat::Uint32,
                );
                if !glow_draws.is_empty() {
                    render_pass.set_pipeline(glow.linear_pipeline(descriptors));
                    for (bind_group, v_buf, offsets) in &glow_draws {
                        render_pass.set_vertex_buffer(0, v_buf.slice(..));
                        for (index, offset) in offsets.iter().enumerate() {
                            render_pass.set_bind_group(0, bind_group, &[*offset]);
                            render_pass.draw_indexed(0..6, index as i32 * 4, 0..1);
                        }
                    }
                }
                if let Some((bind_group, v_buf, offsets)) = &cm_draw {
                    render_pass.set_pipeline(color_matrix.dynamic_pipeline(descriptors));
                    render_pass.set_vertex_buffer(0, v_buf.slice(..));
                    for (index, offset) in offsets.iter().enumerate() {
                        render_pass.set_bind_group(0, bind_group, &[*offset]);
                        render_pass.draw_indexed(0..6, index as i32 * 4, 0..1);
                    }
                }
            }
            current = target;
        }

        for &(index, region, _) in &batch {
            let texture = &entries[index].texture;
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &current.texture.0,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: region.x,
                        y: region.y,
                        z: 0,
                    },
                    aspect: Default::default(),
                },
                texture.as_image_copy(),
                texture.size(),
            );
        }
        crate::perf_counters::count_named_n("filter-together", batch.len() as u64);
    }
    skipped
}
