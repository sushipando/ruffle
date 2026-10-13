//! Filters of many cached bitmaps, applied together, whatever their parameters.
//!
//! `layer_atlas::apply_filters_batched` applies *identical* filters to many bitmaps at
//! once, by applying them to an atlas. Here, every bitmap has its own filters: the
//! bitmaps are placed in an atlas (with enough transparent space around them), and each
//! step of their filter lists is done for all of them in a few passes, one quad per
//! bitmap with its own parameters. A blur (or a glow's blur) is done like
//! `BlurFilter::apply` with merged kernels (scaled down, horizontal, vertical), and the
//! final pass of a step reads the scaled-down result directly (bilinear), instead of
//! scaling it back up first. So a step costs about four passes for all bitmaps, instead
//! of four or five for each bitmap.
//!
//! Only Blur, Glow (as drop shadows, without an offset) and ColorMatrix are supported,
//! and only while rendering for display, since the results aren't bit-exact (see
//! `BlurFilter::apply_merged`).

use crate::buffer_pool::TexturePool;
use crate::descriptors::Descriptors;
use crate::filters::blur::{KERNEL_UNIFORM_SIZE, downscale_factors, merged_kernel};
use crate::filters::glow::GlowUniform;
use crate::filters::{FilterVertex, FilterVertexWithBlur};
use crate::surface::layer_atlas::LAYER_TEXTURE_USAGE;
use ruffle_render::filters::Filter;
use std::collections::BTreeMap;
use swf::{BlurFilter as BlurFilterArgs, Rectangle, Twips};
use wgpu::util::DeviceExt;

/// Region positions and sizes are multiples of this, so that they scale down exactly
/// (blurs are scaled down by at most 8 in each direction).
const ALIGN: u32 = 8;

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

/// The bitmaps (by index, with their regions) in one atlas, and its used size.
type Batch = (Vec<(usize, Region)>, u32, u32);

/// The bitmaps blurred in a step (by index in their batch, with their regions and
/// blurs), by how much they're scaled down.
type BlurGroups = BTreeMap<(u32, u32), Vec<(usize, Region, BlurPlan)>>;

/// Where a bitmap is in the atlas: `(x, y, width, height)`, multiples of `ALIGN`.
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

    fn buffer(&self, descriptors: &Descriptors) -> wgpu::Buffer {
        descriptors
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: create_debug_label!("Filter batch uniforms").as_deref(),
                contents: &self.bytes,
                usage: wgpu::BufferUsages::UNIFORM,
            })
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

fn vertex_buffer<T: bytemuck::Pod>(descriptors: &Descriptors, vertices: &[T]) -> wgpu::Buffer {
    descriptors
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: create_debug_label!("Filter batch vertices").as_deref(),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        })
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

/// Applies each texture's filters to it, in place (all of which must be supported, see
/// `can_apply_together`). Returns the indices of textures that were too large, which the
/// caller must filter itself.
pub fn apply_together(
    entries: &[(wgpu::Texture, Vec<Filter>)],
    descriptors: &Descriptors,
    encoder: &mut wgpu::CommandEncoder,
    texture_pool: &mut TexturePool,
    max_size: u32,
) -> Vec<usize> {
    // Shelf-pack the textures, each with transparent space around it as wide as its
    // filters can spread (like `apply_filters_batched`).
    let mut order: Vec<usize> = (0..entries.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(entries[i].0.height()));
    let mut batches: Vec<Batch> = vec![];
    let mut batch = vec![];
    let mut skipped = vec![];
    let (mut x, mut y, mut row_height, mut used_width) = (0, 0, 0, 0);
    for index in order {
        let (texture, filters) = &entries[index];
        let growth = filters
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
        let width = texture.width().next_multiple_of(ALIGN);
        let height = texture.height().next_multiple_of(ALIGN);
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
        batch.push((index, region));
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
        drop(clear_pass(encoder, &current.texture.1));
        for &(index, region) in &batch {
            let texture = &entries[index].0;
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
            .map(|&(index, _)| entries[index].1.len())
            .max()
            .unwrap_or(0);
        for step in 0..steps {
            // Blurs (including those of glows), grouped by how much they're scaled down.
            let mut blur_groups: BlurGroups = BTreeMap::new();
            for (batch_index, &(index, region)) in batch.iter().enumerate() {
                let blur = match entries[index].1.get(step) {
                    Some(Filter::BlurFilter(filter)) => filter.clone(),
                    Some(Filter::GlowFilter(filter)) => filter.inner_blur_filter(),
                    _ => continue,
                };
                let plan = blur_plan(&blur).expect("checked by can_apply_together");
                blur_groups
                    .entry(plan.factors)
                    .or_default()
                    .push((batch_index, region, plan));
            }

            // The blurred (scaled-down) result of each bitmap blurred in this step:
            // the texture, and where it is in it.
            let mut blurred: Vec<Option<(usize, Region)>> = vec![None; batch.len()];
            let mut blurred_textures: Vec<AtlasTexture> = vec![];
            for (factors, group) in &blur_groups {
                // Scale down in steps of 2 (bilinear sampling between texels averages them).
                let mut source: Option<AtlasTexture> = None;
                let mut scale = (1, 1);
                while scale.0 < factors.0 || scale.1 < factors.1 {
                    let previous_scale = scale;
                    if scale.0 < factors.0 {
                        scale.0 *= 2;
                    }
                    if scale.1 < factors.1 {
                        scale.1 *= 2;
                    }
                    let target = atlas_texture(
                        descriptors,
                        texture_pool,
                        (size.0 / scale.0, size.1 / scale.1),
                    );
                    let (source_view, source_size) = match &source {
                        Some(source) => (&source.texture.1, source.size),
                        None => (&current.texture.1, current.size),
                    };
                    let mut uniforms = Uniforms::new(descriptors);
                    let copy = uniforms.push(
                        &kernel_uniform([0.0, 0.0], &[[0.0, 1.0]]),
                        KERNEL_UNIFORM_SIZE as usize,
                    );
                    let vertices: Vec<_> = group
                        .iter()
                        .flat_map(|(_, region, _)| {
                            quad(
                                region.scaled(scale).normalized(target.size),
                                region.scaled(previous_scale).normalized(source_size),
                            )
                        })
                        .collect();
                    let uniforms = uniforms.buffer(descriptors);
                    let vertices = vertex_buffer(descriptors, &vertices);
                    let mut render_pass = clear_pass(encoder, &target.texture.1);
                    draw_kernel_quads(
                        descriptors,
                        &mut render_pass,
                        source_view,
                        linear,
                        &uniforms,
                        &vertices,
                        &vec![copy; group.len()],
                    );
                    drop(render_pass);
                    source = Some(target);
                }

                // Horizontal, then vertical.
                let small_size = (size.0 / factors.0, size.1 / factors.1);
                for direction in 0..2 {
                    let target = atlas_texture(descriptors, texture_pool, small_size);
                    let source_view = match &source {
                        Some(source) => &source.texture.1,
                        None => &current.texture.1,
                    };
                    let mut uniforms = Uniforms::new(descriptors);
                    let texel = if direction == 0 {
                        [1.0 / small_size.0 as f32, 0.0]
                    } else {
                        [0.0, 1.0 / small_size.1 as f32]
                    };
                    let offsets: Vec<u32> = group
                        .iter()
                        .map(|(_, _, plan)| {
                            uniforms.push(
                                &kernel_uniform(texel, &plan.kernels[direction]),
                                KERNEL_UNIFORM_SIZE as usize,
                            )
                        })
                        .collect();
                    let vertices: Vec<_> = group
                        .iter()
                        .flat_map(|(_, region, _)| {
                            let small = region.scaled(*factors).normalized(small_size);
                            quad(small, small)
                        })
                        .collect();
                    let uniforms = uniforms.buffer(descriptors);
                    let vertices = vertex_buffer(descriptors, &vertices);
                    let mut render_pass = clear_pass(encoder, &target.texture.1);
                    // Bilinear: the kernel's taps sample between pairs of texels.
                    draw_kernel_quads(
                        descriptors,
                        &mut render_pass,
                        source_view,
                        linear,
                        &uniforms,
                        &vertices,
                        &offsets,
                    );
                    drop(render_pass);
                    source = Some(target);
                }
                let result = source.expect("blurred at least once");
                for (batch_index, region, _) in group {
                    blurred[*batch_index] = Some((blurred_textures.len(), region.scaled(*factors)));
                }
                blurred_textures.push(result);
            }

            // The step's results: blurs scaled back up, glows, color matrices, and
            // bitmaps without a filter in this step copied.
            let target = atlas_texture(descriptors, texture_pool, size);
            let mut kernel_uniforms = Uniforms::new(descriptors);
            let copy = kernel_uniforms.push(
                &kernel_uniform([0.0, 0.0], &[[0.0, 1.0]]),
                KERNEL_UNIFORM_SIZE as usize,
            );
            let mut other_uniforms = Uniforms::new(descriptors);
            // Quads drawn with the kernel pipeline, by source texture (None: `current`).
            let mut kernel_quads: BTreeMap<Option<usize>, Vec<[FilterVertex; 4]>> = BTreeMap::new();
            let mut glow_quads: Vec<(usize, u32, [FilterVertexWithBlur; 4])> = vec![];
            let mut color_matrix_quads: Vec<(u32, [FilterVertex; 4])> = vec![];
            for (batch_index, &(index, region)) in batch.iter().enumerate() {
                let corners = region.normalized(size);
                match (entries[index].1.get(step), blurred[batch_index]) {
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
                        glow_quads.push((
                            texture,
                            offset,
                            std::array::from_fn(|i| FilterVertexWithBlur {
                                position: corners[i],
                                source_uv: corners[i],
                                blur_uv: small_corners[i],
                            }),
                        ));
                    }
                    (Some(Filter::ColorMatrixFilter(filter)), _) => {
                        let offset = other_uniforms
                            .push(bytemuck::cast_slice(&filter.matrix), size_of::<[f32; 20]>());
                        color_matrix_quads.push((offset, quad(corners, corners)));
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
            let kernel_uniforms = kernel_uniforms.buffer(descriptors);
            let other_uniforms = other_uniforms.buffer(descriptors);
            let kernel_vertices: Vec<_> = kernel_quads
                .into_iter()
                .map(|(texture, quads)| {
                    let count = quads.len();
                    (texture, count, vertex_buffer(descriptors, &quads))
                })
                .collect();
            let glow_vertices = vertex_buffer(
                descriptors,
                &glow_quads.iter().map(|(_, _, q)| *q).collect::<Vec<_>>(),
            );
            let color_matrix_vertices = vertex_buffer(
                descriptors,
                &color_matrix_quads
                    .iter()
                    .map(|(_, q)| *q)
                    .collect::<Vec<_>>(),
            );
            let glow = &descriptors.filters.glow;
            let glow_bind_groups: Vec<_> = glow_quads
                .iter()
                .map(|(texture, offset, _)| {
                    descriptors
                        .device
                        .create_bind_group(&wgpu::BindGroupDescriptor {
                            label: create_debug_label!("Filter batch glow binds").as_deref(),
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
                                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                        buffer: &other_uniforms,
                                        offset: *offset as u64,
                                        size: wgpu::BufferSize::new(
                                            size_of::<GlowUniform>() as u64,
                                        ),
                                    }),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 3,
                                    resource: wgpu::BindingResource::TextureView(
                                        &blurred_textures[*texture].texture.1,
                                    ),
                                },
                            ],
                        })
                })
                .collect();
            let color_matrix = &descriptors.filters.color_matrix;
            let color_matrix_bind_groups: Vec<_> = color_matrix_quads
                .iter()
                .map(|(offset, _)| {
                    descriptors
                        .device
                        .create_bind_group(&wgpu::BindGroupDescriptor {
                            label: create_debug_label!("Filter batch color matrix binds")
                                .as_deref(),
                            layout: &color_matrix.bind_group_layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: wgpu::BindingResource::TextureView(
                                        &current.texture.1,
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::Sampler(nearest),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                        buffer: &other_uniforms,
                                        offset: *offset as u64,
                                        size: wgpu::BufferSize::new(size_of::<[f32; 20]>() as u64),
                                    }),
                                },
                            ],
                        })
                })
                .collect();

            let mut render_pass = clear_pass(encoder, &target.texture.1);
            for (texture, count, vertices) in &kernel_vertices {
                let (source, sampler) = match texture {
                    // Scaling up (bilinear).
                    Some(texture) => (&blurred_textures[*texture].texture.1, linear),
                    None => (&current.texture.1, nearest),
                };
                draw_kernel_quads(
                    descriptors,
                    &mut render_pass,
                    source,
                    sampler,
                    &kernel_uniforms,
                    vertices,
                    &vec![copy; *count],
                );
            }
            render_pass.set_index_buffer(
                descriptors.quad.indices.slice(..),
                wgpu::IndexFormat::Uint32,
            );
            if !glow_bind_groups.is_empty() {
                render_pass.set_pipeline(glow.linear_pipeline(descriptors));
                render_pass.set_vertex_buffer(0, glow_vertices.slice(..));
                for (index, bind_group) in glow_bind_groups.iter().enumerate() {
                    render_pass.set_bind_group(0, bind_group, &[]);
                    render_pass.draw_indexed(0..6, index as i32 * 4, 0..1);
                }
            }
            if !color_matrix_bind_groups.is_empty() {
                render_pass.set_pipeline(color_matrix.pipeline(descriptors, 1));
                render_pass.set_vertex_buffer(0, color_matrix_vertices.slice(..));
                for (index, bind_group) in color_matrix_bind_groups.iter().enumerate() {
                    render_pass.set_bind_group(0, bind_group, &[]);
                    render_pass.draw_indexed(0..6, index as i32 * 4, 0..1);
                }
            }
            drop(render_pass);
            current = target;
        }

        for &(index, region) in &batch {
            let texture = &entries[index].0;
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
