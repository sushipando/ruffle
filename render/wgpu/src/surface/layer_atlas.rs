//! Batching of small, simple blend layers into a shared texture.
//!
//! Every blend layer used to be rendered with its own render pass. Content that
//! uses blend modes on lots of small objects (e.g. the parts of animated objects)
//! ends up with hundreds of tiny passes per frame, and each pass has a large
//! fixed CPU and GPU cost. Layers that only contain plain draws don't depend on
//! anything else, so they are instead all rendered into one atlas texture in a
//! single pass, and then copied out into their own textures.

use crate::MaskState;
use crate::backend::RenderTargetMode;
use crate::buffer_pool::TexturePool;
use crate::descriptors::Descriptors;
use crate::dynamic_transforms::DynamicTransforms;
use crate::mesh::Mesh;
use crate::surface::commands::{Chunk, CommandRenderer, chunk_leaf_layers};
use crate::surface::target::CommandTarget;
use ruffle_render::commands::{Command, CommandList};
use ruffle_render::quality::StageQuality;
use wgpu_profiler::Scope;

/// Layers larger than this (in either dimension) are rendered on their own.
const MAX_LAYER_SIZE: u32 = 1024;

/// The maximum size of an atlas texture (in either dimension).
const MAX_ATLAS_SIZE: u32 = 4096;

pub const LAYER_TEXTURE_USAGE: wgpu::TextureUsages = wgpu::TextureUsages::RENDER_ATTACHMENT
    .union(wgpu::TextureUsages::COPY_SRC)
    .union(wgpu::TextureUsages::COPY_DST)
    .union(wgpu::TextureUsages::TEXTURE_BINDING);

/// A blend layer whose rendering has been deferred to the atlas.
pub struct LeafLayer {
    pub commands: CommandList,
    /// The pixel coordinate of the top-left of the layer.
    pub origin: (i32, i32),
    pub width: u32,
    pub height: u32,
    /// The texture the rendered layer must end up in.
    pub destination: wgpu::Texture,
}

/// Whether a layer with these commands can be rendered in the atlas: it must only
/// contain plain draws (no masks or nested layers), and be reasonably small.
pub fn is_leaf(commands: &CommandList, width: u32, height: u32) -> bool {
    width <= MAX_LAYER_SIZE
        && height <= MAX_LAYER_SIZE
        && commands.commands.iter().all(|command| {
            matches!(
                command,
                Command::RenderBitmap { .. }
                    | Command::RenderShape { .. }
                    | Command::DrawRect { .. }
                    | Command::DrawLine { .. }
                    | Command::DrawLineRect { .. }
            )
        })
}

/// Where a layer was placed in an atlas.
pub struct Slot {
    pub x: u32,
    pub y: u32,
}

/// Layers (as indices with their slots) that fit in one atlas, and the atlas' size.
type AtlasBatch = (Vec<(usize, Slot)>, u32, u32);

/// Packs layers into rows ("shelves"), tallest first. Returns the batches of
/// `(layer index, slot)` that each fit in one atlas, with each atlas' size.
fn pack(layers: &[LeafLayer], max_size: u32) -> Vec<AtlasBatch> {
    let mut order: Vec<usize> = (0..layers.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(layers[i].height));

    let mut batches = vec![];
    let mut batch = vec![];
    let (mut x, mut y, mut shelf_height, mut used_width) = (0, 0, 0, 0);
    for index in order {
        let layer = &layers[index];
        if x + layer.width > max_size {
            // Start a new shelf.
            x = 0;
            y += shelf_height;
            shelf_height = 0;
        }
        if y + layer.height > max_size {
            // Start a new atlas.
            batches.push((std::mem::take(&mut batch), used_width, y + shelf_height));
            (x, y, shelf_height, used_width) = (0, 0, 0, 0);
        }
        batch.push((index, Slot { x, y }));
        x += layer.width;
        used_width = used_width.max(x);
        shelf_height = shelf_height.max(layer.height);
    }
    if !batch.is_empty() {
        batches.push((batch, used_width, y + shelf_height));
    }
    batches
}

#[expect(clippy::too_many_arguments)]
pub fn render_leaf_layers(
    layers: Vec<LeafLayer>,
    descriptors: &Descriptors,
    meshes: &Vec<Mesh>,
    staging_belt: &mut wgpu::util::StagingBelt,
    dynamic_transforms: &DynamicTransforms,
    draw_encoder: &mut Scope<'_, wgpu::CommandEncoder>,
    texture_pool: &mut TexturePool,
    quality: StageQuality,
) {
    let max_size = MAX_ATLAS_SIZE.min(descriptors.limits.max_texture_dimension_2d);
    for (batch, width, height) in pack(&layers, max_size) {
        // Round sizes up, so that textures can be reused between frames.
        let width = width.next_multiple_of(256).min(max_size);
        let height = height.next_multiple_of(256).min(max_size);
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let sample_count = crate::utils::supported_sample_count(
            &descriptors.adapter,
            quality.sample_count(),
            format,
        );
        let pipelines = descriptors.pipelines(sample_count, format);
        let target = CommandTarget::new(
            descriptors,
            texture_pool,
            size,
            format,
            sample_count,
            RenderTargetMode::FreshWithColor(wgpu::Color::TRANSPARENT),
            draw_encoder,
        );

        let chunks = chunk_leaf_layers(
            batch.iter().map(|(index, slot)| {
                let layer = &layers[*index];
                (
                    layer.commands.clone(),
                    (
                        slot.x as i32 - layer.origin.0,
                        slot.y as i32 - layer.origin.1,
                    ),
                    (slot.x, slot.y, layer.width, layer.height),
                )
            }),
            descriptors,
            staging_belt,
            dynamic_transforms,
            draw_encoder,
            meshes,
            quality,
            width,
            height,
            texture_pool,
        );

        let mut chunks: std::collections::VecDeque<_> = chunks
            .into_iter()
            .map(|chunk| {
                let Chunk::Draw {
                    chunk,
                    needs_stencil: _,
                    transforms,
                    vertices,
                } = chunk
                else {
                    unreachable!("Leaf layers only contain draws");
                };
                (chunk, transforms, vertices)
            })
            .collect();
        while !chunks.is_empty() {
            // As many chunks per pass as possible (see `Surface::draw_commands`).
            let mut group = vec![];
            while let Some((chunk, transforms, vertices)) = chunks.pop_front() {
                let offsets =
                    dynamic_transforms.upload(transforms, vertices, staging_belt, draw_encoder);
                group.push((chunk, offsets));
                if offsets.fallback
                    || !ruffle_render::perf_flags::enabled(
                        ruffle_render::perf_flags::PerfFlag::MergeChunks,
                    )
                {
                    break;
                }
            }
            crate::perf_counters::count(crate::perf_counters::Counter::RenderPass);
            let mut render_pass =
                draw_encoder.scoped_render_pass(
                    "Layer atlas",
                    wgpu::RenderPassDescriptor {
                        color_attachments: &[target.color_attachments_for_pass(chunks.is_empty())],
                        ..Default::default()
                    },
                );
            render_pass.set_bind_group(0, target.globals().bind_group(), &[]);
            for (chunk, offsets) in &group {
                render_pass.set_bind_group(
                    1,
                    &dynamic_transforms.bind_group,
                    &[offsets.transforms],
                );
                let mut renderer = CommandRenderer::new(
                    &pipelines,
                    descriptors,
                    &dynamic_transforms.vertex_buffer,
                    offsets.vertices,
                    0,
                    MaskState::NoMask,
                    false,
                );
                for command in chunk {
                    renderer.execute(&mut render_pass.scope(command.name()), command);
                }
            }
        }
        target.ensure_resolved(draw_encoder);

        for (index, slot) in &batch {
            let layer = &layers[*index];
            draw_encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: target.color_texture(),
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: slot.x,
                        y: slot.y,
                        z: 0,
                    },
                    aspect: Default::default(),
                },
                layer.destination.as_image_copy(),
                wgpu::Extent3d {
                    width: layer.width,
                    height: layer.height,
                    depth_or_array_layers: 1,
                },
            );
        }
        if crate::perf_counters::enabled() {
            crate::perf_counters::count_named(format!(
                "atlas/{}layers",
                batch.len().min(1000) / 50 * 50
            ));
        }
    }
}

/// Whether a filter only depends on pixels near each output pixel, so that it can
/// be applied to many sources at once in an atlas (with enough space around them).
pub fn is_local_filter(filter: &ruffle_render::filters::Filter) -> bool {
    use ruffle_render::filters::Filter;
    matches!(
        filter,
        Filter::BlurFilter(_)
            | Filter::GlowFilter(_)
            | Filter::DropShadowFilter(_)
            | Filter::ColorMatrixFilter(_)
    )
}

/// Applies the same `filters` to all of the given (already drawn) textures,
/// in place, by applying them once to an atlas containing all of them.
///
/// Each texture is surrounded by transparent space at least as wide as the filters
/// can spread (see `Filter::calculate_dest_rect`), so the result for each texture
/// only depends on that texture, as when the filters are applied to it alone.
///
/// Returns the indices of textures that were too large to batch, which the caller
/// must filter itself.
/// The largest cached bitmaps whose filters are applied in an atlas (in either dimension).
pub fn max_batched_filter_size() -> u32 {
    if ruffle_render::perf_flags::enabled(ruffle_render::perf_flags::PerfFlag::BatchLargeFilters) {
        1024
    } else {
        512
    }
}

/// The maximum size of a filter atlas (in either dimension).
fn max_filter_atlas_size() -> u32 {
    if ruffle_render::perf_flags::enabled(ruffle_render::perf_flags::PerfFlag::BatchLargeFilters) {
        4096
    } else {
        2048
    }
}

pub fn apply_filters_batched(
    filters: &[ruffle_render::filters::Filter],
    textures: &[wgpu::Texture],
    descriptors: &Descriptors,
    draw_encoder: &mut wgpu::CommandEncoder,
    texture_pool: &mut TexturePool,
    staging_belt: &mut wgpu::util::StagingBelt,
) -> Vec<usize> {
    use swf::{Rectangle, Twips};
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
    let gutter = reach + 2;

    // Shelf-pack the textures (with their gutters) into atlases.
    let max_size = max_filter_atlas_size().min(descriptors.limits.max_texture_dimension_2d);
    // Each batch: the (texture index, x, y) of its entries, and its atlas size.
    type Batch = (Vec<(usize, u32, u32)>, u32, u32);
    let mut batches: Vec<Batch> = vec![];
    let mut batch = vec![];
    let mut skipped = vec![];
    let (mut x, mut y, mut row_height, mut used_width) = (0, 0, 0, 0);
    for (index, texture) in textures.iter().enumerate() {
        let (w, h) = (texture.width() + gutter * 2, texture.height() + gutter * 2);
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
        batch.push((index, x + gutter, y + gutter));
        x += w;
        used_width = used_width.max(x);
        row_height = row_height.max(h);
    }
    if !batch.is_empty() {
        batches.push((batch, used_width, y + row_height));
    }

    for (batch, width, height) in batches {
        let size = wgpu::Extent3d {
            width: width.next_multiple_of(256).min(max_size),
            height: height.next_multiple_of(256).min(max_size),
            depth_or_array_layers: 1,
        };
        let atlas = texture_pool.get_texture(
            descriptors,
            size,
            LAYER_TEXTURE_USAGE,
            wgpu::TextureFormat::Rgba8Unorm,
            1,
        );
        crate::perf_counters::count(crate::perf_counters::Counter::RenderPass);
        draw_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: create_debug_label!("Clear filter atlas").as_deref(),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &atlas.1,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            ..Default::default()
        });
        for &(index, x, y) in &batch {
            let texture = &textures[index];
            draw_encoder.copy_texture_to_texture(
                texture.as_image_copy(),
                wgpu::TexelCopyTextureInfo {
                    texture: &atlas.0,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x, y, z: 0 },
                    aspect: Default::default(),
                },
                texture.size(),
            );
        }

        let mut target: Option<CommandTarget> = None;
        for filter in filters {
            crate::perf_counters::count(crate::perf_counters::Counter::Filter);
            let source = match &target {
                Some(target) => target.color_texture(),
                None => &atlas.0,
            };
            target = Some(descriptors.filters.apply(
                descriptors,
                draw_encoder,
                texture_pool,
                staging_belt,
                crate::filters::FilterSource::for_entire_texture(source),
                filter.clone(),
            ));
        }
        let Some(target) = target else {
            continue;
        };
        let result = target.color_texture();
        for &(index, x, y) in &batch {
            let texture = &textures[index];
            draw_encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: result,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x, y, z: 0 },
                    aspect: Default::default(),
                },
                texture.as_image_copy(),
                texture.size(),
            );
        }
        crate::perf_counters::count_named_n("filter-batched", batch.len() as u64);
    }
    skipped
}
