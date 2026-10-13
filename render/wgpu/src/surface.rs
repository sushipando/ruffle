mod commands;
pub mod layer_atlas;
pub mod target;
mod wave_schedule;

use crate::backend::RenderTargetMode;
use crate::blend::ComplexBlend;
use crate::buffer_pool::TexturePool;
use crate::dynamic_transforms::DynamicTransforms;
use crate::filters::FilterSource;
use crate::mesh::Mesh;
use crate::pixel_bender::{ShaderMode, run_pixelbender_shader_impl};
use crate::surface::commands::{Chunk, CommandRenderer, chunk_blends};
use crate::utils::run_copy_pipeline_into_region;
use crate::utils::supported_sample_count;
use crate::{Descriptors, MaskState, Pipelines};
use ruffle_render::commands::CommandList;
use ruffle_render::pixel_bender_support::{ImageInputTexture, PixelBenderShaderArgument};
use ruffle_render::quality::StageQuality;
use std::collections::VecDeque;
use std::sync::Arc;
use target::CommandTarget;
use tracing::instrument;
use wgpu_profiler::Scope;

pub use crate::surface::commands::LayerRef;

use self::commands::ChunkBlendMode;

#[derive(Debug)]
pub struct Surface {
    size: wgpu::Extent3d,
    quality: StageQuality,
    sample_count: u32,
    pipelines: Arc<Pipelines>,
    format: wgpu::TextureFormat,
    /// The pixel coordinate that maps to the top-left of this surface.
    origin: (i32, i32),
}

impl Surface {
    pub fn new(
        descriptors: &Descriptors,
        quality: StageQuality,
        width: u32,
        height: u32,
        frame_buffer_format: wgpu::TextureFormat,
    ) -> Self {
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let sample_count = supported_sample_count(
            &descriptors.adapter,
            quality.sample_count(),
            frame_buffer_format,
        );
        let pipelines = descriptors.pipelines(sample_count, frame_buffer_format);
        Self {
            size,
            quality,
            sample_count,
            pipelines,
            format: frame_buffer_format,
            origin: (0, 0),
        }
    }

    /// Makes this surface cover the region of pixel space starting at `origin`,
    /// instead of starting at (0, 0).
    pub fn with_origin(mut self, origin: (i32, i32)) -> Self {
        self.origin = origin;
        self
    }

    #[expect(clippy::too_many_arguments)]
    #[instrument(level = "debug", skip_all)]
    pub fn draw_commands_and_copy_to<'encoder, 'global: 'encoder>(
        &self,
        frame_view: &wgpu::TextureView,
        render_target_mode: RenderTargetMode,
        descriptors: &'global Descriptors,
        staging_belt: &'global mut wgpu::util::StagingBelt,
        dynamic_transforms: &'global DynamicTransforms,
        draw_encoder: &'encoder mut Scope<'global, wgpu::CommandEncoder>,
        meshes: &'global Vec<Mesh>,
        commands: CommandList,
        layer: LayerRef<'encoder>,
        texture_pool: &'global mut TexturePool,
        frame_region: Option<(i32, i32, u32, u32)>,
    ) {
        let target = self.draw_commands(
            render_target_mode,
            descriptors,
            meshes,
            commands,
            staging_belt,
            dynamic_transforms,
            draw_encoder,
            layer,
            texture_pool,
        );

        run_copy_pipeline_into_region(
            descriptors,
            self.format,
            frame_view,
            target.color_view(),
            target.globals(),
            1,
            draw_encoder,
            frame_region,
        );
    }

    #[expect(clippy::too_many_arguments)]
    #[instrument(level = "debug", skip_all)]
    pub fn draw_commands<'encoder, 'global: 'encoder>(
        &self,
        render_target_mode: RenderTargetMode,
        descriptors: &'encoder Descriptors,
        meshes: &'encoder Vec<Mesh>,
        commands: CommandList,
        staging_belt: &'encoder mut wgpu::util::StagingBelt,
        dynamic_transforms: &'encoder DynamicTransforms,
        draw_encoder: &'encoder mut Scope<'global, wgpu::CommandEncoder>,
        nearest_layer: LayerRef<'encoder>,
        texture_pool: &'encoder mut TexturePool,
    ) -> CommandTarget {
        // Everything drawn over an opaque background keeps it opaque
        // (except shader blends, which the command handler accounts for).
        let opaque = render_target_mode
            .color()
            .is_some_and(|color| color.a >= 1.0);
        let target = CommandTarget::new(
            descriptors,
            texture_pool,
            self.size,
            self.format,
            self.sample_count,
            render_target_mode,
            draw_encoder,
        );

        let mut num_masks = 0;
        let mut mask_state = MaskState::NoMask;
        // With fetch blends, the blends that waves batch don't split passes anymore.
        let fetch_blends = descriptors.shaders.fetch_blend_shaders.is_some()
            && ruffle_render::perf_flags::enabled(ruffle_render::perf_flags::PerfFlag::FetchBlends);
        let commands = if ruffle_render::perf_flags::enabled(
            ruffle_render::perf_flags::PerfFlag::WaveSchedule,
        ) && !fetch_blends
        {
            wave_schedule::schedule(
                commands,
                self.origin,
                self.size.width,
                self.size.height,
                opaque,
            )
        } else {
            commands
        };
        let (chunks, leaf_layers) = chunk_blends(
            commands,
            descriptors,
            staging_belt,
            dynamic_transforms,
            draw_encoder,
            meshes,
            self.quality,
            target.width(),
            target.height(),
            self.origin,
            opaque,
            match nearest_layer {
                LayerRef::Current => LayerRef::Parent(&target),
                layer => layer,
            },
            texture_pool,
        );

        if !leaf_layers.is_empty() {
            layer_atlas::render_leaf_layers(
                leaf_layers,
                descriptors,
                meshes,
                staging_belt,
                dynamic_transforms,
                draw_encoder,
                texture_pool,
                self.quality,
            );
        }

        // Complex blends whose composites haven't been drawn yet. They're drawn at the
        // start of the next render pass on `target`, to avoid passes just for them.
        // They never overlap each other, so their order doesn't matter.
        let mut pending_blends: Vec<PendingBlend> = vec![];
        let mut chunks: VecDeque<Chunk> = chunks.into();

        while let Some(chunk) = chunks.pop_front() {
            // The last pass on the target can also resolve it.
            let is_last = chunks.is_empty();
            match chunk {
                Chunk::Draw {
                    chunk,
                    needs_stencil,
                    transforms,
                    vertices,
                } => {
                    let offsets =
                        dynamic_transforms.upload(transforms, vertices, staging_belt, draw_encoder);
                    // Chunks only end because their transforms or vertices are full, so
                    // following ones are drawn in the same pass, each with its own offsets.
                    // (Only one may be in the arena's fallback region at a time.)
                    let mut group = vec![(chunk, offsets)];
                    while ruffle_render::perf_flags::enabled(
                        ruffle_render::perf_flags::PerfFlag::MergeChunks,
                    ) && !group.last().is_some_and(|(_, offsets)| offsets.fallback)
                        && matches!(
                            chunks.front(),
                            Some(Chunk::Draw { needs_stencil: next, .. }) if *next == needs_stencil
                        )
                    {
                        let Some(Chunk::Draw {
                            chunk,
                            transforms,
                            vertices,
                            ..
                        }) = chunks.pop_front()
                        else {
                            unreachable!();
                        };
                        let offsets = dynamic_transforms.upload(
                            transforms,
                            vertices,
                            staging_belt,
                            draw_encoder,
                        );
                        group.push((chunk, offsets));
                    }
                    let is_last = chunks.is_empty();
                    if pending_blends
                        .iter()
                        .any(|blend| blend.needs_stencil != needs_stencil)
                    {
                        self.draw_pending_blends_pass(
                            std::mem::take(&mut pending_blends),
                            &target,
                            descriptors,
                            draw_encoder,
                            texture_pool,
                            false,
                        );
                    }
                    crate::perf_counters::count(crate::perf_counters::Counter::RenderPass);
                    let mut render_pass = draw_encoder.scoped_render_pass(
                        format!(
                            "Chunked draw calls {}",
                            if needs_stencil {
                                "(with stencil)"
                            } else {
                                "(Stencilless)"
                            }
                        ),
                        wgpu::RenderPassDescriptor {
                            color_attachments: &[target.color_attachments_for_pass(is_last)],
                            depth_stencil_attachment: if needs_stencil {
                                target.stencil_attachment(descriptors, texture_pool)
                            } else {
                                None
                            },
                            ..Default::default()
                        },
                    );
                    for blend in pending_blends.drain(..) {
                        self.draw_pending_blend(&mut render_pass, &blend, &target, descriptors);
                    }
                    render_pass.set_bind_group(0, target.globals().bind_group(), &[]);
                    for (chunk, offsets) in &group {
                        render_pass.set_bind_group(
                            1,
                            &dynamic_transforms.bind_group,
                            &[offsets.transforms],
                        );
                        let mut renderer = CommandRenderer::new(
                            &self.pipelines,
                            descriptors,
                            &dynamic_transforms.vertex_buffer,
                            offsets.vertices,
                            num_masks,
                            mask_state,
                            needs_stencil,
                        );

                        for command in chunk {
                            if crate::utils::PROFILE_COMMANDS {
                                renderer.execute(&mut render_pass.scope(command.name()), command);
                            } else {
                                renderer.execute(&mut render_pass, command);
                            }
                        }

                        num_masks = renderer.num_masks();
                        mask_state = renderer.mask_state();
                    }
                }
                Chunk::Blend {
                    texture,
                    blend_mode: ChunkBlendMode::Shader(shader),
                    needs_stencil,
                    region: _,
                } => {
                    assert!(!needs_stencil, "Shader blend mode not implemented in masks");
                    if !pending_blends.is_empty() {
                        self.draw_pending_blends_pass(
                            std::mem::take(&mut pending_blends),
                            &target,
                            descriptors,
                            draw_encoder,
                            texture_pool,
                            false,
                        );
                    }
                    let parent_blend_buffer =
                        target.update_blend_buffer(descriptors, texture_pool, draw_encoder);
                    run_pixelbender_shader_impl(
                        descriptors,
                        shader,
                        ShaderMode::Filter,
                        &[
                            PixelBenderShaderArgument::ImageInput {
                                index: 0,
                                channels: 0xFF,
                                name: "background".to_string(),
                                texture: Some(ImageInputTexture::TextureRef(
                                    parent_blend_buffer.texture(),
                                )),
                            },
                            PixelBenderShaderArgument::ImageInput {
                                index: 1,
                                channels: 0xff,
                                name: "foreground".to_string(),
                                texture: Some(ImageInputTexture::TextureRef(texture.texture())),
                            },
                        ],
                        parent_blend_buffer.texture(),
                        draw_encoder,
                        target.color_attachments_for_pass(is_last),
                        target.sample_count(),
                        &FilterSource::for_entire_texture(texture.texture()),
                    )
                    .expect("Failed to run PixelBender blend mode");
                }
                Chunk::Blend {
                    texture,
                    blend_mode: ChunkBlendMode::Complex(blend_mode),
                    needs_stencil,
                    region,
                } => {
                    // Following complex blends that don't overlap this one (or each other)
                    // can use the same backdrop, so they're handled together.
                    let is_batchable = |mode: ComplexBlend| {
                        !matches!(mode, ComplexBlend::Alpha | ComplexBlend::Erase)
                    };
                    let mut batch = vec![(texture, blend_mode, needs_stencil, region)];
                    if let Some(region) = region
                        && is_batchable(blend_mode)
                    {
                        let mut regions = vec![region];
                        while let Some(Chunk::Blend {
                            blend_mode: ChunkBlendMode::Complex(next_mode),
                            region: Some(next_region),
                            ..
                        }) = chunks.front()
                            && is_batchable(*next_mode)
                            && !regions.iter().any(|r| r.overlaps(next_region))
                        {
                            regions.push(*next_region);
                            let Some(Chunk::Blend {
                                texture,
                                blend_mode: ChunkBlendMode::Complex(mode),
                                needs_stencil,
                                region,
                            }) = chunks.pop_front()
                            else {
                                unreachable!();
                            };
                            batch.push((texture, mode, needs_stencil, region));
                        }
                    }

                    // The backdrops must include any earlier blends that they overlap.
                    let overlaps_pending = pending_blends.iter().any(|pending| {
                        batch
                            .iter()
                            .any(|(_, _, _, region)| match (pending.region, region) {
                                (Some(a), Some(b)) => a.overlaps(b),
                                _ => true,
                            })
                    });
                    if overlaps_pending {
                        self.draw_pending_blends_pass(
                            std::mem::take(&mut pending_blends),
                            &target,
                            descriptors,
                            draw_encoder,
                            texture_pool,
                            false,
                        );
                    }

                    // Bounded blends only ever use `target` as their parent
                    // (Alpha and Erase are never bounded), so regions are in its space.
                    let backdrops: Vec<Option<target::BlendBuffer>> = if batch.len() > 1 {
                        crate::perf_counters::count_named("complex-batch");
                        crate::perf_counters::count_named_n("complex-batched", batch.len() as u64);
                        let regions: Vec<_> = batch
                            .iter()
                            .filter_map(|(_, _, _, region)| *region)
                            .collect();
                        target
                            .copy_regions_for_blends(
                                descriptors,
                                texture_pool,
                                draw_encoder,
                                &regions,
                            )
                            .into_iter()
                            .map(Some)
                            .collect()
                    } else {
                        vec![region.map(|region| {
                            target.copy_region_for_blend(
                                descriptors,
                                texture_pool,
                                draw_encoder,
                                region,
                            )
                        })]
                    };

                    for ((texture, blend_mode, needs_stencil, region), backdrop) in
                        batch.into_iter().zip(backdrops)
                    {
                        let parent = match blend_mode {
                            ComplexBlend::Alpha | ComplexBlend::Erase => {
                                match nearest_layer {
                                    LayerRef::None => {
                                        // An Alpha or Erase with no Layer above it should be ignored
                                        continue;
                                    }
                                    LayerRef::Current => &target,
                                    LayerRef::Parent(layer) => layer,
                                }
                            }
                            _ => &target,
                        };
                        let parent_blend_buffer = match &backdrop {
                            Some(buffer) => buffer,
                            None => {
                                parent.update_blend_buffer(descriptors, texture_pool, draw_encoder)
                            }
                        };

                        let bind_group =
                            descriptors
                                .device
                                .create_bind_group(&wgpu::BindGroupDescriptor {
                                    label: create_debug_label!(
                                        "Complex blend binds {:?} {}",
                                        blend_mode,
                                        if needs_stencil {
                                            "(with stencil)"
                                        } else {
                                            "(Stencilless)"
                                        }
                                    )
                                    .as_deref(),
                                    layout: &descriptors.bind_layouts.blend,
                                    entries: &[
                                        wgpu::BindGroupEntry {
                                            binding: 0,
                                            resource: wgpu::BindingResource::TextureView(
                                                parent_blend_buffer.view(),
                                            ),
                                        },
                                        wgpu::BindGroupEntry {
                                            binding: 1,
                                            resource: wgpu::BindingResource::TextureView(
                                                texture.view(),
                                            ),
                                        },
                                        wgpu::BindGroupEntry {
                                            binding: 2,
                                            resource: wgpu::BindingResource::Sampler(
                                                descriptors
                                                    .bitmap_samplers
                                                    .get_sampler(false, false),
                                            ),
                                        },
                                    ],
                                });

                        pending_blends.push(PendingBlend {
                            bind_group,
                            blend_mode,
                            needs_stencil,
                            region,
                            num_masks,
                            mask_state,
                            _texture: texture,
                            _region_blend_buffer: backdrop,
                        });
                    }
                }
            }
        }

        if !pending_blends.is_empty() {
            self.draw_pending_blends_pass(
                pending_blends,
                &target,
                descriptors,
                draw_encoder,
                texture_pool,
                true,
            );
        }

        // If nothing happened, ensure it's cleared so we don't operate on garbage data,
        // and make sure everything drawn is resolved.
        target.ensure_resolved(draw_encoder);

        target
    }

    /// Draws the composite of a complex blend into an already started render pass on `target`.
    fn draw_pending_blend(
        &self,
        render_pass: &mut wgpu::RenderPass<'_>,
        blend: &PendingBlend,
        target: &CommandTarget,
        descriptors: &Descriptors,
    ) {
        render_pass.set_bind_group(0, target.globals().bind_group(), &[]);
        if let Some(region) = blend.region {
            render_pass.set_viewport(
                region.x as f32,
                region.y as f32,
                region.width as f32,
                region.height as f32,
                0.0,
                1.0,
            );
        }

        let pipeline = &self.pipelines.complex_blends[blend.blend_mode];
        if blend.needs_stencil {
            match blend.mask_state {
                MaskState::NoMask => {}
                MaskState::DrawMaskStencil => {
                    render_pass.set_stencil_reference(blend.num_masks - 1);
                }
                MaskState::DrawMaskedContent | MaskState::ClearMaskStencil => {
                    render_pass.set_stencil_reference(blend.num_masks);
                }
            }
            render_pass.set_pipeline(pipeline.pipeline_for(blend.mask_state));
        } else {
            render_pass.set_pipeline(pipeline.stencilless_pipeline());
        }

        render_pass.set_bind_group(2, &blend.bind_group, &[]);
        render_pass.set_vertex_buffer(0, descriptors.quad.vertices_pos.slice(..));
        render_pass.set_index_buffer(
            descriptors.quad.indices.slice(..),
            wgpu::IndexFormat::Uint32,
        );
        render_pass.draw_indexed(0..6, 0, 0..1);

        if blend.region.is_some() {
            // Restore the default viewport for any draws after this.
            render_pass.set_viewport(
                0.0,
                0.0,
                target.width() as f32,
                target.height() as f32,
                0.0,
                1.0,
            );
        }
    }

    /// Draws the composites of complex blends in their own render pass(es).
    fn draw_pending_blends_pass(
        &self,
        blends: Vec<PendingBlend>,
        target: &CommandTarget,
        descriptors: &Descriptors,
        draw_encoder: &mut wgpu::CommandEncoder,
        texture_pool: &mut TexturePool,
        is_last: bool,
    ) {
        // Blends drawn with and without stencil need differently configured passes.
        let (with_stencil, without_stencil): (Vec<_>, Vec<_>) =
            blends.into_iter().partition(|blend| blend.needs_stencil);
        let groups = [(true, with_stencil), (false, without_stencil)];
        let num_groups = groups
            .iter()
            .filter(|(_, blends)| !blends.is_empty())
            .count();
        for (index, (needs_stencil, blends)) in groups
            .into_iter()
            .filter(|(_, blends)| !blends.is_empty())
            .enumerate()
        {
            crate::perf_counters::count(crate::perf_counters::Counter::RenderPass);
            let mut render_pass = draw_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: create_debug_label!("Complex blends").as_deref(),
                color_attachments: &[
                    target.color_attachments_for_pass(is_last && index + 1 == num_groups)
                ],
                depth_stencil_attachment: if needs_stencil {
                    target.stencil_attachment(descriptors, texture_pool)
                } else {
                    None
                },
                ..Default::default()
            });
            for blend in &blends {
                self.draw_pending_blend(&mut render_pass, blend, target, descriptors);
            }
        }
    }

    pub fn quality(&self) -> StageQuality {
        self.quality
    }

    pub fn sample_count(&self) -> u32 {
        self.sample_count
    }

    pub fn size(&self) -> wgpu::Extent3d {
        self.size
    }
}

/// A complex blend that is ready to be composited onto its parent target.
struct PendingBlend {
    bind_group: wgpu::BindGroup,
    blend_mode: ComplexBlend,
    needs_stencil: bool,
    region: Option<target::BlendRegion>,
    num_masks: u32,
    mask_state: MaskState,
    // Kept alive (out of the texture pool) until the blend is drawn.
    _texture: target::PoolOrArcTexture,
    _region_blend_buffer: Option<target::BlendBuffer>,
}
