use super::layer_atlas;
use super::target::PoolOrArcTexture;
use crate::backend::RenderTargetMode;
use crate::blend::TrivialBlend;
use crate::blend::{BlendType, ComplexBlend};
use crate::buffer_builder::BufferBuilder;
use crate::buffer_pool::TexturePool;
use crate::dynamic_transforms::DynamicTransforms;
use crate::mesh::{DrawType, Mesh, as_mesh};
use crate::surface::Surface;
use crate::surface::target::{BlendRegion, CommandTarget};
use crate::{Descriptors, MaskState, Pipelines, PosUvVertex, Transforms, as_texture};
use ruffle_render::backend::ShapeHandle;
use ruffle_render::bitmap::{BitmapHandle, PixelRegion, PixelSnapping};
use ruffle_render::commands::{Command, CommandHandler, CommandList, RenderBlendMode};
use ruffle_render::lines::{emulate_line, emulate_line_rect};
use ruffle_render::matrix::Matrix;
use ruffle_render::pixel_bender::PixelBenderShaderHandle;
use ruffle_render::quality::StageQuality;
use ruffle_render::transform::Transform;
use std::mem;
use swf::{BlendMode, Color, ColorTransform, Rectangle, Twips};
use wgpu::Backend;
use wgpu_profiler::Scope;

pub struct CommandRenderer<'encoder> {
    pipelines: &'encoder Pipelines,
    descriptors: &'encoder Descriptors,
    num_masks: u32,
    mask_state: MaskState,
    needs_stencil: bool,
    dynamic_vertex_buffer: &'encoder wgpu::Buffer,
    /// Where this chunk's vertices start in `dynamic_vertex_buffer`.
    dynamic_vertex_base: wgpu::BufferAddress,
    atlas_bind_groups: &'encoder [wgpu::BindGroup],
    filter_atlas_bind_groups: &'encoder [wgpu::BindGroup],
    last_pipeline: Option<*const wgpu::RenderPipeline>,
    last_bind_group_2: Option<&'encoder wgpu::BindGroup>,
    last_stencil_ref: Option<u32>,
    last_vertex_buffer: Option<(*const wgpu::Buffer, u64, u64)>,
    last_index_buffer: Option<(*const wgpu::Buffer, u64, u64)>,
}

impl<'encoder> CommandRenderer<'encoder> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        pipelines: &'encoder Pipelines,
        descriptors: &'encoder Descriptors,
        dynamic_vertex_buffer: &'encoder wgpu::Buffer,
        dynamic_vertex_base: wgpu::BufferAddress,
        atlas_bind_groups: &'encoder [wgpu::BindGroup],
        filter_atlas_bind_groups: &'encoder [wgpu::BindGroup],
        num_masks: u32,
        mask_state: MaskState,
        needs_stencil: bool,
    ) -> Self {
        Self {
            dynamic_vertex_base,
            pipelines,
            num_masks,
            mask_state,
            descriptors,
            needs_stencil,
            dynamic_vertex_buffer,
            atlas_bind_groups,
            filter_atlas_bind_groups,
            last_pipeline: None,
            last_bind_group_2: None,
            last_stencil_ref: None,
            last_vertex_buffer: None,
            last_index_buffer: None,
        }
    }

    #[inline]
    fn set_pipeline(
        &mut self,
        render_pass: &mut wgpu::RenderPass<'encoder>,
        pipeline: &'encoder wgpu::RenderPipeline,
    ) {
        let ptr = pipeline as *const wgpu::RenderPipeline;
        if self.last_pipeline != Some(ptr) {
            self.last_pipeline = Some(ptr);
            self.last_bind_group_2 = None;
            render_pass.set_pipeline(pipeline);
        }
    }

    #[inline]
    fn set_bind_group_2(
        &mut self,
        render_pass: &mut wgpu::RenderPass<'encoder>,
        bind_group: &'encoder wgpu::BindGroup,
    ) {
        if self.last_bind_group_2 != Some(bind_group) {
            self.last_bind_group_2 = Some(bind_group);
            render_pass.set_bind_group(2, bind_group, &[]);
        }
    }

    #[inline]
    fn set_stencil_ref(&mut self, render_pass: &mut wgpu::RenderPass<'encoder>, reference: u32) {
        if self.last_stencil_ref != Some(reference) {
            self.last_stencil_ref = Some(reference);
            render_pass.set_stencil_reference(reference);
        }
    }

    pub fn execute(
        &mut self,
        render_pass: &mut wgpu::RenderPass<'encoder>,
        command: &'encoder DrawCommand,
    ) {
        if self.needs_stencil {
            match self.mask_state {
                MaskState::NoMask => {}
                MaskState::DrawMaskStencil => {
                    self.set_stencil_ref(render_pass, self.num_masks - 1);
                }
                MaskState::DrawMaskedContent => {
                    self.set_stencil_ref(render_pass, self.num_masks);
                }
                MaskState::ClearMaskStencil => {
                    self.set_stencil_ref(render_pass, self.num_masks);
                }
            }
        }

        match command {
            DrawCommand::RenderBitmap {
                bitmap,
                instance_index,
                vertex_offset,
                smoothing,
                blend_mode,
                render_stage3d,
            } => self.render_bitmap(
                render_pass,
                bitmap,
                *instance_index,
                *smoothing,
                *blend_mode,
                *render_stage3d,
                *vertex_offset,
            ),
            DrawCommand::RenderTexture {
                _texture,
                binds,
                instance_index,
                vertex_offset,
                blend_mode,
            } => self.render_texture(
                render_pass,
                *instance_index,
                binds,
                *vertex_offset,
                *blend_mode,
            ),
            DrawCommand::FetchBlend {
                _texture,
                binds,
                instance_index,
                vertex_offset,
                blend_mode,
            } => self.render_fetch_blend(
                render_pass,
                *instance_index,
                binds,
                *vertex_offset,
                *blend_mode,
            ),
            DrawCommand::RenderAtlasLayer {
                atlas_index,
                instance_index,
                vertex_offset,
                blend_mode,
                ..
            } => self.render_atlas_layer(
                render_pass,
                *atlas_index,
                *instance_index,
                *vertex_offset,
                *blend_mode,
            ),
            DrawCommand::RenderFilterAtlas {
                atlas_index,
                instance_index,
                vertex_offset,
                blend_mode,
            } => self.render_filter_atlas(
                render_pass,
                *atlas_index,
                *instance_index,
                *vertex_offset,
                *blend_mode,
            ),
            DrawCommand::RenderShape {
                shape,
                instance_index,
                blend_mode,
            } => self.render_shape(render_pass, shape, *instance_index, *blend_mode),
            DrawCommand::DrawRect { instance_index } => {
                self.draw_rect(render_pass, *instance_index)
            }
            DrawCommand::DrawLine { instance_index } => {
                self.draw_lines::<false>(render_pass, *instance_index)
            }
            DrawCommand::DrawLineRect { instance_index } => {
                self.draw_lines::<true>(render_pass, *instance_index)
            }
            DrawCommand::PushMask => self.push_mask(render_pass),
            DrawCommand::ActivateMask => self.activate_mask(render_pass),
            DrawCommand::DeactivateMask => self.deactivate_mask(render_pass),
            DrawCommand::PopMask => self.pop_mask(render_pass),
            DrawCommand::SetScissor((x, y, width, height)) => {
                render_pass.set_scissor_rect(*x, *y, *width, *height)
            }
            DrawCommand::RenderAlphaMask {
                maskee,
                mask,
                binds,
                instance_index,
            } => self.render_alpha_mask(render_pass, maskee, mask, binds, *instance_index),
        }
    }

    pub fn prep_color(&mut self, render_pass: &mut wgpu::RenderPass<'encoder>) {
        let pipeline = if self.needs_stencil {
            self.pipelines.color.pipeline_for(self.mask_state)
        } else {
            self.pipelines.color.stencilless_pipeline()
        };
        self.set_pipeline(render_pass, pipeline);
    }

    pub fn prep_lines(&mut self, render_pass: &mut wgpu::RenderPass<'encoder>) {
        let pipeline = if self.needs_stencil {
            self.pipelines.lines.pipeline_for(self.mask_state)
        } else {
            self.pipelines.lines.stencilless_pipeline()
        };
        self.set_pipeline(render_pass, pipeline);
    }

    pub fn prep_gradient(
        &mut self,
        render_pass: &mut wgpu::RenderPass<'encoder>,
        bind_group: &'encoder wgpu::BindGroup,
    ) {
        let pipeline = if self.needs_stencil {
            self.pipelines.gradients.pipeline_for(self.mask_state)
        } else {
            self.pipelines.gradients.stencilless_pipeline()
        };
        self.set_pipeline(render_pass, pipeline);
        self.set_bind_group_2(render_pass, bind_group);
    }

    pub fn prep_bitmap(
        &mut self,
        render_pass: &mut wgpu::RenderPass<'encoder>,
        bind_group: &'encoder wgpu::BindGroup,
        blend_mode: TrivialBlend,
        render_stage3d: bool,
    ) {
        let pipeline = match (self.needs_stencil, render_stage3d) {
            (true, true) => &self.pipelines.bitmap_opaque_dummy_stencil,
            (true, false) => self.pipelines.bitmap[blend_mode].pipeline_for(self.mask_state),
            (false, true) => &self.pipelines.bitmap_opaque,
            (false, false) => self.pipelines.bitmap[blend_mode].stencilless_pipeline(),
        };
        self.set_pipeline(render_pass, pipeline);
        self.set_bind_group_2(render_pass, bind_group);
    }

    pub fn prep_alpha_mask(
        &mut self,
        render_pass: &mut wgpu::RenderPass<'encoder>,
        bind_group: &'encoder wgpu::BindGroup,
    ) {
        let pipeline = if self.needs_stencil {
            self.pipelines.alpha_mask.pipeline_for(self.mask_state)
        } else {
            self.pipelines.alpha_mask.stencilless_pipeline()
        };
        self.set_pipeline(render_pass, pipeline);
        self.set_bind_group_2(render_pass, bind_group);
    }

    pub fn draw(
        &mut self,
        render_pass: &mut wgpu::RenderPass<'encoder>,
        vertices: wgpu::BufferSlice<'encoder>,
        indices: wgpu::BufferSlice<'encoder>,
        num_indices: u32,
        instance_index: u32,
    ) {
        self.draw_with_base_vertex(render_pass, vertices, indices, num_indices, 0, instance_index);
    }

    pub fn draw_with_base_vertex(
        &mut self,
        render_pass: &mut wgpu::RenderPass<'encoder>,
        vertices: wgpu::BufferSlice<'encoder>,
        indices: wgpu::BufferSlice<'encoder>,
        num_indices: u32,
        base_vertex: i32,
        instance_index: u32,
    ) {
        let vb_key = (
            vertices.buffer() as *const wgpu::Buffer,
            vertices.offset(),
            vertices.size(),
        );
        if self.last_vertex_buffer != Some(vb_key) {
            self.last_vertex_buffer = Some(vb_key);
            render_pass.set_vertex_buffer(0, vertices);
        }
        let ib_key = (
            indices.buffer() as *const wgpu::Buffer,
            indices.offset(),
            indices.size(),
        );
        if self.last_index_buffer != Some(ib_key) {
            self.last_index_buffer = Some(ib_key);
            render_pass.set_index_buffer(indices, wgpu::IndexFormat::Uint32);
        }

        render_pass.draw_indexed(
            0..num_indices,
            base_vertex,
            instance_index..(instance_index + 1),
        );
    }

    #[inline]
    fn draw_quad_with_vertex_offset(
        &mut self,
        render_pass: &mut wgpu::RenderPass<'encoder>,
        vertex_offset: Option<wgpu::BufferAddress>,
        instance_index: u32,
    ) {
        if let Some(vertex_offset) = vertex_offset {
            let stride = size_of::<PosUvVertex>() as u64;
            debug_assert_eq!(vertex_offset % stride, 0);
            let base_vertex = (vertex_offset / stride) as i32;
            self.draw_with_base_vertex(
                render_pass,
                self.dynamic_vertex_buffer.slice(self.dynamic_vertex_base..),
                self.descriptors.quad.indices.slice(..),
                6,
                base_vertex,
                instance_index,
            );
        } else {
            self.draw(
                render_pass,
                self.descriptors.quad.vertices_pos_uv.slice(..),
                self.descriptors.quad.indices.slice(..),
                6,
                instance_index,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn render_bitmap(
        &mut self,
        render_pass: &mut wgpu::RenderPass<'encoder>,
        bitmap: &'encoder BitmapHandle,
        instance_index: u32,
        smoothing: bool,
        blend_mode: ShapeBlend,
        render_stage3d: bool,
        vertex_offset: Option<wgpu::BufferAddress>,
    ) {
        let texture = as_texture(bitmap);

        let descriptors = self.descriptors;
        let bind = texture.bind_group(
            false,
            smoothing,
            &descriptors.device,
            &descriptors.bind_layouts.bitmap,
            bitmap.clone(),
            &descriptors.bitmap_samplers,
        );
        match blend_mode {
            ShapeBlend::Trivial(mode) => {
                self.prep_bitmap(render_pass, &bind.bind_group, mode, render_stage3d);
            }
            ShapeBlend::Fetch(mode) => {
                let pipeline = self.pipelines.fetch_bitmap_blend(
                    &descriptors.device,
                    &descriptors.shaders,
                    &descriptors.bind_layouts,
                    mode,
                );
                let pipeline = if self.needs_stencil {
                    pipeline.pipeline_for(self.mask_state)
                } else {
                    pipeline.stencilless_pipeline()
                };
                self.set_pipeline(render_pass, pipeline);
                self.set_bind_group_2(render_pass, &bind.bind_group);
            }
        }

        self.draw_quad_with_vertex_offset(render_pass, vertex_offset, instance_index);
    }

    pub fn render_texture(
        &mut self,
        render_pass: &mut wgpu::RenderPass<'encoder>,
        instance_index: u32,
        bind_group: &'encoder wgpu::BindGroup,
        vertex_offset: Option<wgpu::BufferAddress>,
        blend_mode: TrivialBlend,
    ) {
        self.prep_bitmap(render_pass, bind_group, blend_mode, false);

        self.draw_quad_with_vertex_offset(render_pass, vertex_offset, instance_index);
    }

    pub fn render_fetch_blend(
        &mut self,
        render_pass: &mut wgpu::RenderPass<'encoder>,
        instance_index: u32,
        bind_group: &'encoder wgpu::BindGroup,
        vertex_offset: Option<wgpu::BufferAddress>,
        blend_mode: ComplexBlend,
    ) {
        let pipelines = &self
            .pipelines
            .fetch_blends
            .as_ref()
            .expect("fetch blends are only used where supported")[blend_mode];
        let pipeline = if self.needs_stencil {
            pipelines.pipeline_for(self.mask_state)
        } else {
            pipelines.stencilless_pipeline()
        };
        self.set_pipeline(render_pass, pipeline);
        self.set_bind_group_2(render_pass, bind_group);
        self.draw_quad_with_vertex_offset(render_pass, vertex_offset, instance_index);
    }

    pub fn render_atlas_layer(
        &mut self,
        render_pass: &mut wgpu::RenderPass<'encoder>,
        atlas_index: usize,
        instance_index: u32,
        vertex_offset: wgpu::BufferAddress,
        blend_mode: ShapeBlend,
    ) {
        let bind_group = &self.atlas_bind_groups[atlas_index];
        match blend_mode {
            ShapeBlend::Trivial(mode) => {
                self.prep_bitmap(render_pass, bind_group, mode, false);
            }
            ShapeBlend::Fetch(mode) => {
                let pipelines = &self
                    .pipelines
                    .fetch_blends
                    .as_ref()
                    .expect("fetch blends are only used where supported")[mode];
                let pipeline = if self.needs_stencil {
                    pipelines.pipeline_for(self.mask_state)
                } else {
                    pipelines.stencilless_pipeline()
                };
                self.set_pipeline(render_pass, pipeline);
                self.set_bind_group_2(render_pass, bind_group);
            }
        }
        self.draw_quad_with_vertex_offset(render_pass, Some(vertex_offset), instance_index);
    }

    pub fn render_filter_atlas(
        &mut self,
        render_pass: &mut wgpu::RenderPass<'encoder>,
        atlas_index: usize,
        instance_index: u32,
        vertex_offset: wgpu::BufferAddress,
        blend_mode: ShapeBlend,
    ) {
        let bind_group = &self.filter_atlas_bind_groups[atlas_index];
        match blend_mode {
            ShapeBlend::Trivial(mode) => {
                self.prep_bitmap(render_pass, bind_group, mode, false);
            }
            ShapeBlend::Fetch(mode) => {
                let pipeline = self.pipelines.fetch_bitmap_blend(
                    &self.descriptors.device,
                    &self.descriptors.shaders,
                    &self.descriptors.bind_layouts,
                    mode,
                );
                let pipeline = if self.needs_stencil {
                    pipeline.pipeline_for(self.mask_state)
                } else {
                    pipeline.stencilless_pipeline()
                };
                self.set_pipeline(render_pass, pipeline);
                self.set_bind_group_2(render_pass, bind_group);
            }
        }
        self.draw_quad_with_vertex_offset(render_pass, Some(vertex_offset), instance_index);
    }

    pub fn render_shape(
        &mut self,
        render_pass: &mut wgpu::RenderPass<'encoder>,
        shape: &'encoder ShapeHandle,
        instance_index: u32,
        blend_mode: ShapeBlend,
    ) {
        let (blend_mode, fetch) = match blend_mode {
            ShapeBlend::Trivial(mode) => (mode, None),
            ShapeBlend::Fetch(mode) => (TrivialBlend::Normal, Some(mode)),
        };
        let blended = (!matches!(blend_mode, TrivialBlend::Normal)).then(|| {
            self.pipelines.blended_fills(
                &self.descriptors.device,
                &self.descriptors.shaders,
                &self.descriptors.bind_layouts,
                blend_mode,
            )
        });
        let mesh = as_mesh(shape);
        for draw in &mesh.draws {
            let num_indices = if self.mask_state != MaskState::DrawMaskStencil
                && self.mask_state != MaskState::ClearMaskStencil
            {
                draw.num_indices
            } else {
                // Omit strokes when drawing a mask stencil.
                draw.num_mask_indices
            };
            if num_indices == 0 {
                continue;
            }

            match (&draw.draw_type, blended) {
                (DrawType::Color, None) if fetch.is_some() => {
                    let pipeline = self.pipelines.fetch_color_blend(
                        &self.descriptors.device,
                        &self.descriptors.shaders,
                        &self.descriptors.bind_layouts,
                        fetch.expect("checked above"),
                    );
                    let pipeline = if self.needs_stencil {
                        pipeline.pipeline_for(self.mask_state)
                    } else {
                        pipeline.stencilless_pipeline()
                    };
                    self.set_pipeline(render_pass, pipeline);
                }
                (DrawType::Color, None) => {
                    self.prep_color(render_pass);
                }
                (DrawType::Color, Some(blended)) => {
                    let pipeline = if self.needs_stencil {
                        blended.color.pipeline_for(self.mask_state)
                    } else {
                        blended.color.stencilless_pipeline()
                    };
                    self.set_pipeline(render_pass, pipeline);
                }
                (DrawType::Gradient { bind_group, .. }, None) if fetch.is_some() => {
                    let pipeline = self.pipelines.fetch_gradient_blend(
                        &self.descriptors.device,
                        &self.descriptors.shaders,
                        &self.descriptors.bind_layouts,
                        fetch.expect("checked above"),
                    );
                    let pipeline = if self.needs_stencil {
                        pipeline.pipeline_for(self.mask_state)
                    } else {
                        pipeline.stencilless_pipeline()
                    };
                    self.set_pipeline(render_pass, pipeline);
                    self.set_bind_group_2(render_pass, bind_group);
                }
                (DrawType::Gradient { bind_group, .. }, None) => {
                    self.prep_gradient(render_pass, bind_group);
                }
                (DrawType::Gradient { bind_group, .. }, Some(blended)) => {
                    let pipeline = if self.needs_stencil {
                        blended.gradient.pipeline_for(self.mask_state)
                    } else {
                        blended.gradient.stencilless_pipeline()
                    };
                    self.set_pipeline(render_pass, pipeline);
                    self.set_bind_group_2(render_pass, bind_group);
                }
                (DrawType::Bitmap { binds, .. }, None) if fetch.is_some() => {
                    let pipeline = self.pipelines.fetch_bitmap_blend(
                        &self.descriptors.device,
                        &self.descriptors.shaders,
                        &self.descriptors.bind_layouts,
                        fetch.expect("checked above"),
                    );
                    let pipeline = if self.needs_stencil {
                        pipeline.pipeline_for(self.mask_state)
                    } else {
                        pipeline.stencilless_pipeline()
                    };
                    self.set_pipeline(render_pass, pipeline);
                    self.set_bind_group_2(render_pass, &binds.bind_group);
                }
                (DrawType::Bitmap { binds, .. }, _) => {
                    self.prep_bitmap(render_pass, &binds.bind_group, blend_mode, false);
                }
            }

            self.draw(
                render_pass,
                mesh.buffer.slice(draw.vertices.clone()),
                mesh.buffer.slice(draw.indices.clone()),
                num_indices,
                instance_index,
            );
        }
    }

    pub fn render_alpha_mask(
        &mut self,
        render_pass: &mut wgpu::RenderPass<'encoder>,
        _maskee: &PoolOrArcTexture,
        _mask: &PoolOrArcTexture,
        bind_group: &'encoder wgpu::BindGroup,
        instance_index: u32,
    ) {
        if cfg!(feature = "render_debug_labels") {
            render_pass.push_debug_group("render_alpha_mask");
        }

        self.prep_alpha_mask(render_pass, bind_group);

        self.draw(
            render_pass,
            self.descriptors.quad.vertices_pos.slice(..),
            self.descriptors.quad.indices.slice(..),
            6,
            instance_index,
        );

        if cfg!(feature = "render_debug_labels") {
            render_pass.pop_debug_group();
        }
    }

    pub fn draw_rect(&mut self, render_pass: &mut wgpu::RenderPass<'encoder>, instance_index: u32) {
        self.prep_color(render_pass);

        self.draw(
            render_pass,
            self.descriptors.quad.vertices_pos_color.slice(..),
            self.descriptors.quad.indices.slice(..),
            6,
            instance_index,
        );
    }

    pub fn draw_lines<const RECT: bool>(
        &mut self,
        render_pass: &mut wgpu::RenderPass<'encoder>,
        instance_index: u32,
    ) {
        self.prep_lines(render_pass);

        self.draw(
            render_pass,
            self.descriptors.quad.vertices_pos_color.slice(..),
            if RECT {
                self.descriptors.quad.indices_line_rect.slice(..)
            } else {
                self.descriptors.quad.indices_line.slice(..)
            },
            if RECT { 5 } else { 2 },
            instance_index,
        );
    }

    pub fn push_mask(&mut self, render_pass: &mut wgpu::RenderPass<'encoder>) {
        debug_assert!(
            self.mask_state == MaskState::NoMask || self.mask_state == MaskState::DrawMaskedContent
        );
        self.num_masks += 1;
        self.mask_state = MaskState::DrawMaskStencil;
        self.set_stencil_ref(render_pass, self.num_masks - 1);
    }

    pub fn activate_mask(&mut self, render_pass: &mut wgpu::RenderPass<'encoder>) {
        debug_assert!(self.num_masks > 0 && self.mask_state == MaskState::DrawMaskStencil);
        self.mask_state = MaskState::DrawMaskedContent;
        self.set_stencil_ref(render_pass, self.num_masks);
    }

    pub fn deactivate_mask(&mut self, render_pass: &mut wgpu::RenderPass<'encoder>) {
        debug_assert!(self.num_masks > 0 && self.mask_state == MaskState::DrawMaskedContent);
        self.mask_state = MaskState::ClearMaskStencil;
        self.set_stencil_ref(render_pass, self.num_masks);
    }

    pub fn pop_mask(&mut self, render_pass: &mut wgpu::RenderPass<'encoder>) {
        debug_assert!(self.num_masks > 0 && self.mask_state == MaskState::ClearMaskStencil);
        self.num_masks -= 1;
        self.set_stencil_ref(render_pass, self.num_masks);
        if self.num_masks == 0 {
            self.mask_state = MaskState::NoMask;
        } else {
            self.mask_state = MaskState::DrawMaskedContent;
        };
    }

    pub fn num_masks(&self) -> u32 {
        self.num_masks
    }

    pub fn mask_state(&self) -> MaskState {
        self.mask_state
    }
}

pub enum Chunk {
    Draw {
        chunk: Vec<DrawCommand>,
        needs_stencil: bool,
        transforms: BufferBuilder,
        vertices: BufferBuilder,
    },
    Blend {
        texture: PoolOrArcTexture,
        blend_mode: ChunkBlendMode,
        needs_stencil: bool,
        /// The region of the target covered by `texture`, or `None` for the whole target.
        region: Option<BlendRegion>,
    },
}

#[derive(Debug)]
pub enum ChunkBlendMode {
    Complex(ComplexBlend),
    Shader(PixelBenderShaderHandle),
}

#[derive(Debug)]
pub enum DrawCommand {
    RenderBitmap {
        bitmap: BitmapHandle,
        instance_index: u32,
        vertex_offset: Option<wgpu::BufferAddress>,
        smoothing: bool,
        blend_mode: ShapeBlend,
        render_stage3d: bool,
    },
    RenderTexture {
        _texture: PoolOrArcTexture,
        binds: wgpu::BindGroup,
        instance_index: u32,
        vertex_offset: Option<wgpu::BufferAddress>,
        blend_mode: TrivialBlend,
    },
    /// A complex blend that reads the backdrop from the color attachment.
    FetchBlend {
        _texture: PoolOrArcTexture,
        binds: wgpu::BindGroup,
        instance_index: u32,
        vertex_offset: Option<wgpu::BufferAddress>,
        blend_mode: ComplexBlend,
    },
    /// A leaf blend layer sampled directly from its slot in a shared layer atlas.
    RenderAtlasLayer {
        leaf_index: usize,
        atlas_index: usize,
        instance_index: u32,
        vertex_offset: wgpu::BufferAddress,
        blend_mode: ShapeBlend,
    },
    /// A filtered CAB sampled directly from its slot in a shared filter atlas.
    RenderFilterAtlas {
        atlas_index: usize,
        instance_index: u32,
        vertex_offset: wgpu::BufferAddress,
        blend_mode: ShapeBlend,
    },
    RenderAlphaMask {
        maskee: PoolOrArcTexture,
        mask: PoolOrArcTexture,
        binds: wgpu::BindGroup,
        instance_index: u32,
    },
    RenderShape {
        shape: ShapeHandle,
        instance_index: u32,
        /// `Normal`, unless drawn directly with a blend mode (see `direct_blend`).
        blend_mode: ShapeBlend,
    },
    DrawRect {
        instance_index: u32,
    },
    DrawLine {
        instance_index: u32,
    },
    DrawLineRect {
        instance_index: u32,
    },
    PushMask,
    ActivateMask,
    DeactivateMask,
    PopMask,
    /// Sets the scissor rectangle `(x, y, width, height)` for subsequent draws.
    SetScissor((u32, u32, u32, u32)),
}

impl DrawCommand {
    pub fn name(&self) -> &'static str {
        match self {
            DrawCommand::RenderBitmap { .. } => "render bitmap",
            DrawCommand::RenderShape { .. } => "render shape",
            DrawCommand::RenderTexture { .. } => "render texture",
            DrawCommand::FetchBlend { .. } => "fetch blend",
            DrawCommand::RenderAtlasLayer { .. } => "render atlas layer",
            DrawCommand::RenderFilterAtlas { .. } => "render filter atlas",
            DrawCommand::DrawRect { .. } => "draw rect",
            DrawCommand::DrawLine { .. } => "draw line",
            DrawCommand::DrawLineRect { .. } => "draw line rect",
            DrawCommand::PushMask => "push mask",
            DrawCommand::ActivateMask => "activate mask",
            DrawCommand::DeactivateMask => "deactivate mask",
            DrawCommand::PopMask => "pop mask",
            DrawCommand::SetScissor(_) => "set scissor",
            DrawCommand::RenderAlphaMask { .. } => "render alpha mask",
        }
    }
}

#[derive(Copy, Clone)]
pub enum LayerRef<'a> {
    None,
    Current,
    Parent(&'a CommandTarget),
}

/// Replaces every blend with a RenderBitmap, with the subcommands rendered out to a temporary texture
/// Every complex blend will be its own item, but every other draw will be chunked together
pub type PendingBlendPromotion = (
    crate::filters::CabContentKey,
    crate::filters::BlendCacheAllocation,
);

#[expect(clippy::too_many_arguments)]
pub fn chunk_blends<'encoder, 'global: 'encoder>(
    commands: CommandList,
    descriptors: &'encoder Descriptors,
    staging_belt: &'encoder mut wgpu::util::StagingBelt,
    dynamic_transforms: &'encoder DynamicTransforms,
    draw_encoder: &'encoder mut Scope<'global, wgpu::CommandEncoder>,
    meshes: &'encoder Vec<Mesh>,
    quality: StageQuality,
    width: u32,
    height: u32,
    origin: (i32, i32),
    opaque: bool,
    nearest_layer: LayerRef,
    texture_pool: &'encoder mut TexturePool,
    filter_atlas: Option<&'encoder crate::filters::FilterAtlasResult>,
    blend_cache: Option<&'encoder mut crate::filters::BlendLayerCache>,
) -> (
    Vec<Chunk>,
    Vec<layer_atlas::LeafLayer>,
    Vec<PendingBlendPromotion>,
) {
    let mut handler = WgpuCommandHandler::new(
        descriptors,
        staging_belt,
        dynamic_transforms,
        draw_encoder,
        meshes,
        quality,
        width,
        height,
        origin,
        nearest_layer,
        texture_pool,
        filter_atlas,
        blend_cache,
    );
    handler.opaque = opaque;
    let chunks = handler.chunk_blends(commands);
    (
        chunks,
        mem::take(&mut handler.leaf_layers),
        mem::take(&mut handler.pending_blend_promotions),
    )
}

/// Fills in atlas bind-group indices and UV sub-rect vertices for every
/// `DrawCommand::RenderAtlasLayer` after `layer_atlas::render_leaf_layers` packs them.
pub fn bind_atlas_layers(chunks: &mut [Chunk], atlas_layers: &layer_atlas::AtlasLayers) {
    for chunk in chunks {
        let Chunk::Draw {
            chunk, vertices, ..
        } = chunk
        else {
            continue;
        };
        let bytes = vertices.bytes_mut();
        for command in chunk {
            if let DrawCommand::RenderAtlasLayer {
                leaf_index,
                atlas_index,
                vertex_offset,
                ..
            } = command
            {
                let (bg_index, [u0, v0, u1, v1]) = atlas_layers.placements[*leaf_index]
                    .expect("RenderAtlasLayer leaf must have an atlas placement");
                *atlas_index = bg_index;
                let quad = [
                    PosUvVertex::new(0.0, 0.0, u0, v0, 1.0),
                    PosUvVertex::new(1.0, 0.0, u1, v0, 1.0),
                    PosUvVertex::new(1.0, 1.0, u1, v1, 1.0),
                    PosUvVertex::new(0.0, 1.0, u0, v1, 1.0),
                ];
                let raw = bytemuck::cast_slice(&quad);
                let start = *vertex_offset as usize;
                bytes[start..start + raw.len()].copy_from_slice(raw);
            }
        }
    }
}

/// Chunks the draws of several leaf layers into one target (an atlas).
/// Each layer is given as its commands, a pixel translation to apply to them,
/// and the scissor rectangle `(x, y, width, height)` to clip them to.
#[expect(clippy::too_many_arguments)]
pub fn chunk_leaf_layers<'encoder, 'global: 'encoder>(
    layers: impl Iterator<Item = (CommandList, (i32, i32), (i32, i32), (u32, u32, u32, u32), bool)>,
    descriptors: &'encoder Descriptors,
    staging_belt: &'encoder mut wgpu::util::StagingBelt,
    dynamic_transforms: &'encoder DynamicTransforms,
    draw_encoder: &'encoder mut Scope<'global, wgpu::CommandEncoder>,
    meshes: &'encoder Vec<Mesh>,
    quality: StageQuality,
    width: u32,
    height: u32,
    texture_pool: &'encoder mut TexturePool,
    filter_atlas: Option<&'encoder crate::filters::FilterAtlasResult>,
) -> (Vec<Chunk>, Vec<layer_atlas::LeafLayer>) {
    let mut handler = WgpuCommandHandler::new(
        descriptors,
        staging_belt,
        dynamic_transforms,
        draw_encoder,
        meshes,
        quality,
        width,
        height,
        (0, 0),
        LayerRef::None,
        texture_pool,
        filter_atlas,
        None,
    );
    for (commands, (dx, dy), scissor_origin, scissor, is_layer) in layers {
        handler.scissor = Some(scissor);
        handler.scissor_origin = scissor_origin;
        handler.nearest_layer = if is_layer {
            LayerRef::Current
        } else {
            LayerRef::None
        };
        handler.current.push(DrawCommand::SetScissor(scissor));
        handler.translation = (Twips::from_pixels_i32(dx), Twips::from_pixels_i32(dy));
        commands.execute(&mut handler);
    }
    let chunks = handler.chunk_blends(CommandList::new());
    (chunks, handler.leaf_layers)
}

/// How a shape is drawn onto its target (see `WgpuCommandHandler::direct_blend_mode`).
#[derive(Clone, Copy, Debug)]
pub enum ShapeBlend {
    Trivial(TrivialBlend),
    /// A complex blend done by reading the backdrop from the color attachment (only for
    /// solid fills).
    Fetch(ComplexBlend),
}

struct WgpuCommandHandler<'encoder, 'global: 'encoder> {
    descriptors: &'encoder Descriptors,
    quality: StageQuality,
    width: u32,
    height: u32,
    /// The pixel coordinate of the top-left of the target being drawn to.
    origin: (i32, i32),
    nearest_layer: LayerRef<'encoder>,
    meshes: &'encoder Vec<Mesh>,
    staging_belt: &'encoder mut wgpu::util::StagingBelt,
    dynamic_transforms: &'encoder DynamicTransforms,
    draw_encoder: &'encoder mut Scope<'global, wgpu::CommandEncoder>,
    texture_pool: &'encoder mut TexturePool,
    emulate_lines: bool,

    result: Vec<Chunk>,
    current: Vec<DrawCommand>,
    /// Blend layers to be rendered in an atlas before the chunks are drawn.
    leaf_layers: Vec<layer_atlas::LeafLayer>,
    /// A translation applied to all draws (used when drawing into an atlas).
    translation: (Twips, Twips),
    /// The current scissor rectangle, to be restored when starting a new chunk.
    scissor: Option<(u32, u32, u32, u32)>,
    /// The world/parent pixel origin of the layer currently being drawn into `scissor`.
    scissor_origin: (i32, i32),
    /// Whether the target is known to be fully opaque (and stays so), allowing
    /// some complex blends to be done as trivial ones.
    opaque: bool,
    transforms: BufferBuilder,
    vertices: BufferBuilder,
    needs_stencil: bool,
    num_masks: i32,
    /// While drawing the content of a blend directly (instead of in a layer), its mode.
    direct_blend: Option<ShapeBlend>,
    /// Optional filtered CAB atlas for direct sampling without copy-out blits.
    filter_atlas: Option<&'encoder crate::filters::FilterAtlasResult>,
    /// Optional content-addressed cache for pure-vector offscreen blend layers.
    blend_cache: Option<&'encoder mut crate::filters::BlendLayerCache>,
    /// Leaf-layer cache promotions deferred until `render_leaf_layers` populates them.
    pending_blend_promotions: Vec<PendingBlendPromotion>,
}

impl<'encoder, 'global: 'encoder> WgpuCommandHandler<'encoder, 'global> {
    #[expect(clippy::too_many_arguments)]
    fn new(
        descriptors: &'encoder Descriptors,
        staging_belt: &'encoder mut wgpu::util::StagingBelt,
        dynamic_transforms: &'encoder DynamicTransforms,
        draw_encoder: &'encoder mut Scope<'global, wgpu::CommandEncoder>,
        meshes: &'encoder Vec<Mesh>,
        quality: StageQuality,
        width: u32,
        height: u32,
        origin: (i32, i32),
        nearest_layer: LayerRef<'encoder>,
        texture_pool: &'encoder mut TexturePool,
        filter_atlas: Option<&'encoder crate::filters::FilterAtlasResult>,
        blend_cache: Option<&'encoder mut crate::filters::BlendLayerCache>,
    ) -> Self {
        let transforms = Self::new_transforms(descriptors, dynamic_transforms);
        let vertices = Self::new_vertices(descriptors, dynamic_transforms);

        // DirectX does support drawing lines, but it's very inconsistent.
        // With MSAA, lines have 1.4px thickness, which makes them too thick.
        // Without MSAA, lines have 1px thickness, but their placement is sometimes off.
        let emulate_lines = descriptors.backend == Backend::Dx12;

        Self {
            descriptors,
            quality,
            width,
            height,
            origin,
            nearest_layer,
            meshes,
            staging_belt,
            dynamic_transforms,
            draw_encoder,
            texture_pool,
            emulate_lines,

            result: vec![],
            current: vec![],
            leaf_layers: vec![],
            opaque: false,
            // Draws are in absolute pixel coordinates, but the target starts at `origin`.
            translation: (
                Twips::from_pixels_i32(-origin.0),
                Twips::from_pixels_i32(-origin.1),
            ),
            scissor: None,
            scissor_origin: (0, 0),
            transforms,
            vertices,
            needs_stencil: false,
            num_masks: 0,
            direct_blend: None,
            filter_atlas,
            blend_cache,
            pending_blend_promotions: vec![],
        }
    }

    fn new_transforms(
        descriptors: &'encoder Descriptors,
        dynamic_transforms: &'encoder DynamicTransforms,
    ) -> BufferBuilder {
        let mut transforms = BufferBuilder::new(&descriptors.limits, 0);
        transforms.set_buffer_limit(dynamic_transforms.binding_size);
        transforms
    }

    fn new_vertices(
        descriptors: &'encoder Descriptors,
        dynamic_transforms: &'encoder DynamicTransforms,
    ) -> BufferBuilder {
        let mut vertices = BufferBuilder::new(&descriptors.limits, 0);
        vertices.set_buffer_limit(dynamic_transforms.chunk_vertices_size);
        vertices
    }

    /// Replaces every blend with a RenderBitmap, with the subcommands rendered out to a temporary texture
    /// Every complex blend will be its own item, but every other draw will be chunked together
    fn chunk_blends(&mut self, commands: CommandList) -> Vec<Chunk> {
        commands.execute(self);

        let current = mem::take(&mut self.current);
        let mut result = mem::take(&mut self.result);
        let needs_stencil = mem::take(&mut self.needs_stencil);
        let transforms = mem::replace(
            &mut self.transforms,
            Self::new_transforms(self.descriptors, self.dynamic_transforms),
        );
        let vertices = mem::replace(
            &mut self.vertices,
            Self::new_vertices(self.descriptors, self.dynamic_transforms),
        );

        if !current.is_empty() {
            result.push(Chunk::Draw {
                chunk: current,
                needs_stencil,
                transforms,
                vertices,
            });
        }

        result
    }

    fn add_to_current(
        &mut self,
        matrix: Matrix,
        tz: f64,
        color_transform: ColorTransform,
        command_builder: impl FnOnce(u32) -> DrawCommand,
    ) {
        self.add_to_current_with_vertices(matrix, tz, color_transform, None, |instance_index, _| {
            command_builder(instance_index)
        })
    }

    fn add_to_current_with_vertices(
        &mut self,
        matrix: Matrix,
        tz: f64,
        color_transform: ColorTransform,
        vertices: Option<&[PosUvVertex]>,
        command_builder: impl FnOnce(u32, Option<wgpu::BufferAddress>) -> DrawCommand,
    ) {
        let mut matrix = matrix;
        matrix.tx += self.translation.0;
        matrix.ty += self.translation.1;
        let transform = Transforms {
            world_matrix: [
                [matrix.a, matrix.b, 0.0, 0.0],
                [matrix.c, matrix.d, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [
                    matrix.tx.to_pixels() as f32,
                    matrix.ty.to_pixels() as f32,
                    tz as f32,
                    1.0,
                ],
            ],
            mult_color: color_transform.mult_rgba_normalized(),
            add_color: color_transform.add_rgba_normalized(),
        };
        if let (Ok(transform_range), Ok(vertices_range)) = (
            self.transforms.add(&[transform]),
            vertices.map(|v| self.vertices.add(v)).transpose(),
        ) {
            self.current.push(command_builder(
                (transform_range.start as usize / size_of::<Transforms>()) as u32,
                vertices_range.map(|v| v.start),
            ));
        } else {
            self.result.push(Chunk::Draw {
                chunk: mem::take(&mut self.current),
                needs_stencil: self.needs_stencil,
                transforms: mem::replace(
                    &mut self.transforms,
                    Self::new_transforms(self.descriptors, self.dynamic_transforms),
                ),
                vertices: mem::replace(
                    &mut self.vertices,
                    Self::new_vertices(self.descriptors, self.dynamic_transforms),
                ),
            });
            if let Some(scissor) = self.scissor {
                self.current.push(DrawCommand::SetScissor(scissor));
            }
            let transform_range = self
                .transforms
                .add(&[transform])
                .expect("Buffer must be able to fit a new thing, it was just emptied");
            let vertices_range = vertices.map(|v| {
                self.vertices
                    .add(v)
                    .expect("Buffer must be able to fit a new thing, it was just emptied")
            });
            self.current.push(command_builder(
                (transform_range.start as usize / size_of::<Transforms>()) as u32,
                vertices_range.map(|v| v.start),
            ));
        }
    }
}

impl WgpuCommandHandler<'_, '_> {
    /// Computes the region of the current target that a blend with the given bounds
    /// needs, in absolute pixel coordinates as `(x, y, width, height)`.
    ///
    /// Returns `Some(None)` if the blend is entirely outside of the target,
    /// and `None` if the whole target should be used instead.
    fn bounded_blend_rect(
        &self,
        blend_mode: &RenderBlendMode,
        bounds: Rectangle<Twips>,
    ) -> Option<Option<(i32, i32, u32, u32)>> {
        if !ruffle_render::perf_flags::enabled(ruffle_render::perf_flags::PerfFlag::BoundedBlends) {
            return None;
        }
        // Some blends read pixels outside of their own bounds.
        if !matches!(
            BlendType::from(blend_mode.clone()),
            BlendType::Trivial(_)
                | BlendType::Complex(
                    ComplexBlend::Multiply
                        | ComplexBlend::Lighten
                        | ComplexBlend::Darken
                        | ComplexBlend::Difference
                        | ComplexBlend::Invert
                        | ComplexBlend::Overlay
                        | ComplexBlend::HardLight
                )
        ) || !bounds.is_valid()
        {
            return None;
        }

        // Padding for anti-aliasing and bitmap smoothing at the edges.
        const PADDING: i32 = 2;
        let (ox, oy, target_w, target_h) = match self.scissor {
            Some((_, _, sw, sh)) => (self.scissor_origin.0, self.scissor_origin.1, sw, sh),
            None => (self.origin.0, self.origin.1, self.width, self.height),
        };
        let clamp_x = |v: f64| (v as i32).clamp(ox, ox + target_w as i32);
        let clamp_y = |v: f64| (v as i32).clamp(oy, oy + target_h as i32);
        let x0 = clamp_x(bounds.x_min.to_pixels().floor() - PADDING as f64);
        let y0 = clamp_y(bounds.y_min.to_pixels().floor() - PADDING as f64);
        let x1 = clamp_x(bounds.x_max.to_pixels().ceil() + PADDING as f64);
        let y1 = clamp_y(bounds.y_max.to_pixels().ceil() + PADDING as f64);
        if x1 <= x0 || y1 <= y0 {
            return Some(None);
        }
        // Round sizes up, so that textures can be reused between frames
        // as content moves and changes size.
        const SIZE_STEP: i32 = 32;
        let round = |start: i32, end: i32, min: i32, max: i32| {
            let size = ((end - start + SIZE_STEP - 1) / SIZE_STEP * SIZE_STEP).min(max - min);
            let start = start.min(max - size).max(min);
            (start, start + size)
        };
        let (x0, x1) = round(x0, x1, ox, ox + target_w as i32);
        let (y0, y1) = round(y0, y1, oy, oy + target_h as i32);
        let (width, height) = ((x1 - x0) as u32, (y1 - y0) as u32);

        // Keep bounded rects whenever the layer fits in the shared layer atlas, so we
        // don't drop out of `leaf_layers` batching into a standalone full-target pass.
        if !layer_atlas::fits(width, height)
            && (width as u64 * height as u64) * 4 > (target_w as u64 * target_h as u64) * 3
        {
            return None;
        }
        Some(Some((x0, y0, width, height)))
    }

    /// A bind group for drawing a blend layer's texture with the bitmap pipelines.
    /// Pooled textures keep theirs, since they're reused every frame.
    fn texture_bind_group(&self, texture: &PoolOrArcTexture) -> wgpu::BindGroup {
        let create = || {
            self.descriptors
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &self.descriptors.bind_layouts.bitmap,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(texture.view()),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(
                                self.descriptors.bitmap_samplers.get_sampler(false, false),
                            ),
                        },
                    ],
                    label: None,
                })
        };
        match texture {
            PoolOrArcTexture::Pool(entry)
                if ruffle_render::perf_flags::enabled(
                    ruffle_render::perf_flags::PerfFlag::CacheBindGroups,
                ) =>
            {
                entry.2.get_or_init(create).clone()
            }
            _ => create(),
        }
    }
}

/// The blend mode to draw `commands` with directly, without a layer, if that gives the
/// same result: when they're a single bitmap, or a single shape with a single fill,
/// which affect each pixel at most once.
fn command_screen_aabb(command: &Command) -> Option<[f32; 4]> {
    let (local, matrix) = match command {
        Command::RenderShape { shape, transform } => {
            let mesh = as_mesh(shape);
            if !(mesh.non_overlapping || (mesh.draws.len() == 1 && mesh.draws[0].single_fill)) {
                return None;
            }
            let (x0, y0, x1, y1) = mesh.bounds?;
            ((x0, y0, x1, y1), &transform.matrix)
        }
        Command::RenderBitmap {
            transform, region, ..
        } => {
            let w = region.width() as f32;
            let h = region.height() as f32;
            ((0.0, 0.0, w, h), &transform.matrix)
        }
        _ => return None,
    };
    let (x0, y0, x1, y1) = local;
    let tx = matrix.tx.to_pixels() as f32;
    let ty = matrix.ty.to_pixels() as f32;
    let corners = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)];
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for (x, y) in corners {
        let sx = matrix.a * x + matrix.c * y + tx;
        let sy = matrix.b * x + matrix.d * y + ty;
        min_x = min_x.min(sx);
        min_y = min_y.min(sy);
        max_x = max_x.max(sx);
        max_y = max_y.max(sy);
    }
    if !(min_x.is_finite() && min_y.is_finite() && max_x.is_finite() && max_y.is_finite()) {
        return None;
    }
    // Pad by 1px so adjacent shapes can't share an MSAA or bilinear-filtered pixel.
    Some([min_x - 1.0, min_y - 1.0, max_x + 1.0, max_y + 1.0])
}

fn unwrap_passthrough_blend_ref(mut cmds: &[Command]) -> &[Command] {
    while let [Command::Blend(inner, mode) | Command::BlendBounded(inner, mode, _)] = cmds {
        if matches!(
            mode,
            RenderBlendMode::Builtin(BlendMode::Normal | BlendMode::Layer)
        ) {
            cmds = inner.commands.as_slice();
        } else {
            break;
        }
    }
    cmds
}

fn unwrap_passthrough_blend_owned(mut commands: CommandList) -> CommandList {
    while commands.commands.len() == 1 {
        let is_passthrough = matches!(
            commands.commands.first(),
            Some(Command::Blend(_, mode) | Command::BlendBounded(_, mode, _))
                if matches!(
                    mode,
                    RenderBlendMode::Builtin(BlendMode::Normal | BlendMode::Layer)
                )
        );
        if !is_passthrough {
            break;
        }
        match commands.commands.pop().unwrap() {
            Command::Blend(inner, _) | Command::BlendBounded(inner, _, _) => commands = inner,
            _ => unreachable!(),
        }
    }
    commands
}

pub(crate) fn direct_blend_mode_for(
    descriptors: &Descriptors,
    commands: &CommandList,
    blend_mode: &RenderBlendMode,
    opaque: bool,
    in_current_layer: bool,
) -> Option<ShapeBlend> {
    if !ruffle_render::perf_flags::enabled(ruffle_render::perf_flags::PerfFlag::DirectBlends) {
        return None;
    }
    let single_fill = match unwrap_passthrough_blend_ref(commands.commands.as_slice()) {
        [command] => match command {
            Command::RenderShape { shape, .. } => {
                let mesh = as_mesh(shape);
                mesh.non_overlapping || (mesh.draws.len() == 1 && mesh.draws[0].single_fill)
            }
            Command::RenderBitmap { .. } => true,
            _ => false,
        },
        cmds if cmds.len() >= 2 && cmds.len() <= 16 => {
            let mut aabbs = [[0.0f32; 4]; 16];
            let mut ok = true;
            for (i, cmd) in cmds.iter().enumerate() {
                if let Some(aabb) = command_screen_aabb(cmd) {
                    for prev in &aabbs[..i] {
                        if aabb[0] < prev[2]
                            && prev[0] < aabb[2]
                            && aabb[1] < prev[3]
                            && prev[1] < aabb[3]
                        {
                            ok = false;
                            break;
                        }
                    }
                    if !ok {
                        break;
                    }
                    aabbs[i] = aabb;
                } else {
                    ok = false;
                    break;
                }
            }
            ok
        }
        _ => false,
    };
    if !single_fill {
        return None;
    }
    match BlendType::from(blend_mode.clone()) {
        BlendType::Trivial(mode) => Some(ShapeBlend::Trivial(mode)),
        BlendType::Complex(ComplexBlend::Multiply)
            if opaque
                && ruffle_render::perf_flags::enabled(
                    ruffle_render::perf_flags::PerfFlag::OpaqueMultiply,
                ) =>
        {
            Some(ShapeBlend::Trivial(TrivialBlend::MultiplyOpaque))
        }
        // Solid, gradient, and bitmap fills can read the backdrop themselves (see
        // `fetch.metal`). Alpha and Erase blend against their enclosing Layer, so they
        // can use framebuffer fetch directly when the current target IS that Layer.
        BlendType::Complex(mode)
            if (in_current_layer || !matches!(mode, ComplexBlend::Alpha | ComplexBlend::Erase))
                && descriptors.shaders.fetch_color_shaders.is_some()
                && ruffle_render::perf_flags::enabled(
                    ruffle_render::perf_flags::PerfFlag::FetchBlends,
                ) =>
        {
            Some(ShapeBlend::Fetch(mode))
        }
        _ => None,
    }
}

impl WgpuCommandHandler<'_, '_> {
    /// See `direct_blend_mode_for`.
    fn direct_blend_mode(
        &self,
        commands: &CommandList,
        blend_mode: &RenderBlendMode,
    ) -> Option<ShapeBlend> {
        if self.direct_blend.is_some() {
            return None;
        }
        direct_blend_mode_for(
            self.descriptors,
            commands,
            blend_mode,
            self.opaque,
            matches!(self.nearest_layer, LayerRef::Current),
        )
    }

    fn blend_impl(
        &mut self,
        commands: CommandList,
        blend_mode: RenderBlendMode,
        rect: Option<(i32, i32, u32, u32)>,
    ) {
        // An Alpha or Erase blend with no enclosing Layer is ignored in Flash (and would
        // otherwise render into a full-screen offscreen target and split the main render
        // pass only to be discarded in `Surface::draw_commands`).
        if matches!(
            blend_mode,
            RenderBlendMode::Builtin(BlendMode::Alpha | BlendMode::Erase)
        ) && matches!(self.nearest_layer, LayerRef::None)
        {
            return;
        }
        if let Some(mode) = self.direct_blend_mode(&commands, &blend_mode) {
            crate::perf_counters::count_named("direct-blend");
            let unwrapped = unwrap_passthrough_blend_owned(commands);
            self.direct_blend = Some(mode);
            unwrapped.execute(self);
            self.direct_blend = None;
            return;
        }
        if crate::perf_counters::enabled() {
            let mode = match &blend_mode {
                RenderBlendMode::Builtin(mode) => format!("{mode:?}"),
                RenderBlendMode::Shader(_) => "Shader".to_string(),
            };
            let size = match rect {
                Some((_, _, w, h)) if w * h <= 64 * 64 => "small",
                Some((_, _, w, h)) if w * h <= 256 * 256 => "medium",
                Some(_) => "large",
                None => "full",
            };
            let simple = if commands.commands.len() == 1 {
                "/1cmd"
            } else {
                ""
            };
            crate::perf_counters::count_named(format!("{mode}/{size}{simple}"));
        }
        let (origin, width, height) = match rect {
            Some((x, y, width, height)) => ((x, y), width, height),
            None => match self.scissor {
                Some((_, _, sw, sh)) => (self.scissor_origin, sw, sh),
                None => (self.origin, self.width, self.height),
            },
        };
        let is_layer = matches!(&blend_mode, RenderBlendMode::Builtin(BlendMode::Layer));
        let target_layer = if is_layer {
            LayerRef::Current
        } else {
            self.nearest_layer
        };
        let in_current_layer = matches!(self.nearest_layer, LayerRef::Current);
        let blend_type = BlendType::from(blend_mode);
        let clear_color = blend_type.default_color();

        // We currently do not support shader blends in masks. In order not to
        // break other parts of the scene, we just fall back to a normal blend.
        //
        // TODO Add support for shader blends in masks.
        let is_shader_blend_in_mask =
            self.num_masks > 0 && matches!(blend_type, BlendType::Shader(_));
        let blend_type = if is_shader_blend_in_mask {
            BlendType::Trivial(TrivialBlend::Normal)
        } else {
            blend_type
        };
        let blend_type = match blend_type {
            BlendType::Complex(ComplexBlend::Multiply)
                if self.opaque
                    && ruffle_render::perf_flags::enabled(
                        ruffle_render::perf_flags::PerfFlag::OpaqueMultiply,
                    ) =>
            {
                BlendType::Trivial(TrivialBlend::MultiplyOpaque)
            }
            BlendType::Complex(_)
                if ruffle_render::perf_flags::enabled(
                    ruffle_render::perf_flags::PerfFlag::ComplexAsNormal,
                ) =>
            {
                BlendType::Trivial(TrivialBlend::Normal)
            }
            // A shader may produce any alpha.
            BlendType::Shader(shader) => {
                self.opaque = false;
                BlendType::Shader(shader)
            }
            blend_type => blend_type,
        };

        let inline_blend = match blend_type {
            BlendType::Complex(blend_mode)
                if (in_current_layer
                    || !matches!(blend_mode, ComplexBlend::Alpha | ComplexBlend::Erase))
                    && self.descriptors.shaders.fetch_blend_shaders.is_some()
                    && ruffle_render::perf_flags::enabled(
                        ruffle_render::perf_flags::PerfFlag::FetchBlends,
                    ) =>
            {
                Some(ShapeBlend::Fetch(blend_mode))
            }
            BlendType::Trivial(blend_mode) => Some(ShapeBlend::Trivial(blend_mode)),
            _ => None,
        };

        // Helper to emit an already-rendered (or cached) texture inline with `shape_blend`.
        let emit_inline_texture = |this: &mut Self,
                                   texture: PoolOrArcTexture,
                                   bind_group: wgpu::BindGroup,
                                   uv_rect: Option<[f32; 4]>,
                                   shape_blend: ShapeBlend| {
            let matrix = Matrix::translate(
                Twips::from_pixels_i32(origin.0),
                Twips::from_pixels_i32(origin.1),
            ) * Matrix::scale(width as f32, height as f32);
            let quad = uv_rect.map(|[u0, v0, u1, v1]| {
                [
                    PosUvVertex::new(0.0, 0.0, u0, v0, 1.0),
                    PosUvVertex::new(1.0, 0.0, u1, v0, 1.0),
                    PosUvVertex::new(1.0, 1.0, u1, v1, 1.0),
                    PosUvVertex::new(0.0, 1.0, u0, v1, 1.0),
                ]
            });
            let vertices_slice = quad.as_ref().map(|q| &q[..]);
            match shape_blend {
                ShapeBlend::Fetch(blend_mode) => {
                    crate::perf_counters::count_named("fetch-blend");
                    this.add_to_current_with_vertices(
                        matrix,
                        0.0,
                        Default::default(),
                        vertices_slice,
                        |instance_index, vertex_offset| DrawCommand::FetchBlend {
                            _texture: texture,
                            binds: bind_group,
                            instance_index,
                            vertex_offset,
                            blend_mode,
                        },
                    );
                }
                ShapeBlend::Trivial(blend_mode) => {
                    crate::perf_counters::count(crate::perf_counters::Counter::BlendTrivial);
                    this.add_to_current_with_vertices(
                        matrix,
                        0.0,
                        Default::default(),
                        vertices_slice,
                        |instance_index, vertex_offset| DrawCommand::RenderTexture {
                            _texture: texture,
                            binds: bind_group,
                            instance_index,
                            vertex_offset,
                            blend_mode,
                        },
                    );
                }
            }
        };

        // Content-addressed blend-layer LRU cache: skip 4× MSAA offscreen rasterization
        // completely for static/24fps-animated/looping blend layers.
        let blend_key = if let Some(shape_blend) = inline_blend
            && matches!(self.nearest_layer, LayerRef::None)
            && self.blend_cache.is_some()
        {
            let clear_swf = Color {
                r: (clear_color.r * 255.0).round() as u8,
                g: (clear_color.g * 255.0).round() as u8,
                b: (clear_color.b * 255.0).round() as u8,
                a: (clear_color.a * 255.0).round() as u8,
            };
            let key = crate::filters::CabContentKey::try_from_blend_layer(
                (width, height),
                origin,
                clear_swf,
                self.quality,
                is_layer,
                &commands,
                self.filter_atlas,
            );
            if let Some(ref k) = key
                && let Some(cached) = self.blend_cache.as_deref_mut().and_then(|c| c.get(k))
            {
                let tex = PoolOrArcTexture::Manual((cached.texture.clone(), cached.view.clone()));
                let bg = cached.bind_group.clone();
                let uv_rect = cached.uv_rect;
                crate::perf_counters::count_named("blend-lru/hit");
                emit_inline_texture(self, tex, bg, uv_rect, shape_blend);
                return;
            }
            key
        } else {
            None
        };

        let promote_key = if let Some(key) = blend_key
            && !self
                .pending_blend_promotions
                .iter()
                .any(|(pending_key, _)| *pending_key == key)
            && self
                .blend_cache
                .as_deref_mut()
                .is_some_and(|c| c.should_promote(&key, (width, height)))
        {
            Some(key)
        } else {
            None
        };

        let use_leaf_atlas = (rect.is_some() || self.scissor.is_some())
            && ruffle_render::perf_flags::enabled(ruffle_render::perf_flags::PerfFlag::LayerAtlas)
            && layer_atlas::is_leaf(self.descriptors, &commands, width, height, is_layer);

        if use_leaf_atlas && let Some(shape_blend) = inline_blend {
            if let Some(key) = promote_key
                && let Some(cache) = self.blend_cache.as_deref_mut()
            {
                let alloc = cache.allocate_entry(self.descriptors, (width, height));
                let tex = alloc.texture.clone();
                let view = alloc.view.clone();
                let bind_group = alloc.bind_group.clone();
                let uv_rect = alloc.uv_rect;
                let (dst_x, dst_y) = alloc.dst_origin;
                self.leaf_layers.push(layer_atlas::LeafLayer {
                    commands,
                    origin,
                    width,
                    height,
                    destination: Some((tex.clone(), dst_x, dst_y)),
                    is_layer,
                });
                self.pending_blend_promotions.push((key, alloc));
                crate::perf_counters::count_named("blend-lru/promote");
                emit_inline_texture(
                    self,
                    PoolOrArcTexture::Manual((tex, view)),
                    bind_group,
                    uv_rect,
                    shape_blend,
                );
                return;
            }
            // Sample directly from the layer atlas via UV sub-rect, avoiding a per-layer
            // texture allocation, copy-out blit, and per-layer bind group.
            match shape_blend {
                ShapeBlend::Fetch(_) => crate::perf_counters::count_named("fetch-blend"),
                ShapeBlend::Trivial(_) => {
                    crate::perf_counters::count(crate::perf_counters::Counter::BlendTrivial)
                }
            }
            let leaf_index = self.leaf_layers.len();
            self.leaf_layers.push(layer_atlas::LeafLayer {
                commands,
                origin,
                width,
                height,
                destination: None,
                is_layer,
            });
            let placeholder = [PosUvVertex::new(0.0, 0.0, 0.0, 0.0, 1.0); 4];
            self.add_to_current_with_vertices(
                Matrix::translate(
                    Twips::from_pixels_i32(origin.0),
                    Twips::from_pixels_i32(origin.1),
                ) * Matrix::scale(width as f32, height as f32),
                0.0,
                Default::default(),
                Some(&placeholder),
                |instance_index, vertex_offset| DrawCommand::RenderAtlasLayer {
                    leaf_index,
                    atlas_index: 0,
                    instance_index,
                    vertex_offset: vertex_offset.expect("vertices were provided"),
                    blend_mode: shape_blend,
                },
            );
            return;
        }

        let texture = if use_leaf_atlas {
            // Rendered later, together with other such layers (see `layer_atlas`).
            let texture = self.texture_pool.get_texture(
                self.descriptors,
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                layer_atlas::LAYER_TEXTURE_USAGE,
                wgpu::TextureFormat::Rgba8Unorm,
                1,
            );
            self.leaf_layers.push(layer_atlas::LeafLayer {
                commands,
                origin,
                width,
                height,
                destination: Some((texture.0.clone(), 0, 0)),
                is_layer,
            });
            PoolOrArcTexture::Pool(texture)
        } else {
            let surface = Surface::new(
                self.descriptors,
                self.quality,
                width,
                height,
                wgpu::TextureFormat::Rgba8Unorm,
            )
            .with_origin(origin);
            let target = surface.draw_commands(
                RenderTargetMode::FreshWithColor(clear_color),
                self.descriptors,
                self.meshes,
                commands,
                self.staging_belt,
                self.dynamic_transforms,
                self.draw_encoder,
                target_layer,
                self.texture_pool,
                self.filter_atlas,
                self.blend_cache.as_deref_mut(),
            );
            target.ensure_cleared(self.draw_encoder);
            let rendered = target.take_color_texture();
            if let Some(key) = promote_key
                && let Some(shape_blend) = inline_blend
                && let Some(cache) = self.blend_cache.as_deref_mut()
            {
                let alloc = cache.allocate_entry(self.descriptors, (width, height));
                let tex = alloc.texture.clone();
                let view = alloc.view.clone();
                let bind_group = alloc.bind_group.clone();
                let uv_rect = alloc.uv_rect;
                let (dst_x, dst_y) = alloc.dst_origin;
                self.draw_encoder.copy_texture_to_texture(
                    rendered.texture().as_image_copy(),
                    wgpu::TexelCopyTextureInfo {
                        texture: &tex,
                        mip_level: 0,
                        origin: wgpu::Origin3d {
                            x: dst_x,
                            y: dst_y,
                            z: 0,
                        },
                        aspect: Default::default(),
                    },
                    wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                );
                cache.insert_entry(key, alloc);
                crate::perf_counters::count_named("blend-lru/promote");
                emit_inline_texture(
                    self,
                    PoolOrArcTexture::Manual((tex, view)),
                    bind_group,
                    uv_rect,
                    shape_blend,
                );
                return;
            }
            rendered
        };

        match inline_blend {
            Some(ShapeBlend::Fetch(blend_mode)) => {
                // Drawn like a trivial blend, but its shader blends with the pixel
                // underneath (framebuffer fetch), so there's no need to split the pass
                // and copy the backdrop. (Alpha and Erase blend with their layer,
                // which may be another target.)
                crate::perf_counters::count_named("fetch-blend");
                let bind_group = self.texture_bind_group(&texture);
                self.add_to_current(
                    Matrix::translate(
                        Twips::from_pixels_i32(origin.0),
                        Twips::from_pixels_i32(origin.1),
                    ) * Matrix::scale(width as f32, height as f32),
                    0.0,
                    Default::default(),
                    |instance_index| DrawCommand::FetchBlend {
                        _texture: texture,
                        binds: bind_group,
                        instance_index,
                        vertex_offset: None,
                        blend_mode,
                    },
                );
            }
            Some(ShapeBlend::Trivial(blend_mode)) => {
                crate::perf_counters::count(crate::perf_counters::Counter::BlendTrivial);
                let transform = Transform {
                    matrix: Matrix::translate(
                        Twips::from_pixels_i32(origin.0),
                        Twips::from_pixels_i32(origin.1),
                    ) * Matrix::scale(width as f32, height as f32),
                    tz: 0.0,
                    color_transform: Default::default(),
                    perspective_projection: None,
                };
                let bind_group = self.texture_bind_group(&texture);
                self.add_to_current(
                    transform.matrix,
                    transform.tz,
                    transform.color_transform,
                    |instance_index| DrawCommand::RenderTexture {
                        _texture: texture,
                        binds: bind_group,
                        instance_index,
                        vertex_offset: None,
                        blend_mode,
                    },
                );
            }
            None => {
                crate::perf_counters::count(crate::perf_counters::Counter::BlendComplex);
                if !self.current.is_empty() {
                    self.result.push(Chunk::Draw {
                        chunk: mem::take(&mut self.current),
                        needs_stencil: self.needs_stencil,
                        transforms: mem::replace(
                            &mut self.transforms,
                            Self::new_transforms(self.descriptors, self.dynamic_transforms),
                        ),
                        vertices: mem::replace(
                            &mut self.vertices,
                            Self::new_vertices(self.descriptors, self.dynamic_transforms),
                        ),
                    });
                }
                let chunk_blend_mode = match blend_type {
                    BlendType::Complex(complex) => ChunkBlendMode::Complex(complex),
                    BlendType::Shader(shader) => ChunkBlendMode::Shader(shader),
                    _ => unreachable!(),
                };
                let region = rect.map(|(x, y, width, height)| BlendRegion {
                    x: (x - self.origin.0) as u32,
                    y: (y - self.origin.1) as u32,
                    width,
                    height,
                });
                self.result.push(Chunk::Blend {
                    texture,
                    blend_mode: chunk_blend_mode,
                    needs_stencil: self.num_masks > 0,
                    region,
                });
                self.needs_stencil = self.num_masks > 0;
            }
        }
    }
}

impl CommandHandler for WgpuCommandHandler<'_, '_> {
    fn blend(&mut self, commands: CommandList, blend_mode: RenderBlendMode) {
        self.blend_impl(commands, blend_mode, None);
    }

    fn blend_bounded(
        &mut self,
        commands: CommandList,
        blend_mode: RenderBlendMode,
        bounds: Rectangle<Twips>,
    ) {
        match self.bounded_blend_rect(&blend_mode, bounds) {
            // Entirely outside of the target, so it can't affect anything.
            Some(None) => {}
            Some(Some(rect)) => self.blend_impl(commands, blend_mode, Some(rect)),
            None => self.blend_impl(commands, blend_mode, None),
        }
    }

    fn render_bitmap(
        &mut self,
        bitmap: BitmapHandle,
        transform: Transform,
        smoothing: bool,
        pixel_snapping: PixelSnapping,
        region: PixelRegion,
    ) {
        let direct_blend = self
            .direct_blend
            .unwrap_or(ShapeBlend::Trivial(TrivialBlend::Normal));
        let mut matrix = transform.matrix;
        pixel_snapping.apply(&mut matrix);
        matrix *= Matrix::scale(region.width() as f32, region.height() as f32);

        if let Some(filter_atlas) = self.filter_atlas
            && let Some(&(atlas_index, slot_x, slot_y, atlas_w, atlas_h)) = filter_atlas
                .slots
                .get(&(std::sync::Arc::as_ptr(&bitmap.0) as *const () as usize))
        {
            let (u0, u1, v0, v1) = (
                (slot_x + region.x_min) as f32 / atlas_w as f32,
                (slot_x + region.x_max) as f32 / atlas_w as f32,
                (slot_y + region.y_min) as f32 / atlas_h as f32,
                (slot_y + region.y_max) as f32 / atlas_h as f32,
            );
            let vertices = [
                PosUvVertex::new(0.0, 0.0, u0, v0, 1.0),
                PosUvVertex::new(1.0, 0.0, u1, v0, 1.0),
                PosUvVertex::new(1.0, 1.0, u1, v1, 1.0),
                PosUvVertex::new(0.0, 1.0, u0, v1, 1.0),
            ];
            self.add_to_current_with_vertices(
                matrix,
                transform.tz,
                transform.color_transform,
                Some(&vertices),
                |instance_index, vertex_offset| DrawCommand::RenderFilterAtlas {
                    atlas_index,
                    instance_index,
                    vertex_offset: vertex_offset.expect("vertices were provided"),
                    blend_mode: direct_blend,
                },
            );
            return;
        }

        let texture = as_texture(&bitmap);

        let vertices: &[PosUvVertex] = {
            let (u0, u1, v0, v1) = (
                region.x_min as f32 / texture.texture.width() as f32,
                region.x_max as f32 / texture.texture.width() as f32,
                region.y_min as f32 / texture.texture.height() as f32,
                region.y_max as f32 / texture.texture.height() as f32,
            );
            &[
                PosUvVertex::new(0.0, 0.0, u0, v0, 1.0),
                PosUvVertex::new(1.0, 0.0, u1, v0, 1.0),
                PosUvVertex::new(1.0, 1.0, u1, v1, 1.0),
                PosUvVertex::new(0.0, 1.0, u0, v1, 1.0),
            ]
        };

        self.add_to_current_with_vertices(
            matrix,
            transform.tz,
            transform.color_transform,
            Some(vertices),
            |instance_index, vertex_offset| DrawCommand::RenderBitmap {
                bitmap,
                instance_index,
                vertex_offset,
                smoothing,
                blend_mode: direct_blend,
                render_stage3d: false,
            },
        );
    }

    fn render_stage3d(&mut self, bitmap: BitmapHandle, transform: Transform) {
        let mut matrix = transform.matrix;
        {
            let texture = as_texture(&bitmap);
            matrix *= Matrix::scale(
                texture.texture.width() as f32,
                texture.texture.height() as f32,
            );
        }
        self.add_to_current(
            matrix,
            transform.tz,
            transform.color_transform,
            |instance_index| DrawCommand::RenderBitmap {
                bitmap,
                instance_index,
                vertex_offset: None,
                smoothing: false,
                blend_mode: ShapeBlend::Trivial(TrivialBlend::Normal),
                render_stage3d: true,
            },
        );
    }

    fn render_shape(&mut self, shape: ShapeHandle, transform: Transform) {
        let blend_mode = self
            .direct_blend
            .unwrap_or(ShapeBlend::Trivial(TrivialBlend::Normal));
        self.add_to_current(
            transform.matrix,
            transform.tz,
            transform.color_transform,
            |instance_index| DrawCommand::RenderShape {
                shape,
                instance_index,
                blend_mode,
            },
        );
    }

    fn draw_rect(&mut self, color: Color, matrix: Matrix) {
        self.add_to_current(
            matrix,
            0.0,
            ColorTransform::multiply_from(color),
            |instance_index| DrawCommand::DrawRect { instance_index },
        );
    }

    fn draw_line(&mut self, color: Color, mut matrix: Matrix) {
        if self.emulate_lines {
            let mut cl = CommandList::new();
            emulate_line(&mut cl, color, matrix);
            cl.execute(self);
        } else {
            matrix.tx += Twips::HALF_PX;
            matrix.ty += Twips::HALF_PX;
            self.add_to_current(
                matrix,
                0.0,
                ColorTransform::multiply_from(color),
                |instance_index| DrawCommand::DrawLine { instance_index },
            );
        }
    }

    fn draw_line_rect(&mut self, color: Color, mut matrix: Matrix) {
        if self.emulate_lines {
            let mut cl = CommandList::new();
            emulate_line_rect(&mut cl, color, matrix);
            cl.execute(self);
        } else {
            matrix.tx += Twips::HALF_PX;
            matrix.ty += Twips::HALF_PX;
            self.add_to_current(
                matrix,
                0.0,
                ColorTransform::multiply_from(color),
                |instance_index| DrawCommand::DrawLineRect { instance_index },
            );
        }
    }

    fn push_mask(&mut self) {
        self.needs_stencil = true;
        self.num_masks += 1;
        self.current.push(DrawCommand::PushMask);
    }

    fn activate_mask(&mut self) {
        self.needs_stencil = true;
        self.current.push(DrawCommand::ActivateMask);
    }

    fn deactivate_mask(&mut self) {
        self.needs_stencil = true;
        self.current.push(DrawCommand::DeactivateMask);
    }

    fn pop_mask(&mut self) {
        self.needs_stencil = true;
        self.num_masks -= 1;
        self.current.push(DrawCommand::PopMask);
    }

    fn render_alpha_mask(&mut self, maskee_commands: CommandList, mask_commands: CommandList) {
        crate::perf_counters::count(crate::perf_counters::Counter::AlphaMask);
        let surface = Surface::new(
            self.descriptors,
            self.quality,
            self.width,
            self.height,
            wgpu::TextureFormat::Rgba8Unorm,
        )
        .with_origin(self.origin);

        let maskee = surface.draw_commands(
            RenderTargetMode::FreshWithColor(wgpu::Color::TRANSPARENT),
            self.descriptors,
            self.meshes,
            maskee_commands,
            self.staging_belt,
            self.dynamic_transforms,
            self.draw_encoder,
            LayerRef::None,
            self.texture_pool,
            self.filter_atlas,
            None,
        );
        maskee.ensure_cleared(self.draw_encoder);
        let matrix = Matrix::translate(
            Twips::from_pixels_i32(self.origin.0),
            Twips::from_pixels_i32(self.origin.1),
        ) * Matrix::scale(maskee.width() as f32, maskee.height() as f32);
        let maskee = maskee.take_color_texture();

        let mask = surface.draw_commands(
            RenderTargetMode::FreshWithColor(wgpu::Color::TRANSPARENT),
            self.descriptors,
            self.meshes,
            mask_commands,
            self.staging_belt,
            self.dynamic_transforms,
            self.draw_encoder,
            LayerRef::None,
            self.texture_pool,
            self.filter_atlas,
            None,
        );
        mask.ensure_cleared(self.draw_encoder);
        let mask = mask.take_color_texture();

        let binds = self
            .descriptors
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.descriptors.bind_layouts.alpha_mask,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(maskee.view()),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(mask.view()),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(
                            self.descriptors.bitmap_samplers.get_sampler(false, false),
                        ),
                    },
                ],
                label: None,
            });

        self.add_to_current(matrix, 0.0, Default::default(), |instance_index| {
            DrawCommand::RenderAlphaMask {
                maskee,
                mask,
                binds,
                instance_index,
            }
        });
    }
}
