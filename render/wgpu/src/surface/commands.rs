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
}

impl<'encoder> CommandRenderer<'encoder> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        pipelines: &'encoder Pipelines,
        descriptors: &'encoder Descriptors,
        dynamic_vertex_buffer: &'encoder wgpu::Buffer,
        dynamic_vertex_base: wgpu::BufferAddress,
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
                    render_pass.set_stencil_reference(self.num_masks - 1);
                }
                MaskState::DrawMaskedContent => {
                    render_pass.set_stencil_reference(self.num_masks);
                }
                MaskState::ClearMaskStencil => {
                    render_pass.set_stencil_reference(self.num_masks);
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
                blend_mode,
            } => self.render_texture(render_pass, *instance_index, binds, *blend_mode),
            DrawCommand::FetchBlend {
                _texture,
                binds,
                instance_index,
                blend_mode,
            } => self.render_fetch_blend(render_pass, *instance_index, binds, *blend_mode),
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

    pub fn prep_color(&self, render_pass: &mut wgpu::RenderPass<'encoder>) {
        if self.needs_stencil {
            render_pass.set_pipeline(self.pipelines.color.pipeline_for(self.mask_state));
        } else {
            render_pass.set_pipeline(self.pipelines.color.stencilless_pipeline());
        }
    }

    pub fn prep_lines(&self, render_pass: &mut wgpu::RenderPass<'encoder>) {
        if self.needs_stencil {
            render_pass.set_pipeline(self.pipelines.lines.pipeline_for(self.mask_state));
        } else {
            render_pass.set_pipeline(self.pipelines.lines.stencilless_pipeline());
        }
    }

    pub fn prep_gradient(
        &self,
        render_pass: &mut wgpu::RenderPass<'encoder>,
        bind_group: &'encoder wgpu::BindGroup,
    ) {
        if self.needs_stencil {
            render_pass.set_pipeline(self.pipelines.gradients.pipeline_for(self.mask_state));
        } else {
            render_pass.set_pipeline(self.pipelines.gradients.stencilless_pipeline());
        }

        render_pass.set_bind_group(2, bind_group, &[]);
    }

    pub fn prep_bitmap(
        &self,
        render_pass: &mut wgpu::RenderPass<'encoder>,
        bind_group: &'encoder wgpu::BindGroup,
        blend_mode: TrivialBlend,
        render_stage3d: bool,
    ) {
        match (self.needs_stencil, render_stage3d) {
            (true, true) => {
                render_pass.set_pipeline(&self.pipelines.bitmap_opaque_dummy_stencil);
            }
            (true, false) => {
                render_pass
                    .set_pipeline(self.pipelines.bitmap[blend_mode].pipeline_for(self.mask_state));
            }
            (false, true) => {
                render_pass.set_pipeline(&self.pipelines.bitmap_opaque);
            }
            (false, false) => {
                render_pass.set_pipeline(self.pipelines.bitmap[blend_mode].stencilless_pipeline());
            }
        }

        render_pass.set_bind_group(2, bind_group, &[]);
    }

    pub fn prep_alpha_mask(
        &self,
        render_pass: &mut wgpu::RenderPass<'encoder>,
        bind_group: &'encoder wgpu::BindGroup,
    ) {
        if self.needs_stencil {
            render_pass.set_pipeline(self.pipelines.alpha_mask.pipeline_for(self.mask_state));
        } else {
            render_pass.set_pipeline(self.pipelines.alpha_mask.stencilless_pipeline());
        }

        render_pass.set_bind_group(2, bind_group, &[]);
    }

    pub fn draw(
        &self,
        render_pass: &mut wgpu::RenderPass<'encoder>,
        vertices: wgpu::BufferSlice<'encoder>,
        indices: wgpu::BufferSlice<'encoder>,
        num_indices: u32,
        instance_index: u32,
    ) {
        render_pass.set_vertex_buffer(0, vertices);
        render_pass.set_index_buffer(indices, wgpu::IndexFormat::Uint32);

        render_pass.draw_indexed(0..num_indices, 0, instance_index..(instance_index + 1));
    }

    #[allow(clippy::too_many_arguments)]
    pub fn render_bitmap(
        &self,
        render_pass: &mut wgpu::RenderPass<'encoder>,
        bitmap: &'encoder BitmapHandle,
        instance_index: u32,
        smoothing: bool,
        blend_mode: TrivialBlend,
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
        self.prep_bitmap(render_pass, &bind.bind_group, blend_mode, render_stage3d);

        let vertex_slice = if let Some(vertex_offset) = vertex_offset {
            self.dynamic_vertex_buffer
                .slice(self.dynamic_vertex_base + vertex_offset..)
        } else {
            self.descriptors.quad.vertices_pos_uv.slice(..)
        };

        self.draw(
            render_pass,
            vertex_slice,
            self.descriptors.quad.indices.slice(..),
            6,
            instance_index,
        );
    }

    pub fn render_texture(
        &self,
        render_pass: &mut wgpu::RenderPass<'encoder>,
        instance_index: u32,
        bind_group: &'encoder wgpu::BindGroup,
        blend_mode: TrivialBlend,
    ) {
        self.prep_bitmap(render_pass, bind_group, blend_mode, false);

        self.draw(
            render_pass,
            self.descriptors.quad.vertices_pos_uv.slice(..),
            self.descriptors.quad.indices.slice(..),
            6,
            instance_index,
        );
    }

    pub fn render_fetch_blend(
        &self,
        render_pass: &mut wgpu::RenderPass<'encoder>,
        instance_index: u32,
        bind_group: &'encoder wgpu::BindGroup,
        blend_mode: ComplexBlend,
    ) {
        let pipelines = &self
            .pipelines
            .fetch_blends
            .as_ref()
            .expect("fetch blends are only used where supported")[blend_mode];
        if self.needs_stencil {
            render_pass.set_pipeline(pipelines.pipeline_for(self.mask_state));
        } else {
            render_pass.set_pipeline(pipelines.stencilless_pipeline());
        }
        render_pass.set_bind_group(2, bind_group, &[]);
        self.draw(
            render_pass,
            self.descriptors.quad.vertices_pos_uv.slice(..),
            self.descriptors.quad.indices.slice(..),
            6,
            instance_index,
        );
    }

    pub fn render_shape(
        &self,
        render_pass: &mut wgpu::RenderPass<'encoder>,
        shape: &'encoder ShapeHandle,
        instance_index: u32,
        blend_mode: TrivialBlend,
    ) {
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
                (DrawType::Color, None) => {
                    self.prep_color(render_pass);
                }
                (DrawType::Color, Some(blended)) => {
                    render_pass.set_pipeline(if self.needs_stencil {
                        blended.color.pipeline_for(self.mask_state)
                    } else {
                        blended.color.stencilless_pipeline()
                    });
                }
                (DrawType::Gradient { bind_group, .. }, None) => {
                    self.prep_gradient(render_pass, bind_group);
                }
                (DrawType::Gradient { bind_group, .. }, Some(blended)) => {
                    render_pass.set_pipeline(if self.needs_stencil {
                        blended.gradient.pipeline_for(self.mask_state)
                    } else {
                        blended.gradient.stencilless_pipeline()
                    });
                    render_pass.set_bind_group(2, bind_group, &[]);
                }
                (DrawType::Bitmap { binds, .. }, _) => {
                    self.prep_bitmap(render_pass, &binds.bind_group, blend_mode, false);
                }
            }

            self.draw(
                render_pass,
                mesh.vertex_buffer.slice(draw.vertices.clone()),
                mesh.index_buffer.slice(draw.indices.clone()),
                num_indices,
                instance_index,
            );
        }
    }

    pub fn render_alpha_mask(
        &self,
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

    pub fn draw_rect(&self, render_pass: &mut wgpu::RenderPass<'encoder>, instance_index: u32) {
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
        &self,
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
        render_pass.set_stencil_reference(self.num_masks - 1);
    }

    pub fn activate_mask(&mut self, render_pass: &mut wgpu::RenderPass<'encoder>) {
        debug_assert!(self.num_masks > 0 && self.mask_state == MaskState::DrawMaskStencil);
        self.mask_state = MaskState::DrawMaskedContent;
        render_pass.set_stencil_reference(self.num_masks);
    }

    pub fn deactivate_mask(&mut self, render_pass: &mut wgpu::RenderPass<'encoder>) {
        debug_assert!(self.num_masks > 0 && self.mask_state == MaskState::DrawMaskedContent);
        self.mask_state = MaskState::ClearMaskStencil;
        render_pass.set_stencil_reference(self.num_masks);
    }

    pub fn pop_mask(&mut self, render_pass: &mut wgpu::RenderPass<'encoder>) {
        debug_assert!(self.num_masks > 0 && self.mask_state == MaskState::ClearMaskStencil);
        self.num_masks -= 1;
        render_pass.set_stencil_reference(self.num_masks);
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
        blend_mode: TrivialBlend,
        render_stage3d: bool,
    },
    RenderTexture {
        _texture: PoolOrArcTexture,
        binds: wgpu::BindGroup,
        instance_index: u32,
        blend_mode: TrivialBlend,
    },
    /// A complex blend that reads the backdrop from the color attachment.
    FetchBlend {
        _texture: PoolOrArcTexture,
        binds: wgpu::BindGroup,
        instance_index: u32,
        blend_mode: ComplexBlend,
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
        blend_mode: TrivialBlend,
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
        origin,
        nearest_layer,
        texture_pool,
    );
    handler.opaque = opaque;
    let chunks = handler.chunk_blends(commands);
    (chunks, mem::take(&mut handler.leaf_layers))
}

/// Chunks the draws of several leaf layers into one target (an atlas).
/// Each layer is given as its commands, a pixel translation to apply to them,
/// and the scissor rectangle `(x, y, width, height)` to clip them to.
#[expect(clippy::too_many_arguments)]
pub fn chunk_leaf_layers<'encoder, 'global: 'encoder>(
    layers: impl Iterator<Item = (CommandList, (i32, i32), (u32, u32, u32, u32))>,
    descriptors: &'encoder Descriptors,
    staging_belt: &'encoder mut wgpu::util::StagingBelt,
    dynamic_transforms: &'encoder DynamicTransforms,
    draw_encoder: &'encoder mut Scope<'global, wgpu::CommandEncoder>,
    meshes: &'encoder Vec<Mesh>,
    quality: StageQuality,
    width: u32,
    height: u32,
    texture_pool: &'encoder mut TexturePool,
) -> Vec<Chunk> {
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
    );
    for (commands, (dx, dy), scissor) in layers {
        handler.scissor = Some(scissor);
        handler.current.push(DrawCommand::SetScissor(scissor));
        handler.translation = (Twips::from_pixels_i32(dx), Twips::from_pixels_i32(dy));
        commands.execute(&mut handler);
    }
    let chunks = handler.chunk_blends(CommandList::new());
    debug_assert!(handler.leaf_layers.is_empty());
    chunks
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
    /// Whether the target is known to be fully opaque (and stays so), allowing
    /// some complex blends to be done as trivial ones.
    opaque: bool,
    transforms: BufferBuilder,
    vertices: BufferBuilder,
    needs_stencil: bool,
    num_masks: i32,
    /// While drawing the content of a blend directly (instead of in a layer), its mode.
    direct_blend: Option<TrivialBlend>,
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
            transforms,
            vertices,
            needs_stencil: false,
            num_masks: 0,
            direct_blend: None,
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
        let (ox, oy) = self.origin;
        let clamp_x = |v: f64| (v as i32).clamp(ox, ox + self.width as i32);
        let clamp_y = |v: f64| (v as i32).clamp(oy, oy + self.height as i32);
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
        let (x0, x1) = round(x0, x1, ox, ox + self.width as i32);
        let (y0, y1) = round(y0, y1, oy, oy + self.height as i32);
        let (width, height) = ((x1 - x0) as u32, (y1 - y0) as u32);

        // Not worth it if we'd cover most of the target anyway.
        if (width as u64 * height as u64) * 4 > (self.width as u64 * self.height as u64) * 3 {
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

    /// The trivial blend mode to draw `commands` with directly, without a layer, if that
    /// gives the same result: when they're a single bitmap, or a single shape with a
    /// single fill, which affect each pixel at most once.
    fn direct_blend_mode(
        &self,
        commands: &CommandList,
        blend_mode: &RenderBlendMode,
    ) -> Option<TrivialBlend> {
        if self.direct_blend.is_some()
            || !ruffle_render::perf_flags::enabled(
                ruffle_render::perf_flags::PerfFlag::DirectBlends,
            )
        {
            return None;
        }
        let mode = match BlendType::from(blend_mode.clone()) {
            BlendType::Trivial(mode) => mode,
            BlendType::Complex(ComplexBlend::Multiply)
                if self.opaque
                    && ruffle_render::perf_flags::enabled(
                        ruffle_render::perf_flags::PerfFlag::OpaqueMultiply,
                    ) =>
            {
                TrivialBlend::MultiplyOpaque
            }
            _ => return None,
        };
        let [command] = commands.commands.as_slice() else {
            return None;
        };
        let single = match command {
            Command::RenderShape { shape, .. } => {
                let mesh = as_mesh(shape);
                mesh.draws.len() == 1 && mesh.draws[0].single_fill
            }
            Command::RenderBitmap { .. } => true,
            _ => false,
        };
        single.then_some(mode)
    }

    fn blend_impl(
        &mut self,
        commands: CommandList,
        blend_mode: RenderBlendMode,
        rect: Option<(i32, i32, u32, u32)>,
    ) {
        if let Some(mode) = self.direct_blend_mode(&commands, &blend_mode) {
            crate::perf_counters::count_named("direct-blend");
            self.direct_blend = Some(mode);
            commands.execute(self);
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
            None => (self.origin, self.width, self.height),
        };
        let target_layer = if let RenderBlendMode::Builtin(BlendMode::Layer) = &blend_mode {
            LayerRef::Current
        } else {
            self.nearest_layer
        };
        let blend_type = BlendType::from(blend_mode);
        let clear_color = blend_type.default_color();

        let texture = if rect.is_some()
            && ruffle_render::perf_flags::enabled(ruffle_render::perf_flags::PerfFlag::LayerAtlas)
            && layer_atlas::is_leaf(&commands, width, height)
        {
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
                destination: texture.0.clone(),
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
            );
            target.ensure_cleared(self.draw_encoder);
            target.take_color_texture()
        };

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

        match blend_type {
            BlendType::Complex(blend_mode)
                if !matches!(blend_mode, ComplexBlend::Alpha | ComplexBlend::Erase)
                    && self.descriptors.shaders.fetch_blend_shaders.is_some()
                    && ruffle_render::perf_flags::enabled(
                        ruffle_render::perf_flags::PerfFlag::FetchBlends,
                    ) =>
            {
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
                        blend_mode,
                    },
                );
            }
            BlendType::Trivial(blend_mode) => {
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
                        blend_mode,
                    },
                );
            }
            blend_type => {
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
        let direct_blend = self.direct_blend.unwrap_or(TrivialBlend::Normal);
        let texture = as_texture(&bitmap);

        let mut matrix = transform.matrix;
        pixel_snapping.apply(&mut matrix);
        matrix *= Matrix::scale(region.width() as f32, region.height() as f32);

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
                blend_mode: TrivialBlend::Normal,
                render_stage3d: true,
            },
        );
    }

    fn render_shape(&mut self, shape: ShapeHandle, transform: Transform) {
        let blend_mode = self.direct_blend.unwrap_or(TrivialBlend::Normal);
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
