use crate::backend::RenderTargetMode;
use crate::buffer_pool::TexturePool;
use crate::descriptors::Descriptors;
use crate::filters::blur::BlurFilter;
use crate::filters::{
    FilterSource, FilterVertexWithBlur, VERTEX_BUFFERS_DESCRIPTION_FILTERS_WITH_BLUR,
};
use crate::surface::target::CommandTarget;
use crate::utils::SampleCountMap;
use bytemuck::{Pod, Zeroable};
use std::sync::OnceLock;
use swf::GlowFilter as GlowFilterArgs;
use wgpu::util::StagingBelt;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, PartialEq)]
pub(super) struct GlowUniform {
    pub(super) color: [f32; 4],
    pub(super) strength: f32,
    pub(super) inner: u32, // a wasteful bool, but we need to be aligned anyway
    pub(super) knockout: u32, // a wasteful bool, but we need to be aligned anyway
    pub(super) composite_source: u32, // undocumented flash feature, another bool
}

/// Maximum number of vertical blur taps supported by the fused V-blur + Glow shader.
pub(super) const MAX_FUSED_V_TAPS: usize = 32;

/// Uniform payload for `main_fragment_vblur_rgba` and `main_fragment_vblur_r8` (`binding(4)`).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, PartialEq)]
pub(super) struct GlowVBlurUniform {
    pub(super) glow: GlowUniform,
    pub(super) dir_x: f32,
    pub(super) dir_y: f32,
    pub(super) count: u32,
    pub(super) _padding: u32,
    pub(super) taps: [[f32; 4]; MAX_FUSED_V_TAPS / 2],
}

/// Uniform payload for `main_fragment_cm` and `main_fragment_r8_cm` (`binding(5)`).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, PartialEq)]
pub(super) struct GlowColorMatrixUniform {
    pub(super) glow: GlowUniform,
    pub(super) row_r: [f32; 4],
    pub(super) row_g: [f32; 4],
    pub(super) row_b: [f32; 4],
    pub(super) row_a: [f32; 4],
    pub(super) extra: [f32; 4],
}

impl GlowColorMatrixUniform {
    pub(super) fn from_glow_and_matrix(glow: GlowUniform, m: &[f32; 20]) -> Self {
        Self {
            glow,
            row_r: [m[0], m[1], m[2], m[3]],
            row_g: [m[5], m[6], m[7], m[8]],
            row_b: [m[10], m[11], m[12], m[13]],
            row_a: [m[15], m[16], m[17], m[18]],
            extra: [
                m[4] / 255.0,
                m[9] / 255.0,
                m[14] / 255.0,
                m[19] / 255.0,
            ],
        }
    }
}

pub struct GlowFilter {
    bind_group_layout: wgpu::BindGroupLayout,
    /// Like `bind_group_layout`, with a filtering sampler and dynamic uniform offset
    /// (single-sampled, see `linear_pipeline` / `linear_r8_pipeline`).
    pub(super) linear_bind_group_layout: wgpu::BindGroupLayout,
    linear_pipeline_layout: wgpu::PipelineLayout,
    linear_pipeline: OnceLock<wgpu::RenderPipeline>,
    linear_r8_pipeline: OnceLock<wgpu::RenderPipeline>,
    /// Bind group layout for fused V-blur + Glow (`binding(4)` = `GlowVBlurUniform`).
    pub(super) vblur_bind_group_layout: wgpu::BindGroupLayout,
    vblur_pipeline_layout: wgpu::PipelineLayout,
    vblur_rgba_pipeline: OnceLock<wgpu::RenderPipeline>,
    vblur_r8_pipeline: OnceLock<wgpu::RenderPipeline>,
    /// Bind group layout for fused Glow + ColorMatrix (`binding(5)` = `GlowColorMatrixUniform`).
    pub(super) cm_bind_group_layout: wgpu::BindGroupLayout,
    cm_pipeline_layout: wgpu::PipelineLayout,
    cm_rgba_pipeline: OnceLock<wgpu::RenderPipeline>,
    cm_r8_pipeline: OnceLock<wgpu::RenderPipeline>,
    pipeline_layout: wgpu::PipelineLayout,
    vertex_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    vertices_size: wgpu::BufferSize,
    uniform_size: wgpu::BufferSize,
    pipeline: SampleCountMap<OnceLock<wgpu::RenderPipeline>>,
}

impl GlowFilter {
    pub fn new(device: &wgpu::Device) -> Self {
        let uniform_size = std::mem::size_of::<GlowUniform>() as u64;
        // The batched path (`filters::batch`) samples scaled-down blurs bilinearly
        // and indexes per-quad uniforms with a dynamic offset.
        let make_layout = |filterable: bool, dynamic: bool, uniform_binding: u32, u_size: u64| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            sample_type: wgpu::TextureSampleType::Float { filterable },
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(if filterable {
                            wgpu::SamplerBindingType::Filtering
                        } else {
                            wgpu::SamplerBindingType::NonFiltering
                        }),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: uniform_binding,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: dynamic,
                            min_binding_size: wgpu::BufferSize::new(u_size),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            sample_type: wgpu::TextureSampleType::Float { filterable },
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
                label: create_debug_label!("Blur filter binds").as_deref(),
            })
        };
        let bind_group_layout = make_layout(false, false, 2, uniform_size);
        let linear_bind_group_layout = make_layout(true, true, 2, uniform_size);
        let vblur_bind_group_layout =
            make_layout(true, true, 4, std::mem::size_of::<GlowVBlurUniform>() as u64);
        let cm_bind_group_layout = make_layout(
            true,
            true,
            5,
            std::mem::size_of::<GlowColorMatrixUniform>() as u64,
        );

        let vertices_size = std::mem::size_of::<[FilterVertexWithBlur; 4]>() as u64;
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: vertices_size,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: uniform_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let linear_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&linear_bind_group_layout)],
                immediate_size: 0,
            });
        let vblur_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&vblur_bind_group_layout)],
                immediate_size: 0,
            });
        let cm_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&cm_bind_group_layout)],
            immediate_size: 0,
        });

        Self {
            linear_bind_group_layout,
            linear_pipeline_layout,
            linear_pipeline: Default::default(),
            linear_r8_pipeline: Default::default(),
            vblur_bind_group_layout,
            vblur_pipeline_layout,
            vblur_rgba_pipeline: Default::default(),
            vblur_r8_pipeline: Default::default(),
            cm_bind_group_layout,
            cm_pipeline_layout,
            cm_rgba_pipeline: Default::default(),
            cm_r8_pipeline: Default::default(),
            pipeline: Default::default(),
            pipeline_layout,
            vertex_buffer,
            uniform_buffer,
            bind_group_layout,
            uniform_size: wgpu::BufferSize::new(uniform_size).expect("Definitely not zero."),
            vertices_size: wgpu::BufferSize::new(vertices_size).expect("Definitely not zero."),
        }
    }

    fn pipeline(&self, descriptors: &Descriptors, msaa_sample_count: u32) -> &wgpu::RenderPipeline {
        self.pipeline.get_or_init(msaa_sample_count, || {
            self.create_pipeline(
                descriptors,
                msaa_sample_count,
                &self.pipeline_layout,
                "main_fragment",
            )
        })
    }

    /// The pipeline for `linear_bind_group_layout` (sampling `.a` from RGBA blur).
    pub(super) fn linear_pipeline(&self, descriptors: &Descriptors) -> &wgpu::RenderPipeline {
        self.linear_pipeline.get_or_init(|| {
            self.create_pipeline(
                descriptors,
                1,
                &self.linear_pipeline_layout,
                "main_fragment",
            )
        })
    }

    /// Like `linear_pipeline`, but reads blurred alpha from `.r` of an `R8Unorm` texture.
    pub(super) fn linear_r8_pipeline(&self, descriptors: &Descriptors) -> &wgpu::RenderPipeline {
        self.linear_r8_pipeline.get_or_init(|| {
            self.create_pipeline(
                descriptors,
                1,
                &self.linear_pipeline_layout,
                "main_fragment_r8",
            )
        })
    }

    /// Fused V-blur + Glow pipeline sampling `.a` from an RGBA horizontal blur atlas.
    pub(super) fn vblur_rgba_pipeline(&self, descriptors: &Descriptors) -> &wgpu::RenderPipeline {
        self.vblur_rgba_pipeline.get_or_init(|| {
            self.create_pipeline(
                descriptors,
                1,
                &self.vblur_pipeline_layout,
                "main_fragment_vblur_rgba",
            )
        })
    }

    /// Fused V-blur + Glow pipeline sampling `.r` from an `R8Unorm` horizontal blur atlas.
    pub(super) fn vblur_r8_pipeline(&self, descriptors: &Descriptors) -> &wgpu::RenderPipeline {
        self.vblur_r8_pipeline.get_or_init(|| {
            self.create_pipeline(
                descriptors,
                1,
                &self.vblur_pipeline_layout,
                "main_fragment_vblur_r8",
            )
        })
    }

    /// Fused Glow + ColorMatrix pipeline sampling `.a` from an RGBA blur atlas.
    pub(super) fn cm_rgba_pipeline(&self, descriptors: &Descriptors) -> &wgpu::RenderPipeline {
        self.cm_rgba_pipeline.get_or_init(|| {
            self.create_pipeline(
                descriptors,
                1,
                &self.cm_pipeline_layout,
                "main_fragment_cm",
            )
        })
    }

    /// Fused Glow + ColorMatrix pipeline sampling `.r` from an `R8Unorm` blur atlas.
    pub(super) fn cm_r8_pipeline(&self, descriptors: &Descriptors) -> &wgpu::RenderPipeline {
        self.cm_r8_pipeline.get_or_init(|| {
            self.create_pipeline(
                descriptors,
                1,
                &self.cm_pipeline_layout,
                "main_fragment_r8_cm",
            )
        })
    }

    fn create_pipeline(
        &self,
        descriptors: &Descriptors,
        msaa_sample_count: u32,
        layout: &wgpu::PipelineLayout,
        fragment_entry: &str,
    ) -> wgpu::RenderPipeline {
        {
            let label = create_debug_label!("Glow Filter ({} msaa, {})", msaa_sample_count, fragment_entry);
            descriptors
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: label.as_deref(),
                    layout: Some(layout),
                    vertex: wgpu::VertexState {
                        module: &descriptors.shaders.glow_filter,
                        entry_point: Some("main_vertex"),
                        buffers: &VERTEX_BUFFERS_DESCRIPTION_FILTERS_WITH_BLUR,
                        compilation_options: Default::default(),
                    },
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        strip_index_format: None,
                        front_face: wgpu::FrontFace::Ccw,
                        cull_mode: None,
                        polygon_mode: wgpu::PolygonMode::default(),
                        unclipped_depth: false,
                        conservative: false,
                    },
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState {
                        count: msaa_sample_count,
                        mask: !0,
                        alpha_to_coverage_enabled: false,
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &descriptors.shaders.glow_filter,
                        entry_point: Some(fragment_entry),
                        targets: &[Some(wgpu::TextureFormat::Rgba8Unorm.into())],
                        compilation_options: Default::default(),
                    }),
                    multiview_mask: None,
                    cache: None,
                })
        }
    }

    #[expect(clippy::too_many_arguments)]
    pub fn apply(
        &self,
        descriptors: &Descriptors,
        texture_pool: &mut TexturePool,
        draw_encoder: &mut wgpu::CommandEncoder,
        staging_belt: &mut StagingBelt,
        source: &FilterSource,
        filter: &GlowFilterArgs,
        blur_filter: &BlurFilter,
        blur_offset: (f32, f32),
    ) -> CommandTarget {
        let sample_count = source.texture.sample_count();
        let format = source.texture.format();
        let pipeline = self.pipeline(descriptors, sample_count);
        let blurred = blur_filter.apply(
            descriptors,
            texture_pool,
            draw_encoder,
            staging_belt,
            source,
            &filter.inner_blur_filter(),
        );
        let blurred_texture = if let Some(blurred) = &blurred {
            blurred.ensure_cleared(draw_encoder);
            blurred.color_texture()
        } else {
            source.texture
        };
        let source_view = source.texture.create_view(&Default::default());
        let blurred_view = blurred_texture.create_view(&Default::default());

        let target = CommandTarget::new(
            descriptors,
            texture_pool,
            wgpu::Extent3d {
                width: source.size.0,
                height: source.size.1,
                depth_or_array_layers: 1,
            },
            format,
            sample_count,
            RenderTargetMode::FreshWithColor(wgpu::Color::TRANSPARENT),
            draw_encoder,
        );
        staging_belt
            .write_buffer(draw_encoder, &self.uniform_buffer, 0, self.uniform_size)
            .copy_from_slice(bytemuck::cast_slice(&[GlowUniform {
                color: [
                    f32::from(filter.color.r) / 255.0,
                    f32::from(filter.color.g) / 255.0,
                    f32::from(filter.color.b) / 255.0,
                    f32::from(filter.color.a) / 255.0,
                ],
                strength: filter.strength.to_f32(),
                inner: if filter.is_inner() { 1 } else { 0 },
                knockout: if filter.is_knockout() { 1 } else { 0 },
                composite_source: if filter.composite_source() { 1 } else { 0 },
            }]));
        staging_belt
            .write_buffer(draw_encoder, &self.vertex_buffer, 0, self.vertices_size)
            .copy_from_slice(bytemuck::cast_slice(&[
                source.vertices_with_blur_offset(blur_offset)
            ]));
        let filter_group = descriptors
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: create_debug_label!("Filter group").as_deref(),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&source_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(
                            descriptors.bitmap_samplers.get_sampler(false, false),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&blurred_view),
                    },
                ],
            });
        crate::perf_counters::count(crate::perf_counters::Counter::RenderPass);
        let mut render_pass = draw_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: create_debug_label!("Glow filter").as_deref(),
            color_attachments: &[target.color_attachments()],
            ..Default::default()
        });
        render_pass.set_pipeline(pipeline);

        render_pass.set_bind_group(0, &filter_group, &[]);

        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_index_buffer(
            descriptors.quad.indices.slice(..),
            wgpu::IndexFormat::Uint32,
        );
        render_pass.draw_indexed(0..6, 0, 0..1);
        target
    }
}
