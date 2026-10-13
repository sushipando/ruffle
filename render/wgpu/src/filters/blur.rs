use crate::backend::RenderTargetMode;
use crate::buffer_pool::TexturePool;
use crate::descriptors::Descriptors;
use crate::filters::{FilterSource, FilterVertex, VERTEX_BUFFERS_DESCRIPTION_FILTERS};
use crate::surface::target::CommandTarget;
use crate::utils::SampleCountMap;
use bytemuck::{Pod, Zeroable};
use std::sync::OnceLock;
use swf::BlurFilter as BlurFilterArgs;
use swf::Fixed16;
use wgpu::util::StagingBelt;
use wgpu::{BufferSlice, CommandEncoder, RenderPipeline, TextureView};

/// This is a 1:1 match of `struct Filter` in `blur.wgsl`. See that, and the usage below, for more info.
/// Since WebGL requires 16 byte struct size (alignment), some of these fields (namely m2 and last_weight)
/// are passed in precomputed, even though they are trivial to get (addition/multiplication by constant).
/// The struct would have to be padded with dummy data otherwise anyway - these are at least useful.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, PartialEq)]
struct BlurUniform {
    direction: [f32; 2],
    full_size: f32,
    m: f32,
    m2: f32,
    first_weight: f32,
    last_offset: f32,
    last_weight: f32,
}

/// A blur "pass" that just copies (with bilinear sampling).
const COPY_UNIFORM: BlurUniform = BlurUniform {
    direction: [0.0, 0.0],
    full_size: 1.0,
    m: 0.0,
    m2: 0.0,
    first_weight: 1.0,
    last_offset: 0.0,
    last_weight: 0.0,
};

/// The most taps a merged kernel can have (see `kernel.taps` in `blur.wgsl`).
const MAX_KERNEL_TAPS: usize = 256;

/// The size of `struct Kernel` in `blur.wgsl`.
pub(super) const KERNEL_UNIFORM_SIZE: u64 = 16 + 16 * (MAX_KERNEL_TAPS as u64 / 2);

/// Combines `passes` box blurs of `strength` (as `BlurFilter::apply_full` does them, one
/// after the other) into one kernel. Returns its taps as `(offset, weight)`, each sampling
/// between two neighboring texels with bilinear filtering, or `None` if there are too many.
pub(super) fn merged_kernel(strength: f32, passes: u8) -> Option<Vec<[f32; 2]>> {
    // See `apply_full` for what these mean.
    let full_size = strength.min(255.0);
    let radius = (full_size - 1.0) / 2.0;
    let m = radius.ceil() - 1.0;
    let alpha = ((radius - m) * 255.0).floor() / 255.0;
    let r = m as usize + 1;
    let mut box_kernel = vec![1.0 / full_size as f64; 2 * r + 1];
    box_kernel[0] = alpha as f64 / full_size as f64;
    box_kernel[2 * r] = box_kernel[0];

    let mut kernel = vec![1.0f64];
    for _ in 0..passes {
        let mut next = vec![0.0; kernel.len() + box_kernel.len() - 1];
        for (i, a) in kernel.iter().enumerate() {
            for (j, b) in box_kernel.iter().enumerate() {
                next[i + j] += a * b;
            }
        }
        kernel = next;
    }
    if kernel.len().div_ceil(2) > MAX_KERNEL_TAPS {
        return None;
    }
    let center = (kernel.len() / 2) as f64;
    Some(
        kernel
            .chunks(2)
            .enumerate()
            .filter_map(|(index, pair)| {
                let (w0, w1) = (pair[0], pair.get(1).copied().unwrap_or(0.0));
                let weight = w0 + w1;
                (weight > 0.0).then(|| {
                    [
                        ((2 * index) as f64 - center + w1 / weight) as f32,
                        weight as f32,
                    ]
                })
            })
            .collect(),
    )
}

/// How many times lower the resolution of a blur may be in each direction (see
/// `BlurFilter::apply_downscaled`).
///
/// Only when the stage is magnified (never in the tests), and for each direction
/// separately: 2 if the blur is at least 8 pixels in that direction (so never below the
/// movie's own resolution), 4 if it's at least 64 and 8 if it's at least 128 (so still at
/// least 16 pixels at the lower resolution; such large blurs have no detail to lose).
/// A direction without any blur (such as `blurX = 0`) keeps its resolution.
pub(super) fn downscale_factors(filter: &BlurFilterArgs) -> (u32, u32) {
    if !ruffle_render::perf_flags::enabled(ruffle_render::perf_flags::PerfFlag::BlurDownscale)
        || ruffle_render::filters::view_scale() < 2.0
    {
        return (1, 1);
    }
    let factor = |strength: Fixed16| match strength.to_f32() {
        s if s < 8.0 => 1,
        s if s < 64.0 => 2,
        s if s < 128.0 => 4,
        _ => 8,
    };
    (factor(filter.blur_x), factor(filter.blur_y))
}

pub struct BlurFilter {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    vertex_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    vertices_size: wgpu::BufferSize,
    uniform_size: wgpu::BufferSize,
    pipelines: SampleCountMap<OnceLock<wgpu::RenderPipeline>>,
    kernel_bind_group_layout: wgpu::BindGroupLayout,
    kernel_pipeline_layout: wgpu::PipelineLayout,
    kernel_uniform_buffer: wgpu::Buffer,
    kernel_pipelines: SampleCountMap<OnceLock<wgpu::RenderPipeline>>,
    /// Like `kernel_bind_group_layout`, with a dynamic offset into the kernels (see
    /// `filters::batch`).
    pub(super) dynamic_kernel_bind_group_layout: wgpu::BindGroupLayout,
    dynamic_kernel_pipeline_layout: wgpu::PipelineLayout,
    dynamic_kernel_pipeline: OnceLock<wgpu::RenderPipeline>,
}

impl BlurFilter {
    pub fn new(device: &wgpu::Device) -> Self {
        let texture = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                multisampled: false,
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        };
        let sampling = wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        };
        let uniform_size = std::mem::size_of::<BlurUniform>() as u64;
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                texture,
                sampling,
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(uniform_size),
                    },
                    count: None,
                },
            ],
            label: create_debug_label!("Blur filter binds (with buffer)").as_deref(),
        });

        let vertices_size = std::mem::size_of::<[FilterVertex; 4]>() as u64;
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

        let kernel_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    texture,
                    sampling,
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(KERNEL_UNIFORM_SIZE),
                        },
                        count: None,
                    },
                ],
                label: create_debug_label!("Blur filter binds (kernel)").as_deref(),
            });
        let kernel_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&kernel_bind_group_layout)],
                immediate_size: 0,
            });
        let dynamic_kernel_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    texture,
                    sampling,
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: true,
                            min_binding_size: wgpu::BufferSize::new(KERNEL_UNIFORM_SIZE),
                        },
                        count: None,
                    },
                ],
                label: create_debug_label!("Blur filter binds (dynamic kernel)").as_deref(),
            });
        let dynamic_kernel_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&dynamic_kernel_bind_group_layout)],
                immediate_size: 0,
            });
        let kernel_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: KERNEL_UNIFORM_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            kernel_bind_group_layout,
            kernel_pipeline_layout,
            kernel_uniform_buffer,
            kernel_pipelines: Default::default(),
            dynamic_kernel_bind_group_layout,
            dynamic_kernel_pipeline_layout,
            dynamic_kernel_pipeline: Default::default(),
            pipelines: Default::default(),
            pipeline_layout,
            vertex_buffer,
            uniform_buffer,
            bind_group_layout,
            vertices_size: wgpu::BufferSize::new(vertices_size).expect("Definitely not zero."),
            uniform_size: wgpu::BufferSize::new(uniform_size).expect("Definitely not zero."),
        }
    }

    fn pipeline(&self, descriptors: &Descriptors, msaa_sample_count: u32) -> &wgpu::RenderPipeline {
        self.pipelines.get_or_init(msaa_sample_count, || {
            self.create_pipeline(descriptors, msaa_sample_count, false)
        })
    }

    /// The kernel pipeline with `dynamic_kernel_bind_group_layout` (single-sampled).
    pub(super) fn dynamic_kernel_pipeline(
        &self,
        descriptors: &Descriptors,
    ) -> &wgpu::RenderPipeline {
        self.dynamic_kernel_pipeline.get_or_init(|| {
            descriptors
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: create_debug_label!("Blur Filter (dynamic kernel)").as_deref(),
                    layout: Some(&self.dynamic_kernel_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &descriptors.shaders.blur_filter,
                        entry_point: Some("kernel_vertex"),
                        buffers: &VERTEX_BUFFERS_DESCRIPTION_FILTERS,
                        compilation_options: Default::default(),
                    },
                    primitive: Default::default(),
                    depth_stencil: None,
                    multisample: Default::default(),
                    fragment: Some(wgpu::FragmentState {
                        module: &descriptors.shaders.blur_filter,
                        entry_point: Some("kernel_fragment"),
                        targets: &[Some(wgpu::TextureFormat::Rgba8Unorm.into())],
                        compilation_options: Default::default(),
                    }),
                    multiview_mask: None,
                    cache: None,
                })
        })
    }

    fn kernel_pipeline(
        &self,
        descriptors: &Descriptors,
        msaa_sample_count: u32,
    ) -> &wgpu::RenderPipeline {
        self.kernel_pipelines.get_or_init(msaa_sample_count, || {
            self.create_pipeline(descriptors, msaa_sample_count, true)
        })
    }

    fn create_pipeline(
        &self,
        descriptors: &Descriptors,
        msaa_sample_count: u32,
        kernel: bool,
    ) -> wgpu::RenderPipeline {
        {
            let label = create_debug_label!("Blur Filter ({} msaa)", msaa_sample_count);
            descriptors
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: label.as_deref(),
                    layout: Some(if kernel {
                        &self.kernel_pipeline_layout
                    } else {
                        &self.pipeline_layout
                    }),
                    vertex: wgpu::VertexState {
                        module: &descriptors.shaders.blur_filter,
                        entry_point: Some(if kernel {
                            "kernel_vertex"
                        } else {
                            "main_vertex"
                        }),
                        buffers: &VERTEX_BUFFERS_DESCRIPTION_FILTERS,
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
                        module: &descriptors.shaders.blur_filter,
                        entry_point: Some(if kernel {
                            "kernel_fragment"
                        } else {
                            "main_fragment"
                        }),
                        targets: &[Some(wgpu::TextureFormat::Rgba8Unorm.into())],
                        compilation_options: Default::default(),
                    }),
                    multiview_mask: None,
                    cache: None,
                })
        }
    }

    pub fn apply(
        &self,
        descriptors: &Descriptors,
        texture_pool: &mut TexturePool,
        draw_encoder: &mut wgpu::CommandEncoder,
        staging_belt: &mut StagingBelt,
        source: &FilterSource,
        filter: &BlurFilterArgs,
    ) -> Option<CommandTarget> {
        let factors = downscale_factors(filter);
        if factors != (1, 1) {
            return self.apply_downscaled(
                descriptors,
                texture_pool,
                draw_encoder,
                staging_belt,
                source,
                filter,
                factors,
            );
        }
        self.apply_full(
            descriptors,
            texture_pool,
            draw_encoder,
            staging_belt,
            source,
            filter,
        )
    }

    /// Blurs at `factors` (powers of 2) times lower resolution in each direction: the
    /// source is scaled down in steps of 2 (bilinear sampling between two texels averages
    /// them), blurred with that many times smaller sizes, and scaled back up (bilinear).
    /// In each direction, the blur work is the factor squared times smaller (fewer pixels,
    /// each with fewer taps).
    #[expect(clippy::too_many_arguments)]
    fn apply_downscaled(
        &self,
        descriptors: &Descriptors,
        texture_pool: &mut TexturePool,
        draw_encoder: &mut wgpu::CommandEncoder,
        staging_belt: &mut StagingBelt,
        source: &FilterSource,
        filter: &BlurFilterArgs,
        factors: (u32, u32),
    ) -> Option<CommandTarget> {
        let sample_count = source.texture.sample_count();
        let format = source.texture.format();
        let pipeline = self.pipeline(descriptors, sample_count);
        staging_belt
            .write_buffer(draw_encoder, &self.vertex_buffer, 0, self.vertices_size)
            .copy_from_slice(bytemuck::cast_slice(&[source.vertices()]));
        let source_view = source.texture.create_view(&Default::default());
        let mut small: Option<CommandTarget> = None;
        let mut small_size = source.size;
        let mut step = (1, 1);
        while step.0 < factors.0 || step.1 < factors.1 {
            if step.0 < factors.0 {
                small_size.0 = small_size.0.div_ceil(2);
                step.0 *= 2;
            }
            if step.1 < factors.1 {
                small_size.1 = small_size.1.div_ceil(2);
                step.1 *= 2;
            }
            let next = CommandTarget::new(
                descriptors,
                texture_pool,
                wgpu::Extent3d {
                    width: small_size.0,
                    height: small_size.1,
                    depth_or_array_layers: 1,
                },
                format,
                sample_count,
                RenderTargetMode::FreshWithColor(wgpu::Color::TRANSPARENT),
                draw_encoder,
            );
            let (view, vertices) = match &small {
                // The first step reads the source's region.
                None => (&source_view, self.vertex_buffer.slice(..)),
                Some(previous) => (
                    previous.color_view(),
                    descriptors.quad.filter_vertices.slice(..),
                ),
            };
            self.write_uniform(draw_encoder, staging_belt, COPY_UNIFORM);
            self.render_with_uniform_buffers(
                descriptors,
                draw_encoder,
                pipeline,
                &next,
                view,
                vertices,
            );
            small = Some(next);
        }
        let small = small?;

        let small_filter = BlurFilterArgs {
            blur_x: Fixed16::from_f32(filter.blur_x.to_f32() / factors.0 as f32),
            blur_y: Fixed16::from_f32(filter.blur_y.to_f32() / factors.1 as f32),
            flags: filter.flags,
        };
        let small_source = FilterSource {
            texture: small.color_texture(),
            point: (0, 0),
            size: small_size,
        };
        let blurred = self.apply_full(
            descriptors,
            texture_pool,
            draw_encoder,
            staging_belt,
            &small_source,
            &small_filter,
        )?;

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
        self.write_uniform(draw_encoder, staging_belt, COPY_UNIFORM);
        self.render_with_uniform_buffers(
            descriptors,
            draw_encoder,
            pipeline,
            &target,
            blurred.color_view(),
            descriptors.quad.filter_vertices.slice(..),
        );
        Some(target)
    }

    fn write_uniform(
        &self,
        draw_encoder: &mut wgpu::CommandEncoder,
        staging_belt: &mut StagingBelt,
        uniform: BlurUniform,
    ) {
        staging_belt
            .write_buffer(draw_encoder, &self.uniform_buffer, 0, self.uniform_size)
            .copy_from_slice(bytemuck::cast_slice(&[uniform]));
    }

    fn apply_full(
        &self,
        descriptors: &Descriptors,
        texture_pool: &mut TexturePool,
        draw_encoder: &mut wgpu::CommandEncoder,
        staging_belt: &mut StagingBelt,
        source: &FilterSource,
        filter: &BlurFilterArgs,
    ) -> Option<CommandTarget> {
        if ruffle_render::perf_flags::enabled(ruffle_render::perf_flags::PerfFlag::BlurMerge)
            && ruffle_render::filters::rendering_for_display()
            && let Some(result) = self.apply_merged(
                descriptors,
                texture_pool,
                draw_encoder,
                staging_belt,
                source,
                filter,
            )
        {
            return result;
        }
        let sample_count = source.texture.sample_count();
        let format = source.texture.format();
        let pipeline = self.pipeline(descriptors, sample_count);

        let mut flip = CommandTarget::new(
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
        let mut flop = CommandTarget::new(
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
            .write_buffer(draw_encoder, &self.vertex_buffer, 0, self.vertices_size)
            .copy_from_slice(bytemuck::cast_slice(&[source.vertices()]));

        if crate::perf_counters::enabled() {
            let taps = |s: f32| ((s.min(255.0) - 1.0) / 2.0).ceil() as u64 + 2;
            count_blur(
                filter,
                source,
                (taps(filter.blur_x.to_f32()) + taps(filter.blur_y.to_f32()))
                    * filter.num_passes() as u64,
            );
        }

        let source_view = source.texture.create_view(&Default::default());
        let mut first = true;
        for _ in 0..(filter.num_passes() as usize) {
            for i in 0..2 {
                let horizontal = i % 2 == 0;
                let strength = if horizontal {
                    filter.blur_x.to_f32()
                } else {
                    filter.blur_y.to_f32()
                };
                // Full width of the kernel (left edge to right edge)
                let full_size = strength.min(255.0);
                if full_size <= 1.0 {
                    // A width of 1 or less is a noop (it'd just sample itself and nothing else)
                    continue;
                }

                let (previous_view, previous_vertices, previous_width, previous_height) = if first {
                    first = false;
                    (
                        &source_view,
                        self.vertex_buffer.slice(..),
                        source.texture.width() as f32,
                        source.texture.height() as f32,
                    )
                } else {
                    (
                        flip.color_view(),
                        descriptors.quad.filter_vertices.slice(..),
                        flip.width() as f32,
                        flip.height() as f32,
                    )
                };

                // See this article for additional information on the fractional blur algorithm, as this
                // implementation was inspired by it: https://fgiesen.wordpress.com/2012/08/01/fast-blurs-2/

                // This is how much the blur "extends past" the center pixel to either side.
                let radius = (full_size - 1.0) / 2.0;

                // This is how many simple double-1 weighted pixel pairs we can sample in the center.
                // Note how we're not using floor() here. This is to guarantee that alpha is not 0 when
                // radius is a whole number: That would cause the division below to end the universe,
                // and more importantly, also waste at least one sampling of the texture (the first one).
                // This way, alpha is 1 instead in those cases (with m being one smaller), and the last
                // two samplings can be fused into one, at the right place and with the right weight.
                let m = radius.ceil() - 1.0;
                // Not the transparency kind. It's almost the fractional part of radius.
                // If radius is a whole number, however, it's 1 instead of 0.
                // The rounding is done to imitate the fixed-point calculations in Flash Player,
                // improving emulation accuracy somewhat.
                let alpha = ((radius - m) * 255.0).floor() / 255.0;

                // These control how and where the last pair of pixels are to be sampled,
                // so that the next-to-last will end up with an effective weight of 1.0,
                // and the last one with a weight of alpha. Note that the offset is relative
                // to the center of the next-to-last sampled pixel, in the range of 0 to 0.5.
                let last_offset = 1.0 / ((1.0 / alpha) + 1.0);
                let last_weight = alpha + 1.0;

                let uniform = BlurUniform {
                    direction: if horizontal {
                        [1.0 / previous_width, 0.0]
                    } else {
                        [0.0, 1.0 / previous_height]
                    },
                    full_size,
                    m,
                    m2: m * 2.0,
                    first_weight: alpha,
                    last_offset,
                    last_weight,
                };
                staging_belt
                    .write_buffer(draw_encoder, &self.uniform_buffer, 0, self.uniform_size)
                    .copy_from_slice(bytemuck::cast_slice(&[uniform]));

                self.render_with_uniform_buffers(
                    descriptors,
                    draw_encoder,
                    pipeline,
                    &flop,
                    previous_view,
                    previous_vertices,
                );

                std::mem::swap(&mut flip, &mut flop);
            }
        }

        if first {
            // Nothing happened, don't return an empty unused texture
            None
        } else {
            Some(flip)
        }
    }

    /// Like `apply_full`, but does all passes in each direction at once with a combined
    /// kernel (see `merged_kernel`): two render passes instead of two per pass. The result
    /// isn't bit-exact: it's rounded after each direction instead of after each pass, and
    /// the first pass reads further outside of the source (`passes` times the radius,
    /// which only matters for sources with opaque edges). So it's only used for frames
    /// rendered for display. Returns `None` if a kernel would be too large.
    fn apply_merged(
        &self,
        descriptors: &Descriptors,
        texture_pool: &mut TexturePool,
        draw_encoder: &mut wgpu::CommandEncoder,
        staging_belt: &mut StagingBelt,
        source: &FilterSource,
        filter: &BlurFilterArgs,
    ) -> Option<Option<CommandTarget>> {
        let kernel = |strength: f32| {
            if strength.min(255.0) <= 1.0 || filter.num_passes() == 0 {
                // A no-op in this direction (see `apply_full`).
                Some(None)
            } else {
                merged_kernel(strength, filter.num_passes()).map(Some)
            }
        };
        let kernels = [
            (true, kernel(filter.blur_x.to_f32())?),
            (false, kernel(filter.blur_y.to_f32())?),
        ];
        if crate::perf_counters::enabled() {
            let taps: usize = kernels.iter().flat_map(|(_, k)| k).map(Vec::len).sum();
            count_blur(filter, source, taps as u64);
        }

        let sample_count = source.texture.sample_count();
        let format = source.texture.format();
        let pipeline = self.kernel_pipeline(descriptors, sample_count);
        staging_belt
            .write_buffer(draw_encoder, &self.vertex_buffer, 0, self.vertices_size)
            .copy_from_slice(bytemuck::cast_slice(&[source.vertices()]));
        let source_view = source.texture.create_view(&Default::default());
        let mut previous: Option<CommandTarget> = None;
        for (horizontal, taps) in kernels {
            let Some(taps) = taps else {
                continue;
            };
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
            let (view, vertices, width, height) = match &previous {
                // The first pass reads the source's region.
                None => (
                    &source_view,
                    self.vertex_buffer.slice(..),
                    source.texture.width() as f32,
                    source.texture.height() as f32,
                ),
                Some(previous) => (
                    previous.color_view(),
                    descriptors.quad.filter_vertices.slice(..),
                    previous.width() as f32,
                    previous.height() as f32,
                ),
            };
            // `struct Kernel` in `blur.wgsl`, up to the last pair of taps.
            let mut uniform = vec![0.0f32; 4 + taps.len().div_ceil(2) * 4];
            if horizontal {
                uniform[0] = 1.0 / width;
            } else {
                uniform[1] = 1.0 / height;
            }
            uniform[2] = f32::from_bits(taps.len() as u32);
            for (index, tap) in taps.iter().enumerate() {
                uniform[4 + index * 2..][..2].copy_from_slice(tap);
            }
            let size = wgpu::BufferSize::new((uniform.len() * 4) as u64).expect("not empty");
            staging_belt
                .write_buffer(draw_encoder, &self.kernel_uniform_buffer, 0, size)
                .copy_from_slice(bytemuck::cast_slice(&uniform));
            self.render_kernel_pass(descriptors, draw_encoder, pipeline, &target, view, vertices);
            previous = Some(target);
        }
        Some(previous)
    }

    fn render_kernel_pass(
        &self,
        descriptors: &Descriptors,
        draw_encoder: &mut CommandEncoder,
        pipeline: &RenderPipeline,
        destination: &CommandTarget,
        source: &TextureView,
        vertices: BufferSlice,
    ) {
        let filter_group = descriptors
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: create_debug_label!("Filter group").as_deref(),
                layout: &self.kernel_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(source),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(
                            descriptors.bitmap_samplers.get_sampler(false, true),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.kernel_uniform_buffer.as_entire_binding(),
                    },
                ],
            });

        crate::perf_counters::count(crate::perf_counters::Counter::RenderPass);
        let mut render_pass = draw_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: create_debug_label!("Blur filter").as_deref(),
            color_attachments: &[destination.color_attachments()],
            ..Default::default()
        });
        render_pass.set_pipeline(pipeline);
        render_pass.set_bind_group(0, &filter_group, &[]);
        render_pass.set_vertex_buffer(0, vertices);
        render_pass.set_index_buffer(
            descriptors.quad.indices.slice(..),
            wgpu::IndexFormat::Uint32,
        );
        render_pass.draw_indexed(0..6, 0, 0..1);
    }

    fn render_with_uniform_buffers(
        &self,
        descriptors: &Descriptors,
        draw_encoder: &mut CommandEncoder,
        pipeline: &RenderPipeline,
        destination: &CommandTarget,
        source: &TextureView,
        vertices: BufferSlice,
    ) {
        let filter_group = descriptors
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: create_debug_label!("Filter group").as_deref(),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(source),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(
                            descriptors.bitmap_samplers.get_sampler(false, true),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.uniform_buffer.as_entire_binding(),
                    },
                ],
            });

        crate::perf_counters::count(crate::perf_counters::Counter::RenderPass);
        let mut render_pass = draw_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: create_debug_label!("Blur filter").as_deref(),
            color_attachments: &[destination.color_attachments()],
            ..Default::default()
        });
        render_pass.set_pipeline(pipeline);

        render_pass.set_bind_group(0, &filter_group, &[]);

        render_pass.set_vertex_buffer(0, vertices);
        render_pass.set_index_buffer(
            descriptors.quad.indices.slice(..),
            wgpu::IndexFormat::Uint32,
        );
        render_pass.draw_indexed(0..6, 0, 0..1);
    }
}

/// Counts a blur's work for `RUFFLE_PERF` (`[gpu-named]`), with `taps` texture samples per pixel.
fn count_blur(filter: &BlurFilterArgs, source: &FilterSource, taps: u64) {
    let strength = filter.blur_x.to_f32().max(filter.blur_y.to_f32());
    let bucket = match strength {
        s if s <= 4.0 => "s<=4",
        s if s <= 8.0 => "s<=8",
        s if s <= 16.0 => "s<=16",
        s if s <= 32.0 => "s<=32",
        _ => "s>32",
    };
    let pixels = source.size.0 as u64 * source.size.1 as u64;
    let total_taps = pixels * taps;
    if total_taps > 20_000_000 && std::env::var_os("RUFFLE_BLUR_DEBUG").is_some() {
        eprintln!(
            "[blur] {:.1}x{:.1} q{} on {}x{} (of a {}x{} texture): {}M taps",
            filter.blur_x.to_f32(),
            filter.blur_y.to_f32(),
            filter.num_passes(),
            source.size.0,
            source.size.1,
            source.texture.width(),
            source.texture.height(),
            total_taps / 1_000_000
        );
    }
    crate::perf_counters::count_named(format!("blur/{bucket}/q{}", filter.num_passes()));
    crate::perf_counters::count_named_n("blur-kpx", pixels / 1000);
    crate::perf_counters::count_named_n("blur-mtaps", total_taps / 1_000_000);
    crate::perf_counters::count_named_n(format!("blur-mtaps/{bucket}"), total_taps / 1_000_000);
}
