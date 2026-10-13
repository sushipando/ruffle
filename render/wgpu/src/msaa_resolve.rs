//! Resolving only part of a multisampled texture.
//!
//! Resolving a whole multisampled render target at the end of every render pass is
//! extremely expensive on tile-based GPUs when a target is split into many passes
//! (as complex blend modes do). Instead, targets are resolved once when finished,
//! and complex blends resolve just the region they need with this shader.

use crate::descriptors::Descriptors;
use fnv::FnvHashMap;
use std::sync::Mutex;

#[derive(Debug)]
pub struct MsaaRegionResolve {
    layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    shader: wgpu::ShaderModule,
    pipelines: Mutex<FnvHashMap<wgpu::TextureFormat, wgpu::RenderPipeline>>,
}

impl MsaaRegionResolve {
    pub fn new(device: &wgpu::Device) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: create_debug_label!("MSAA region resolve bind group layout").as_deref(),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: true,
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: create_debug_label!("MSAA region resolve pipeline layout").as_deref(),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: create_debug_label!("MSAA region resolve shader").as_deref(),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/msaa_region_resolve.wgsl").into(),
            ),
        });
        Self {
            layout,
            pipeline_layout,
            shader,
            pipelines: Default::default(),
        }
    }

    fn pipeline(&self, device: &wgpu::Device, format: wgpu::TextureFormat) -> wgpu::RenderPipeline {
        let mut pipelines = self.pipelines.lock().expect("pipelines lock");
        pipelines
            .entry(format)
            .or_insert_with(|| {
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: create_debug_label!("MSAA region resolve pipeline {format:?}")
                        .as_deref(),
                    layout: Some(&self.pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &self.shader,
                        entry_point: Some("main_vertex"),
                        compilation_options: Default::default(),
                        buffers: &[],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &self.shader,
                        entry_point: Some("main_fragment"),
                        compilation_options: Default::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    primitive: Default::default(),
                    depth_stencil: None,
                    multisample: Default::default(),
                    multiview_mask: None,
                    cache: None,
                })
            })
            .clone()
    }

    /// Resolves the region of `source` (multisampled) starting at `offset` into
    /// the whole of `destination` (single-sampled, with `format`).
    pub fn resolve(
        descriptors: &Descriptors,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::TextureView,
        destination: &wgpu::TextureView,
        format: wgpu::TextureFormat,
        offset: (u32, u32),
    ) {
        Self::resolve_many(
            descriptors,
            encoder,
            source,
            destination,
            format,
            &[ResolveRegion {
                source: offset,
                destination: (0, 0),
                size: None,
            }],
        );
    }

    /// Resolves several regions of `source` (multisampled) into `destination`
    /// (single-sampled, with `format`) in a single render pass.
    pub fn resolve_many(
        descriptors: &Descriptors,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::TextureView,
        destination: &wgpu::TextureView,
        format: wgpu::TextureFormat,
        regions: &[ResolveRegion],
    ) {
        let this = &descriptors.msaa_region_resolve;
        let bind_group = descriptors
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: create_debug_label!("MSAA region resolve bind group").as_deref(),
                layout: &this.layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(source),
                }],
            });
        let pipeline = this.pipeline(&descriptors.device, format);

        crate::perf_counters::count(crate::perf_counters::Counter::RenderPass);
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: create_debug_label!("MSAA region resolve").as_deref(),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: destination,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            ..Default::default()
        });
        render_pass.set_pipeline(&pipeline);
        render_pass.set_bind_group(0, &bind_group, &[]);
        for region in regions {
            if let Some((width, height)) = region.size {
                render_pass.set_viewport(
                    region.destination.0 as f32,
                    region.destination.1 as f32,
                    width as f32,
                    height as f32,
                    0.0,
                    1.0,
                );
            }
            let dx = region.source.0 as i32 - region.destination.0 as i32;
            let dy = region.source.1 as i32 - region.destination.1 as i32;
            let packed_offset = (dx as u16 as u32) | ((dy as u16 as u32) << 16);
            render_pass.draw(0..3, packed_offset..packed_offset + 1);
        }
    }
}

/// A region to resolve: from `source` in the source texture, to `destination`
/// in the destination texture, of `size` (or the whole destination if `None`).
pub struct ResolveRegion {
    pub source: (u32, u32),
    pub destination: (u32, u32),
    pub size: Option<(u32, u32)>,
}
