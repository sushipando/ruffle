use crate::gui::MENU_HEIGHT;
use ruffle_render_wgpu::descriptors::Descriptors;
use ruffle_render_wgpu::target::{RenderTarget, RenderTargetFrame};
use std::borrow::Cow;
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[derive(Debug)]
pub struct MovieViewRenderer {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
    /// A quad over the whole viewport (the window under the menu, if it's shown).
    vertices: wgpu::Buffer,
    /// A quad over the whole render pass viewport, for drawing a region of the movie
    /// into a viewport set to its place (see `MovieView::render`).
    unit_vertices: wgpu::Buffer,
}

fn get_vertices(has_menu: bool, height: u32, scale_factor: f64) -> [[f32; 4]; 6] {
    let top = if has_menu {
        let menu_height = MENU_HEIGHT as f64 * scale_factor;
        1.0 - ((menu_height / height as f64) * 2.0) as f32
    } else {
        1.0
    };
    // x y u v
    [
        [-1.0, top, 0.0, 0.0],  // tl
        [1.0, top, 1.0, 0.0],   // tr
        [1.0, -1.0, 1.0, 1.0],  // br
        [1.0, -1.0, 1.0, 1.0],  // br
        [-1.0, -1.0, 0.0, 1.0], // bl
        [-1.0, top, 0.0, 0.0],  // tl
    ]
}

impl MovieViewRenderer {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        has_menu: bool,
        height: u32,
        scale_factor: f64,
    ) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("blit.wgsl"))),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                entry_point: Some("vs_main"),
                module: &module,
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: 4 * 4,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    // 0: vec2 position
                    // 1: vec2 texture coordinates
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2],
                })],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                unclipped_depth: false,
                conservative: false,
                cull_mode: None,
                front_face: wgpu::FrontFace::default(),
                polygon_mode: wgpu::PolygonMode::default(),
                strip_index_format: None,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                alpha_to_coverage_enabled: false,
                count: 1,
                mask: !0,
            },

            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some(if surface_format.is_srgb() {
                    "fs_main_srgb_framebuffer"
                } else {
                    "fs_main_linear_framebuffer"
                }),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
        });
        let vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&get_vertices(has_menu, height, scale_factor)),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        let unit_vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&get_vertices(false, 1, 1.0)),
            usage: wgpu::BufferUsages::VERTEX,
        });

        Self {
            bind_group_layout,
            pipeline,
            sampler,
            vertices,
            unit_vertices,
        }
    }

    pub fn update_resolution(
        &self,
        descriptors: &Descriptors,
        has_menu: bool,
        height: u32,
        scale_factor: f64,
    ) {
        descriptors.queue.write_buffer(
            &self.vertices,
            0,
            bytemuck::cast_slice(&get_vertices(has_menu, height, scale_factor)),
        );
    }
}

/// The texture the movie is drawn into, drawn onto the window by the GUI's render pass.
#[derive(Debug)]
pub struct MovieView {
    renderer: Arc<MovieViewRenderer>,
    /// The viewport's size.
    width: u32,
    height: u32,
    /// The part of the viewport that `texture` holds, if not all of it (the rest is the
    /// letterbox, see `RenderTarget::set_visible_region`).
    region: Option<(i32, i32, u32, u32)>,
    texture: wgpu::Texture,
    bind_group: wgpu::BindGroup,
    #[cfg(feature = "tracy_images")]
    tracy_frame_captures: crate::tracy::FrameCapturesHolder,
}

impl MovieView {
    pub fn new(
        renderer: Arc<MovieViewRenderer>,
        device: &wgpu::Device,
        width: u32,
        height: u32,
        #[cfg(feature = "tracy_images")] tracy_frame_captures: crate::tracy::FrameCapturesHolder,
    ) -> Self {
        Self::with_region(
            renderer,
            device,
            width,
            height,
            None,
            #[cfg(feature = "tracy_images")]
            tracy_frame_captures,
        )
    }

    fn with_region(
        renderer: Arc<MovieViewRenderer>,
        device: &wgpu::Device,
        width: u32,
        height: u32,
        region: Option<(i32, i32, u32, u32)>,
        #[cfg(feature = "tracy_images")] tracy_frame_captures: crate::tracy::FrameCapturesHolder,
    ) -> Self {
        let (texture_width, texture_height) =
            region.map_or((width, height), |(_, _, width, height)| (width, height));
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: texture_width,
                height: texture_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            // Frames are drawn straight into it (see `RenderTargetFrame::texture`).
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &renderer.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&renderer.sampler),
                },
            ],
        });
        #[cfg(feature = "tracy_images")]
        tracy_frame_captures.set_target(device, Some(&texture));
        Self {
            renderer,
            width,
            height,
            region,
            texture,
            bind_group,
            #[cfg(feature = "tracy_images")]
            tracy_frame_captures,
        }
    }

    /// Draws the movie onto the window, in a pass over the whole window (of
    /// `window_size`, with the menu's height above the viewport); a region of it in its
    /// place, the rest being left as cleared.
    pub fn render(
        &self,
        renderer: &MovieViewRenderer,
        render_pass: &mut wgpu::RenderPass<'static>,
        window_size: (u32, u32),
        menu_height: f32,
    ) {
        render_pass.set_pipeline(&renderer.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        if let Some((x, y, width, height)) = self.region {
            render_pass.set_viewport(
                x as f32,
                y as f32 + menu_height,
                width as f32,
                height as f32,
                0.0,
                1.0,
            );
            render_pass.set_vertex_buffer(0, renderer.unit_vertices.slice(..));
            render_pass.draw(0..6, 0..1);
            render_pass.set_viewport(
                0.0,
                0.0,
                window_size.0 as f32,
                window_size.1 as f32,
                0.0,
                1.0,
            );
        } else {
            render_pass.set_vertex_buffer(0, renderer.vertices.slice(..));
            render_pass.draw(0..6, 0..1);
        }
    }
}

impl RenderTarget for MovieView {
    type Frame = MovieViewFrame;

    fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        // The region is set again for the new size before the next frame.
        *self = MovieView::with_region(
            self.renderer.clone(),
            device,
            width,
            height,
            None,
            #[cfg(feature = "tracy_images")]
            self.tracy_frame_captures.clone(),
        );
    }

    fn format(&self) -> wgpu::TextureFormat {
        self.texture.format()
    }

    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn get_next_texture(&mut self) -> Option<Self::Frame> {
        Some(MovieViewFrame {
            view: self.texture.create_view(&Default::default()),
            texture: self.texture.clone(),
        })
    }

    fn set_visible_region(
        &mut self,
        device: &wgpu::Device,
        region: Option<(i32, i32, u32, u32)>,
    ) -> bool {
        if region != self.region {
            *self = MovieView::with_region(
                self.renderer.clone(),
                device,
                self.width,
                self.height,
                region,
                #[cfg(feature = "tracy_images")]
                self.tracy_frame_captures.clone(),
            );
        }
        true
    }

    fn submit<I: IntoIterator<Item = wgpu::CommandBuffer>>(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        command_buffers: I,
        _frame: Self::Frame,
    ) -> wgpu::SubmissionIndex {
        queue.submit(command_buffers)
    }

    fn can_submit_asynchronously(&self) -> bool {
        true
    }
}

#[derive(Debug)]
pub struct MovieViewFrame {
    view: wgpu::TextureView,
    texture: wgpu::Texture,
}

impl RenderTargetFrame for MovieViewFrame {
    fn into_view(self) -> wgpu::TextureView {
        self.view
    }

    fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    fn texture(&self) -> Option<&wgpu::Texture> {
        Some(&self.texture)
    }
}
