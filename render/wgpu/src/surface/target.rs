use crate::backend::RenderTargetMode;
use crate::buffer_pool::{AlwaysCompatible, PoolEntry, TexturePool};
use crate::descriptors::Descriptors;
use crate::globals::Globals;
use crate::utils::run_copy_pipeline;
use std::cell::{Cell, OnceCell};
use std::sync::Arc;

#[derive(Debug)]
pub struct ResolveBuffer {
    texture: PoolOrArcTexture,
}

impl ResolveBuffer {
    pub fn new(
        descriptors: &Descriptors,
        size: wgpu::Extent3d,
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
        pool: &mut TexturePool,
    ) -> Self {
        let texture = pool.get_texture(descriptors, size, usage, format, 1);
        Self {
            texture: PoolOrArcTexture::Pool(texture),
        }
    }

    pub fn new_manual(texture: wgpu::Texture) -> Self {
        Self {
            texture: PoolOrArcTexture::Manual((
                texture.clone(),
                texture.create_view(&Default::default()),
            )),
        }
    }

    pub fn view(&self) -> &wgpu::TextureView {
        match self.texture {
            PoolOrArcTexture::Pool(ref texture) => &texture.1,
            PoolOrArcTexture::Manual(ref texture) => &texture.1,
        }
    }

    pub fn texture(&self) -> &wgpu::Texture {
        match self.texture {
            PoolOrArcTexture::Pool(ref texture) => &texture.0,
            PoolOrArcTexture::Manual(ref texture) => &texture.0,
        }
    }

    pub fn take_texture(self) -> PoolOrArcTexture {
        self.texture
    }
}

#[derive(Debug)]
pub struct FrameBuffer {
    texture: PoolOrArcTexture,
    size: wgpu::Extent3d,
}

#[derive(Debug)]
/// Holds either a `PoolEntry` texture, or an `Arc`-wrapped texture.
/// This is used to select between using a texture pool for our framebuffer/resolve-buffer
/// (when rendering to the main screen), or rendering to a non-pooled `Texture`
/// (when doing an offscreen render to a BitmapData texture)
pub enum PoolOrArcTexture {
    Pool(PoolEntry<(wgpu::Texture, wgpu::TextureView), AlwaysCompatible>),
    Manual((wgpu::Texture, wgpu::TextureView)),
}

impl PoolOrArcTexture {
    pub fn texture(&self) -> &wgpu::Texture {
        match self {
            PoolOrArcTexture::Pool(texture) => &texture.0,
            PoolOrArcTexture::Manual(texture) => &texture.0,
        }
    }
    pub fn view(&self) -> &wgpu::TextureView {
        match self {
            PoolOrArcTexture::Pool(texture) => &texture.1,
            PoolOrArcTexture::Manual(texture) => &texture.1,
        }
    }
}

impl FrameBuffer {
    pub fn new(
        descriptors: &Descriptors,
        sample_count: u32,
        size: wgpu::Extent3d,
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
        pool: &mut TexturePool,
    ) -> Self {
        let texture = pool.get_texture(descriptors, size, usage, format, sample_count);

        Self {
            texture: PoolOrArcTexture::Pool(texture),
            size,
        }
    }

    pub fn new_manual(texture: wgpu::Texture, size: wgpu::Extent3d) -> Self {
        Self {
            texture: PoolOrArcTexture::Manual((
                texture.clone(),
                texture.create_view(&Default::default()),
            )),
            size,
        }
    }

    pub fn view(&self) -> &wgpu::TextureView {
        match self.texture {
            PoolOrArcTexture::Pool(ref texture) => &texture.1,
            PoolOrArcTexture::Manual(ref texture) => &texture.1,
        }
    }

    pub fn texture(&self) -> &wgpu::Texture {
        match self.texture {
            PoolOrArcTexture::Pool(ref texture) => &texture.0,
            PoolOrArcTexture::Manual(ref texture) => &texture.0,
        }
    }

    pub fn take_texture(self) -> PoolOrArcTexture {
        self.texture
    }

    pub fn size(&self) -> wgpu::Extent3d {
        self.size
    }
}

#[derive(Debug)]
pub struct BlendBuffer {
    texture: PoolEntry<(wgpu::Texture, wgpu::TextureView), AlwaysCompatible>,
}

impl BlendBuffer {
    pub fn new(
        descriptors: &Descriptors,
        size: wgpu::Extent3d,
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
        pool: &mut TexturePool,
    ) -> Self {
        let texture = pool.get_texture(descriptors, size, usage, format, 1);

        Self { texture }
    }

    pub fn view(&self) -> &wgpu::TextureView {
        &self.texture.1
    }

    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture.0
    }
}

#[derive(Debug)]
pub struct StencilBuffer {
    texture: PoolEntry<(wgpu::Texture, wgpu::TextureView), AlwaysCompatible>,
}

impl StencilBuffer {
    pub fn new(
        descriptors: &Descriptors,
        msaa_sample_count: u32,
        size: wgpu::Extent3d,
        pool: &mut TexturePool,
    ) -> Self {
        let texture = pool.get_texture(
            descriptors,
            size,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
            wgpu::TextureFormat::Stencil8,
            msaa_sample_count,
        );

        Self { texture }
    }

    pub fn view(&self) -> &wgpu::TextureView {
        &self.texture.1
    }
}

pub struct CommandTarget {
    frame_buffer: FrameBuffer,
    blend_buffer: OnceCell<BlendBuffer>,
    resolve_buffer: Option<ResolveBuffer>,
    depth: OnceCell<StencilBuffer>,
    globals: Arc<Globals>,
    size: wgpu::Extent3d,
    format: wgpu::TextureFormat,
    sample_count: u32,
    color_needs_clear: OnceCell<bool>,
    render_target_mode: RenderTargetMode,
    /// Whether the multisampled frame buffer has been drawn to since it was
    /// last resolved into the resolve buffer.
    needs_resolve: Cell<bool>,
}

impl CommandTarget {
    pub fn new(
        descriptors: &Descriptors,
        pool: &mut TexturePool,
        size: wgpu::Extent3d,
        format: wgpu::TextureFormat,
        sample_count: u32,
        render_target_mode: RenderTargetMode,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Self {
        crate::perf_counters::count(crate::perf_counters::Counter::CommandTarget);
        let globals = pool.get_globals(descriptors, size.width, size.height);

        let mut make_pooled_frame_buffer = || {
            FrameBuffer::new(
                descriptors,
                sample_count,
                size,
                format,
                if sample_count > 1 {
                    wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING
                } else {
                    wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::COPY_SRC
                        | wgpu::TextureUsages::COPY_DST
                        | wgpu::TextureUsages::TEXTURE_BINDING
                },
                pool,
            )
        };

        let (frame_buffer, resolve_buffer) =
            if let RenderTargetMode::ExistingWithColor(texture, _) = &render_target_mode {
                if sample_count > 1 {
                    (
                        make_pooled_frame_buffer(),
                        Some(ResolveBuffer::new_manual(texture.clone())),
                    )
                } else {
                    (
                        FrameBuffer::new_manual(texture.clone(), texture.size()),
                        None,
                    )
                }
            } else if sample_count > 1 {
                (
                    make_pooled_frame_buffer(),
                    Some(ResolveBuffer::new(
                        descriptors,
                        size,
                        format,
                        wgpu::TextureUsages::COPY_SRC
                            | wgpu::TextureUsages::COPY_DST
                            | wgpu::TextureUsages::TEXTURE_BINDING
                            | wgpu::TextureUsages::RENDER_ATTACHMENT,
                        pool,
                    )),
                )
            } else {
                (make_pooled_frame_buffer(), None)
            };

        if let RenderTargetMode::FreshWithTexture(texture) = &render_target_mode {
            if let Some(resolve_buffer) = &resolve_buffer {
                encoder.copy_texture_to_texture(
                    texture.as_image_copy(),
                    resolve_buffer.texture().as_image_copy(),
                    size,
                );
            }

            if sample_count > 1 {
                // Both our frame buffer and resolve buffer need to start out
                // in the same state, so copy our existing texture to the freshly
                // allocated frame buffer. We cannot use `copy_texture_to_texture`,
                // since the sample counts are different.
                run_copy_pipeline(
                    descriptors,
                    format,
                    frame_buffer.texture.view(),
                    &texture.create_view(&Default::default()),
                    &globals,
                    sample_count,
                    encoder,
                );
            } else {
                encoder.copy_texture_to_texture(
                    texture.as_image_copy(),
                    frame_buffer.texture().as_image_copy(),
                    size,
                );
            }
        }

        Self {
            frame_buffer,
            blend_buffer: OnceCell::new(),
            resolve_buffer,
            depth: OnceCell::new(),
            globals,
            size,
            format,
            sample_count,
            color_needs_clear: OnceCell::new(),
            render_target_mode,
            needs_resolve: Cell::new(false),
        }
    }

    pub fn width(&self) -> u32 {
        self.size.width
    }

    pub fn height(&self) -> u32 {
        self.size.height
    }

    pub fn ensure_cleared(&self, encoder: &mut wgpu::CommandEncoder) {
        if self.color_needs_clear.get().is_some() {
            return;
        }
        // If we aren't clearing with a color (eg a texture instead)
        // the there's no point in creating a new render pass that does nothing.
        if self.render_target_mode.color().is_some() {
            crate::perf_counters::count(crate::perf_counters::Counter::RenderPass);
            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: create_debug_label!("Clearing command target").as_deref(),
                color_attachments: &[self.color_attachments()],
                ..Default::default()
            });
        }
    }

    /// Ensures the target is cleared (if it was never drawn to), and that the
    /// resolve buffer (if any) contains everything drawn so far.
    pub fn ensure_resolved(&self, encoder: &mut wgpu::CommandEncoder) {
        self.ensure_cleared(encoder);
        if !self.needs_resolve.replace(false) {
            return;
        }
        let Some(resolve_buffer) = &self.resolve_buffer else {
            return;
        };
        // An empty pass, just to resolve.
        crate::perf_counters::count(crate::perf_counters::Counter::RenderPass);
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: create_debug_label!("Resolving command target").as_deref(),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: self.frame_buffer.view(),
                resolve_target: Some(resolve_buffer.view()),
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            ..Default::default()
        });
    }

    pub fn take_color_texture(self) -> PoolOrArcTexture {
        debug_assert!(!self.needs_resolve.get(), "target must be resolved first");
        self.resolve_buffer
            .map(|b| b.take_texture())
            .unwrap_or_else(|| self.frame_buffer.take_texture())
    }

    pub fn globals(&self) -> &Globals {
        &self.globals
    }

    /// Like `color_attachments`, but if `is_last` is set, the pass also resolves the
    /// target (so that no separate resolve is needed afterwards).
    pub fn color_attachments_for_pass(
        &self,
        is_last: bool,
    ) -> Option<wgpu::RenderPassColorAttachment<'_>> {
        let mut attachment = self.color_attachments();
        if is_last
            && let (Some(attachment), Some(resolve_buffer)) =
                (&mut attachment, &self.resolve_buffer)
        {
            attachment.resolve_target = Some(resolve_buffer.view());
            self.needs_resolve.set(false);
        }
        attachment
    }

    pub fn color_attachments(&self) -> Option<wgpu::RenderPassColorAttachment<'_>> {
        let mut load = wgpu::LoadOp::Load;
        if self.color_needs_clear.set(false).is_ok()
            && let Some(clear_color) = self.render_target_mode.color()
        {
            load = wgpu::LoadOp::Clear(clear_color);
        }
        let lazy_resolve = self.resolve_buffer.is_some()
            && ruffle_render::perf_flags::enabled(ruffle_render::perf_flags::PerfFlag::LazyResolve);
        let resolve_target = if lazy_resolve {
            // Resolved later, by `ensure_resolved` or `copy_region_for_blend`.
            self.needs_resolve.set(true);
            None
        } else {
            self.resolve_buffer.as_ref().map(|b| b.view())
        };
        Some(wgpu::RenderPassColorAttachment {
            view: self.frame_buffer.view(),
            resolve_target,
            ops: wgpu::Operations {
                load,
                store: wgpu::StoreOp::Store,
            },
            depth_slice: None,
        })
    }

    pub fn sample_count(&self) -> u32 {
        self.sample_count
    }

    pub fn stencil_attachment(
        &self,
        descriptors: &Descriptors,
        pool: &mut TexturePool,
    ) -> Option<wgpu::RenderPassDepthStencilAttachment<'_>> {
        let new_buffer = self.depth.get().is_none();
        let stencil = self
            .depth
            .get_or_init(|| StencilBuffer::new(descriptors, self.sample_count, self.size, pool));
        Some(wgpu::RenderPassDepthStencilAttachment {
            view: stencil.view(),
            depth_ops: None,
            stencil_ops: Some(wgpu::Operations {
                load: if new_buffer {
                    wgpu::LoadOp::Clear(0)
                } else {
                    wgpu::LoadOp::Load
                },
                store: wgpu::StoreOp::Store,
            }),
        })
    }

    pub fn update_blend_buffer(
        &self,
        descriptors: &Descriptors,
        pool: &mut TexturePool,
        encoder: &mut wgpu::CommandEncoder,
    ) -> &BlendBuffer {
        let blend_buffer = self.blend_buffer.get_or_init(|| {
            BlendBuffer::new(
                descriptors,
                self.size,
                self.format,
                wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::COPY_SRC,
                pool,
            )
        });
        self.ensure_resolved(encoder);
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: self
                    .resolve_buffer
                    .as_ref()
                    .map(|b| b.texture())
                    .unwrap_or_else(|| self.frame_buffer.texture()),
                mip_level: 0,
                origin: Default::default(),
                aspect: Default::default(),
            },
            wgpu::TexelCopyTextureInfo {
                texture: blend_buffer.texture(),
                mip_level: 0,
                origin: Default::default(),
                aspect: Default::default(),
            },
            self.frame_buffer.size(),
        );
        blend_buffer
    }

    /// Copies the given region of this target into a new texture of the region's size,
    /// for use as the background of a bounded complex blend.
    pub fn copy_region_for_blend(
        &self,
        descriptors: &Descriptors,
        pool: &mut TexturePool,
        encoder: &mut wgpu::CommandEncoder,
        region: BlendRegion,
    ) -> BlendBuffer {
        let blend_buffer = BlendBuffer::new(
            descriptors,
            region.size(),
            self.format,
            wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            pool,
        );
        self.ensure_cleared(encoder);
        if self.needs_resolve.get() {
            // Only resolve the part of the frame buffer we need.
            crate::msaa_resolve::MsaaRegionResolve::resolve(
                descriptors,
                encoder,
                self.frame_buffer.view(),
                blend_buffer.view(),
                self.format,
                (region.x, region.y),
            );
            return blend_buffer;
        }
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: self.color_texture(),
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: region.x,
                    y: region.y,
                    z: 0,
                },
                aspect: Default::default(),
            },
            wgpu::TexelCopyTextureInfo {
                texture: blend_buffer.texture(),
                mip_level: 0,
                origin: Default::default(),
                aspect: Default::default(),
            },
            region.size(),
        );
        blend_buffer
    }

    /// Like `copy_region_for_blend`, but for several non-overlapping regions at once.
    /// If the target needs resolving, this takes a single render pass for all of them.
    pub fn copy_regions_for_blends(
        &self,
        descriptors: &Descriptors,
        pool: &mut TexturePool,
        encoder: &mut wgpu::CommandEncoder,
        regions: &[BlendRegion],
    ) -> Vec<BlendBuffer> {
        const USAGE: wgpu::TextureUsages = wgpu::TextureUsages::TEXTURE_BINDING
            .union(wgpu::TextureUsages::COPY_DST)
            .union(wgpu::TextureUsages::COPY_SRC)
            .union(wgpu::TextureUsages::RENDER_ATTACHMENT);
        self.ensure_cleared(encoder);
        let buffers: Vec<BlendBuffer> = regions
            .iter()
            .map(|region| BlendBuffer::new(descriptors, region.size(), self.format, USAGE, pool))
            .collect();

        let (source, positions) = if self.needs_resolve.get() {
            // Resolve all regions into rows of an atlas, then copy them out of it.
            let max_width = descriptors.limits.max_texture_dimension_2d;
            let atlas_width = regions
                .iter()
                .map(|r| r.width)
                .max()
                .unwrap_or(1)
                .max(4096.min(max_width));
            let mut positions = Vec::with_capacity(regions.len());
            let (mut x, mut y, mut row_height) = (0, 0, 0);
            for region in regions {
                if x + region.width > atlas_width {
                    (x, y, row_height) = (0, y + row_height, 0);
                }
                positions.push((x, y));
                x += region.width;
                row_height = row_height.max(region.height);
            }
            // Round sizes up, so that textures can be reused between frames.
            let atlas_width = atlas_width.next_multiple_of(256).min(max_width);
            let atlas_height = (y + row_height).next_multiple_of(256).min(max_width);
            if y + row_height > max_width {
                // Too much to fit in one texture; do them one by one.
                drop(buffers);
                return regions
                    .iter()
                    .map(|region| self.copy_region_for_blend(descriptors, pool, encoder, *region))
                    .collect();
            }
            let atlas = BlendBuffer::new(
                descriptors,
                wgpu::Extent3d {
                    width: atlas_width,
                    height: atlas_height,
                    depth_or_array_layers: 1,
                },
                self.format,
                USAGE,
                pool,
            );
            let resolve_regions: Vec<_> = regions
                .iter()
                .zip(&positions)
                .map(
                    |(region, &destination)| crate::msaa_resolve::ResolveRegion {
                        source: (region.x, region.y),
                        destination,
                        size: Some((region.width, region.height)),
                    },
                )
                .collect();
            crate::msaa_resolve::MsaaRegionResolve::resolve_many(
                descriptors,
                encoder,
                self.frame_buffer.view(),
                atlas.view(),
                self.format,
                &resolve_regions,
            );
            (Some(atlas), positions)
        } else {
            (None, regions.iter().map(|r| (r.x, r.y)).collect())
        };

        let source_texture = source
            .as_ref()
            .map(|atlas| atlas.texture())
            .unwrap_or_else(|| self.color_texture());
        for ((region, buffer), (x, y)) in regions.iter().zip(&buffers).zip(positions) {
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: source_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x, y, z: 0 },
                    aspect: Default::default(),
                },
                buffer.texture().as_image_copy(),
                region.size(),
            );
        }
        buffers
    }

    pub fn color_view(&self) -> &wgpu::TextureView {
        debug_assert!(!self.needs_resolve.get(), "target must be resolved first");
        self.resolve_buffer
            .as_ref()
            .map(|b| b.view())
            .unwrap_or_else(|| self.frame_buffer.view())
    }

    pub fn color_texture(&self) -> &wgpu::Texture {
        debug_assert!(!self.needs_resolve.get(), "target must be resolved first");
        self.resolve_buffer
            .as_ref()
            .map(|b| b.texture())
            .unwrap_or_else(|| self.frame_buffer.texture())
    }
}

/// A rectangle of pixels within a `CommandTarget`, in the target's own texture coordinates.
#[derive(Debug, Clone, Copy)]
pub struct BlendRegion {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl BlendRegion {
    pub fn overlaps(&self, other: &BlendRegion) -> bool {
        self.x < other.x + other.width
            && other.x < self.x + self.width
            && self.y < other.y + other.height
            && other.y < self.y + self.height
    }

    pub fn size(&self) -> wgpu::Extent3d {
        wgpu::Extent3d {
            width: self.width,
            height: self.height,
            depth_or_array_layers: 1,
        }
    }
}
