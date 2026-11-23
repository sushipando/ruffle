use crate::buffer_pool::{BufferPool, TexturePool};
use crate::context3d::WgpuContext3D;
use crate::dynamic_transforms::DynamicTransforms;
use crate::filters::FilterSource;
use crate::mesh::{Mesh, PendingShape, tessellate_all};
use crate::pixel_bender::{ShaderMode, run_pixelbender_shader_impl};
use crate::surface::target::CommandTarget;
use crate::surface::{LayerRef, Surface};
use crate::target::{MaybeOwnedBuffer, TextureTarget};
use crate::target::{RenderTargetFrame, TextureBufferInfo};
use crate::utils::{BufferDimensions, run_copy_pipeline};
use crate::{
    Descriptors, Error, QueueSyncHandle, RenderTarget, SwapChainTarget, Texture, as_texture,
    format_list, get_backend_names,
};
use image::imageops::FilterType;
use ruffle_render::backend::{
    BitmapCacheEntry, CacheVolatility, Context3D, Context3DProfile, PixelBenderOutput,
    PixelBenderTarget,
};
use ruffle_render::backend::{RenderBackend, ShapeHandle, ShapeHandleImpl, ViewportDimensions};
use ruffle_render::bitmap::{
    Bitmap, BitmapFormat, BitmapHandle, BitmapSource, PixelRegion, RgbaBufRead, SyncHandle,
};
use ruffle_render::commands::CommandList;
use ruffle_render::error::Error as BitmapError;
use ruffle_render::filters::Filter;
use ruffle_render::pixel_bender::{PixelBenderShader, PixelBenderShaderHandle};
use ruffle_render::pixel_bender_support::PixelBenderShaderArgument;
use ruffle_render::quality::StageQuality;
use ruffle_render::shape_utils::DistilledShape;
use ruffle_render::tessellator::ShapeTessellator;
use std::any::Any;
use std::borrow::Cow;
use std::num::NonZeroU32;
use std::sync::Arc;
use swf::Color;
use tracing::instrument;
use wgpu::SubmissionIndex;
use wgpu_profiler::{GpuProfiler, GpuProfilerSettings};

/// Creates a wgpu instance with Ruffle's required configuration.
///
/// This disables indirect call validation because wgpu's validation runs a compute
/// shader that uses `array<u32>`, which requires the `DYNAMIC_ARRAY_SIZE` feature.
/// However, wgpu runs this shader without first checking if the device supports
/// that feature, causing device creation to fail on GPUs that lack it.
/// Since Ruffle doesn't use indirect draws, disabling this validation has no
/// functional impact.
///
/// See <https://github.com/gfx-rs/wgpu/issues/8799>
pub fn create_wgpu_instance(
    backends: wgpu::Backends,
    backend_options: wgpu::BackendOptions,
    display: Option<Box<dyn wgpu::wgt::WgpuHasDisplayHandle>>,
) -> wgpu::Instance {
    let descriptor = match display {
        Some(display) => wgpu::InstanceDescriptor::new_with_display_handle(display),
        None => wgpu::InstanceDescriptor::new_without_display_handle(),
    };
    wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends,
        flags: wgpu::InstanceFlags::default()
            .difference(wgpu::InstanceFlags::VALIDATION_INDIRECT_CALL)
            .with_env(),
        backend_options,
        ..descriptor
    })
}

/// Result of lazily acquiring a swapchain texture for direct-to-swapchain MSAA resolve
/// right before the final main pass is encoded.
pub enum DirectOutputAcquire {
    Ready(wgpu::Texture),
    SkipFrame,
}

pub struct WgpuRenderBackend<T: RenderTarget> {
    pub(crate) descriptors: Arc<Descriptors>,
    target: T,
    surface: Surface,
    /// See `RenderBackend::set_visible_region`.
    visible_region: Option<(i32, i32, u32, u32)>,
    /// Whether the target's frames hold just `visible_region` (see
    /// `RenderTarget::set_visible_region`).
    frame_is_region: bool,
    /// A surface covering just `visible_region`, used instead of `surface` when set.
    region_surface: Option<((i32, i32, u32, u32), Surface)>,
    meshes: Vec<Mesh>,
    shape_tessellator: ShapeTessellator,
    /// Shapes registered since the last frame, still to be tessellated (see `Mesh`).
    pending_shapes: Vec<std::sync::Weak<dyn ShapeHandleImpl>>,
    // This is currently unused - we just store it to report in
    // `get_viewport_dimensions`
    viewport_scale_factor: f64,
    texture_pool: TexturePool,
    offscreen_texture_pool: TexturePool,
    pub(crate) offscreen_buffer_pool: Arc<BufferPool<wgpu::Buffer, BufferDimensions>>,
    active_frame: ActiveFrame,
    profiler: GpuProfiler,
    /// Transient filtered CAB atlases kept alive for direct sampling during the current
    /// frame (and, for tick-scoped entries, across interpolated sub-frames of the same tick).
    filter_atlas: crate::filters::FilterAtlasResult,
    filter_atlas_tick: u32,
    /// Content-addressed LRU cache for looping pure-vector filtered CABs.
    cab_cache: crate::filters::FilteredCabCache,
    /// Content-addressed LRU cache for pure-vector offscreen blend layers.
    blend_cache: crate::filters::BlendLayerCache,
    /// When true, `submit_frame` leaves `active_frame.command_encoder` open so the desktop
    /// shell can append its `egui_render` pass onto the same Metal command buffer before
    /// submitting + presenting in one job.
    defer_submit: bool,
    deferred_frame_pending: bool,
    /// Optional lazy acquirer for a swapchain texture to resolve the frame straight into
    /// (bypassing `MovieView` and the fullscreen `egui_render` blit pass when no egui UI is
    /// visible). Invoked right before the final main pass so display-list building,
    /// tessellation, and offscreen cache/filter passes overlap with waiting for the next
    /// drawable.
    direct_output_acquirer: Option<Box<dyn FnOnce() -> DirectOutputAcquire>>,
    /// Cached full-window `Surface` matching `direct_output`'s dimensions, format, and quality.
    direct_surface: Option<((u32, u32), wgpu::TextureFormat, StageQuality, Surface)>,
}

impl WgpuRenderBackend<SwapChainTarget> {
    #[cfg(target_family = "wasm")]
    pub async fn for_canvas(
        canvas: web_sys::HtmlCanvasElement,
        webgpu: bool,
    ) -> Result<Self, Error> {
        let backends = if webgpu {
            wgpu::Backends::BROWSER_WEBGPU
        } else {
            wgpu::Backends::GL
        };
        let instance = create_wgpu_instance(
            backends,
            wgpu::BackendOptions {
                gl: wgpu::GlBackendOptions {
                    // See <https://github.com/gfx-rs/wgpu/releases/tag/v25.0.0>
                    fence_behavior: wgpu::GlFenceBehavior::AutoFinish,
                    ..Default::default()
                },
                ..Default::default()
            },
            None,
        );
        let surface = instance.create_surface(wgpu::SurfaceTarget::Canvas(canvas))?;
        let (adapter, device, queue) = request_adapter_and_device(
            backends,
            &instance,
            Some(&surface),
            wgpu::PowerPreference::HighPerformance,
        )
        .await?;
        let descriptors = Descriptors::new(instance, adapter, device, queue);
        let target =
            SwapChainTarget::new(surface, &descriptors.adapter, (1, 1), &descriptors.device);
        Self::new(Arc::new(descriptors), target)
    }

    /// # Safety
    ///  See [`wgpu::SurfaceTargetUnsafe`] variants for safety requirements.
    ///
    /// Since wgpu 29, a display handle is needed at instance creation time:
    /// pass one via `display`, or make sure the `window` target carries a raw
    /// display handle (note that `SurfaceTargetUnsafe::from_window` does not
    /// provide one). Prefer passing `display` - some backends (e.g. GL via
    /// EGL) select their platform when the instance is created, before the
    /// target's display handle is seen.
    #[cfg(not(target_family = "wasm"))]
    pub unsafe fn for_window_unsafe(
        window: wgpu::SurfaceTargetUnsafe,
        size: (u32, u32),
        backend: wgpu::Backends,
        power_preference: wgpu::PowerPreference,
        display: Option<Box<dyn wgpu::wgt::WgpuHasDisplayHandle>>,
    ) -> Result<Self, Error> {
        if wgpu::Backends::SECONDARY.contains(backend) {
            tracing::warn!(
                "{} graphics backend support may not be fully supported.",
                format_list(&get_backend_names(backend), "and")
            );
        }
        let instance = create_wgpu_instance(backend, wgpu::BackendOptions::default(), display);
        let surface = unsafe { instance.create_surface_unsafe(window)? };
        let (adapter, device, queue) = futures::executor::block_on(request_adapter_and_device(
            backend,
            &instance,
            Some(&surface),
            power_preference,
        ))?;
        let descriptors = Descriptors::new(instance, adapter, device, queue);
        let target = SwapChainTarget::new(surface, &descriptors.adapter, size, &descriptors.device);
        Self::new(Arc::new(descriptors), target)
    }

    /// # Safety
    ///  See [`wgpu::SurfaceTargetUnsafe`] variants for safety requirements.
    #[cfg(not(target_family = "wasm"))]
    pub unsafe fn recreate_surface_unsafe(
        &mut self,
        window: wgpu::SurfaceTargetUnsafe,
        size: (u32, u32),
    ) -> Result<(), Error> {
        let descriptors = &self.descriptors;
        let surface = unsafe { descriptors.wgpu_instance.create_surface_unsafe(window)? };
        self.target =
            SwapChainTarget::new(surface, &descriptors.adapter, size, &descriptors.device);
        Ok(())
    }
}

#[cfg(not(target_family = "wasm"))]
impl WgpuRenderBackend<crate::target::TextureTarget> {
    pub fn for_offscreen(
        size: (u32, u32),
        backend: wgpu::Backends,
        power_preference: wgpu::PowerPreference,
    ) -> Result<Self, Error> {
        if wgpu::Backends::SECONDARY.contains(backend) {
            tracing::warn!(
                "{} graphics backend support may not be fully supported.",
                format_list(&get_backend_names(backend), "and")
            );
        }
        let instance = create_wgpu_instance(backend, wgpu::BackendOptions::default(), None);
        let (adapter, device, queue) = futures::executor::block_on(request_adapter_and_device(
            backend,
            &instance,
            None,
            power_preference,
        ))?;
        let descriptors = Descriptors::new(instance, adapter, device, queue);
        let target = crate::target::TextureTarget::new(&descriptors.device, size)?;
        Self::new(Arc::new(descriptors), target)
    }

    pub fn capture_frame(&self) -> Option<image::RgbaImage> {
        use crate::utils::buffer_to_image;
        if let Some(buffer) = &self.target.buffer {
            let (buffer, dimensions) = buffer.buffer.inner();
            Some(buffer_to_image(
                &self.descriptors.device,
                buffer,
                dimensions,
                None,
                self.target.size,
            ))
        } else {
            None
        }
    }
}

impl<T: RenderTarget> WgpuRenderBackend<T> {
    pub fn new(descriptors: Arc<Descriptors>, target: T) -> Result<Self, Error> {
        if target.width() > descriptors.limits.max_texture_dimension_2d
            || target.height() > descriptors.limits.max_texture_dimension_2d
        {
            return Err(format!(
                "Render target texture cannot be larger than {}px on either dimension (requested {} x {})",
                descriptors.limits.max_texture_dimension_2d,
                target.width(),
                target.height()
            )
                .into());
        }

        let surface = Surface::new(
            &descriptors,
            StageQuality::Low,
            target.width(),
            target.height(),
            target.format(),
        );

        let offscreen_buffer_pool = BufferPool::new(Box::new(
            |descriptors: &Descriptors, dimensions: &BufferDimensions| {
                descriptors.device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: dimensions.size(),
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                })
            },
        ));

        let active_frame = ActiveFrame::new(&descriptors);

        let profiler_settings = GpuProfilerSettings {
            enable_timer_queries: cfg!(feature = "profile-with-tracy"),
            enable_debug_groups: cfg!(feature = "render_debug_labels"),
            ..Default::default()
        };
        #[cfg(feature = "profile-with-tracy")]
        let profiler = GpuProfiler::new_with_tracy_client(
            profiler_settings,
            descriptors.backend,
            &descriptors.device,
            &descriptors.queue,
        )?;
        #[cfg(not(feature = "profile-with-tracy"))]
        let profiler = GpuProfiler::new(&descriptors.device, profiler_settings)?;

        Ok(Self {
            descriptors,
            target,
            surface,
            visible_region: None,
            frame_is_region: false,
            region_surface: None,
            meshes: Vec::new(),
            shape_tessellator: ShapeTessellator::new(),
            pending_shapes: Vec::new(),
            viewport_scale_factor: 1.0,
            texture_pool: TexturePool::new(),
            offscreen_texture_pool: TexturePool::new(),
            offscreen_buffer_pool: Arc::new(offscreen_buffer_pool),
            active_frame,
            profiler,
            filter_atlas: crate::filters::FilterAtlasResult::default(),
            filter_atlas_tick: 0,
            cab_cache: crate::filters::FilteredCabCache::new(),
            blend_cache: crate::filters::BlendLayerCache::new(),
            defer_submit: false,
            deferred_frame_pending: false,
            direct_output_acquirer: None,
            direct_surface: None,
        })
    }

    fn finish_or_queue_pending(&mut self, pending: PendingShape) -> ShapeHandle {
        if ruffle_render::perf_flags::enabled(
            ruffle_render::perf_flags::PerfFlag::ParallelTessellation,
        ) {
            let handle = ShapeHandle(Arc::new(Mesh::pending(pending)));
            self.pending_shapes.push(Arc::downgrade(&handle.0));
            return handle;
        }
        let PendingShape { job, finish } = pending;
        let mesh = job.tessellate(&mut self.shape_tessellator);
        ShapeHandle(Arc::new(Mesh::ready(finish.build(mesh))))
    }

    fn register_shape_internal(
        &mut self,
        shape: DistilledShape,
        bitmap_source: &dyn BitmapSource,
        scale: f32,
    ) -> ShapeHandle {
        let pending =
            PendingShape::new(self.descriptors.clone(), shape, bitmap_source, self, scale);
        self.finish_or_queue_pending(pending)
    }

    /// Tessellates the shapes registered since the last frame (see `Mesh`), on several
    /// threads when there are enough of them.
    fn finish_pending_shapes(&mut self) {
        if self.pending_shapes.is_empty() {
            return;
        }
        let start = ruffle_render::tessellator::stats::enabled().then(std::time::Instant::now);
        let handles: Vec<_> = self
            .pending_shapes
            .drain(..)
            .filter_map(|shape| shape.upgrade())
            .collect();
        let mut meshes = Vec::with_capacity(handles.len());
        let mut jobs = Vec::with_capacity(handles.len());
        for handle in &handles {
            let mesh: &Mesh =
                <dyn Any>::downcast_ref(&**handle).expect("Shape handle must be a WGPU ShapeData");
            if let Some(PendingShape { job, finish }) = mesh.take_pending() {
                meshes.push((mesh, finish));
                jobs.push(job);
            }
        }
        let tessellated = tessellate_all(jobs, &mut self.shape_tessellator);
        for ((mesh, finish), tessellated) in meshes.into_iter().zip(tessellated) {
            mesh.set_tessellated(finish, tessellated);
        }
        if let Some(start) = start {
            ruffle_render::tessellator::stats::record(start.elapsed(), 0);
        }
    }

    fn clamp_bitmap(&self, bitmap: &mut Bitmap) -> bool {
        let max_size = self.descriptors.limits.max_texture_dimension_2d;
        if bitmap.width() > max_size || bitmap.height() > max_size {
            let image =
                image::RgbaImage::from_raw(bitmap.width(), bitmap.height(), bitmap.data().to_vec())
                    .expect("Width and height of bitmap must match bitmap data");

            let ratio = bitmap.width() as f32 / bitmap.height() as f32;
            let mut width = bitmap.width();
            let mut height = bitmap.height();
            if width > max_size {
                width = max_size;
                height = (max_size as f32 / ratio) as u32;
            }
            if height > max_size {
                height = max_size;
                width = (max_size as f32 * ratio) as u32;
            }
            let resized = image::imageops::resize(&image, width, height, FilterType::CatmullRom);
            *bitmap = Bitmap::new(width, height, BitmapFormat::Rgba, resized.into_raw());
            true
        } else {
            false
        }
    }

    pub fn descriptors(&self) -> &Arc<Descriptors> {
        &self.descriptors
    }

    pub fn target(&self) -> &T {
        &self.target
    }

    /// Runs `task` once everything submitted so far is on the queue. If the last
    /// frame is still being submitted on another thread, `task` runs there after
    /// it, so the caller doesn't have to wait (for example, to present a frame
    /// that shows this one).
    pub fn run_after_submissions(&mut self, task: Box<dyn FnOnce() + Send>) {
        self.active_frame.run_after_submissions(task);
    }

    pub fn set_defer_submit(&mut self, defer: bool) {
        self.defer_submit = defer;
    }

    /// Sets a one-shot lazy swapchain-texture acquirer for the next `submit_frame` to
    /// resolve straight into, bypassing `target` (`MovieView`) and any fullscreen GUI blit
    /// pass.
    pub fn set_direct_output_acquirer(
        &mut self,
        acquirer: Option<Box<dyn FnOnce() -> DirectOutputAcquire>>,
    ) {
        self.direct_output_acquirer = acquirer;
    }

    pub fn has_deferred_frame(&self) -> bool {
        self.deferred_frame_pending
    }

    /// Borrows the render target and the open deferred `CommandEncoder` simultaneously so
    /// the host GUI can append its present pass onto the movie's own command buffer.
    pub fn with_target_and_encoder<R>(
        &mut self,
        f: impl FnOnce(&T, &mut wgpu::CommandEncoder) -> R,
    ) -> R {
        f(&self.target, &mut self.active_frame.command_encoder)
    }

    /// Submits a deferred movie frame (plus any appended GUI pass) on the async submit
    /// thread, running `after` (e.g. `set_present_time` + `queue.present`) immediately
    /// after `queue.submit`.
    pub fn submit_deferred_frame(
        &mut self,
        extra_before: Vec<wgpu::CommandBuffer>,
        after: Option<Box<dyn FnOnce() + Send>>,
    ) {
        self.defer_submit = false;
        if !self.deferred_frame_pending {
            if let Some(after) = after {
                self.run_after_submissions(after);
            }
            return;
        }
        self.deferred_frame_pending = false;
        self.active_frame
            .submit_async_with_post(&self.descriptors, extra_before, after);
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.descriptors.device
    }

    pub fn make_queue_sync_handle(
        &self,
        target: TextureTarget,
        index: Option<SubmissionIndex>,
        destination: BitmapHandle,
        copy_area: PixelRegion,
    ) -> Box<QueueSyncHandle> {
        match target.take_buffer() {
            None => Box::new(QueueSyncHandle::NotCopied {
                handle: destination,
                copy_area,
                descriptors: self.descriptors.clone(),
                pool: self.offscreen_buffer_pool.clone(),
            }),
            Some(TextureBufferInfo {
                buffer: MaybeOwnedBuffer::Borrowed(buffer, copy_dimensions),
                ..
            }) => Box::new(QueueSyncHandle::AlreadyCopied {
                index,
                buffer,
                copy_dimensions,
                descriptors: self.descriptors.clone(),
            }),
            Some(TextureBufferInfo {
                buffer: MaybeOwnedBuffer::Owned(..),
                ..
            }) => unreachable!("Buffer must be Borrowed as it was set to be Borrowed earlier"),
        }
    }
}

impl<T: RenderTarget + 'static> RenderBackend for WgpuRenderBackend<T> {
    fn set_viewport_dimensions(&mut self, dimensions: ViewportDimensions) {
        self.cab_cache.clear();
        self.blend_cache.clear();
        self.filter_atlas.clear();
        // Avoid panics from creating 0-sized framebuffers.
        // TODO: find a way to bubble an error when the size is too large
        let width = std::cmp::max(
            std::cmp::min(
                dimensions.width,
                self.descriptors.limits.max_texture_dimension_2d,
            ),
            1,
        );
        let height = std::cmp::max(
            std::cmp::min(
                dimensions.height,
                self.descriptors.limits.max_texture_dimension_2d,
            ),
            1,
        );
        self.target.resize(&self.descriptors.device, width, height);

        self.surface = Surface::new(
            &self.descriptors,
            self.surface.quality(),
            width,
            height,
            self.target.format(),
        );

        self.viewport_scale_factor = dimensions.scale_factor;
        self.texture_pool = TexturePool::new();
        self.region_surface = None;
    }

    fn set_visible_region(&mut self, region: Option<(i32, i32, u32, u32)>) {
        let (width, height) = (self.target.width(), self.target.height());
        self.visible_region = region.filter(|&(x, y, w, h)| {
            ruffle_render::perf_flags::enabled(ruffle_render::perf_flags::PerfFlag::VisibleRegion)
                && x >= 0
                && y >= 0
                && w > 0
                && h > 0
                && x as u32 + w <= width
                && y as u32 + h <= height
                // Only worth it if it saves a meaningful amount.
                && (w as u64 * h as u64) * 10 < (width as u64 * height as u64) * 9
        });
        self.frame_is_region = self
            .target
            .set_visible_region(&self.descriptors.device, self.visible_region);
    }

    fn create_context3d(
        &mut self,
        profile: Context3DProfile,
    ) -> Result<Box<dyn Context3D>, BitmapError> {
        Ok(Box::new(WgpuContext3D::new(
            self.descriptors.clone(),
            profile,
        )))
    }

    fn debug_info(&self) -> Cow<'static, str> {
        let mut result = vec![];
        result.push("Renderer: wgpu".to_string());

        let info = self.descriptors.adapter.get_info();
        result.push(format!("Adapter Backend: {:?}", info.backend));
        result.push(format!("Adapter Name: {:?}", info.name));
        result.push(format!("Adapter Device Type: {:?}", info.device_type));
        result.push(format!("Adapter Driver Name: {:?}", info.driver));
        result.push(format!("Adapter Driver Info: {:?}", info.driver_info));

        let enabled_features = self.descriptors.device.features();
        let available_features = self.descriptors.adapter.features() - enabled_features;
        let current_limits = &self.descriptors.limits;

        result.push(format!("Enabled features: {enabled_features:?}"));
        result.push(format!("Available features: {available_features:?}"));
        result.push(format!("Current limits: {current_limits:?}"));
        result.push(format!("Surface quality: {}", self.surface.quality()));
        result.push(format!("Surface samples: {}", self.surface.sample_count()));
        result.push(format!("Surface size: {:?}", self.surface.size()));

        Cow::Owned(result.join("\n"))
    }

    fn name(&self) -> &'static str {
        if cfg!(target_family = "wasm") {
            let info = self.descriptors.adapter.get_info();
            if info.backend == wgpu::Backend::BrowserWebGpu {
                "webgpu"
            } else {
                "wgpu-webgl"
            }
        } else {
            "wgpu"
        }
    }

    fn set_quality(&mut self, quality: StageQuality) {
        if quality != self.surface.quality() {
            self.cab_cache.clear();
            self.blend_cache.clear();
            self.filter_atlas.clear();
        }
        self.surface = Surface::new(
            &self.descriptors,
            quality,
            self.surface.size().width,
            self.surface.size().height,
            self.target.format(),
        );
    }

    fn viewport_dimensions(&self) -> ViewportDimensions {
        ViewportDimensions {
            width: self.target.width(),
            height: self.target.height(),
            scale_factor: self.viewport_scale_factor,
        }
    }

    #[instrument(level = "debug", skip_all)]
    fn register_shape(
        &mut self,
        shape: DistilledShape,
        bitmap_source: &dyn BitmapSource,
    ) -> ShapeHandle {
        self.register_shape_internal(shape, bitmap_source, 1.0)
    }

    #[instrument(level = "debug", skip_all)]
    fn register_shape_with_scale(
        &mut self,
        shape: DistilledShape,
        bitmap_source: &dyn BitmapSource,
        scale: f32,
    ) -> ShapeHandle {
        self.register_shape_internal(shape, bitmap_source, scale)
    }

    #[instrument(level = "debug", skip_all)]
    fn register_raw_shape_with_scale(
        &mut self,
        raw: ruffle_render::backend::RawSwfShape,
        bitmap_source: &dyn BitmapSource,
        scale: f32,
    ) -> ShapeHandle {
        let pending =
            PendingShape::from_raw(self.descriptors.clone(), raw, bitmap_source, self, scale);
        self.finish_or_queue_pending(pending)
    }

    #[instrument(level = "debug", skip_all)]
    fn submit_frame(
        &mut self,
        clear: Color,
        commands: CommandList,
        cache_entries: Vec<BitmapCacheEntry>,
    ) {
        self.finish_pending_shapes();
        let mut cache_entries = cache_entries;
        if self.deferred_frame_pending {
            self.submit_deferred_frame(Vec::new(), None);
        }
        if ruffle_render::perf_flags::enabled(ruffle_render::perf_flags::PerfFlag::SkipCacheFilters)
        {
            for entry in &mut cache_entries {
                entry.filters.clear();
            }
        }
        // The previous frame may still be being submitted on another thread.
        // Recording this one meanwhile is fine: it uses another staging belt, and
        // everything that must come after the previous frame on the queue (the
        // transforms arena flush, recalling belts, other submissions) waits for it.
        if !ruffle_render::perf_flags::enabled(ruffle_render::perf_flags::PerfFlag::OverlapRecord) {
            self.active_frame.wait_for_submission();
        }
        let Some(frame_output) = self.target.get_next_texture() else {
            // Attempt to recreate the swap chain in this case.
            self.target.resize(
                &self.descriptors.device,
                self.target.width(),
                self.target.height(),
            );
            return;
        };

        // Cache entries are drawn and filtered in levels: an entry that draws other cache
        // entries' textures comes after them, so that they're complete by then. Within a
        // level, filters are applied to many entries at once where possible (see
        // `filters::batch` and `layer_atlas::apply_filters_batched`), after all of the
        // level's entries are drawn.
        let cache_textures: Vec<wgpu::Texture> = cache_entries
            .iter()
            .map(|entry| as_texture(&entry.handle).texture.clone())
            .collect();
        let cache_handles: Vec<_> = cache_entries.iter().map(|e| e.handle.clone()).collect();
        let references: Vec<Vec<usize>> = cache_entries
            .iter()
            .enumerate()
            .map(|(index, entry)| {
                let mut bitmaps = vec![];
                collect_bitmaps(&entry.commands, &mut bitmaps);
                bitmaps
                    .iter()
                    .filter_map(|bitmap| cache_handles.iter().position(|h| h == bitmap))
                    .filter(|&other| other != index)
                    .collect()
            })
            .collect();
        let mut levels = vec![0usize; cache_entries.len()];
        for _ in 0..cache_entries.len() {
            let mut changed = false;
            for index in 0..cache_entries.len() {
                let level = references[index]
                    .iter()
                    .map(|&other| levels[other] + 1)
                    .max()
                    .unwrap_or(0)
                    // Cycles can't happen, but don't loop forever if they do.
                    .min(cache_entries.len());
                if level != levels[index] {
                    levels[index] = level;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        let max_level = levels.iter().copied().max().unwrap_or(0);
        let batching =
            ruffle_render::perf_flags::enabled(ruffle_render::perf_flags::PerfFlag::FilterBatching);
        let together =
            ruffle_render::perf_flags::enabled(ruffle_render::perf_flags::PerfFlag::FilterTogether)
                && ruffle_render::perf_flags::enabled(
                    ruffle_render::perf_flags::PerfFlag::BlurMerge,
                )
                && ruffle_render::filters::rendering_for_display();

        self.cab_cache.begin_frame();
        self.blend_cache.begin_frame();
        let cur_tick = ruffle_render::filters::render_tick();
        if cur_tick == 0 || cur_tick != self.filter_atlas_tick {
            self.filter_atlas.clear();
            self.filter_atlas_tick = cur_tick;
        }
        for entry in &cache_entries {
            self.filter_atlas
                .remove_slot(Arc::as_ptr(&entry.handle.0) as *const () as usize);
        }

        // Cache entries whose contents are simple enough are drawn together in an atlas
        // per topological dependency level (so level-N entries can sample completed
        // level-<N cache entries), and copied into their textures (or drawn straight into
        // the filter atlas when they're filtered together).
        let used_sizes: Vec<(u32, u32)> = cache_entries
            .iter()
            .enumerate()
            .map(|(index, entry)| {
                let tex = &cache_textures[index];
                (
                    entry.size.0.min(tex.width()).max(1),
                    entry.size.1.min(tex.height()).max(1),
                )
            })
            .collect();
        let is_atlased: Vec<bool> = cache_entries
            .iter()
            .enumerate()
            .map(|(index, entry)| {
                use ruffle_render::commands::Command;
                let (width, height) = used_sizes[index];
                // Why an entry isn't atlased, for the `cab-atlas` counters.
                let reason = if !ruffle_render::perf_flags::enabled(
                    ruffle_render::perf_flags::PerfFlag::LayerAtlas,
                ) {
                    Some("off")
                } else if entry.clear.a != 0
                    || entry.clear.r != 0
                    || entry.clear.g != 0
                    || entry.clear.b != 0
                {
                    Some("clear")
                } else if !crate::surface::layer_atlas::fits(width, height) {
                    Some("size")
                } else if !crate::surface::layer_atlas::is_cab_leaf(
                    &self.descriptors,
                    &entry.commands,
                    width,
                    height,
                ) {
                    Some(
                        if entry.commands.commands.iter().any(|command| {
                            matches!(command, Command::Blend(..) | Command::BlendBounded(..))
                        }) {
                            "blend"
                        } else if entry.commands.commands.iter().any(|command| {
                            matches!(command, Command::PushMask | Command::RenderAlphaMask { .. })
                        }) {
                            "mask"
                        } else {
                            "commands"
                        },
                    )
                } else {
                    None
                };
                if crate::perf_counters::enabled() {
                    crate::perf_counters::count_named(format!(
                        "cab-atlas/{}",
                        reason.unwrap_or("yes")
                    ));
                }
                reason.is_none()
            })
            .collect();
        let max_filter_size = crate::surface::layer_atlas::max_batched_filter_size();
        let direct_to_filter_atlas: Vec<bool> = cache_entries
            .iter()
            .enumerate()
            .map(|(index, entry)| {
                let (uw, uh) = used_sizes[index];
                is_atlased[index]
                    && batching
                    && together
                    && !entry.filters.is_empty()
                    && uw <= max_filter_size
                    && uh <= max_filter_size
                    && entry.filters.iter().all(crate::filters::can_apply_together)
            })
            .collect();
        let mut cache_entries: Vec<_> = cache_entries.into_iter().map(Some).collect();

        for level in 0..=max_level {
            let members: Vec<usize> = (0..levels.len())
                .filter(|&index| levels[index] == level)
                .collect();
            let quality = self.surface.quality();
            let mut preserved_unfiltered_key = vec![false; levels.len()];
            let filter_atlas_key_ref =
                (!self.filter_atlas.slots.is_empty()).then_some(&self.filter_atlas);
            let mut pending_unfiltered_promotions: Vec<(
                crate::filters::CabContentKey,
                wgpu::Texture,
                crate::filters::CacheAllocation,
                (u32, u32),
            )> = Vec::new();
            let atlased: Vec<_> = members
                .iter()
                .copied()
                .filter(|&index| is_atlased[index] && !direct_to_filter_atlas[index])
                .filter_map(|index| {
                    let entry = cache_entries[index].as_mut()?;
                    let texture = as_texture(&entry.handle);
                    // Unfiltered CABs only ever sample `used_sizes[index]`; filtered CABs
                    // outside `filter-together` filter the full backing texture.
                    let (w, h) = if entry.filters.is_empty() {
                        used_sizes[index]
                    } else {
                        (texture.texture.width(), texture.texture.height())
                    };
                    if entry.filters.is_empty()
                        && entry.volatility == CacheVolatility::Persistent
                        && let Some(key) = crate::filters::CabContentKey::try_from_any_entry(
                            (w, h),
                            entry.clear,
                            quality,
                            &entry.filters,
                            &entry.commands,
                            filter_atlas_key_ref,
                        )
                    {
                        if texture.last_cab_key.borrow().as_ref() == Some(&key) {
                            entry.commands.commands.clear();
                            preserved_unfiltered_key[index] = true;
                            crate::perf_counters::count_named("cab-same/unfiltered");
                            return None;
                        }
                        if let Some(cached) = self.cab_cache.get(&key) {
                            entry.commands.commands.clear();
                            self.active_frame.command_encoder.copy_texture_to_texture(
                                wgpu::TexelCopyTextureInfo {
                                    texture: &cached.texture,
                                    mip_level: 0,
                                    origin: wgpu::Origin3d {
                                        x: cached.src_origin.0,
                                        y: cached.src_origin.1,
                                        z: 0,
                                    },
                                    aspect: Default::default(),
                                },
                                texture.texture.as_image_copy(),
                                wgpu::Extent3d {
                                    width: w,
                                    height: h,
                                    depth_or_array_layers: 1,
                                },
                            );
                            texture.bump_generation_keeping_key();
                            *texture.last_cab_key.borrow_mut() = Some(key);
                            preserved_unfiltered_key[index] = true;
                            crate::perf_counters::count_named("cab-lru/unfiltered-hit");
                            return None;
                        }
                        if self.cab_cache.should_promote(&key, (w, h), false) {
                            let alloc = self.cab_cache.allocate_entry(
                                &self.descriptors,
                                &mut self.active_frame.command_encoder,
                                (w, h),
                            );
                            pending_unfiltered_promotions.push((
                                key.clone(),
                                texture.texture.clone(),
                                alloc,
                                (w, h),
                            ));
                        }
                        *texture.last_cab_key.borrow_mut() = Some(key);
                        preserved_unfiltered_key[index] = true;
                    }
                    texture.bump_generation_keeping_key();
                    Some(crate::surface::layer_atlas::LeafLayer {
                        commands: std::mem::take(&mut entry.commands),
                        origin: (0, 0),
                        width: w,
                        height: h,
                        destination: Some((texture.texture.clone(), 0, 0)),
                        is_layer: false,
                    })
                })
                .collect();
            if !atlased.is_empty() {
                let filter_atlas_ref =
                    (!self.filter_atlas.slots.is_empty()).then_some(&self.filter_atlas);
                drop(crate::surface::layer_atlas::render_leaf_layers(
                    atlased,
                    &self.descriptors,
                    &self.meshes,
                    &mut self.active_frame.staging_belt,
                    &self.active_frame.dynamic_transforms,
                    &mut self
                        .profiler
                        .scope("Draw to CABs", &mut self.active_frame.command_encoder),
                    &mut self.offscreen_texture_pool,
                    self.surface.quality(),
                    filter_atlas_ref,
                ));
                for (key, src_tex, alloc, (w, h)) in pending_unfiltered_promotions {
                    self.active_frame.command_encoder.copy_texture_to_texture(
                        src_tex.as_image_copy(),
                        wgpu::TexelCopyTextureInfo {
                            texture: &alloc.texture,
                            mip_level: 0,
                            origin: wgpu::Origin3d {
                                x: alloc.dst_origin.0,
                                y: alloc.dst_origin.1,
                                z: 0,
                            },
                            aspect: Default::default(),
                        },
                        wgpu::Extent3d {
                            width: w,
                            height: h,
                            depth_or_array_layers: 1,
                        },
                    );
                    self.cab_cache.insert_entry(key, alloc, false);
                    crate::perf_counters::count_named("cab-lru/unfiltered-promote");
                }
            }
            let mut filter_groups: Vec<(Vec<Filter>, Vec<usize>)> = vec![];
            // Entries whose filters are applied together, whatever they are.
            let mut filtered_together: Vec<usize> = vec![];
            if batching {
                for &index in &members {
                    let entry = cache_entries[index]
                        .as_ref()
                        .expect("entries are only taken once");
                    if entry.filters.is_empty() {
                        continue;
                    }
                    let (uw, uh) = used_sizes[index];
                    if together
                        && uw <= max_filter_size
                        && uh <= max_filter_size
                        && entry.filters.iter().all(crate::filters::can_apply_together)
                    {
                        filtered_together.push(index);
                        continue;
                    }
                    let texture = &cache_textures[index];
                    if texture.width() > max_filter_size || texture.height() > max_filter_size {
                        continue;
                    }
                    if !entry
                        .filters
                        .iter()
                        .all(crate::surface::layer_atlas::is_local_filter)
                    {
                        continue;
                    }
                    match filter_groups
                        .iter_mut()
                        .find(|(filters, _)| *filters == entry.filters)
                    {
                        Some((_, group)) => group.push(index),
                        None => filter_groups.push((entry.filters.clone(), vec![index])),
                    }
                }
                filter_groups.retain(|(_, group)| group.len() >= 2);
            }
            if std::env::var_os("RUFFLE_FILTER_DEBUG").is_some() {
                log_cache_filters(
                    level,
                    &members,
                    &cache_entries,
                    &cache_textures,
                    &filter_groups,
                    &filtered_together,
                );
            }
            let mut filter_batched = vec![false; levels.len()];
            let mut in_filtered_together = vec![false; levels.len()];
            for (_, group) in &filter_groups {
                for &index in group {
                    filter_batched[index] = true;
                }
            }
            for &index in &filtered_together {
                filter_batched[index] = true;
                in_filtered_together[index] = true;
            }
            let quality = self.surface.quality();
            let mut filtered_together_entries: Vec<crate::filters::FilterTogetherEntry> =
                Vec::with_capacity(filtered_together.len());
            let mut pending_promotions: Vec<(
                usize,
                crate::filters::CabContentKey,
                crate::filters::CacheAllocation,
                bool,
            )> = Vec::new();
            let mut pending_persistent_keys: Vec<(usize, crate::filters::CabContentKey)> =
                Vec::new();
            let mut pending_slot_keys: Vec<(usize, Arc<crate::filters::CabContentKey>)> =
                Vec::new();
            let mut dedup_leaders: std::collections::HashMap<
                crate::filters::CabContentKey,
                usize,
            > = std::collections::HashMap::new();
            let mut dedup_followers: Vec<(
                usize,
                BitmapHandle,
                usize,
                wgpu::Texture,
                (u32, u32),
                CacheVolatility,
                crate::filters::CabContentKey,
            )> = Vec::new();
            for index in filtered_together {
                let entry = cache_entries[index]
                    .as_mut()
                    .expect("entries are only taken once");
                let handle_ptr = Arc::as_ptr(&entry.handle.0) as *const () as usize;
                let used_size = used_sizes[index];
                let is_persistent = entry.volatility == CacheVolatility::Persistent;
                let content_key = if direct_to_filter_atlas[index] && entry.clear.a == 0 {
                    let filter_atlas_ref =
                        (!self.filter_atlas.slots.is_empty()).then_some(&self.filter_atlas);
                    crate::filters::CabContentKey::try_from_entry(
                        used_size,
                        entry.clear,
                        quality,
                        &entry.filters,
                        &entry.commands,
                        filter_atlas_ref,
                    )
                } else {
                    None
                };
                // Same-handle content-hash skip: if this Persistent CAB's GPU texture
                // already holds the exact filtered result for `key`, skip all GPU work.
                if is_persistent
                    && let Some(ref key) = content_key
                    && as_texture(&entry.handle).last_cab_key.borrow().as_ref() == Some(key)
                {
                    entry.commands.commands.clear();
                    crate::perf_counters::count_named("cab-same/hit");
                    continue;
                }
                if let Some(ref key) = content_key
                    && let Some(cached) = self.cab_cache.get(key)
                {
                    entry.commands.commands.clear();
                    if !is_persistent {
                        self.filter_atlas.insert_external_slot(
                            entry.handle.clone(),
                            handle_ptr,
                            cached.bind_group.clone(),
                            cached.src_origin,
                            cached.atlas_size,
                            entry.volatility,
                            Some(Arc::new(key.clone())),
                        );
                    } else {
                        self.active_frame.command_encoder.copy_texture_to_texture(
                            wgpu::TexelCopyTextureInfo {
                                texture: &cached.texture,
                                mip_level: 0,
                                origin: wgpu::Origin3d {
                                    x: cached.src_origin.0,
                                    y: cached.src_origin.1,
                                    z: 0,
                                },
                                aspect: Default::default(),
                            },
                            cache_textures[index].as_image_copy(),
                            wgpu::Extent3d {
                                width: used_size.0,
                                height: used_size.1,
                                depth_or_array_layers: 1,
                            },
                        );
                        let tex = as_texture(&entry.handle);
                        tex.bump_generation_keeping_key();
                        *tex.last_cab_key.borrow_mut() = Some(key.clone());
                    }
                    crate::perf_counters::count_named("cab-lru/hit");
                    continue;
                }
                if let Some(ref key) = content_key
                    && let Some(&leader_ft_idx) = dedup_leaders.get(key)
                {
                    let leader_persistent = filtered_together_entries[leader_ft_idx].volatility
                        == CacheVolatility::Persistent;
                    // Same-volatility duplicates always share; a Persistent follower can also
                    // copy directly from a Tick/Subframe leader's finished atlas slot.
                    if leader_persistent == is_persistent || is_persistent {
                        entry.commands.commands.clear();
                        dedup_followers.push((
                            leader_ft_idx,
                            entry.handle.clone(),
                            handle_ptr,
                            cache_textures[index].clone(),
                            used_size,
                            entry.volatility,
                            key.clone(),
                        ));
                        crate::perf_counters::count_named("cab-lru/dedup");
                        continue;
                    }
                }
                let ft_idx = filtered_together_entries.len();
                let mut promote_copy = None;
                if let Some(key) = content_key {
                    let is_subframe = entry.volatility == CacheVolatility::Subframe;
                    let promote_eligible = !is_subframe
                        || u64::from(used_size.0) * u64::from(used_size.1)
                            <= crate::filters::SUBFRAME_PROMOTE_MAX_PIXELS;
                    if promote_eligible
                        && self.cab_cache.should_promote(&key, used_size, is_subframe)
                    {
                        let alloc = self.cab_cache.allocate_entry(
                            &self.descriptors,
                            &mut self.active_frame.command_encoder,
                            used_size,
                        );
                        promote_copy = Some((
                            alloc.texture.clone(),
                            alloc.dst_origin.0,
                            alloc.dst_origin.1,
                        ));
                        pending_promotions.push((ft_idx, key.clone(), alloc, is_subframe));
                    }
                    if is_persistent {
                        pending_persistent_keys.push((ft_idx, key.clone()));
                    } else {
                        pending_slot_keys.push((ft_idx, Arc::new(key.clone())));
                    }
                    dedup_leaders.insert(key, ft_idx);
                } else if is_persistent {
                    *as_texture(&entry.handle).last_cab_key.borrow_mut() = None;
                }
                let direct_commands = direct_to_filter_atlas[index]
                    .then(|| std::mem::take(&mut entry.commands));
                filtered_together_entries.push(crate::filters::FilterTogetherEntry {
                    texture: cache_textures[index].clone(),
                    used_size,
                    filters: entry.filters.clone(),
                    direct_commands,
                    handle: entry.handle.clone(),
                    handle_ptr,
                    volatility: entry.volatility,
                    promote_copy,
                });
            }
            let mut filtered_together = filtered_together_entries;

            for &index in &members {
                let entry = cache_entries[index]
                    .take()
                    .expect("entries are only taken once");
                let atlased = is_atlased[index];
                crate::perf_counters::count(crate::perf_counters::Counter::CacheEntry);
                if !entry.filters.is_empty() {
                    crate::perf_counters::count(crate::perf_counters::Counter::CacheEntryFiltered);
                }
                let texture = as_texture(&entry.handle);
                if !in_filtered_together[index] && !preserved_unfiltered_key[index] {
                    texture.bump_generation();
                } else if !atlased || !filter_batched[index] {
                    texture.bump_generation_keeping_key();
                }
                let surface = Surface::new(
                    &self.descriptors,
                    self.surface.quality(),
                    texture.texture.width(),
                    texture.texture.height(),
                    wgpu::TextureFormat::Rgba8Unorm,
                );
                let clear = wgpu::Color {
                    r: f64::from(entry.clear.r) / 255.0,
                    g: f64::from(entry.clear.g) / 255.0,
                    b: f64::from(entry.clear.b) / 255.0,
                    a: f64::from(entry.clear.a) / 255.0,
                };
                if entry.filters.is_empty() || filter_batched[index] {
                    if !atlased {
                        let filter_atlas_ref =
                            (!self.filter_atlas.slots.is_empty()).then_some(&self.filter_atlas);
                        surface.draw_commands(
                            RenderTargetMode::ExistingWithColor(texture.texture.clone(), clear),
                            &self.descriptors,
                            &self.meshes,
                            entry.commands,
                            &mut self.active_frame.staging_belt,
                            &self.active_frame.dynamic_transforms,
                            &mut self
                                .profiler
                                .scope("Draw to CAB", &mut self.active_frame.command_encoder),
                            LayerRef::None,
                            &mut self.offscreen_texture_pool,
                            filter_atlas_ref,
                            None,
                        );
                    }
                } else {
                    let mut scope = self
                        .profiler
                        .scope("Filters", &mut self.active_frame.command_encoder);
                    // We're relying on there being no impotent filters here,
                    // so that we can safely start by using the actual CAB texture.
                    // It's guaranteed that at least one filter would have used it and moved the target to something else,
                    // letting us safely copy back to it later.
                    let mut target: Option<CommandTarget> = if atlased {
                        // Already drawn into the texture.
                        None
                    } else {
                        let filter_atlas_ref =
                            (!self.filter_atlas.slots.is_empty()).then_some(&self.filter_atlas);
                        Some(surface.draw_commands(
                            RenderTargetMode::ExistingWithColor(texture.texture.clone(), clear),
                            &self.descriptors,
                            &self.meshes,
                            entry.commands,
                            &mut self.active_frame.staging_belt,
                            &self.active_frame.dynamic_transforms,
                            &mut scope.scope("Draw to CAB"),
                            LayerRef::None,
                            &mut self.offscreen_texture_pool,
                            filter_atlas_ref,
                            None,
                        ))
                    };
                    if crate::perf_counters::enabled() {
                        let (w, h) = (texture.texture.width(), texture.texture.height());
                        let size = match w * h {
                            0..=4096 => "tiny",
                            4097..=65536 => "small",
                            65537..=262144 => "medium",
                            _ => "large",
                        };
                        let names: Vec<_> = entry.filters.iter().map(|f| f.name()).collect();
                        crate::perf_counters::count_named(format!(
                            "cab/{size}/{}",
                            names.join("+")
                        ));
                    }
                    for filter in entry.filters {
                        crate::perf_counters::count(crate::perf_counters::Counter::Filter);
                        let source = match &target {
                            Some(target) => target.color_texture(),
                            None => &texture.texture,
                        };
                        target = Some(self.descriptors.filters.apply(
                            &self.descriptors,
                            &mut scope.scope(filter.name()),
                            &mut self.offscreen_texture_pool,
                            &mut self.active_frame.staging_belt,
                            FilterSource::for_entire_texture(source),
                            filter,
                        ));
                    }
                    let target = target.expect("there's at least one filter");
                    let result = target.color_texture();
                    if result.sample_count() == 1
                        && result.size() == texture.texture.size()
                        && result.format() == texture.texture.format()
                    {
                        // Same as the copy pipeline below, but without a render pass.
                        scope.copy_texture_to_texture(
                            result.as_image_copy(),
                            texture.texture.as_image_copy(),
                            result.size(),
                        );
                    } else {
                        run_copy_pipeline(
                            &self.descriptors,
                            texture.texture.format(),
                            &texture.texture.create_view(&Default::default()),
                            target.color_view(),
                            target.globals(),
                            target.color_texture().sample_count(),
                            &mut scope.scope("Copy filtered to CAB"),
                        );
                    }
                }
                // Periodically flush GPU work to prevent OOM when many unbatched cache
                // entries accumulate; batched atlas/filter entries share a single pass.
                if !atlased || !filter_batched[index] {
                    self.active_frame.maybe_flush(&self.descriptors);
                }
            }

            if !filtered_together.is_empty() {
                let skipped = crate::filters::apply_together(
                    &mut filtered_together,
                    &self.descriptors,
                    &self.meshes,
                    &mut self
                        .profiler
                        .scope("Filter together", &mut self.active_frame.command_encoder),
                    &mut self.active_frame.staging_belt,
                    &self.active_frame.dynamic_transforms,
                    &mut self.offscreen_texture_pool,
                    self.surface.quality(),
                    crate::surface::layer_atlas::max_filter_atlas_size()
                        .min(self.descriptors.limits.max_texture_dimension_2d),
                    &mut self.filter_atlas,
                );
                for &index in &skipped {
                    let entry = &mut filtered_together[index];
                    *as_texture(&entry.handle).last_cab_key.borrow_mut() = None;
                    if let Some(commands) = entry.direct_commands.take() {
                        let surface = Surface::new(
                            &self.descriptors,
                            self.surface.quality(),
                            entry.texture.width(),
                            entry.texture.height(),
                            wgpu::TextureFormat::Rgba8Unorm,
                        );
                        let filter_atlas_ref =
                            (!self.filter_atlas.slots.is_empty()).then_some(&self.filter_atlas);
                        surface.draw_commands(
                            RenderTargetMode::ExistingWithColor(
                                entry.texture.clone(),
                                wgpu::Color::TRANSPARENT,
                            ),
                            &self.descriptors,
                            &self.meshes,
                            commands,
                            &mut self.active_frame.staging_belt,
                            &self.active_frame.dynamic_transforms,
                            &mut self
                                .profiler
                                .scope("Draw to CAB", &mut self.active_frame.command_encoder),
                            LayerRef::None,
                            &mut self.offscreen_texture_pool,
                            filter_atlas_ref,
                            None,
                        );
                    }
                    filter_texture_in_place(
                        &self.descriptors,
                        &mut self.active_frame.command_encoder,
                        &mut self.offscreen_texture_pool,
                        &mut self.active_frame.staging_belt,
                        &entry.texture,
                        &entry.filters,
                    );
                }
                for entry in &filtered_together {
                    if entry.volatility == CacheVolatility::Persistent {
                        as_texture(&entry.handle).bump_generation_keeping_key();
                    }
                }
                for (ft_idx, key) in pending_persistent_keys {
                    if !skipped.contains(&ft_idx) {
                        *as_texture(&filtered_together[ft_idx].handle)
                            .last_cab_key
                            .borrow_mut() = Some(key);
                    }
                }
                for (ft_idx, key) in pending_slot_keys {
                    if !skipped.contains(&ft_idx) {
                        self.filter_atlas
                            .set_slot_key(filtered_together[ft_idx].handle_ptr, key);
                    }
                }
                for (ft_idx, key, alloc, is_subframe) in pending_promotions {
                    if !skipped.contains(&ft_idx) {
                        self.cab_cache.insert_entry(key, alloc, is_subframe);
                        crate::perf_counters::count_named("cab-lru/promote");
                    } else {
                        self.cab_cache.cancel_allocation(alloc);
                    }
                }
                for (leader_ft_idx, handle, handle_ptr, tex, used_size, volatility, key) in
                    dedup_followers
                {
                    let leader = &filtered_together[leader_ft_idx];
                    let leader_skipped = skipped.contains(&leader_ft_idx);
                    if volatility != CacheVolatility::Persistent && !leader_skipped {
                        let aliased = self.filter_atlas.alias_slot(
                            leader.handle_ptr,
                            handle,
                            handle_ptr,
                            volatility,
                        );
                        debug_assert!(aliased);
                    } else {
                        let copied_from_atlas = !leader_skipped
                            && leader.volatility != CacheVolatility::Persistent
                            && self.filter_atlas.copy_slot_to_texture(
                                &mut self.active_frame.command_encoder,
                                leader.handle_ptr,
                                &tex,
                                used_size,
                            );
                        if !copied_from_atlas {
                            self.active_frame.command_encoder.copy_texture_to_texture(
                                leader.texture.as_image_copy(),
                                tex.as_image_copy(),
                                wgpu::Extent3d {
                                    width: used_size.0,
                                    height: used_size.1,
                                    depth_or_array_layers: 1,
                                },
                            );
                        }
                        if volatility == CacheVolatility::Persistent {
                            let tex_obj = as_texture(&handle);
                            tex_obj.bump_generation_keeping_key();
                            *tex_obj.last_cab_key.borrow_mut() = (!leader_skipped).then_some(key);
                        }
                    }
                }
            }
            for (filters, members) in filter_groups {
                let textures: Vec<_> = members.iter().map(|&i| cache_textures[i].clone()).collect();
                let skipped = crate::surface::layer_atlas::apply_filters_batched(
                    &filters,
                    &textures,
                    &self.descriptors,
                    &mut self.active_frame.command_encoder,
                    &mut self.offscreen_texture_pool,
                    &mut self.active_frame.staging_belt,
                );
                for index in skipped {
                    filter_texture_in_place(
                        &self.descriptors,
                        &mut self.active_frame.command_encoder,
                        &mut self.offscreen_texture_pool,
                        &mut self.active_frame.staging_belt,
                        &textures[index],
                        &filters,
                    );
                }
            }
        }

        let clear_wgpu = wgpu::Color {
            r: f64::from(clear.r) / 255.0,
            g: f64::from(clear.g) / 255.0,
            b: f64::from(clear.b) / 255.0,
            a: f64::from(clear.a) / 255.0,
        };
        let mut scope = self
            .profiler
            .scope("Frame commands", &mut self.active_frame.command_encoder);
        let filter_atlas_ref =
            (!self.filter_atlas.slots.is_empty()).then_some(&self.filter_atlas);
        if let Some(acquirer) = self.direct_output_acquirer.take() {
            if let DirectOutputAcquire::Ready(swap_tex) = acquirer() {
                let swap_w = swap_tex.width();
                let swap_h = swap_tex.height();
                let swap_fmt = swap_tex.format();
                let quality = self.surface.quality();
                let fetch_blends = self.descriptors.shaders.fetch_blend_shaders.is_some()
                    && ruffle_render::perf_flags::enabled(
                        ruffle_render::perf_flags::PerfFlag::FetchBlends,
                    );
                let has_stage_blends = commands.commands.iter().any(|cmd| match cmd {
                    ruffle_render::commands::Command::Blend(_, mode)
                    | ruffle_render::commands::Command::BlendBounded(_, mode, _) => match mode {
                        ruffle_render::commands::RenderBlendMode::Shader(_) => true,
                        ruffle_render::commands::RenderBlendMode::Builtin(b) if !fetch_blends => {
                            matches!(
                                crate::blend::BlendType::from(
                                    ruffle_render::commands::RenderBlendMode::Builtin(*b)
                                ),
                                crate::blend::BlendType::Complex(_)
                            )
                        }
                        _ => false,
                    },
                    _ => false,
                });
                if !has_stage_blends && clear.a == 255 {
                    let mut commands = commands;
                    let has_letterbox = self
                        .visible_region
                        .is_some_and(|(x, y, w, h)| x != 0 || y != 0 || w != swap_w || h != swap_h);
                    let (scissor, pass_clear) = if has_letterbox {
                        let (rx, ry, rw, rh) = self.visible_region.expect("checked above");
                        let sx = rx.max(0) as u32;
                        let sy = ry.max(0) as u32;
                        let sw = rw.min(swap_w.saturating_sub(sx));
                        let sh = rh.min(swap_h.saturating_sub(sy));
                        if clear != Color::BLACK && sw > 0 && sh > 0 {
                            commands.commands.insert(
                                0,
                                ruffle_render::commands::Command::DrawRect {
                                    color: clear,
                                    matrix: ruffle_render::matrix::Matrix::translate(
                                        swf::Twips::from_pixels_i32(rx),
                                        swf::Twips::from_pixels_i32(ry),
                                    ) * ruffle_render::matrix::Matrix::scale(
                                        rw as f32, rh as f32,
                                    ),
                                },
                            );
                        }
                        (Some((sx, sy, sw, sh)), wgpu::Color::BLACK)
                    } else {
                        (None, clear_wgpu)
                    };
                    if !matches!(
                        &self.direct_surface,
                        Some((size, fmt, q, _))
                            if *size == (swap_w, swap_h) && *fmt == swap_fmt && *q == quality
                    ) {
                        self.direct_surface = Some((
                            (swap_w, swap_h),
                            swap_fmt,
                            quality,
                            Surface::new(&self.descriptors, quality, swap_w, swap_h, swap_fmt),
                        ));
                    }
                    let direct_surface = &mut self.direct_surface.as_mut().expect("just created").3;
                    direct_surface.set_scissor(scissor);
                    let target = direct_surface.draw_commands(
                        RenderTargetMode::ExistingWithColor(swap_tex, pass_clear),
                        &self.descriptors,
                        &self.meshes,
                        commands,
                        &mut self.active_frame.staging_belt,
                        &self.active_frame.dynamic_transforms,
                        &mut scope,
                        LayerRef::None,
                        &mut self.texture_pool,
                        filter_atlas_ref,
                        Some(&mut self.blend_cache),
                    );
                    target.ensure_cleared(&mut scope);
                } else {
                    // Rare fallback (PixelBender shader blend on stage or transparent stage):
                    // render to a pooled surface and copy into the swapchain without requiring
                    // `TEXTURE_BINDING` / `COPY_SRC` on the swapchain texture.
                    let surface = match self.visible_region {
                        Some(region) => {
                            if !matches!(&self.region_surface, Some((r, s)) if *r == region && s.quality() == quality)
                            {
                                let (x, y, width, height) = region;
                                self.region_surface = Some((
                                    region,
                                    Surface::new(
                                        &self.descriptors,
                                        quality,
                                        width,
                                        height,
                                        self.target.format(),
                                    )
                                    .with_origin((x, y)),
                                ));
                            }
                            &self.region_surface.as_ref().expect("just created").1
                        }
                        None => &self.surface,
                    };
                    let target = surface.draw_commands(
                        RenderTargetMode::FreshWithColor(clear_wgpu),
                        &self.descriptors,
                        &self.meshes,
                        commands,
                        &mut self.active_frame.staging_belt,
                        &self.active_frame.dynamic_transforms,
                        &mut scope,
                        LayerRef::None,
                        &mut self.texture_pool,
                        filter_atlas_ref,
                        Some(&mut self.blend_cache),
                    );
                    let swap_view = swap_tex.create_view(&Default::default());
                    crate::utils::run_copy_pipeline_into_region(
                        &self.descriptors,
                        swap_fmt,
                        &swap_view,
                        target.color_view(),
                        target.globals(),
                        1,
                        &mut scope,
                        self.visible_region,
                    );
                }
            }
        } else {
            let surface = match self.visible_region {
                Some(region) => {
                    let quality = self.surface.quality();
                    if !matches!(&self.region_surface, Some((r, s)) if *r == region && s.quality() == quality)
                    {
                        let (x, y, width, height) = region;
                        self.region_surface = Some((
                            region,
                            Surface::new(
                                &self.descriptors,
                                quality,
                                width,
                                height,
                                self.target.format(),
                            )
                            .with_origin((x, y)),
                        ));
                    }
                    &self.region_surface.as_ref().expect("just created").1
                }
                None => &self.surface,
            };
            // Drawn straight into the frame's texture when it's the size drawn (resolved into
            // it, when multisampled): a copy of the whole frame fewer.
            let direct = frame_output.texture().filter(|texture| {
                ruffle_render::perf_flags::enabled(ruffle_render::perf_flags::PerfFlag::DirectFrame)
                    && texture.width() == surface.size().width
                    && texture.height() == surface.size().height
            });
            if let Some(texture) = direct {
                surface.draw_commands(
                    RenderTargetMode::ExistingWithColor(texture.clone(), clear_wgpu),
                    &self.descriptors,
                    &self.meshes,
                    commands,
                    &mut self.active_frame.staging_belt,
                    &self.active_frame.dynamic_transforms,
                    &mut scope,
                    LayerRef::None,
                    &mut self.texture_pool,
                    filter_atlas_ref,
                    Some(&mut self.blend_cache),
                );
            } else {
                surface.draw_commands_and_copy_to(
                    frame_output.view(),
                    RenderTargetMode::FreshWithColor(clear_wgpu),
                    &self.descriptors,
                    &mut self.active_frame.staging_belt,
                    &self.active_frame.dynamic_transforms,
                    &mut scope,
                    &self.meshes,
                    commands,
                    LayerRef::None,
                    &mut self.texture_pool,
                    // A frame that holds just the region is copied into whole.
                    if self.frame_is_region {
                        None
                    } else {
                        self.visible_region
                    },
                    filter_atlas_ref,
                    Some(&mut self.blend_cache),
                );
            }
        }
        drop(scope);
        self.filter_atlas.end_frame();
        self.profiler
            .resolve_queries(&mut self.active_frame.command_encoder);
        self.active_frame.staging_belt.finish();

        if self.defer_submit
            && self.target.can_submit_asynchronously()
            && ruffle_render::perf_flags::enabled(ruffle_render::perf_flags::PerfFlag::AsyncSubmit)
        {
            drop(frame_output);
            self.deferred_frame_pending = true;
        } else {
            self.active_frame
                .submit_for_target_async(&self.descriptors, &self.target, frame_output);
        }
        self.offscreen_texture_pool.end_frame();
        self.texture_pool.end_frame();
        crate::perf_counters::end_frame();
        self.profiler
            .end_frame()
            .expect("Frame should end successfully");
        let timestamp_period = self.descriptors.queue.get_timestamp_period();
        self.profiler.process_finished_frame(timestamp_period);
    }

    #[instrument(level = "debug", skip_all)]
    fn register_bitmap(&mut self, bitmap: Bitmap<'_>) -> Result<BitmapHandle, BitmapError> {
        let mut bitmap = bitmap.to_rgba();

        self.clamp_bitmap(&mut bitmap);

        let extent = wgpu::Extent3d {
            width: bitmap.width(),
            height: bitmap.height(),
            depth_or_array_layers: 1,
        };

        let texture_label = create_debug_label!("Bitmap");
        let texture = self
            .descriptors
            .device
            .create_texture(&wgpu::TextureDescriptor {
                label: texture_label.as_deref(),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                view_formats: &[wgpu::TextureFormat::Rgba8Unorm],
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_SRC,
            });

        self.descriptors.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: Default::default(),
                aspect: wgpu::TextureAspect::All,
            },
            bitmap.data(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * extent.width),
                rows_per_image: None,
            },
            extent,
        );

        let handle = BitmapHandle(Arc::new(Texture::new(texture)));

        Ok(handle)
    }

    #[instrument(level = "debug", skip_all)]
    fn update_texture(
        &mut self,
        handle: &BitmapHandle,
        bitmap: Bitmap<'_>,
        mut region: PixelRegion,
    ) -> Result<(), BitmapError> {
        if region.width() == 0 || region.height() == 0 {
            // Nothing to do. It's important to bail out now, as the
            // write_texture call panics when the source buffer is of zero size.
            return Ok(());
        }

        let texture = as_texture(handle);
        texture.bump_generation();

        let mut bitmap = bitmap.to_rgba();
        if self.clamp_bitmap(&mut bitmap) {
            // If we're updating a resized texture, just redo the whole thing.
            // We can't trivially map pixel regions as we use a filter to resize.
            region = PixelRegion::for_whole_size(bitmap.width(), bitmap.height());
        }

        let extent = wgpu::Extent3d {
            width: region.width(),
            height: region.height(),
            depth_or_array_layers: 1,
        };

        self.active_frame.submit_direct(&self.descriptors);
        self.descriptors.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture.texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: region.x_min,
                    y: region.y_min,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            &bitmap.data()[(region.y_min * texture.texture.width() * 4) as usize
                ..(region.y_max * texture.texture.width() * 4) as usize],
            wgpu::TexelCopyBufferLayout {
                offset: (region.x_min * 4) as wgpu::BufferAddress,
                bytes_per_row: Some(4 * texture.texture.width()),
                rows_per_image: None,
            },
            extent,
        );

        Ok(())
    }

    #[instrument(level = "debug", skip_all)]
    fn render_offscreen(
        &mut self,
        handle: BitmapHandle,
        commands: CommandList,
        quality: StageQuality,
        bounds: PixelRegion,
    ) -> Option<Box<dyn SyncHandle>> {
        self.finish_pending_shapes();
        let texture = as_texture(&handle);
        texture.bump_generation();

        let extent = wgpu::Extent3d {
            width: texture.texture.width(),
            height: texture.texture.height(),
            depth_or_array_layers: 1,
        };

        let mut target = TextureTarget {
            size: extent,
            texture: texture.texture.clone(),
            format: wgpu::TextureFormat::Rgba8Unorm,
            buffer: None,
        };

        let frame_output = target
            .get_next_texture()
            .expect("TextureTargetFrame.get_next_texture is infallible");

        let surface = Surface::new(
            &self.descriptors,
            quality,
            texture.texture.width(),
            texture.texture.height(),
            wgpu::TextureFormat::Rgba8Unorm,
        );
        let filter_atlas_ref =
            (!self.filter_atlas.slots.is_empty()).then_some(&self.filter_atlas);
        surface.draw_commands_and_copy_to(
            frame_output.view(),
            RenderTargetMode::FreshWithTexture(target.get_texture()),
            &self.descriptors,
            &mut self.active_frame.staging_belt,
            &self.active_frame.dynamic_transforms,
            &mut self
                .profiler
                .scope("Offscreen commands", &mut self.active_frame.command_encoder),
            &self.meshes,
            commands,
            LayerRef::Current,
            &mut self.offscreen_texture_pool,
            None,
            filter_atlas_ref,
            None,
        );

        self.active_frame.maybe_flush(&self.descriptors);
        Some(self.make_queue_sync_handle(target, None, handle, bounds))
    }

    fn is_filter_supported(&self, filter: &Filter) -> bool {
        matches!(
            filter,
            Filter::BlurFilter(_)
                | Filter::GlowFilter(_)
                | Filter::DropShadowFilter(_)
                | Filter::ColorMatrixFilter(_)
                | Filter::ShaderFilter(_)
                | Filter::BevelFilter(_)
                | Filter::DisplacementMapFilter(_)
        )
    }

    fn is_offscreen_supported(&self) -> bool {
        true
    }

    fn apply_filter(
        &mut self,
        source: BitmapHandle,
        source_point: (u32, u32),
        source_size: (u32, u32),
        destination: BitmapHandle,
        dest_point: (i32, i32),
        filter: Filter,
    ) -> Option<Box<dyn SyncHandle>> {
        let source_texture = as_texture(&source);
        let dest_texture = as_texture(&destination);
        dest_texture.bump_generation();

        let copy_area = PixelRegion::for_whole_size(
            dest_texture.texture.width(),
            dest_texture.texture.height(),
        );

        let target = TextureTarget {
            size: wgpu::Extent3d {
                width: dest_texture.texture.width(),
                height: dest_texture.texture.height(),
                depth_or_array_layers: 1,
            },
            texture: dest_texture.texture.clone(),
            format: wgpu::TextureFormat::Rgba8Unorm,
            buffer: None,
        };

        let applied_filter = self.descriptors.filters.apply(
            &self.descriptors,
            &mut self.active_frame.command_encoder,
            &mut self.offscreen_texture_pool,
            &mut self.active_frame.staging_belt,
            FilterSource {
                texture: &source_texture.texture,
                point: source_point,
                size: source_size,
            },
            filter,
        );

        let (dest_x, dest_y) = dest_point;

        let src_offset_x = dest_x.min(0).unsigned_abs();
        let src_offset_y = dest_y.min(0).unsigned_abs();

        let final_dest_x = dest_x.max(0) as u32;
        let final_dest_y = dest_y.max(0) as u32;

        let available_width = applied_filter.width().saturating_sub(src_offset_x);
        let available_height = applied_filter.height().saturating_sub(src_offset_y);
        let dest_available_width = dest_texture.texture.width().saturating_sub(final_dest_x);
        let dest_available_height = dest_texture.texture.height().saturating_sub(final_dest_y);

        let copy_width = available_width.min(dest_available_width);
        let copy_height = available_height.min(dest_available_height);

        if copy_width == 0 || copy_height == 0 {
            return None;
        }

        self.active_frame.command_encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: applied_filter.color_texture(),
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: src_offset_x,
                    y: src_offset_y,
                    z: 0,
                },
                aspect: Default::default(),
            },
            wgpu::TexelCopyTextureInfo {
                texture: &dest_texture.texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: final_dest_x,
                    y: final_dest_y,
                    z: 0,
                },
                aspect: Default::default(),
            },
            wgpu::Extent3d {
                width: copy_width,
                height: copy_height,
                depth_or_array_layers: 1,
            },
        );

        self.active_frame.maybe_flush(&self.descriptors);
        Some(self.make_queue_sync_handle(target, None, destination, copy_area))
    }

    fn compile_pixelbender_shader(
        &mut self,
        shader: PixelBenderShader,
    ) -> Result<PixelBenderShaderHandle, BitmapError> {
        self.compile_pixelbender_shader_impl(shader)
    }

    fn run_pixelbender_shader(
        &mut self,
        shader: PixelBenderShaderHandle,
        arguments: &[PixelBenderShaderArgument],
        target: &PixelBenderTarget,
    ) -> Result<PixelBenderOutput, BitmapError> {
        let output_channels = shader
            .0
            .parsed_shader()
            .output_channels()
            .expect("No output parameter");
        let has_padding = output_channels == 3;

        let texture_format =
            crate::pixel_bender::temporary_texture_format_for_channels(output_channels as u32);

        let target_handle = match target {
            PixelBenderTarget::Bitmap(handle) => handle.clone(),
            PixelBenderTarget::Bytes { width, height } => {
                let extent = wgpu::Extent3d {
                    width: *width,
                    height: *height,
                    depth_or_array_layers: 1,
                };
                // FIXME - cache this texture somehow. We might also want to consider using
                // a compute shader
                let texture_label = create_debug_label!("Temporary pixelbender output texture");
                let texture = self
                    .descriptors
                    .device
                    .create_texture(&wgpu::TextureDescriptor {
                        label: texture_label.as_deref(),
                        size: extent,
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: texture_format,
                        view_formats: &[texture_format],
                        usage: wgpu::TextureUsages::TEXTURE_BINDING
                            | wgpu::TextureUsages::COPY_DST
                            | wgpu::TextureUsages::RENDER_ATTACHMENT
                            | wgpu::TextureUsages::COPY_SRC,
                    });
                BitmapHandle(Arc::new(Texture::new(texture)))
            }
        };

        let target_texture = as_texture(&target_handle);
        target_texture.bump_generation();

        let extent = wgpu::Extent3d {
            width: target_texture.texture.width(),
            height: target_texture.texture.height(),
            depth_or_array_layers: 1,
        };

        let copy_dimensions = BufferDimensions::new(
            target_texture.texture.width() as usize,
            target_texture.texture.height() as usize,
            target_texture.texture.format(),
        );
        let buffer_info = Some(TextureBufferInfo {
            buffer: MaybeOwnedBuffer::Borrowed(
                self.offscreen_buffer_pool
                    .take(&self.descriptors, copy_dimensions.clone()),
                copy_dimensions,
            ),
            copy_area: PixelRegion::for_whole_size(
                target_texture.texture.width(),
                target_texture.texture.height(),
            ),
        });

        let mut texture_target = TextureTarget {
            size: extent,
            texture: target_texture.texture.clone(),
            format: target_texture.texture.format(),
            buffer: buffer_info,
        };

        let frame_output = texture_target
            .get_next_texture()
            .expect("TextureTargetFrame.get_next_texture is infallible");

        run_pixelbender_shader_impl(
            &self.descriptors,
            shader,
            ShaderMode::ShaderJob,
            arguments,
            &target_texture.texture,
            &mut self.active_frame.command_encoder,
            Some(wgpu::RenderPassColorAttachment {
                view: frame_output.view(),
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            }),
            1,
            // When running a standalone shader, we always process the entire image
            &FilterSource::for_entire_texture(&target_texture.texture),
        )?;

        let index = Some(self.active_frame.submit_for_target(
            &self.descriptors,
            &texture_target,
            frame_output,
        ));

        let sync_handle = self.make_queue_sync_handle(
            texture_target,
            index,
            target_handle,
            PixelRegion::for_whole_size(extent.width, extent.height),
        );

        match target {
            PixelBenderTarget::Bitmap(_) => Ok(PixelBenderOutput::Bitmap(sync_handle)),
            PixelBenderTarget::Bytes { width, .. } => {
                let mut output = None;
                self.resolve_sync_handle(
                    sync_handle,
                    Box::new(|raw_pixels, buffer_width| {
                        let width = *width as usize;

                        if buffer_width as usize
                            != width * output_channels * std::mem::size_of::<f32>()
                        {
                            let mut new_pixels = Vec::new();
                            for row in raw_pixels.chunks(buffer_width as usize) {
                                let actual_row = &row[0..(width * std::mem::size_of::<[f32; 4]>())];

                                for pixel in actual_row
                                    .as_chunks::<{ std::mem::size_of::<[f32; 4]>() }>()
                                    .0
                                {
                                    if has_padding {
                                        // Take the first three channels
                                        new_pixels.extend_from_slice(
                                            &pixel[0..(3 * std::mem::size_of::<f32>())],
                                        );
                                    } else {
                                        // Copy the pixel as-is
                                        new_pixels.extend_from_slice(pixel);
                                    }
                                }
                            }
                            output = Some(new_pixels);
                        } else {
                            output = Some(raw_pixels.to_vec());
                        };
                    }),
                )?;
                Ok(PixelBenderOutput::Bytes(output.unwrap()))
            }
        }
    }

    fn create_empty_texture(
        &mut self,
        width: NonZeroU32,
        height: NonZeroU32,
    ) -> Result<BitmapHandle, BitmapError> {
        let width = width.get();
        let height = height.get();

        if width > self.descriptors.limits.max_texture_dimension_2d
            || height > self.descriptors.limits.max_texture_dimension_2d
        {
            return Err(BitmapError::TooLarge);
        }

        let extent = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let texture_label = create_debug_label!("Bitmap");
        let texture = self
            .descriptors
            .device
            .create_texture(&wgpu::TextureDescriptor {
                label: texture_label.as_deref(),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                view_formats: &[wgpu::TextureFormat::Rgba8Unorm],
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_SRC,
            });
        Ok(BitmapHandle(Arc::new(Texture::new(texture))))
    }

    fn resolve_sync_handle(
        &mut self,
        handle: Box<dyn SyncHandle>,
        with_rgba: RgbaBufRead,
    ) -> Result<(), ruffle_render::error::Error> {
        let handle = Box::<dyn Any>::downcast::<QueueSyncHandle>(handle).unwrap();
        handle.capture(with_rgba, &mut self.active_frame);
        Ok(())
    }
}

pub async fn request_adapter_and_device(
    backend: wgpu::Backends,
    instance: &wgpu::Instance,
    surface: Option<&wgpu::Surface<'static>>,
    power_preference: wgpu::PowerPreference,
) -> Result<(wgpu::Adapter, wgpu::Device, wgpu::Queue), Error> {
    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference,
        compatible_surface: surface,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }).await
        .map_err(|_e| {
            let names = get_backend_names(backend);
            if names.is_empty() {
                "Ruffle requires hardware acceleration, but no compatible graphics device was found (no backend provided?)".to_string()
            } else if cfg!(target_vendor = "apple") {
                "Ruffle does not support OpenGL on macOS/iOS.".to_string()
            } else {
                format!("Ruffle requires hardware acceleration, but no compatible graphics device was found supporting {}", format_list(&names, "or"))
            }
        })?;

    let (device, queue) = request_device(&adapter).await?;
    Ok((adapter, device, queue))
}

// We try to request the highest limits we can get away with
async fn request_device(
    adapter: &wgpu::Adapter,
) -> Result<(wgpu::Device, wgpu::Queue), wgpu::RequestDeviceError> {
    // We start off with the lowest limits we actually need - basically GL-ES 3.0
    let mut limits = wgpu::Limits::downlevel_webgl2_defaults();
    // Then we increase parts of it to the maximum supported by the adapter, to take advantage of
    // more powerful hardware or capabilities
    limits = limits.using_resolution(adapter.limits());
    limits = limits.using_alignment(adapter.limits());
    limits.max_uniform_buffer_binding_size = adapter.limits().max_uniform_buffer_binding_size;
    limits.max_inter_stage_shader_variables = adapter.limits().max_inter_stage_shader_variables;
    // This will be a default limit in a future wgpu version (down from 8).
    // It's required for some WebGL devices to be supported.
    limits.max_color_attachments = 4;

    let mut features = Default::default();

    let optional_features = wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
        | wgpu::Features::TEXTURE_COMPRESSION_BC
        | wgpu::Features::FLOAT32_FILTERABLE
        | GpuProfiler::ALL_WGPU_TIMER_FEATURES;

    features |= optional_features & adapter.features();
    if crate::shaders::supports_fetch_blends(adapter) {
        features |= wgpu::Features::PASSTHROUGH_SHADERS;
    }

    adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: features,
            required_limits: limits,
            memory_hints: Default::default(),
            trace: wgpu::Trace::Off,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
        })
        .await
}

/// Determines how we choose our frame buffer
#[derive(Clone)]
pub enum RenderTargetMode {
    // Construct a new frame buffer, clearng it with the provided color.
    // This is used when rendering to the actual display,
    // or when applying a filter. In both cases, we have a fixed background color,
    // and don't need to blend with anything else
    FreshWithColor(wgpu::Color),
    // Construct a new frame buffer, cleared with an existing texture.
    // we will blend with the previous contents of the texture.
    // This is used in `render_offscreen`, as we need to blend with the previous
    // contents of our `BitmapData` texture
    FreshWithTexture(wgpu::Texture),
    // Use the provided texture as our frame buffer, and clear it with the given color.
    ExistingWithColor(wgpu::Texture, wgpu::Color),
}

impl RenderTargetMode {
    pub fn color(&self) -> Option<wgpu::Color> {
        match self {
            RenderTargetMode::FreshWithColor(color) => Some(*color),
            RenderTargetMode::FreshWithTexture(_) => None,
            RenderTargetMode::ExistingWithColor(_, color) => Some(*color),
        }
    }
}

/// Shows the next frame presented on this thread at the given time (on Metal, see the
/// vendored wgpu-core's `ruffle_patch::set_present_time`), which paces frames evenly on the
/// display's refresh grid.
pub fn set_present_time(time: Option<std::time::Instant>) {
    #[cfg(not(target_family = "wasm"))]
    wgpu_core::ruffle_patch::set_present_time(time);
    #[cfg(target_family = "wasm")]
    let _ = time;
}

/// Measures how busy the GPU is, for `gpu_busy_fraction` (on Metal; it adds a completion
/// handler to each command buffer).
pub fn set_gpu_timing(enabled: bool) {
    #[cfg(not(target_family = "wasm"))]
    wgpu_core::ruffle_patch::set_gpu_timing(enabled);
    #[cfg(target_family = "wasm")]
    let _ = enabled;
}

/// The fraction of the last measured second that the GPU spent on Ruffle's work, while
/// `set_gpu_timing` is enabled (`None` where it can't be measured).
pub fn gpu_busy_fraction() -> Option<f64> {
    #[cfg(not(target_family = "wasm"))]
    return wgpu_core::ruffle_patch::gpu_busy_fraction();
    #[cfg(target_family = "wasm")]
    None
}

/// Passes Ruffle's perf flags on to the switches of the vendored wgpu-core's patches.
fn apply_wgpu_core_switches() {
    #[cfg(not(target_family = "wasm"))]
    wgpu_core::ruffle_patch::set_share_command_buffers(ruffle_render::perf_flags::enabled(
        ruffle_render::perf_flags::PerfFlag::ShareCommandBuffers,
    ));
}

pub struct ActiveFrame {
    pub staging_belt: wgpu::util::StagingBelt,
    pub command_encoder: wgpu::CommandEncoder,
    pub dynamic_transforms: DynamicTransforms,
    draws_since_flush: u32,
    /// Finishes and submits frames on another thread (see `submit_for_target_async`).
    submitter: Option<AsyncSubmitter>,
    /// How many jobs sent to `submitter` aren't done yet.
    jobs_in_flight: usize,
    /// The staging belt used by the commands being submitted by `submitter`.
    /// It's only recalled once they have been submitted; meanwhile, recording
    /// continues with another belt. (Recalling a belt while some of its chunks
    /// are used by unsubmitted commands makes the submission fail.)
    in_flight_belt: Option<wgpu::util::StagingBelt>,
    spare_belt: Option<wgpu::util::StagingBelt>,
}

/// A thread that finishes command encoders and submits them.
///
/// Finishing an encoder (where wgpu actually encodes the commands for the GPU)
/// and submitting it is the most expensive part of rendering a complex frame.
/// Doing it on another thread lets the next frame's scripts run meanwhile.
struct AsyncSubmitter {
    jobs: std::sync::mpsc::Sender<SubmitJob>,
    done: std::sync::mpsc::Receiver<()>,
}

/// Work for the submit thread, which does it in order.
enum SubmitJob {
    /// Finish the encoder, submit any preceding buffers + it, and optionally run a
    /// post-submit callback (e.g. `set_present_time` + `queue.present`).
    Frame {
        encoder: wgpu::CommandEncoder,
        extra_before: Vec<wgpu::CommandBuffer>,
        queue: wgpu::Queue,
        after: Option<Box<dyn FnOnce() + Send>>,
    },
    /// Anything that must happen after the frames before it were submitted
    /// (see `WgpuRenderBackend::run_after_submissions`).
    Task(Box<dyn FnOnce() + Send>),
}

impl AsyncSubmitter {
    fn new() -> Self {
        let (jobs, job_receiver) = std::sync::mpsc::channel::<SubmitJob>();
        let (done_sender, done) = std::sync::mpsc::channel();
        std::thread::Builder::new()
            .name("ruffle-wgpu-submit".to_string())
            .spawn(move || {
                // On Apple Silicon, default-QoS threads are often scheduled on E-cores
                // and preempted by background work, turning ~2ms `encoder.finish()` into
                // 6-14ms stalls that block the main thread. Request USER_INTERACTIVE QoS
                // so submission stays on P-cores alongside the main render loop.
                #[cfg(target_os = "macos")]
                {
                    const QOS_CLASS_USER_INTERACTIVE: u32 = 0x21;
                    unsafe extern "C" {
                        fn pthread_set_qos_class_self_np(
                            qos_class: u32,
                            relative_priority: i32,
                        ) -> i32;
                    }
                    unsafe {
                        pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
                    }
                }
                for job in job_receiver {
                    match job {
                        SubmitJob::Frame {
                            encoder,
                            mut extra_before,
                            queue,
                            after,
                        } => {
                            extra_before.push(encoder.finish());
                            queue.submit(extra_before);
                            if let Some(after) = after {
                                after();
                            }
                        }
                        SubmitJob::Task(task) => task(),
                    }
                    if done_sender.send(()).is_err() {
                        break;
                    }
                }
            })
            .expect("Submit thread should be created");
        Self { jobs, done }
    }
}

impl ActiveFrame {
    const MAX_DRAWS_PER_FLUSH: u32 = 100;

    pub fn new(descriptors: &Descriptors) -> Self {
        Self {
            command_encoder: descriptors
                .device
                .create_command_encoder(&Default::default()),
            staging_belt: wgpu::util::StagingBelt::new(descriptors.device.clone(), 262144),
            dynamic_transforms: DynamicTransforms::new(descriptors),
            draws_since_flush: 0,
            submitter: None,
            jobs_in_flight: 0,
            in_flight_belt: None,
            spare_belt: None,
        }
    }

    /// Waits until a frame being submitted on another thread has been submitted.
    /// Anything that must happen after the previous frame on the queue
    /// (such as further submissions or buffer writes) must call this first.
    pub fn wait_for_submission(&mut self) {
        if self.jobs_in_flight > 0 {
            let submitter = self
                .submitter
                .as_ref()
                .expect("jobs in flight imply a submitter");
            while self.jobs_in_flight > 0 {
                if submitter.done.recv().is_err() {
                    self.jobs_in_flight = 0;
                    break;
                }
                self.jobs_in_flight -= 1;
            }
            if let Some(mut belt) = self.in_flight_belt.take() {
                belt.recall();
                self.spare_belt = Some(belt);
            }
        }
    }

    /// Runs `task` once everything submitted so far is on the queue: on the
    /// submit thread if a frame is still being submitted there, otherwise now.
    pub fn run_after_submissions(&mut self, task: Box<dyn FnOnce() + Send>) {
        match &self.submitter {
            Some(submitter) if self.jobs_in_flight > 0 => {
                submitter
                    .jobs
                    .send(SubmitJob::Task(task))
                    .expect("Submit thread should be running");
                self.jobs_in_flight += 1;
            }
            _ => task(),
        }
    }

    pub fn submit_for_target<T: RenderTarget>(
        &mut self,
        descriptors: &Descriptors,
        target: &T,
        frame: T::Frame,
    ) -> SubmissionIndex {
        self.wait_for_submission();
        self.draws_since_flush = 0;
        apply_wgpu_core_switches();
        self.staging_belt.finish();
        self.dynamic_transforms.flush();
        let draw_encoder = std::mem::replace(
            &mut self.command_encoder,
            descriptors
                .device
                .create_command_encoder(&Default::default()),
        );
        let index = target.submit(
            &descriptors.device,
            &descriptors.queue,
            Some(draw_encoder.finish()),
            frame,
        );
        self.staging_belt.recall();
        index
    }

    /// Like `submit_for_target`, but if the target allows it, the frame is finished
    /// and submitted on another thread, and this returns without waiting for that.
    pub fn submit_for_target_async<T: RenderTarget>(
        &mut self,
        descriptors: &Descriptors,
        target: &T,
        frame: T::Frame,
    ) {
        if !target.can_submit_asynchronously()
            || !ruffle_render::perf_flags::enabled(ruffle_render::perf_flags::PerfFlag::AsyncSubmit)
        {
            self.submit_for_target(descriptors, target, frame);
            return;
        }
        drop(frame);
        self.submit_async(descriptors);
    }

    /// Finishes and submits the current commands on the submit thread.
    fn submit_async(&mut self, descriptors: &Descriptors) {
        self.submit_async_with_post(descriptors, Vec::new(), None);
    }

    fn submit_async_with_post(
        &mut self,
        descriptors: &Descriptors,
        extra_before: Vec<wgpu::CommandBuffer>,
        after: Option<Box<dyn FnOnce() + Send>>,
    ) {
        self.wait_for_submission();
        apply_wgpu_core_switches();
        self.draws_since_flush = 0;
        self.staging_belt.finish();
        self.dynamic_transforms.flush();
        let draw_encoder = std::mem::replace(
            &mut self.command_encoder,
            descriptors
                .device
                .create_command_encoder(&Default::default()),
        );
        // This belt is recalled once the submission is done; record with another one meanwhile.
        let next_belt = self
            .spare_belt
            .take()
            .unwrap_or_else(|| wgpu::util::StagingBelt::new(descriptors.device.clone(), 262144));
        self.in_flight_belt = Some(std::mem::replace(&mut self.staging_belt, next_belt));
        let submitter = self.submitter.get_or_insert_with(AsyncSubmitter::new);
        submitter
            .jobs
            .send(SubmitJob::Frame {
                encoder: draw_encoder,
                extra_before,
                queue: descriptors.queue.clone(),
                after,
            })
            .expect("Submit thread should be running");
        self.jobs_in_flight += 1;
    }

    pub fn submit_direct(&mut self, descriptors: &Descriptors) -> SubmissionIndex {
        self.wait_for_submission();
        apply_wgpu_core_switches();
        self.draws_since_flush = 0;
        self.staging_belt.finish();
        self.dynamic_transforms.flush();
        let draw_encoder = std::mem::replace(
            &mut self.command_encoder,
            descriptors
                .device
                .create_command_encoder(&Default::default()),
        );
        let index = descriptors.queue.submit(Some(draw_encoder.finish()));
        self.staging_belt.recall();
        index
    }

    pub fn maybe_flush(&mut self, descriptors: &Descriptors) {
        // [NA] This is kind of a hack.
        // If we do "too much" during one frame, the submission ends up being way too large and goes OutOfMemory.
        // What it is that we're OOMing on is likely buffers and temporary textures and such from render_offscreen
        // Hard to track that though... so let's just flush it out if we do more than X draws per frame
        self.draws_since_flush += 1;

        if self.draws_since_flush > Self::MAX_DRAWS_PER_FLUSH {
            crate::perf_counters::count(crate::perf_counters::Counter::Flush);
            if self.submitter.is_some()
                && ruffle_render::perf_flags::enabled(
                    ruffle_render::perf_flags::PerfFlag::AsyncSubmit,
                )
            {
                // Nothing waits for this submission, so it can happen in the background.
                self.submit_async(descriptors);
            } else {
                self.submit_direct(descriptors);
            }
        }
    }
}

/// Prints the filters of a level's cache entries, and how they're applied
/// (`RUFFLE_FILTER_DEBUG`), every 120 frames.
fn log_cache_filters(
    level: usize,
    members: &[usize],
    cache_entries: &[Option<BitmapCacheEntry>],
    cache_textures: &[wgpu::Texture],
    filter_groups: &[(Vec<Filter>, Vec<usize>)],
    filtered_together: &[usize],
) {
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
    static FRAME: AtomicU32 = AtomicU32::new(0);
    static PRINTING: AtomicBool = AtomicBool::new(false);
    if level == 0 {
        let frame = FRAME.fetch_add(1, Ordering::Relaxed);
        PRINTING.store(frame.is_multiple_of(120), Ordering::Relaxed);
    }
    if !PRINTING.load(Ordering::Relaxed) {
        return;
    }
    let describe = |filters: &[Filter]| {
        filters
            .iter()
            .map(|f| match f {
                Filter::BlurFilter(b) => format!(
                    "Blur({:.1}x{:.1} q{})",
                    b.blur_x.to_f32(),
                    b.blur_y.to_f32(),
                    b.num_passes()
                ),
                Filter::GlowFilter(g) => format!(
                    "Glow({:.1}x{:.1} q{} #{:02x}{:02x}{:02x}{:02x} s{:.2}{}{})",
                    g.blur_x.to_f32(),
                    g.blur_y.to_f32(),
                    g.num_passes(),
                    g.color.r,
                    g.color.g,
                    g.color.b,
                    g.color.a,
                    g.strength.to_f32(),
                    if g.is_inner() { " inner" } else { "" },
                    if g.is_knockout() { " knockout" } else { "" }
                ),
                other => other.name().to_string(),
            })
            .collect::<Vec<_>>()
            .join("+")
    };
    for (filters, group) in filter_groups {
        let sizes: Vec<_> = group
            .iter()
            .map(|&i| {
                format!(
                    "{}x{}",
                    cache_textures[i].width(),
                    cache_textures[i].height()
                )
            })
            .collect();
        eprintln!(
            "[filters] level {level}, group of {}: {} on {}",
            group.len(),
            describe(filters),
            sizes.join(" ")
        );
    }
    for &index in members {
        let Some(entry) = &cache_entries[index] else {
            continue;
        };
        if entry.filters.is_empty() || filter_groups.iter().any(|(_, g)| g.contains(&index)) {
            continue;
        }
        let texture = &cache_textures[index];
        let path = if filtered_together.contains(&index) {
            "together"
        } else if texture.width().max(texture.height())
            > crate::surface::layer_atlas::max_batched_filter_size()
        {
            "single (too large)"
        } else {
            "single"
        };
        eprintln!(
            "[filters] level {level}, {path}: {} on {}x{}",
            describe(&entry.filters),
            texture.width(),
            texture.height()
        );
    }
}

/// Collects the bitmaps drawn by `commands` (including nested ones).
fn collect_bitmaps(commands: &CommandList, bitmaps: &mut Vec<BitmapHandle>) {
    use ruffle_render::commands::Command;
    for command in &commands.commands {
        match command {
            Command::RenderBitmap { bitmap, .. } | Command::RenderStage3D { bitmap, .. } => {
                bitmaps.push(bitmap.clone())
            }
            Command::Blend(commands, _) | Command::BlendBounded(commands, _, _) => {
                collect_bitmaps(commands, bitmaps)
            }
            Command::RenderAlphaMask {
                maskee_commands,
                mask_commands,
            } => {
                collect_bitmaps(maskee_commands, bitmaps);
                collect_bitmaps(mask_commands, bitmaps);
            }
            _ => {}
        }
    }
}

/// Applies `filters` to `texture` (which must not be multisampled), in place.
fn filter_texture_in_place(
    descriptors: &Descriptors,
    encoder: &mut wgpu::CommandEncoder,
    texture_pool: &mut TexturePool,
    staging_belt: &mut wgpu::util::StagingBelt,
    texture: &wgpu::Texture,
    filters: &[Filter],
) {
    let mut target: Option<CommandTarget> = None;
    for filter in filters {
        let source = match &target {
            Some(target) => target.color_texture(),
            None => texture,
        };
        target = Some(descriptors.filters.apply(
            descriptors,
            encoder,
            texture_pool,
            staging_belt,
            FilterSource::for_entire_texture(source),
            filter.clone(),
        ));
    }
    if let Some(target) = target {
        let result = target.color_texture();
        encoder.copy_texture_to_texture(
            result.as_image_copy(),
            texture.as_image_copy(),
            result.size(),
        );
    }
}
