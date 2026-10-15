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
    BitmapCacheEntry, Context3D, Context3DProfile, PixelBenderOutput, PixelBenderTarget,
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
use std::cell::Cell;
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

pub struct WgpuRenderBackend<T: RenderTarget> {
    pub(crate) descriptors: Arc<Descriptors>,
    target: T,
    surface: Surface,
    /// See `RenderBackend::set_visible_region`.
    visible_region: Option<(i32, i32, u32, u32)>,
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
        })
    }

    fn register_shape_internal(
        &mut self,
        shape: DistilledShape,
        bitmap_source: &dyn BitmapSource,
        scale: f32,
    ) -> ShapeHandle {
        let pending =
            PendingShape::new(self.descriptors.clone(), shape, bitmap_source, self, scale);
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
    fn submit_frame(
        &mut self,
        clear: Color,
        commands: CommandList,
        cache_entries: Vec<BitmapCacheEntry>,
    ) {
        self.finish_pending_shapes();
        let mut cache_entries = cache_entries;
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

        // Cache entries whose contents are simple enough are all drawn together in an
        // atlas, and copied into their textures. They mustn't use other cache entries'
        // textures, as those may not be drawn yet.
        let is_atlased: Vec<bool> = cache_entries
            .iter()
            .map(|entry| {
                let texture = as_texture(&entry.handle);
                ruffle_render::perf_flags::enabled(ruffle_render::perf_flags::PerfFlag::LayerAtlas)
                    && entry.clear.a == 0
                    && entry.clear.r == 0
                    && entry.clear.g == 0
                    && entry.clear.b == 0
                    && crate::surface::layer_atlas::is_leaf(
                        &entry.commands,
                        texture.texture.width(),
                        texture.texture.height(),
                    ) && !entry.commands.commands.iter().any(|command| {
                    matches!(command, ruffle_render::commands::Command::RenderBitmap { bitmap, .. }
                            if cache_handles.contains(bitmap))
                })
            })
            .collect();
        let mut cache_entries: Vec<_> = cache_entries.into_iter().map(Some).collect();
        let atlased: Vec<_> = cache_entries
            .iter_mut()
            .zip(&is_atlased)
            .filter(|(_, atlased)| **atlased)
            .filter_map(|(entry, _)| {
                let entry = entry.as_mut()?;
                let texture = as_texture(&entry.handle);
                Some(crate::surface::layer_atlas::LeafLayer {
                    commands: std::mem::take(&mut entry.commands),
                    origin: (0, 0),
                    width: texture.texture.width(),
                    height: texture.texture.height(),
                    destination: texture.texture.clone(),
                })
            })
            .collect();
        if !atlased.is_empty() {
            crate::surface::layer_atlas::render_leaf_layers(
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
            );
        }

        for level in 0..=max_level {
            let members: Vec<usize> = (0..levels.len())
                .filter(|&index| levels[index] == level)
                .collect();
            let mut filter_groups: Vec<(Vec<Filter>, Vec<usize>)> = vec![];
            // Entries whose filters are applied together, whatever they are.
            let mut filtered_together: Vec<usize> = vec![];
            if batching {
                for &index in &members {
                    let entry = cache_entries[index]
                        .as_ref()
                        .expect("entries are only taken once");
                    let texture = &cache_textures[index];
                    if entry.filters.is_empty()
                        || texture.width() > crate::surface::layer_atlas::max_batched_filter_size()
                        || texture.height() > crate::surface::layer_atlas::max_batched_filter_size()
                    {
                        continue;
                    }
                    if together && entry.filters.iter().all(crate::filters::can_apply_together) {
                        filtered_together.push(index);
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
            for (_, group) in &filter_groups {
                for &index in group {
                    filter_batched[index] = true;
                }
            }
            for &index in &filtered_together {
                filter_batched[index] = true;
            }
            let filtered_together: Vec<(wgpu::Texture, Vec<Filter>)> = filtered_together
                .into_iter()
                .map(|index| {
                    (
                        cache_textures[index].clone(),
                        cache_entries[index]
                            .as_ref()
                            .expect("entries are only taken once")
                            .filters
                            .clone(),
                    )
                })
                .collect();

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
                // Periodically flush GPU work to prevent OOM when many cache entries
                // accumulate (e.g. when a large container's cacheAsBitmap is skipped
                // but its hundreds of children each have their own bitmap caches).
                self.active_frame.maybe_flush(&self.descriptors);
            }

            if !filtered_together.is_empty() {
                let skipped = crate::filters::apply_together(
                    &filtered_together,
                    &self.descriptors,
                    &mut self.active_frame.command_encoder,
                    &mut self.active_frame.staging_belt,
                    &mut self.offscreen_texture_pool,
                    crate::surface::layer_atlas::max_filter_atlas_size()
                        .min(self.descriptors.limits.max_texture_dimension_2d),
                );
                for index in skipped {
                    let (texture, filters) = &filtered_together[index];
                    filter_texture_in_place(
                        &self.descriptors,
                        &mut self.active_frame.command_encoder,
                        &mut self.offscreen_texture_pool,
                        &mut self.active_frame.staging_belt,
                        texture,
                        filters,
                    );
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
        surface.draw_commands_and_copy_to(
            frame_output.view(),
            RenderTargetMode::FreshWithColor(wgpu::Color {
                r: f64::from(clear.r) / 255.0,
                g: f64::from(clear.g) / 255.0,
                b: f64::from(clear.b) / 255.0,
                a: f64::from(clear.a) / 255.0,
            }),
            &self.descriptors,
            &mut self.active_frame.staging_belt,
            &self.active_frame.dynamic_transforms,
            &mut self
                .profiler
                .scope("Frame commands", &mut self.active_frame.command_encoder),
            &self.meshes,
            commands,
            LayerRef::None,
            &mut self.texture_pool,
            self.visible_region,
        );
        self.profiler
            .resolve_queries(&mut self.active_frame.command_encoder);
        self.active_frame.staging_belt.finish();

        self.active_frame
            .submit_for_target_async(&self.descriptors, &self.target, frame_output);
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

        let handle = BitmapHandle(Arc::new(Texture {
            texture,
            repeating_linear: Default::default(),
            repeating_nearest: Default::default(),
            clamped_linear: Default::default(),
            clamped_nearest: Default::default(),
            copy_count: Cell::new(0),
        }));

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
                BitmapHandle(Arc::new(Texture {
                    texture,
                    repeating_linear: Default::default(),
                    repeating_nearest: Default::default(),
                    clamped_linear: Default::default(),
                    clamped_nearest: Default::default(),
                    copy_count: Cell::new(0),
                }))
            }
        };

        let target_texture = as_texture(&target_handle);

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
        Ok(BitmapHandle(Arc::new(Texture {
            texture,
            repeating_linear: Default::default(),
            repeating_nearest: Default::default(),
            clamped_linear: Default::default(),
            clamped_nearest: Default::default(),
            copy_count: Cell::new(0),
        })))
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

/// Passes Ruffle's perf flags on to the switches of the vendored wgpu-core's patches.
/// Presents frames so that each stays on screen for at least `seconds` (on Metal, see the
/// vendored wgpu-core's `ruffle_patch::set_present_min_duration`), which paces them evenly
/// on the display's refresh grid.
pub fn set_present_min_duration(seconds: Option<f64>) {
    #[cfg(not(target_family = "wasm"))]
    wgpu_core::ruffle_patch::set_present_min_duration(seconds);
    #[cfg(target_family = "wasm")]
    let _ = seconds;
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
    /// Finish the encoder and submit it.
    Frame(wgpu::CommandEncoder, wgpu::Queue),
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
                for job in job_receiver {
                    match job {
                        SubmitJob::Frame(encoder, queue) => {
                            queue.submit(Some(encoder.finish()));
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
            staging_belt: wgpu::util::StagingBelt::new(descriptors.device.clone(), 65536),
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
        self.dynamic_transforms.flush(&descriptors.queue);
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
        self.wait_for_submission();
        apply_wgpu_core_switches();
        self.draws_since_flush = 0;
        self.staging_belt.finish();
        self.dynamic_transforms.flush(&descriptors.queue);
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
            .unwrap_or_else(|| wgpu::util::StagingBelt::new(descriptors.device.clone(), 65536));
        self.in_flight_belt = Some(std::mem::replace(&mut self.staging_belt, next_belt));
        let submitter = self.submitter.get_or_insert_with(AsyncSubmitter::new);
        submitter
            .jobs
            .send(SubmitJob::Frame(draw_encoder, descriptors.queue.clone()))
            .expect("Submit thread should be running");
        self.jobs_in_flight += 1;
    }

    pub fn submit_direct(&mut self, descriptors: &Descriptors) -> SubmissionIndex {
        self.wait_for_submission();
        apply_wgpu_core_switches();
        self.draws_since_flush = 0;
        self.staging_belt.finish();
        self.dynamic_transforms.flush(&descriptors.queue);
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
