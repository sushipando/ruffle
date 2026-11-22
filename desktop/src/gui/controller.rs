use super::dialogs::export_bundle_dialog::ExportBundleDialogConfiguration;
use super::{DialogDescriptor, FilePicker};
use crate::backends::DesktopUiBackend;
use crate::custom_event::RuffleEvent;
use crate::gui::movie::{MovieView, MovieViewRenderer};
use crate::gui::theme::ThemeController;
use crate::gui::{MENU_HEIGHT, RuffleGui};
use crate::player::{LaunchOptions, PlayerController};
use crate::preferences::GlobalPreferences;
use anyhow::anyhow;
use egui::{Context, FontData, FontDefinitions, ViewportId};
use fontdb::{Database, Family, Query, Source};
use ruffle_core::events::{ImeCursorArea, ImePurpose};
use ruffle_core::{Player, PlayerEvent};
use ruffle_frontend_utils::content::ContentDescriptor;
use ruffle_render_wgpu::backend::{
    WgpuRenderBackend, create_wgpu_instance, request_adapter_and_device,
};
use ruffle_render_wgpu::descriptors::Descriptors;
use ruffle_render_wgpu::utils::{format_list, get_backend_names};
use std::any::Any;
use std::fs::File;
use std::path::Path;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};
use url::Url;
use winit::dpi::{PhysicalPosition, PhysicalSize};
use winit::event::WindowEvent;
use winit::event_loop::EventLoopProxy;
use winit::keyboard::{Key, NamedKey};
use winit::window::{ImePurpose as WinitImePurpose, Theme, Window};

/// Integration layer connecting wgpu+winit to egui.
pub struct GuiController {
    descriptors: Arc<Descriptors>,
    egui_winit: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    gui: RuffleGui,
    window: Arc<Window>,
    last_update: Instant,
    repaint_after: Duration,
    /// Whether egui drew anything (menu bar, context menu, dialogs, debug tools) last time.
    shows_ui: bool,
    /// True after a frame bypassed `MovieView` by resolving directly into the swapchain,
    /// so the next frame that needs `MovieView` (e.g. when an egui menu opens) redraws it.
    movie_view_stale: bool,
    surface: Arc<wgpu::Surface<'static>>,
    surface_format: wgpu::TextureFormat,
    movie_view_renderer: Arc<MovieViewRenderer>,
    // Note that `window.get_inner_size` can change at any point on x11, even between two lines of code.
    // Use this instead.
    size: PhysicalSize<u32>,
    /// If this is set, we should not render the main menu.
    no_gui: bool,
    theme_controller: ThemeController,
    /// Presents handed to the renderer's submit thread (see `render`).
    pending_presents: Arc<PendingPresents>,
    #[cfg(feature = "tracy_images")]
    tracy_frame_captures: crate::tracy::FrameCapturesHolder,
}

/// Counts the frames that were handed to another thread to be presented, but
/// haven't been yet. A new surface texture can only be acquired (and the surface
/// only reconfigured) once the previous one was presented.
#[derive(Default)]
struct PendingPresents {
    count: Mutex<usize>,
    presented: Condvar,
}

impl PendingPresents {
    /// Marks a present as pending, until the returned guard is dropped.
    fn start(self: &Arc<Self>) -> PendingPresentGuard {
        *self.count.lock().expect("pending presents lock") += 1;
        PendingPresentGuard(self.clone())
    }

    fn wait(&self) {
        let mut count = self.count.lock().expect("pending presents lock");
        while *count > 0 {
            count = self.presented.wait(count).expect("pending presents lock");
        }
    }
}

struct PendingPresentGuard(Arc<PendingPresents>);

impl Drop for PendingPresentGuard {
    fn drop(&mut self) {
        *self.0.count.lock().expect("pending presents lock") -= 1;
        self.0.presented.notify_all();
    }
}

impl GuiController {
    pub fn new(
        window: Arc<Window>,
        event_loop: EventLoopProxy<RuffleEvent>,
        preferences: GlobalPreferences,
        font_database: &Database,
        initial_movie_url: Option<Url>,
        no_gui: bool,
    ) -> anyhow::Result<Self> {
        let (instance, backend) =
            select_wgpu_backend(preferences.graphics_backends().into(), &window)?;
        let surface = instance.create_surface(window.clone())?;
        let (adapter, device, queue) = futures::executor::block_on(request_adapter_and_device(
            backend,
            &instance,
            Some(&surface),
            preferences.graphics_power_preference().into(),
        ))
        .map_err(|e| anyhow!(e.to_string()))?;
        let adapter_info = adapter.get_info();
        tracing::info!(
            "Using graphics API {} on {} (type: {:?})",
            adapter_info.backend.to_str(),
            adapter_info.name,
            adapter_info.device_type
        );
        let preferred_formats = [
            // by egui
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureFormat::Bgra8Unorm,
        ];
        let supported_formats = surface.get_capabilities(&adapter).formats;
        let surface_format = preferred_formats
            .iter()
            .find(|format| supported_formats.contains(format))
            .copied()
            .unwrap_or_else(|| {
                supported_formats
                    .first()
                    .copied()
                    .expect("At least one format should be supported")
            });
        tracing::info!("Using surface format {:?}", surface_format);
        let size = window.inner_size();
        surface.configure(
            &device,
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: surface_format,
                color_space: wgpu::SurfaceColorSpace::Auto,
                width: size.width,
                height: size.height,
                present_mode: Default::default(),
                desired_maximum_frame_latency: 2,
                alpha_mode: Default::default(),
                view_formats: Default::default(),
            },
        );
        let descriptors = Descriptors::new(instance, adapter, device, queue);
        let egui_ctx = Context::default();

        let theme_controller = futures::executor::block_on(ThemeController::new(
            window.clone(),
            preferences.clone(),
            egui_ctx.clone(),
        ));
        let mut egui_winit = egui_winit::State::new(
            egui_ctx,
            ViewportId::ROOT,
            window.as_ref(),
            None,
            None,
            None,
        );
        egui_winit.set_max_texture_side(descriptors.limits.max_texture_dimension_2d as usize);

        let movie_view_renderer = Arc::new(MovieViewRenderer::new(
            &descriptors.device,
            surface_format,
            window.fullscreen().is_none() && !no_gui,
            size.height,
            window.scale_factor(),
        ));
        let egui_renderer = egui_wgpu::Renderer::new(
            &descriptors.device,
            surface_format,
            egui_wgpu::RendererOptions {
                msaa_samples: 1,
                depth_stencil_format: None,
                dithering: false,
                predictable_texture_filtering: false,
            },
        );
        let descriptors = Arc::new(descriptors);
        let gui = RuffleGui::new(
            Arc::downgrade(&window),
            event_loop,
            initial_movie_url.map(|url| ContentDescriptor {
                url,
                root_content_path: None,
            }),
            LaunchOptions::from(&preferences),
            preferences.clone(),
        );
        let system_fonts = load_system_fonts(font_database, preferences.language());
        egui_winit.egui_ctx().set_fonts(system_fonts);

        egui_extras::install_image_loaders(egui_winit.egui_ctx());

        #[cfg(feature = "tracy_images")]
        let tracy_frame_captures = crate::tracy::FrameCapturesHolder::new(&descriptors.device);

        Ok(Self {
            descriptors,
            egui_winit,
            egui_renderer,
            gui,
            window,
            last_update: Instant::now(),
            repaint_after: Duration::ZERO,
            shows_ui: true,
            movie_view_stale: false,
            surface: Arc::new(surface),
            surface_format,
            movie_view_renderer,
            size,
            no_gui,
            theme_controller,
            pending_presents: Default::default(),
            #[cfg(feature = "tracy_images")]
            tracy_frame_captures,
        })
    }

    pub fn set_theme(&self, theme: Theme) {
        self.theme_controller.set_theme(theme);
    }

    pub fn descriptors(&self) -> &Arc<Descriptors> {
        &self.descriptors
    }

    pub fn file_picker(&self) -> FilePicker {
        self.gui.dialogs.file_picker()
    }

    pub fn window(&self) -> &Arc<Window> {
        &self.window
    }

    pub fn resize(&mut self, size: PhysicalSize<u32>) {
        if size.width > 0 && size.height > 0 {
            self.size = size;
            self.reconfigure_surface();
        }
    }

    pub fn reconfigure_surface(&self) {
        self.pending_presents.wait();
        self.surface.configure(
            &self.descriptors.device,
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: self.surface_format,
                color_space: wgpu::SurfaceColorSpace::Auto,
                width: self.size.width,
                height: self.size.height,
                present_mode: Default::default(),
                desired_maximum_frame_latency: 2,
                alpha_mode: Default::default(),
                view_formats: Default::default(),
            },
        );
        self.movie_view_renderer.update_resolution(
            &self.descriptors,
            self.window.fullscreen().is_none() && !self.no_gui,
            self.size.height,
            self.window.scale_factor(),
        );
    }

    #[must_use]
    pub fn handle_event(&mut self, event: &WindowEvent) -> bool {
        if let WindowEvent::Resized(size) = &event {
            self.resize(*size);
        }

        if let WindowEvent::ThemeChanged(theme) = &event {
            self.set_theme(*theme);
        }

        if matches!(
            &event,
            WindowEvent::KeyboardInput {
                event: winit::event::KeyEvent {
                    logical_key: Key::Named(NamedKey::Tab),
                    ..
                },
                ..
            }
        ) {
            // Prevent egui from consuming the Tab key.
            return false;
        }

        let response = self.egui_winit.on_window_event(&self.window, event);
        // egui asks for a repaint on every mouse move. When it shows nothing, that would
        // only present the movie's last frame again (and it gets the input with the next
        // frame anyway).
        let pointer_only = matches!(event, WindowEvent::CursorMoved { .. });
        if response.repaint && (self.shows_ui || !pointer_only) {
            self.window.request_redraw();
        }
        response.consumed
    }

    pub fn close_movie(&mut self, player: &mut PlayerController) {
        player.destroy();
        self.gui.on_player_destroyed();
    }

    pub fn create_movie(
        &mut self,
        player: &mut PlayerController,
        opt: LaunchOptions,
        content_descriptor: ContentDescriptor,
    ) {
        tracing::info!("Opening {}", content_descriptor.describe());

        self.close_movie(player);
        let movie_view = MovieView::new(
            self.movie_view_renderer.clone(),
            &self.descriptors.device,
            self.size.width,
            self.size.height,
            #[cfg(feature = "tracy_images")]
            self.tracy_frame_captures.clone(),
        );
        player.create(&opt, &content_descriptor, movie_view);
        self.gui.on_player_created(
            opt,
            content_descriptor,
            player
                .get()
                .expect("Player must exist after being created."),
        );
    }

    pub fn height_offset(&self) -> f64 {
        if self.window.fullscreen().is_some() || self.no_gui {
            0.0
        } else {
            MENU_HEIGHT as f64 * self.window.scale_factor()
        }
    }

    pub fn window_to_movie_position(&self, position: PhysicalPosition<f64>) -> (f64, f64) {
        let x = position.x;
        let y = position.y - self.height_offset();
        (x, y)
    }

    pub fn movie_to_window_position(&self, x: f64, y: f64) -> PhysicalPosition<f64> {
        let y = y + self.height_offset();
        PhysicalPosition::new(x, y)
    }

    /// Prepares the player's renderer to defer submitting its command buffer so
    /// `GuiController::render` can append `egui_render` onto the same Metal command buffer.
    pub fn begin_movie_render(&self, player: &mut Player) {
        if ruffle_render::perf_flags::enabled(ruffle_render::perf_flags::PerfFlag::AsyncPresent)
            && let Some(renderer) =
                <dyn Any>::downcast_mut::<WgpuRenderBackend<MovieView>>(player.renderer_mut())
        {
            renderer.set_defer_submit(true);
        }
    }

    fn flush_deferred_movie(&self, player: &mut Option<MutexGuard<Player>>) {
        if let Some(player) = player.as_deref_mut()
            && let Some(renderer) =
                <dyn Any>::downcast_mut::<WgpuRenderBackend<MovieView>>(player.renderer_mut())
            && renderer.has_deferred_frame()
        {
            renderer.submit_deferred_frame(Vec::new(), None);
        }
    }

    /// Runs the GUI and presents the frame (calling `render_movie` when the movie needs
    /// to be drawn). Returns `(acquire_wait, movie_drawn)`.
    ///
    /// When no egui UI or menu bar is shown and `DirectFrame` + `AsyncPresent` are active
    /// on a non-sRGB surface, the movie resolves directly into the acquired swapchain
    /// texture, skipping `MovieView` and the fullscreen `egui_render` pass completely.
    /// When neither the movie nor the GUI has anything new to show, nothing is presented.
    /// The frame is shown at `present_time` if given (see `MainWindow::present_time`),
    /// otherwise as soon as possible.
    pub fn render<F>(
        &mut self,
        mut player: Option<MutexGuard<Player>>,
        want_movie: bool,
        present_time: Option<Instant>,
        render_movie: F,
    ) -> (Duration, bool)
    where
        F: FnOnce(&mut Player),
    {
        let raw_input = self.egui_winit.take_egui_input(&self.window);
        let show_menu = self.window.fullscreen().is_none() && !self.no_gui;
        let mut full_output = self.egui_winit.egui_ctx().run_ui(raw_input, |ui| {
            self.gui.update(
                self.egui_winit.egui_ctx(),
                ui,
                show_menu,
                player.as_deref_mut(),
                if show_menu {
                    MENU_HEIGHT as f64 * self.window.scale_factor()
                } else {
                    0.0
                },
            );
        });
        self.repaint_after = full_output
            .viewport_output
            .get(&ViewportId::ROOT)
            .expect("Root viewport must exist")
            .repaint_delay;

        // If we're not in a UI, tell egui which cursor we prefer to use instead
        if !self.egui_winit.egui_ctx().egui_wants_pointer_input()
            && let Some(player) = player.as_deref()
        {
            full_output.platform_output.cursor_icon =
                <dyn Any>::downcast_ref::<DesktopUiBackend>(player.ui())
                    .unwrap_or_else(|| panic!("UI Backend should be DesktopUiBackend"))
                    .cursor();
        }
        self.egui_winit
            .handle_platform_output(&self.window, full_output.platform_output);

        let clipped_primitives = self
            .egui_winit
            .egui_ctx()
            .tessellate(full_output.shapes, full_output.pixels_per_point);
        let has_egui = !clipped_primitives.is_empty();
        let showed_ui = std::mem::replace(&mut self.shows_ui, has_egui);
        // Texture updates don't need a frame to be presented (nor do they wait for one).
        for (id, image_deltas) in full_output.textures_delta.set.drain() {
            for image_delta in &image_deltas {
                self.egui_renderer.update_texture(
                    &self.descriptors.device,
                    &self.descriptors.queue,
                    id,
                    image_delta,
                );
            }
        }
        let need_movie = player.is_some() && (want_movie || (has_egui && self.movie_view_stale));
        if !need_movie && !showed_ui && !self.shows_ui {
            self.free_textures(std::mem::take(&mut full_output.textures_delta.free));
            return (Duration::ZERO, false);
        }

        let can_direct_swapchain = need_movie
            && !has_egui
            && !show_menu
            && self.height_offset() == 0.0
            && !self.surface_format.is_srgb()
            && cfg!(not(feature = "tracy_images"))
            && ruffle_render::perf_flags::enabled(
                ruffle_render::perf_flags::PerfFlag::DirectFrame,
            )
            && ruffle_render::perf_flags::enabled(
                ruffle_render::perf_flags::PerfFlag::AsyncPresent,
            );

        let mut render_movie = Some(render_movie);
        let mut movie_drawn = false;

        if can_direct_swapchain
            && let Some(ref mut p) = player
            && let Some(renderer) =
                <dyn Any>::downcast_mut::<WgpuRenderBackend<MovieView>>(p.renderer_mut())
        {
            enum AcquiredSurface {
                Ready(wgpu::SurfaceTexture),
                Reconfigure,
                Skip,
            }
            let surface = self.surface.clone();
            let pending_presents = self.pending_presents.clone();
            let expected_size = self.size;
            let acquired: Arc<Mutex<Option<(AcquiredSurface, Duration)>>> =
                Arc::new(Mutex::new(None));
            let acquired_cb = acquired.clone();

            let defer = ruffle_render::perf_flags::enabled(
                ruffle_render::perf_flags::PerfFlag::AsyncSubmit,
            );
            renderer.set_defer_submit(defer);
            renderer.set_direct_output_acquirer(Some(Box::new(move || {
                let wait_start = Instant::now();
                pending_presents.wait();
                let res = surface.get_current_texture();
                let wait = wait_start.elapsed();
                ruffle_core::perf_stats::record_render_wait(wait);
                match res {
                    wgpu::CurrentSurfaceTexture::Success(t)
                    | wgpu::CurrentSurfaceTexture::Suboptimal(t) => {
                        if t.texture.width() == expected_size.width
                            && t.texture.height() == expected_size.height
                        {
                            let tex = t.texture.clone();
                            *acquired_cb.lock().expect("acquired lock") =
                                Some((AcquiredSurface::Ready(t), wait));
                            ruffle_render_wgpu::backend::DirectOutputAcquire::Ready(tex)
                        } else {
                            drop(t);
                            *acquired_cb.lock().expect("acquired lock") =
                                Some((AcquiredSurface::Skip, wait));
                            ruffle_render_wgpu::backend::DirectOutputAcquire::SkipFrame
                        }
                    }
                    state @ (wgpu::CurrentSurfaceTexture::Lost
                    | wgpu::CurrentSurfaceTexture::Outdated) => {
                        tracing::warn!("Surface became unavailable: {:?}, reconfiguring", state);
                        *acquired_cb.lock().expect("acquired lock") =
                            Some((AcquiredSurface::Reconfigure, wait));
                        ruffle_render_wgpu::backend::DirectOutputAcquire::SkipFrame
                    }
                    wgpu::CurrentSurfaceTexture::Timeout => {
                        tracing::warn!("Surface acquisition timed out, skipping a frame");
                        *acquired_cb.lock().expect("acquired lock") =
                            Some((AcquiredSurface::Skip, wait));
                        ruffle_render_wgpu::backend::DirectOutputAcquire::SkipFrame
                    }
                    wgpu::CurrentSurfaceTexture::Occluded => {
                        tracing::trace!("Surface is occluded, skipping a frame");
                        *acquired_cb.lock().expect("acquired lock") =
                            Some((AcquiredSurface::Skip, wait));
                        ruffle_render_wgpu::backend::DirectOutputAcquire::SkipFrame
                    }
                    wgpu::CurrentSurfaceTexture::Validation => {
                        panic!("wgpu: Acquiring a texture failed with a validation error");
                    }
                }
            })));

            if let Some(f) = render_movie.take() {
                f(p);
            }
            let renderer =
                <dyn Any>::downcast_mut::<WgpuRenderBackend<MovieView>>(p.renderer_mut())
                    .expect("checked above");
            renderer.set_direct_output_acquirer(None);

            match acquired.lock().expect("acquired lock").take() {
                Some((AcquiredSurface::Ready(surface_texture), acquire_wait)) => {
                    self.window.pre_present_notify();
                    let queue = self.descriptors.queue.clone();
                    if renderer.has_deferred_frame() {
                        let guard = self.pending_presents.start();
                        renderer.submit_deferred_frame(
                            Vec::new(),
                            Some(Box::new(move || {
                                ruffle_render_wgpu::backend::set_present_time(present_time);
                                queue.present(surface_texture);
                                drop(guard);
                            })),
                        );
                    } else {
                        let guard = self.pending_presents.start();
                        renderer.run_after_submissions(Box::new(move || {
                            ruffle_render_wgpu::backend::set_present_time(present_time);
                            queue.present(surface_texture);
                            drop(guard);
                        }));
                    }
                    self.movie_view_stale = true;
                    self.free_textures(std::mem::take(&mut full_output.textures_delta.free));
                    #[cfg(feature = "tracy")]
                    tracing_tracy::client::frame_mark();
                    return (acquire_wait, true);
                }
                Some((AcquiredSurface::Reconfigure, acquire_wait)) => {
                    if renderer.has_deferred_frame() {
                        renderer.submit_deferred_frame(Vec::new(), None);
                    }
                    self.reconfigure_surface();
                    self.free_textures(std::mem::take(&mut full_output.textures_delta.free));
                    return (acquire_wait, false);
                }
                Some((AcquiredSurface::Skip, acquire_wait)) => {
                    if renderer.has_deferred_frame() {
                        renderer.submit_deferred_frame(Vec::new(), None);
                    }
                    self.free_textures(std::mem::take(&mut full_output.textures_delta.free));
                    return (acquire_wait, false);
                }
                None => {
                    // `f(p)` did not submit a frame (e.g. player paused without drawing).
                    self.free_textures(std::mem::take(&mut full_output.textures_delta.free));
                    return (Duration::ZERO, false);
                }
            }
        }

        if need_movie
            && let Some(ref mut p) = player
            && let Some(f) = render_movie.take()
        {
            self.begin_movie_render(p);
            f(p);
            self.movie_view_stale = false;
            movie_drawn = true;
        }

        self.pending_presents.wait();
        let acquire_start = Instant::now();
        let surface_texture = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(surface_texture) => surface_texture,
            wgpu::CurrentSurfaceTexture::Suboptimal(surface_texture) => {
                // The acquired texture is still usable - no reason to waste
                // this frame. The surface should ideally be reconfigured
                // before the next acquire; usually a resize event arrives and
                // does that, but not always (e.g. on a display transform
                // change) - we accept staying suboptimal in that case for now.
                surface_texture
            }
            state @ (wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Outdated) => {
                // Reconfigure the surface if lost or outdated.
                // Some sources suggest ignoring `Outdated` and waiting for the next frame,
                // but I suspect this advice is related explicitly to resizing,
                // because the future resize event will reconfigure the surface.
                // However, resizing is not the only possible reason for the surface
                // to become outdated (resolution / refresh rate change, some internal
                // platform-specific reasons, wgpu bugs?).
                // Testing on Vulkan shows that reconfiguring the surface works in that case.
                tracing::warn!("Surface became unavailable: {:?}, reconfiguring", state);
                self.flush_deferred_movie(&mut player);
                self.reconfigure_surface();
                self.free_textures(std::mem::take(&mut full_output.textures_delta.free));
                return (acquire_start.elapsed(), movie_drawn);
            }
            wgpu::CurrentSurfaceTexture::Timeout => {
                // Acquiring the frame took too long (a full second, as of wgpu 29).
                // The best thing we can do is skip a frame and wait.
                tracing::warn!("Surface acquisition timed out, skipping a frame");
                self.flush_deferred_movie(&mut player);
                self.free_textures(std::mem::take(&mut full_output.textures_delta.free));
                return (acquire_start.elapsed(), movie_drawn);
            }
            wgpu::CurrentSurfaceTexture::Occluded => {
                // The window is occluded (e.g. minimized or fully covered by
                // another window), so there is no point in drawing anything.
                // We normally don't even get here, as `WindowEvent::Occluded`
                // makes us skip rendering entirely - but the two can race, and
                // not every platform reports occlusion through winit.
                // Unlike `Timeout`, this returns immediately, so it would repeat
                // at the movie's frame rate: keep it quiet, it's not an error.
                tracing::trace!("Surface is occluded, skipping a frame");
                self.flush_deferred_movie(&mut player);
                self.free_textures(std::mem::take(&mut full_output.textures_delta.free));
                return (acquire_start.elapsed(), movie_drawn);
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                // Generic error, not much we can do.
                panic!("wgpu: Acquiring a texture failed with a validation error");
            }
        };
        let acquire_wait = acquire_start.elapsed();

        #[cfg(feature = "tracy_images")]
        if player.is_none() {
            self.tracy_frame_captures
                .set_target(&self.descriptors.device, None);
        }

        let scale_factor = self.window.scale_factor() as f32;
        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.size.width, self.size.height],
            pixels_per_point: scale_factor,
        };
        let window_size = (self.size.width, self.size.height);
        let menu_height = self.height_offset() as f32;
        let surface_view = surface_texture.texture.create_view(&Default::default());

        // Fast path: append `egui_render` directly onto the movie's deferred
        // `CommandEncoder` and submit + present in one async job (avoiding an extra
        // Metal command buffer and queue.submit every frame).
        let mut renderer = player.as_deref_mut().and_then(|player| {
            <dyn Any>::downcast_mut::<WgpuRenderBackend<MovieView>>(player.renderer_mut())
        });
        if let Some(ref mut renderer) = renderer
            && renderer.has_deferred_frame()
        {
            let egui_renderer = &mut self.egui_renderer;
            let descriptors = &self.descriptors;
            let movie_view_renderer = &self.movie_view_renderer;
            #[cfg(feature = "tracy_images")]
            let tracy_frame_captures = &mut self.tracy_frame_captures;
            let extra_buffers = renderer.with_target_and_encoder(|movie_view, encoder| {
                let extra = if has_egui {
                    egui_renderer.update_buffers(
                        &descriptors.device,
                        &descriptors.queue,
                        encoder,
                        &clipped_primitives,
                        &screen_descriptor,
                    )
                } else {
                    Vec::new()
                };
                let mut render_pass = encoder
                    .begin_render_pass(&wgpu::RenderPassDescriptor {
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &surface_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        label: Some("egui_render"),
                        ..Default::default()
                    })
                    .forget_lifetime();
                movie_view.render(movie_view_renderer, &mut render_pass, window_size, menu_height);
                if has_egui {
                    egui_renderer.render(&mut render_pass, &clipped_primitives, &screen_descriptor);
                }
                #[cfg(feature = "tracy_images")]
                {
                    drop(render_pass);
                    tracy_frame_captures.capture_frame(&descriptors.device, encoder);
                }
                extra
            });
            self.window.pre_present_notify();
            let queue = self.descriptors.queue.clone();
            let guard = self.pending_presents.start();
            renderer.submit_deferred_frame(
                extra_buffers,
                Some(Box::new(move || {
                    ruffle_render_wgpu::backend::set_present_time(present_time);
                    queue.present(surface_texture);
                    drop(guard);
                })),
            );
        } else {
            let mut encoder =
                self.descriptors
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("egui encoder"),
                    });

            let mut command_buffers = if has_egui {
                self.egui_renderer.update_buffers(
                    &self.descriptors.device,
                    &self.descriptors.queue,
                    &mut encoder,
                    &clipped_primitives,
                    &screen_descriptor,
                )
            } else {
                Vec::new()
            };

            let movie_view = renderer.as_deref().map(|r| r.target());

            {
                let mut render_pass = encoder
                    .begin_render_pass(&wgpu::RenderPassDescriptor {
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &surface_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        label: Some("egui_render"),
                        ..Default::default()
                    })
                    .forget_lifetime();

                if let Some(movie_view) = movie_view {
                    movie_view.render(
                        &self.movie_view_renderer,
                        &mut render_pass,
                        window_size,
                        menu_height,
                    );
                }

                if has_egui {
                    self.egui_renderer
                        .render(&mut render_pass, &clipped_primitives, &screen_descriptor);
                }

                #[cfg(feature = "tracy_images")]
                {
                    drop(render_pass);
                    self.tracy_frame_captures
                        .capture_frame(&self.descriptors.device, &mut encoder);
                }
            }

            command_buffers.push(encoder.finish());
            self.window.pre_present_notify();
            let queue = self.descriptors.queue.clone();
            let present = move || {
                queue.submit(command_buffers);
                ruffle_render_wgpu::backend::set_present_time(present_time);
                queue.present(surface_texture);
            };
            // The movie's frame may still be being submitted on the renderer's submit
            // thread. This frame shows it, so it must be submitted after it: let that
            // thread do it, instead of waiting here.
            match renderer {
                Some(renderer)
                    if ruffle_render::perf_flags::enabled(
                        ruffle_render::perf_flags::PerfFlag::AsyncPresent,
                    ) =>
                {
                    let guard = self.pending_presents.start();
                    renderer.run_after_submissions(Box::new(move || {
                        present();
                        drop(guard);
                    }));
                }
                _ => present(),
            }
        }

        self.free_textures(std::mem::take(&mut full_output.textures_delta.free));

        #[cfg(feature = "tracy")]
        tracing_tracy::client::frame_mark();
        #[cfg(feature = "tracy_images")]
        {
            self.pending_presents.wait();
            self.tracy_frame_captures.finish_frame();
        }

        (acquire_wait, movie_drawn)
    }

    /// Frees textures the GUI no longer uses: only after submitting the command buffers
    /// that may still reference them (destroying a texture before its usage is submitted
    /// causes a wgpu validation panic).
    fn free_textures(&mut self, textures: impl IntoIterator<Item = egui::TextureId>) {
        let mut textures = textures.into_iter().peekable();
        if textures.peek().is_some() {
            self.pending_presents.wait();
        }
        for id in textures {
            self.egui_renderer.free_texture(&id);
        }
    }

    pub fn show_context_menu(
        &mut self,
        menu: Vec<ruffle_core::ContextMenuItem>,
        close_event: PlayerEvent,
    ) {
        self.gui.show_context_menu(menu, close_event);
    }

    pub fn is_context_menu_visible(&self) -> bool {
        self.gui.is_context_menu_visible()
    }

    pub fn needs_render(&self) -> bool {
        Instant::now().duration_since(self.last_update) >= self.repaint_after
    }

    pub fn show_open_dialog(&mut self) {
        self.gui.dialogs.open_file_advanced()
    }

    pub fn open_dialog(&mut self, dialog_event: DialogDescriptor) {
        self.gui.dialogs.open_dialog(dialog_event);
    }

    pub fn set_ime_allowed(&self, allowed: bool) {
        self.window.set_ime_allowed(allowed);
    }

    pub fn set_ime_purpose(&self, purpose: ImePurpose) {
        self.window.set_ime_purpose(match purpose {
            ImePurpose::Standard => WinitImePurpose::Normal,
            ImePurpose::Password => WinitImePurpose::Password,
        });
    }

    pub fn set_ime_cursor_area(&self, cursor_area: ImeCursorArea) {
        self.window.set_ime_cursor_area(
            self.movie_to_window_position(cursor_area.x, cursor_area.y),
            PhysicalSize::new(cursor_area.width, cursor_area.height),
        );
    }

    pub fn export_bundle(&mut self) {
        let Some(content_descriptor) = self.gui.dialogs.saved_content_descriptor() else {
            return;
        };

        let launch_options = self.gui.dialogs.saved_launch_options();
        let player_options = launch_options.player.clone();
        self.gui
            .dialogs
            .open_dialog(DialogDescriptor::ExportBundle(Box::new(
                ExportBundleDialogConfiguration::new(content_descriptor, player_options),
            )));
        self.gui.on_player_destroyed();
    }
}

fn select_wgpu_backend(
    preferred_backends: wgpu::Backends,
    window: &Arc<Window>,
) -> anyhow::Result<(wgpu::Instance, wgpu::Backends)> {
    for backend in preferred_backends.iter() {
        if let Some(instance) = try_wgpu_backend(backend, window.clone()) {
            tracing::info!(
                "Using preferred backend {}",
                format_list(&get_backend_names(backend), "and")
            );
            return Ok((instance, backend));
        }
    }

    tracing::warn!(
        "Preferred backend(s) of {} not available; falling back to any",
        format_list(&get_backend_names(preferred_backends), "or")
    );

    for backend in wgpu::Backends::all() - preferred_backends {
        if let Some(instance) = try_wgpu_backend(backend, window.clone()) {
            tracing::info!(
                "Using fallback backend {}",
                format_list(&get_backend_names(backend), "and")
            );
            return Ok((instance, backend));
        }
    }

    Err(anyhow!(
        "No compatible graphics backends of any kind were available"
    ))
}

fn try_wgpu_backend(backend: wgpu::Backends, window: Arc<Window>) -> Option<wgpu::Instance> {
    let instance = create_wgpu_instance(
        backend,
        wgpu::BackendOptions::default(),
        Some(Box::new(window)),
    );
    if futures::executor::block_on(instance.enumerate_adapters(backend)).is_empty() {
        None
    } else {
        Some(instance)
    }
}

// Load fallback fonts
fn load_system_fonts(
    font_database: &Database,
    locale: unic_langid::LanguageIdentifier,
) -> egui::FontDefinitions {
    let mut fd: FontDefinitions = egui::FontDefinitions::default();

    let lang = locale.language.as_str();
    let is_ja = lang == "ja";
    let is_ko = lang == "ko";
    let is_zh = lang == "zh";
    let is_sc = is_zh && locale.to_string().as_str() == "zh-CN";
    let is_tc = is_zh && !is_sc;

    let mut queries: PrioritizedQueries = Vec::new();

    // The main font
    queries.push((1, vec![Family::SansSerif]));

    // Pan-CJK fonts
    queries.push((
        2,
        vec![
            Family::Name("Noto Sans CJK"),     // Open font
            Family::Name("Source Han Sans"),   // Open font, same as Noto Sans CJK
            Family::Name("WenQuanYi Zen Hei"), // Open font
            Family::Name("Arial Unicode MS"),  // MacOS
        ],
    ));

    // Korean
    queries.push((
        3 + if is_ko { 0 } else { 1 },
        vec![
            Family::Name("Noto Sans CJK KR"), // Open font
            Family::Name("Malgun Gothic"),    // Windows
        ],
    ));

    // Japanese
    queries.push((
        3 + if is_ja { 0 } else { 1 },
        vec![
            Family::Name("Noto Sans CJK JP"), // Open font
            Family::Name("MS UI Gothic"),     // Windows
        ],
    ));

    // Chinese Simplified
    queries.push((
        3 + if is_sc { 0 } else { 1 },
        vec![
            Family::Name("Noto Sans CJK SC"), // Open font
            Family::Name("Microsoft YaHei"),  // Windows
        ],
    ));

    // Chinese Traditional
    queries.push((
        3 + if is_tc { 0 } else { 1 },
        vec![
            Family::Name("Noto Sans CJK TC"),   // Open font
            Family::Name("Microsoft JhengHei"), // Windows
        ],
    ));

    // Hebrew
    queries.push((
        4,
        vec![
            Family::Name("Noto Sans Hebrew"), // Open font
            Family::Name("Tahoma"),           // Windows
        ],
    ));

    // Arabic
    queries.push((
        5,
        vec![
            Family::Name("Noto Sans Arabic"), // Open font
            Family::Name("Tahoma"),           // Windows
        ],
    ));

    // Thai
    queries.push((
        6,
        vec![
            Family::Name("Noto Sans Thai"), // Open font
            Family::Name("Tahoma"),         // Windows
        ],
    ));

    register_family(
        font_database,
        &mut fd,
        egui::FontFamily::Proportional,
        queries,
    );

    fd
}

type FamilyQuery<'a> = Vec<Family<'a>>;
type PrioritizedQueries<'a> = Vec<(usize, FamilyQuery<'a>)>;

fn register_family(
    font_database: &Database,
    fd: &mut FontDefinitions,
    family: egui::FontFamily,
    mut queries: PrioritizedQueries<'_>,
) {
    queries.sort_by_key(|(priority, _)| *priority);
    for (_, query) in queries {
        register_family_font(font_database, fd, family.clone(), &query);
    }
}

fn register_family_font(
    font_database: &Database,
    fd: &mut FontDefinitions,
    family: egui::FontFamily,
    query: &FamilyQuery<'_>,
) {
    let (name, fontdata) = match load_system_font(font_database, query) {
        Ok((name, fontdata)) => (name, fontdata),
        Err(e) => {
            tracing::warn!("Failed to register {query:?} as {family}: {e}");
            return;
        }
    };

    tracing::debug!("Registering font {name} as {family}");

    fd.font_data.insert(name.clone(), fontdata.into());
    fd.families.entry(family).or_default().push(name);
}

fn load_system_font(
    font_database: &Database,
    families: &Vec<Family<'_>>,
) -> anyhow::Result<(String, FontData)> {
    let system_unicode_fonts = Query {
        families,
        ..Query::default()
    };

    let id = font_database
        .query(&system_unicode_fonts)
        .ok_or(anyhow!("no unicode fonts found!"))?;
    let (name, src, index) = font_database
        .face(id)
        .map(|f| (f.post_script_name.clone(), f.source.clone(), f.index))
        .expect("id not found in font database");

    let mut fontdata = match src {
        Source::File(path) | Source::SharedFile(path, _) => {
            let data = mmap_system_font(&path)?;

            // egui accepts only static data, so we have to leak mmapped fonts.
            // This is acceptable, as we're doing it only once.
            let data = Box::leak(Box::new(data));

            egui::FontData::from_static(data)
        }
        Source::Binary(bin) => {
            let data = bin.as_ref().as_ref().to_vec();
            egui::FontData::from_owned(data)
        }
    };
    fontdata.index = index;

    Ok((name, fontdata))
}

fn mmap_system_font(path: &Path) -> anyhow::Result<memmap2::Mmap> {
    let file = File::open(path).map_err(|e| anyhow!("Couldn't open font file at {path:?}: {e}"))?;

    // SAFETY: We have to assume that the font file won't change.
    // This assumption is realistic, as we're using system fonts only.
    let mmap = unsafe { memmap2::Mmap::map(&file) };

    let mmap = mmap.map_err(|e| anyhow!("Failed to mmap font file at {path:?}: {e}"))?;
    Ok(mmap)
}
