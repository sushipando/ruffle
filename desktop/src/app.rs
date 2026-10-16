use crate::custom_event::{OpenType, RuffleEvent};
use crate::gui::{GuiController, MENU_HEIGHT};
use crate::player::{LaunchOptions, PlayerController};
use crate::preferences::GlobalPreferences;
use crate::util::{
    get_screen_size, gilrs_button_to_gamepad_button, winit_input_to_ruffle_key_descriptor,
    winit_to_ruffle_text_control,
};
use anyhow::Error;
use gilrs::{Event, EventType, Gilrs};
use ruffle_core::FloatDuration;
use ruffle_core::PlayerEvent;
use ruffle_core::events::{ImeEvent, ImeNotification, PlayerNotification};
use ruffle_core::swf::HeaderExt;
use ruffle_frontend_utils::content::ContentDescriptor;
use ruffle_render::backend::ViewportDimensions;
use std::sync::Arc;
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalPosition, PhysicalSize, Size};
use winit::event::{ElementState, Ime, KeyEvent, Modifiers, StartCause, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Fullscreen, Icon, WindowAttributes, WindowId};

struct MainWindow {
    preferences: GlobalPreferences,
    gui: GuiController,
    player: PlayerController,
    minimized: bool,
    /// Set from `WindowEvent::Occluded`, which winit only reports on macOS and X11
    /// (and on X11 only without a compositor). Always false elsewhere.
    occluded: bool,
    mouse_pos: PhysicalPosition<f64>,
    modifiers: Modifiers,
    min_window_size: LogicalSize<u32>,
    /// The window's size limit: the area covered by all monitors (see `get_screen_size`),
    /// raised when monitors are added or rearranged.
    max_window_size: LogicalSize<u32>,
    /// When to check the monitors again (see `update_max_window_size`).
    next_monitor_check: Instant,
    no_gui: bool,
    preferred_width: Option<f64>,
    preferred_height: Option<f64>,
    start_fullscreen: bool,
    loaded: LoadingState,
    time: Instant,
    next_frame_time: Option<Instant>,
    /// When the player next needs to tick (a movie frame or a timer), apart from the
    /// frames drawn between movie frames (which `next_frame_time` also includes).
    next_tick_time: Option<Instant>,
    /// How many frames are drawn per movie frame (see `--interpolate`), at most.
    max_interpolate: u32,
    /// How many frames are drawn per movie frame now: lowered (to a divisor of
    /// `max_interpolate`, so that frames stay evenly spaced) while frames can't be drawn
    /// in time, and raised again later (see `adapt_interpolation`).
    interpolate: u32,
    adaptation: InterpolationAdaptation,
    /// The ideal time of the last movie frame, from which frames drawn between movie
    /// frames are timed.
    frame_start: Option<Instant>,
    /// The next frame to draw (counted from `frame_start`).
    next_subframe: u32,
    /// Whether the movie must be drawn again for the next redraw, apart from when the player
    /// says so (`Player::needs_render`). Other redraws (asked for by the GUI) only draw the
    /// GUI over the movie's last frame.
    movie_dirty: bool,
    event_loop_proxy: EventLoopProxy<RuffleEvent>,
}

/// Counts for `MainWindow::adapt_interpolation`.
struct InterpolationAdaptation {
    window_start: Instant,
    /// Frames drawn between movie frames in the current second.
    drawn: u32,
    /// Frames between movie frames skipped in the current second.
    skipped: u32,
    /// Seconds without skipped frames since the number of frames last changed.
    clean_windows: u32,
    /// How many such seconds to wait before drawing more frames again.
    retry_after: u32,
    /// Seconds since the number of frames was raised, while that may still turn out not to
    /// work (until `TRIAL_SECONDS`; one second of skipped frames lowers it again then).
    since_raise: Option<u32>,
    /// The load (the busier of the GPU and the main thread) in the second before the last
    /// raise.
    load_before_raise: f64,
    /// The load before the last raise that didn't work: once the load is clearly lower than
    /// that, it tries again without waiting for `retry_after`.
    failed_load: Option<f64>,
    /// Whether frames were skipped in the last second (it's lowered after two such).
    behind: bool,
    /// Time the main thread spent ticking and drawing in the current second.
    busy: Duration,
    /// Whether ticking or drawing once took at least `STALL` in the current second.
    stalled: bool,
    /// Never changes the number of frames (`RUFFLE_FIXED_INTERPOLATION`, for measuring).
    fixed: bool,
}

impl InterpolationAdaptation {
    /// Seconds without skipped frames before drawing more frames again, at least.
    const MIN_WAIT: u32 = 4;
    /// Seconds without skipped frames before drawing more frames again, at most (after
    /// several raises that didn't work).
    const MAX_WAIT: u32 = 64;
    /// Seconds after which a raise counts as having worked.
    const TRIAL_SECONDS: u32 = 10;
    /// How long ticking or drawing takes for it to count as a stall, as when loading new
    /// content (longer than any steady-state hitch).
    const STALL: Duration = Duration::from_millis(100);

    fn new() -> Self {
        Self {
            window_start: Instant::now(),
            drawn: 0,
            skipped: 0,
            clean_windows: 0,
            retry_after: Self::MIN_WAIT,
            since_raise: None,
            load_before_raise: 0.0,
            failed_load: None,
            behind: false,
            busy: Duration::ZERO,
            stalled: false,
            fixed: std::env::var_os("RUFFLE_FIXED_INTERPOLATION").is_some(),
        }
    }

    /// Counts time the main thread spent ticking or drawing.
    fn add_busy(&mut self, time: Duration) {
        self.busy += time;
        self.stalled |= time >= Self::STALL;
    }
}

impl MainWindow {
    pub fn window_event(&mut self, event_loop: &ActiveEventLoop, event: WindowEvent) {
        if matches!(event, WindowEvent::RedrawRequested) {
            // Don't render when nobody can see the result: while minimized, the
            // surface also keeps the size it had before, as `GuiController::resize`
            // refuses to reconfigure it to a zero size.
            if !self.minimized && !self.occluded {
                let start = Instant::now();
                let mut player = self.player.get();
                if let Some(ref mut player) = player
                    // Even if the movie is paused, user interaction with debug tools can change the render output
                    && (self.movie_dirty || player.needs_render() || !player.is_playing())
                {
                    self.movie_dirty = false;
                    let fraction = self.render_fraction(player.frame_duration_seconds());
                    player.set_render_fraction(fraction);
                    self.pace_present(player.frame_duration_seconds());
                    player.render();
                }

                self.gui.render(player);
                self.adaptation.add_busy(start.elapsed());
            }

            // Important that we return here, or we'll get a feedback loop with egui
            // (winit says redraw, egui hears redraw and says redraw, we hear redraw and tell winit to redraw...)
            return;
        }

        if self.gui.handle_event(&event) {
            // Event consumed by GUI.
            return;
        }
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                // How minimizing shows up on Windows; winit still has no
                // `WindowEvent::Minimize`. (`Window::is_minimized` does exist by
                // now, but it has to be polled, and it wouldn't tell us anything
                // new: macOS reports minimized windows as occluded instead, and
                // on Wayland it always returns `None`.)
                self.minimized = size.width == 0 && size.height == 0;

                if let Some(mut player) = self.player.get() {
                    let viewport_scale_factor = self.gui.window().scale_factor();
                    player.set_viewport_dimensions(ViewportDimensions {
                        width: size.width,
                        height: size.height.saturating_sub(self.gui.height_offset() as u32),
                        scale_factor: viewport_scale_factor,
                    });
                }
                self.movie_dirty = true;
                self.gui.window().request_redraw();
                if matches!(self.loaded, LoadingState::WaitingForResize) {
                    self.loaded = LoadingState::Loaded;
                }
            }
            WindowEvent::Occluded(occluded) => {
                self.occluded = occluded;
                if !occluded {
                    // Nothing repaints us on its own if the movie is paused.
                    self.movie_dirty = true;
                    self.gui.window().request_redraw();
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if self.gui.is_context_menu_visible() {
                    return;
                }

                // This is needed because some platforms (like macOS)
                // will fire this event at the same time as WindowEvent::MouseInput,
                // which may cause inaccurate behavior in the movie
                if position == self.mouse_pos {
                    return;
                }

                self.mouse_pos = position;
                let (x, y) = self.gui.window_to_movie_position(position);
                let event = PlayerEvent::MouseMove { x, y };
                self.player.handle_event(event);
                self.check_redraw();
            }
            WindowEvent::DroppedFile(file) => {
                if let Some(content_descriptor) = ContentDescriptor::new_local(&file, None) {
                    self.gui.create_movie(
                        &mut self.player,
                        LaunchOptions::from(&self.preferences),
                        content_descriptor,
                    );
                }
            }
            WindowEvent::Focused(true) => {
                self.player.handle_event(PlayerEvent::FocusGained);
            }
            WindowEvent::Focused(false) => {
                self.player.handle_event(PlayerEvent::FocusLost);
            }
            WindowEvent::MouseInput { button, state, .. } => {
                if self.gui.is_context_menu_visible() {
                    return;
                }

                use ruffle_core::events::MouseButton as RuffleMouseButton;
                use winit::event::MouseButton;
                let (x, y) = self.gui.window_to_movie_position(self.mouse_pos);
                let button = match button {
                    MouseButton::Left => RuffleMouseButton::Left,
                    MouseButton::Right => RuffleMouseButton::Right,
                    MouseButton::Middle => RuffleMouseButton::Middle,
                    _ => RuffleMouseButton::Unknown,
                };
                let event = match state {
                    // TODO We should get information about click index from the OS,
                    //   but winit does not support that yet.
                    ElementState::Pressed => PlayerEvent::MouseDown {
                        x,
                        y,
                        button,
                        index: None,
                    },
                    ElementState::Released => PlayerEvent::MouseUp { x, y, button },
                };
                let handled = self.player.handle_event(event);
                if !handled && state == ElementState::Pressed && button == RuffleMouseButton::Right
                {
                    // Show context menu.
                    if let Some(mut player) = self.player.get() {
                        let context_menu = player.prepare_context_menu();

                        // MouseUp event will be ignored when the context menu is shown,
                        // but it has to be dispatched when the menu closes.
                        let close_event = PlayerEvent::MouseUp {
                            x,
                            y,
                            button: RuffleMouseButton::Right,
                        };
                        self.gui.show_context_menu(context_menu, close_event);
                    }
                }
                self.check_redraw();
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if self.gui.is_context_menu_visible() {
                    return;
                }

                use ruffle_core::events::MouseWheelDelta;
                use winit::event::MouseScrollDelta;
                let delta = match delta {
                    MouseScrollDelta::LineDelta(_, dy) => MouseWheelDelta::Lines(dy.into()),
                    MouseScrollDelta::PixelDelta(pos) => MouseWheelDelta::Pixels(pos.y),
                };
                let event = PlayerEvent::MouseWheel { delta };
                self.player.handle_event(event);
                self.check_redraw();
            }
            WindowEvent::CursorEntered { .. } => {
                if let Some(mut player) = self.player.get() {
                    player.set_mouse_in_stage(true);
                    if player.needs_render() {
                        self.gui.window().request_redraw();
                    }
                }
            }
            WindowEvent::CursorLeft { .. } => {
                if let Some(mut player) = self.player.get() {
                    player.set_mouse_in_stage(false);
                }
                self.player.handle_event(PlayerEvent::MouseLeave);
                self.check_redraw();
            }
            WindowEvent::ModifiersChanged(new_modifiers) => {
                self.modifiers = new_modifiers;
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if self.gui.is_context_menu_visible() {
                    return;
                }

                // Handle escaping from fullscreen.
                if let KeyEvent {
                    state: ElementState::Pressed,
                    logical_key: Key::Named(NamedKey::Escape),
                    ..
                } = event
                {
                    let _ = self
                        .event_loop_proxy
                        .send_event(RuffleEvent::ExitFullScreen);
                }

                let key = winit_input_to_ruffle_key_descriptor(&event);
                match event.state {
                    ElementState::Pressed => {
                        self.player.handle_event(PlayerEvent::KeyDown { key });
                        if let Some(control_code) =
                            winit_to_ruffle_text_control(&event, self.modifiers)
                        {
                            self.player
                                .handle_event(PlayerEvent::TextControl { code: control_code });
                        } else if let Some(text) = event.text {
                            for codepoint in text.chars() {
                                self.player
                                    .handle_event(PlayerEvent::TextInput { codepoint });
                            }
                        }
                    }
                    ElementState::Released => {
                        self.player.handle_event(PlayerEvent::KeyUp { key });
                    }
                };
                self.check_redraw();
            }
            WindowEvent::Ime(ime) => match ime {
                Ime::Enabled => {}
                Ime::Preedit(text, cursor) => {
                    self.player
                        .handle_event(PlayerEvent::Ime(ImeEvent::Preedit(text, cursor)));
                }
                Ime::Commit(text) => {
                    self.player
                        .handle_event(PlayerEvent::Ime(ImeEvent::Commit(text)));
                }
                Ime::Disabled => {}
            },
            _ => (),
        }
    }

    fn on_metadata(&mut self, swf_header: HeaderExt) {
        let height_offset = if self.gui.window().fullscreen().is_some() || self.no_gui {
            0.0
        } else {
            MENU_HEIGHT as f64
        };

        // To prevent issues like waiting on resize indefinitely (#11364) or desyncing the window state on Windows,
        // do not resize while window is maximized.
        let should_resize = !self.gui.window().is_maximized();

        let (viewport_size, state) = if should_resize {
            let movie_width = swf_header.stage_size().width().to_pixels();
            let movie_height = swf_header.stage_size().height().to_pixels();

            let window_size: Size = match (self.preferred_width, self.preferred_height) {
                (None, None) => LogicalSize::new(movie_width, movie_height + height_offset).into(),
                (Some(width), None) => {
                    let scale = width / movie_width;
                    let height = movie_height * scale;
                    PhysicalSize::new(
                        width.max(1.0),
                        height.max(1.0) + height_offset * self.gui.window().scale_factor(),
                    )
                    .into()
                }
                (None, Some(height)) => {
                    let scale = height / movie_height;
                    let width = movie_width * scale;
                    PhysicalSize::new(
                        width.max(1.0),
                        height.max(1.0) + height_offset * self.gui.window().scale_factor(),
                    )
                    .into()
                }
                (Some(width), Some(height)) => PhysicalSize::new(
                    width.max(1.0),
                    height.max(1.0) + height_offset * self.gui.window().scale_factor(),
                )
                .into(),
            };

            let window_size = Size::clamp(
                window_size,
                self.min_window_size.into(),
                self.max_window_size.into(),
                self.gui.window().scale_factor(),
            );

            let viewport_size = self.gui.window().inner_size();
            let mut window_resize_denied = false;

            if let Some(new_viewport_size) = self.gui.window().request_inner_size(window_size) {
                if new_viewport_size != viewport_size {
                    self.gui.resize(new_viewport_size);
                } else {
                    tracing::warn!("Unable to resize window");
                    window_resize_denied = true;
                }
            }

            let viewport_size = self.gui.window().inner_size();

            // On X11 (and possibly other platforms), the window size is not updated immediately.
            // On a successful resize request, wait for the window to be resized to the requested size
            // before we start running the SWF (which can observe the viewport size in "noScale" mode)
            let state = if !window_resize_denied && window_size != viewport_size.into() {
                LoadingState::WaitingForResize
            } else {
                LoadingState::Loaded
            };

            (viewport_size, state)
        } else {
            (self.gui.window().inner_size(), LoadingState::Loaded)
        };

        self.loaded = state;

        self.gui.window().set_fullscreen(if self.start_fullscreen {
            Some(Fullscreen::Borderless(None))
        } else {
            None
        });
        self.gui.window().set_visible(true);

        let viewport_scale_factor = self.gui.window().scale_factor();
        if let Some(mut player) = self.player.get() {
            player.set_viewport_dimensions(ViewportDimensions {
                width: viewport_size.width,
                height: viewport_size.height - (height_offset * viewport_scale_factor) as u32,
                scale_factor: viewport_scale_factor,
            });
        }
    }

    fn about_to_wait(&mut self, gilrs: Option<&mut Gilrs>) {
        if let Some(Event { event, .. }) = gilrs.and_then(|gilrs| gilrs.next_event()) {
            match event {
                EventType::ButtonPressed(button, _) => {
                    if let Some(button) = gilrs_button_to_gamepad_button(button) {
                        self.player
                            .handle_event(PlayerEvent::GamepadButtonDown { button });
                        self.check_redraw();
                    }
                }
                EventType::ButtonReleased(button, _) => {
                    if let Some(button) = gilrs_button_to_gamepad_button(button) {
                        self.player
                            .handle_event(PlayerEvent::GamepadButtonUp { button });
                        self.check_redraw();
                    }
                }
                _ => {}
            }
        }

        self.update_max_window_size();

        // Core loop
        // [NA] This used to be called `MainEventsCleared`, but I think the behaviour is different now.
        // We should look at changing our tick to happen somewhere else if we see any behavioural problems.
        if matches!(self.loaded, LoadingState::Loaded) {
            let new_time = Instant::now();
            let dt = FloatDuration::from_std(new_time.duration_since(self.time));
            // Waking up only to draw a frame between movie frames doesn't tick the player
            // (ticking also runs timers and mouse picking, which can take milliseconds).
            let tick_due =
                self.interpolate <= 1 || self.next_tick_time.is_none_or(|time| new_time >= time);
            if dt.as_millis() > 0.0 && tick_due {
                self.time = new_time;
                let interpolate = self.interpolate;
                let mut new_frame_start = None;
                let tick_start = Instant::now();
                self.next_frame_time = self.player.get().map(|mut player| {
                    player.set_frame_interpolation(interpolate > 1);
                    let frames_before = player.frames_run();
                    player.tick(dt);
                    let next = new_time + player.time_til_next_frame();
                    if player.frames_run() != frames_before {
                        new_frame_start =
                            Some(next.checked_sub(Duration::from_secs_f64(
                                player.frame_duration_seconds(),
                            )));
                    }
                    next
                });
                self.adaptation.add_busy(tick_start.elapsed());
                self.next_tick_time = self.next_frame_time;
                if let Some(start) = new_frame_start {
                    let missed = self.interpolate.saturating_sub(self.next_subframe);
                    if self.frame_start.is_some() && missed <= 2 {
                        // Frames of the previous movie frame that weren't drawn.
                        self.adaptation.skipped += missed;
                    }
                    self.frame_start = start.or(Some(new_time));
                    self.next_subframe = 0;
                }
                self.check_redraw();
            }
            self.schedule_subframes();
        }
    }

    /// The time between frames drawn between movie frames, if they are (with
    /// `--interpolate`, also while only one is drawn per movie frame because more didn't
    /// keep up: they're counted then, to tell when to draw more again).
    fn subframe_duration(&self) -> Option<Duration> {
        let player = self.player.get()?;
        (self.max_interpolate > 1 && player.is_playing()).then(|| {
            Duration::from_secs_f64(player.frame_duration_seconds() / self.interpolate as f64)
        })
    }

    /// Requests drawing frames between movie frames when they're due, and makes the event
    /// loop wake up for the next one (see `--interpolate`).
    fn schedule_subframes(&mut self) {
        let (Some(step), Some(start)) = (self.subframe_duration(), self.frame_start) else {
            return;
        };
        let now = Instant::now();
        let due = (now.saturating_duration_since(start).as_secs_f64() / step.as_secs_f64()) as u32;
        if due >= self.next_subframe && self.next_subframe < self.interpolate {
            // Skip frames that are already late.
            let index = due.min(self.interpolate - 1);
            self.adaptation.drawn += 1;
            // A stall (such as loading) skips many frames at once; that's not a sign that
            // drawing fewer frames would help.
            if index - self.next_subframe <= 2 {
                self.adaptation.skipped += index - self.next_subframe;
            }
            self.next_subframe = index + 1;
            self.movie_dirty = true;
            self.gui.window().request_redraw();
        }
        self.adapt_interpolation(now);
        self.next_frame_time = self.next_tick_time;
        if self.next_subframe < self.interpolate {
            let next = start + step * self.next_subframe;
            self.next_frame_time = Some(match self.next_tick_time {
                Some(tick_time) => tick_time.min(next),
                None => next,
            });
        }
    }

    /// Lowers the number of frames drawn per movie frame while more than a tenth of them
    /// are skipped (they couldn't be drawn in time), to the next divisor of
    /// `max_interpolate`: evenly spaced frames look smoother than more frames at uneven
    /// times. Tries the next higher number after a few seconds without skipped frames.
    /// How busy the GPU is doesn't tell whether that will fit: GPUs whose clock follows
    /// their load (such as Apple's) look about as busy with less work. So when frames are
    /// skipped soon after raising, it goes back down and waits twice as long before trying
    /// again, unless the load gets clearly lower than it was before that raise, or a stall
    /// (such as loading new content) suggests that what's shown changed.
    fn adapt_interpolation(&mut self, now: Instant) {
        let adaptation = &mut self.adaptation;
        let window = now.duration_since(adaptation.window_start);
        if window < Duration::from_secs(1) || adaptation.fixed {
            return;
        }
        let (drawn, skipped) = (adaptation.drawn, adaptation.skipped);
        let cpu = adaptation.busy.as_secs_f64() / window.as_secs_f64();
        let load = ruffle_render_wgpu::backend::gpu_busy_fraction().map_or(cpu, |gpu| gpu.max(cpu));
        let stalled = std::mem::take(&mut adaptation.stalled);
        adaptation.window_start = now;
        adaptation.drawn = 0;
        adaptation.skipped = 0;
        adaptation.busy = Duration::ZERO;
        if stalled {
            // What was learned about the content may no longer hold.
            adaptation.clean_windows = 0;
            adaptation.retry_after = InterpolationAdaptation::MIN_WAIT;
            adaptation.failed_load = None;
        }
        let divisors: Vec<u32> = (1..=self.max_interpolate)
            .filter(|n| self.max_interpolate.is_multiple_of(*n))
            .collect();
        let total = drawn + skipped;
        let behind = total > 0 && skipped * 10 > total;
        let was_behind = std::mem::replace(&mut adaptation.behind, behind);
        let on_trial = adaptation.since_raise.is_some();
        if behind && (was_behind || on_trial) {
            if let Some(&lower) = divisors.iter().rev().find(|&&n| n < self.interpolate) {
                if on_trial {
                    adaptation.retry_after =
                        (adaptation.retry_after * 2).min(InterpolationAdaptation::MAX_WAIT);
                    adaptation.failed_load = Some(adaptation.load_before_raise);
                }
                adaptation.since_raise = None;
                adaptation.clean_windows = 0;
                self.interpolate = lower;
                tracing::info!("[interpolate] {lower} frames per movie frame (couldn't keep up)");
            }
            return;
        }
        if let Some(seconds) = &mut adaptation.since_raise {
            *seconds += 1;
            if *seconds >= InterpolationAdaptation::TRIAL_SECONDS {
                // The last raise worked.
                adaptation.since_raise = None;
                adaptation.retry_after = InterpolationAdaptation::MIN_WAIT;
                adaptation.failed_load = None;
            }
        }
        if skipped == 0 {
            adaptation.clean_windows += 1;
            let lighter = adaptation
                .failed_load
                .is_some_and(|failed| load <= failed * 0.85);
            if (adaptation.clean_windows >= adaptation.retry_after
                || (adaptation.clean_windows >= InterpolationAdaptation::MIN_WAIT && lighter))
                && let Some(&higher) = divisors.iter().find(|&&n| n > self.interpolate)
            {
                adaptation.since_raise = Some(0);
                adaptation.load_before_raise = load;
                adaptation.clean_windows = 0;
                tracing::info!(
                    "[interpolate] {higher} frames per movie frame (load {:.0}% at {})",
                    load * 100.0,
                    self.interpolate
                );
                self.interpolate = higher;
            }
        }
    }

    /// How far between the previous movie frame and the current one a frame drawn now
    /// should be (`None` without interpolation): the frames between movie frames are
    /// spread evenly, the last one showing the current movie frame.
    fn render_fraction(&self, frame_duration: f64) -> Option<f32> {
        if self.interpolate <= 1 {
            return None;
        }
        let start = self.frame_start?;
        let step = frame_duration / self.interpolate as f64;
        let index = (Instant::now()
            .saturating_duration_since(start)
            .as_secs_f64()
            / step)
            .min((self.interpolate - 1) as f64) as u32;
        Some((index + 1) as f32 / self.interpolate as f32)
    }

    /// Raises the window's size limit when monitors were added or rearranged: otherwise
    /// the window couldn't be made fullscreen on (or tiled on) a monitor that was connected
    /// after it opened, if that monitor is larger. The limit is never lowered, which would
    /// resize the window.
    fn update_max_window_size(&mut self) {
        let now = Instant::now();
        if now < self.next_monitor_check {
            return;
        }
        self.next_monitor_check = now + Duration::from_secs(1);
        let screen_size = get_screen_size(self.gui.window());
        let max_window_size = LogicalSize::new(
            self.max_window_size.width.max(screen_size.width),
            self.max_window_size.height.max(screen_size.height),
        );
        // Only when it changed: setting it also sets the window's size.
        if max_window_size != self.max_window_size {
            self.max_window_size = max_window_size;
            self.gui.window().set_max_inner_size(Some(max_window_size));
        }
    }

    /// Paces presentation: when the display's refresh rate is a whole multiple of the rate
    /// frames are drawn at (for example 144 Hz, and a 24 fps movie drawn 6 times per
    /// frame), each frame is shown for the same number of refreshes, instead of whenever
    /// it happens to be ready (which made animations look uneven). Otherwise frames are
    /// presented as soon as possible.
    fn pace_present(&self, frame_duration: f64) {
        let refresh = self
            .gui
            .window()
            .current_monitor()
            .and_then(|monitor| monitor.refresh_rate_millihertz())
            .map(|millihertz| 1000.0 / millihertz as f64);
        let interval = frame_duration / self.interpolate.max(1) as f64;
        let min_duration = refresh.and_then(|refresh| {
            let refreshes = interval / refresh;
            let whole = refreshes.round();
            (whole >= 1.0 && (refreshes - whole).abs() < 0.05).then_some((whole - 0.5) * refresh)
        });
        ruffle_render_wgpu::backend::set_present_min_duration(min_duration.filter(|_| {
            ruffle_render::perf_flags::enabled(ruffle_render::perf_flags::PerfFlag::PacedPresent)
        }));
    }

    fn check_redraw(&self) {
        let player = self.player.get();
        if player.map(|p| p.needs_render()).unwrap_or_default() || self.gui.needs_render() {
            self.gui.window().request_redraw();
        }
    }
}

pub struct App {
    main_window: Option<MainWindow>,
    runtime: Option<tokio::runtime::Runtime>,
    gilrs: Option<Gilrs>,
    event_loop_proxy: EventLoopProxy<RuffleEvent>,
    preferences: GlobalPreferences,
    font_database: fontdb::Database,
}

/// Enters the tokio runtime context.
/// This cannot be a method, as the borrow-checker would complain.
macro_rules! enter_runtime {
    ($this:expr) => {
        let _guard = $this.runtime.as_ref().map(|runtime| runtime.enter());
    };
}

impl App {
    pub fn new(preferences: GlobalPreferences) -> Result<(Self, EventLoop<RuffleEvent>), Error> {
        let event_loop = EventLoop::with_user_event().build()?;

        let mut font_database = fontdb::Database::default();
        font_database.load_system_fonts();

        let gilrs = Gilrs::new()
            .inspect_err(|err| {
                tracing::warn!("Gamepad support could not be initialized: {err}");
            })
            .ok();
        let event_loop_proxy = event_loop.create_proxy();
        let runtime = tokio::runtime::Runtime::new()?;

        Ok((
            Self {
                main_window: None,
                runtime: Some(runtime),
                gilrs,
                event_loop_proxy,
                font_database,
                preferences,
            },
            event_loop,
        ))
    }
}

impl ApplicationHandler<RuffleEvent> for App {
    fn new_events(&mut self, event_loop: &ActiveEventLoop, cause: StartCause) {
        enter_runtime!(self);

        if cause == StartCause::Init {
            let movie_url = self.preferences.cli.movie_url.clone();
            let icon_bytes = include_bytes!("../assets/favicon-32.rgba");
            let icon =
                Icon::from_rgba(icon_bytes.to_vec(), 32, 32).expect("App icon should be correct");

            let no_gui = self.preferences.cli.no_gui;
            let min_window_size = if no_gui {
                (16, 16)
            } else {
                (350, MENU_HEIGHT + 16)
            }
            .into();
            let preferred_width = self.preferences.cli.width;
            let preferred_height = self.preferences.cli.height;
            let start_fullscreen = self.preferences.cli.fullscreen;

            #[cfg_attr(not(target_os = "linux"), allow(unused_mut))]
            let mut window_attributes = WindowAttributes::default()
                .with_visible(false)
                .with_title("Ruffle")
                .with_window_icon(Some(icon))
                .with_min_inner_size(min_window_size);

            #[cfg(target_os = "linux")]
            {
                use winit::platform::startup_notify::{
                    self, EventLoopExtStartupNotify, WindowAttributesExtStartupNotify,
                };
                use winit::platform::wayland::WindowAttributesExtWayland;
                window_attributes = window_attributes.with_name("rs.ruffle.Ruffle", "main");
                if let Some(token) = event_loop.read_token_from_env() {
                    startup_notify::reset_activation_token_env();
                    window_attributes = window_attributes.with_activation_token(token);
                }
            }

            let event_loop_proxy = self.event_loop_proxy.clone();
            let preferences = self.preferences.clone();
            let window = event_loop
                .create_window(window_attributes)
                .expect("Window should be created");
            let max_window_size = get_screen_size(&window);
            window.set_max_inner_size(Some(max_window_size));
            let window = Arc::new(window);
            let font_database = self.font_database.clone();

            let mut gui = GuiController::new(
                window.clone(),
                event_loop_proxy.clone(),
                preferences.clone(),
                &font_database,
                movie_url.clone(),
                no_gui,
            )
            .expect("GUI controller should be created");

            let mut player = PlayerController::new(
                event_loop_proxy.clone(),
                window.clone(),
                gui.descriptors().clone(),
                font_database,
                preferences.clone(),
                gui.file_picker(),
            );

            if let Some(movie_url) = &movie_url {
                gui.create_movie(
                    &mut player,
                    LaunchOptions::from(&preferences),
                    ContentDescriptor {
                        url: movie_url.clone(),
                        root_content_path: None,
                    },
                );
            } else {
                gui.show_open_dialog();
            }

            let mut loaded = LoadingState::Loading;

            if movie_url.is_none() {
                // No SWF provided on command line; show window with dummy movie immediately.
                window.set_visible(true);
                loaded = LoadingState::Loaded;
            }

            ruffle_render_wgpu::backend::set_gpu_timing(self.preferences.cli.interpolate > 1);
            self.main_window = Some(MainWindow {
                preferences,
                gui,
                player,
                min_window_size,
                max_window_size,
                next_monitor_check: Instant::now() + Duration::from_secs(1),
                no_gui,
                preferred_width,
                preferred_height,
                start_fullscreen,
                loaded,
                minimized: false,
                occluded: false,
                mouse_pos: PhysicalPosition::new(0.0, 0.0),
                modifiers: Modifiers::default(),
                time: Instant::now(),
                next_frame_time: None,
                next_tick_time: None,
                max_interpolate: self.preferences.cli.interpolate.max(1),
                interpolate: self.preferences.cli.interpolate.max(1),
                adaptation: InterpolationAdaptation::new(),
                frame_start: None,
                next_subframe: 0,
                movie_dirty: true,
                event_loop_proxy,
            });
        }
    }

    fn resumed(&mut self, _event_loop: &ActiveEventLoop) {}

    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: RuffleEvent) {
        enter_runtime!(self);

        match (&mut self.main_window, event) {
            (Some(main_window), RuffleEvent::TaskPoll(task)) => main_window.player.poll(task),

            (Some(main_window), RuffleEvent::OnMetadata(swf_header)) => {
                main_window.on_metadata(swf_header)
            }

            (Some(main_window), RuffleEvent::ContextMenuItemClicked(index)) => {
                if let Some(mut player) = main_window.player.get() {
                    player.run_context_menu_callback(index);
                }
            }

            (Some(main_window), RuffleEvent::BrowseAndOpen(options, open_type)) => {
                let event_loop = main_window.event_loop_proxy.clone();
                let picker = main_window.gui.file_picker();
                tokio::spawn(async move {
                    let picked = match open_type {
                        OpenType::File => {
                            picker.pick_ruffle_file(None).await.map(|file| (file, None))
                        }
                        OpenType::Directory => picker
                            .pick_ruffle_directory_and_content(None)
                            .await
                            .map(|(dir, file)| (file, Some(dir))),
                    };

                    if let Some(desc) =
                        picked.and_then(|(file, dir)| ContentDescriptor::new_local(&file, dir))
                    {
                        let _ = event_loop.send_event(RuffleEvent::Open(desc, options));
                    }
                });
            }

            (Some(main_window), RuffleEvent::Open(desc, options)) => {
                main_window
                    .gui
                    .create_movie(&mut main_window.player, *options, desc);
            }

            (Some(main_window), RuffleEvent::OpenDialog(descriptor)) => {
                main_window.gui.open_dialog(descriptor);
            }

            (Some(main_window), RuffleEvent::CloseFile) => {
                main_window.gui.window().set_title("Ruffle"); // Reset title since file has been closed.
                main_window.gui.close_movie(&mut main_window.player);
            }

            (Some(main_window), RuffleEvent::ExportBundle) => {
                main_window.gui.export_bundle();
            }

            (Some(main_window), RuffleEvent::EnterFullScreen) => {
                if let Some(mut player) = main_window.player.get()
                    && player.is_playing()
                {
                    player.set_fullscreen(true);
                }
            }

            (Some(main_window), RuffleEvent::ExitFullScreen) => {
                if let Some(mut player) = main_window.player.get()
                    && player.is_playing()
                {
                    player.set_fullscreen(false);
                }
            }

            (Some(main_window), RuffleEvent::PlayerNotification(notification)) => {
                match notification {
                    PlayerNotification::ImeNotification(ImeNotification::ImeReady {
                        purpose,
                        cursor_area,
                    }) => {
                        let ime_enabled = main_window.preferences.ime_enabled().unwrap_or(false);
                        main_window.gui.set_ime_allowed(ime_enabled);
                        main_window.gui.set_ime_purpose(purpose);
                        main_window.gui.set_ime_cursor_area(cursor_area);
                    }
                    PlayerNotification::ImeNotification(ImeNotification::ImePurposeUpdated(
                        purpose,
                    )) => {
                        main_window.gui.set_ime_purpose(purpose);
                    }
                    PlayerNotification::ImeNotification(ImeNotification::ImeCursorAreaUpdated(
                        cursor_area,
                    )) => {
                        main_window.gui.set_ime_cursor_area(cursor_area);
                    }
                    PlayerNotification::ImeNotification(ImeNotification::ImeNotReady) => {
                        main_window.gui.set_ime_allowed(false);
                    }
                }
            }

            (_, RuffleEvent::ExitRequested) => {
                event_loop.exit();
            }

            _ => {}
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        enter_runtime!(self);

        if let Some(main_window) = &mut self.main_window {
            main_window.window_event(event_loop, event);
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        enter_runtime!(self);

        if let Some(main_window) = &mut self.main_window {
            main_window.about_to_wait(self.gilrs.as_mut());

            // The event loop is finished; let's find out how long we need to wait for.
            // We don't need to worry about earlier update requests, as it's the
            // only place where we're setting control flow, and events cancel wait.
            // Note: the control flow might be set to `ControlFlow::WaitUntil` with a
            // timestamp in the past! Take that into consideration when changing this code.
            if let Some(next_frame_time) = main_window.next_frame_time {
                event_loop.set_control_flow(ControlFlow::WaitUntil(next_frame_time));
            }
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        // `MainWindow` needs a tokio context to properly drop.
        {
            enter_runtime!(self);
            let _ = self.main_window.take();
        }

        // Manually stop the tokio runtime: this makes sure that any pending Player-bound futures
        // are properly cancelled and put back on the winit event loop before it closes, preventing
        // them from being dropped on the wrong thread and causing a panic.
        if let Some(runtime) = self.runtime.take() {
            runtime.shutdown_timeout(std::time::Duration::from_secs(1));
        }
    }
}

enum LoadingState {
    Loading,
    WaitingForResize,
    Loaded,
}
