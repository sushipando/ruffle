//! Interpolation of display objects' transforms between movie frames, so that a movie
//! can be drawn more often than its frame rate (see `Player::set_frame_interpolation`).
//!
//! Flash animations mostly move, rotate, scale and fade objects by changing their
//! transforms on every frame (tweens), and draw other things by replacing what objects
//! contain (frame-by-frame animation). When interpolation is enabled, each display object
//! remembers its transform from before the current frame first changed it, and frames
//! drawn between two movie frames blend the two. What objects contain still changes only
//! at movie frames. Objects created or placed in the current frame aren't blended, and
//! neither are transforms that change too much to be motion (for example, a jump back to
//! the start of a loop, or a flip).
//!
//! Drawing a frame between movie frames `n - 1` and `n` shows the blend of their
//! transforms, so interpolation delays what's shown by up to one movie frame.

use ruffle_render::matrix::Matrix;
use std::cell::Cell;
use swf::{ColorTransform, Fixed8, Twips};

/// Transforms that move further than this (in the parent's pixels) aren't blended.
const MAX_MOVE: f32 = 100.0;

/// Rotations (of either axis) by more than this (in radians) aren't blended.
const MAX_TURN: f32 = std::f32::consts::FRAC_PI_2;

/// Scale changes by more than this factor (of either axis) aren't blended.
const MAX_SCALE: f32 = 4.0;

#[derive(Clone, Copy)]
struct State {
    enabled: bool,
    /// Increased before every movie frame.
    tick: u32,
    /// While drawing a frame between movie frames: how far it is from the previous movie
    /// frame (0) to the current one (1).
    render_fraction: Option<f32>,
    /// The fraction to draw the next frame at (see `set_next_render_fraction`).
    next_render_fraction: Option<f32>,
}

thread_local! {
    static STATE: Cell<State> = const {
        Cell::new(State {
            enabled: false,
            tick: 1,
            render_fraction: None,
            next_render_fraction: None,
        })
    };
}

fn state() -> State {
    STATE.with(|state| state.get())
}

fn update(f: impl FnOnce(&mut State)) {
    STATE.with(|cell| {
        let mut state = cell.get();
        f(&mut state);
        cell.set(state);
    });
}

/// Whether display objects remember their previous transforms (on this thread).
pub fn set_enabled(enabled: bool) {
    update(|state| state.enabled = enabled);
}

/// Called before every movie frame.
pub fn advance_tick() {
    update(|state| state.tick = state.tick.wrapping_add(1).max(1));
}

pub fn tick() -> u32 {
    state().tick
}

/// Sets how far between the previous movie frame (0) and the current one (1) the next
/// frame is drawn (see `Player::render`).
pub fn set_next_render_fraction(fraction: Option<f32>) {
    update(|state| state.next_render_fraction = fraction);
}

pub fn next_render_fraction() -> Option<f32> {
    state().next_render_fraction
}

/// Runs `f` with transforms blended by `fraction` between the previous movie frame (0)
/// and the current one (1), if it's `Some`.
pub fn with_render_fraction<R>(fraction: Option<f32>, f: impl FnOnce() -> R) -> R {
    let previous = state().render_fraction;
    update(|state| {
        state.render_fraction = fraction
            .filter(|_| state.enabled)
            .map(|fraction| fraction.clamp(0.0, 1.0));
    });
    let result = f();
    update(|state| state.render_fraction = previous);
    result
}

/// Whether frames are being drawn between movie frames, and `tick` is the current one.
pub fn is_blending_tick(tick: u32) -> bool {
    let state = state();
    state.render_fraction.is_some() && state.tick == tick
}

/// A display object's transform from before the movie frame `tick` first changed it.
#[derive(Clone, Copy)]
struct PreviousTransform {
    tick: u32,
    matrix: Matrix,
    color_transform: ColorTransform,
}

/// A blended transform, for the movie frame and fraction it was blended at. Rendering
/// reads an object's transform many times (bounds, layers, cached bitmaps), and blending
/// takes trigonometry.
#[derive(Clone, Copy)]
struct Blended {
    tick: u32,
    fraction: f32,
    matrix: Matrix,
    color_transform: ColorTransform,
}

/// What a display object needs for interpolation (in `DisplayObjectBase`).
pub struct InterpolationData {
    /// The movie frame in which the object was created or placed: it isn't blended then.
    born: Cell<u32>,
    previous: Cell<Option<PreviousTransform>>,
    blended: Cell<Option<Blended>>,
}

impl Default for InterpolationData {
    fn default() -> Self {
        Self {
            born: Cell::new(tick()),
            previous: Cell::new(None),
            blended: Cell::new(None),
        }
    }
}

/// A copy of a display object (an instance of a library symbol) is a new object.
impl Clone for InterpolationData {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl InterpolationData {
    /// The object was placed (again) in the current movie frame, so it isn't blended.
    pub fn reset(&self) {
        self.born.set(tick());
        self.previous.set(None);
    }

    /// Called before the object's transform changes.
    pub fn before_change(&self, matrix: Matrix, color_transform: ColorTransform) {
        let state = state();
        if !state.enabled
            || state.render_fraction.is_some()
            || self.born.get() == state.tick
            || self
                .previous
                .get()
                .is_some_and(|previous| previous.tick == state.tick)
        {
            return;
        }
        self.previous.set(Some(PreviousTransform {
            tick: state.tick,
            matrix,
            color_transform,
        }));
    }

    /// The blended transform, if the object changed in the current movie frame and a
    /// frame between movie frames is being drawn.
    fn blended(&self, matrix: Matrix, color_transform: ColorTransform) -> Option<Blended> {
        let state = state();
        let fraction = state.render_fraction?;
        let previous = self.previous.get()?;
        if previous.tick != state.tick {
            return None;
        }
        if let Some(blended) = self.blended.get()
            && blended.tick == state.tick
            && blended.fraction == fraction
        {
            return Some(blended);
        }
        let blended = Blended {
            tick: state.tick,
            fraction,
            matrix: blend_matrix(previous.matrix, matrix, fraction).unwrap_or(matrix),
            color_transform: blend_color_transform(
                previous.color_transform,
                color_transform,
                fraction,
            ),
        };
        self.blended.set(Some(blended));
        Some(blended)
    }

    /// The matrix to draw with: `current`, or blended with the previous one.
    pub fn matrix(&self, current: Matrix, color_transform: ColorTransform) -> Matrix {
        self.blended(current, color_transform)
            .map_or(current, |blended| blended.matrix)
    }

    /// The color transform to draw with: `current`, or blended with the previous one.
    pub fn color_transform(&self, matrix: Matrix, current: ColorTransform) -> ColorTransform {
        self.blended(matrix, current)
            .map_or(current, |blended| blended.color_transform)
    }
}

/// Blends two matrices by blending each axis's angle and length (so that rotations don't
/// shrink halfway) and the translation. `None` if they're too different to be motion.
fn blend_matrix(from: Matrix, to: Matrix, t: f32) -> Option<Matrix> {
    if from == to {
        return Some(to);
    }
    let lerp = |a: f32, b: f32| a + (b - a) * t;
    let (dx, dy) = (
        (to.tx - from.tx).to_pixels() as f32,
        (to.ty - from.ty).to_pixels() as f32,
    );
    if dx.abs() > MAX_MOVE || dy.abs() > MAX_MOVE {
        return None;
    }
    let axis = |x: f32, y: f32, x2: f32, y2: f32| -> Option<(f32, f32)> {
        let (length, length2) = (x.hypot(y), x2.hypot(y2));
        if length < 1e-6 || length2 < 1e-6 {
            return None;
        }
        let ratio = length2 / length;
        if !(1.0 / MAX_SCALE..=MAX_SCALE).contains(&ratio) {
            return None;
        }
        let (angle, angle2) = (y.atan2(x), y2.atan2(x2));
        let mut turn = angle2 - angle;
        if turn > std::f32::consts::PI {
            turn -= std::f32::consts::TAU;
        } else if turn < -std::f32::consts::PI {
            turn += std::f32::consts::TAU;
        }
        if turn.abs() > MAX_TURN {
            return None;
        }
        Some((angle + turn * t, lerp(length, length2)))
    };
    let (x_angle, x_length) = axis(from.a, from.b, to.a, to.b)?;
    let (y_angle, y_length) = axis(from.c, from.d, to.c, to.d)?;
    Some(Matrix {
        a: x_length * x_angle.cos(),
        b: x_length * x_angle.sin(),
        c: y_length * y_angle.cos(),
        d: y_length * y_angle.sin(),
        tx: Twips::new(lerp(from.tx.get() as f32, to.tx.get() as f32).round() as i32),
        ty: Twips::new(lerp(from.ty.get() as f32, to.ty.get() as f32).round() as i32),
    })
}

fn blend_color_transform(from: ColorTransform, to: ColorTransform, t: f32) -> ColorTransform {
    if from == to {
        return to;
    }
    let multiply =
        |a: Fixed8, b: Fixed8| Fixed8::from_f32(a.to_f32() + (b.to_f32() - a.to_f32()) * t);
    let add = |a: i16, b: i16| (a as f32 + (b as f32 - a as f32) * t).round() as i16;
    ColorTransform {
        r_multiply: multiply(from.r_multiply, to.r_multiply),
        g_multiply: multiply(from.g_multiply, to.g_multiply),
        b_multiply: multiply(from.b_multiply, to.b_multiply),
        a_multiply: multiply(from.a_multiply, to.a_multiply),
        r_add: add(from.r_add, to.r_add),
        g_add: add(from.g_add, to.g_add),
        b_add: add(from.b_add, to.b_add),
        a_add: add(from.a_add, to.a_add),
    }
}
