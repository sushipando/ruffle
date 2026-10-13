//! Reordering of commands so that complex blends can be processed in batches.
//!
//! A complex blend (e.g. Overlay or Multiply) needs a copy of everything drawn
//! below it, so the target's render pass must end before it, and a new pass
//! begins after it. Content with many complex blends spread over the screen
//! (e.g. many animated objects whose parts use Overlay shading)
//! ends up with hundreds of passes per frame.
//!
//! However, commands that don't overlap can be drawn in any order. This module
//! reorders a target's commands into "waves": every command whose dependencies
//! are already drawn, followed by a batch of complex blends that don't overlap each
//! other or anything still waiting to be drawn. The surface then handles each
//! batch with a single pass break instead of one per blend.
//!
//! Dependencies are determined conservatively from bounding boxes on a coarse
//! grid; anything that can't be bounded (masks, unbounded layers, 3D) acts as a
//! barrier that nothing is moved across. Since only non-overlapping commands are
//! ever swapped, the output is identical.

use crate::blend::{BlendType, ComplexBlend};
use crate::mesh::as_mesh;
use ruffle_render::commands::{Command, CommandList, RenderBlendMode};
use ruffle_render::matrix::Matrix;

/// The size (in pixels) of the cells of the dependency grid.
const CELL_SIZE: f32 = 32.0;

/// Extra space around bounds, for anti-aliasing and pixel snapping.
const PADDING: f32 = 2.0;

#[derive(Clone, Copy)]
enum Kind {
    /// Draws within the given bounds.
    Draw(Rect),
    /// A complex blend affecting the given bounds.
    ComplexBlend(Rect),
    /// Can't be bounded, so nothing may be moved across it.
    Barrier,
}

/// Bounds in cells: `(x0, y0, x1, y1)`, exclusive end. Empty if outside the target.
#[derive(Clone, Copy)]
struct Rect {
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
}

struct Grid {
    columns: usize,
    rows: usize,
    origin: (f32, f32),
    /// For each cell, the earliest wave that later commands touching it may be in.
    min_wave: Vec<u32>,
}

impl Grid {
    fn new(origin: (i32, i32), width: u32, height: u32) -> Self {
        let columns = (width as f32 / CELL_SIZE).ceil().max(1.0) as usize;
        let rows = (height as f32 / CELL_SIZE).ceil().max(1.0) as usize;
        Self {
            columns,
            rows,
            origin: (origin.0 as f32, origin.1 as f32),
            min_wave: vec![0; columns * rows],
        }
    }

    /// Converts pixel bounds (`x0, y0, x1, y1`) into cells.
    fn rect(&self, (x0, y0, x1, y1): (f32, f32, f32, f32)) -> Rect {
        let to_cell = |v: f32, origin: f32, max: usize| {
            (((v - origin) / CELL_SIZE).floor().max(0.0) as usize).min(max)
        };
        Rect {
            x0: to_cell(x0 - PADDING, self.origin.0, self.columns),
            y0: to_cell(y0 - PADDING, self.origin.1, self.rows),
            x1: to_cell(x1 + PADDING + CELL_SIZE, self.origin.0, self.columns),
            y1: to_cell(y1 + PADDING + CELL_SIZE, self.origin.1, self.rows),
        }
    }

    fn rows(&self, rect: Rect) -> impl Iterator<Item = std::ops::Range<usize>> + '_ {
        (rect.y0..rect.y1).map(move |y| y * self.columns + rect.x0..y * self.columns + rect.x1)
    }

    /// The earliest wave a command touching `rect` may be in.
    fn earliest_wave(&self, rect: Rect) -> u32 {
        self.rows(rect)
            .flat_map(|row| self.min_wave[row].iter().copied())
            .max()
            .unwrap_or(0)
    }

    /// Requires later commands touching `rect` to be in `wave` or later.
    fn require(&mut self, rect: Rect, wave: u32) {
        let rows: Vec<_> = self.rows(rect).collect();
        for row in rows {
            for cell in &mut self.min_wave[row] {
                *cell = (*cell).max(wave);
            }
        }
    }
}

impl Rect {
    const EMPTY: Rect = Rect {
        x0: 0,
        y0: 0,
        x1: 0,
        y1: 0,
    };

    fn is_empty(&self) -> bool {
        self.x0 >= self.x1 || self.y0 >= self.y1
    }

    fn union(self, other: Rect) -> Rect {
        if self.is_empty() {
            return other;
        }
        if other.is_empty() {
            return self;
        }
        Rect {
            x0: self.x0.min(other.x0),
            y0: self.y0.min(other.y0),
            x1: self.x1.max(other.x1),
            y1: self.y1.max(other.y1),
        }
    }
}

/// The axis-aligned bounds of the unit square transformed by `matrix`, scaled by `(w, h)`.
fn transformed_box(matrix: &Matrix, w: f32, h: f32) -> (f32, f32, f32, f32) {
    transformed_rect(matrix, (0.0, 0.0, w, h))
}

fn transformed_rect(
    matrix: &Matrix,
    (x0, y0, x1, y1): (f32, f32, f32, f32),
) -> (f32, f32, f32, f32) {
    let tx = matrix.tx.to_pixels() as f32;
    let ty = matrix.ty.to_pixels() as f32;
    let mut bounds = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for (x, y) in [(x0, y0), (x1, y0), (x0, y1), (x1, y1)] {
        let px = matrix.a * x + matrix.c * y + tx;
        let py = matrix.b * x + matrix.d * y + ty;
        bounds = (
            bounds.0.min(px),
            bounds.1.min(py),
            bounds.2.max(px),
            bounds.3.max(py),
        );
    }
    bounds
}

fn is_complex(blend_mode: &RenderBlendMode) -> bool {
    match blend_mode {
        RenderBlendMode::Builtin(mode) => matches!(
            BlendType::from(RenderBlendMode::Builtin(*mode)),
            BlendType::Complex(
                ComplexBlend::Multiply
                    | ComplexBlend::Lighten
                    | ComplexBlend::Darken
                    | ComplexBlend::Difference
                    | ComplexBlend::Invert
                    | ComplexBlend::Overlay
                    | ComplexBlend::HardLight
            )
        ),
        RenderBlendMode::Shader(_) => false,
    }
}

fn classify(command: &Command, grid: &Grid) -> Kind {
    let simple_transform = |transform: &ruffle_render::transform::Transform| {
        transform.tz == 0.0 && transform.perspective_projection.is_none()
    };
    match command {
        Command::RenderShape { shape, transform } if simple_transform(transform) => {
            match as_mesh(shape).bounds {
                Some(bounds) => Kind::Draw(grid.rect(transformed_rect(&transform.matrix, bounds))),
                // Nothing to draw.
                None => Kind::Draw(Rect::EMPTY),
            }
        }
        Command::RenderBitmap {
            transform, region, ..
        } if simple_transform(transform) => Kind::Draw(grid.rect(transformed_box(
            &transform.matrix,
            region.width() as f32,
            region.height() as f32,
        ))),
        Command::DrawRect { matrix, .. }
        | Command::DrawLine { matrix, .. }
        | Command::DrawLineRect { matrix, .. } => {
            Kind::Draw(grid.rect(transformed_box(matrix, 1.0, 1.0)))
        }
        Command::BlendBounded(_, blend_mode, bounds) if bounds.is_valid() => {
            let bounds = grid.rect((
                bounds.x_min.to_pixels() as f32,
                bounds.y_min.to_pixels() as f32,
                bounds.x_max.to_pixels() as f32,
                bounds.y_max.to_pixels() as f32,
            ));
            if is_complex(blend_mode) {
                Kind::ComplexBlend(bounds)
            } else if matches!(blend_mode, RenderBlendMode::Builtin(_)) {
                Kind::Draw(bounds)
            } else {
                Kind::Barrier
            }
        }
        _ => Kind::Barrier,
    }
}

/// A command, or a whole masked group (from a `PushMask` to its `PopMask`), which
/// restores the stencil buffer when it ends and so can be moved as one unit.
struct Unit {
    commands: Vec<Command>,
    kind: Kind,
}

/// Groups commands into units.
fn units(commands: Vec<Command>, grid: &Grid) -> Vec<Unit> {
    let mut units = Vec::with_capacity(commands.len());
    let mut group: Option<(Vec<Command>, Kind, u32)> = None;
    for command in commands {
        let kind = classify(&command, grid);
        let is_mask_command = matches!(
            command,
            Command::PushMask | Command::ActivateMask | Command::DeactivateMask | Command::PopMask
        );
        let mask_depth_change = match command {
            Command::PushMask => 1,
            Command::PopMask => -1,
            _ => 0,
        };
        match &mut group {
            None if mask_depth_change == 1 => {
                group = Some((vec![command], Kind::Draw(Rect::EMPTY), 1));
            }
            None => units.push(Unit {
                commands: vec![command],
                kind,
            }),
            Some((commands, group_kind, depth)) => {
                commands.push(command);
                if !is_mask_command {
                    // Masked content is drawn normally, so complex blends in it
                    // just count as draws.
                    *group_kind = match (*group_kind, kind) {
                        (Kind::Draw(a), Kind::Draw(b) | Kind::ComplexBlend(b)) => {
                            Kind::Draw(a.union(b))
                        }
                        _ => Kind::Barrier,
                    };
                }
                *depth = depth.saturating_add_signed(mask_depth_change);
                if *depth == 0 {
                    let (commands, kind, _) = group.take().expect("group exists");
                    units.push(Unit { commands, kind });
                }
            }
        }
    }
    if let Some((commands, _, _)) = group {
        // Unbalanced masks; don't move anything across them.
        units.push(Unit {
            commands,
            kind: Kind::Barrier,
        });
    }
    units
}

/// Reorders `commands` (drawn into a target covering `width`x`height` pixels at
/// `origin`) so that complex blends that can share a pass break are adjacent.
///
/// Every unit is assigned the earliest "wave" that keeps it after everything it
/// overlaps: draws may share a wave with what they depend on, as each wave's
/// draws are output (in their original order) before its complex blends, but
/// anything that depends on a complex blend must be in a later wave. Then the
/// waves are output in order. Units that overlap thus keep their order, and the
/// complex blends of a wave never overlap each other.
pub fn schedule(commands: CommandList, origin: (i32, i32), width: u32, height: u32) -> CommandList {
    let num_complex = commands
        .commands
        .iter()
        .filter(|command| matches!(command, Command::BlendBounded(_, mode, _) if is_complex(mode)))
        .count();
    if num_complex < 2 {
        return commands;
    }

    let start = std::time::Instant::now();
    let mut grid = Grid::new(origin, width, height);
    let units = units(commands.commands, &grid);

    // The earliest wave anything may be in (raised by barriers),
    // and the earliest wave a barrier may be in (after everything so far).
    let mut floor = 0;
    let mut barrier_wave = 0;
    let mut waves: Vec<(Vec<Unit>, Vec<Unit>)> = vec![];
    for unit in units {
        let (wave, is_blend) = match unit.kind {
            Kind::Draw(rect) => {
                let wave = grid.earliest_wave(rect).max(floor);
                grid.require(rect, wave);
                barrier_wave = barrier_wave.max(wave);
                (wave, false)
            }
            Kind::ComplexBlend(rect) => {
                let wave = grid.earliest_wave(rect).max(floor);
                grid.require(rect, wave + 1);
                barrier_wave = barrier_wave.max(wave + 1);
                (wave, true)
            }
            Kind::Barrier => {
                floor = barrier_wave;
                (barrier_wave, false)
            }
        };
        let wave = wave as usize;
        if waves.len() <= wave {
            waves.resize_with(wave + 1, Default::default);
        }
        if is_blend {
            waves[wave].1.push(unit);
        } else {
            waves[wave].0.push(unit);
        }
    }

    crate::perf_counters::count_named_n("sched-waves", waves.len() as u64);
    crate::perf_counters::count_named_n("sched-complex", num_complex as u64);
    crate::perf_counters::count_named_n("sched-us", start.elapsed().as_micros() as u64);

    let mut result = CommandList::new();
    result.commands = waves
        .into_iter()
        .flat_map(|(draws, blends)| draws.into_iter().chain(blends))
        .flat_map(|unit| unit.commands)
        .collect();
    result
}
