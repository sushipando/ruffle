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

    /// The earliest wave a command touching `rect` may be in.
    #[inline]
    fn earliest_wave(&self, rect: Rect) -> u32 {
        let mut wave = 0;
        for y in rect.y0..rect.y1 {
            let row = y * self.columns;
            for &cell in &self.min_wave[row + rect.x0..row + rect.x1] {
                wave = wave.max(cell);
            }
        }
        wave
    }

    /// Requires later commands touching `rect` to be in `wave` or later.
    #[inline]
    fn require(&mut self, rect: Rect, wave: u32) {
        for y in rect.y0..rect.y1 {
            let row = y * self.columns;
            for cell in &mut self.min_wave[row + rect.x0..row + rect.x1] {
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

fn is_complex(blend_mode: &RenderBlendMode, opaque: bool) -> bool {
    match blend_mode {
        // Done as a trivial blend on opaque targets (see `TrivialBlend::MultiplyOpaque`).
        RenderBlendMode::Builtin(swf::BlendMode::Multiply)
            if opaque
                && ruffle_render::perf_flags::enabled(
                    ruffle_render::perf_flags::PerfFlag::OpaqueMultiply,
                ) =>
        {
            false
        }
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

fn classify(command: &Command, grid: &Grid, opaque: bool) -> Kind {
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
            if is_complex(blend_mode, opaque) {
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
fn units(commands: Vec<Command>, grid: &Grid, opaque: bool) -> Vec<Unit> {
    let mut units = Vec::with_capacity(commands.len());
    let mut group: Option<(Vec<Command>, Kind, u32)> = None;
    for command in commands {
        let kind = classify(&command, grid, opaque);
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
pub fn schedule(
    commands: CommandList,
    origin: (i32, i32),
    width: u32,
    height: u32,
    opaque: bool,
) -> CommandList {
    let num_complex = commands
        .commands
        .iter()
        .filter(|command| matches!(command, Command::BlendBounded(_, mode, _) if is_complex(mode, opaque)))
        .count();
    if num_complex < 2 {
        return commands;
    }

    let start = std::time::Instant::now();
    let mut grid = Grid::new(origin, width, height);
    let units = units(commands.commands, &grid, opaque);

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

/// Coarse pipeline bucket for grouping non-overlapping draws when Metal framebuffer-fetch
/// blends are active. Grouping identical blend & fill pipelines cuts redundant `set_pipeline`
/// / `set_bind_group` switches and clusters Metal tile-fetch draws together.
fn pipeline_bucket(unit: &Unit, opaque: bool) -> u8 {
    if unit.commands.len() != 1 {
        return 3;
    }
    match &unit.commands[0] {
        Command::DrawRect { .. } => 0,
        Command::RenderShape { shape, .. } => {
            let mesh = as_mesh(shape);
            match mesh.draws.first().map(|d| &d.draw_type) {
                Some(crate::mesh::DrawType::Color) if mesh.draws.len() == 1 => 0,
                Some(crate::mesh::DrawType::Gradient { .. }) if mesh.draws.len() == 1 => 1,
                Some(crate::mesh::DrawType::Bitmap { .. }) if mesh.draws.len() == 1 => 2,
                _ => 3,
            }
        }
        Command::RenderBitmap { .. } => 2,
        Command::DrawLine { .. } | Command::DrawLineRect { .. } => 4,
        Command::BlendBounded(_, mode, _) => match BlendType::from(mode.clone()) {
            BlendType::Trivial(t) => 10 + t as u8,
            BlendType::Complex(ComplexBlend::Multiply)
                if opaque
                    && ruffle_render::perf_flags::enabled(
                        ruffle_render::perf_flags::PerfFlag::OpaqueMultiply,
                    ) =>
            {
                10 + crate::blend::TrivialBlend::MultiplyOpaque as u8
            }
            BlendType::Complex(c) => 20 + c as u8,
            BlendType::Shader(_) => 30,
        },
        _ => 31,
    }
}

/// Dependency-preserving pipeline grouping for targets with Metal framebuffer-fetch blends.
///
/// Assigns each unit to the earliest layer where it does not overlap any other unit in that
/// layer (using the same padded 32px dependency grid as `schedule`), then stably sorts each
/// independent layer by pipeline bucket. Since no two units in the same layer overlap, the
/// rendered output is bit-for-bit identical while consecutive draws share pipelines.
pub fn group_pipelines(
    commands: CommandList,
    origin: (i32, i32),
    width: u32,
    height: u32,
    opaque: bool,
) -> CommandList {
    if commands.commands.len() < 16 {
        return commands;
    }
    let mut grid = Grid::new(origin, width, height);
    let units = units(commands.commands, &grid, opaque);
    if units.len() < 16 {
        let mut result = CommandList::new();
        result.commands = units.into_iter().flat_map(|u| u.commands).collect();
        return result;
    }

    let mut floor = 0u32;
    let mut barrier_wave = 0u32;
    let mut layers: Vec<Vec<(u8, Unit)>> = Vec::new();
    for unit in units {
        let (wave, bucket) = match unit.kind {
            Kind::Draw(rect) | Kind::ComplexBlend(rect) => {
                let wave = grid.earliest_wave(rect).max(floor);
                // Require `wave + 1` so no two units in the same layer can overlap.
                grid.require(rect, wave + 1);
                barrier_wave = barrier_wave.max(wave + 1);
                let bucket = pipeline_bucket(&unit, opaque);
                (wave, bucket)
            }
            Kind::Barrier => {
                let wave = barrier_wave;
                barrier_wave += 1;
                floor = barrier_wave;
                (wave, 255)
            }
        };
        let wave = wave as usize;
        if layers.len() <= wave {
            layers.resize_with(wave + 1, Vec::new);
        }
        layers[wave].push((bucket, unit));
    }

    let mut result = CommandList::new();
    result.commands.reserve(layers.iter().map(Vec::len).sum());
    for mut layer in layers {
        if layer.len() > 1 {
            layer.sort_by_key(|(bucket, _)| *bucket);
        }
        for (_, unit) in layer {
            result.commands.extend(unit.commands);
        }
    }
    result
}
