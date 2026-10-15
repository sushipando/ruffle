//! 9-slice scaling (`DisplayObject.scale9Grid`, or a `DefineScalingGrid` tag).
//!
//! When a clip with a scaling grid is scaled without rotation or skew, Flash draws its own
//! drawing and the shapes directly inside it (or a shape with a scaling grid draws itself)
//! so that their parts outside the grid keep
//! their size: the corners aren't scaled, the parts along the sides are only scaled along
//! their side, and the center is scaled normally. Each point of an outline (and each curve
//! control point) is moved on its own, per axis: left of the grid it keeps its distance to
//! the clip's scaled left edge, right of it its distance to the scaled right edge, and
//! inside the grid it's spread evenly between the two (the same for y). Line widths keep
//! their size, and each gradient or bitmap fill is stretched from the bounds of its path to
//! the bounds of the sliced path (so a fill in a corner keeps its size, and one in the
//! center fills the scaled center). Nested clips, text and everything else are scaled
//! normally, as is everything when the clip is rotated. Hit tests and bounds use the shapes
//! as they are, scaled normally.

use crate::display_object::{BoundsMode, DisplayObject, TDisplayObject};
use ruffle_render::backend::ShapeHandle;
use ruffle_render::matrix::Matrix;
use ruffle_render::shape_utils::{DistilledShape, DrawCommand, DrawPath, FillRule};
use swf::{CharacterId, FillStyle, LineStyle, Point, Rectangle, Twips};

/// How a clip with a scaling grid draws its own drawing and its child shapes (or a shape
/// with a scaling grid draws itself).
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Slicing {
    /// The scaling grid, in the object's coordinates.
    grid: Rectangle<Twips>,
    /// The bounds of the object's content, in its coordinates.
    bounds: Rectangle<Twips>,
    /// The object's scale.
    scale_x: f32,
    scale_y: f32,
}

impl Slicing {
    /// The slicing of `object`, if it has a scaling grid and is scaled without rotation or
    /// skew.
    pub fn of(object: DisplayObject<'_>) -> Option<Self> {
        let grid = object.scaling_grid();
        if !grid.is_valid() || grid.width() <= Twips::ZERO || grid.height() <= Twips::ZERO {
            return None;
        }
        let matrix = object.base().matrix();
        if matrix.b != 0.0 || matrix.c != 0.0 || (matrix.a == 1.0 && matrix.d == 1.0) {
            return None;
        }
        if matrix.a == 0.0 || matrix.d == 0.0 {
            return None;
        }
        // Without strokes (in the AIR runtime, a stroked box keeps its exact place and size).
        let bounds = object.bounds(BoundsMode::ScriptWithoutStrokes);
        if !bounds.is_valid() {
            return None;
        }
        Some(Self {
            grid,
            bounds,
            scale_x: matrix.a,
            scale_y: matrix.d,
        })
    }

    /// Where `value` goes along one axis.
    fn map(value: f64, bounds: (f64, f64), grid: (f64, f64), scale: f64) -> f64 {
        let (b0, b1) = bounds;
        let (g0, g1) = grid;
        let left = b0 * scale + (g0 - b0);
        let right = b1 * scale - (b1 - g1);
        if value <= g0 {
            b0 * scale + (value - b0)
        } else if value >= g1 {
            b1 * scale - (b1 - value)
        } else {
            left + (value - g0) * (right - left) / (g1 - g0)
        }
    }

    /// Where a point in the clip's coordinates goes, in its scaled coordinates.
    fn map_point(&self, point: Point<Twips>) -> Point<Twips> {
        let x = Self::map(
            point.x.get() as f64,
            (
                self.bounds.x_min.get() as f64,
                self.bounds.x_max.get() as f64,
            ),
            (self.grid.x_min.get() as f64, self.grid.x_max.get() as f64),
            self.scale_x as f64,
        );
        let y = Self::map(
            point.y.get() as f64,
            (
                self.bounds.y_min.get() as f64,
                self.bounds.y_max.get() as f64,
            ),
            (self.grid.y_min.get() as f64, self.grid.y_max.get() as f64),
            self.scale_y as f64,
        );
        Point::new(Twips::new(x.round() as i32), Twips::new(y.round() as i32))
    }

    /// The matrix to draw sliced content with, given the clip's world matrix: the same,
    /// without the clip's scale (the sliced points include it).
    pub fn draw_matrix(&self, clip_world: &Matrix) -> Matrix {
        *clip_world
            * Matrix {
                a: 1.0 / self.scale_x,
                d: 1.0 / self.scale_y,
                ..Matrix::IDENTITY
            }
    }

    /// The matrix that stretches the bounds of the points of `commands` (in the clip's
    /// coordinates) to the bounds of their sliced points, for a fill of that path.
    fn fill_stretch(&self, commands: &[DrawCommand]) -> Matrix {
        // The points of the segments (a move that isn't followed by one doesn't count).
        let mut cursor = Point::ZERO;
        let mut points = commands
            .iter()
            .flat_map(|command| {
                let start = cursor;
                let (a, b, c) = match *command {
                    DrawCommand::MoveTo(p) => {
                        cursor = p;
                        return [None, None, None, None];
                    }
                    DrawCommand::LineTo(p) => (p, None, None),
                    DrawCommand::QuadraticCurveTo { control, anchor } => {
                        (anchor, Some(control), None)
                    }
                    DrawCommand::CubicCurveTo {
                        control_a,
                        control_b,
                        anchor,
                    } => (anchor, Some(control_a), Some(control_b)),
                };
                cursor = a;
                [Some(start), Some(a), b, c]
            })
            .flatten();
        let Some(first) = points.next() else {
            return Matrix::IDENTITY;
        };
        let (min, max) = points.fold((first, first), |(min, max), p| {
            (
                Point::new(min.x.min(p.x), min.y.min(p.y)),
                Point::new(max.x.max(p.x), max.y.max(p.y)),
            )
        });
        let (sliced_min, sliced_max) = (self.map_point(min), self.map_point(max));
        let axis = |min: Twips, max: Twips, sliced_min: Twips, sliced_max: Twips| {
            let scale = if max > min {
                (sliced_max - sliced_min).get() as f32 / (max - min).get() as f32
            } else {
                1.0
            };
            let offset = sliced_min.get() as f32 - scale * min.get() as f32;
            (scale, Twips::new(offset.round() as i32))
        };
        let (a, tx) = axis(min.x, max.x, sliced_min.x, sliced_max.x);
        let (d, ty) = axis(min.y, max.y, sliced_min.y, sliced_max.y);
        Matrix {
            a,
            d,
            tx,
            ty,
            ..Matrix::IDENTITY
        }
    }

    /// Slices a shape drawn with `matrix` in the clip (the identity for the clip's own
    /// drawing).
    pub fn slice(&self, shape: DistilledShape<'_>, matrix: &Matrix) -> SlicedShape {
        // Line widths keep their size in the clip, but are still scaled with the shape.
        let line_scale = ((matrix.a * matrix.d - matrix.b * matrix.c).abs()).sqrt();
        let point = |point: Point<Twips>| self.map_point(point);
        let command = |command: DrawCommand| match command {
            DrawCommand::MoveTo(p) => DrawCommand::MoveTo(point(p)),
            DrawCommand::LineTo(p) => DrawCommand::LineTo(point(p)),
            DrawCommand::QuadraticCurveTo { control, anchor } => DrawCommand::QuadraticCurveTo {
                control: point(control),
                anchor: point(anchor),
            },
            DrawCommand::CubicCurveTo {
                control_a,
                control_b,
                anchor,
            } => DrawCommand::CubicCurveTo {
                control_a: point(control_a),
                control_b: point(control_b),
                anchor: point(anchor),
            },
        };
        // The path's commands in the clip's coordinates, and the matrix for its fill (the
        // shape's matrix, then the stretch of the path).
        let place = |commands: Vec<DrawCommand>| {
            let commands: Vec<_> = commands
                .into_iter()
                .map(|command| transform_command(command, matrix))
                .collect();
            let fill_matrix = self.fill_stretch(&commands) * *matrix;
            (commands.into_iter().map(command).collect(), fill_matrix)
        };

        let mut fills = Vec::new();
        let mut lines = Vec::new();
        let paths = shape
            .paths
            .into_iter()
            .map(|path| match path {
                DrawPath::Fill {
                    style,
                    commands,
                    winding_rule,
                } => {
                    let (commands, fill_matrix) = place(commands);
                    fills.push(transform_fill(style, &fill_matrix));
                    SlicedPath::Fill {
                        style: fills.len() - 1,
                        commands,
                        winding_rule,
                    }
                }
                DrawPath::Stroke {
                    style,
                    is_closed,
                    commands,
                } => {
                    let (commands, fill_matrix) = place(commands);
                    let width =
                        Twips::new((style.width().get() as f32 * line_scale).round() as i32);
                    lines.push(
                        style
                            .clone()
                            .with_width(width)
                            .with_fill_style(transform_fill(style.fill_style(), &fill_matrix)),
                    );
                    SlicedPath::Stroke {
                        style: lines.len() - 1,
                        is_closed,
                        commands,
                    }
                }
            })
            .collect();

        let bounds = |rect: Rectangle<Twips>| {
            if !rect.is_valid() {
                return rect;
            }
            let rect = *matrix * rect;
            let min = self.map_point(Point::new(rect.x_min, rect.y_min));
            let max = self.map_point(Point::new(rect.x_max, rect.y_max));
            Rectangle {
                x_min: min.x.min(max.x),
                x_max: min.x.max(max.x),
                y_min: min.y.min(max.y),
                y_max: min.y.max(max.y),
            }
        };
        SlicedShape {
            id: shape.id,
            shape_bounds: bounds(shape.shape_bounds),
            edge_bounds: bounds(shape.edge_bounds),
            fills,
            lines,
            paths,
        }
    }
}

fn transform_command(command: DrawCommand, matrix: &Matrix) -> DrawCommand {
    match command {
        DrawCommand::MoveTo(p) => DrawCommand::MoveTo(*matrix * p),
        DrawCommand::LineTo(p) => DrawCommand::LineTo(*matrix * p),
        DrawCommand::QuadraticCurveTo { control, anchor } => DrawCommand::QuadraticCurveTo {
            control: *matrix * control,
            anchor: *matrix * anchor,
        },
        DrawCommand::CubicCurveTo {
            control_a,
            control_b,
            anchor,
        } => DrawCommand::CubicCurveTo {
            control_a: *matrix * control_a,
            control_b: *matrix * control_b,
            anchor: *matrix * anchor,
        },
    }
}

fn transform_fill(style: &FillStyle, matrix: &Matrix) -> FillStyle {
    let transform = |m: swf::Matrix| swf::Matrix::from(*matrix * Matrix::from(m));
    match style {
        FillStyle::Color(color) => FillStyle::Color(*color),
        FillStyle::LinearGradient(gradient) => {
            let mut gradient = gradient.clone();
            gradient.matrix = transform(gradient.matrix);
            FillStyle::LinearGradient(gradient)
        }
        FillStyle::RadialGradient(gradient) => {
            let mut gradient = gradient.clone();
            gradient.matrix = transform(gradient.matrix);
            FillStyle::RadialGradient(gradient)
        }
        FillStyle::FocalGradient {
            gradient,
            focal_point,
        } => {
            let mut gradient = gradient.clone();
            gradient.matrix = transform(gradient.matrix);
            FillStyle::FocalGradient {
                gradient,
                focal_point: *focal_point,
            }
        }
        FillStyle::Bitmap {
            id,
            matrix: bitmap_matrix,
            is_smoothed,
            is_repeating,
        } => FillStyle::Bitmap {
            id: *id,
            matrix: transform(*bitmap_matrix),
            is_smoothed: *is_smoothed,
            is_repeating: *is_repeating,
        },
    }
}

/// A sliced shape, which owns its styles.
pub struct SlicedShape {
    id: CharacterId,
    shape_bounds: Rectangle<Twips>,
    edge_bounds: Rectangle<Twips>,
    fills: Vec<FillStyle>,
    lines: Vec<LineStyle>,
    paths: Vec<SlicedPath>,
}

enum SlicedPath {
    Fill {
        style: usize,
        commands: Vec<DrawCommand>,
        winding_rule: FillRule,
    },
    Stroke {
        style: usize,
        is_closed: bool,
        commands: Vec<DrawCommand>,
    },
}

impl SlicedShape {
    pub fn distilled(&self) -> DistilledShape<'_> {
        DistilledShape {
            paths: self
                .paths
                .iter()
                .map(|path| match path {
                    SlicedPath::Fill {
                        style,
                        commands,
                        winding_rule,
                    } => DrawPath::Fill {
                        style: &self.fills[*style],
                        commands: commands.clone(),
                        winding_rule: *winding_rule,
                    },
                    SlicedPath::Stroke {
                        style,
                        is_closed,
                        commands,
                    } => DrawPath::Stroke {
                        style: &self.lines[*style],
                        is_closed: *is_closed,
                        commands: commands.clone(),
                    },
                })
                .collect(),
            shape_bounds: self.shape_bounds,
            edge_bounds: self.edge_bounds,
            id: self.id,
        }
    }
}

/// The last sliced tessellation of a shape, and what it was made for.
#[derive(Clone, Debug, Default)]
pub struct SliceCache(Option<(SliceKey, ShapeHandle)>);

#[derive(Clone, Copy, PartialEq, Debug)]
struct SliceKey {
    slicing: Slicing,
    matrix: Matrix,
    /// The tessellation scale, rounded (see `SliceCache::get_or_insert`).
    scale: i32,
}

impl SliceCache {
    /// The handle for `slicing` of a shape drawn with `matrix` in the clip, tessellated at
    /// about `scale`, made with `make` unless the last one fits.
    pub fn get_or_insert(
        &mut self,
        slicing: Slicing,
        matrix: Matrix,
        scale: f32,
        make: impl FnOnce() -> ShapeHandle,
    ) -> ShapeHandle {
        let key = SliceKey {
            slicing,
            matrix,
            scale: (scale * 16.0).round() as i32,
        };
        match &self.0 {
            Some((cached, handle)) if *cached == key => handle.clone(),
            _ => {
                let handle = make();
                self.0 = Some((key, handle.clone()));
                handle
            }
        }
    }

    pub fn clear(&mut self) {
        self.0 = None;
    }
}

/// The tessellation scale for drawing with `matrix` (as `Graphic` computes it).
pub fn tessellation_scale(matrix: &Matrix) -> f32 {
    let scale_x = f32::abs(matrix.a + matrix.c);
    let scale_y = f32::abs(matrix.b + matrix.d);
    ((scale_x * scale_x + scale_y * scale_y) / 2.0).sqrt()
}
