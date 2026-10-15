use crate::avm1::Object as Avm1Object;
use crate::avm2::{
    Activation as Avm2Activation, Avm2, ClassObject as Avm2ClassObject,
    StageObject as Avm2StageObject,
};
use crate::context::{RenderContext, UpdateContext};
use crate::display_object::scale9::{SliceCache, Slicing, tessellation_scale};
use crate::display_object::{BoundsMode, DisplayObjectBase};
use crate::drawing::Drawing;
use crate::library::MovieLibrarySource;
use crate::prelude::*;
use crate::tag_utils::SwfMovie;
use crate::tessellation_cache::TessellationCache;
use crate::vminterface::Instantiator;
use core::fmt;
use gc_arena::barrier::unlock;
use gc_arena::lock::Lock;
use gc_arena::{Collect, Gc, Mutation};
use ruffle_common::utils::HasPrefixField;
use ruffle_render::backend::ShapeHandle;
use ruffle_render::commands::CommandHandler;
use ruffle_render::transform::Transform;
use std::cell::{OnceCell, RefCell, RefMut};
use std::ops::Range;
use std::rc::Rc;

#[derive(Clone, Collect, Copy)]
#[collect(no_drop)]
pub struct Graphic<'gc>(Gc<'gc, GraphicData<'gc>>);

impl fmt::Debug for Graphic<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Graphic")
            .field("ptr", &Gc::as_ptr(self.0))
            .finish()
    }
}

#[derive(Clone, Collect, HasPrefixField)]
#[collect(no_drop)]
#[repr(C, align(8))]
pub struct GraphicData<'gc> {
    base: DisplayObjectBase<'gc>,
    shared: Lock<Gc<'gc, GraphicShared<'gc>>>,
    class: Lock<Option<Avm2ClassObject<'gc>>>,
    avm2_object: Lock<Option<Avm2StageObject<'gc>>>,
    /// This is lazily allocated on demand, to make `GraphicData` smaller in the common case.
    #[collect(require_static)]
    drawing: OnceCell<Box<RefCell<Drawing>>>,
    /// The last 9-slice scaled tessellation of the shape (see `scale9`).
    #[collect(require_static)]
    sliced: RefCell<SliceCache>,
}

impl<'gc> Graphic<'gc> {
    /// Renders this shape with 9-slice scaling: `matrix` is how it's placed in the object
    /// with the scaling grid, and `transform` is what to draw the sliced shape with.
    fn render_sliced(
        self,
        context: &mut RenderContext<'_, 'gc>,
        slicing: Slicing,
        matrix: Matrix,
        transform: Transform,
    ) {
        if let Some(drawing) = self.0.drawing.get() {
            drawing
                .borrow()
                .render_sliced(context, slicing, matrix, transform);
        } else if !self.0.shared.get().is_empty {
            let scale = tessellation_scale(&transform.matrix);
            let handle = self
                .0
                .sliced
                .borrow_mut()
                .get_or_insert(slicing, matrix, scale, || {
                    let shared = self.0.shared.get();
                    let library = context
                        .library
                        .library_for_movie(shared.movie)
                        .expect("Graphic's movie has a library");
                    let shape = shared.shape.get(shared.movie);
                    let sliced = slicing.slice((&*shape).into(), &matrix);
                    let handle = context.renderer.register_shape_with_scale(
                        sliced.distilled(),
                        &MovieLibrarySource { library },
                        scale,
                    );
                    drop(shape);
                    shared.shape.release();
                    handle
                });
            context.commands.render_shape(handle, transform);
        }
    }

    /// Construct a `Graphic` from it's associated `Shape` tag.
    ///
    /// `source` is where the tag's body is in the movie's data, so that the shape can be
    /// parsed again after its records are released (see `ShapeRecords`).
    pub fn from_swf_tag(
        context: &mut UpdateContext<'gc>,
        swf_shape: swf::Shape,
        movie: SwfMovie<'gc>,
        source: Option<Range<usize>>,
    ) -> Self {
        let shared = GraphicShared {
            id: swf_shape.id,
            shape_bounds: swf_shape.shape_bounds,
            edge_bounds: swf_shape.edge_bounds,
            is_empty: swf_shape.shape.is_empty(),
            shape: ShapeRecords::new(swf_shape, source),
            movie,
            scaled_handle: RefCell::new(TessellationCache::new()),
        };
        // Most shapes are never drawn (other frames, hidden parts of symbols), and drawn
        // ones are parsed again when they're tessellated.
        shared.shape.release();

        Graphic(Gc::new(
            context.gc(),
            GraphicData {
                base: Default::default(),
                shared: Lock::new(Gc::new(context.gc(), shared)),
                class: Lock::new(None),
                avm2_object: Lock::new(None),
                drawing: OnceCell::new(),
                sliced: Default::default(),
            },
        ))
    }

    /// Construct an empty `Graphic`.
    pub fn empty(context: &mut UpdateContext<'gc>) -> Self {
        let shared = GraphicShared {
            id: 0,
            shape_bounds: Default::default(),
            edge_bounds: Default::default(),
            is_empty: true,
            shape: ShapeRecords::new(
                swf::Shape {
                    version: 32,
                    id: 0,
                    shape_bounds: Default::default(),
                    edge_bounds: Default::default(),
                    flags: swf::ShapeFlag::empty(),
                    styles: swf::ShapeStyles {
                        fill_styles: Vec::new(),
                        line_styles: Vec::new(),
                    },
                    shape: Vec::new(),
                },
                None,
            ),
            movie: *context.root_swf,
            scaled_handle: RefCell::new(TessellationCache::new()),
        };

        Graphic(Gc::new(
            context.gc(),
            GraphicData {
                base: Default::default(),
                shared: Lock::new(Gc::new(context.gc(), shared)),
                class: Lock::new(None),
                avm2_object: Lock::new(None),
                drawing: OnceCell::new(),
                sliced: Default::default(),
            },
        ))
    }

    pub fn instantiate(self, mc: &Mutation<'gc>) -> Self {
        Self(Gc::new(mc, (*self.0).clone()))
    }

    pub fn drawing_mut(&self) -> RefMut<'_, Drawing> {
        self.0.drawing.get_or_init(Default::default).borrow_mut()
    }

    pub fn set_avm2_class(self, mc: &Mutation<'gc>, class: Avm2ClassObject<'gc>) {
        unlock!(Gc::write(mc, self.0), GraphicData, class).set(Some(class));
    }

    fn set_shared(self, mc: &Mutation<'gc>, shared: Gc<'gc, GraphicShared<'gc>>) {
        unlock!(Gc::write(mc, self.0), GraphicData, shared).set(shared);
        self.0.sliced.borrow_mut().clear();
    }

    /// Returns the best shape handle for the current scale, tessellating if necessary.
    ///
    /// Shapes are not tessellated when they are defined: the first draw tessellates
    /// at the scale it is drawn at.
    fn get_or_retessellate_handle(
        self,
        context: &mut RenderContext<'_, 'gc>,
        current_scale: f32,
    ) -> ShapeHandle {
        // Since graphics are created from a shared shape, we may be able to reuse a
        // cached tessellation from another instance at a similar scale.
        let shared = self.0.shared.get();

        {
            let mut cache = shared.scaled_handle.borrow_mut();
            if let Some(handle) = cache.find_near_and_touch(current_scale) {
                // Found a cached handle at a similar scale; reuse it.
                return handle;
            }
        }

        // Tessellate at the current scale
        let start = crate::perf_stats::enabled().then(web_time::Instant::now);
        let library = context
            .library
            .library_for_movie(shared.movie)
            .expect("Graphic's movie has a library");
        let shape = shared.shape.get(shared.movie);
        let new_handle = context.renderer.register_shape_with_scale(
            (&*shape).into(),
            &MovieLibrarySource { library },
            current_scale,
        );
        // Not needed until the next tessellation (or hit test).
        drop(shape);
        shared.shape.release();
        if let Some(start) = start {
            crate::perf_stats::record_tessellation(start.elapsed());
        }

        {
            let mut cache = shared.scaled_handle.borrow_mut();
            tracing::debug!(
                "Graphic id={} retessellated: new_scale={:.2}, cache_size={}",
                shared.id,
                current_scale,
                cache.len()
            );
            cache.insert(current_scale, new_handle.clone());
        }

        new_handle
    }
}

impl<'gc> Graphic<'gc> {
    /// Drops this shape's tessellations that weren't drawn since `cutoff` (see
    /// `tessellation_cache::clock`), returning how many were dropped.
    pub fn evict_unused_tessellations(self, cutoff: u32) -> usize {
        self.0
            .shared
            .get()
            .scaled_handle
            .borrow_mut()
            .evict_unused_since(cutoff)
    }
}

impl<'gc> TDisplayObject<'gc> for Graphic<'gc> {
    fn base(self) -> Gc<'gc, DisplayObjectBase<'gc>> {
        HasPrefixField::as_prefix_gc(self.0)
    }

    fn id(self) -> CharacterId {
        self.0.shared.get().id
    }

    fn self_bounds(self, mode: BoundsMode) -> Rectangle<Twips> {
        let include_strokes = mode.includes_strokes();

        if let Some(drawing) = self.0.drawing.get() {
            drawing.borrow().self_bounds(include_strokes)
        } else if include_strokes {
            self.0.shared.get().shape_bounds
        } else {
            self.0.shared.get().edge_bounds
        }
    }

    fn construct_frame(self, context: &mut UpdateContext<'gc>) {
        if self.movie().is_action_script_3() && self.object2().is_none() {
            let class_object = self
                .0
                .class
                .get()
                .unwrap_or_else(|| context.avm2.classes().shape);

            let mut activation = Avm2Activation::from_nothing(context);

            match Avm2StageObject::for_display_object_childless(
                &mut activation,
                self.into(),
                class_object,
            ) {
                Ok(object) => self.set_object2(activation.context, object),
                Err(err) => {
                    Avm2::uncaught_error(
                        &mut activation,
                        Some(self.into()),
                        err,
                        "Error running AVM2 construction for shape",
                    );
                }
            }

            self.on_construction_complete(context);
        }
    }

    fn replace_with(self, context: &mut UpdateContext<'gc>, id: CharacterId) {
        // Static assets like Graphics can replace themselves via a PlaceObject tag with PlaceObjectAction::Replace.
        // This does not create a new instance, but instead swaps out the underlying static data to point to the new art.
        if let Some(new_graphic) = context
            .library
            .library_for_movie_mut(self.movie())
            .get_graphic(id)
        {
            self.set_shared(context.gc(), new_graphic.0.shared.get());
        } else {
            tracing::warn!("PlaceObject: expected Graphic at character ID {}", id);
        }
        self.invalidate_cached_bitmap();
    }

    fn render_self(self, context: &mut RenderContext<'_, 'gc>) {
        if !context.is_offscreen
            && !self
                .world_bounds(BoundsMode::Engine)
                .intersects(&context.stage.view_bounds())
        {
            // Off-screen; culled
            return;
        }

        if let Some(slicing) = Slicing::of(self.into()) {
            let world = context.transform_stack.transform();
            let transform = Transform {
                matrix: slicing.draw_matrix(&world.matrix),
                ..world
            };
            self.render_sliced(context, slicing, Matrix::IDENTITY, transform);
            return;
        }
        if let Some(clip) = self
            .parent()
            .filter(|parent| parent.as_movie_clip().is_some())
            && let Some(slicing) = Slicing::of(clip)
            && let matrix = self.base().matrix()
            && let Some(inverse) = matrix.inverse()
        {
            let world = context.transform_stack.transform();
            let transform = Transform {
                matrix: slicing.draw_matrix(&(world.matrix * inverse)),
                ..world
            };
            self.render_sliced(context, slicing, matrix, transform);
            return;
        }

        if let Some(drawing) = self.0.drawing.get() {
            drawing.borrow().render(context);
        } else if !self.0.shared.get().is_empty {
            let transform = context.transform_stack.transform();

            // Calculate the current scale from the transform, to determine if
            // we can reuse a cached tessellation or need to retessellate.
            let matrix = &transform.matrix;
            let scale_x = f32::abs(matrix.a + matrix.c);
            let scale_y = f32::abs(matrix.b + matrix.d);
            let current_scale = ((scale_x * scale_x + scale_y * scale_y) / 2.0).sqrt();

            let handle = self.get_or_retessellate_handle(context, current_scale);

            context.commands.render_shape(handle, transform)
        }
    }

    fn hit_test_shape(
        self,
        _context: &mut UpdateContext<'gc>,
        point: Point<Twips>,
        options: HitTestOptions,
    ) -> bool {
        // Transform point to local coordinates and test.
        if (!options.contains(HitTestOptions::SKIP_INVISIBLE) || self.visible())
            && self.world_bounds(BoundsMode::Engine).contains(point)
        {
            let Some(local_matrix) = self.global_to_local_matrix() else {
                return false;
            };
            let point = local_matrix * point;
            if let Some(drawing) = self.0.drawing.get() {
                if drawing.borrow().hit_test(point, &local_matrix) {
                    return true;
                }
            } else {
                let shared = self.0.shared.get();
                let shape = shared.shape.get(shared.movie);
                return ruffle_render::shape_utils::shape_hit_test(&shape, point, &local_matrix);
            }
        }

        false
    }

    fn post_instantiation(
        self,
        context: &mut UpdateContext<'gc>,
        _init_object: Option<Avm1Object<'gc>>,
        _instantiated_by: Instantiator,
        _run_frame: bool,
    ) {
        if self.movie().is_action_script_3() {
            self.set_default_instance_name(context);
        }
    }

    fn movie(self) -> SwfMovie<'gc> {
        self.0.shared.get().movie
    }

    fn object1(self) -> Option<Avm1Object<'gc>> {
        None
    }

    fn object2(self) -> Option<Avm2StageObject<'gc>> {
        self.0.avm2_object.get()
    }

    fn set_object2(self, context: &mut UpdateContext<'gc>, to: Avm2StageObject<'gc>) {
        let mc = context.gc();
        unlock!(Gc::write(mc, self.0), GraphicData, avm2_object).set(Some(to));
    }

    fn as_drawing(&self) -> Option<RefMut<'_, Drawing>> {
        Some(self.drawing_mut())
    }
}

/// Data shared between all instances of a Graphic.
#[derive(Collect)]
#[collect(no_drop)]
struct GraphicShared<'gc> {
    movie: SwfMovie<'gc>,
    id: CharacterId,

    #[collect(require_static)]
    shape: ShapeRecords,

    /// Whether the shape has no records (and so isn't drawn).
    is_empty: bool,

    #[collect(require_static)]
    shape_bounds: Rectangle<Twips>,

    #[collect(require_static)]
    edge_bounds: Rectangle<Twips>,

    #[collect(require_static)]
    scaled_handle: RefCell<TessellationCache>,
}

/// A shape's definition (its styles and records), which is only needed to tessellate
/// it and to hit test it. A movie can define thousands of shapes, and
/// most are never drawn, so shapes don't keep it: it's parsed again from the tag in the
/// movie's data (which stays in memory) when it's needed, and dropped again after
/// tessellating (for example, at another scale after the window is resized). Shapes that
/// are hit tested keep it until they're tessellated again.
struct ShapeRecords {
    parsed: RefCell<Option<Rc<swf::Shape>>>,
    /// Where the DefineShape tag's body is in the movie's data, and its version.
    source: Option<(Range<usize>, u8)>,
}

/// The approximate memory of all parsed shape definitions (for `RUFFLE_PERF`).
pub static PARSED_SHAPE_BYTES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

fn parsed_size(shape: &swf::Shape) -> usize {
    shape.shape.capacity() * std::mem::size_of::<swf::ShapeRecord>()
        + shape
            .shape
            .iter()
            .map(|record| match record {
                swf::ShapeRecord::StyleChange(change) => {
                    std::mem::size_of::<swf::StyleChangeData>()
                        + change.new_styles.as_ref().map_or(0, |styles| {
                            styles.fill_styles.capacity() * std::mem::size_of::<swf::FillStyle>()
                                + styles.line_styles.capacity()
                                    * std::mem::size_of::<swf::LineStyle>()
                        })
                }
                _ => 0,
            })
            .sum::<usize>()
        + shape.styles.fill_styles.capacity() * std::mem::size_of::<swf::FillStyle>()
        + shape.styles.line_styles.capacity() * std::mem::size_of::<swf::LineStyle>()
}

impl ShapeRecords {
    fn new(shape: swf::Shape, source: Option<Range<usize>>) -> Self {
        let version = shape.version;
        PARSED_SHAPE_BYTES.fetch_add(parsed_size(&shape), std::sync::atomic::Ordering::Relaxed);
        Self {
            parsed: RefCell::new(Some(Rc::new(shape))),
            source: source.map(|range| (range, version)),
        }
    }

    /// The shape's definition, parsed again if it was released.
    fn get(&self, movie: SwfMovie<'_>) -> Rc<swf::Shape> {
        if let Some(shape) = &*self.parsed.borrow() {
            return shape.clone();
        }
        let (range, version) = self
            .source
            .clone()
            .expect("only shapes with a source are released");
        let mut reader = swf::read::Reader::new(&movie.data()[range], movie.version());
        let shape = Rc::new(
            reader
                .read_define_shape(version)
                .expect("the shape was parsed from this data before"),
        );
        PARSED_SHAPE_BYTES.fetch_add(parsed_size(&shape), std::sync::atomic::Ordering::Relaxed);
        *self.parsed.borrow_mut() = Some(shape.clone());
        shape
    }

    /// Drops the parsed definition, if it can be parsed again.
    fn release(&self) {
        if self.source.is_some()
            && ruffle_render::perf_flags::enabled(
                ruffle_render::perf_flags::PerfFlag::ReleaseShapes,
            )
            && let Some(shape) = self.parsed.borrow_mut().take()
        {
            PARSED_SHAPE_BYTES.fetch_sub(parsed_size(&shape), std::sync::atomic::Ordering::Relaxed);
        }
    }
}

impl Drop for ShapeRecords {
    fn drop(&mut self) {
        if let Some(shape) = self.parsed.get_mut() {
            PARSED_SHAPE_BYTES.fetch_sub(parsed_size(shape), std::sync::atomic::Ordering::Relaxed);
        }
    }
}
