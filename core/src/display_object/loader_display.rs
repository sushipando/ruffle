use crate::avm1::Object as Avm1Object;
use crate::avm2::Activation;
use crate::avm2::StageObject as Avm2StageObject;
use crate::backend::ui::MouseCursor;
use crate::context::RenderContext;
use crate::context::UpdateContext;
use crate::display_object::TInteractiveObject;
use crate::display_object::{BoundsMode, DisplayObjectBase, DisplayObjectPtr};
use crate::events::{ClipEvent, ClipEventResult};
use crate::prelude::*;

use crate::display_object::container::ChildContainer;
use crate::display_object::interactive::InteractiveObjectBase;
use crate::tag_utils::SwfMovie;
use crate::vminterface::Instantiator;
use core::fmt;
use gc_arena::barrier::unlock;
use gc_arena::lock::{Lock, RefLock};
use gc_arena::{Collect, Gc, GcWeak, Mutation};
use ruffle_common::utils::HasPrefixField;
use std::cell::{Cell, Ref, RefMut};

use super::interactive::Avm2MousePick;

#[derive(Clone, Collect, Copy)]
#[collect(no_drop)]
pub struct LoaderDisplay<'gc>(Gc<'gc, LoaderDisplayData<'gc>>);

impl fmt::Debug for LoaderDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LoaderDisplay")
            .field("ptr", &Gc::as_ptr(self.0))
            .finish()
    }
}

#[derive(Clone, Collect, HasPrefixField)]
#[collect(no_drop)]
#[repr(C, align(8))]
pub struct LoaderDisplayData<'gc> {
    base: InteractiveObjectBase<'gc>,
    container: RefLock<ChildContainer<'gc>>,
    avm2_object: Lock<Option<Avm2StageObject<'gc>>>,
    movie: SwfMovie<'gc>,
    settled_epoch: Cell<u64>,
}

impl<'gc> LoaderDisplay<'gc> {
    pub fn empty(activation: &mut Activation<'_, 'gc>, movie: SwfMovie<'gc>) -> Self {
        let obj = LoaderDisplay(Gc::new(
            activation.gc(),
            LoaderDisplayData {
                base: Default::default(),
                container: RefLock::new(ChildContainer::new(&movie)),
                avm2_object: Lock::new(None),
                movie,
                settled_epoch: Cell::new(0),
            },
        ));

        obj.set_placed_by_avm2_script(true);
        activation.context.orphan_manager.add_orphan_obj(obj.into());
        obj
    }

    pub fn downgrade(self) -> LoaderDisplayWeak<'gc> {
        LoaderDisplayWeak(Gc::downgrade(self.0))
    }

    #[inline(always)]
    pub(crate) fn settled_epoch(self) -> u64 {
        self.0.settled_epoch.get()
    }

    #[inline(always)]
    pub(crate) fn clear_settled_epoch(self) -> u64 {
        self.0.settled_epoch.replace(0)
    }

    pub(crate) fn unsettle(self) {
        crate::display_object::unsettle_ancestors(Some(self.into()));
    }

    fn is_idle(self) -> bool {
        crate::display_object::is_packed_idle(self.0.settled_epoch.get())
    }

    fn is_settled(self, context: &UpdateContext<'gc>) -> bool {
        crate::display_object::is_packed_settled(
            self.0.settled_epoch.get(),
            context.stage.frame_epoch(),
        )
    }

    fn settle(self, context: &UpdateContext<'gc>) {
        if !self.movie().is_action_script_3() {
            return;
        }
        let mut idle = !self.base().should_skip_next_enter_frame();
        let epoch = context.stage.frame_epoch();
        let generation = crate::display_object::button_state_unsettle_gen();
        for child in self.iter_render_list() {
            let (settled, child_idle) =
                crate::display_object::child_settled_state(child, epoch, generation);
            if !settled {
                return;
            }
            if !child_idle {
                idle = false;
            }
        }
        self.0
            .settled_epoch
            .set(crate::display_object::pack_settled_epoch(
                generation,
                if idle {
                    crate::display_object::SETTLED_IDLE_LOW
                } else {
                    epoch
                },
            ));
    }
}

impl<'gc> TDisplayObject<'gc> for LoaderDisplay<'gc> {
    fn base(self) -> Gc<'gc, DisplayObjectBase<'gc>> {
        HasPrefixField::as_prefix_gc(self.raw_interactive())
    }

    fn id(self) -> CharacterId {
        u16::MAX
    }

    fn render_self(self, context: &mut RenderContext<'_, 'gc>) {
        self.render_children(context);
    }

    fn self_bounds(self, _mode: BoundsMode) -> Rectangle<Twips> {
        Default::default()
    }

    fn object1(self) -> Option<crate::avm1::Object<'gc>> {
        None
    }

    fn object2(self) -> Option<Avm2StageObject<'gc>> {
        self.0.avm2_object.get()
    }

    fn set_object2(self, context: &mut UpdateContext<'gc>, to: Avm2StageObject<'gc>) {
        let mc = context.gc();
        unlock!(Gc::write(mc, self.0), LoaderDisplayData, avm2_object).set(Some(to))
    }

    fn enter_frame(self, context: &mut UpdateContext<'gc>) {
        let skip_frame = self.base().should_skip_next_enter_frame();
        if !skip_frame && self.is_idle() {
            return;
        }
        for child in self.iter_render_list() {
            // See MovieClip::enter_frame for an explanation of this.
            if skip_frame {
                child.base().set_skip_next_enter_frame(true);
            }
            child.enter_frame(context);
        }
        self.base().set_skip_next_enter_frame(false);
    }

    fn construct_frame(self, context: &mut UpdateContext<'gc>) {
        if self.is_settled(context) {
            return;
        }
        for child in self.iter_render_list() {
            child.construct_frame(context);
        }
    }

    fn run_frame_scripts(self, context: &mut UpdateContext<'gc>) {
        if self.is_settled(context) {
            return;
        }
        for child in self.iter_render_list() {
            child.run_frame_scripts(context);
        }
        self.settle(context);
    }

    fn post_instantiation(
        self,
        context: &mut UpdateContext<'gc>,
        _init_object: Option<Avm1Object<'gc>>,
        _instantiated_by: Instantiator,
        _run_frame: bool,
    ) {
        self.set_default_instance_name(context);
    }

    fn movie(self) -> SwfMovie<'gc> {
        self.0.movie
    }

    fn on_parent_removed(self, context: &mut UpdateContext<'gc>) {
        if self.movie().is_action_script_3() {
            context.orphan_manager.add_orphan_obj(self.into())
        }
    }
}

impl<'gc> TInteractiveObject<'gc> for LoaderDisplay<'gc> {
    fn raw_interactive(self) -> Gc<'gc, InteractiveObjectBase<'gc>> {
        HasPrefixField::as_prefix_gc(self.0)
    }

    fn as_displayobject(self) -> DisplayObject<'gc> {
        self.into()
    }

    fn filter_clip_event(
        self,
        _context: &mut UpdateContext<'gc>,
        _event: ClipEvent,
    ) -> ClipEventResult {
        if !self.visible() {
            return ClipEventResult::NotHandled;
        }

        ClipEventResult::Handled
    }
    fn event_dispatch(
        self,
        _context: &mut UpdateContext<'gc>,
        _event: ClipEvent<'gc>,
    ) -> ClipEventResult {
        ClipEventResult::NotHandled
    }

    fn mouse_pick_avm2(
        self,
        context: &mut UpdateContext<'gc>,
        point: Point<Twips>,
        require_button_mode: bool,
    ) -> Avm2MousePick<'gc> {
        let parent_to_global = self
            .parent()
            .filter(|p| p.as_stage().is_none())
            .map_or(Matrix::IDENTITY, |p| p.local_to_global_matrix());
        self.mouse_pick_avm2_with_matrix(context, point, require_button_mode, parent_to_global)
    }

    fn mouse_pick_avm2_with_matrix(
        self,
        context: &mut UpdateContext<'gc>,
        point: Point<Twips>,
        require_button_mode: bool,
        parent_to_global: Matrix,
    ) -> Avm2MousePick<'gc> {
        let mut options = HitTestOptions::SKIP_INVISIBLE;
        options.set(HitTestOptions::SKIP_MASK, self.maskee().is_none());

        if self.visible() {
            let mut self_to_global = parent_to_global * self.base().raw_matrix();
            if let Some(rect) = self.scroll_rect() {
                self_to_global *= Matrix::translate(-rect.x_min, -rect.y_min);
            }
            // A loader has at most one child.
            if let Some(child) = self.iter_render_list().next() {
                if let Some(int) = child.as_interactive() {
                    if int.as_displayobject().movie().is_action_script_3() {
                        return int
                            .mouse_pick_avm2_with_matrix(
                                context,
                                point,
                                require_button_mode,
                                self_to_global,
                            )
                            .combine_with_parent(self.into());
                    } else if let Some(result) =
                        int.mouse_pick_avm1(context, point, require_button_mode)
                    {
                        return Avm2MousePick::Hit(result).combine_with_parent(self.into());
                    }
                }

                let hit = match child {
                    DisplayObject::Graphic(g) if !g.base().has_scroll_rect() => {
                        g.hit_test_shape_with_matrix(
                            point,
                            options,
                            self_to_global * g.base().raw_matrix(),
                        )
                    }
                    _ => child.hit_test_shape(context, point, options),
                };
                if hit {
                    if self.mouse_enabled() {
                        return Avm2MousePick::Hit(self.into());
                    } else {
                        return Avm2MousePick::PropagateToParent;
                    }
                }
            }
        }
        Avm2MousePick::Miss
    }

    fn mouse_cursor(self, _context: &mut UpdateContext<'gc>) -> MouseCursor {
        MouseCursor::Arrow
    }
}

impl<'gc> TDisplayObjectContainer<'gc> for LoaderDisplay<'gc> {
    fn raw_container(&self) -> Ref<'_, ChildContainer<'gc>> {
        self.0.container.borrow()
    }

    fn raw_container_mut(&self, gc_context: &Mutation<'gc>) -> RefMut<'_, ChildContainer<'gc>> {
        self.unsettle();
        unlock!(Gc::write(gc_context, self.0), LoaderDisplayData, container).borrow_mut()
    }
}

#[derive(Clone, Debug, Collect, Copy)]
#[collect(no_drop)]
pub struct LoaderDisplayWeak<'gc>(GcWeak<'gc, LoaderDisplayData<'gc>>);

impl<'gc> LoaderDisplayWeak<'gc> {
    pub fn upgrade(self, mc: &Mutation<'gc>) -> Option<LoaderDisplay<'gc>> {
        self.0.upgrade(mc).map(LoaderDisplay)
    }

    pub fn as_ptr(self) -> *const DisplayObjectPtr {
        self.0.as_ptr() as *const DisplayObjectPtr
    }
}
