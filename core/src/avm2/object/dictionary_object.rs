//! Object representation for `flash.utils.Dictionary`

use crate::avm2::Error;
use crate::avm2::activation::Activation;
use crate::avm2::dynamic_map::DynamicKey;
use crate::avm2::object::script_object::ScriptObjectData;
use crate::avm2::object::{ClassObject, Object, TObject, WeakObject};
use crate::avm2::value::Value;
use crate::string::AvmString;
use core::fmt;
use fnv::FnvHashMap;
use gc_arena::barrier::unlock;
use gc_arena::collect::Trace;
use gc_arena::lock::RefLock;
use gc_arena::{Collect, Gc, GcWeak, Mutation};
use ruffle_common::utils::HasPrefixField;

/// A class instance allocator that allocates Dictionary objects.
pub fn dictionary_allocator<'gc>(
    class: ClassObject<'gc>,
    activation: &mut Activation<'_, 'gc>,
) -> Result<Object<'gc>, Error<'gc>> {
    let base = ScriptObjectData::new(class);

    Ok(DictionaryObject(Gc::new(
        activation.gc(),
        DictionaryObjectData {
            base,
            weak_entries: RefLock::new(None),
        },
    ))
    .into())
}

/// An object that allows associations between objects and values.
///
/// This is implemented by way of "object space", parallel to the property
/// space that ordinary properties live in. This space has no namespaces, and
/// keys are objects instead of strings.
#[derive(Clone, Collect, Copy)]
#[collect(no_drop)]
pub struct DictionaryObject<'gc>(pub Gc<'gc, DictionaryObjectData<'gc>>);

#[derive(Clone, Collect, Copy, Debug)]
#[collect(no_drop)]
pub struct DictionaryObjectWeak<'gc>(pub GcWeak<'gc, DictionaryObjectData<'gc>>);

impl fmt::Debug for DictionaryObject<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DictionaryObject")
            .field("ptr", &Gc::as_ptr(self.0))
            .finish()
    }
}

#[derive(Clone, Collect, HasPrefixField)]
#[collect(no_drop)]
#[repr(C, align(8))]
pub struct DictionaryObjectData<'gc> {
    /// Base script object
    base: ScriptObjectData<'gc>,

    /// For dictionaries created with `weakKeys`: the entries with object keys,
    /// which don't keep their keys alive. Once a key is garbage collected, its
    /// entry is gone. (Other keys are stored in `base`, like in other dictionaries.)
    weak_entries: RefLock<Option<WeakEntries<'gc>>>,
}

/// Entries with weakly held object keys.
///
/// These are ephemerons: a value is only kept alive while its key is reachable from
/// outside the dictionary, even if the value refers to the key. (Otherwise, an entry
/// like `dict[clip] = {owner: clip.parent}` would keep itself alive forever.) So once a
/// key is collected, its value may have been collected too, and must not be accessed.
#[derive(Clone, Default)]
struct WeakEntries<'gc> {
    /// Entries in insertion order; `None` for removed entries (which keeps the
    /// indices of the other entries stable during enumeration).
    slots: Vec<Option<(WeakObject<'gc>, Value<'gc>)>>,
    /// The slot of each key, by the key's pointer. (`WeakObject` keeps its
    /// allocation alive, so pointers can't be reused while they're in here.)
    index: FnvHashMap<usize, usize>,
    /// When to next look for collected keys.
    prune_at: usize,
}

unsafe impl<'gc> Collect<'gc> for WeakEntries<'gc> {
    fn trace<C: Trace<'gc>>(&self, cc: &mut C) {
        for (key, value) in self.slots.iter().flatten() {
            // Even collected keys must be traced (weakly), to keep their allocations (and so
            // their pointers in `index`) from being reused while they're in here.
            cc.trace(key);
            // The value of a collected key may have been collected too.
            if !key.is_dropped() && cc.ephemeron_key_reachable(key.erase()) {
                cc.trace(value);
            }
        }
    }
}

impl<'gc> WeakEntries<'gc> {
    fn key(object: Object<'gc>) -> usize {
        object.as_ptr() as usize
    }

    fn get(&self, key: Object<'gc>) -> Option<Value<'gc>> {
        let slot = *self.index.get(&Self::key(key))?;
        self.slots[slot].map(|(_, value)| value)
    }

    fn set(&mut self, key: Object<'gc>, value: Value<'gc>) {
        if let Some(&slot) = self.index.get(&Self::key(key)) {
            self.slots[slot] = Some((key.downgrade(), value));
            return;
        }
        if self.slots.len() >= self.prune_at {
            self.prune();
        }
        self.index.insert(Self::key(key), self.slots.len());
        self.slots.push(Some((key.downgrade(), value)));
    }

    fn delete(&mut self, key: Object<'gc>) {
        if let Some(slot) = self.index.remove(&Self::key(key)) {
            self.slots[slot] = None;
        }
    }

    fn has(&self, key: Object<'gc>) -> bool {
        self.index.contains_key(&Self::key(key))
    }

    /// Removes the entries of collected keys, and compacts the slots if most are empty.
    fn prune(&mut self) {
        for slot in self.slots.iter_mut() {
            if let Some((key, _)) = slot
                && key.is_dropped()
            {
                self.index.remove(&(key.as_ptr() as usize));
                *slot = None;
            }
        }
        let live = self.index.len();
        if self.slots.len() > 64 && live * 4 < self.slots.len() {
            self.slots.retain(Option::is_some);
            for (slot, entry) in self.slots.iter().enumerate() {
                if let Some((key, _)) = entry {
                    self.index.insert(key.as_ptr() as usize, slot);
                }
            }
        }
        self.prune_at = (self.slots.len() * 2).max(64);
    }

    /// The slot of the next live entry at or after `start`.
    fn next_live(&self, start: usize, mc: &Mutation<'gc>) -> Option<usize> {
        (start..self.slots.len())
            .find(|&slot| matches!(self.slots[slot], Some((key, _)) if key.upgrade(mc).is_some()))
    }
}

/// Enumerant indices of weakly keyed entries start here (above those of `base`).
const WEAK_ENUMERANT_BASE: u32 = 1 << 30;

impl<'gc> DictionaryObject<'gc> {
    /// Makes this dictionary hold its object keys weakly. Must be called while it's empty.
    pub fn set_weak_keys(self, mc: &Mutation<'gc>) {
        *unlock!(Gc::write(mc, self.0), DictionaryObjectData, weak_entries).borrow_mut() =
            Some(WeakEntries::default());
    }

    pub fn has_weak_keys(self) -> bool {
        self.0.weak_entries.borrow().is_some()
    }

    fn weak_entries_mut(
        self,
        mc: &Mutation<'gc>,
    ) -> std::cell::RefMut<'gc, Option<WeakEntries<'gc>>> {
        unlock!(Gc::write(mc, self.0), DictionaryObjectData, weak_entries).borrow_mut()
    }

    /// Retrieve a value in the dictionary's object space.
    pub fn get_property_by_object(self, name: Object<'gc>) -> Value<'gc> {
        if let Some(weak) = &*self.0.weak_entries.borrow() {
            return weak.get(name).unwrap_or(Value::Undefined);
        }
        self.base()
            .values()
            .get(&DynamicKey::Object(name))
            .map(|v| v.value)
            .unwrap_or(Value::Undefined)
    }

    /// Set a value in the dictionary's object space.
    pub fn set_property_by_object(self, name: Object<'gc>, value: Value<'gc>, mc: &Mutation<'gc>) {
        if let Some(weak) = &mut *self.weak_entries_mut(mc) {
            weak.set(name, value);
            return;
        }
        self.base()
            .values_mut(mc)
            .insert(DynamicKey::Object(name), value);
    }

    /// Delete a value from the dictionary's object space.
    pub fn delete_property_by_object(self, name: Object<'gc>, mc: &Mutation<'gc>) {
        if let Some(weak) = &mut *self.weak_entries_mut(mc) {
            weak.delete(name);
            return;
        }
        self.base().values_mut(mc).remove(&DynamicKey::Object(name));
    }

    pub fn has_property_by_object(self, name: Object<'gc>) -> bool {
        if let Some(weak) = &*self.0.weak_entries.borrow() {
            return weak.has(name);
        }
        self.base().values().contains_key(&DynamicKey::Object(name))
    }
}

impl<'gc> TObject<'gc> for DictionaryObject<'gc> {
    fn gc_base(&self) -> Gc<'gc, ScriptObjectData<'gc>> {
        HasPrefixField::as_prefix_gc(self.0)
    }

    // Calling `setPropertyIsEnumerable` on a `Dictionary` has no effect -
    // stringified properties are always enumerable.
    fn set_local_property_is_enumerable(
        &self,
        _mc: &Mutation<'gc>,
        _name: AvmString<'gc>,
        _is_enumerable: bool,
    ) {
    }

    fn get_next_enumerant(
        self,
        last_index: u32,
        activation: &mut Activation<'_, 'gc>,
    ) -> Result<u32, Error<'gc>> {
        let start = if last_index < WEAK_ENUMERANT_BASE {
            let next = self.base().get_next_enumerant(last_index);
            if next != 0 {
                return Ok(next);
            }
            0
        } else {
            (last_index - WEAK_ENUMERANT_BASE) as usize
        };
        // Weakly keyed entries come after the others.
        Ok(self
            .0
            .weak_entries
            .borrow()
            .as_ref()
            .and_then(|weak| weak.next_live(start, activation.gc()))
            .map(|slot| WEAK_ENUMERANT_BASE + slot as u32 + 1)
            .unwrap_or(0))
    }

    fn get_enumerant_name(
        self,
        index: u32,
        activation: &mut Activation<'_, 'gc>,
    ) -> Result<Value<'gc>, Error<'gc>> {
        if index > WEAK_ENUMERANT_BASE {
            let slot = (index - WEAK_ENUMERANT_BASE - 1) as usize;
            let weak = self.0.weak_entries.borrow();
            let key = weak
                .as_ref()
                .and_then(|weak| weak.slots.get(slot).copied().flatten())
                .and_then(|(key, _)| key.upgrade(activation.gc()));
            return Ok(key.map(Value::Object).unwrap_or(Value::Null));
        }
        Ok(self.base().get_enumerant_name(index).unwrap_or(Value::Null))
    }

    fn get_enumerant_value(
        self,
        index: u32,
        activation: &mut Activation<'_, 'gc>,
    ) -> Result<Value<'gc>, Error<'gc>> {
        if index > WEAK_ENUMERANT_BASE {
            let slot = (index - WEAK_ENUMERANT_BASE - 1) as usize;
            let weak = self.0.weak_entries.borrow();
            // Not just `is_dropped`: while the collector is sweeping, a key that is about
            // to be dropped can't be upgraded, but its value may already be gone.
            return Ok(weak
                .as_ref()
                .and_then(|weak| weak.slots.get(slot).copied().flatten())
                .filter(|(key, _)| key.upgrade(activation.gc()).is_some())
                .map(|(_, value)| value)
                .unwrap_or(Value::Undefined));
        }
        Ok(*self
            .base()
            .values()
            .value_at(index as usize)
            .unwrap_or(&Value::Undefined))
    }
}
