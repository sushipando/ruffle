use crate::descriptors::Descriptors;
use crate::globals::Globals;
use fnv::FnvHashMap;
use std::fmt::{Debug, Formatter};
use std::ops::Deref;
use std::sync::{Arc, Mutex, Weak};

type PoolInner<T> = Mutex<Vec<T>>;
type Constructor<Type, Description> = Box<dyn Fn(&Descriptors, &Description) -> Type>;

/// A pooled texture, its view, and a bind group for drawing it with the bitmap
/// pipelines (created on first use, see `PooledTexture::bitmap_bind_group`).
pub type PooledTexture = (
    wgpu::Texture,
    wgpu::TextureView,
    std::sync::OnceLock<wgpu::BindGroup>,
);

type TexturePoolEntry = BufferPool<PooledTexture, AlwaysCompatible>;

/// Textures and globals of sizes unused for this many frames are freed.
const UNUSED_FRAMES_TO_KEEP: u64 = 4;

#[derive(Debug, Default)]
pub struct TexturePool {
    /// Each pool, with the frame it was last used in.
    pools: FnvHashMap<TextureKey, (TexturePoolEntry, u64)>,
    globals_cache: FnvHashMap<GlobalsKey, (Arc<Globals>, u64)>,
    frame: u64,
}

impl TexturePool {
    pub fn new() -> Self {
        Default::default()
    }

    /// Frees textures and globals of sizes that haven't been used recently.
    /// Without this, a pool used for content of varying sizes would grow forever.
    pub fn end_frame(&mut self) {
        self.frame += 1;
        let oldest = self.frame.saturating_sub(UNUSED_FRAMES_TO_KEEP);
        self.pools.retain(|_, (_, last_used)| *last_used >= oldest);
        self.globals_cache
            .retain(|_, (_, last_used)| *last_used >= oldest);
    }

    pub fn get_texture(
        &mut self,
        descriptors: &Descriptors,
        size: wgpu::Extent3d,
        usage: wgpu::TextureUsages,
        format: wgpu::TextureFormat,
        sample_count: u32,
    ) -> PoolEntry<PooledTexture, AlwaysCompatible> {
        let key = TextureKey {
            size,
            usage,
            format,
            sample_count,
        };
        let frame = self.frame;
        let (pool, last_used) = self.pools.entry(key).or_insert_with(|| {
            let label = if cfg!(feature = "render_debug_labels") {
                use std::sync::atomic::{AtomicU32, Ordering};
                static ID_COUNT: AtomicU32 = AtomicU32::new(0);
                let id = ID_COUNT.fetch_add(1, Ordering::Relaxed);
                create_debug_label!("Pooled texture {}", id)
            } else {
                None
            };
            let pool = BufferPool::new(Box::new(move |descriptors, _description| {
                let texture = descriptors.device.create_texture(&wgpu::TextureDescriptor {
                    label: label.as_deref(),
                    size,
                    mip_level_count: 1,
                    sample_count,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    view_formats: &[format],
                    usage,
                });
                let view = texture.create_view(&Default::default());
                (texture, view, Default::default())
            }));
            (pool, frame)
        });
        *last_used = frame;
        pool.take(descriptors, AlwaysCompatible)
    }

    pub fn get_globals(
        &mut self,
        descriptors: &Descriptors,
        viewport_width: u32,
        viewport_height: u32,
    ) -> Arc<Globals> {
        let frame = self.frame;
        let (globals, last_used) = self
            .globals_cache
            .entry(GlobalsKey {
                viewport_width,
                viewport_height,
            })
            .or_insert_with(|| {
                (
                    Arc::new(Globals::new(
                        &descriptors.device,
                        &descriptors.bind_layouts.globals,
                        viewport_width,
                        viewport_height,
                    )),
                    frame,
                )
            });
        *last_used = frame;
        globals.clone()
    }
}

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
struct TextureKey {
    size: wgpu::Extent3d,
    usage: wgpu::TextureUsages,
    format: wgpu::TextureFormat,
    sample_count: u32,
}

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
struct GlobalsKey {
    viewport_width: u32,
    viewport_height: u32,
}

pub trait BufferDescription: Clone + Debug {
    type Cost: Ord;

    /// If the potential buffer represented by this description (`self`)
    /// fits another existing buffer and its description (`other`),
    /// return the cost to use that buffer instead of making a new one.
    ///
    /// Cost is an arbitrary unit, but lower is better.
    /// None means that the other buffer cannot be used in place of this one.
    fn cost_to_use(&self, other: &Self) -> Option<Self::Cost>;
}

#[derive(Clone, Debug)]
pub struct AlwaysCompatible;

impl BufferDescription for AlwaysCompatible {
    type Cost = ();

    fn cost_to_use(&self, _other: &Self) -> Option<()> {
        Some(())
    }
}

pub struct BufferPool<Type, Description: BufferDescription> {
    available: Arc<PoolInner<(Type, Description)>>,
    constructor: Constructor<Type, Description>,
}

impl<Type, Description: BufferDescription> Debug for BufferPool<Type, Description> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BufferPool").finish()
    }
}

impl<Type, Description: BufferDescription> BufferPool<Type, Description> {
    pub fn new(constructor: Constructor<Type, Description>) -> Self {
        Self {
            available: Arc::new(Mutex::new(vec![])),
            constructor,
        }
    }

    pub fn take(
        &self,
        descriptors: &Descriptors,
        description: Description,
    ) -> PoolEntry<Type, Description> {
        let mut guard = self
            .available
            .lock()
            .expect("Should not be able to lock recursively");
        let mut best: Option<(Description::Cost, usize)> = None;
        for i in 0..guard.len() {
            if let Some(cost) = description.cost_to_use(&guard[i].1) {
                if let Some(best) = &mut best {
                    if best.0 > cost {
                        *best = (cost, i);
                    }
                } else if best.is_none() {
                    best = Some((cost, i));
                }
            }
        }

        let (item, used_description) = if let Some((_, best)) = best {
            guard.swap_remove(best)
        } else {
            let item = (self.constructor)(descriptors, &description);
            (item, description)
        };
        PoolEntry {
            item: Some(item),
            description: used_description,
            pool: Arc::downgrade(&self.available),
        }
    }
}

pub struct PoolEntry<Type, Description: BufferDescription> {
    item: Option<Type>,
    description: Description,
    pool: Weak<PoolInner<(Type, Description)>>,
}

impl<Type, Description: BufferDescription> Debug for PoolEntry<Type, Description>
where
    Type: Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("PoolEntry").field(&self.item).finish()
    }
}

impl<Type, Description: BufferDescription> Drop for PoolEntry<Type, Description> {
    fn drop(&mut self) {
        if let Some(item) = self.item.take()
            && let Some(pool) = self.pool.upgrade()
        {
            pool.lock()
                .expect("Should not be able to lock recursively")
                .push((item, self.description.clone()))
        }
    }
}

impl<Type, Description: BufferDescription> Deref for PoolEntry<Type, Description> {
    type Target = Type;

    fn deref(&self) -> &Self::Target {
        self.item.as_ref().expect("Item should exist until dropped")
    }
}
