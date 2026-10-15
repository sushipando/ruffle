//! Runtime toggles for experimental rendering optimizations, so that they can be
//! compared against each other in the same scene.

use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Copy, Debug)]
pub enum PerfFlag {
    /// Render blend layers into textures sized to the blended object's bounds.
    BoundedBlends,
    /// Resolve multisampled targets only when their contents are needed,
    /// instead of at the end of every render pass.
    LazyResolve,
    /// Render small, simple blend layers together in one pass (see `layer_atlas`).
    LayerAtlas,
    /// Only render the part of the viewport that isn't hidden by the letterbox.
    VisibleRegion,
    /// Reorder commands so that complex blends can be batched (see `wave_schedule`).
    WaveSchedule,
    /// Cache global matrices while picking the object under the mouse.
    PickCache,
    /// Finish and submit frames on another thread, where supported.
    AsyncSubmit,
    /// Do multiply blends onto opaque targets as trivial blends.
    OpaqueMultiply,
    /// Apply identical filters to many cached bitmaps at once.
    FilterBatching,
    /// Let render passes share Metal command buffers (vendored wgpu-core patch).
    ShareCommandBuffers,
    /// Compute large blurs at a lower resolution when the stage is magnified.
    /// Off by default, because the result isn't exactly Flash's (it's on in the desktop
    /// player).
    BlurDownscale,
    /// Submit and present the desktop GUI's frame on the submit thread, after the
    /// movie's frame, instead of waiting for that on the main thread.
    AsyncPresent,
    /// Record the next frame while the previous one is still being submitted on
    /// another thread (only wait for it when submitting).
    OverlapRecord,
    /// Do all passes of a blur in each direction at once, with a combined kernel.
    /// Off by default, because the result isn't exactly Flash's (it's on in the desktop
    /// player).
    BlurMerge,
    /// For measurements only: don't apply the filters of cached bitmaps.
    SkipCacheFilters,
    /// For measurements only: draw complex blends (which need to read the backdrop)
    /// as normal ones.
    ComplexAsNormal,
    /// Draw complex blends (except Alpha and Erase) in the current pass, reading the
    /// backdrop from the color attachment, where supported (Metal on Apple GPUs).
    FetchBlends,
    /// Draw consecutive chunks of draws in the same render pass.
    MergeChunks,
    /// Let cached bitmaps' textures be larger than needed, so that they aren't recreated
    /// whenever their size changes a little (see `BitmapCache::update` in core).
    CacheSlack,
    /// Batch the filters of larger cached bitmaps, in larger atlases.
    BatchLargeFilters,
    /// Keep the bind groups of pooled blend layer textures (see `texture_bind_group`).
    CacheBindGroups,
    /// Apply the filters of many cached bitmaps together, whatever their parameters
    /// (only with `BlurMerge`, see `filters::batch` in the wgpu backend).
    FilterTogether,
    /// Draw a blend's content directly with the blend mode, instead of in a layer, when
    /// that's the same (a single bitmap, or a single shape with a single fill).
    DirectBlends,
    /// Drop the parsed definitions of tessellated shapes, and parse them again from the
    /// movie's data when needed (see `ShapeRecords` in core).
    ReleaseShapes,
    /// Drop the tessellations of shapes that weren't drawn for a while (see
    /// `Player::evict_unused_tessellations` in core).
    EvictMeshes,
    /// Present frames so that each stays on screen for a whole number of display refreshes
    /// (desktop, see `MainWindow::pace_present` there).
    PacedPresent,
    /// Tessellate the shapes registered for a frame together, on several threads, before
    /// the frame is drawn (see `Mesh` in the wgpu backend).
    ParallelTessellation,
}

const NAMES: [&str; 27] = [
    "bounded-blends",
    "lazy-resolve",
    "layer-atlas",
    "visible-region",
    "wave-schedule",
    "pick-cache",
    "async-submit",
    "opaque-multiply",
    "filter-batching",
    "share-cmdbufs",
    "blur-downscale",
    "async-present",
    "overlap-record",
    "blur-merge",
    "skip-cache-filters",
    "complex-as-normal",
    "fetch-blends",
    "merge-chunks",
    "cache-slack",
    "batch-large-filters",
    "cache-bind-groups",
    "filter-together",
    "direct-blends",
    "release-shapes",
    "evict-meshes",
    "paced-present",
    "parallel-tessellation",
];

static FLAGS: [AtomicBool; 27] = [
    AtomicBool::new(true),  // bounded-blends
    AtomicBool::new(true),  // lazy-resolve
    AtomicBool::new(true),  // layer-atlas
    AtomicBool::new(true),  // visible-region
    AtomicBool::new(true),  // wave-schedule
    AtomicBool::new(true),  // pick-cache
    AtomicBool::new(true),  // async-submit
    AtomicBool::new(true),  // opaque-multiply
    AtomicBool::new(true),  // filter-batching
    AtomicBool::new(true),  // share-cmdbufs
    AtomicBool::new(false), // blur-downscale
    AtomicBool::new(true),  // async-present
    AtomicBool::new(true),  // overlap-record
    AtomicBool::new(false), // blur-merge
    AtomicBool::new(false), // skip-cache-filters
    AtomicBool::new(false), // complex-as-normal
    AtomicBool::new(true),  // fetch-blends
    AtomicBool::new(true),  // merge-chunks
    AtomicBool::new(true),  // cache-slack
    AtomicBool::new(true),  // batch-large-filters
    AtomicBool::new(true),  // cache-bind-groups
    AtomicBool::new(true),  // filter-together
    AtomicBool::new(true),  // direct-blends
    AtomicBool::new(true),  // release-shapes
    AtomicBool::new(true),  // evict-meshes
    AtomicBool::new(true),  // paced-present
    AtomicBool::new(true),  // parallel-tessellation
];

#[inline]
pub fn enabled(flag: PerfFlag) -> bool {
    FLAGS[flag as usize].load(Ordering::Relaxed)
}

/// Sets a flag by name. Returns `false` if there is no such flag.
pub fn set_by_name(name: &str, value: bool) -> bool {
    match NAMES.iter().position(|n| *n == name) {
        Some(index) => {
            FLAGS[index].store(value, Ordering::Relaxed);
            true
        }
        None => false,
    }
}
