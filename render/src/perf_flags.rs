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
}

const NAMES: [&str; 16] = [
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
];

static FLAGS: [AtomicBool; 16] = [
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
