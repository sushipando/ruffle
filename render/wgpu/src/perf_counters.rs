//! Per-frame counters for render work, used to diagnose GPU overhead.
//!
//! Enabled by setting the `RUFFLE_PERF` environment variable. Once a second,
//! the average count per frame is written to stderr.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

#[derive(Clone, Copy)]
pub enum Counter {
    RenderPass,
    CommandTarget,
    CacheEntry,
    CacheEntryFiltered,
    Filter,
    BlendTrivial,
    BlendComplex,
    AlphaMask,
    Flush,
}

const NAMES: [&str; 9] = [
    "passes",
    "targets",
    "cache",
    "cache-filtered",
    "filters",
    "blend-trivial",
    "blend-complex",
    "alpha-mask",
    "flushes",
];

static COUNTS: [AtomicU64; 9] = [const { AtomicU64::new(0) }; 9];

pub fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("RUFFLE_PERF").is_some())
}

#[inline]
pub fn count(counter: Counter) {
    if enabled() {
        COUNTS[counter as usize].fetch_add(1, Ordering::Relaxed);
    }
}

static NAMED: std::sync::Mutex<Vec<(String, u64)>> = std::sync::Mutex::new(Vec::new());

/// Counts an arbitrary named event; used for ad-hoc breakdowns.
pub fn count_named(name: impl AsRef<str>) {
    count_named_n(name, 1);
}

/// Like `count_named`, but adds `n`.
pub fn count_named_n(name: impl AsRef<str>, n: u64) {
    if enabled() {
        let name = name.as_ref();
        let mut named = NAMED.lock().expect("perf counters lock");
        match named.iter_mut().find(|(n, _)| n == name) {
            Some((_, count)) => *count += n,
            None => named.push((name.to_string(), n)),
        }
    }
}

/// Called once per submitted frame; prints averages roughly once a second.
pub fn end_frame() {
    if !enabled() {
        return;
    }
    static STATE: std::sync::Mutex<Option<(Instant, u64)>> = std::sync::Mutex::new(None);
    let mut state = STATE.lock().expect("perf counters lock");
    let (start, frames) = state.get_or_insert_with(|| (Instant::now(), 0));
    *frames += 1;
    if start.elapsed().as_secs_f64() < 1.0 {
        return;
    }
    let frames_f = *frames as f64;
    let mut line = String::from("[gpu]");
    for (name, count) in NAMES.iter().zip(COUNTS.iter()) {
        let value = count.swap(0, Ordering::Relaxed);
        line.push_str(&format!(" {name} {:.1}", value as f64 / frames_f));
    }
    eprintln!("{line}  (per frame)");
    let mut named = std::mem::take(&mut *NAMED.lock().expect("perf counters lock"));
    if !named.is_empty() {
        named.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
        let mut line = String::from("[gpu-named]");
        for (name, count) in named {
            line.push_str(&format!(" {name}={:.1}", count as f64 / frames_f));
        }
        eprintln!("{line}");
    }
    *state = Some((Instant::now(), 0));
}
