//! Lightweight per-second frame timing statistics.
//!
//! Enabled by setting the `RUFFLE_PERF` environment variable. Once a second,
//! a summary line is written to stderr describing how much time was spent
//! running frames (scripts + timeline), rendering, and in the rest of `tick`
//! (timers, sockets, streams).

use ruffle_render::tessellator::stats as tessellation_stats;
use std::time::Duration;
use web_time::Instant;

/// Whether `RUFFLE_PERF` statistics are being collected.
pub fn enabled() -> bool {
    tessellation_stats::enabled()
}

static INNER_GOTO_NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static INNER_GOTOS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Adds the time a goto spent running frame construction and frame scripts for the whole
/// stage (see `run_inner_goto_frame`).
pub fn record_inner_goto(d: Duration) {
    use std::sync::atomic::Ordering;
    INNER_GOTO_NANOS.fetch_add(d.as_nanos() as u64, Ordering::Relaxed);
    INNER_GOTOS.fetch_add(1, Ordering::Relaxed);
}

/// Adds the time it took to register a shape for tessellation (parsing it, and
/// tessellating it unless the renderer does that later) during the current render.
pub fn record_tessellation(d: Duration) {
    tessellation_stats::record(d, 1);
}

#[derive(Default)]
struct Bucket {
    count: u32,
    total: Duration,
    max: Duration,
}

impl Bucket {
    fn add(&mut self, d: Duration) {
        self.count += 1;
        self.total += d;
        self.max = self.max.max(d);
    }

    fn avg_ms(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.total.as_secs_f64() * 1000.0 / self.count as f64
        }
    }

    fn max_ms(&self) -> f64 {
        self.max.as_secs_f64() * 1000.0
    }

    fn total_ms(&self) -> f64 {
        self.total.as_secs_f64() * 1000.0
    }
}

pub struct PerfStats {
    window_start: Instant,
    run_frame: Bucket,
    render: Bucket,
    tick_other: Bucket,
    /// The number of live GC objects, and of orphan display objects.
    gc_objects: usize,
    orphans: usize,
    loaders: usize,
    /// Time spent tessellating shapes, per render.
    tessellation: Bucket,
    tessellated_shapes: u32,
}

impl PerfStats {
    /// Returns `Some` if perf logging was requested through `RUFFLE_PERF`.
    pub fn from_env() -> Option<Self> {
        std::env::var_os("RUFFLE_PERF").map(|_| {
            tessellation_stats::enable();
            Self {
                window_start: Instant::now(),
                run_frame: Bucket::default(),
                render: Bucket::default(),
                tick_other: Bucket::default(),
                gc_objects: 0,
                orphans: 0,
                loaders: 0,
                tessellation: Bucket::default(),
                tessellated_shapes: 0,
            }
        })
    }

    pub fn record_run_frame(&mut self, d: Duration) {
        self.run_frame.add(d);
    }

    pub fn set_heap_stats(&mut self, gc_objects: usize, orphans: usize, loaders: usize) {
        self.gc_objects = gc_objects;
        self.orphans = orphans;
        self.loaders = loaders;
    }

    pub fn record_render(&mut self, d: Duration) {
        self.render.add(d);
        let (tessellation, shapes) = tessellation_stats::take();
        self.tessellation.add(tessellation);
        self.tessellated_shapes += shapes;
        self.maybe_report();
    }

    pub fn record_tick_other(&mut self, d: Duration) {
        self.tick_other.add(d);
        self.maybe_report();
    }

    fn maybe_report(&mut self) {
        let elapsed = self.window_start.elapsed();
        if elapsed < Duration::from_secs(1) {
            return;
        }
        let secs = elapsed.as_secs_f64();
        let busy_ms =
            self.run_frame.total_ms() + self.render.total_ms() + self.tick_other.total_ms();
        eprintln!(
            "[perf] fps {:5.1} | frame avg {:6.2}ms max {:6.2}ms | render {:5.1}/s avg {:6.2}ms max {:6.2}ms | tick-other avg {:5.2}ms max {:6.2}ms | busy {:4.0}% | gc objects {} orphans {} loaders {} | parsed shapes {}MB | tessellation {:.1}ms max {:.1}ms ({} shapes) | nested gotos {} {:.1}ms",
            self.run_frame.count as f64 / secs,
            self.run_frame.avg_ms(),
            self.run_frame.max_ms(),
            self.render.count as f64 / secs,
            self.render.avg_ms(),
            self.render.max_ms(),
            self.tick_other.avg_ms(),
            self.tick_other.max_ms(),
            busy_ms / (secs * 10.0),
            self.gc_objects,
            self.orphans,
            self.loaders,
            crate::display_object::graphic::PARSED_SHAPE_BYTES
                .load(std::sync::atomic::Ordering::Relaxed)
                / 1_000_000,
            self.tessellation.total_ms(),
            self.tessellation.max_ms(),
            self.tessellated_shapes,
            INNER_GOTOS.swap(0, std::sync::atomic::Ordering::Relaxed),
            INNER_GOTO_NANOS.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1e6,
        );
        self.window_start = Instant::now();
        self.run_frame = Bucket::default();
        self.render = Bucket::default();
        self.tick_other = Bucket::default();
        self.tessellation = Bucket::default();
        self.tessellated_shapes = 0;
    }
}
