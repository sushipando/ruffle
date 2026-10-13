//! Lightweight per-second frame timing statistics.
//!
//! Enabled by setting the `RUFFLE_PERF` environment variable. Once a second,
//! a summary line is written to stderr describing how much time was spent
//! running frames (scripts + timeline), rendering, and in the rest of `tick`
//! (timers, sockets, streams).

use std::time::Duration;
use web_time::Instant;

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
}

impl PerfStats {
    /// Returns `Some` if perf logging was requested through `RUFFLE_PERF`.
    pub fn from_env() -> Option<Self> {
        std::env::var_os("RUFFLE_PERF").map(|_| Self {
            window_start: Instant::now(),
            run_frame: Bucket::default(),
            render: Bucket::default(),
            tick_other: Bucket::default(),
        })
    }

    pub fn record_run_frame(&mut self, d: Duration) {
        self.run_frame.add(d);
    }

    pub fn record_render(&mut self, d: Duration) {
        self.render.add(d);
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
            "[perf] fps {:5.1} | frame avg {:6.2}ms max {:6.2}ms | render {:5.1}/s avg {:6.2}ms max {:6.2}ms | tick-other avg {:5.2}ms max {:6.2}ms | busy {:4.0}%",
            self.run_frame.count as f64 / secs,
            self.run_frame.avg_ms(),
            self.run_frame.max_ms(),
            self.render.count as f64 / secs,
            self.render.avg_ms(),
            self.render.max_ms(),
            self.tick_other.avg_ms(),
            self.tick_other.max_ms(),
            busy_ms / (secs * 10.0),
        );
        self.window_start = Instant::now();
        self.run_frame = Bucket::default();
        self.render = Bucket::default();
        self.tick_other = Bucket::default();
    }
}
