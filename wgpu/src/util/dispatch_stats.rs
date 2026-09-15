//! Host record/submit and pass counters at the wgpu dispatch boundary.
//!
//! Blade's custom backend and wgpu-core both go through [`crate::CommandEncoder::finish`],
//! [`crate::Queue::submit`], and `begin_*_pass`. Timing those entry points — and
//! not bind-group creation — is what makes a matched host comparison possible.
//! GPU elapsed is wait-to-idle via [`crate::Device::poll`] with [`crate::PollType::Wait`];
//! timestamp queries are not required.

use crate::cmp::{AtomicU64, Ordering};

static RECORD_NS: AtomicU64 = AtomicU64::new(0);
static SUBMIT_NS: AtomicU64 = AtomicU64::new(0);
static WAIT_NS: AtomicU64 = AtomicU64::new(0);
static RENDER_PASSES: AtomicU64 = AtomicU64::new(0);
static COMPUTE_PASSES: AtomicU64 = AtomicU64::new(0);

/// How [`DispatchStats::gpu_ns`] is produced when timestamp queries are absent.
pub const GPU_TIMING_METHOD: &str = "wait-to-idle";

/// Accumulated host timings and pass counts since the last [`reset`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DispatchStats {
    /// Nanoseconds spent in [`crate::CommandEncoder::finish`].
    pub record_ns: u64,
    /// Nanoseconds spent in [`crate::Queue::submit`].
    pub submit_ns: u64,
    /// Nanoseconds spent in blocking [`crate::Device::poll`] waits.
    pub wait_ns: u64,
    /// `begin_render_pass` + `begin_compute_pass` since reset.
    pub gpu_pass_count: u64,
    /// `begin_render_pass` since reset.
    pub render_pass_count: u64,
    /// `begin_compute_pass` since reset.
    pub compute_pass_count: u64,
}

impl DispatchStats {
    /// Wait-to-idle GPU elapsed for this window. Same as [`Self::wait_ns`].
    pub fn gpu_ns(&self) -> u64 {
        self.wait_ns
    }
}

/// Zero every counter. Call at the start of a measured iteration.
pub fn reset() {
    RECORD_NS.store(0, Ordering::Relaxed);
    SUBMIT_NS.store(0, Ordering::Relaxed);
    WAIT_NS.store(0, Ordering::Relaxed);
    RENDER_PASSES.store(0, Ordering::Relaxed);
    COMPUTE_PASSES.store(0, Ordering::Relaxed);
}

/// Copy the counters without clearing them.
pub fn snapshot() -> DispatchStats {
    let render_pass_count = RENDER_PASSES.load(Ordering::Relaxed);
    let compute_pass_count = COMPUTE_PASSES.load(Ordering::Relaxed);
    DispatchStats {
        record_ns: RECORD_NS.load(Ordering::Relaxed),
        submit_ns: SUBMIT_NS.load(Ordering::Relaxed),
        wait_ns: WAIT_NS.load(Ordering::Relaxed),
        gpu_pass_count: render_pass_count.saturating_add(compute_pass_count),
        render_pass_count,
        compute_pass_count,
    }
}

/// Copy the counters and [`reset`].
pub fn take() -> DispatchStats {
    let stats = snapshot();
    reset();
    stats
}

pub(crate) fn note_render_pass() {
    RENDER_PASSES.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn note_compute_pass() {
    COMPUTE_PASSES.fetch_add(1, Ordering::Relaxed);
}

pub(crate) struct TimingScope {
    target: &'static AtomicU64,
    #[cfg(std)]
    start: std::time::Instant,
}

impl TimingScope {
    fn new(target: &'static AtomicU64) -> Self {
        Self {
            target,
            #[cfg(std)]
            start: std::time::Instant::now(),
        }
    }
}

impl Drop for TimingScope {
    fn drop(&mut self) {
        #[cfg(std)]
        {
            let nanos = self.start.elapsed().as_nanos();
            let nanos = if nanos > u128::from(u64::MAX) {
                u64::MAX
            } else {
                nanos as u64
            };
            self.target.fetch_add(nanos, Ordering::Relaxed);
        }
        #[cfg(not(std))]
        {
            let _ = self.target;
        }
    }
}

pub(crate) fn record_scope() -> TimingScope {
    TimingScope::new(&RECORD_NS)
}

pub(crate) fn submit_scope() -> TimingScope {
    TimingScope::new(&SUBMIT_NS)
}

pub(crate) fn wait_scope() -> TimingScope {
    TimingScope::new(&WAIT_NS)
}
