use crate::stats::LatencyStats;
use crate::timer::{DefaultTimer, Timer};

/// Measures and records the latency of a profiled scope.
///
/// The model stays a plain local value; profiling a call is a scope around it,
/// mirroring `torch.profiler.profile`:
///
/// ```ignore
/// let mut model = Model::new();
/// let mut prof = Profiler::new();
/// let output = prof.profile(|| model.run(&input));
/// info!("{}", prof.stats());
/// ```
///
/// Every [`profile`](Profiler::profile) call records its measured duration into
/// the accumulated [`LatencyStats`]. The timer is chosen automatically by the
/// crate's features ([`DefaultTimer`]) and can be overridden with
/// [`with_timer`](Profiler::with_timer).
pub struct Profiler<T = DefaultTimer> {
    timer: T,
    stats: LatencyStats,
}

impl Profiler {
    /// Creates a profiler using the crate's default timer.
    pub fn new() -> Self {
        Self::with_timer(DefaultTimer::default())
    }
}

impl<T: Timer> Profiler<T> {
    /// Creates a profiler using a caller-provided timer.
    pub fn with_timer(timer: T) -> Self {
        Self {
            timer,
            stats: LatencyStats::default(),
        }
    }

    /// Runs `f`, records its wall-clock duration, and returns `f`'s result.
    ///
    /// This is the Rust counterpart of the `with prof:` block: the profiled
    /// work is whatever the closure performs, including any direct
    /// `ModelInference::run` call.
    pub fn profile<R>(&mut self, f: impl FnOnce() -> R) -> R {
        let start = self.timer.now();
        let result = f();
        let elapsed = self.timer.elapsed(start);
        self.stats.record(elapsed);
        result
    }

    /// Returns the latency statistics accumulated so far.
    pub fn stats(&self) -> &LatencyStats {
        &self.stats
    }

    /// Clears the accumulated latency statistics.
    pub fn reset_stats(&mut self) {
        self.stats.reset();
    }
}

impl<T: Timer + Default> Default for Profiler<T> {
    fn default() -> Self {
        Self::with_timer(T::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::cell::Cell;
    use oneliner_runtime::ModelInference;

    /// Timer whose `now` advances by 10 us per read, so every elapsed interval
    /// is 10 us.
    struct DummyTimer {
        tick: Cell<u64>,
    }

    impl DummyTimer {
        fn new() -> Self {
            Self { tick: Cell::new(0) }
        }
    }

    impl Timer for DummyTimer {
        type Instant = u64;

        fn now(&self) -> Self::Instant {
            let tick = self.tick.get();
            self.tick.set(tick + 10);
            tick
        }

        fn elapsed(&self, start: Self::Instant) -> core::time::Duration {
            core::time::Duration::from_micros(self.tick.get() - start)
        }
    }

    struct DummyModel;

    impl ModelInference for DummyModel {
        type InputTensor = u8;
        type OutputTensor = u16;

        fn create_input_tensor() -> Self::InputTensor {
            0
        }

        fn run(&mut self, input: &Self::InputTensor) -> Self::OutputTensor {
            *input as u16 + 1
        }
    }

    #[test]
    fn profiles_closures_and_accumulates_stats() {
        let mut profiler = Profiler::with_timer(DummyTimer::new());
        let mut model = DummyModel;

        let output = profiler.profile(|| model.run(&7));
        assert_eq!(output, 8);
        assert_eq!(profiler.stats().samples, 1);
        assert_eq!(profiler.stats().average(), Some(core::time::Duration::from_micros(10)));

        let output = profiler.profile(|| model.run(&8));
        assert_eq!(output, 9);
        assert_eq!(profiler.stats().samples, 2);
        assert_eq!(
            profiler.stats().min,
            Some(core::time::Duration::from_micros(10))
        );
        assert_eq!(
            profiler.stats().max,
            Some(core::time::Duration::from_micros(10))
        );
        assert_eq!(
            profiler.stats().average(),
            Some(core::time::Duration::from_micros(10))
        );
    }

    #[test]
    fn reset_clears_statistics() {
        let mut profiler = Profiler::with_timer(DummyTimer::new());
        let mut model = DummyModel;

        profiler.profile(|| model.run(&1));
        assert_eq!(profiler.stats().samples, 1);

        profiler.reset_stats();
        assert_eq!(profiler.stats().samples, 0);
        assert_eq!(profiler.stats().min, None);
        assert_eq!(profiler.stats().max, None);
    }

    #[test]
    fn default_timer_builds() {
        let _profiler = Profiler::new();
    }
}