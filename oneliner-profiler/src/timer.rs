/// Source of timestamps used to measure inference latency.
///
/// The trait keeps profiling usable on `no_std` targets: implementors read a
/// hardware or OS clock and convert the measured interval into a
/// [`core::time::Duration`]. Provide your own implementation to profile with a
/// custom clock.
pub trait Timer {
    /// A snapshot of the timer's clock.
    type Instant: Copy + core::fmt::Debug;

    /// Takes a timestamp snapshot.
    fn now(&self) -> Self::Instant;

    /// Converts the interval between `start` and the current time into a
    /// [`core::time::Duration`].
    fn elapsed(&self, start: Self::Instant) -> core::time::Duration;
}

/// Timer selected by the profiler crate's enabled features.
///
/// Prefers the host `std` clock, then the Ariel OS clock, then the Embassy
/// clock. [`Profiler::new`](crate::Profiler::new) uses this type so callers do
/// not need to pick a clock explicitly.
#[cfg(feature = "std")]
pub type DefaultTimer = StdTimer;

/// Timer selected by the profiler crate's enabled features.
#[cfg(all(feature = "ariel-os", not(feature = "std")))]
pub type DefaultTimer = ArielOsTimer;

/// Timer selected by the profiler crate's enabled features.
#[cfg(all(feature = "embassy", not(feature = "std"), not(feature = "ariel-os")))]
pub type DefaultTimer = EmbassyTimer;

/// [`Timer`] backed by the host's `std::time::Instant` clock.
#[cfg(feature = "std")]
#[derive(Debug, Clone, Copy, Default)]
pub struct StdTimer;

#[cfg(feature = "std")]
impl Timer for StdTimer {
    type Instant = std::time::Instant;

    fn now(&self) -> Self::Instant {
        std::time::Instant::now()
    }

    fn elapsed(&self, start: Self::Instant) -> core::time::Duration {
        start.elapsed()
    }
}

/// [`Timer`] backed by Ariel OS's monotonic clock.
#[cfg(feature = "ariel-os")]
#[derive(Debug, Clone, Copy, Default)]
pub struct ArielOsTimer;

#[cfg(feature = "ariel-os")]
impl Timer for ArielOsTimer {
    type Instant = ariel_os::time::Instant;

    fn now(&self) -> Self::Instant {
        ariel_os::time::Instant::now()
    }

    fn elapsed(&self, start: Self::Instant) -> core::time::Duration {
        let micros = ariel_os::time::Instant::now().as_micros() - start.as_micros();
        core::time::Duration::from_micros(micros)
    }
}

/// [`Timer`] backed by Embassy's monotonic clock.
#[cfg(feature = "embassy")]
#[derive(Debug, Clone, Copy, Default)]
pub struct EmbassyTimer;

#[cfg(feature = "embassy")]
impl Timer for EmbassyTimer {
    type Instant = embassy_time::Instant;

    fn now(&self) -> Self::Instant {
        embassy_time::Instant::now()
    }

    fn elapsed(&self, start: Self::Instant) -> core::time::Duration {
        let micros = embassy_time::Instant::now().as_micros() - start.as_micros();
        core::time::Duration::from_micros(micros)
    }
}