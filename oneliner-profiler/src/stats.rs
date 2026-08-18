use core::fmt;
use core::time::Duration;

/// Accumulated latency measurements for one profiled model.
///
/// Statistics are cheap to keep in `no_std` firmware: one counter plus running
/// total, minimum, and maximum durations.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LatencyStats {
    /// Number of recorded inference calls.
    pub samples: u64,
    /// Sum of all recorded durations.
    pub total: Duration,
    /// Shortest recorded duration.
    pub min: Option<Duration>,
    /// Longest recorded duration.
    pub max: Option<Duration>,
}

impl LatencyStats {
    /// Records one measured inference duration.
    pub fn record(&mut self, elapsed: Duration) {
        self.samples += 1;
        self.total += elapsed;
        self.min = Some(self.min.map_or(elapsed, |min| min.min(elapsed)));
        self.max = Some(self.max.map_or(elapsed, |max| max.max(elapsed)));
    }

    /// Average duration per sample, or `None` when nothing has been recorded.
    pub fn average(&self) -> Option<Duration> {
        if self.samples == 0 {
            return None;
        }
        let avg_nanos = (self.total.as_nanos() / self.samples as u128) as u64;
        Some(Duration::from_nanos(avg_nanos))
    }

    /// Average latency in microseconds, or `0` when nothing has been recorded.
    pub fn average_micros(&self) -> u64 {
        self.average().map_or(0, |duration| duration.as_micros() as u64)
    }

    /// Shortest latency in microseconds, or `0` when nothing has been recorded.
    pub fn min_micros(&self) -> u64 {
        self.min.map_or(0, |duration| duration.as_micros() as u64)
    }

    /// Longest latency in microseconds, or `0` when nothing has been recorded.
    pub fn max_micros(&self) -> u64 {
        self.max.map_or(0, |duration| duration.as_micros() as u64)
    }

    /// Resets all statistics to their initial state.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

impl fmt::Display for LatencyStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "samples={}", self.samples)?;
        if let Some(avg) = self.average() {
            write!(
                f,
                " avg={} min={} max={}",
                FormatDuration(avg),
                FormatDuration(self.min.unwrap_or_default()),
                FormatDuration(self.max.unwrap_or_default()),
            )?;
        }
        Ok(())
    }
}

/// Logs latency statistics through the `defmt` facade used on embedded boards.
impl defmt::Format for LatencyStats {
    fn format(&self, fmt: defmt::Formatter) {
        defmt::write!(
            fmt,
            "samples={} avg_us={} min_us={} max_us={}",
            self.samples,
            self.average_micros(),
            self.min_micros(),
            self.max_micros(),
        );
    }
}

/// Formats a duration using a human-friendly unit (us, ms, or s).
struct FormatDuration(Duration);

impl fmt::Display for FormatDuration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let duration = self.0;
        if duration.as_secs() >= 1 {
            write!(f, "{}.{:03}s", duration.as_secs(), duration.subsec_millis())
        } else if duration.as_millis() >= 1 {
            write!(f, "{}ms", duration.as_millis())
        } else {
            write!(f, "{}us", duration.as_micros())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate std;

    #[test]
    fn tracks_min_max_total_and_average() {
        let mut stats = LatencyStats::default();
        assert_eq!(stats.average(), None);

        stats.record(Duration::from_micros(100));
        stats.record(Duration::from_micros(200));
        stats.record(Duration::from_micros(150));

        assert_eq!(stats.samples, 3);
        assert_eq!(stats.total, Duration::from_micros(450));
        assert_eq!(stats.min, Some(Duration::from_micros(100)));
        assert_eq!(stats.max, Some(Duration::from_micros(200)));
        assert_eq!(stats.average(), Some(Duration::from_micros(150)));
    }

    #[test]
    fn reset_clears_statistics() {
        let mut stats = LatencyStats::default();
        stats.record(Duration::from_micros(10));
        stats.reset();
        assert_eq!(stats, LatencyStats::default());
    }

#[cfg(feature = "std")]
fn formats_durations() {
    use std::string::ToString;

    assert_eq!(FormatDuration(Duration::from_micros(5)).to_string(), "5us");
    assert_eq!(
        FormatDuration(Duration::from_millis(12)).to_string(),
        "12ms"
    );
    assert_eq!(
        FormatDuration(Duration::from_secs(3)).to_string(),
        "3.000s"
    );
}
}