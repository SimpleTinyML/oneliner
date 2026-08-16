#![no_std]

#[cfg(feature = "std")]
extern crate std;

mod profiler;
mod stats;
mod timer;

pub use profiler::Profiler;
pub use stats::LatencyStats;
#[cfg(feature = "ariel-os")]
pub use timer::ArielOsTimer;
pub use timer::{DefaultTimer, Timer};
#[cfg(feature = "embassy")]
pub use timer::EmbassyTimer;
#[cfg(feature = "std")]
pub use timer::StdTimer;