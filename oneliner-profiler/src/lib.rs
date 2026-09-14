//! Measure elapsed time around model inference or any other closure.
//!
//! [`Profiler`] records profiling count with total, minimal and maximal time usage in [`LatencyStats`].
//!
//! # Timer selection
//!
//! The default `std` feature uses the timer provided by `std`. For embedded, `no_std` targets, disable default features and select exactly one of `embassy` or `ariel-os`. Selecting both
//! leads to compilation panic. 
//!
//! # Example
//!
//! ```
//! use oneliner_profiler::Profiler;
//! let mut profiler = Profiler::new();
//! let result = profiler.profile(|| [1, 2, 3].iter().sum::<u32>());
//! assert_eq!(result, 6);
//! assert_eq!(profiler.stats().samples, 1);
//! profiler.reset_stats();
//! assert_eq!(profiler.stats().average(), None);
//! ```
//!
//! A measured scope includes all work and waiting inside the closure. Warm-up,
//! input preparation, logging and preemption affect the result if included in
//! that interval. If the closure unwinds, that call is not recorded.
#![no_std]

mod profiler;
mod stats;
mod timer;

pub use profiler::Profiler;
pub use stats::LatencyStats;
pub use timer::{DefaultTimer, Timer};
