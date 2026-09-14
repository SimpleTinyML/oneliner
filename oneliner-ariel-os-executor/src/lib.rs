//! Ariel OS model inference executor for multi-core workload scheduling.
//!
//! This crate provides `ArielOsExecutor`. The
//! default selects multi-core cocurrent execution when Ariel OS reports more than one core (via [`ariel_os::thread::CORE_COUNT`](https://github.com/ariel-os/ariel-os/blob/b3fe318f90752e3ab94cf65f7c488e6190591490/src/ariel-os-threads/src/lib.rs#L96)),
//! and falls back to equential execution otherwise. Workers, queue and completion counter
//! are global to the executor, so instances share them. Creating an executor
//! does not start a separate worker pool.
//!
//! See the repository's [Ariel OS examples](https://github.com/SimpleTinyML/oneliner/tree/main/examples/ariel-os-minimal) for more details. 
//! 
//! # Features
//! 
//! - `enabled`: Dummy feature to enable the executor implementation for Ariel OS 
//!     and avoid cargo test failed. Must be enabled when using this crate. It is 
//!     automatically set to ture when enabling feature `ariel-os` in crate `oneliner`.
#![no_std]

#[cfg(feature = "enabled")]
mod ariel_os_executor;
#[cfg(feature = "enabled")]
pub use ariel_os_executor::{ArielOsExecutor, DefaultExecutor};
