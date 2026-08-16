#![no_std]

#[cfg(feature = "enabled")]
mod ariel_os_executor;
#[cfg(feature = "enabled")]
pub use ariel_os_executor::{ArielOsExecutor, DefaultExecutor};
