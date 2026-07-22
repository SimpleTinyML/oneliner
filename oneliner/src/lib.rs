#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub use oneliner_macro::model;

pub mod runtime;

pub use runtime::{ModelArtifacts, Prediction};

