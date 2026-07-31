#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub use oneliner_macro::model;

pub mod runtime;

/// Implementation details used by code generated from `#[model]`.
#[doc(hidden)]
pub mod __private {
    #[cfg(feature = "microflow-runtime")]
    pub use microflow;
}

pub use runtime::ModelArtifacts;
