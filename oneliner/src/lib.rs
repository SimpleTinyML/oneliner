#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub use oneliner_macro::model;

pub mod runtime;

pub use runtime::{ModelArtifacts, Prediction};
pub type Result<T> = runtime::Result<T>;

/// Builds a `TensorRef` for converter-generated tensor storage.
///
/// Input: generated static byte array or `TensorRef` descriptor name.
/// Output: dispatch-ready `TensorRef`.
#[macro_export]
macro_rules! tensor_ref {
    ($name:ident) => {{
        unsafe { $crate::runtime::tensor_ref_from_raw(core::ptr::addr_of!($name)) }
    }};
}
