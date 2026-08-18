#![no_std]

#[cfg(feature = "alloc")]
extern crate alloc;

pub use oneliner_macro::model;

pub mod runtime {
    pub use oneliner_runtime::*;
}

