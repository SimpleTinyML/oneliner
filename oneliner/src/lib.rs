//! Oneline eases TinyML model inference on low-power devices with one-line code. Its main focus is `no_std` embedded targets.
//! For the verified host and embedded targets, see [Target support](https://github.com/SimpleTinyML/oneliner/blob/main/docs/target_support.md).
//! 
//! # Getting started
//!
//! Install the [host toolchain](https://github.com/SimpleTinyML/oneliner/blob/main/docs/installation.md)
//! before using Oneliner. The following snippet uses an LeNet5 model file. 
//! See [model zoo](https://github.com/SimpleTinyML/oneliner/tree/main/examples/models).
//!
//! ```ignore (requires the model file and the IREE toolchain)
//! use oneliner::{model, runtime::ModelInference};
//!
//! #[model("models/lenet5_quantized.tflite")]
//! struct MyModel; 
//!
//! let mut model = MyModel::new();
//! let mut input = MyModel::create_input_tensor();
//! input.fill(0); // Use values and preprocessing appropriate for your model.
//! let output = model.run(&input);
//! ```
//!
//! The proc macro [`model`] expands struct `MyModel` and implements [`runtime::ModelInference`] for model inference control.
//! The model `input` and `output` implements [`runtime::Tensor`] for tensor-like operation.
//! 
//! # Accepted Model Formats
//! 
//! The Oneliner's frontend accepts TFLite, ONNX, PyTorch [ExportedProgram `.pt2`](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/export/pt2_archive.html), and TensorFlow SavedModel v2 models. 
//! Onliner automatically recognizes model format by file extension.
//! Model paths are resolved relative to the application's `Cargo.toml`.
//! For more details, please check [model formats](https://github.com/SimpleTinyML/oneliner/blob/main/docs/model_formats.md).
//!
//! # Features
//! 
//! - `ariel-os` : Use Ariel OS's multi-core executor and benchmarking utilities.
//! - `alloc` : Enable heap-allocation for "owned" model workspace. Needs a global allocator supplied by the application.
//! - `ndarray` : Enable ndarray views for tensors; ndarray depends on `alloc`.
//!

#![no_std]

#[cfg(feature = "alloc")]
extern crate alloc;

#[doc(inline)]
pub use oneliner_macro::model;

/// Tensor, arena, inference and backend interfaces used by generated models.
pub mod runtime {
    #[doc(inline)]
    pub use oneliner_runtime::*;
}
