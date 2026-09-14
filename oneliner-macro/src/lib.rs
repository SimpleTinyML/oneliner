//! Onliner's model import frontend and compilation+codegen backend.
//!
//! Applications normally uses public entry point [`model`] from the `oneliner` facade. This
//! procedural macro invokes model compiler (currently IREE only) and expands
//! the wrapped struct to implement `oneliner::runtime::ModelInference` trait.
//!
//! The frontend resolves the model format and converts model into compiler-compatible form. The types and shapes of model's input/output are identified and normalized as well. The compiler transpiles a model into a executable library for specific target.

mod args;
mod backend;
mod frontend;
mod utils;

use proc_macro::TokenStream;
use syn::{AttributeArgs, ItemStruct, parse_macro_input};

/// Expands `#[model("path/to/model", args)]` on a struct into backend-specific model bindings.
///
/// # Arguments
///
/// | Argument | Accepted values | Default |
/// | --- | --- | --- |
/// | path/to/model | Path string to Model file or Tensorflow SavedModel directory | Required |
/// | `backend` | `"iree"` | `"iree"` |
/// | `arena` | `"owned"`, `"shared"` | `"owned"` |
/// | `format` | `"mlir"`, `"onnx"`, `"pytorch"`/`"pt2"`, `"tensorflow"`/`"tf"`, `"tflite"` | None, Oneliner identifies via file extension. |
///
/// Relative paths resolve against the application's `Cargo.toml` directory.
/// 
/// A TensorFlow SavedModel directory requires `format = "tensorflow"`.
/// PyTorch input must be an exported `.pt2` program, not a `.pt`/`.pth` checkpoint.
/// For more details, please check [model formats](https://github.com/SimpleTinyML/oneliner/blob/main/docs/model_formats.md).
///
/// # Remarks on `arena`
///
/// `owned` gives each model instance an independent work arena. `shared` gives all instances one shared
/// static arena. Please check [memory model](https://github.com/SimpleTinyML/oneliner/blob/main/docs/memory_model.md)
/// for more details.
///
/// # Example
///
/// This snippet requires an application-provided model.
///
/// ```ignore (requires the application's model file and the IREE toolchain)
/// use oneliner::{model, runtime::{ModelInference, ModelSource}};
/// #[model("models/model.tflite", arena = "shared")]
/// struct MyModel;
/// let mut model = MyModel::new();
/// let input = MyModel::create_input_tensor();
/// let output = model.run(&input);
/// println!("{} workspace arena bytes", MyModel::ARTIFACTS.ram_size);
/// ```
///
/// # Toolchain requirements
///
/// Macro expansion invokes host tools and writes generated artifacts. Install the
/// IREE/Python toolchain and any format-specific importer first. For more details,
/// please check [host toolchain](https://github.com/SimpleTinyML/oneliner/blob/main/docs/installation.md).
#[proc_macro_attribute]
pub fn model(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attr = parse_macro_input!(attr as AttributeArgs);
    let input_struct = parse_macro_input!(item as ItemStruct);

    let expanded = args::ModelArgs::parse(attr).and_then(|args| {
        let model = frontend::prepare(&args, &input_struct)?;
        backend::expand(args.backend, args.arena, input_struct, model)
    });
    match expanded {
        Ok(tokens) => tokens.into(),
        Err(error) => error.into_compile_error().into(),
    }
}
