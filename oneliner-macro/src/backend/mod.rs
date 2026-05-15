mod iree;
mod microflow;

use std::path::PathBuf;
use syn::ItemStruct;

use crate::args::{BackendArg, ModelArgs};

/// Resolves model path and delegates macro expansion to the selected backend.
///
/// Input: parsed macro arguments and the annotated struct.
/// Output: generated Rust tokens or a `syn::Error` for invalid input/backend failure.
pub fn expand(args: ModelArgs, input_struct: ItemStruct) -> syn::Result<proc_macro2::TokenStream> {
    let span = args.model_path.span();
    let caller_manifest_dir = std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .ok_or_else(|| {
            syn::Error::new(
                span,
                "CARGO_MANIFEST_DIR is not set; Cargo must expand #[model] inside a package build",
            )
        })?;
    let model_path = PathBuf::from(args.model_path.value());
    let model_path = if model_path.is_absolute() {
        model_path
    } else {
        caller_manifest_dir.join(model_path)
    };

    if !model_path.exists() {
        return Err(syn::Error::new(
            span,
            format!("model file does not exist: {}", model_path.display()),
        ));
    }

    match args.backend {
        BackendArg::Iree => iree::expand(input_struct, model_path),
        BackendArg::Microflow => Ok(microflow::expand(input_struct, model_path)),
    }
}
