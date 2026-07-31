mod common;
mod iree;
mod llvm_target_info;
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
    if matches!(args.backend, BackendArg::Microflow) {
        if let Some(arena_span) = args.arena_span {
            return Err(syn::Error::new(
                arena_span,
                "the arena option is only supported by the IREE backend",
            ));
        }
    }
    if !input_struct.generics.params.is_empty() || input_struct.generics.where_clause.is_some() {
        return Err(syn::Error::new_spanned(
            &input_struct.generics,
            "#[model] currently supports only non-generic structs",
        ));
    }
    if !matches!(input_struct.fields, syn::Fields::Unit) {
        return Err(syn::Error::new_spanned(
            &input_struct.fields,
            "#[model] must be applied to a unit struct",
        ));
    }
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

    if !model_path.is_file() {
        return Err(syn::Error::new(
            span,
            format!("model path is not a file: {}", model_path.display()),
        ));
    }

    match args.backend {
        BackendArg::Iree => iree::expand(input_struct, model_path, args.arena),
        BackendArg::Microflow => microflow::expand(input_struct, model_path),
    }
}
