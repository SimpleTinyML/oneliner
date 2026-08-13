mod mlir;
mod mlir_tensor;
mod onnx;
mod types;

use std::path::Path;

use proc_macro2::Span;

pub(crate) use types::{ModelIo, TensorInfo};

pub(super) fn load_mlir(path: &Path) -> syn::Result<ModelIo> {
    mlir::load(path)
}

pub(super) fn load_onnx(path: &Path) -> syn::Result<ModelIo> {
    onnx::load(path)
}

pub(super) fn load_tensorflow_metadata(path: &Path) -> syn::Result<ModelIo> {
    let text = std::fs::read_to_string(path).map_err(|error| {
        syn::Error::new(
            Span::call_site(),
            format!("failed to read model I/O from {}: {error}", path.display()),
        )
    })?;
    serde_json::from_str(&text).map_err(|parse_error| error(path, parse_error))
}

fn error(path: &Path, message: impl std::fmt::Display) -> syn::Error {
    syn::Error::new(
        Span::call_site(),
        format!("failed to parse model I/O at {}: {message}", path.display()),
    )
}
