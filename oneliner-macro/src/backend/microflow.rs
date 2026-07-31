mod codegen;
mod signature;

use std::path::PathBuf;

/// Expands the Microflow backend for a model struct.
///
/// Input: annotated struct and absolute model path.
/// Output: generated Rust tokens backed by MicroFlow's native buffers.
pub fn expand(
    input_struct: syn::ItemStruct,
    model_path: PathBuf,
) -> syn::Result<proc_macro2::TokenStream> {
    let signature = signature::load_model_signature(&model_path)?;
    Ok(codegen::expand(input_struct, model_path, signature))
}
