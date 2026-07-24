mod artifacts;
mod codegen;
mod discovery;
mod metadata;
mod signature;
mod toolchain;

use std::path::PathBuf;

use proc_macro2::TokenStream;
use syn::Ident;

#[derive(Debug)]
struct ArtifactPaths {
    model: PathBuf,
    compile_input: PathBuf,
    object: PathBuf,
    ir: PathBuf,
    flow_rs: PathBuf,
    metadata_json: PathBuf,
}

#[derive(Debug, Clone)]
struct BindingArtifact {
    size: usize,
}

#[derive(Debug)]
struct IreeArtifacts {
    paths: ArtifactPaths,
    query_fn: Ident,
    execute_fns: Vec<Ident>,
    input: BindingArtifact,
    output: BindingArtifact,
    input_tensor: signature::TensorArtifact,
    output_tensor: signature::TensorArtifact,
}

pub fn expand(input_struct: syn::ItemStruct, model_path: PathBuf) -> syn::Result<TokenStream> {
    let artifacts = artifacts::build(&input_struct.ident, model_path)?;

    let expanded = codegen::expand(input_struct, artifacts);
    // eprintln!("generated:\n{expanded}");
    Ok(expanded.into())
}
