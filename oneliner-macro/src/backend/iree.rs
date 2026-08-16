mod artifacts;
mod codegen;
mod discovery;
mod metadata;
mod toolchain;

use std::path::PathBuf;

use proc_macro2::TokenStream;
use syn::Ident;

use crate::args::ArenaArg;
use crate::frontend::{Model, TensorInfo};

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
    query_link_name: String,
    execute_fns: Vec<Ident>,
    input: BindingArtifact,
    output: BindingArtifact,
    input_tensor: TensorInfo,
    output_tensor: TensorInfo,
    flash_size: usize,
    ram_size: usize,
}

pub fn expand(
    input_struct: syn::ItemStruct,
    model: Model,
    arena: ArenaArg,
) -> syn::Result<TokenStream> {
    let artifacts = artifacts::build(&input_struct.ident, model)?;

    eprintln!(
        "[oneliner-profiler] {}: flash = {} B ({} KiB), ram = {} B ({} KiB), input = {} B, output = {} B",
        input_struct.ident,
        artifacts.flash_size,
        artifacts.flash_size / 1024,
        artifacts.ram_size,
        artifacts.ram_size / 1024,
        artifacts.input.size,
        artifacts.output.size,
    );

    let expanded = codegen::expand(input_struct, artifacts, arena);
    // eprintln!("generated:\n{expanded}");
    Ok(expanded.into())
}
