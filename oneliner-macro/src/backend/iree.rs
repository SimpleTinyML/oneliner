mod artifacts;
mod codegen;
mod discovery;
mod metadata;
mod object_size;
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
    params_size: usize,
    code_size: usize,
    rodata_size: usize,
    ram_size: usize,
}

pub fn expand(
    input_struct: syn::ItemStruct,
    model: Model,
    arena: ArenaArg,
) -> syn::Result<TokenStream> {
    let artifacts = artifacts::build(&input_struct.ident, model)?;

    let params_size = artifacts.params_size;
    let code_size = artifacts.code_size;
    let rodata_size = artifacts.rodata_size;
    let total_flash_size = params_size + code_size + rodata_size;
    let ram_size = artifacts.ram_size;
    let input_size = artifacts.input.size;
    let output_size = artifacts.output.size;

    eprintln!("[oneliner-profiler] {} memory footprint:", input_struct.ident);
    eprintln!(
        "  Flash Usage: params = {} B ({} KiB), text(code) = {} B ({} KiB), rodata = {} B ({} KiB), total = {} B ({} KiB)",
        params_size,
        params_size / 1024,
        code_size,
        code_size / 1024,
        rodata_size,
        rodata_size / 1024,
        total_flash_size,
        total_flash_size / 1024,
    );
    eprintln!(
        "  RAM Usage: arena = {} B ({} KiB), input = {} B ({} KiB), output = {} B ({} KiB)",
        ram_size,
        ram_size / 1024,
        input_size,
        input_size / 1024,
        output_size,
        output_size / 1024,
    );

    let expanded = codegen::expand(input_struct, artifacts, arena);
    // eprintln!("generated:\n{expanded}");
    Ok(expanded.into())
}
