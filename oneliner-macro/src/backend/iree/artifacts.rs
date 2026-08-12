use std::ffi::OsStr;
use std::fs;
use std::path::PathBuf;

use proc_macro2::Span;
use syn::Ident;

use super::discovery::parse_query_function;
use super::metadata::load_metadata;
use super::toolchain::{run_converter, run_iree_compile};
use super::{ArtifactPaths, BindingArtifact, IreeArtifacts};
use crate::frontend::{Model, TensorArtifact};
use crate::utils::{required_path_env, rust_ident};

pub(super) fn build(struct_ident: &Ident, model: Model) -> syn::Result<IreeArtifacts> {
    let Model {
        source_path: model_path,
        compile_input_path,
        ir_dump_stem,
        signature,
    } = model;
    let struct_name = rust_ident(&struct_ident.to_string());
    let manifest_dir = required_path_env("CARGO_MANIFEST_DIR")?;
    let out_root = std::env::var_os("OUT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest_dir.join("target").join("oneliner"));
    let model_stem = model_path
        .file_stem()
        .and_then(OsStr::to_str)
        .map(rust_ident)
        .unwrap_or_else(|| struct_name.clone());
    let artifact_dir = out_root.join(format!("{struct_name}_iree_{model_stem}"));
    let ir_dump_dir = artifact_dir.join("iree-ir-dumps");
    fs::create_dir_all(&ir_dump_dir).map_err(|error| syn::Error::new(Span::call_site(), error))?;

    let vmfb_path = artifact_dir.join(format!("{model_stem}.vmfb"));
    let object_path = artifact_dir.join(format!("{model_stem}.o"));

    run_iree_compile(&compile_input_path, &vmfb_path, &object_path, &ir_dump_dir)?;
    let (query_fn, query_link_name) = parse_query_function(&object_path)?;

    let ir_path = ir_dump_dir.join(format!("{ir_dump_stem}.10.executable-targets.mlir"));
    let flow_rs = artifact_dir.join(format!("{model_stem}.flow.rs"));
    let metadata_json = artifact_dir.join(format!("{model_stem}.flow.json"));
    run_converter(&ir_path, &flow_rs, &metadata_json)?;

    let metadata = load_metadata(&metadata_json)?;
    let input = metadata.input.ok_or_else(|| {
        syn::Error::new(
            Span::call_site(),
            "IREE metadata does not contain an input binding",
        )
    })?;
    validate_tensor_size("input", &input, &signature.input)?;
    validate_tensor_size("output", &metadata.output, &signature.output)?;

    Ok(IreeArtifacts {
        paths: ArtifactPaths {
            model: model_path,
            compile_input: compile_input_path,
            object: object_path,
            ir: ir_path,
            flow_rs,
            metadata_json,
        },
        query_fn,
        query_link_name,
        execute_fns: metadata.execute_fns,
        input,
        output: metadata.output,
        input_tensor: signature.input,
        output_tensor: signature.output,
    })
}

fn validate_tensor_size(
    label: &str,
    binding: &BindingArtifact,
    tensor: &TensorArtifact,
) -> syn::Result<()> {
    let tensor_size = tensor
        .byte_len()
        .expect("frontend validated tensor byte size");
    if tensor_size != binding.size {
        return Err(syn::Error::new(
            Span::call_site(),
            format!(
                "{label} tensor {:?} with element width {} occupies {} bytes, but the IREE binding occupies {} bytes",
                tensor.shape,
                tensor.element_type.byte_width(),
                tensor_size,
                binding.size,
            ),
        ));
    }
    Ok(())
}
