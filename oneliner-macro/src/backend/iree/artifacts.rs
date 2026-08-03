use std::ffi::OsStr;
use std::fs;
use std::path::PathBuf;

use proc_macro2::Span;
use syn::Ident;

use super::super::common::rust_ident;
use super::discovery::{find_stream_flow_ir, parse_query_function, validate_file};
use super::metadata::load_metadata;
use super::signature::{load_model_signature, TensorArtifact};
use super::toolchain::{
    run_converter, run_iree_compile, run_onnx_converter, run_pytorch_importer, run_tosa_converter,
};
use super::{ArtifactPaths, BindingArtifact, IreeArtifacts};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ModelFormat {
    Mlir,
    Onnx,
    PytorchExport,
    Tflite,
}

pub(super) fn build(struct_ident: &Ident, model_path: PathBuf) -> syn::Result<IreeArtifacts> {
    let manifest_dir = required_path_env("CARGO_MANIFEST_DIR")?;
    let out_root = std::env::var_os("OUT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest_dir.join("target").join("oneliner"));
    let struct_name = rust_ident(&struct_ident.to_string());
    let model_stem = model_path
        .file_stem()
        .and_then(OsStr::to_str)
        .map(rust_ident)
        .unwrap_or_else(|| struct_name.clone());
    let artifact_dir = out_root.join(format!("{struct_name}_iree_{model_stem}"));
    let ir_dump_dir = artifact_dir.join("iree-ir-dumps");
    fs::create_dir_all(&ir_dump_dir).map_err(call_site_error)?;

    let vmfb_path = artifact_dir.join(format!("{model_stem}.vmfb"));
    let object_path = artifact_dir.join(format!("{model_stem}.o"));
    let compile_input_path = match detect_model_format(&model_path)? {
        ModelFormat::Tflite => {
            let imported_path = artifact_dir.join(format!("{model_stem}.tosa.mlir"));
            run_tosa_converter(&model_path, &imported_path)?;
            validate_file(&imported_path, "IREE imported TFLite MLIR")?;
            imported_path
        }
        ModelFormat::Onnx => {
            let imported_path = artifact_dir.join(format!("{model_stem}.tosa.mlir"));
            run_onnx_converter(&model_path, &imported_path)?;
            validate_file(&imported_path, "IREE imported ONNX MLIR")?;
            imported_path
        }
        ModelFormat::PytorchExport => {
            let imported_path = artifact_dir.join(format!("{model_stem}.torch.mlir"));
            run_pytorch_importer(&model_path, &imported_path, &struct_name)?;
            validate_file(&imported_path, "IREE imported PyTorch MLIR")?;
            imported_path
        }
        ModelFormat::Mlir => model_path.clone(),
    };

    run_iree_compile(&compile_input_path, &vmfb_path, &object_path, &ir_dump_dir)?;
    validate_file(&object_path, "IREE object file")?;
    let (query_fn, query_link_name) = parse_query_function(&object_path)?;

    let ir_path = find_stream_flow_ir(&artifact_dir)?;
    let flow_rs = artifact_dir.join(format!("{model_stem}.flow.rs"));
    let metadata_json = artifact_dir.join(format!("{model_stem}.flow.json"));
    run_converter(&ir_path, &flow_rs, &metadata_json)?;
    validate_file(&flow_rs, "generated IREE Rust flow")?;
    validate_file(&metadata_json, "generated IREE metadata JSON")?;

    let metadata = load_metadata(&metadata_json)?;
    let signature = load_model_signature(&model_path, &compile_input_path)?;
    let input = metadata
        .input
        .ok_or_else(|| call_site_error("IREE metadata does not contain an input binding"))?;
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

fn detect_model_format(path: &std::path::Path) -> syn::Result<ModelFormat> {
    let extension = path
        .extension()
        .and_then(OsStr::to_str)
        .map(str::to_ascii_lowercase)
        .ok_or_else(|| {
            call_site_error(format!(
                "model path has no supported extension: {}; expected .mlir, .onnx, .pt2, or .tflite",
                path.display()
            ))
        })?;

    match extension.as_str() {
        "mlir" => Ok(ModelFormat::Mlir),
        "onnx" => Ok(ModelFormat::Onnx),
        "pt2" => Ok(ModelFormat::PytorchExport),
        "tflite" => Ok(ModelFormat::Tflite),
        "pt" | "pth" => Err(call_site_error(format!(
            "PyTorch checkpoint '.{extension}' at {} is not a self-contained export; use torch.export.save(..., \"model.pt2\") and pass the .pt2 file to #[model]",
            path.display()
        ))),
        _ => Err(call_site_error(format!(
            "unsupported model format '.{extension}' at {}; expected .mlir, .onnx, .pt2, or .tflite",
            path.display()
        ))),
    }
}

fn validate_tensor_size(
    label: &str,
    binding: &BindingArtifact,
    tensor: &TensorArtifact,
) -> syn::Result<()> {
    let tensor_size = tensor.byte_len().ok_or_else(|| {
        call_site_error(format!(
            "{label} tensor byte size overflows usize for shape {:?}",
            tensor.shape
        ))
    })?;
    if tensor_size != binding.size {
        return Err(call_site_error(format!(
            "{label} tensor {:?} with element width {} occupies {} bytes, but the IREE binding occupies {} bytes",
            tensor.shape,
            tensor.element_type.byte_width(),
            tensor_size,
            binding.size,
        )));
    }
    Ok(())
}

fn required_path_env(name: &str) -> syn::Result<PathBuf> {
    std::env::var_os(name).map(PathBuf::from).ok_or_else(|| {
        syn::Error::new(
            Span::call_site(),
            format!("{name} is not set; Cargo must expand #[model] inside a package build"),
        )
    })
}

fn call_site_error(error: impl std::fmt::Display) -> syn::Error {
    syn::Error::new(Span::call_site(), error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_supported_model_formats_case_insensitively() {
        assert_eq!(
            detect_model_format(std::path::Path::new("model.mlir")).unwrap(),
            ModelFormat::Mlir
        );
        assert_eq!(
            detect_model_format(std::path::Path::new("model.ONNX")).unwrap(),
            ModelFormat::Onnx
        );
        assert_eq!(
            detect_model_format(std::path::Path::new("model.pt2")).unwrap(),
            ModelFormat::PytorchExport
        );
        assert_eq!(
            detect_model_format(std::path::Path::new("model.TFLITE")).unwrap(),
            ModelFormat::Tflite
        );
    }

    #[test]
    fn rejects_ambiguous_pytorch_checkpoint_extensions() {
        let error = detect_model_format(std::path::Path::new("model.pth")).unwrap_err();
        assert!(error.to_string().contains("use torch.export.save"));
    }
}
