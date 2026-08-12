mod model_io;
mod normalize;

use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

use proc_macro2::Span;
use syn::{ItemStruct, LitStr};

use crate::utils::rust_ident;

pub(crate) use model_io::{ModelIo, TensorInfo};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ModelFormat {
    Mlir,
    Onnx,
    PytorchExport,
    Tflite,
}

#[derive(Debug)]
/// A frontend-prepared model ready for backend compilation.
pub(crate) struct Model {
    /// Original model path supplied to `#[model]`.
    pub(crate) source_path: PathBuf,
    /// Model TOSA MLIR consumed by the backend compiler.
    pub(crate) compile_input_path: PathBuf,
    /// File stem of `compile_input_path`.
    pub(crate) ir_dump_stem: String,
    /// Validated model input and output tensor metadata.
    pub(crate) model_io: ModelIo,
}

pub(crate) fn prepare(model_path: &LitStr, input_struct: &ItemStruct) -> syn::Result<Model> {
    let caller_manifest_dir = std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .ok_or_else(|| {
            syn::Error::new(
                model_path.span(),
                "CARGO_MANIFEST_DIR is not set; Cargo must expand #[model] inside a package build",
            )
        })?;
    let path = PathBuf::from(model_path.value());
    let path = if path.is_absolute() {
        path
    } else {
        caller_manifest_dir.join(path)
    };
    let struct_ident = &input_struct.ident;
    if !path.is_file() {
        return Err(syn::Error::new(
            model_path.span(),
            format!("model path is not a file: {}", path.display()),
        ));
    }

    let format = detect_format(&path)?;
    let struct_name = rust_ident(&struct_ident.to_string());
    let model_stem = path
        .file_stem()
        .and_then(OsStr::to_str)
        .map(rust_ident)
        .unwrap_or_else(|| struct_name.clone());

    let (compile_input_path, ir_dump_stem, model_io) = match format {
        ModelFormat::Mlir => {
            let model_io = model_io::load_mlir(&path)?;
            (path.clone(), model_stem.clone(), model_io)
        }
        ModelFormat::Onnx => {
            let model_io = model_io::load_onnx(&path)?;
            let output = normalized_path(&struct_name, &model_stem, "tosa.mlir")?;
            normalize::onnx(&path, &output)?;
            let ir_dump_stem = rust_ident(output.file_stem().and_then(OsStr::to_str).unwrap());
            (output, ir_dump_stem, model_io)
        }
        ModelFormat::PytorchExport => {
            let output = normalized_path(&struct_name, &model_stem, "torch.mlir")?;
            normalize::pytorch(&path, &output, &struct_name)?;
            let model_io = model_io::load_mlir(&output)?;
            (output, struct_name, model_io)
        }
        ModelFormat::Tflite => {
            let output = normalized_path(&struct_name, &model_stem, "tosa.mlir")?;
            normalize::tflite(&path, &output)?;
            let model_io = model_io::load_mlir(&output)?;
            let ir_dump_stem = rust_ident(output.file_stem().and_then(OsStr::to_str).unwrap());
            (output, ir_dump_stem, model_io)
        }
    };
    model_io.validate()?;

    Ok(Model {
        source_path: path,
        compile_input_path,
        ir_dump_stem,
        model_io,
    })
}

fn normalized_path(struct_name: &str, model_stem: &str, suffix: &str) -> syn::Result<PathBuf> {
    let manifest_dir = std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .ok_or_else(|| {
            syn::Error::new(
                Span::call_site(),
                "CARGO_MANIFEST_DIR is not set; Cargo must expand #[model] inside a package build",
            )
        })?;
    let out_root = std::env::var_os("OUT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest_dir.join("target").join("oneliner"));
    let output_dir = out_root
        .join("frontend")
        .join(format!("{struct_name}_{model_stem}"));
    fs::create_dir_all(&output_dir).map_err(|error| syn::Error::new(Span::call_site(), error))?;
    Ok(output_dir.join(format!("{model_stem}.{suffix}")))
}

fn detect_format(path: &Path) -> syn::Result<ModelFormat> {
    let extension = path
        .extension()
        .and_then(OsStr::to_str)
        .map(str::to_ascii_lowercase)
        .ok_or_else(|| {
            syn::Error::new(
                Span::call_site(),
                format!(
                    "model path has no supported extension: {}; expected .mlir, .onnx, .pt2, or .tflite",
                    path.display()
                ),
            )
        })?;

    match extension.as_str() {
        "mlir" => Ok(ModelFormat::Mlir),
        "onnx" => Ok(ModelFormat::Onnx),
        "pt2" => Ok(ModelFormat::PytorchExport),
        "tflite" => Ok(ModelFormat::Tflite),
        "pt" | "pth" => Err(syn::Error::new(
            Span::call_site(),
            format!(
                "PyTorch checkpoint '.{extension}' at {} is not a self-contained export; use torch.export.save(..., \"model.pt2\") and pass the .pt2 file to #[model]",
                path.display()
            ),
        )),
        _ => Err(syn::Error::new(
            Span::call_site(),
            format!(
                "unsupported model format '.{extension}' at {}; expected .mlir, .onnx, .pt2, or .tflite",
                path.display()
            ),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_supported_model_formats_case_insensitively() {
        assert_eq!(
            detect_format(Path::new("model.mlir")).unwrap(),
            ModelFormat::Mlir
        );
        assert_eq!(
            detect_format(Path::new("model.ONNX")).unwrap(),
            ModelFormat::Onnx
        );
        assert_eq!(
            detect_format(Path::new("model.pt2")).unwrap(),
            ModelFormat::PytorchExport
        );
        assert_eq!(
            detect_format(Path::new("model.TFLITE")).unwrap(),
            ModelFormat::Tflite
        );
    }

    #[test]
    fn rejects_ambiguous_pytorch_checkpoint_extensions() {
        let error = detect_format(Path::new("model.pth")).unwrap_err();
        assert!(error.to_string().contains("use torch.export.save"));
    }
}
