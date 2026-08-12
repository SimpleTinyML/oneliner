mod normalize;
mod signature;

use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

use proc_macro2::Span;
use syn::{ItemStruct, LitStr};

use crate::utils::rust_ident;

pub(crate) use signature::{ModelSignature, TensorArtifact};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ModelFormat {
    Mlir,
    Onnx,
    PytorchExport,
    Tflite,
}

#[derive(Debug)]
pub(crate) enum ModelSource {
    Mlir(PathBuf),
    Onnx { path: PathBuf, imported: PathBuf },
    Pytorch { path: PathBuf, imported: PathBuf },
    Tflite { path: PathBuf, imported: PathBuf },
}

impl ModelSource {
    pub(crate) fn into_paths(self) -> (PathBuf, PathBuf) {
        match self {
            Self::Mlir(path) => (path.clone(), path),
            Self::Onnx { path, imported }
            | Self::Pytorch { path, imported }
            | Self::Tflite { path, imported } => (path, imported),
        }
    }
}

#[derive(Debug)]
pub(crate) struct Model {
    pub(crate) source: ModelSource,
    pub(crate) signature: ModelSignature,
}

pub(crate) fn prepare(model_path: &LitStr, input_struct: &ItemStruct) -> syn::Result<Model> {
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

    if format == ModelFormat::Onnx {
        let signature = signature::load(format, &path, &path)?;
        signature.validate()?;
        let compile_input = normalized_path(&struct_name, &model_stem, "tosa.mlir")?;
        normalize::onnx(&path, &compile_input)?;
        return Ok(Model {
            source: ModelSource::Onnx {
                path,
                imported: compile_input,
            },
            signature,
        });
    }

    let compile_input = match format {
        ModelFormat::Mlir => path.clone(),
        ModelFormat::PytorchExport => {
            let output = normalized_path(&struct_name, &model_stem, "torch.mlir")?;
            normalize::pytorch(&path, &output, &struct_name)?;
            output
        }
        ModelFormat::Tflite => {
            let output = normalized_path(&struct_name, &model_stem, "tosa.mlir")?;
            normalize::tflite(&path, &output)?;
            output
        }
        ModelFormat::Onnx => unreachable!("ONNX is handled before normalization"),
    };
    let signature = signature::load(format, &path, &compile_input)?;
    signature.validate()?;

    let source = match format {
        ModelFormat::Mlir => ModelSource::Mlir(path),
        ModelFormat::PytorchExport => ModelSource::Pytorch {
            path,
            imported: compile_input,
        },
        ModelFormat::Tflite => ModelSource::Tflite {
            path,
            imported: compile_input,
        },
        ModelFormat::Onnx => unreachable!("ONNX is handled before normalization"),
    };
    Ok(Model { source, signature })
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
