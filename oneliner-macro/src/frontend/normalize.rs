use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::utils::{env_first_os, run_command};

pub(super) fn tflite(input: &Path, output: &Path) -> syn::Result<()> {
    let converter = env_first_os(&["ONELINER_TOSA_CONVERTER_FOR_TFLITE"])
        .unwrap_or_else(|| OsString::from("tosa-converter-for-tflite"));
    let mut command = Command::new(converter);
    command.arg(input).arg("--text").arg("-o").arg(output);
    run_command(&mut command, "tosa-converter-for-tflite")
}

pub(super) fn onnx(input: &Path, output: &Path) -> syn::Result<()> {
    let converter = env_first_os(&["ONELINER_IREE_IMPORT_ONNX"])
        .unwrap_or_else(|| OsString::from("iree-import-onnx"));
    let mut command = Command::new(converter);
    command.arg("-o").arg(output).arg(input);
    run_command(&mut command, "iree-import-onnx")
}

pub(super) fn pytorch(input: &Path, output: &Path, module_name: &str) -> syn::Result<()> {
    let python =
        env_first_os(&["ONELINER_PYTHON", "PYTHON"]).unwrap_or_else(|| OsString::from("python"));
    let mut command = Command::new(python);
    command
        .arg(pytorch_importer_path())
        .arg(input)
        .arg("--output")
        .arg(output)
        .arg("--module-name")
        .arg(module_name);
    if let Some(model_dir) = input.parent() {
        command.current_dir(model_dir);
    }
    run_command(&mut command, "PyTorch ExportedProgram importer")
}

fn pytorch_importer_path() -> PathBuf {
    env_first_os(&["ONELINER_PYTORCH_IMPORTER"])
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("python")
                .join("import_pytorch.py")
        })
}
