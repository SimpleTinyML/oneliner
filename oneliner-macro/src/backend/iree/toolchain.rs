use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use std::env;

use proc_macro2::Span;

use super::super::llvm_target_info::llvm_target_info_from_rust_triple;

pub(super) fn run_tosa_converter(input: &Path, output: &Path) -> syn::Result<()> {
    let converter = env_first_os(&["ONELINER_TOSA_CONVERTER_FOR_TFLITE"])
        .unwrap_or_else(|| OsString::from("tosa-converter-for-tflite"));
    let mut command = Command::new(converter);
    command.arg(input).arg("--text").arg("-o").arg(output);
    run_command(&mut command, "tosa-converter-for-tflite")
}

pub(super) fn run_onnx_converter(input: &Path, output: &Path) -> syn::Result<()> {
    let converter = env_first_os(&["ONELINER_IREE_IMPORT_ONNX"])
        .unwrap_or_else(|| OsString::from("iree-import-onnx"));
    let mut command = Command::new(converter);
    command.arg("-o").arg(output).arg(input);
    run_command(&mut command, "iree-import-onnx")
}

pub(super) fn run_pytorch_importer(
    input: &Path,
    output: &Path,
    module_name: &str,
) -> syn::Result<()> {
    let python =
        env_first_os(&["ONELINER_PYTHON", "PYTHON"]).unwrap_or_else(|| OsString::from("python"));
    let importer = pytorch_importer_path();
    let mut command = Command::new(python);
    command
        .arg(importer)
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
    if let Some(path) = env_first_os(&["ONELINER_PYTORCH_IMPORTER"]) {
        return PathBuf::from(path);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("python")
        .join("import_pytorch.py")
}

//TODO: hacky, shall be improved.
fn target_from_process_args() -> Option<OsString> {
    let mut args = env::args_os();

    while let Some(arg) = args.next() {
        if arg == OsStr::new("--target") {
            return args.next();
        }

        if let Some(arg) = arg.to_str() {
            if let Some(target) = arg.strip_prefix("--target=") {
                return Some(target.into());
            }
        }
    }

    None
}

//TODO: hacky, shall be improved.
fn query_rustc_host() -> Option<String> {
    let compiler = env::current_exe().ok()?;

    let output = Command::new(compiler).arg("-vV").output().ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8(output.stdout).ok()?;

    stdout
        .lines()
        .find_map(|line| line.strip_prefix("host: "))
        .map(str::to_owned)
}

pub(super) fn run_iree_compile(
    compile_input: &Path,
    vmfb: &Path,
    object: &Path,
    ir_dump_dir: &Path,
) -> syn::Result<()> {
    let compiler = env_first_os(&["ONELINER_IREE_COMPILE", "IREE_COMPILE"])
        .unwrap_or_else(|| OsString::from("iree-compile"));

    //TODO: hacky, shall be improved.
    let rust_target = match required_env(&["CARGO_BUILD_TARGET", "TARGET"], "Rust target triple") {
        Ok(t) => t,
        Err(_) => target_from_process_args()
            .unwrap_or(query_rustc_host().unwrap().into())
            .into_string()
            .map_err(|_| {
                syn::Error::new(
                    proc_macro2::Span::call_site(),
                    "target argument is not valid UTF-8",
                )
            })?,
    };
    let target_info = llvm_target_info_from_rust_triple(&rust_target).map_err(|error| {
        syn::Error::new(
            Span::call_site(),
            format!("failed to get LLVM target info for Rust target {rust_target}: {error}"),
        )
    })?;
    let llvm_triple =
        env_first(&["ONELINER_IREE_TARGET_TRIPLE"]).unwrap_or(target_info.llvm_triple);
    let target_cpu = env_first(&["ONELINER_IREE_TARGET_CPU"])
        .or(target_info.cpu)
        .filter(|value| !value.is_empty());
    let cpu_features = env_first(&["ONELINER_IREE_CPU_FEATURES"])
        .or(target_info.features)
        .filter(|value| !value.is_empty());

    let mut command = Command::new(compiler);
    command
        .arg(compile_input)
        .arg("--iree-hal-target-device=local")
        .arg("--iree-hal-local-target-device-backends=llvm-cpu")
        .arg(format!("--iree-llvmcpu-target-triple={llvm_triple}"));
    if let Some(cpu) = target_cpu {
        command.arg(format!("--iree-llvmcpu-target-cpu={cpu}"));
    }
    if let Some(features) = cpu_features {
        command.arg(format!("--iree-llvmcpu-target-cpu-features={features}"));
    }
    command
        .arg("--iree-stream-partitioning-favor=min-peak-memory")
        .arg("--iree-llvmcpu-link-embedded=false")
        .arg("--iree-llvmcpu-link-static")
        .arg(format!(
            "--iree-llvmcpu-static-library-output-path={}",
            object.display()
        ))
        .arg(format!(
            "--dump-compilation-phases-to={}",
            ir_dump_dir.display()
        ))
        .arg("-o")
        .arg(vmfb);

    if let Some(extra_args) =
        env_first(&["ONELINER_IREE_COMPILE_FLAGS", "IREE_MODEL_COMPILE_FLAGS"])
    {
        command.args(extra_args.split_whitespace());
    }
    run_command(&mut command, "iree-compile")
}

pub(super) fn run_converter(
    input: &Path,
    rust_output: &Path,
    json_output: &Path,
) -> syn::Result<()> {
    let python =
        env_first_os(&["ONELINER_PYTHON", "PYTHON"]).unwrap_or_else(|| OsString::from("python"));
    let converter = converter_path();
    let mut command = Command::new(python);
    command
        .arg(converter)
        .arg(input)
        .arg("--rust-output")
        .arg(rust_output)
        .arg("--json-output")
        .arg(json_output);
    run_command(&mut command, "IREE Stream/Flow converter")
}

fn converter_path() -> PathBuf {
    if let Some(path) = env_first_os(&[
        "ONELINER_IREE_STREAM_FLOW_TO_RUST",
        "IREE_STREAM_FLOW_TO_RUST",
    ]) {
        return PathBuf::from(path);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("python")
        .join("iree_stream_flow_to_rust_using_re.py")
}

fn run_command(command: &mut Command, display_name: &str) -> syn::Result<()> {
    let rendered_command = format!("{command:?}");
    let output = command.output().map_err(|error| {
        syn::Error::new(
            Span::call_site(),
            format!("failed to run {display_name} ({rendered_command}): {error}"),
        )
    })?;
    if output.status.success() {
        Ok(())
    } else {
        Err(command_error(display_name, &rendered_command, output))
    }
}

fn command_error(display_name: &str, command: &str, output: Output) -> syn::Error {
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    syn::Error::new(
        Span::call_site(),
        format!(
            "{display_name} failed with status {}\ncommand: {command}\nstdout:\n{}\nstderr:\n{}",
            output.status, stdout, stderr
        ),
    )
}

fn required_env(names: &[&str], label: &str) -> syn::Result<String> {
    env_first(names).ok_or_else(|| {
        syn::Error::new(
            Span::call_site(),
            format!("{label} is unavailable; set one of {}", names.join(", ")),
        )
    })
}

fn env_first(names: &[&str]) -> Option<String> {
    names.iter().find_map(|name| {
        std::env::var(name)
            .ok()
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty())
    })
}

fn env_first_os(names: &[&str]) -> Option<OsString> {
    names.iter().find_map(|name| {
        std::env::var_os(name).filter(|value| !value.to_string_lossy().trim().is_empty())
    })
}
