use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

use proc_macro2::Span;

use super::super::llvm_target_info::llvm_target_info_from_rust_triple;
use crate::utils::{
    env_first, env_first_os, query_rustc_host, required_env, run_command, target_from_process_args,
};

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
