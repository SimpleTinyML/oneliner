use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use serde::Deserialize;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::ptr::null;
use syn::{Ident, LitStr};
use walkdir::WalkDir;
use super::llvm_target_info::llvm_target_info_from_rust_triple;

#[derive(Debug)]
struct IreeArtifacts {
    model_path: PathBuf,
    compile_input_path: PathBuf,
    object_path: PathBuf,
    ir_path: PathBuf,
    flow_rs: PathBuf,
    metadata_json: PathBuf,
    execute_count: usize,
    input: Option<BindingArtifact>,
    output: Option<BindingArtifact>,
}

#[derive(Debug, Clone)]
struct BindingArtifact {
    static_ident: String,
    size: usize,
    role: String,
}

#[derive(Debug, Deserialize)]
struct Metadata {
    cmd_executes: Vec<CmdExecute>,
}

#[derive(Debug, Deserialize)]
struct CmdExecute {
    resources: Vec<ResourceBinding>,
}

#[derive(Debug, Deserialize)]
struct ResourceBinding {
    arg: String,
    kind: String,
    size: Option<usize>,
    role: String,
}

/// Expands the IREE backend for a model struct.
///
/// Input: annotated struct and absolute model path.
/// Output: generated Rust tokens after IREE artifacts have been built.
pub fn expand(input_struct: syn::ItemStruct, model_path: PathBuf) -> syn::Result<TokenStream> {
    let artifacts = build_iree_artifacts(&input_struct.ident, model_path)?;
    Ok(expand_iree_items(input_struct, artifacts))
}

/// Renders the Rust items injected by the IREE backend.
///
/// Input: original struct and built artifact metadata.
/// Output: generated Rust tokens implementing model metadata and prediction.
fn expand_iree_items(input_struct: syn::ItemStruct, artifacts: IreeArtifacts) -> TokenStream {
    let struct_ident = &input_struct.ident;
    let module_ident = format_ident!("__oneliner_iree_{}", rust_ident(&struct_ident.to_string()));
    let flow_rs = path_lit(&artifacts.flow_rs);
    let model_path = path_lit(&artifacts.model_path);
    let compile_input_path = path_lit(&artifacts.compile_input_path);
    let object_path = path_lit(&artifacts.object_path);
    let ir_path = path_lit(&artifacts.ir_path);
    let metadata_json_path = path_lit(&artifacts.metadata_json);
    let input_size = artifacts
        .input
        .as_ref()
        .map(|binding| binding.size)
        .unwrap_or(0);
    let output_size = artifacts
        .output
        .as_ref()
        .map(|binding| binding.size)
        .unwrap_or(0);
    let execute_calls = (0..artifacts.execute_count)
        .map(|index| format_ident!("cmd_execute_{}", index))
        .collect::<Vec<_>>();

    let model_stem = artifacts
        .model_path
        .file_stem()
        .and_then(OsStr::to_str).unwrap();

    let query_fn_name = Ident::new(parse_iree_query_function_name_from_object_header(&artifacts.object_path)
        .unwrap_or_else(|_| model_stem.to_string() + "_linked_library_query").as_str(), proc_macro2::Span::call_site());

    let input_write = if let Some(binding) = &artifacts.input {
        let ident = Ident::new(&binding.static_ident, proc_macro2::Span::call_site());
        quote! {
            unsafe {
                let slot = core::ptr::addr_of_mut!(#module_ident::#ident);
                ::OneLiner::runtime::bind_static_input(slot, #input_size, input)?;
            }
        }
    } else {
        quote! {
            let _ = input;
        }
    };
    let output_read = if let Some(binding) = &artifacts.output {
        let ident = Ident::new(&binding.static_ident, proc_macro2::Span::call_site());
        quote! {
            unsafe {
                let src = core::ptr::addr_of!(#module_ident::#ident) as *const u8;
                ::OneLiner::runtime::read_static_output(src, #output_size)
            }
        }
    } else {

        syn::Error::new(
            proc_macro2::Span::call_site(),
            "Cannot locate output binding in IREE metadata; the model may not have any outputs or the converter may have failed to generate them.",
        ).to_compile_error().into()

    };

    quote! {
        use ::OneLiner::runtime::Predict as _;

        #input_struct

        #[allow(improper_ctypes, non_camel_case_types, non_snake_case, non_upper_case_globals)]
        #[link(name = #object_path, kind = "static", modifiers = "+verbatim")]
        unsafe extern "C" {
            // The actual query function is looked up dynamically from the object header
            // and may not be directly callable, but this declaration allows Rust to link
            // the object file without undefined symbol errors.
        }

        #[allow(
            dead_code,
            improper_ctypes,
            mutable_transmutes,
            non_camel_case_types,
            non_snake_case,
            non_upper_case_globals,
            unused_imports,
            unused_macros,
            unused_mut,
            unused_variables
        )]
        mod #module_ident {
            use ::OneLiner::runtime::{
                concurrent, dispatch, dispatch_fn_from_library, fill, Access, TensorRange,
                iree_hal_executable_dispatch_state_v0_t, iree_hal_executable_environment_v0_t,
                iree_hal_executable_library_header_t, iree_hal_executable_workgroup_state_v0_t,
                iree_hal_executable_library_query_fn_t, 
                TensorRef, Aligned, AlignedType,
            };
            use ::OneLiner::tensor_ref;

            unsafe extern "C" {
                pub unsafe fn #query_fn_name(
                    max_version: u32,
                    environment: *const iree_hal_executable_environment_v0_t,
                ) -> *const *const iree_hal_executable_library_header_t;
            }

            static query_fn_ptr: iree_hal_executable_library_query_fn_t = #query_fn_name;

            include!(#flow_rs);
        }

        impl ::OneLiner::runtime::ModelSource for #struct_ident {
            const MODEL_PATH: &'static str = #model_path;
            const ARTIFACTS: ::OneLiner::runtime::ModelArtifacts = ::OneLiner::runtime::ModelArtifacts {
                backend: "iree",
                expansion: "static-flow",
                model_path: #model_path,
                compile_input_path: #compile_input_path,
                object_path: #object_path,
                link_path: #object_path,
                ir_path: #ir_path,
                flow_rs_path: #flow_rs,
                metadata_json_path: #metadata_json_path,
                input_size: #input_size,
                output_size: #output_size,
            };
        }

        impl ::OneLiner::runtime::Predict<[u8]> for #struct_ident {
            type Error = ::OneLiner::runtime::Error;
            type Output = ::OneLiner::runtime::Prediction<'static>;

            /// Runs the generated IREE dispatch flow for one input buffer.
            ///
            /// Input: exact-size model input bytes.
            /// Output: borrowed prediction bytes or runtime error.
            fn try_predict(input: &[u8]) -> ::core::result::Result<Self::Output, Self::Error> {
                #input_write
                #(#module_ident::#execute_calls();)*
                Ok(#output_read)
            }
        }
    }
}

fn parse_iree_query_function_name_from_object_header(object_path: &Path) -> syn::Result<String> {
    let header_path = object_path.with_extension("h");
    validate_file(&header_path, "IREE object header")?;

    let header_text = fs::read_to_string(&header_path).map_err(to_syn_error)?;

    header_text
        .lines()
        .find_map(|raw_line| {
            let line = raw_line.trim();

            // Skip empty lines and comments such as:
            // //  - Query library from main_dispatch_0_library_query()<<
            if line.is_empty() || line.starts_with("//") {
                return None;
            }

            if !line.contains("_library_query") || !line.contains('(') {
                return None;
            }

            let before_paren = line.split('(').next()?.trim();

            // IREE generated header usually has the function name alone:
            // main_dispatch_0_library_query(
            //
            // But this also handles the case where return type and function name
            // are on the same line.
            let function_name = before_paren
                .split_whitespace()
                .last()?
                .trim_start_matches('*')
                .trim_end_matches('*')
                .trim();

            if function_name.ends_with("_library_query") {
                Some(function_name.to_string())
            } else {
                None
            }
        })
        .ok_or_else(|| {
            syn::Error::new(
                proc_macro2::Span::call_site(),
                format!(
                    "failed to parse IREE query function name from {}",
                    header_path.display()
                ),
            )
        })
}

/// Builds all compile-time artifacts needed by the IREE backend.
///
/// Input: model struct name and source model path.
/// Output: paths and binding metadata for generated Rust code.
fn build_iree_artifacts(struct_ident: &Ident, model_path: PathBuf) -> syn::Result<IreeArtifacts> {
    let caller_manifest_dir = std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .ok_or_else(|| {
            syn::Error::new(
                proc_macro2::Span::call_site(),
                "CARGO_MANIFEST_DIR is not set; Cargo must expand #[model] inside a package build",
            )
        })?;
    let out_root = std::env::var_os("OUT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| caller_manifest_dir.join("target").join("oneliner"));
    let model_stem = model_path
        .file_stem()
        .and_then(OsStr::to_str)
        .map(rust_ident)
        .unwrap_or_else(|| rust_ident(&struct_ident.to_string()));
    let artifact_dir = out_root.join(format!(
        "{}_iree_{}",
        rust_ident(&struct_ident.to_string()),
        model_stem
    ));
    let ir_dump_dir = artifact_dir.join("iree-ir-dumps");
    fs::create_dir_all(&ir_dump_dir).map_err(to_syn_error)?;

    let vmfb_path = artifact_dir.join(format!("{model_stem}.vmfb"));
    let object_path = artifact_dir.join(format!("{model_stem}.o"));
    let compile_input_path = if model_path
        .extension()
        .and_then(OsStr::to_str)
        .map(|extension| extension.eq_ignore_ascii_case("tflite"))
        .unwrap_or(false)
    {
        let import_mlir_path = artifact_dir.join(format!("{model_stem}.tosa.mlir"));
        run_tosa_converter_for_tflite(&model_path, &import_mlir_path)?;
        validate_file(&import_mlir_path, "IREE imported TFLite MLIR")?;
        import_mlir_path
    } else {
        model_path.clone()
    };
    run_iree_compile(&compile_input_path, &vmfb_path, &object_path, &ir_dump_dir)?;
    validate_file(&object_path, "IREE object file")?;

    let ir_path = find_stream_flow_ir(&artifact_dir).ok_or_else(|| {
        syn::Error::new(
            proc_macro2::Span::call_site(),
            format!(
                "iree-compile succeeded, but no Stream/Flow IR dump with executable targets was found under {}",
                artifact_dir.display()
            ),
        )
    })?;
    let flow_rs = artifact_dir.join(format!("{model_stem}.flow.rs"));
    let metadata_json = artifact_dir.join(format!("{model_stem}.flow.json"));
    run_converter(&ir_path, &flow_rs, "rust")?;
    run_converter(&ir_path, &metadata_json, "json")?;
    validate_file(&flow_rs, "generated IREE Rust flow")?;
    validate_file(&metadata_json, "generated IREE metadata JSON")?;

    let metadata_text = fs::read_to_string(&metadata_json).map_err(to_syn_error)?;
    let metadata: Metadata = serde_json::from_str(&metadata_text).map_err(to_syn_error)?;
    if metadata.cmd_executes.is_empty() {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            format!(
                "no cmd_execute entries were found in {}",
                metadata_json.display()
            ),
        ));
    }
    let bindings = flatten_bindings(&metadata);
    let input = bindings
        .iter()
        .find(|binding| binding.role == "input" || binding.role == "inout")
        .cloned();
    let output = bindings
        .iter()
        .find(|binding| binding.role == "output" || binding.role == "inout")
        .cloned();

    Ok(IreeArtifacts {
        model_path,
        compile_input_path,
        object_path,
        ir_path,
        flow_rs,
        metadata_json,
        execute_count: metadata.cmd_executes.len(),
        input,
        output,
    })
}

/// Imports a TFLite model into MLIR before compiling with IREE.
///
/// Input: TFLite input path and destination MLIR path.
/// Output: `Ok(())` when the importer succeeds.
fn run_iree_import_tflite(input_path: &Path, output_path: &Path) -> syn::Result<()> {
    let importer = env_first(&["ONELINER_IREE_IMPORT_TFLITE", "IREE_IMPORT_TFLITE"])
        .unwrap_or_else(|| "iree-import-tflite".to_string());
    let mut args = vec![
        input_path.display().to_string(),
        "-o".to_string(),
        output_path.display().to_string(),
    ];

    run_command(Command::new(importer).args(args), "iree-import-tflite")
}

fn run_tosa_converter_for_tflite(input_path: &Path, output_path: &Path) -> syn::Result<()> {
    let converter = "tosa-converter-for-tflite".to_string();
    let args = [
        input_path.display().to_string(),
        "--text".to_string(),
        "-o".to_string(),
        output_path.display().to_string(),
    ];
    run_command(Command::new(converter).args(args), "tosa-converter")
}

/// Runs `iree-compile` to produce VMFB, object file, and IR dumps.
///
/// Input: compiler input path, VMFB path, object path, and IR dump directory.
/// Output: `Ok(())` when compilation succeeds.
fn run_iree_compile(
    compile_input_path: &Path,
    vmfb_path: &Path,
    object_path: &Path,
    ir_dump_dir: &Path,
) -> syn::Result<()> {
    let compiler = env_first(&["ONELINER_IREE_COMPILE", "IREE_COMPILE"])
        .unwrap_or_else(|| "iree-compile".to_string());
    
    let rust_target_triple = rust_target_triple();
    let llvm_target_info = llvm_target_info_from_rust_triple(&rust_target_triple)
        .map_err(|error| syn::Error::new(proc_macro2::Span::call_site(), format!("failed to get LLVM target info for triple {rust_target_triple}: {error}")))?; 
    
    let llvm_triple = &llvm_target_info.llvm_triple;
    let target_cpu = &llvm_target_info.cpu;
    let cpu_features = &llvm_target_info.features;

    // println!("Using LLVM triple: {llvm_triple}");
    // println!("Using target CPU: {}", target_cpu.as_deref().unwrap_or("<empty>"));
    // println!("Using target CPU features: {}", cpu_features.as_deref().unwrap_or("<empty>"));
    let mut args = vec![
        compile_input_path.display().to_string(),
        "--iree-hal-target-device=local".to_string(),
        "--iree-hal-local-target-device-backends=llvm-cpu".to_string(),
        // "--iree-opt-level=O2".to_string(),
        format!("--iree-llvmcpu-target-triple={llvm_triple}"),
        {
            match target_cpu {
                Some(cpu) if !cpu.is_empty() => format!("--iree-llvmcpu-target-cpu={cpu}"),
                _ =>  String::new(),
            }
        },
        {
            match cpu_features {
                Some(features) if !features.is_empty() => format!("--iree-llvmcpu-target-cpu-features={features}"),
                _ => String::new(),
            }
        },
        // "--align-all-functions=4".to_string(),
        // "--align-all-blocks=4".to_string(),
        // "--iree-llvmcpu-stack-allocation-limit=4096".to_string(),
        "--iree-stream-partitioning-favor=min-peak-memory".to_string(),
        // "--iree-vm-bytecode-module-strip-source-map=true".to_string(),
        // "--iree-vm-emit-polyglot-zip=false".to_string(),
        "--iree-llvmcpu-link-embedded=false".to_string(),
        "--iree-llvmcpu-link-static".to_string(),
        format!(
            "--iree-llvmcpu-static-library-output-path={}",
            object_path.display()
        ),

        format!("--dump-compilation-phases-to={}", ir_dump_dir.display()),
        "-o".to_string(),
        vmfb_path.display().to_string(),
    ];
    
    if let Some(extra_args) =
        env_first(&["ONELINER_IREE_COMPILE_FLAGS", "IREE_MODEL_COMPILE_FLAGS"])
    {
        args.extend(extra_args.split_whitespace().map(str::to_string));
    }

    args.retain(| x| !x.is_empty());

    run_command(Command::new(compiler).args(args), "iree-compile")
}

/// Runs the Stream/Flow converter script for Rust or JSON output.
///
/// Input: stage-10 MLIR input, destination path, and output format.
/// Output: `Ok(())` when converter execution succeeds.
fn run_converter(input: &Path, output: &Path, format: &str) -> syn::Result<()> {
    let python = env_first(&["ONELINER_PYTHON", "PYTHON"]).unwrap_or_else(|| "python".to_string());
    let converter = converter_path();
    let args = [
        converter.display().to_string(),
        input.display().to_string(),
        "-o".to_string(),
        output.display().to_string(),
        "--format".to_string(),
        format.to_string(),
    ];
    run_command(
        Command::new(python).args(args),
        "iree_stream_flow_to_rust_using_re.py",
    )
}

/// Executes an external command and maps process failure into `syn::Error`.
///
/// Input: configured command and user-facing display name.
/// Output: `Ok(())` on success or a detailed compiler error.
fn run_command(command: &mut Command, display_name: &str) -> syn::Result<()> {
    let output = command.output().map_err(|error| {
        syn::Error::new(
            proc_macro2::Span::call_site(),
            format!("failed to run {display_name}: {error}"),
        )
    })?;
    if output.status.success() {
        Ok(())
    } else {
        Err(command_error(display_name, output))
    }
}

/// Formats stdout/stderr from a failed external command.
///
/// Input: display name and completed process output.
/// Output: `syn::Error` suitable for macro expansion failure.
fn command_error(display_name: &str, output: Output) -> syn::Error {
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    syn::Error::new(
        proc_macro2::Span::call_site(),
        format!(
            "{display_name} failed with status {}\nstdout:\n{}\nstderr:\n{}",
            output.status, stdout, stderr
        ),
    )
}

/// Locates the Stream/Flow converter script.
///
/// Input: environment variables or the macro crate location.
/// Output: converter path used by `run_converter`.
fn converter_path() -> PathBuf {
    if let Some(path) = env_first(&[
        "ONELINER_IREE_STREAM_FLOW_TO_RUST",
        "IREE_STREAM_FLOW_TO_RUST",
    ]) {
        return PathBuf::from(path);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("iree_stream_flow_to_rust_using_re.py")
}

/// Flattens converter JSON resources into binding metadata.
///
/// Input: parsed converter metadata.
/// Output: external resource bindings with resolved positive sizes.
fn flatten_bindings(metadata: &Metadata) -> Vec<BindingArtifact> {
    metadata
        .cmd_executes
        .iter()
        .flat_map(|execute| execute.resources.iter())
        .filter(|binding| binding.kind == "external" && binding.size.unwrap_or(0) > 0)
        .map(|binding| BindingArtifact {
            static_ident: const_ident(&binding_name(binding)),
            size: binding.size.unwrap_or(0),
            role: binding.role.clone(),
        })
        .collect()
}

/// Builds the generated Rust storage name for one resource binding.
///
/// Input: converter resource binding.
/// Output: stable role-prefixed Rust identifier text.
fn binding_name(binding: &ResourceBinding) -> String {
    match binding.role.as_str() {
        "input" => format!("input_{}", rust_ident(&binding.arg)),
        "output" => format!("output_{}", rust_ident(&binding.arg)),
        "inout" => format!("inout_{}", rust_ident(&binding.arg)),
        "temporary" => format!("temp_{}", rust_ident(&binding.arg)),
        "constant" => format!("const_{}", rust_ident(&binding.arg)),
        role => format!("{}_{}", rust_ident(role), rust_ident(&binding.arg)),
    }
}

/// Finds the Stream/Flow MLIR dump used by the Rust flow converter.
///
/// Input: artifact directory to scan.
/// Output: first compatible IR dump containing stream commands and executable targets.
fn find_stream_flow_ir(root: &Path) -> Option<PathBuf> {
    let files = WalkDir::new(root)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_file())
        .map(|entry| entry.into_path())
        .filter(|path| {
            path.extension()
                .and_then(OsStr::to_str)
                .map(|extension| extension.eq_ignore_ascii_case("mlir"))
                .unwrap_or(false)
        })
        .collect::<Vec<_>>();

    if let Some(path) = files.iter().find(|path| {
        path.file_name()
            .and_then(OsStr::to_str)
            .map(|name| name.ends_with(".10.executable-targets.mlir"))
            .unwrap_or(false)
    }) {
        return Some(path.clone());
    }
    // TODO
    let mut candidates = files
        .iter()
        .filter(|path| {
            path.file_name()
                .and_then(OsStr::to_str)
                .map(|name| name.contains("iree-hal-prune-executables")) 
                .unwrap_or(false)
        })
        .filter(|path| is_converter_compatible_ir(path))
        .cloned()
        .collect::<Vec<_>>();

    if candidates.is_empty() {
        candidates = files
            .into_iter()
            .filter(|path| is_converter_compatible_ir(path))
            .collect();
    }

    candidates.sort_by_key(|path| (pass_order(path), path.to_string_lossy().into_owned()));
    candidates.into_iter().next()
}

/// Checks whether an MLIR dump has the operations the converter needs.
///
/// Input: path to an MLIR dump.
/// Output: `true` when the file contains stream commands and HAL executable targets.
fn is_converter_compatible_ir(path: &Path) -> bool {
    let Ok(bytes) = fs::read(path) else {
        return false;
    };
    contains_bytes(&bytes, b"stream.cmd.execute")
        && contains_bytes(&bytes, b"hal.executable")
        && contains_bytes(&bytes, b"#hal.executable.target")
}

/// Finds a numeric pass prefix in an IREE dump path.
///
/// Input: path such as `113_iree-hal-prune-executables.mlir`.
/// Output: parsed pass order, or `usize::MAX` if no prefix exists.
fn pass_order(path: &Path) -> usize {
    path.file_name()
        .and_then(OsStr::to_str)
        .and_then(|name| name.split_once('_').map(|(prefix, _)| prefix))
        .and_then(|prefix| prefix.parse().ok())
        .unwrap_or(usize::MAX)
}

/// Searches a byte slice without requiring UTF-8 decoding.
///
/// Input: haystack and needle bytes.
/// Output: `true` when `needle` appears in `haystack`.
fn contains_bytes(haystack: &[u8], needle: &[u8]) -> bool {
    haystack
        .windows(needle.len())
        .any(|window| window == needle)
}

/// Resolves the target triple passed to `iree-compile`.
///
/// Input: environment variables or Cargo `TARGET`.
/// Output: target triple string.
fn rust_target_triple() -> String {
    //TODO: temporary fix to get target triple, only tested under ariel-os.
    let env_target = std::env::var("CARGO_BUILD_TARGET").unwrap();
    env_target
}

/// Converts a filesystem path into a Rust string literal for generated code.
///
/// Input: filesystem path.
/// Output: `LitStr` with Windows separators normalized to `/`.
fn path_lit(path: &Path) -> LitStr {
    LitStr::new(
        &path.to_string_lossy().replace('\\', "/"),
        proc_macro2::Span::call_site(),
    )
}

/// Converts any displayable error into a `syn::Error`.
///
/// Input: error value.
/// Output: macro expansion error at call-site span.
fn to_syn_error<T: std::fmt::Display>(error: T) -> syn::Error {
    syn::Error::new(proc_macro2::Span::call_site(), error)
}

/// Verifies that a generated artifact exists and is non-empty.
///
/// Input: path and label used in diagnostics.
/// Output: `Ok(())` if the file is usable, otherwise `syn::Error`.
fn validate_file(path: &Path, label: &str) -> syn::Result<()> {
    if !path.is_file() {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            format!("{label} does not exist: {}", path.display()),
        ));
    }
    if path.metadata().map_err(to_syn_error)?.len() == 0 {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            format!("{label} is empty: {}", path.display()),
        ));
    }
    Ok(())
}

/// Returns the first non-empty environment variable value from a list.
///
/// Input: ordered environment variable names.
/// Output: first non-empty value, if any.
fn env_first(names: &[&str]) -> Option<String> {
    names.iter().find_map(|name| {
        std::env::var(name)
            .ok()
            .filter(|value| !value.trim().is_empty())
    })
}

/// Converts arbitrary text into an uppercase Rust constant identifier.
///
/// Input: raw resource name.
/// Output: sanitized uppercase identifier string.
fn const_ident(raw: &str) -> String {
    rust_ident(raw).to_uppercase()
}

/// Converts arbitrary text into a valid Rust identifier.
///
/// Input: raw name from MLIR, metadata, or filesystem.
/// Output: lowercase Rust identifier, keyword-safe and non-empty.
fn rust_ident(raw: &str) -> String {
    let mut ident = String::new();
    let mut previous_underscore = false;
    for ch in raw.trim().chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            ident.push(ch.to_ascii_lowercase());
            previous_underscore = false;
        } else if !previous_underscore {
            ident.push('_');
            previous_underscore = true;
        }
    }
    let mut ident = ident.trim_matches('_').to_string();
    if ident.is_empty() {
        ident.push_str("value");
    }
    if ident
        .as_bytes()
        .first()
        .map(u8::is_ascii_digit)
        .unwrap_or(false)
    {
        ident.insert_str(0, "v_");
    }
    if is_rust_keyword(&ident) {
        ident.push('_');
    }
    ident
}

/// Checks whether an identifier is a Rust keyword.
///
/// Input: candidate identifier.
/// Output: `true` if the identifier must be renamed.
fn is_rust_keyword(ident: &str) -> bool {
    matches!(
        ident,
        "as" | "break"
            | "const"
            | "continue"
            | "crate"
            | "else"
            | "enum"
            | "extern"
            | "false"
            | "fn"
            | "for"
            | "if"
            | "impl"
            | "in"
            | "let"
            | "loop"
            | "match"
            | "mod"
            | "move"
            | "mut"
            | "pub"
            | "ref"
            | "return"
            | "self"
            | "Self"
            | "static"
            | "struct"
            | "super"
            | "trait"
            | "true"
            | "type"
            | "unsafe"
            | "use"
            | "where"
            | "while"
            | "async"
            | "await"
            | "dyn"
    )
}
