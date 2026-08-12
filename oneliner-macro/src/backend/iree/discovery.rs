use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

use proc_macro2::Span;
use syn::Ident;
use walkdir::WalkDir;

use crate::utils::{contains_bytes, has_extension, parse_ident, rust_ident};

pub(super) fn parse_query_function(object_path: &Path) -> syn::Result<(Ident, String)> {
    let header_path = object_path.with_extension("h");
    let header = fs::read_to_string(&header_path)
        .map_err(|error| syn::Error::new(Span::call_site(), error))?;

    let name = header
        .lines()
        .filter_map(query_name_from_line)
        .next()
        .ok_or_else(|| {
            syn::Error::new(
                Span::call_site(),
                format!(
                    "failed to parse IREE query function name from {}",
                    header_path.display()
                ),
            )
        })?;
    let rust_name = rust_ident(name);
    Ok((
        parse_ident(&rust_name, "IREE query function")?,
        name.to_owned(),
    ))
}

fn query_name_from_line(raw_line: &str) -> Option<&str> {
    let line = raw_line.trim();
    if line.is_empty() || line.starts_with("//") || !line.contains("_library_query") {
        return None;
    }

    let (before_paren, _) = line.split_once('(')?;
    let name = before_paren.split_whitespace().last()?.trim_matches('*');
    name.ends_with("_library_query").then_some(name)
}

pub(super) fn find_stream_flow_ir(root: &Path) -> syn::Result<PathBuf> {
    let mut files = WalkDir::new(root)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_file())
        .map(|entry| entry.into_path())
        .filter(|path| has_extension(path, "mlir"))
        .collect::<Vec<_>>();
    files.sort();

    let exact = files
        .iter()
        .filter(|path| {
            path.file_name()
                .and_then(OsStr::to_str)
                .is_some_and(|name| name.ends_with(".10.executable-targets.mlir"))
        })
        .filter(|path| is_converter_compatible_ir(path))
        .cloned()
        .collect::<Vec<_>>();
    if let Some(path) = unique_candidate(exact, "stage-10 executable-targets IR")? {
        return Ok(path);
    }

    let preferred = files
        .iter()
        .filter(|path| {
            path.file_name()
                .and_then(OsStr::to_str)
                .is_some_and(|name| name.contains("iree-hal-prune-executables"))
        })
        .filter(|path| is_converter_compatible_ir(path))
        .cloned()
        .collect::<Vec<_>>();
    if let Some(path) = first_by_pass_order(preferred) {
        return Ok(path);
    }

    first_by_pass_order(
        files
            .into_iter()
            .filter(|path| is_converter_compatible_ir(path))
            .collect(),
    )
    .ok_or_else(|| {
        syn::Error::new(
            Span::call_site(),
            format!(
                "no Stream/Flow IR dump with executable targets was found under {}",
                root.display()
            ),
        )
    })
}

fn unique_candidate(mut candidates: Vec<PathBuf>, label: &str) -> syn::Result<Option<PathBuf>> {
    match candidates.len() {
        0 => Ok(None),
        1 => Ok(candidates.pop()),
        _ => Err(syn::Error::new(
            Span::call_site(),
            format!(
                "multiple {label} files were found: {}",
                candidates
                    .iter()
                    .map(|path| path.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        )),
    }
}

fn first_by_pass_order(mut candidates: Vec<PathBuf>) -> Option<PathBuf> {
    candidates.sort_by_key(|path| (pass_order(path), path.to_string_lossy().into_owned()));
    candidates.into_iter().next()
}

fn is_converter_compatible_ir(path: &Path) -> bool {
    let Ok(bytes) = fs::read(path) else {
        return false;
    };
    contains_bytes(&bytes, b"stream.cmd.execute")
        && contains_bytes(&bytes, b"hal.executable")
        && contains_bytes(&bytes, b"#hal.executable.target")
}

fn pass_order(path: &Path) -> usize {
    path.file_name()
        .and_then(OsStr::to_str)
        .and_then(|name| name.split_once('_').map(|(prefix, _)| prefix))
        .and_then(|prefix| prefix.parse().ok())
        .unwrap_or(usize::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_query_declaration_variants() {
        assert_eq!(
            query_name_from_line("model_library_query("),
            Some("model_library_query")
        );
        assert_eq!(
            query_name_from_line("const void** *model_library_query("),
            Some("model_library_query")
        );
        assert_eq!(query_name_from_line("// model_library_query("), None);
    }

    #[test]
    fn sanitizes_pytorch_query_function_names_for_rust() {
        let raw = "main$async_dispatch_0_library_query";
        let rust_name = rust_ident(raw);

        assert_eq!(rust_name, "main_async_dispatch_0_library_query");
        assert_eq!(
            parse_ident(&rust_name, "test").unwrap().to_string(),
            rust_name
        );
    }

    #[test]
    fn extracts_pass_order() {
        assert_eq!(pass_order(Path::new("113_pass.mlir")), 113);
        assert_eq!(pass_order(Path::new("pass.mlir")), usize::MAX);
    }
}
