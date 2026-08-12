use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use proc_macro2::Span;
use syn::{Ident, LitStr};

pub(crate) fn path_lit(path: &Path) -> LitStr {
    LitStr::new(
        &path.to_string_lossy().replace('\\', "/"),
        Span::call_site(),
    )
}

pub(crate) fn parse_ident(raw: &str, context: &str) -> syn::Result<Ident> {
    syn::parse_str(raw).map_err(|error| {
        syn::Error::new(
            Span::call_site(),
            format!("invalid Rust identifier '{raw}' for {context}: {error}"),
        )
    })
}

pub(crate) fn rust_ident(raw: &str) -> String {
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
    if ident.as_bytes().first().is_some_and(u8::is_ascii_digit) {
        ident.insert_str(0, "v_");
    }
    if is_rust_keyword(&ident) {
        ident.push('_');
    }
    ident
}

pub(crate) fn query_rustc_host() -> Option<String> {
    let compiler = std::env::current_exe().ok()?;
    let output = Command::new(compiler).arg("-vV").output().ok()?;
    if !output.status.success() {
        return None;
    }

    String::from_utf8(output.stdout)
        .ok()?
        .lines()
        .find_map(|line| line.strip_prefix("host: "))
        .map(str::to_owned)
}

pub(crate) fn required_path_env(name: &str) -> syn::Result<PathBuf> {
    std::env::var_os(name).map(PathBuf::from).ok_or_else(|| {
        syn::Error::new(
            Span::call_site(),
            format!("{name} is not set; Cargo must expand #[model] inside a package build"),
        )
    })
}

pub(crate) fn run_command(command: &mut Command, display_name: &str) -> syn::Result<()> {
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

pub(crate) fn target_from_process_args() -> Option<OsString> {
    let mut args = std::env::args_os();
    while let Some(arg) = args.next() {
        if arg == OsStr::new("--target") {
            return args.next();
        }
        if let Some(target) = arg.to_str().and_then(|arg| arg.strip_prefix("--target=")) {
            return Some(target.into());
        }
    }
    None
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

fn is_rust_keyword(ident: &str) -> bool {
    matches!(
        ident,
        "as" | "async"
            | "await"
            | "break"
            | "const"
            | "continue"
            | "crate"
            | "dyn"
            | "else"
            | "enum"
            | "extern"
            | "false"
            | "fn"
            | "for"
            | "gen"
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
            | "try"
            | "type"
            | "unsafe"
            | "use"
            | "where"
            | "while"
            | "yield"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitizes_identifiers_and_keywords() {
        assert_eq!(rust_ident(" 9-model.value "), "v_9_model_value");
        assert_eq!(rust_ident("struct"), "struct_");
        assert_eq!(rust_ident("gen"), "gen_");
    }

    #[test]
    fn rejects_invalid_identifiers() {
        assert!(parse_ident("not-an-ident", "test").is_err());
    }
}
