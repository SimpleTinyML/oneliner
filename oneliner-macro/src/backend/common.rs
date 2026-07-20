use std::path::Path;

use proc_macro2::Span;
use syn::{Ident, LitStr};

pub(super) fn path_lit(path: &Path) -> LitStr {
    LitStr::new(
        &path.to_string_lossy().replace('\\', "/"),
        Span::call_site(),
    )
}

pub(super) fn parse_ident(raw: &str, context: &str) -> syn::Result<Ident> {
    syn::parse_str(raw).map_err(|error| {
        syn::Error::new(
            Span::call_site(),
            format!("invalid Rust identifier '{raw}' for {context}: {error}"),
        )
    })
}

pub(super) fn rust_ident(raw: &str) -> String {
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
