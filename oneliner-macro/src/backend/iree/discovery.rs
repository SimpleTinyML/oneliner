use std::fs;
use std::path::Path;

use proc_macro2::Span;
use syn::Ident;

use crate::utils::{parse_ident, rust_ident};

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
}
