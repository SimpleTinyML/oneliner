use syn::spanned::Spanned;
use syn::{AttributeArgs, Lit, LitStr, Meta, NestedMeta};

pub struct ModelArgs {
    pub model_path: LitStr,
    pub arena: ArenaArg,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArenaArg {
    Owned,
    Shared,
}

impl ModelArgs {
    /// Parses `#[model("path", backend = "...", arena = "...")]` arguments.
    ///
    /// Input: `syn::AttributeArgs` from the procedural macro entry point.
    /// Output: normalized model path literal and backend selector.
    pub fn parse(args: AttributeArgs) -> syn::Result<Self> {
        let mut args = args.into_iter();
        let model_path = match args.next() {
            Some(NestedMeta::Lit(Lit::Str(path))) => path,
            Some(arg) => return Err(syn::Error::new(arg.span(), "expected model path string")),
            None => {
                return Err(syn::Error::new(
                    proc_macro2::Span::call_site(),
                    "missing model path",
                ));
            }
        };

        let mut has_backend = false;
        let mut arena = None;
        for arg in args {
            match arg {
                NestedMeta::Meta(Meta::NameValue(meta)) if meta.path.is_ident("backend") => {
                    if has_backend {
                        return Err(syn::Error::new(meta.span(), "duplicate backend option"));
                    }
                    parse_backend(meta.lit)?;
                    has_backend = true;
                }
                NestedMeta::Meta(Meta::NameValue(meta)) if meta.path.is_ident("arena") => {
                    if arena.is_some() {
                        return Err(syn::Error::new(meta.span(), "duplicate arena option"));
                    }
                    arena = Some(parse_arena(meta.lit)?);
                }
                other => {
                    return Err(syn::Error::new(
                        other.span(),
                        "unknown #[model] option, expected backend = \"...\" or arena = \"...\"",
                    ));
                }
            }
        }

        Ok(Self {
            model_path,
            arena: arena.unwrap_or(ArenaArg::Owned),
        })
    }
}

/// Parses a backend string literal into a known backend selector.
///
/// Input: `backend = "..."` literal.
/// Output: `Ok(())` for IREE or a `syn::Error` for unsupported names.
fn parse_backend(lit: Lit) -> syn::Result<()> {
    let value = match lit {
        Lit::Str(value) => value,
        other => {
            return Err(syn::Error::new(
                other.span(),
                "backend must be a string literal, for example backend = \"iree\"",
            ));
        }
    };

    match value.value().trim().to_ascii_lowercase().as_str() {
        "iree" => Ok(()),
        other => Err(syn::Error::new(
            value.span(),
            format!("unknown backend '{other}', expected 'iree'"),
        )),
    }
}

fn parse_arena(lit: Lit) -> syn::Result<ArenaArg> {
    let value = match lit {
        Lit::Str(value) => value,
        other => {
            return Err(syn::Error::new(
                other.span(),
                "arena must be a string literal, for example arena = \"shared\"",
            ));
        }
    };

    match value.value().trim().to_ascii_lowercase().as_str() {
        "owned" => Ok(ArenaArg::Owned),
        "shared" => Ok(ArenaArg::Shared),
        other => Err(syn::Error::new(
            value.span(),
            format!("unknown arena '{other}', expected 'owned' or 'shared'"),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_duplicate_backend_options() {
        let args: AttributeArgs = vec![
            syn::parse_quote!("model.tflite"),
            syn::parse_quote!(backend = "iree"),
            syn::parse_quote!(backend = "iree"),
        ];

        assert!(ModelArgs::parse(args).is_err());
    }

    #[test]
    fn defaults_to_iree() {
        let args: AttributeArgs = vec![syn::parse_quote!("model.tflite")];

        assert!(ModelArgs::parse(args).is_ok());
    }
}
