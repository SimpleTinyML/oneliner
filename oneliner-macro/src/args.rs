use syn::spanned::Spanned;
use syn::{AttributeArgs, Lit, LitStr, Meta, NestedMeta};

pub struct ModelArgs {
    pub model_path: LitStr,
    pub backend: BackendArg,
}

pub enum BackendArg {
    Iree,
    Microflow,
}

impl ModelArgs {
    /// Parses `#[model("path", backend = "...")]` arguments.
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

        let mut backend = BackendArg::Iree;
        for arg in args {
            match arg {
                NestedMeta::Meta(Meta::NameValue(meta)) if meta.path.is_ident("backend") => {
                    backend = parse_backend(meta.lit)?;
                }
                other => {
                    return Err(syn::Error::new(
                        other.span(),
                        "unknown #[model] option, expected backend = \"...\"",
                    ));
                }
            }
        }

        Ok(Self {
            model_path,
            backend,
        })
    }
}

/// Parses a backend string literal into a known backend selector.
///
/// Input: `backend = "..."` literal.
/// Output: `BackendArg` or a `syn::Error` for unsupported names.
fn parse_backend(lit: Lit) -> syn::Result<BackendArg> {
    let value = match lit {
        Lit::Str(value) => value,
        other => {
            return Err(syn::Error::new(
                other.span(),
                "backend must be a string literal, for example backend = \"microflow\"",
            ));
        }
    };

    match value.value().trim().to_ascii_lowercase().as_str() {
        "iree" => Ok(BackendArg::Iree),
        "microflow" | "microflow-rs" | "microflow_rs" => Ok(BackendArg::Microflow),
        other => Err(syn::Error::new(
            value.span(),
            format!("unknown backend '{other}', expected 'iree' or 'microflow'"),
        )),
    }
}
