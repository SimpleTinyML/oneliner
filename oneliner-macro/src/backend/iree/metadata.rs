use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use proc_macro2::Span;
use serde::Deserialize;
use syn::Ident;

use super::super::common::parse_ident;
use super::BindingArtifact;

#[derive(Debug)]
pub(super) struct FlowMetadata {
    pub execute_fns: Vec<Ident>,
    pub input: Option<BindingArtifact>,
    pub output: BindingArtifact,
}

#[derive(Debug, Deserialize)]
struct MetadataDocument {
    schema_version: u32,
    cmd_executes: Vec<CmdExecute>,
}

#[derive(Debug, Deserialize)]
struct CmdExecute {
    name: String,
    resources: Vec<ResourceBinding>,
}

#[derive(Debug, Deserialize)]
struct ResourceBinding {
    static_ident: String,
    kind: BindingKind,
    size: Option<usize>,
    role: BindingRole,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum BindingKind {
    External,
    #[serde(other)]
    Other,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum BindingRole {
    Input,
    Output,
    Inout,
    Temporary,
    Constant,
    #[serde(other)]
    Other,
}

pub(super) fn load_metadata(path: &Path) -> syn::Result<FlowMetadata> {
    let text = fs::read_to_string(path).map_err(call_site_error)?;
    let document: MetadataDocument = serde_json::from_str(&text).map_err(|error| {
        syn::Error::new(
            Span::call_site(),
            format!("invalid IREE metadata JSON at {}: {error}", path.display()),
        )
    })?;
    if document.schema_version != 1 {
        return Err(error(format!(
            "unsupported IREE metadata schema version {} in {}",
            document.schema_version,
            path.display()
        )));
    }
    if document.cmd_executes.is_empty() {
        return Err(error(format!(
            "no cmd_execute entries were found in {}",
            path.display()
        )));
    }

    let execute_fns = document
        .cmd_executes
        .iter()
        .map(|execute| parse_ident(&execute.name, "generated IREE execute function"))
        .collect::<syn::Result<Vec<_>>>()?;
    ensure_unique_idents(&execute_fns, "execute function")?;

    let mut bindings = BTreeMap::<String, (usize, BindingRole)>::new();
    for binding in document
        .cmd_executes
        .iter()
        .flat_map(|execute| execute.resources.iter())
        .filter(|binding| binding.kind == BindingKind::External)
    {
        let Some(size) = binding.size.filter(|size| *size > 0) else {
            continue;
        };
        match bindings.entry(binding.static_ident.clone()) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert((size, binding.role));
            }
            std::collections::btree_map::Entry::Occupied(entry)
                if *entry.get() != (size, binding.role) =>
            {
                return Err(error(format!(
                    "binding '{}' has inconsistent size or role across execute blocks",
                    binding.static_ident
                )));
            }
            std::collections::btree_map::Entry::Occupied(_) => {}
        }
    }

    let input = unique_binding(
        bindings
            .iter()
            .filter(|(_, (_, role))| matches!(role, BindingRole::Input | BindingRole::Inout)),
        "input",
    )?;
    let output = unique_binding(
        bindings
            .iter()
            .filter(|(_, (_, role))| matches!(role, BindingRole::Output | BindingRole::Inout)),
        "output",
    )?
    .ok_or_else(|| error("IREE metadata does not contain an output binding"))?;

    Ok(FlowMetadata {
        execute_fns,
        input,
        output,
    })
}

fn unique_binding<'a>(
    candidates: impl Iterator<Item = (&'a String, &'a (usize, BindingRole))>,
    label: &str,
) -> syn::Result<Option<BindingArtifact>> {
    let candidates = candidates.collect::<Vec<_>>();
    if candidates.len() > 1 {
        return Err(error(format!(
            "multiple {label} bindings are not supported: {}",
            candidates
                .iter()
                .map(|(name, _)| name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        )));
    }
    let Some((_name, (size, _))) = candidates.into_iter().next() else {
        return Ok(None);
    };
    Ok(Some(BindingArtifact { size: *size }))
}

fn ensure_unique_idents(idents: &[Ident], label: &str) -> syn::Result<()> {
    let mut names = idents.iter().map(ToString::to_string).collect::<Vec<_>>();
    names.sort();
    if let Some(duplicate) = names.windows(2).find(|pair| pair[0] == pair[1]) {
        return Err(error(format!(
            "duplicate {label} '{}': metadata is invalid",
            duplicate[0]
        )));
    }
    Ok(())
}

fn error(message: impl std::fmt::Display) -> syn::Error {
    syn::Error::new(Span::call_site(), message)
}

fn call_site_error(error: impl std::fmt::Display) -> syn::Error {
    syn::Error::new(Span::call_site(), error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_duplicate_execute_functions() {
        let ident = syn::parse_str::<Ident>("cmd_execute_0").unwrap();
        assert!(ensure_unique_idents(&[ident.clone(), ident], "execute function").is_err());
    }
}
