//! Read converter metadata and count unique constant/temporary resource payloads.

use std::{fs, path::Path};

use proc_macro2::Span;
use serde::Deserialize;
use syn::Ident;

use super::BindingArtifact;
use crate::utils::parse_ident;

#[derive(Debug)]
pub(super) struct FlowMetadata {
    pub execute_fns: Vec<Ident>,
    pub input: Option<BindingArtifact>,
    pub output: BindingArtifact,
    /// Constant/weight bytes placed in flash.
    pub params_size: usize,
    /// Model RAM workspace size in bytes, excluding possible padding.
    pub arena_size: usize,
}

#[derive(Deserialize)]
struct Metadata {
    cmd_executes: Vec<Execute>,
}

#[derive(Deserialize)]
struct Execute {
    name: String,
    resources: Vec<Resource>,
}

#[derive(Deserialize)]
struct Resource {
    static_ident: Option<String>,
    size: Option<usize>,
    role: Role,
}

#[derive(Deserialize, PartialEq, Debug)]
#[serde(rename_all = "lowercase")]
enum Role {
    Input,
    Output,
    Constant,
    Temporary,
    #[serde(other)]
    Other,
}

pub(super) fn load_metadata(path: &Path) -> syn::Result<FlowMetadata> {
    let metadata: Metadata = serde_json::from_str(
        &fs::read_to_string(path).map_err(|error| syn::Error::new(Span::call_site(), error))?,
    )
    .map_err(|error| syn::Error::new(Span::call_site(), error))?;

    let execute_fns = metadata
        .cmd_executes
        .iter()
        .map(|execute| parse_ident(&execute.name, "generated IREE execute function"))
        .collect::<syn::Result<_>>()?;
    let resources: Vec<&Resource> = metadata
        .cmd_executes
        .iter()
        .flat_map(|execute| execute.resources.iter())
        .collect();
    let input = resources
        .iter()
        .find(|resource| resource.role == Role::Input)
        .and_then(|resource| resource.size)
        .map(|size| BindingArtifact { size });
    let output = resources
        .iter()
        .find(|resource| resource.role == Role::Output)
        .and_then(|resource| resource.size)
        .map(|size| BindingArtifact { size })
        .unwrap();

    let (params_size, arena_size) = footprint_sizes(resources.iter().copied());

    Ok(FlowMetadata {
        execute_fns,
        input,
        output,
        params_size,
        arena_size,
    })
}

/// Sums constant and transient resource sizes, counting each storage blob once.
///
/// A constant or temporary shared by several `cmd_execute` blocks is rendered
/// once in the generated storage (a constant static or a workspace field), so
/// its payload contributes once. Alignment padding and input/output tensors
/// are excluded.
fn footprint_sizes<'a>(resources: impl Iterator<Item = &'a Resource>) -> (usize, usize) {
    let mut params_size = 0usize;
    let mut arena_size = 0usize;
    let mut seen_flash = std::collections::HashSet::new();
    let mut seen_ram = std::collections::HashSet::new();
    for resource in resources {
        match (&resource.role, resource.size) {
            (Role::Constant, Some(size)) => {
                if seen_flash.insert(resource.static_ident.as_deref()) {
                    params_size += size;
                }
            }
            (Role::Temporary, Some(size)) => {
                if seen_ram.insert(resource.static_ident.as_deref()) {
                    arena_size += size;
                }
            }
            _ => {}
        }
    }
    (params_size, arena_size)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_metadata_and_deduplicates_footprints() {
        let value: Metadata = serde_json::from_str(
            r#"{
                "cmd_executes": [{
                    "name": "cmd_execute_0",
                    "resources": [
                        {"static_ident": "CONST_ARG2", "kind": "constant", "size": 100, "role": "constant"},
                        {"static_ident": "CONST_ARG2", "kind": "constant", "size": 100, "role": "constant"},
                        {"static_ident": "CONST_ARG5", "kind": "constant", "size": 7, "role": "constant"},
                        {"static_ident": "TEMP_ARG4", "kind": "transient", "size": 50, "role": "temporary"},
                        {"static_ident": "TEMP_ARG9", "kind": "transient", "size": 50, "role": "temporary"},
                        {"static_ident": "INPUT_ARG1", "kind": "input", "size": 16, "role": "input"}
                    ]
                }]
            }"#,
        )
        .unwrap();
        let resources: Vec<&Resource> = value
            .cmd_executes
            .iter()
            .flat_map(|execute| execute.resources.iter())
            .collect();

        assert_eq!(resources[0].role, Role::Constant);
        assert_eq!(resources[1].role, Role::Constant);
        assert_eq!(resources[3].role, Role::Temporary);
        assert_eq!(resources[5].role, Role::Input);

        let (params_size, arena_size) = footprint_sizes(resources.into_iter());
        assert_eq!(params_size, 107);
        assert_eq!(arena_size, 100);
    }
}
