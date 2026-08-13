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
    size: Option<usize>,
    role: Role,
}

#[derive(Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
enum Role {
    Input,
    Output,
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
    let mut resources = metadata
        .cmd_executes
        .iter()
        .flat_map(|execute| execute.resources.iter());
    let input = resources
        .clone()
        .find(|resource| resource.role == Role::Input)
        .and_then(|resource| resource.size)
        .map(|size| BindingArtifact { size });
    let output = resources
        .find(|resource| resource.role == Role::Output)
        .and_then(|resource| resource.size)
        .map(|size| BindingArtifact { size })
        .unwrap();

    Ok(FlowMetadata {
        execute_fns,
        input,
        output,
    })
}
