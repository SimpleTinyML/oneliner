use quote::quote;
use std::path::{Path, PathBuf};
use syn::ItemStruct;

/// Expands the Microflow backend for a model struct.
///
/// Input: annotated struct and absolute model path.
/// Output: generated Rust tokens implementing runtime traits.
pub fn expand(input_struct: ItemStruct, model_path: PathBuf) -> proc_macro2::TokenStream {
    let struct_ident = &input_struct.ident;
    let model_path_lit = path_lit(&model_path);

    quote! {
        use ::OneLiner::runtime::Predict as _;

        #input_struct

        impl ::OneLiner::runtime::ModelSource for #struct_ident {
            const MODEL_PATH: &'static str = #model_path_lit;
            const ARTIFACTS: ::OneLiner::runtime::ModelArtifacts = ::OneLiner::runtime::ModelArtifacts {
                backend: "microflow",
                expansion: "microflow-rs",
                model_path: #model_path_lit,
                compile_input_path: "",
                object_path: "",
                link_path: "",
                ir_path: "",
                flow_rs_path: "",
                metadata_json_path: "",
                input_size: 0,
                output_size: 0,
            };
        }

        impl ::OneLiner::runtime::Predict<[u8]> for #struct_ident
        where
            #struct_ident: ::OneLiner::runtime::MicroflowModel,
        {
            type Error = <#struct_ident as ::OneLiner::runtime::MicroflowModel>::Error;
            type Output = <#struct_ident as ::OneLiner::runtime::MicroflowModel>::Output;

            /// Runs prediction through the user-provided Microflow backend hook.
            ///
            /// Input: model input bytes.
            /// Output: backend-defined prediction output or error.
            fn try_predict(input: &[u8]) -> ::core::result::Result<Self::Output, Self::Error> {
                <#struct_ident as ::OneLiner::runtime::MicroflowModel>::try_predict_microflow(input)
            }
        }
    }
}

/// Converts a filesystem path into a Rust string literal for generated code.
///
/// Input: filesystem path.
/// Output: `syn::LitStr` with Windows separators normalized to `/`.
fn path_lit(path: &Path) -> syn::LitStr {
    syn::LitStr::new(
        &path.to_string_lossy().replace('\\', "/"),
        proc_macro2::Span::call_site(),
    )
}
