use quote::quote;
use std::path::PathBuf;
use syn::ItemStruct;

use super::common::path_lit;

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
            type Output<'prediction> = <#struct_ident as ::OneLiner::runtime::MicroflowModel>::Output
            where
                Self: 'prediction;

            /// Runs prediction through the user-provided Microflow backend hook.
            ///
            /// Input: model input bytes.
            /// Output: backend-defined prediction output or error.
            fn try_predict<'prediction>(
                &'prediction mut self,
                input: &[u8],
            ) -> ::core::result::Result<Self::Output<'prediction>, Self::Error> {
                <#struct_ident as ::OneLiner::runtime::MicroflowModel>::try_predict_microflow(input)
            }
        }
    }
}
