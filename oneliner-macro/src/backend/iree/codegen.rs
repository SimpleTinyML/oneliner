use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::ItemStruct;

use super::super::common::{path_lit, rust_ident};
use super::IreeArtifacts;

pub(super) fn expand(input_struct: ItemStruct, artifacts: IreeArtifacts) -> TokenStream {
    let struct_ident = &input_struct.ident;
    let struct_vis = &input_struct.vis;
    let module_ident = format_ident!("__oneliner_iree_{}", rust_ident(&struct_ident.to_string()));
    let workspace_ident = format_ident!("{}Workspace", struct_ident);
    let session_ident = format_ident!("{}Session", struct_ident);
    let paths = &artifacts.paths;
    let flow_rs = path_lit(&paths.flow_rs);
    let model_path = path_lit(&paths.model);
    let compile_input_path = path_lit(&paths.compile_input);
    let object_path = path_lit(&paths.object);
    let ir_path = path_lit(&paths.ir);
    let metadata_json_path = path_lit(&paths.metadata_json);
    let input_size = artifacts.input.size;
    let output_size = artifacts.output.size;
    let execute_fns = &artifacts.execute_fns;
    let query_fn = &artifacts.query_fn;
    let input_ident = &artifacts.input.static_ident;
    let output_ident = &artifacts.output.static_ident;
    let input_type = artifacts.input_tensor.element_type.rust_tokens();
    let output_type = artifacts.output_tensor.element_type.rust_tokens();
    let input_element_size = artifacts.input_tensor.element_type.byte_width();
    let output_element_size = artifacts.output_tensor.element_type.byte_width();
    let [input_d0, input_d1, input_d2, input_d3] = artifacts.input_tensor.shape;
    let [output_d0, output_d1, output_d2, output_d3] = artifacts.output_tensor.shape;
    let input_shape = quote!((#input_d0, #input_d1, #input_d2, #input_d3));
    let output_shape = quote!((#output_d0, #output_d1, #output_d2, #output_d3));

    quote! {

        #input_struct

        #[allow(improper_ctypes, non_camel_case_types, non_snake_case, non_upper_case_globals)]
        #[link(name = #object_path, kind = "static", modifiers = "+verbatim")]
        unsafe extern "C" {}

        #[allow(
            dead_code,
            improper_ctypes,
            mutable_transmutes,
            non_camel_case_types,
            non_snake_case,
            non_upper_case_globals,
            unused_imports,
            unused_macros,
            unused_mut,
            unused_variables
        )]


        mod #module_ident {

            use ::OneLiner::runtime::{
                concurrent, dispatch_fn_from_library, fill, try_dispatch, Access, Aligned,
                AlignedType, AnyBufferRange, iree_hal_executable_environment_v0_t, BufferSource, Error,
                iree_hal_executable_library_header_t, iree_hal_executable_library_query_fn_t,
            };

            unsafe extern "C" {
                pub unsafe fn #query_fn(
                    max_version: u32,
                    environment: *const iree_hal_executable_environment_v0_t,
                ) -> *const *const iree_hal_executable_library_header_t;
            }

            static QUERY_FN_PTR: iree_hal_executable_library_query_fn_t = #query_fn;

            include!(#flow_rs);
        }

        #struct_vis type #workspace_ident = #module_ident::Workspace;

        #struct_vis struct #session_ident<'workspace> {
            workspace: &'workspace mut #workspace_ident,
        }

        impl #struct_ident {
            /// Creates an independently reusable inference session over caller-owned storage.
            pub fn session<'workspace>(
                workspace: &'workspace mut #workspace_ident,
            ) -> #session_ident<'workspace> {
                #session_ident { workspace }
            }
        }

        impl ::OneLiner::runtime::ModelSource for #struct_ident {
            const MODEL_PATH: &'static str = #model_path;
            const ARTIFACTS: ::OneLiner::runtime::ModelArtifacts = ::OneLiner::runtime::ModelArtifacts {
                backend: "iree",
                expansion: "static-flow",
                model_path: #model_path,
                compile_input_path: #compile_input_path,
                object_path: #object_path,
                link_path: #object_path,
                ir_path: #ir_path,
                flow_rs_path: #flow_rs,
                metadata_json_path: #metadata_json_path,
                input_size: #input_size,
                output_size: #output_size,
            };
        }

        impl ::OneLiner::runtime::ModelInference for #session_ident<'_> {
            type InputTensor = ::OneLiner::runtime::Tensor<#input_type>;
            type OutputTensor = ::OneLiner::runtime::Tensor<#output_type>;

            fn create_input_tensor() -> Self::InputTensor {
                ::OneLiner::runtime::Tensor::<#input_type>::zeros(#input_shape)
            }

            fn run(&mut self, input: &Self::InputTensor) -> Self::OutputTensor {
                assert_eq!(
                    input.dim(),
                    #input_shape,
                    "OneLiner input tensor shape mismatch",
                );

                let mut input_elements = input.iter();
                for destination in self.workspace.#input_ident.chunks_exact_mut(#input_element_size) {
                    let value = input_elements
                        .next()
                        .expect("OneLiner input tensor contains too few elements");
                    destination.copy_from_slice(&value.to_ne_bytes());
                }
                assert!(
                    input_elements.next().is_none(),
                    "OneLiner input tensor contains too many elements",
                );

                #(
                    #module_ident::#execute_fns(&mut *self.workspace)
                        .expect("OneLiner inference dispatch failed");
                )*

                let mut output =
                    ::OneLiner::runtime::Tensor::<#output_type>::zeros(#output_shape);
                for (value, source) in output
                    .iter_mut()
                    .zip(self.workspace.#output_ident.chunks_exact(#output_element_size))
                {
                    let mut bytes = [0u8; #output_element_size];
                    bytes.copy_from_slice(source);
                    *value = <#output_type>::from_ne_bytes(bytes);
                }
                output
            }
        }
    }
}
