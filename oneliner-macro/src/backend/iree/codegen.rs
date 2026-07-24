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
    let input_type = artifacts.input_tensor.element_type.rust_tokens();
    let output_type = artifacts.output_tensor.element_type.rust_tokens();
    let [input_d0, input_d1, input_d2, input_d3] = artifacts.input_tensor.shape;
    let [output_d0, output_d1, output_d2, output_d3] = artifacts.output_tensor.shape;

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
                AlignedType, AnyBufferRange, Buffer, BufferMut, BufferSource, Error,
                iree_hal_executable_environment_v0_t, iree_hal_executable_library_header_t,
                iree_hal_executable_library_query_fn_t,
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
            type InputTensor = ::OneLiner::runtime::Tensor<
                #input_type,
                #input_d0,
                #input_d1,
                #input_d2,
                #input_d3,
            >;
            type OutputTensor = ::OneLiner::runtime::Tensor<
                #output_type,
                #output_d0,
                #output_d1,
                #output_d2,
                #output_d3,
            >;

            fn create_input_tensor() -> Self::InputTensor {
                Self::InputTensor::filled(0 as #input_type)
            }

            fn run(&mut self, input: &Self::InputTensor) -> Self::OutputTensor {
                let input_buffer = ::OneLiner::runtime::Buffer::new(
                    input.as_ptr().cast::<u8>(),
                    input.byte_len(),
                );

                let mut output = Self::OutputTensor::filled(0 as #output_type);
                let output_buffer = ::OneLiner::runtime::BufferMut::new(
                    output.as_mut_ptr().cast::<u8>(),
                    output.byte_len(),
                );

                #(
                    #module_ident::#execute_fns(
                        &mut *self.workspace,
                        input_buffer,
                        output_buffer,
                    )
                        .expect("OneLiner inference dispatch failed");
                )*

                output
            }
        }
    }
}
