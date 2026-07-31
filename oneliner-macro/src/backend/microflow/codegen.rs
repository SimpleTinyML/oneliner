use std::path::PathBuf;

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{ItemStruct, Meta, NestedMeta};

use super::signature::{ModelSignature, TensorSignature};
use crate::backend::common::{path_lit, rust_ident};

pub(super) fn expand(
    input_struct: ItemStruct,
    model_path: PathBuf,
    signature: ModelSignature,
) -> TokenStream {
    let struct_ident = &input_struct.ident;
    let module_ident = format_ident!(
        "__oneliner_microflow_{}",
        rust_ident(&struct_ident.to_string())
    );
    let model_path = path_lit(&model_path);
    let input_type = buffer_type(&signature.input);
    let output_type = buffer_type(&signature.output);
    let input_zero = zero_buffer(&signature.input);
    let input_size = signature.input.byte_len;
    let output_size = signature.output.byte_len;

    let default_impl = if derives_default(&input_struct) {
        quote! {}
    } else {
        quote! {
            impl ::core::default::Default for #struct_ident {
                fn default() -> Self {
                    Self::new()
                }
            }
        }
    };

    quote! {
        #input_struct

        #[doc(hidden)]
        mod #module_ident {
            #[allow(unused_imports)]
            use ::OneLiner::__private::microflow as microflow;

            #[::OneLiner::__private::microflow::model(#model_path)]
            pub(super) struct Backend;
        }

        impl #struct_ident {
            /// Creates a stateless MicroFlow model value.
            pub const fn new() -> Self {
                Self
            }
        }

        #default_impl

        impl ::OneLiner::runtime::ModelSource for #struct_ident {
            const MODEL_PATH: &'static str = #model_path;
            const ARTIFACTS: ::OneLiner::runtime::ModelArtifacts =
                ::OneLiner::runtime::ModelArtifacts {
                    backend: "microflow",
                    expansion: "microflow-rs",
                    model_path: #model_path,
                    compile_input_path: "",
                    object_path: "",
                    link_path: "",
                    ir_path: "",
                    flow_rs_path: "",
                    metadata_json_path: "",
                    input_size: #input_size,
                    output_size: #output_size,
                };
        }

        impl ::OneLiner::runtime::ModelInference for #struct_ident {
            type InputTensor = #input_type;
            type InputRefOrVal<'input> = Self::InputTensor;
            type OutputTensor = #output_type;

            fn create_input_tensor() -> Self::InputTensor {
                #input_zero
            }

            fn run<'input>(
                &mut self,
                input: Self::InputRefOrVal<'input>,
            ) -> Self::OutputTensor {
                #module_ident::Backend::predict(input)
            }
        }
    }
}

fn buffer_type(tensor: &TensorSignature) -> TokenStream {
    match tensor.shape.as_slice() {
        [rows, columns] => quote! {
            ::OneLiner::__private::microflow::buffer::Buffer2D<
                f32,
                #rows,
                #columns,
            >
        },
        [batches, rows, columns, channels] => quote! {
            ::OneLiner::__private::microflow::buffer::Buffer4D<
                f32,
                #batches,
                #rows,
                #columns,
                #channels,
            >
        },
        _ => unreachable!("MicroFlow tensor rank was validated"),
    }
}

fn zero_buffer(tensor: &TensorSignature) -> TokenStream {
    match tensor.shape.as_slice() {
        [_, _] => quote! {
            ::OneLiner::__private::microflow::buffer::Buffer2D::from_element(0.0)
        },
        [_, _, _, channels] => quote! {
            ::core::array::from_fn(|_| {
                ::OneLiner::__private::microflow::buffer::Buffer2D::from_element(
                    [0.0; #channels],
                )
            })
        },
        _ => unreachable!("MicroFlow tensor rank was validated"),
    }
}

fn derives_default(input_struct: &ItemStruct) -> bool {
    input_struct.attrs.iter().any(|attribute| {
        if !attribute.path.is_ident("derive") {
            return false;
        }

        let Ok(Meta::List(derive)) = attribute.parse_meta() else {
            return false;
        };

        derive.nested.iter().any(|item| {
            let NestedMeta::Meta(Meta::Path(path)) = item else {
                return false;
            };

            path.segments
                .last()
                .is_some_and(|segment| segment.ident == "Default")
        })
    })
}
