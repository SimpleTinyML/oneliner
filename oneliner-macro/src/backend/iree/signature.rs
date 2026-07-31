use std::fs;
use std::path::Path;

use onnx_extractor::{DataType, Graph, Model, Tensor};
use proc_macro2::Span;

#[derive(Debug, Clone, Copy)]
pub(super) enum ElementType {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F32,
    F64,
}

impl ElementType {
    pub(super) fn rust_tokens(self) -> proc_macro2::TokenStream {
        match self {
            Self::I8 => quote::quote!(i8),
            Self::I16 => quote::quote!(i16),
            Self::I32 => quote::quote!(i32),
            Self::I64 => quote::quote!(i64),
            Self::U8 => quote::quote!(u8),
            Self::U16 => quote::quote!(u16),
            Self::U32 => quote::quote!(u32),
            Self::U64 => quote::quote!(u64),
            Self::F32 => quote::quote!(f32),
            Self::F64 => quote::quote!(f64),
        }
    }

    pub(super) const fn byte_width(self) -> usize {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::I64 | Self::U64 | Self::F64 => 8,
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct TensorArtifact {
    pub element_type: ElementType,
    pub shape: [usize; 4],
}

impl TensorArtifact {
    pub(super) fn byte_len(&self) -> Option<usize> {
        self.shape
            .iter()
            .try_fold(self.element_type.byte_width(), |size, dimension| {
                size.checked_mul(*dimension)
            })
    }
}

#[derive(Debug)]
pub(super) struct ModelSignature {
    pub input: TensorArtifact,
    pub output: TensorArtifact,
}

pub(super) fn load_model_signature(
    model_path: &Path,
    compile_input_path: &Path,
) -> syn::Result<ModelSignature> {
    if has_extension(model_path, "onnx") {
        load_onnx_model_signature(model_path)
    } else {
        load_mlir_model_signature(compile_input_path)
    }
}

fn load_mlir_model_signature(path: &Path) -> syn::Result<ModelSignature> {
    let text = fs::read_to_string(path).map_err(|error| {
        syn::Error::new(
            Span::call_site(),
            format!(
                "failed to read model signature from {}: {error}",
                path.display()
            ),
        )
    })?;
    parse_model_signature(&text, path)
}

fn load_onnx_model_signature(path: &Path) -> syn::Result<ModelSignature> {
    let model = Model::load_from_file(path)
        .map_err(|parse_error| error(path, format!("failed to parse ONNX model: {parse_error}")))?;
    let graph = model.graph();

    let input = exactly_one_onnx_tensor(graph, graph.inputs(), path, "input")?;
    let output = exactly_one_onnx_tensor(graph, graph.outputs(), path, "output")?;

    Ok(ModelSignature { input, output })
}

fn exactly_one_onnx_tensor(
    graph: &Graph,
    tensor_names: &[String],
    path: &Path,
    label: &str,
) -> syn::Result<TensorArtifact> {
    if tensor_names.len() != 1 {
        return Err(error(
            path,
            format!(
                "ModelInference requires exactly one {label} tensor, but the ONNX graph declares {}",
                tensor_names.len()
            ),
        ));
    }

    let tensor_name = tensor_names.first().expect("length was checked");
    let tensor = graph.tensors().get(tensor_name).ok_or_else(|| {
        error(
            path,
            format!("ONNX graph {label} tensor '{tensor_name}' has no type information"),
        )
    })?;
    parse_onnx_tensor(tensor, path, label)
}

fn parse_onnx_tensor(
    tensor: &Tensor,
    path: &Path,
    label: &str,
) -> syn::Result<TensorArtifact> {
    let element_type = parse_onnx_element_type(tensor.data_type()).ok_or_else(|| {
        error(
            path,
            format!(
                "unsupported ONNX graph {label} tensor element type '{:?}' for '{name}'",
                tensor.data_type(),
                name = tensor.name()
            ),
        )
    })?;

    let dimensions = tensor.shape();
    if dimensions.len() > 4 {
        return Err(error(
            path,
            format!(
                "ONNX graph {label} tensor '{name}' rank {} exceeds Tensor's four dimensions",
                dimensions.len(),
                name = tensor.name()
            ),
        ));
    }

    let mut shape = [1usize; 4];
    let shape_offset = 4 - dimensions.len();
    for (index, dimension) in dimensions.iter().enumerate() {
        shape[shape_offset + index] = usize::try_from(*dimension).map_err(|_| {
            error(
                path,
                format!(
                    "ONNX graph {label} tensor '{name}' has a dynamic dimension",
                    name = tensor.name()
                ),
            )
        })?;
    }

    Ok(TensorArtifact {
        element_type,
        shape,
    })
}

fn parse_onnx_element_type(value: DataType) -> Option<ElementType> {
    match value {
        DataType::Int8 => Some(ElementType::I8),
        DataType::Int16 => Some(ElementType::I16),
        DataType::Int32 => Some(ElementType::I32),
        DataType::Int64 => Some(ElementType::I64),
        DataType::Uint8 => Some(ElementType::U8),
        DataType::Uint16 => Some(ElementType::U16),
        DataType::Uint32 => Some(ElementType::U32),
        DataType::Uint64 => Some(ElementType::U64),
        DataType::Float => Some(ElementType::F32),
        DataType::Double => Some(ElementType::F64),
        _ => None,
    }
}

fn parse_model_signature(text: &str, path: &Path) -> syn::Result<ModelSignature> {
    let after_main = find_main_signature(text)
        .ok_or_else(|| error(path, "model does not contain an @main entry function"))?;
    let input_open = after_main
        .find('(')
        .ok_or_else(|| error(path, "@main does not contain an input argument list"))?;
    let input_close = matching_delimiter(after_main, input_open, '(', ')')
        .ok_or_else(|| error(path, "@main has an unterminated input argument list"))?;
    let input_list = &after_main[input_open + 1..input_close];
    let inputs = parse_tensor_types(input_list, path, "input")?;
    let input = exactly_one_tensor(inputs, path, "input")?;

    let after_inputs = after_main[input_close + 1..].trim_start();
    let after_arrow = after_inputs
        .strip_prefix("->")
        .ok_or_else(|| error(path, "@main does not contain an output type"))?
        .trim_start();
    let output_text = if after_arrow.starts_with('(') {
        let close = matching_delimiter(after_arrow, 0, '(', ')')
            .ok_or_else(|| error(path, "@main has an unterminated output type list"))?;
        &after_arrow[1..close]
    } else {
        let tensor_text = after_arrow
            .strip_prefix("tensor<")
            .ok_or_else(|| error(path, "@main output is not a tensor"))?;
        let close = matching_angle_bracket(tensor_text)
            .ok_or_else(|| error(path, "@main has an unterminated output tensor type"))?;
        &after_arrow[.."tensor<".len() + close + 1]
    };
    let outputs = parse_tensor_types(output_text, path, "output")?;
    let output = exactly_one_tensor(outputs, path, "output")?;

    Ok(ModelSignature { input, output })
}

fn find_main_signature(text: &str) -> Option<&str> {
    for (main_offset, _) in text.match_indices("@main") {
        let after_name = &text[main_offset + "@main".len()..];
        if !after_name
            .chars()
            .next()
            .is_some_and(|character| character == '(' || character.is_whitespace())
        {
            continue;
        }

        let before_main = &text[..main_offset];
        let Some(function_offset) = ["func.func", "util.func"]
            .into_iter()
            .filter_map(|operation| {
                before_main
                    .rfind(operation)
                    .map(|offset| (offset, operation.len()))
            })
            .max_by_key(|(offset, _)| *offset)
        else {
            continue;
        };
        let declaration_prefix = &text[function_offset.0 + function_offset.1..main_offset];
        if !declaration_prefix.contains(['{', '}']) {
            return Some(after_name);
        }
    }

    None
}

fn exactly_one_tensor(
    tensors: Vec<TensorArtifact>,
    path: &Path,
    label: &str,
) -> syn::Result<TensorArtifact> {
    if tensors.len() != 1 {
        return Err(error(
            path,
            format!(
                "ModelInference requires exactly one {label} tensor, but @main declares {}",
                tensors.len()
            ),
        ));
    }
    Ok(tensors.into_iter().next().expect("length was checked"))
}

fn parse_tensor_types(text: &str, path: &Path, label: &str) -> syn::Result<Vec<TensorArtifact>> {
    let mut tensors = Vec::new();
    let mut remaining = text;
    while let Some(offset) = remaining.find("tensor<") {
        let tensor_text = &remaining[offset + "tensor<".len()..];
        let close = matching_angle_bracket(tensor_text).ok_or_else(|| {
            error(
                path,
                format!("unterminated tensor type in @main {label} declaration"),
            )
        })?;
        tensors.push(parse_tensor_type(&tensor_text[..close], path, label)?);
        remaining = &tensor_text[close + 1..];
    }
    Ok(tensors)
}

fn parse_tensor_type(text: &str, path: &Path, label: &str) -> syn::Result<TensorArtifact> {
    let shape_and_type = text
        .split(',')
        .next()
        .expect("split always yields one item")
        .trim();
    let mut parts = shape_and_type.split('x').collect::<Vec<_>>();
    let element = parts
        .pop()
        .ok_or_else(|| error(path, format!("empty @main {label} tensor type")))?;
    let element_type = parse_element_type(element.trim()).ok_or_else(|| {
        error(
            path,
            format!(
                "unsupported @main {label} tensor element type '{}'",
                element.trim()
            ),
        )
    })?;

    if parts.len() > 4 {
        return Err(error(
            path,
            format!(
                "@main {label} tensor rank {} exceeds Tensor's four dimensions",
                parts.len()
            ),
        ));
    }

    let mut shape = [1usize; 4];
    let shape_offset = 4 - parts.len();
    for (index, dimension) in parts.into_iter().enumerate() {
        let dimension = dimension.trim();
        if dimension == "?" {
            return Err(error(
                path,
                format!("@main {label} tensor has a dynamic dimension"),
            ));
        }
        shape[shape_offset + index] = dimension.parse::<usize>().map_err(|_| {
            error(
                path,
                format!("invalid @main {label} tensor dimension '{}'", dimension),
            )
        })?;
    }

    Ok(TensorArtifact {
        element_type,
        shape,
    })
}

fn parse_element_type(value: &str) -> Option<ElementType> {
    match value {
        "i8" | "si8" => Some(ElementType::I8),
        "i16" | "si16" => Some(ElementType::I16),
        "i32" | "si32" => Some(ElementType::I32),
        "i64" | "si64" => Some(ElementType::I64),
        "ui8" => Some(ElementType::U8),
        "ui16" => Some(ElementType::U16),
        "ui32" => Some(ElementType::U32),
        "ui64" => Some(ElementType::U64),
        "f32" => Some(ElementType::F32),
        "f64" => Some(ElementType::F64),
        _ => None,
    }
}

fn matching_delimiter(text: &str, open_offset: usize, open: char, close: char) -> Option<usize> {
    let mut depth = 0usize;
    for (relative_offset, character) in text[open_offset..].char_indices() {
        if character == open {
            depth += 1;
        } else if character == close {
            depth = depth.checked_sub(1)?;
            if depth == 0 {
                return Some(open_offset + relative_offset);
            }
        }
    }
    None
}

fn matching_angle_bracket(text: &str) -> Option<usize> {
    let mut depth = 1usize;
    for (offset, character) in text.char_indices() {
        if character == '<' {
            depth += 1;
        } else if character == '>' {
            depth = depth.checked_sub(1)?;
            if depth == 0 {
                return Some(offset);
            }
        }
    }
    None
}

fn error(path: &Path, message: impl std::fmt::Display) -> syn::Error {
    syn::Error::new(
        Span::call_site(),
        format!(
            "failed to parse model signature at {}: {message}",
            path.display()
        ),
    )
}

fn has_extension(path: &Path, expected: &str) -> bool {
    path.extension()
        .and_then(std::ffi::OsStr::to_str)
        .is_some_and(|extension| extension.eq_ignore_ascii_case(expected))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_util_main_signature_with_tensor_attributes() {
        let text = r#"
            module {
              util.func public @main(
                %arg0: tensor<1x64x64x3xi8> {ml_program.identifier = "input"}
              ) -> (tensor<1x2xi8> {ml_program.identifier = "output"}) {
                util.return %arg0 : tensor<1x2xi8>
              }
            }
        "#;

        let signature = parse_model_signature(text, Path::new("model.mlir")).unwrap();

        assert!(matches!(signature.input.element_type, ElementType::I8));
        assert_eq!(signature.input.shape, [1, 64, 64, 3]);
        assert!(matches!(signature.output.element_type, ElementType::I8));
        assert_eq!(signature.output.shape, [1, 1, 1, 2]);
    }

    #[test]
    fn selects_main_instead_of_an_earlier_helper_function() {
        let text = r#"
            func.func private @helper(%arg0: tensor<4xi32>) -> tensor<4xi32>
            func.func @main(%arg0: tensor<2xf32>) -> tensor<2xf32>
        "#;

        let signature = parse_model_signature(text, Path::new("model.mlir")).unwrap();

        assert!(matches!(signature.input.element_type, ElementType::F32));
        assert_eq!(signature.input.shape, [1, 1, 1, 2]);
        assert!(matches!(signature.output.element_type, ElementType::F32));
        assert_eq!(signature.output.shape, [1, 1, 1, 2]);
    }
}
