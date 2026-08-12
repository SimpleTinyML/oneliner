use std::path::Path;

use onnx_extractor::{DataType, Model, Tensor};

use super::error;
use super::types::{ElementType, ModelIo, TensorInfo};

pub(super) fn load(path: &Path) -> syn::Result<ModelIo> {
    let model = Model::load_from_file(path)
        .map_err(|parse_error| error(path, format!("failed to parse ONNX model: {parse_error}")))?;
    let graph = model.graph();

    Ok(ModelIo {
        input: parse_tensor(&graph.tensors()[&graph.inputs()[0]], path, "input")?,
        output: parse_tensor(&graph.tensors()[&graph.outputs()[0]], path, "output")?,
    })
}

fn parse_tensor(tensor: &Tensor, path: &Path, label: &str) -> syn::Result<TensorInfo> {
    let element_type = element_type(tensor.data_type()).ok_or_else(|| {
        error(
            path,
            format!(
                "unsupported ONNX graph {label} tensor element type '{:?}' for '{}'",
                tensor.data_type(),
                tensor.name()
            ),
        )
    })?;
    let dimensions = tensor.shape();
    if dimensions.len() > 4 {
        return Err(error(
            path,
            format!(
                "ONNX graph {label} tensor '{}' rank {} exceeds Tensor's four dimensions",
                tensor.name(),
                dimensions.len()
            ),
        ));
    }

    let mut shape = [1; 4];
    let offset = 4 - dimensions.len();
    for (index, dimension) in dimensions.iter().enumerate() {
        shape[offset + index] = usize::try_from(*dimension).map_err(|_| {
            error(
                path,
                format!(
                    "ONNX graph {label} tensor '{}' has a dynamic dimension",
                    tensor.name()
                ),
            )
        })?;
    }
    Ok(TensorInfo {
        element_type,
        shape,
    })
}

fn element_type(value: DataType) -> Option<ElementType> {
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
