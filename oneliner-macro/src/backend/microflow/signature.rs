use std::fs;
use std::path::Path;

use proc_macro2::Span;

#[allow(
    dead_code,
    deprecated,
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    unused_imports
)]
#[path = "tflite_schema_generated.rs"]
mod tflite_schema_generated;

use tflite_schema_generated::tflite::{root_as_model, SubGraph, Tensor, TensorType};

const TFLITE_FILE_IDENTIFIER: &str = "TFL3";

#[derive(Debug)]
pub(super) struct ModelSignature {
    pub input: TensorSignature,
    pub output: TensorSignature,
}

#[derive(Debug)]
pub(super) struct TensorSignature {
    /// MicroFlow accepts two- and four-dimensional buffers. Rank-one tensors
    /// are normalized to `[1, N]`, matching its own model macro.
    pub shape: Vec<usize>,
    /// Size of the dequantized `f32` buffer exposed by MicroFlow.
    pub byte_len: usize,
}

pub(super) fn load_model_signature(path: &Path) -> syn::Result<ModelSignature> {
    let bytes = fs::read(path).map_err(|error| {
        syn::Error::new(
            Span::call_site(),
            format!("failed to read MicroFlow model {}: {error}", path.display()),
        )
    })?;

    parse_model_signature(&bytes).map_err(|message| {
        syn::Error::new(
            Span::call_site(),
            format!("invalid MicroFlow model {}: {message}", path.display()),
        )
    })
}

fn parse_model_signature(bytes: &[u8]) -> Result<ModelSignature, String> {
    if bytes.len() < 8 || !flatbuffers::buffer_has_identifier(bytes, TFLITE_FILE_IDENTIFIER, false)
    {
        return Err(format!(
            "expected the {TFLITE_FILE_IDENTIFIER:?} FlatBuffer identifier"
        ));
    }

    let model =
        root_as_model(bytes).map_err(|error| format!("FlatBuffer verification failed: {error}"))?;
    let subgraphs = model
        .subgraphs()
        .ok_or_else(|| "missing subgraphs vector".to_owned())?;
    let subgraph = subgraphs
        .iter()
        .next()
        .ok_or_else(|| "the TFLite model does not contain a subgraph".to_owned())?;

    let inputs = subgraph
        .inputs()
        .ok_or_else(|| "missing inputs vector".to_owned())?;
    let outputs = subgraph
        .outputs()
        .ok_or_else(|| "missing outputs vector".to_owned())?;

    if inputs.len() != 1 {
        return Err(format!(
            "MicroFlow requires exactly one input tensor, but the first subgraph declares {}",
            inputs.len()
        ));
    }
    if outputs.len() != 1 {
        return Err(format!(
            "MicroFlow requires exactly one output tensor, but the first subgraph declares {}",
            outputs.len()
        ));
    }

    let input = parse_tensor(subgraph, inputs.get(0), "input")?;
    let output = parse_tensor(subgraph, outputs.get(0), "output")?;

    Ok(ModelSignature { input, output })
}

fn parse_tensor(
    subgraph: SubGraph<'_>,
    tensor_index: i32,
    label: &str,
) -> Result<TensorSignature, String> {
    let tensors = subgraph
        .tensors()
        .ok_or_else(|| "missing tensors vector".to_owned())?;
    let tensor_index = usize::try_from(tensor_index)
        .map_err(|_| format!("the {label} tensor index is negative"))?;
    if tensor_index >= tensors.len() {
        return Err(format!(
            "the {label} tensor index {tensor_index} exceeds the tensor table length {}",
            tensors.len()
        ));
    }

    let tensor = tensors.get(tensor_index);
    let shape = tensor
        .shape()
        .ok_or_else(|| format!("the {label} tensor is missing its shape"))?;
    validate_static_shape(tensor, shape.len(), label)?;

    let mut dimensions = shape
        .iter()
        .map(|dimension| {
            usize::try_from(dimension)
                .ok()
                .filter(|dimension| *dimension > 0)
                .ok_or_else(|| {
                    format!(
                        "the {label} tensor has a dynamic or non-positive dimension {dimension}"
                    )
                })
        })
        .collect::<Result<Vec<_>, _>>()?;

    if dimensions.len() == 1 {
        dimensions.insert(0, 1);
    }
    if !matches!(dimensions.len(), 2 | 4) {
        return Err(format!(
            "the {label} tensor has rank {}; MicroFlow supports ranks 1, 2, and 4",
            dimensions.len()
        ));
    }

    let tensor_type = tensor.type_();
    if !matches!(tensor_type, TensorType::INT8 | TensorType::UINT8) {
        return Err(format!(
            "the {label} tensor type {} is unsupported; MicroFlow supports INT8 and UINT8 model tensors",
            tensor_type.0
        ));
    }

    validate_quantization(tensor, label)?;

    let byte_len = dimensions
        .iter()
        .try_fold(core::mem::size_of::<f32>(), |size, dimension| {
            size.checked_mul(*dimension)
        })
        .ok_or_else(|| format!("the {label} tensor byte size overflows usize"))?;

    Ok(TensorSignature {
        shape: dimensions,
        byte_len,
    })
}

fn validate_static_shape(
    tensor: Tensor<'_>,
    concrete_rank: usize,
    label: &str,
) -> Result<(), String> {
    let Some(signature) = tensor.shape_signature() else {
        return Ok(());
    };
    if signature.len() != concrete_rank {
        return Err(format!(
            "the {label} tensor shape signature rank does not match its concrete shape"
        ));
    }
    for dimension in signature.iter() {
        if dimension <= 0 {
            return Err(format!(
                "the {label} tensor has a dynamic shape signature dimension {dimension}"
            ));
        }
    }
    Ok(())
}

fn validate_quantization(tensor: Tensor<'_>, label: &str) -> Result<(), String> {
    let quantization = tensor
        .quantization()
        .ok_or_else(|| format!("the {label} tensor does not contain quantization parameters"))?;
    let scales = quantization
        .scale()
        .ok_or_else(|| format!("the {label} tensor does not contain quantization scales"))?;
    let zero_points = quantization
        .zero_point()
        .ok_or_else(|| format!("the {label} tensor does not contain quantization zero points"))?;
    if scales.len() != 1 || zero_points.len() != 1 {
        return Err(format!(
            "the {label} tensor uses per-axis quantization; MicroFlow requires one scale and one zero point"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_non_tflite_data() {
        let error = parse_model_signature(b"not a tflite model").unwrap_err();
        assert!(error.contains("TFL3"));
    }

    #[test]
    fn parses_quantized_microflow_fixture() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../examples/models/ds_cnn_s_quantized.tflite");
        let signature = load_model_signature(&path).unwrap();
        assert_eq!(signature.input.shape, vec![1, 490]);
        assert_eq!(signature.output.shape, vec![1, 12]);
    }
}
