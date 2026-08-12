use std::{fs, path::Path};

use super::error;
use super::mlir_tensor;
use super::types::ModelIo;

pub(super) fn load(path: &Path) -> syn::Result<ModelIo> {
    let text = fs::read_to_string(path).map_err(|error| {
        syn::Error::new(
            proc_macro2::Span::call_site(),
            format!("failed to read model I/O from {}: {error}", path.display()),
        )
    })?;
    parse(&text, path)
}

fn parse(text: &str, path: &Path) -> syn::Result<ModelIo> {
    let main = find_main(text)
        .ok_or_else(|| error(path, "model does not contain an @main entry function"))?;
    let input_open = main
        .find('(')
        .ok_or_else(|| error(path, "@main does not contain an input argument list"))?;
    let input_close = matching_delimiter(main, input_open, '(', ')')
        .ok_or_else(|| error(path, "@main has an unterminated input argument list"))?;
    let input = mlir_tensor::parse_all(&main[input_open + 1..input_close], path, "input")?
        .into_iter()
        .next()
        .unwrap();

    let output = main[input_close + 1..]
        .trim_start()
        .strip_prefix("->")
        .ok_or_else(|| error(path, "@main does not contain an output type"))?
        .trim_start();
    let output = if output.starts_with('(') {
        let close = matching_delimiter(output, 0, '(', ')')
            .ok_or_else(|| error(path, "@main has an unterminated output type list"))?;
        &output[1..close]
    } else {
        let end = mlir_tensor::type_end(output)
            .ok_or_else(|| error(path, "@main output is not a supported tensor"))?;
        &output[..end]
    };
    let output = mlir_tensor::parse_all(output, path, "output")?
        .into_iter()
        .next()
        .unwrap();

    Ok(ModelIo { input, output })
}

fn find_main(text: &str) -> Option<&str> {
    for (offset, _) in text.match_indices("@main") {
        let after_name = &text[offset + "@main".len()..];
        if !after_name
            .chars()
            .next()
            .is_some_and(|character| character == '(' || character.is_whitespace())
        {
            continue;
        }

        let before_name = &text[..offset];
        let Some((function_offset, operation_len)) = ["func.func", "util.func"]
            .into_iter()
            .filter_map(|operation| {
                before_name
                    .rfind(operation)
                    .map(|offset| (offset, operation.len()))
            })
            .max_by_key(|(offset, _)| *offset)
        else {
            continue;
        };
        if !text[function_offset + operation_len..offset].contains(['{', '}']) {
            return Some(after_name);
        }
    }
    None
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frontend::model_io::types::ElementType;

    #[test]
    fn parses_builtin_tensors_with_attributes() {
        let text = r#"
            module {
              util.func public @main(
                %arg0: tensor<1x64x64x3xi8> {ml_program.identifier = "input"}
              ) -> (tensor<1x2xi8> {ml_program.identifier = "output"}) {
                util.return %arg0 : tensor<1x2xi8>
              }
            }
        "#;
        let model_io = parse(text, Path::new("model.mlir")).unwrap();

        assert!(matches!(model_io.input.element_type, ElementType::I8));
        assert_eq!(model_io.input.shape, [1, 64, 64, 3]);
        assert_eq!(model_io.output.shape, [1, 1, 1, 2]);
    }

    #[test]
    fn selects_main_instead_of_helper() {
        let text = r#"
            func.func private @helper(%arg0: tensor<4xi32>) -> tensor<4xi32>
            func.func @main(%arg0: tensor<2xf32>) -> tensor<2xf32>
        "#;
        let model_io = parse(text, Path::new("model.mlir")).unwrap();

        assert!(matches!(model_io.input.element_type, ElementType::F32));
        assert_eq!(model_io.input.shape, [1, 1, 1, 2]);
        assert_eq!(model_io.output.shape, [1, 1, 1, 2]);
    }

    #[test]
    fn parses_torch_tensors() {
        let text = r#"
            module @lenet5_pytorch {
              func.func @main(
                %arg0: !torch.vtensor<[1, 1, 32, 32],f32>
              ) -> !torch.vtensor<[1, 10],f32> {
                return %arg0 : !torch.vtensor<[1, 10],f32>
              }
            }
        "#;
        let model_io = parse(text, Path::new("model.torch.mlir")).unwrap();

        assert_eq!(model_io.input.shape, [1, 1, 32, 32]);
        assert_eq!(model_io.output.shape, [1, 1, 1, 10]);
    }
}
