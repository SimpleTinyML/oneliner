use std::path::Path;

use super::error;
use super::types::{ElementType, TensorInfo};

#[derive(Clone, Copy)]
enum Syntax {
    Builtin,
    Torch,
}

impl Syntax {
    const ALL: [Self; 2] = [Self::Torch, Self::Builtin];

    const fn marker(self) -> &'static str {
        match self {
            Self::Builtin => "tensor<",
            Self::Torch => "!torch.vtensor<",
        }
    }
}

pub(super) fn parse_all(text: &str, path: &Path, label: &str) -> syn::Result<Vec<TensorInfo>> {
    let mut tensors = Vec::new();
    let mut remaining = text;
    while let Some((offset, syntax)) = find_syntax(remaining) {
        let body = &remaining[offset + syntax.marker().len()..];
        let close = matching_angle_bracket(body).ok_or_else(|| {
            error(
                path,
                format!("unterminated tensor type in @main {label} declaration"),
            )
        })?;
        tensors.push(match syntax {
            Syntax::Builtin => parse_builtin(&body[..close], path, label)?,
            Syntax::Torch => parse_torch(&body[..close], path, label)?,
        });
        remaining = &body[close + 1..];
    }
    Ok(tensors)
}

pub(super) fn type_end(text: &str) -> Option<usize> {
    let syntax = Syntax::ALL
        .into_iter()
        .find(|syntax| text.starts_with(syntax.marker()))?;
    let body = &text[syntax.marker().len()..];
    matching_angle_bracket(body).map(|close| syntax.marker().len() + close + 1)
}

fn find_syntax(text: &str) -> Option<(usize, Syntax)> {
    Syntax::ALL
        .into_iter()
        .filter_map(|syntax| text.find(syntax.marker()).map(|offset| (offset, syntax)))
        .min_by_key(|(offset, _)| *offset)
}

fn parse_builtin(text: &str, path: &Path, label: &str) -> syn::Result<TensorInfo> {
    let mut parts = text
        .split(',')
        .next()
        .expect("split always yields one item")
        .trim()
        .split('x')
        .collect::<Vec<_>>();
    let element = parts
        .pop()
        .ok_or_else(|| error(path, format!("empty @main {label} tensor type")))?;
    build(parts, element, path, label)
}

fn parse_torch(text: &str, path: &Path, label: &str) -> syn::Result<TensorInfo> {
    let text = text
        .strip_prefix('[')
        .ok_or_else(|| error(path, format!("invalid PyTorch @main {label} tensor shape")))?;
    let (shape, rest) = text.split_once(']').ok_or_else(|| {
        error(
            path,
            format!("unterminated PyTorch @main {label} tensor shape"),
        )
    })?;
    let element = rest
        .strip_prefix(',')
        .and_then(|rest| rest.split(',').next())
        .map(str::trim)
        .filter(|element| !element.is_empty())
        .ok_or_else(|| error(path, format!("missing PyTorch @main {label} element type")))?;
    let dimensions = if shape.trim().is_empty() {
        Vec::new()
    } else {
        shape.split(',').collect()
    };
    build(dimensions, element, path, label)
}

fn build(
    dimensions: Vec<&str>,
    element: &str,
    path: &Path,
    label: &str,
) -> syn::Result<TensorInfo> {
    let element_type = element_type(element.trim()).ok_or_else(|| {
        error(
            path,
            format!(
                "unsupported @main {label} tensor element type '{}'",
                element.trim()
            ),
        )
    })?;
    if dimensions.len() > 4 {
        return Err(error(
            path,
            format!(
                "@main {label} tensor rank {} exceeds Tensor's four dimensions",
                dimensions.len()
            ),
        ));
    }

    let mut shape = [1; 4];
    let offset = 4 - dimensions.len();
    for (index, dimension) in dimensions.into_iter().enumerate() {
        let dimension = dimension.trim();
        if dimension == "?" {
            return Err(error(
                path,
                format!("@main {label} tensor has a dynamic dimension"),
            ));
        }
        shape[offset + index] = dimension.parse().map_err(|_| {
            error(
                path,
                format!("invalid @main {label} tensor dimension '{dimension}'"),
            )
        })?;
    }
    Ok(TensorInfo {
        element_type,
        shape,
    })
}

fn element_type(value: &str) -> Option<ElementType> {
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

fn matching_angle_bracket(text: &str) -> Option<usize> {
    let mut depth = 1usize;
    for (offset, character) in text.char_indices() {
        match character {
            '<' => depth += 1,
            '>' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(offset);
                }
            }
            _ => {}
        }
    }
    None
}
