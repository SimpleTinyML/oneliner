use proc_macro2::Span;
use serde::Deserialize;

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum ElementType {
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
    pub(crate) fn rust_tokens(self) -> proc_macro2::TokenStream {
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

    pub(crate) const fn byte_width(self) -> usize {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::I64 | Self::U64 | Self::F64 => 8,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct TensorInfo {
    pub(crate) element_type: ElementType,
    pub(crate) shape: [usize; 4],
}

impl TensorInfo {
    pub(crate) fn byte_len(&self) -> Option<usize> {
        self.shape
            .iter()
            .try_fold(self.element_type.byte_width(), |size, dimension| {
                size.checked_mul(*dimension)
            })
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct ModelIo {
    pub(crate) input: TensorInfo,
    pub(crate) output: TensorInfo,
}

impl ModelIo {
    pub(crate) fn validate(&self) -> syn::Result<()> {
        for (label, tensor) in [("input", &self.input), ("output", &self.output)] {
            tensor.byte_len().ok_or_else(|| {
                syn::Error::new(
                    Span::call_site(),
                    format!(
                        "{label} tensor byte size overflows usize for shape {:?}",
                        tensor.shape
                    ),
                )
            })?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_tensor_byte_size_overflow() {
        let model_io = ModelIo {
            input: TensorInfo {
                element_type: ElementType::F64,
                shape: [usize::MAX, 2, 1, 1],
            },
            output: TensorInfo {
                element_type: ElementType::F32,
                shape: [1; 4],
            },
        };

        assert!(model_io.validate().is_err());
    }
}
