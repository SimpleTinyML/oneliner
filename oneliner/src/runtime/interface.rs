#[cfg(feature = "alloc")]
use super::Prediction;

use super::Access;

use ndarray::Array4;

pub type Tensor4D<T> = Array4<T>;

pub type Shape4D = (usize, usize, usize, usize);

pub type Shape = Shape4D;

pub type Tensor<T> = Tensor4D<T>;

/// Typed tensor inference interface implemented by generated model sessions.
pub trait ModelInference {
    type InputTensor;
    type OutputTensor;

    /// Runs inference and returns an owned output tensor.
    fn run(&mut self, input: &Self::InputTensor) -> Self::OutputTensor;

    /// Creates a zero-filled input tensor with the model's element type and shape.
    fn create_input_tensor() -> Self::InputTensor;
}

/// Common prediction interface implemented by backend model values and sessions.
///
/// Input: a backend-selected input type, usually `[u8]`.
/// Output: the backend-specific prediction value or an error from `try_predict`.
pub trait Predict<Input: ?Sized = [u8]> {
    type Error;
    type Output<'prediction>
    where
        Self: 'prediction;

    /// Runs prediction and panics if the backend returns an error.
    ///
    /// Input: any value that can be borrowed as `Input`.
    /// Output: `Self::Output` on success.
    fn predict<T>(&mut self, input: T) -> Self::Output<'_>
    where
        T: AsRef<Input>,
        Self::Error: core::fmt::Debug,
    {
        self.try_predict(input.as_ref())
            .expect("OneLiner prediction failed")
    }

    /// Runs prediction and returns the backend error instead of panicking.
    ///
    /// Input: a borrowed model input value.
    /// Output: `Ok(Self::Output)` on success or `Err(Self::Error)` on failure.
    fn try_predict<'prediction>(
        &'prediction mut self,
        input: &Input,
    ) -> core::result::Result<Self::Output<'prediction>, Self::Error>;

    /// Runs prediction and copies byte output into an owned prediction.
    ///
    /// Input: a borrowed model input value.
    /// Output: owned prediction bytes that remain valid after this predictor is reused.
    #[cfg(feature = "alloc")]
    fn try_predict_owned<'prediction>(
        &'prediction mut self,
        input: &Input,
    ) -> core::result::Result<Prediction<'static>, Self::Error>
    where
        Self::Output<'prediction>: AsRef<[u8]>,
    {
        let output = self.try_predict(input)?;
        Ok(Prediction::from_bytes(output.as_ref().to_vec()))
    }
}

/// Metadata exposed by a model generated with `#[model]`.
///
/// Input: no runtime input; metadata is provided as associated constants.
/// Output: model path and generated artifact metadata.
pub trait ModelSource {
    const MODEL_PATH: &'static str;
    const ARTIFACTS: ModelArtifacts;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    TooManyBindings {
        provided: usize,
        capacity: usize,
    },
    TooManyConstants {
        provided: usize,
        capacity: usize,
    },
    MissingDispatchFunction {
        ordinal: usize,
    },
    WorkgroupCountTooLarge {
        dimension: char,
        value: u32,
    },
    BufferRangeOutOfBounds {
        offset: usize,
        length: usize,
        capacity: usize,
    },
    DispatchFailed {
        status: i32,
    },
    InvalidAccess {
        access: Access,
        required: Access,
    },
    NullPointer,
}

impl core::fmt::Display for Error {
    /// Formats a runtime error for diagnostics.
    ///
    /// Input: formatter supplied by Rust formatting machinery.
    /// Output: `Ok(())` when the message was written.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::TooManyBindings { provided, capacity } => write!(
                f,
                "dispatch has {provided} bindings, but runtime stack storage holds {capacity}"
            ),
            Self::TooManyConstants { provided, capacity } => write!(
                f,
                "dispatch has {provided} constants, but the IREE ABI supports {capacity}"
            ),
            Self::MissingDispatchFunction { ordinal } => {
                write!(f, "IREE library does not expose dispatch ordinal {ordinal}")
            }
            Self::WorkgroupCountTooLarge { dimension, value } => write!(
                f,
                "workgroup count {dimension}={value} exceeds the IREE ABI limit"
            ),
            Self::BufferRangeOutOfBounds {
                offset,
                length,
                capacity,
            } => write!(
                f,
                "tensor range offset={offset} length={length} exceeds capacity {capacity}"
            ),
            Self::DispatchFailed { status } => {
                write!(f, "backend dispatch returned status {status}")
            }

            Self::InvalidAccess { access, required } => write!(
                f,
                "tensor access {access:?} is invalid; expected {required:?}"
            ),
            Self::NullPointer => write!(f, "tensor pointer is null"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelArtifacts {
    pub backend: &'static str,
    pub expansion: &'static str,
    pub model_path: &'static str,
    pub compile_input_path: &'static str,
    pub object_path: &'static str,
    pub link_path: &'static str,
    pub ir_path: &'static str,
    pub flow_rs_path: &'static str,
    pub metadata_json_path: &'static str,
    pub input_size: usize,
    pub output_size: usize,
}
