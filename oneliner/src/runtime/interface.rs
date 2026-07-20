pub type Result<T> = core::result::Result<T, Error>;

/// Common prediction interface implemented by every backend-generated model.
///
/// Input: a backend-selected input type, usually `[u8]`.
/// Output: the backend-specific prediction value or an error from `try_predict`.
pub trait Predict<Input: ?Sized = [u8]> {
    type Error;
    type Output;

    /// Runs prediction and panics if the backend returns an error.
    ///
    /// Input: any value that can be borrowed as `Input`.
    /// Output: `Self::Output` on success.
    fn predict<T>(input: T) -> Self::Output
    where
        T: AsRef<Input>,
        Self::Error: core::fmt::Debug,
    {
        Self::try_predict(input.as_ref()).expect("OneLiner prediction failed")
    }

    /// Runs prediction and returns the backend error instead of panicking.
    ///
    /// Input: a borrowed model input value.
    /// Output: `Ok(Self::Output)` on success or `Err(Self::Error)` on failure.
    fn try_predict(input: &Input) -> core::result::Result<Self::Output, Self::Error>;
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
    InputSizeMismatch {
        provided: usize,
        expected: usize,
    },
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
    TensorRangeOutOfBounds {
        offset: usize,
        length: usize,
        capacity: usize,
    },
    DispatchFailed {
        status: i32,
    },
}

impl core::fmt::Display for Error {
    /// Formats a runtime error for diagnostics.
    ///
    /// Input: formatter supplied by Rust formatting machinery.
    /// Output: `Ok(())` when the message was written.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InputSizeMismatch { provided, expected } => write!(
                f,
                "input has {provided} bytes, but the model expects exactly {expected} bytes"
            ),
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
            Self::TensorRangeOutOfBounds {
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
