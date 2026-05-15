use super::interface::ModelSource;

/// User-implemented hook for the Microflow backend.
///
/// Input: raw model input bytes supplied by `Predict`.
/// Output: backend-defined prediction output or backend-defined error.
pub trait MicroflowModel: ModelSource {
    type Error;
    type Output;

    /// Runs Microflow prediction for the generated model type.
    ///
    /// Input: byte slice passed to `MyModel::try_predict`.
    /// Output: `Ok(Output)` on success or `Err(Error)` on failure.
    fn try_predict_microflow(input: &[u8]) -> core::result::Result<Self::Output, Self::Error>;
}
