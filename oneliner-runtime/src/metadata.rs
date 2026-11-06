//! Model provenance and compile-time estimates of tensor and memory sizes.

/// Model provenance exposed by a generated model's `METADATA` constant.
///
/// Resource estimates are available separately through the model's
/// `RESOURCE_USAGE` constant of type [`ResourceUsage`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelMetadata {
    /// Backend name, currently `iree`.
    pub backend: &'static str,
    /// Original model file or SavedModel directory on the build host.
    ///
    /// This path records the model's source; inference does not read it.
    pub model_path: &'static str,
}

/// Tensor sizes and model-only memory estimates exposed by `RESOURCE_USAGE`.
///
/// All sizes are byte counts. Estimates describe generated resources and object
/// sections before final application linking; they omit application/OS state,
/// stack usage and allocator overhead. Alignment and final linker placement can
/// change the actual footprint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceUsage {
    /// Input and output tensor payload sizes.
    pub io: IOSize,
    /// Temporary workspace payload size.
    pub ram: RAMUsage,
    /// Model weights and compiled object-section sizes.
    pub flash: FlashUsage,
}

/// Input and output tensor payload sizes in bytes, excluding alignment padding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IOSize {
    /// Size in bytes of the model's input tensor.
    pub input_size: usize,
    /// Size in bytes of the model's output tensor.
    pub output_size: usize,
}

/// Model workspace payload estimate in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RAMUsage {
    /// Deduplicated temporary resource bytes for one workspace.
    ///
    /// Excludes alignment padding, input/output tensors, stack usage and
    /// allocator overhead; this is not the complete arena allocation size.
    pub arena_size: usize,
}

/// Model-only flash footprint estimate in bytes before final linking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlashUsage {
    /// Deduplicated model parameter/weight bytes.
    pub params_size: usize,
    /// Machine code bytes of the compiled model.
    pub code_size: usize,
    /// Read-only data bytes embedded in the compiled model object (lookup
    /// tables, library metadata, unwind tables).
    pub rodata_size: usize,
    /// Sum of `params_size`, `code_size` and `rodata_size` before final linking.
    pub total_flash_size: usize,
}
