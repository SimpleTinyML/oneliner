use oneliner::runtime::{ModelMetadata, ResourceUsage};

/// Checks the public metadata/resource-usage contract once on a non-trivial model.
///
/// Tensor byte sizes are already validated against the frontend model at macro
/// expansion time; this keeps one end-to-end check of the public constants.
#[allow(dead_code)] // Each integration-test target compiles this module separately.
pub fn assert_usage(
    metadata: &ModelMetadata,
    usage: &ResourceUsage,
    input_bytes: usize,
    output_bytes: usize,
) {
    assert_eq!(metadata.backend, "iree", "unexpected backend");
    assert_eq!(
        usage.io.input_size, input_bytes,
        "input size must match tensor byte length"
    );
    assert_eq!(
        usage.io.output_size, output_bytes,
        "output size must match tensor byte length"
    );
    assert!(
        usage.flash.code_size > 0,
        "compiled model object must contain machine code"
    );
    assert_eq!(
        usage.flash.total_flash_size,
        usage.flash.params_size + usage.flash.code_size + usage.flash.rodata_size,
        "total flash must equal params + code + rodata"
    );
}

#[allow(dead_code)] // Each integration-test target compiles this module separately.
pub fn assert_f32_slice_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len(), "output length differs");
    for (index, (&actual_value, &expected_value)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual_value - expected_value).abs() <= tolerance,
            "output[{index}] differs: expected {expected_value}, got {actual_value} (tolerance {tolerance}); \
             expected output: {expected_slice:?}; actual output: {actual_slice:?}",
            expected_slice = expected,
            actual_slice = actual,
        );
    }
}
