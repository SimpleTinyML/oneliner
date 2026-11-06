use std::path::Path;

use oneliner::runtime::{ModelMetadata, ResourceUsage};

pub fn assert_metadata(
    model_name: &str,
    metadata: &ModelMetadata,
    usage: &ResourceUsage,
    input_bytes: usize,
    output_bytes: usize,
) {
    let ModelMetadata {
        backend,
        model_path,
    } = *metadata;
    let ResourceUsage { io, ram: _, flash } = *usage;

    assert_eq!(backend, "iree", "{model_name}: unexpected backend");
    assert!(io.input_size > 0, "{model_name}: empty input binding");
    assert!(io.output_size > 0, "{model_name}: empty output binding");
    assert_eq!(
        io.input_size, input_bytes,
        "{model_name}: input size must match tensor byte length"
    );
    assert_eq!(
        io.output_size, output_bytes,
        "{model_name}: output size must match tensor byte length"
    );
    // Footprints may be zero for models without weights or scratch buffers
    // (for example a trivial identity/abs graph); they must still be reported.
    assert!(
        flash.code_size > 0,
        "{model_name}: compiled model object must contain machine code"
    );
    assert_eq!(
        flash.total_flash_size,
        flash.params_size + flash.code_size + flash.rodata_size,
        "{model_name}: total flash must equal params + code + rodata"
    );

    assert!(
        Path::new(model_path).exists(),
        "{model_name}: model does not exist: {model_path}"
    );
}

#[allow(dead_code)]
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
