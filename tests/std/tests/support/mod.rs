use std::path::Path;

use OneLiner::runtime::{ModelArtifacts, ModelSource};

pub fn assert_artifacts<M: ModelSource>(model_name: &str) {
    let ModelArtifacts {
        backend,
        expansion,
        model_path,
        compile_input_path,
        object_path,
        link_path,
        ir_path,
        flow_rs_path,
        metadata_json_path,
        input_size,
        output_size,
    } = M::ARTIFACTS;

    assert_eq!(backend, "iree", "{model_name}: unexpected backend");
    assert_eq!(
        expansion, "static-flow",
        "{model_name}: unexpected expansion"
    );
    assert!(input_size > 0, "{model_name}: empty input binding");
    assert!(output_size > 0, "{model_name}: empty output binding");

    for (label, path) in [
        ("model", model_path),
        ("compile input", compile_input_path),
        ("object", object_path),
        ("link", link_path),
        ("stream/flow IR", ir_path),
        ("generated Rust flow", flow_rs_path),
        ("metadata", metadata_json_path),
    ] {
        assert!(
            Path::new(path).is_file(),
            "{model_name}: {label} artifact does not exist: {path}"
        );
    }
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
