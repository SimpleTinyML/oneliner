use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

const MODEL_EXTENSIONS: [&str; 3] = ["mlir", "onnx", "tflite"];
const ARENAS: [&str; 2] = ["owned", "shared"];

#[test]
fn every_example_model_has_an_owned_and_shared_test() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let model_dir = manifest_dir.join("../../examples/models");
    let test_dir = manifest_dir.join("tests");
    let test_sources = rust_sources(&test_dir);

    let model_names = fs::read_dir(&model_dir)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", model_dir.display()))
        .map(|entry| entry.expect("failed to read model directory entry").path())
        .filter(|path| {
            path.extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| MODEL_EXTENSIONS.contains(&extension))
        })
        .map(|path| {
            path.file_name()
                .expect("model path has no file name")
                .to_string_lossy()
                .into_owned()
        })
        .collect::<BTreeSet<_>>();

    assert!(!model_names.is_empty(), "no example models were discovered");

    for model_name in model_names {
        for arena in ARENAS {
            let arena_option = format!("arena = \"{arena}\"");
            assert!(
                test_sources.iter().any(|source| {
                    source.contains(&model_name) && source.contains(&arena_option)
                }),
                "missing std end-to-end test for {model_name} with the {arena} arena"
            );
        }
    }
}

fn rust_sources(test_dir: &Path) -> Vec<String> {
    fs::read_dir(test_dir)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", test_dir.display()))
        .map(|entry| entry.expect("failed to read test directory entry").path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "rs"))
        .map(|path| {
            fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
        })
        .collect()
}
