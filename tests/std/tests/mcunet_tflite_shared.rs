mod support;

use OneLiner::model;
use OneLiner::runtime::ModelInference;

#[model(
    "../../examples/models/mcunet-10fps_vww.tflite",
    backend = "iree",
    arena = "shared"
)]
struct McunetTfliteShared;

#[test]
fn mcunet_tflite_runs_with_shared_arena() {
    support::assert_artifacts::<McunetTfliteShared>("mcunet-10fps_vww.tflite (shared)");

    let mut model = McunetTfliteShared::new();
    let mut input = McunetTfliteShared::create_input_tensor();
    input.fill(7);

    let output = model.run(&input);
    assert_eq!(output.as_slice(), &[4, -5]);
}
