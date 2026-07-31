mod support;

use OneLiner::model;
use OneLiner::runtime::ModelInference;

#[model(
    "../../examples/models/mcunet-10fps_vww.tflite",
    backend = "iree",
    arena = "owned"
)]
struct McunetTfliteOwned;

#[test]
fn mcunet_tflite_runs_with_owned_arena() {
    support::assert_artifacts::<McunetTfliteOwned>("mcunet-10fps_vww.tflite (owned)");

    let mut model = McunetTfliteOwned::new();
    let mut input = McunetTfliteOwned::create_input_tensor();
    input.fill(7);

    let output = model.run(&input);
    assert_eq!(output.as_slice(), &[4, -5]);
}
