use oneliner::model;
use oneliner::runtime::ModelInference;

#[model(
    "../../examples/models/mcunet-10fps_vww.tflite",
    backend = "iree",
    arena = "owned"
)]
struct McunetTfliteOwned;

#[test]
fn mcunet_tflite_runs_with_owned_arena() {
    let mut model = McunetTfliteOwned::new();
    let mut input = McunetTfliteOwned::create_input_tensor();
    input.fill(7);

    let output = model.run(&input);
    assert_eq!(output.as_slice(), &[4, -5]);
}
