use oneliner::model;
use oneliner::runtime::ModelInference;

#[model("../../examples/models/mcunet-10fps_vww.tflite", backend = "iree")]
struct McunetTflite;

#[test]
fn mcunet_tflite_runs() {
    let mut model = McunetTflite::new();
    let mut input = McunetTflite::create_input_tensor();
    input.fill(7);

    let output = model.run(&input);
    assert_eq!(output.as_slice(), &[4, -5]);
}
