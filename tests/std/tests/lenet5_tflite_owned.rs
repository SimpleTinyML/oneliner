mod support;

use oneliner::model;
use oneliner::runtime::ModelInference;

#[model(
    "../../examples/models/lenet5_quantized.tflite",
    backend = "iree",
    arena = "owned"
)]
struct LeNet5TfliteOwned;

const EXPECTED: [f32; 10] = [
    0.11666615, 0.11666615, 0.13124943, 0.68541366, 0.0, 0.36458173, 0.0, 0.0, 1.2104113,
    0.16041596,
];

#[test]
fn lenet5_tflite_runs_with_owned_arena() {
    support::assert_artifacts::<LeNet5TfliteOwned>("lenet5_quantized.tflite (owned)");

    let mut model = LeNet5TfliteOwned::new();
    let mut input = LeNet5TfliteOwned::create_input_tensor();
    input.fill(7.0);

    let output = model.run(&input);
    support::assert_f32_slice_close(output.as_slice(), &EXPECTED, 1.0e-5);
}
