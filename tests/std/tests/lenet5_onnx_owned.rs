mod support;

use OneLiner::model;
use OneLiner::runtime::ModelInference;

#[model(
    "../../examples/models/lenet5_quantized.onnx",
    backend = "iree",
    arena = "owned"
)]
struct LeNet5OnnxOwned;

const EXPECTED: [f32; 10] = [
    0.11666615, 0.11666615, 0.17499924, 0.68541366, 0.0, 0.33541518, 0.0, 0.0, 1.2541611,
    0.14583269,
];

#[test]
fn lenet5_onnx_runs_with_owned_arena() {
    support::assert_artifacts::<LeNet5OnnxOwned>("lenet5_quantized.onnx (owned)");

    let mut model = LeNet5OnnxOwned::new();
    let mut input = LeNet5OnnxOwned::create_input_tensor();
    input.fill(7.0);

    let output = model.run(&input);
    support::assert_f32_slice_close(output.as_slice(), &EXPECTED, 1.0e-5);
}
