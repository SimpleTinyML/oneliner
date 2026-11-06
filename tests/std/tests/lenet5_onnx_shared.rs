mod support;

use oneliner::model;
use oneliner::runtime::ModelInference;

#[model(
    "../../examples/models/lenet5_quantized.onnx",
    backend = "iree",
    arena = "shared"
)]
struct LeNet5OnnxShared;

const EXPECTED: [f32; 10] = [
    0.11666615, 0.11666615, 0.17499924, 0.68541366, 0.0, 0.33541518, 0.0, 0.0, 1.2541611,
    0.14583269,
];

#[test]
fn lenet5_onnx_runs_with_shared_arena() {
    let mut model = LeNet5OnnxShared::new();
    let mut input = LeNet5OnnxShared::create_input_tensor();
    input.fill(7.0);

    let output = model.run(&input);
    support::assert_metadata(
        "lenet5_quantized.onnx (shared)",
        &LeNet5OnnxShared::METADATA,
        &LeNet5OnnxShared::RESOURCE_USAGE,
        input.byte_len(),
        output.byte_len(),
    );
    support::assert_f32_slice_close(output.as_slice(), &EXPECTED, 1.0e-5);
}
