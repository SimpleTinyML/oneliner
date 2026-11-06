mod support;

use oneliner::model;
use oneliner::runtime::ModelInference;

#[model(
    "../../examples/models/lenet5_pytorch.pt2",
    backend = "iree",
    arena = "shared"
)]
struct LeNet5PytorchShared;

const EXPECTED: [f32; 10] = [
    0.028477922,
    0.062621646,
    0.020966977,
    0.049119197,
    0.07007058,
    0.028415913,
    -0.05843186,
    -0.037480474,
    -0.06808296,
    -0.04674624,
];

#[test]
fn lenet5_pytorch_runs_with_shared_arena() {
    let mut model = LeNet5PytorchShared::new();
    let mut input = LeNet5PytorchShared::create_input_tensor();
    input.fill(1.0);

    let output = model.run(&input);
    support::assert_metadata(
        "lenet5_pytorch.pt2 (shared)",
        &LeNet5PytorchShared::METADATA,
        &LeNet5PytorchShared::RESOURCE_USAGE,
        input.byte_len(),
        output.byte_len(),
    );
    support::assert_f32_slice_close(output.as_slice(), &EXPECTED, 1.0e-5);
}
