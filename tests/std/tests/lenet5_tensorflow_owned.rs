mod support;

use oneliner::model;
use oneliner::runtime::ModelInference;

#[model(
    "../../examples/models/lenet5_tensorflow",
    backend = "iree",
    arena = "owned",
    format = "tensorflow"
)]
struct LeNet5TensorflowOwned;

const EXPECTED: [f32; 10] = [
    0.031603746,
    0.03970769,
    0.050355315,
    0.06100294,
    0.059626535,
    0.06172616,
    -0.042626217,
    -0.05316555,
    -0.059627794,
    -0.04898017,
];

#[test]
fn lenet5_tensorflow_runs_with_owned_arena() {
    support::assert_artifacts::<LeNet5TensorflowOwned>("lenet5_tensorflow (owned)");

    let mut model = LeNet5TensorflowOwned::new();
    let mut input = LeNet5TensorflowOwned::create_input_tensor();
    input.fill(1.0);

    let output = model.run(&input);
    support::assert_f32_slice_close(output.as_slice(), &EXPECTED, 1.0e-5);
}
