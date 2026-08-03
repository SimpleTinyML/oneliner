mod support;

use OneLiner::model;
use OneLiner::runtime::ModelInference;

#[model(
    "../../examples/models/abs2_pytorch.pt2",
    backend = "iree",
    arena = "shared"
)]
struct Abs2PytorchShared;

#[test]
fn abs2_pytorch_runs_with_shared_arena() {
    support::assert_artifacts::<Abs2PytorchShared>("abs2_pytorch.pt2 (shared)");

    let mut model = Abs2PytorchShared::new();
    let mut input = Abs2PytorchShared::create_input_tensor();
    input.as_slice_mut().copy_from_slice(&[-2.5, 3.25]);

    let output = model.run(&input);
    support::assert_f32_slice_close(output.as_slice(), &[2.5, 3.25], f32::EPSILON);
}
