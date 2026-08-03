mod support;

use OneLiner::model;
use OneLiner::runtime::ModelInference;

#[model(
    "../../examples/models/abs2_pytorch.pt2",
    backend = "iree",
    arena = "owned"
)]
struct Abs2PytorchOwned;

#[test]
fn abs2_pytorch_runs_with_owned_arena() {
    support::assert_artifacts::<Abs2PytorchOwned>("abs2_pytorch.pt2 (owned)");

    let mut model = Abs2PytorchOwned::new();
    let mut input = Abs2PytorchOwned::create_input_tensor();
    input.as_slice_mut().copy_from_slice(&[-2.5, 3.25]);

    let output = model.run(&input);
    support::assert_f32_slice_close(output.as_slice(), &[2.5, 3.25], f32::EPSILON);
}
