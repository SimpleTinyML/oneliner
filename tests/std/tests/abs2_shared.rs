mod support;

use OneLiner::model;
use OneLiner::runtime::ModelInference;

#[model("../../examples/models/abs2.mlir", backend = "iree", arena = "shared")]
struct Abs2Shared;

#[test]
fn abs2_mlir_runs_with_shared_arena() {
    support::assert_artifacts::<Abs2Shared>("abs2.mlir (shared)");

    let mut model = Abs2Shared::new();
    let mut input = Abs2Shared::create_input_tensor();
    input.as_slice_mut().copy_from_slice(&[-2.5, 3.25]);

    let output = model.run(&input);
    support::assert_f32_slice_close(output.as_slice(), &[2.5, 3.25], f32::EPSILON);
}
