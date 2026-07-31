mod support;

use OneLiner::model;
use OneLiner::runtime::ModelInference;

#[model("../../examples/models/abs2.mlir", backend = "iree", arena = "owned")]
struct Abs2Owned;

#[test]
fn abs2_mlir_runs_with_owned_arena() {
    support::assert_artifacts::<Abs2Owned>("abs2.mlir (owned)");

    let mut model = Abs2Owned::new();
    let mut input = Abs2Owned::create_input_tensor();
    input.as_slice_mut().copy_from_slice(&[-2.5, 3.25]);

    let output = model.run(&input);
    support::assert_f32_slice_close(output.as_slice(), &[2.5, 3.25], f32::EPSILON);
}
