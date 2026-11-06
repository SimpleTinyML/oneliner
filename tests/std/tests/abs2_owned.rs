mod support;

use oneliner::runtime::ModelInference;

mod models {
    use oneliner::model;

    #[model("../../examples/models/abs2.mlir", backend = "iree", arena = "owned")]
    pub struct Abs2Owned;
}

use models::Abs2Owned;

#[test]
fn abs2_mlir_runs_with_owned_arena() {
    let mut model = Abs2Owned::new();
    let mut input = Abs2Owned::create_input_tensor();
    input.as_slice_mut().copy_from_slice(&[-2.5, 3.25]);

    let output = model.run(&input);
    support::assert_metadata(
        "abs2.mlir (owned)",
        &Abs2Owned::METADATA,
        &Abs2Owned::RESOURCE_USAGE,
        input.byte_len(),
        output.byte_len(),
    );
    support::assert_f32_slice_close(output.as_slice(), &[2.5, 3.25], f32::EPSILON);
}
