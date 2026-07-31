mod support;

use OneLiner::model;
use OneLiner::runtime::ModelInference;

#[model(
    "../../examples/models/mcunet_10fps_vww.mlir",
    backend = "iree",
    arena = "owned"
)]
struct McunetMlirOwned;

#[test]
fn mcunet_mlir_runs_with_owned_arena() {
    support::assert_artifacts::<McunetMlirOwned>("mcunet_10fps_vww.mlir (owned)");

    let mut model = McunetMlirOwned::new();
    let mut input = McunetMlirOwned::create_input_tensor();
    input.fill(7);

    let output = model.run(&input);
    assert_eq!(output.as_slice(), &[4, -5]);
}
