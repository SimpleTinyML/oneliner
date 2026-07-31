mod support;

use OneLiner::model;
use OneLiner::runtime::ModelInference;

#[model(
    "../../examples/models/mcunet_10fps_vww.mlir",
    backend = "iree",
    arena = "shared"
)]
struct McunetMlirShared;

#[test]
fn mcunet_mlir_runs_with_shared_arena() {
    support::assert_artifacts::<McunetMlirShared>("mcunet_10fps_vww.mlir (shared)");

    let mut model = McunetMlirShared::new();
    let mut input = McunetMlirShared::create_input_tensor();
    input.fill(7);

    let output = model.run(&input);
    assert_eq!(output.as_slice(), &[4, -5]);
}
