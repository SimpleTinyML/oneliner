mod support;

use oneliner::model;
use oneliner::runtime::ModelInference;

#[model(
    "../../examples/models/mcunet_10fps_vww.mlir",
    backend = "iree",
    arena = "shared"
)]
struct McunetMlirShared;

#[test]
fn mcunet_mlir_runs_with_shared_arena() {
    let mut model = McunetMlirShared::new();
    let mut input = McunetMlirShared::create_input_tensor();
    input.fill(7);

    let output = model.run(&input);
    support::assert_metadata(
        "mcunet_10fps_vww.mlir (shared)",
        &McunetMlirShared::METADATA,
        &McunetMlirShared::RESOURCE_USAGE,
        input.byte_len(),
        output.byte_len(),
    );
    assert_eq!(output.as_slice(), &[4, -5]);
}
