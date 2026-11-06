mod support;

use oneliner::model;
use oneliner::runtime::ModelInference;

#[model(
    "../../examples/models/mcunet_10fps_vww.mlir",
    backend = "iree",
    arena = "owned"
)]
struct McunetMlirOwned;

#[test]
fn mcunet_mlir_runs_with_owned_arena() {
    let mut model = McunetMlirOwned::new();
    let mut input = McunetMlirOwned::create_input_tensor();
    input.fill(7);

    let output = model.run(&input);
    support::assert_metadata(
        "mcunet_10fps_vww.mlir (owned)",
        &McunetMlirOwned::METADATA,
        &McunetMlirOwned::RESOURCE_USAGE,
        input.byte_len(),
        output.byte_len(),
    );
    assert_eq!(output.as_slice(), &[4, -5]);
}
