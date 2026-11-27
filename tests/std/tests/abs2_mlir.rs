mod support;

use oneliner::model;
use oneliner::runtime::ModelInference;
use oneliner_profiler::Profiler;

#[model("../../examples/models/abs2.mlir", backend = "iree")]
struct Abs2Mlir;

#[test]
fn abs2_mlir_runs() {
    let mut model = Abs2Mlir::new();
    let mut profiler = Profiler::new();
    let mut input = Abs2Mlir::create_input_tensor();
    input.as_slice_mut().copy_from_slice(&[-2.5, 3.25]);

    let output = profiler.profile(|| model.run(&input));
    assert_eq!(profiler.stats().samples, 1);
    support::assert_f32_slice_close(output.as_slice(), &[2.5, 3.25], f32::EPSILON);
}
