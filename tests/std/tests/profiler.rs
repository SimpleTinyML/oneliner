mod support;

use oneliner::model;
use oneliner::profiler::Profiler;
use oneliner::runtime::ModelInference;

#[model("../../examples/models/abs2.mlir", backend = "iree", arena = "owned")]
struct Abs2Profiled;

#[test]
fn profiler_measures_inference_latency() {
    let mut model = Abs2Profiled::new();
    let mut profiler = Profiler::new();
    let mut input = Abs2Profiled::create_input_tensor();
    input.as_slice_mut().copy_from_slice(&[-2.5, 3.25]);

    for _ in 0..5 {
        let output = profiler.profile(|| model.run(&input));
        support::assert_f32_slice_close(output.as_slice(), &[2.5, 3.25], f32::EPSILON);
    }

    let stats = profiler.stats();
    assert_eq!(stats.samples, 5);
    assert!(stats.min.is_some());
    assert!(stats.max.is_some());
    assert!(stats.average().unwrap() >= stats.min.unwrap());
    assert!(stats.average().unwrap() <= stats.max.unwrap());

    profiler.reset_stats();
    assert_eq!(profiler.stats().samples, 0);
}