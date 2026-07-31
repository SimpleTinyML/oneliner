use OneLiner::model;
use OneLiner::runtime::{ModelInference, ModelSource};

#[model("../models/ds_cnn_s_quantized.tflite", backend = "microflow")]
struct Model;

const CLASS_COUNT: usize = 12;

fn main() {
    let artifacts = <Model as ModelSource>::ARTIFACTS;
    println!(
        "MicroFlow model buffers: input={} bytes, output={} bytes",
        artifacts.input_size, artifacts.output_size
    );

    let mut model = Model::new();
    let input = Model::create_input_tensor();

    // MicroFlow's InputRefOrVal is the buffer itself, so run consumes it
    // without converting through OneLiner's row-major Tensor type.
    let output = model.run(input);
    let (class, score) = (0..CLASS_COUNT)
        .map(|class| (class, output[(0, class)]))
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .expect("the model has at least one output class");

    println!("MicroFlow inference completed: class={class}, score={score}");
}
