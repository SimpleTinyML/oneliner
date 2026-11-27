#![no_std]
#![no_main]

use embassy_executor::Spawner;
use embassy_time::{Duration, Timer};
use {defmt_rtt as _, panic_probe as _};
use defmt::{info, error};

use oneliner::model;
use oneliner::runtime::ModelInference;
use static_cell::ConstStaticCell;

#[model("../models/lenet5_quantized.tflite")]
struct Model;
const INPUT_LEN: usize = 28 * 28 * 1;
const OUTPUT_LEN: usize = 10;
const EXPECTED: [f32; OUTPUT_LEN] = [
    0.11666615, 0.11666615, 0.13124943, 0.68541366, 0.0, 0.36458173, 0.0, 0.0, 1.2104113,
    0.16041596,
];

static INPUT_CELL: ConstStaticCell<<Model as ModelInference>::InputTensor> = ConstStaticCell::new(<Model as ModelInference>::InputTensor::new(0.0));
static MODEL: ConstStaticCell<Model> = ConstStaticCell::new(Model::new());

#[embassy_executor::main]
async fn main(_spawner: Spawner) {
    let _p = embassy_rp::init(Default::default());
    let usage = Model::RESOURCE_USAGE;

    info!(
        "Model tensor sizes: input={} output={}",
        usage.io.input_size, usage.io.output_size
    );

    let mut model = MODEL.take();
    // let mut input = Model::create_input_tensor();
    let mut input = INPUT_CELL.take();
    input.fill(7.0);
    let time_begin_us = embassy_time::Instant::now().as_micros();
    let output = model.run(&input);
    let time_end_us = embassy_time::Instant::now().as_micros();
    info!("Model inference time: {:?} us", time_end_us - time_begin_us);

    let actual = output.as_slice();
    if actual == EXPECTED {
        info!("Model IREE validation passed");
    }
    error!(
        "Model validation failed: expected {} output elements, received {} elements with different values",
        EXPECTED.len(),
        actual.len()
    );
    error!(
        "EXPECTED: [{}, {}], received: [{}, {}]",
        EXPECTED[0], EXPECTED[1], actual[0], actual[1]
    );
}
