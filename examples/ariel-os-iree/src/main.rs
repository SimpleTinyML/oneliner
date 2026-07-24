#![no_main]
#![no_std]

use ariel_os::debug::{exit, ExitCode};
use ariel_os::log::{error, info};
use ariel_os::reexports::static_cell::ConstStaticCell;
use ariel_os::time;

use OneLiner::model;
use OneLiner::runtime::{ModelInference, ModelSource};

#[model("models/mcunet-10fps_vww.tflite", backend = "iree")]
struct Model;
const INPUT_LEN: usize = 64 * 64 * 3;
const EXPECTED: [i8; 2] = [4, -5];

static WORKSPACE: ConstStaticCell<ModelWorkspace> = ConstStaticCell::new(ModelWorkspace::new());

// #[model("models/lenet5_quantized.tflite", backend = "iree")]
// struct Model;
// const INPUT_LEN: usize = 28 * 28 * 4;
// static INPUT: Aligned<AlignedType, [u8; INPUT_LEN]> = Aligned([7; INPUT_LEN]);
// const OUTPUT_LEN: usize = 10 * 4;
// const EXPECTED: [u8; OUTPUT_LEN] = [0; OUTPUT_LEN];

#[ariel_os::thread(autostart, priority = 1)]
fn main() {
    let artifacts = <Model as ModelSource>::ARTIFACTS;
    info!(
        "OneLiner IREE example running on {}",
        ariel_os::buildinfo::BOARD
    );
    info!(
        "Model artifact sizes: input={} output={}",
        artifacts.input_size, artifacts.output_size
    );

    if artifacts.input_size != INPUT_LEN || artifacts.output_size != EXPECTED.len() {
        error!("Model validation failed: unexpected artifact sizes");
        exit(ExitCode::FAILURE);
    }
    let mut model = Model::session(WORKSPACE.take());
    let mut input = ModelSession::create_input_tensor();
    input.fill(7);
    let time_begin_us = time::Instant::now().as_micros();
    let output = model.run(&input);
    let time_end_us = time::Instant::now().as_micros();
    info!("Model inference time: {:?} us", time_end_us - time_begin_us);

    let actual = output.as_slice().expect("output tensor is contiguous");
    if actual == EXPECTED {
        info!("Model IREE validation passed");
        exit(ExitCode::SUCCESS);
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
    exit(ExitCode::FAILURE);
}
