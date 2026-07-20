#![no_main]
#![no_std]

use ariel_os::debug::{exit, ExitCode};
use ariel_os::log::{error, info};
use ariel_os::time;

use OneLiner::model;
use OneLiner::runtime::{ModelSource, Predict, Aligned, AlignedType};


#[model("models/mcunet-10fps_vww.tflite", backend = "iree")]
struct Model;
const INPUT_LEN: usize = 64 * 64 * 3;
static INPUT: [u8; INPUT_LEN] = [7; INPUT_LEN];
const EXPECTED: [u8; 2] = [4, 251];


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

    if artifacts.input_size != INPUT.len() || artifacts.output_size != EXPECTED.len() {
        error!("Model validation failed: unexpected artifact sizes");
        exit(ExitCode::FAILURE);
    }
    let time_begin_us = time::Instant::now().as_micros();
    let prediction = Model::try_predict(&INPUT[..]);
    let time_end_us = time::Instant::now().as_micros();
    info!("Model inference time: {:?} us", time_end_us - time_begin_us);

    match prediction {
        Ok(prediction) => {
            let actual = prediction.as_bytes();
            if actual == EXPECTED {
                info!("Model IREE validation passed");
                exit(ExitCode::SUCCESS);
            }
            error!(
                "Model validation failed: expected {} output bytes, received {} bytes with different contents",
                EXPECTED.len(),
                actual.len()
            );
            exit(ExitCode::FAILURE);
        }
        Err(error) => {
            error!("Model validation failed");
            exit(ExitCode::FAILURE);
        }
    }
}
