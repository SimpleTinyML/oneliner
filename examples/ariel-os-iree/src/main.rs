#![no_main]
#![no_std]

use ariel_os::log::{
    error, 
    info,
};

use ariel_os::debug::{
    exit,
    ExitCode,
};

use ariel_os::time;

use OneLiner::model;
use OneLiner::runtime::{ModelSource, Predict, Aligned, AlignedType};


// #[model("models/mcunet-10fps_vww.tflite", backend = "iree")]
// struct Model;
// const INPUT_LEN: usize = 64 * 64 * 3;
// static INPUT: [u8; INPUT_LEN] = [7; INPUT_LEN];
// const EXPECTED: [u8; 2] = [4, 251];


#[model("models/lenet5_quantized.tflite", backend = "iree")]
struct Model;
const INPUT_LEN: usize = 28 * 28 * 4;
static INPUT: Aligned<AlignedType, [u8; INPUT_LEN]> = Aligned([7; INPUT_LEN]);
const OUTPUT_LEN: usize = 10 * 4;
const EXPECTED: [u8; OUTPUT_LEN] = [0; OUTPUT_LEN];

#[ariel_os::thread(autostart, priority = 1)]
fn main() {

    let artifacts = <Model as ModelSource>::ARTIFACTS;
    info!(
        "OneLiner MCUNet IREE example running on {}",
        ariel_os::buildinfo::BOARD
    );
    info!(
        "MCUNet artifact sizes: input={} output={}",
        artifacts.input_size, artifacts.output_size
    );

    if artifacts.input_size != INPUT.len() || artifacts.output_size != EXPECTED.len() {
        error!("MCUNet validation failed: unexpected artifact sizes");
        exit(ExitCode::FAILURE);
    }
    let time_begin_us = time::Instant::now().as_micros();
    let res_prediction = Model::try_predict(&INPUT[..]);
    let time_end_us = time::Instant::now().as_micros();
    info!("Model inference time: {:?} us", time_end_us - time_begin_us);

    match res_prediction {
        Ok(prediction) if prediction.as_bytes() == EXPECTED => {
            info!("MCUNet IREE validation passed");
            exit(ExitCode::SUCCESS);
        }
        Ok(prediction) if prediction.as_bytes().len() == EXPECTED.len() => {
            let actual = prediction.as_bytes();
            panic!(
                "MCUNet IREE validation failed: expected=[{}, {}] actual=[{}, {}]",
                EXPECTED[0], EXPECTED[1], actual[0], actual[1]
            );
            exit(ExitCode::FAILURE);
        }
        Ok(prediction) => {
            error!(
                "MCUNet IREE validation failed: output length {}",
                prediction.as_bytes().len()
            );
            exit(ExitCode::FAILURE);
        }
        Err(_) => {
            error!("MCUNet IREE validation failed: prediction returned an error");
            exit(ExitCode::FAILURE);
        }
    }
}