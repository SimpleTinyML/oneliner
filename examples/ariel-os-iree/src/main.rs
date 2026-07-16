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

use OneLiner::model;
use OneLiner::runtime::{ModelSource, Predict, Aligned, AlignedType};


#[model("models/mcunet_10fps_vww.mlir", backend = "iree")]
// #[model("models/lenet5_quantized.tflite", backend = "iree")]
struct McunetVww;

const INPUT_LEN: usize = 64 * 64 * 3;
static INPUT: [u8; INPUT_LEN] = [7; INPUT_LEN];

const EXPECTED: [u8; 2] = [4, 251];

#[ariel_os::task(autostart)]
async fn main() {
    let artifacts = <McunetVww as ModelSource>::ARTIFACTS;
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

    match McunetVww::try_predict(&INPUT) {
        Ok(prediction) if prediction.as_bytes() == EXPECTED => {
            info!("MCUNet IREE validation passed");
            exit(ExitCode::SUCCESS);
        }
        Ok(prediction) if prediction.as_bytes().len() == EXPECTED.len() => {
            let actual = prediction.as_bytes();
            error!(
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