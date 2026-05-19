#![no_main]
#![no_std]

use ariel_os::debug::{
    exit,
    log::{error, info},
    ExitCode,
};
use OneLiner::model;
use OneLiner::runtime::{ModelSource, Predict};

#[model("models/abs2.mlir", backend = "iree")]
struct Abs2;

const INPUT: [u8; 8] = [
    0x00, 0x00, 0x80, 0xbf, // -1.0f32
    0x00, 0x00, 0x40, 0x40, // 3.0f32
];

const EXPECTED: [u8; 8] = [
    0x00, 0x00, 0x80, 0x3f, // 1.0f32
    0x00, 0x00, 0x40, 0x40, // 3.0f32
];

#[ariel_os::task(autostart)]
async fn main() {
    let artifacts = <Abs2 as ModelSource>::ARTIFACTS;
    info!(
        "OneLiner IREE example running on {}",
        ariel_os::buildinfo::BOARD
    );
    info!(
        "OneLiner artifact sizes: input={} output={}",
        artifacts.input_size, artifacts.output_size
    );

    match Abs2::try_predict(&INPUT) {
        Ok(prediction) if prediction.as_bytes() == EXPECTED => {
            info!("OneLiner IREE validation passed");
            exit(ExitCode::SUCCESS);
        }
        Ok(_) => {
            error!("OneLiner IREE validation failed: unexpected output");
            exit(ExitCode::FAILURE);
        }
        Err(_) => {
            error!("OneLiner IREE validation failed: prediction returned an error");
            exit(ExitCode::FAILURE);
        }
    }
}
