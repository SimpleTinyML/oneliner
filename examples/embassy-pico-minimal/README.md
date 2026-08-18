# Oneliner + IREE on Raspberry Pi Pico

This example runs a Oneliner model on an RP2040 with Embassy. It demonstrates bare-metal, `no_std` model inference with a shared static workspace and no Python or model interpreter on the device.

```rust
#[model(
    "../models/lenet5_quantized.tflite",
    backend = "iree",
    arena = "shared"
)]
struct Model;
```

## What This Example Shows

- Cross-compiling a TFLite model into RP2040-native code
- Running inference from an Embassy async application
- A synchronized shared workspace to avoid per-instance model workspaces
- Static input tensor storage to keep the large input off the task stack
- `defmt` logging over RTT
- On-device inference timing

## Active Model

The example currently uses `../models/lenet5_quantized.tflite`:

- input: `Tensor<f32, 1, 28, 28, 1>`
- output: `Tensor<f32, 1, 1, 1, 10>`
- input data: every element is filled with `7.0`
- memory mode: `shared`
- input storage: a `ConstStaticCell`

`EXPECTED` is currently a ten-element zero-filled placeholder. Replace it with reference output for your validation data before treating the comparison as a correctness test.

## Hardware and Tools

You need:

- a Raspberry Pi Pico or another compatible RP2040 board;
- a debug probe supported by `probe-rs`;
- `probe-rs` installed on the host;
- the stable Rust toolchain and `thumbv6m-none-eabi` target;
- the Python/IREE toolchain from [docs/installation.md](../../docs/installation.md).

The included Rust toolchain and Cargo configuration select the RP2040 target automatically. Keep the Python virtual environment active during the build.

## Build

From this directory:

```sh
cargo build --release
```

## Flash and Run

Connect the board and debug probe, then run:

```sh
cargo run --release
```

The configured Cargo runner uses:

```text
probe-rs run --chip RP2040
```

## Expected Behavior

The application initializes the RP2040, reports model artifact sizes, fills the static input tensor, runs inference, and reports elapsed microseconds through RTT.

It also prints expected-versus-actual diagnostics. The bundled zero-filled `EXPECTED` array is a placeholder, so use output captured from trusted reference inference before evaluating model correctness.

## Why Shared Storage?

The LeNet5 input and IREE workspace are significant on an RP2040 with 264 KiB of RAM. This example uses two complementary choices:

- `arena = "shared"` places one model workspace in static storage;
- `ConstStaticCell` places the input tensor in static storage.

These choices reduce task-stack pressure. They also mean that access must remain exclusive: a shared model arena cannot perform concurrent inference for the same model type.

## Switching Models

When replacing the bundled model:

1. update the path in `#[model(...)]`;
2. update the initial value used by `ConstStaticCell` to match the new input element type;
3. replace the input preparation and `EXPECTED` data;
4. check the generated artifact sizes and the board's RAM/flash limits;
5. adjust `memory.x` only if the board memory map is different.

Validate a new model with the [desktop example](../std-minimal/) before compiling it for RP2040.
