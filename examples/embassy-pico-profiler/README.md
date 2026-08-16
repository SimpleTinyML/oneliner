# Oneliner + IREE + Profiler on Raspberry Pi Pico

This example runs a Oneliner model on an RP2040 with Embassy and measures its
inference latency with the `oneliner-profiler` crate. It is the profiler
counterpart of the [`embassy-pico-iree`](../embassy-pico-iree/) example.

```rust
#[model(
    "../models/lenet5_quantized.tflite",
    backend = "iree",
    arena = "shared"
)]
struct Model;

let mut model = Model::new();
let mut profiler = Profiler::new();
let output = profiler.profile(|| model.run(&input));
```

## What This Example Shows

- Cross-compiling a TFLite model into RP2040-native code
- Running inference from an Embassy async application
- A synchronized shared workspace to avoid per-instance model workspaces
- Static input tensor storage to keep the large input off the task stack
- `defmt` logging over RTT
- Automatic flash/RAM footprint report printed during compilation
- Runtime latency profiling with `Profiler`, using the built-in
  `EmbassyTimer` (chosen automatically as the default timer)

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
- the Python/IREE toolchain from the [project README](../../README.md#1-install-the-host-model-toolchain).

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

The application initializes the RP2040, reports model artifact sizes (input,
output, flash, RAM workspace), fills the static input tensor, runs inference
inside a profiled scope, and reports the latency statistics over RTT.
`LatencyStats` implements `defmt::Format`, so the profiler statistics are
logged directly (`samples=... avg_us=... min_us=... max_us=...`).

It also prints expected-versus-actual diagnostics. The bundled zero-filled
`EXPECTED` array is a placeholder, so use output captured from trusted
reference inference before evaluating model correctness.

## Timer Selection

`Profiler::new()` picks the default timer automatically. In this example the
`embassy` feature is enabled and `std` is disabled, so the profiler uses the
`EmbassyTimer`. To supply a custom clock, construct a profiler with
`Profiler::with_timer(my_timer)`.

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

Validate a new model with the [desktop example](../std-iree/) before compiling it for RP2040.