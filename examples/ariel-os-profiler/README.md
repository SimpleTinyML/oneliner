# Oneliner + IREE + Profiler on Ariel OS

This example runs a Oneliner model inside a `no_std` Ariel OS thread and measures its inference latency with the `oneliner-profiler` crate. It is the profiler counterpart of the [`ariel-os-minimal`](../ariel-os-minimal/) example.

```rust
#[model("../models/lenet5_quantized.tflite", backend = "iree")]
struct Model;

let mut model = Model::new();
let mut profiler = Profiler::new();
let output = profiler.profile(|| model.run(&input));
info!("Profiled inference stats: {}", profiler.stats());
```

## What This Example Shows

- Build-time TFLite import and IREE compilation
- A fully typed model API in `no_std`
- Inference from an Ariel OS thread
- Automatic flash footprint report (parameters, machine code, read-only data, and total) plus RAM workspace, printed during compilation
- Runtime latency profiling with `Profiler`, using the built-in `ArielOsTimer` (chosen automatically as the default timer)
- Model artifact size logging
- Process exit status based on output comparison

## Active Model

The example currently uses `../models/lenet5_quantized.tflite`:

- input: `Tensor<f32, 1, 28, 28, 1>`
- output: `Tensor<f32, 1, 1, 1, 10>`
- input data: every element is filled with `7.0`
- memory mode: `owned`, the Oneliner default

`EXPECTED` is the reference output captured from trusted validation data. Replace it if you change the model or input preparation.

## Prerequisites

Install the Python/IREE model toolchain described in [docs/installation.md](../../docs/installation.md), and keep that environment active while building.

You also need:

- Rust 1.95 or newer for the configured Ariel OS release
- [Laze](https://github.com/kas-gui/laze)
- network access on the first build so Laze can fetch Ariel OS

This example tracks Ariel OS `v0.5.0` through `laze-project.yml`.

## Build for the Native Board

From this directory:

```sh
laze build -b native
```

Build and run the native application:

```sh
laze build -b native run --bin oneliner-ariel-os-profiler
```

If your toolchain is stored in a Conda environment, prefix the commands with `conda run -n <environment>`.

## Expected Behavior

The application:

1. reports the active Ariel OS board;
2. reports the generated artifact sizes (input, output, parameters, machine code, read-only data, total flash, RAM workspace);
3. fills the typed input tensor with `7.0`;
4. runs inference inside a profiled scope and reports the latency statistics;
5. compares the result with `EXPECTED`;
6. exits with success or failure.

## Timer Selection

`Profiler::new()` picks the default timer automatically. In this example the `ariel-os` feature is enabled and `std` is disabled, so the profiler uses the `ArielOsTimer`. To supply a custom clock, construct a profiler with `Profiler::with_timer(my_timer)`.

## Building for Hardware

Select a board supported by the configured Ariel OS release instead of `native`. Oneliner derives the IREE target from the Rust target chosen by Ariel OS.

## Switching Models

Change the path in `#[model(...)]`, then update:

- the input preparation;
- the `EXPECTED` element type and values;
- any stack-size or memory decisions required by the new model.

For the fastest first check of a new model, use the [desktop example](../std-minimal/) before cross-compiling it for Ariel OS.