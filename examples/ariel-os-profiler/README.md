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

- Automatic flash footprint report (parameters, machine code, read-only data, and total) plus RAM workspace, printed during compilation
- Runtime latency profiling with `Profiler`, using the built-in `ArielOsTimer` (chosen automatically as the default timer)
- Model size logging

## Sample Model

The example currently uses `../models/lenet5_quantized.tflite`:

- input: `Tensor<f32, 1, 28, 28, 1>`
- output: `Tensor<f32, 1, 1, 1, 10>`
- input data: every element is filled with `7.0`

`EXPECTED` is the reference output captured from trusted validation data. Replace it if you change the model or input preparation.

## Prerequisites

Install the Python/IREE model toolchain described in [docs/installation.md](../../docs/installation.md), and keep that environment active while building.

This example tracks Ariel OS `v0.5.0` through `laze-project.yml`.

## Build for the Native Board

From this directory:

```sh
laze build -b native
```

Build and run the native application:

```sh
laze build -b native run
```


## Timer Selection

`Profiler::new()` picks the default timer automatically. In this example the `ariel-os` feature is enabled and `std` is disabled, so the profiler uses the `ArielOsTimer`. To supply a custom clock, construct a profiler with `Profiler::with_timer(my_timer)`.

## Building for Hardware

Select a board supported by the configured Ariel OS release instead of `native`. Oneliner derives the IREE target from the Rust target chosen by Ariel OS.
