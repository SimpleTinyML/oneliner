# Oneliner + IREE on Ariel OS

This example runs a Oneliner model inside a `no_std` Ariel OS thread. It demonstrates that the same one-line model binding used on desktop can produce target-native inference code for an operating-system-based embedded application.

```rust
#[model("../models/lenet5_quantized.tflite", backend = "iree")]
struct Model;
```

## What This Example Shows

- Build-time TFLite import and IREE compilation
- A fully typed model API in `no_std`
- Inference from an Ariel OS thread
- Model artifact size logging
- On-target inference timing
- Process exit status based on output comparison

## Active Model

The example currently uses `../models/lenet5_quantized.tflite`:

- input: `Tensor<f32, 1, 28, 28, 1>`
- output: `Tensor<f32, 1, 1, 1, 10>`
- input data: every element is filled with `7.0`
- memory mode: `owned`, the Oneliner default

`EXPECTED` is currently a ten-element zero-filled placeholder. Replace it with reference output from your own validation data before using the comparison as a model-correctness test.

## Prerequisites

Install the Python/IREE model toolchain described in the [project README](../../README.md#1-install-the-host-model-toolchain), and keep that environment active while building.

You also need:

- Rust 1.94 or newer for the configured Ariel OS release
- [Laze](https://github.com/kas-gui/laze)
- network access on the first build so Laze can fetch Ariel OS

This example tracks Ariel OS `v0.4.0` through `laze-project.yml`.

## Build for the Native Board

From this directory:

```sh
laze build -b native
```

Build and run the native application:

```sh
laze build -b native run --bin oneliner-ariel-os-iree
```

If your toolchain is stored in a Conda environment, prefix the commands with `conda run -n <environment>`.

## Expected Behavior

The application:

1. reports the active Ariel OS board;
2. reports the generated input and output artifact sizes;
3. fills the typed input tensor with `7.0`;
4. runs inference and reports elapsed microseconds;
5. compares the result with `EXPECTED`;
6. exits with success or failure.

Because the bundled `EXPECTED` value is a placeholder, a failure exit does not by itself indicate that model compilation or dispatch failed.

## Using Shared Memory

The active example uses the default per-instance workspace. To let every `Model` instance share one synchronized static workspace, change the binding to:

```rust
#[model(
    "../models/lenet5_quantized.tflite",
    backend = "iree",
    arena = "shared"
)]
struct Model;
```

The `ariel-os` feature is already enabled for Oneliner in this example.

## Building for Hardware

Select a board supported by the configured Ariel OS release instead of `native`. Oneliner derives the IREE target from the Rust target chosen by Ariel OS.

## Switching Models

Change the path in `#[model(...)]`, then update:

- the input preparation;
- the `EXPECTED` element type and values;
- any stack-size or memory decisions required by the new model.

For the fastest first check of a new model, use the [desktop example](../std-iree/) before cross-compiling it for Ariel OS.
