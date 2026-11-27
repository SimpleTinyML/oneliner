# Oneliner on Ariel OS

This example runs a Oneliner model inside a `no_std` Ariel OS thread.

```rust
#[model("../models/lenet5_quantized.tflite", backend = "iree")]
struct Model;
```

## Sample Model

The example currently uses `../models/lenet5_quantized.tflite`:

- input: `Tensor<f32, 1, 28, 28, 1>`
- output: `Tensor<f32, 1, 1, 1, 10>`
- input data: every element is filled with `7.0`

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

If your toolchain is stored in a Conda environment, prefix the commands with `conda run -n <environment>`.


## Building for Hardware

Select a board supported by the configured Ariel OS release instead of `native`. Oneliner derives the IREE target from the Rust target chosen by Ariel OS.
