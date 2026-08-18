# Oneliner + IREE on Desktop

This is the recommended first Oneliner example. It validates the complete model-to-Rust path on a standard host with the smallest amount of setup.

```rust
#[model("../models/mcunet-10fps_vww.tflite", backend = "iree")]
struct Model;
```

## What This Example Shows

- One-line TFLite model binding
- Build-time IREE native compilation
- Automatically generated typed tensors
- A complete inference call with no hand-written dispatch code
- Output validation against a known reference value

## Active Model

The example uses the quantized MCUNet visual wake-word model at `../models/mcunet-10fps_vww.tflite`:

- input: `Tensor<i8, 1, 64, 64, 3>`
- output: `Tensor<i8, 1, 1, 1, 2>`
- input data: every element is filled with `7`
- expected output: `[4, -5]`
- memory mode: `owned`, the Oneliner default

## Prerequisites

Install the Python/IREE model toolchain described in [docs/installation.md](../../docs/installation.md), then keep that environment active.

You also need a current Rust toolchain with Cargo.

## Run

From this directory:

```sh
cargo run --release
```

The first build can take longer because Oneliner imports the TFLite model and compiles native model code before Rust links the application.

## Expected Behavior

The application logs:

- the generated input and output artifact sizes;
- `Model IREE validation passed` when the output is `[4, -5]`;
- expected and actual values if validation fails.

`RUST_LOG=info` is already configured in `.cargo/config.toml`.

## Try Another Model

Change the model path in `src/main.rs`:

```rust
#[model("../models/your-model.tflite")]
struct Model;
```

Oneliner also accepts ONNX, PyTorch `ExportedProgram` (`.pt2`), TensorFlow SavedModel v2, and IREE-compatible MLIR. When changing models, update the input preparation and reference output to match the new model.

Once a model works here, move the same binding to the [Ariel OS](../ariel-os-minimal/) or [Embassy Pico](../embassy-pico-minimal/) example.
