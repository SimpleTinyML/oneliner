# Oneliner

[![Current Crates.io Version](https://img.shields.io/crates/v/oneliner.svg)](https://crates.io/crates/oneliner)
[![Minimum Supported Rust Version](https://img.shields.io/crates/msrv/oneliner)](https://crates.io/crates/oneliner)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](#license)

> **TinyML model inference with one-line code. Support `no_std` embedded targets.**

Oneliner turns a model file into a callable Rust type with one attribute:

```rust
#[model("models/model.tflite")]
struct MyModel;
```

At build time, Oneliner imports the model, compiles it for the selected Rust target, generates the Rust binding, and links the native model code. At runtime, your application works with ordinary, strongly typed Rust tensors.

```rust
use oneliner::model;
use oneliner::runtime::ModelInference;

#[model("models/model.tflite")]
struct MyModel;

fn main() {
    let mut model = MyModel::new();
    let mut input = MyModel::create_input_tensor();
    input.fill(1);

    let output = model.run(&input);
    println!("{:?}", output.as_slice());
}
```

## Why Oneliner?

- **One-line model binding:** Replace conversion scripts, native linking setup, tensor declarations, and dispatch glue with `#[model(...)]`.
- **Typed inputs and outputs:** Tensor element types and shapes come from the model, so mismatches surface during the build instead of on the device.
- **Made for on-device inference:** The model is compiled into target-native code. Inference does not depend on a cloud service.
- **Embedded-ready:** The runtime supports `no_std` and is demonstrated with Ariel OS and Embassy on RP2040.
- **Memory-aware by design:** Choose independent per-instance workspaces or one synchronized shared workspace.
- **Profiling built in:** Measure inference latency and build-time flash/RAM footprint with the optional `profiler` feature.

## Quick Start

1. [Install the host model toolchain](docs/INSTALLATION.md).
2. Add the crate to your `Cargo.toml`:

   ```toml
   [dependencies]
   oneliner = "0.2"
   ```

3. Bind and run a model:

   ```rust
   use oneliner::model;
   use oneliner::runtime::ModelInference;

   #[model("models/model.tflite")]
   struct MyModel;

   let mut model = MyModel::new();
   let mut input = MyModel::create_input_tensor();
   input.as_slice_mut().copy_from_slice(&input_data);

   let output = model.run(&input);
   let values = output.as_slice();
   ```

Oneliner generates the input and output tensor types directly from the model. The application does not need to repeat their data types or dimensions.

## Supported Models

The built-in IREE backend accepts:

- TFLite
- ONNX
- PyTorch `ExportedProgram` (`.pt2`)
- TensorFlow SavedModel v2 directories
- MLIR accepted by IREE

See [Model formats](docs/MODEL_FORMATS.md) for per-format guides and the
`owned`/`shared` [memory modes](docs/MODEL_FORMATS.md#memory-modes).

## Profiling

Enable the optional `profiler` feature to measure inference latency:

```toml
[dependencies]
oneliner = { version = "0.2", features = ["profiler"] }
```

Wrap any inference call in a profiler scope:

```rust
use oneliner::profiler::Profiler;

let mut profiler = Profiler::new();
let output = profiler.profile(|| model.run(&input));
println!("{}", profiler.stats());
```

On `no_std` targets, depend on `oneliner-profiler` directly to pick the timer
backend (Ariel OS or Embassy). In addition, every model build prints an
automatic flash/RAM footprint report (parameters, machine code, read-only data,
workspace) — see the profiler examples for live output.

## Examples

Each example is an independent Cargo project. Run its commands from the example directory with the Python environment activated.

| Example | What it demonstrates | Active model |
| --- | --- | --- |
| [Desktop IREE](examples/std-iree/) | The shortest end-to-end validation path on a standard host | Quantized MCUNet visual wake word |
| [Ariel OS + IREE](examples/ariel-os-iree/) | `no_std`, Ariel OS threads, native-board validation, and inference timing | Quantized LeNet5 |
| [Embassy + IREE on Pico](examples/embassy-pico-iree/) | Bare-metal RP2040, shared model workspace, static input storage, and `defmt` logging | Quantized LeNet5 |
| [Ariel OS + Profiler](examples/ariel-os-profiler/) | `no_std` latency profiling with `Profiler` and the automatic flash/RAM footprint report | Quantized LeNet5 |
| [Embassy + Profiler on Pico](examples/embassy-pico-profiler/) | Bare-metal RP2040 latency profiling and footprint report | Quantized LeNet5 |

Start with the [desktop example](examples/std-iree/) to confirm the model toolchain, then move to the operating system or board example that matches your target.

## Project Status

Oneliner is currently at version `0.2.0`. The project focuses on making fixed-shape, single-input, single-output inference straightforward across desktop Rust and memory-constrained `no_std` targets.

The examples are intentionally small and explicit. They are designed to help you validate the toolchain, understand the memory trade-offs, and replace the bundled model with your own.

## Testing

With the host model toolchain active, run the std end-to-end test suite from
the repository root:

```sh
cargo test
```

This runs end-to-end inference for every model in `examples/models`, using both
the `owned` and `shared` arena modes.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you shall be dual-licensed as above, without any additional terms or conditions.

**Other languages:** [简体中文](README-zh-CN.md)