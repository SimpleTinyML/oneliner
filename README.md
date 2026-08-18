# Oneliner

**TinyML model inference with one-line code. Focus on `no_std` embedded targets.**

[![Current Crates.io Version](https://img.shields.io/crates/v/oneliner.svg)](https://crates.io/crates/oneliner)
[![Minimum Supported Rust Version](https://img.shields.io/crates/msrv/oneliner)](https://crates.io/crates/oneliner)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](#license)



## Why Oneliner?

- **One-line model deployment:** Replace conversion scripts, native linking setup, tensor declarations, and dispatch glue with `#[model(...)]`.
- **Embedded-ready:** The runtime supports `no_std` and is demonstrated with Ariel OS and Embassy on ARM Cortex-M targets.

Oneliner turns a model file into a callable Rust type with oneline code:

```rust
#[model("models/model.tflite")]
struct MyModel;
```


## Quick Start

1. [Install the host model compilation toolchain](docs/Installation.md).
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

Oneliner accepts:

- TFLite
- ONNX
- PyTorch `ExportedProgram` (`.pt2`)
- TensorFlow SavedModel v2 directories
- MLIR accepted by IREE

See [Model formats](docs/Model_formats.md) for per-format guides and the `owned`/`shared` [memory modes](docs/Model_formats.md#memory-modes).

## Profiling

Use the optional crate `oneliner-profiler` to measure inference latency:

```toml
[dependencies]
oneliner = { version = "0.2" }
oneliner-profiler = "0.1"
```

Wrap any inference call in a profiler scope:

```rust
use oneliner::profiler::Profiler;

let mut profiler = Profiler::new();
let output = profiler.profile(|| model.run(&input));
println!("{}", profiler.stats());
```

On `no_std` targets, depend on `oneliner-profiler` directly to pick the timer backend (Ariel OS or Embassy). In addition, every model build prints an automatic flash/RAM footprint report (parameters, machine code, read-only data, workspace) — see the profiler examples for live output.

## Examples

Each example is an independent Cargo project. Run its commands from the example directory with the Python environment activated.

| Example | What it demonstrates | Active model |
| --- | --- | --- |
| [Desktop Std](examples/std-minimal/) | The shortest end-to-end validation path on a standard host | Quantized MCUNet visual wake word |
| [Ariel OS](examples/ariel-os-minimal/) | `no_std`, Ariel OS threads, native-board validation, and inference timing | Quantized LeNet5 and MCUNet|
| [Embassy on Rasperry Pi Pico](examples/embassy-pico-minimal/) | Bare-metal RP2040, shared model workspace, static input storage, and `defmt` logging | Quantized LeNet5 |
| [Ariel OS + Profiler](examples/ariel-os-profiler/) | `no_std` latency profiling with `Profiler` and the automatic flash/RAM footprint report | Quantized LeNet5 and MCUNet |
| [Embassy on Rasperry Pi Pico + Profiler](examples/embassy-pico-profiler/) | Bare-metal RP2040 latency profiling and footprint report | Quantized LeNet5 |

Start with the [desktop example](examples/std-minimal/) to confirm the model toolchain, then move to the operating system or board example that matches your target.

## Project Status

Oneliner is currently at version `0.2.0`. The project focuses on making fixed-shape, single-input, single-output inference straightforward across desktop Rust and memory-constrained `no_std` targets.

The examples are intentionally small and explicit. They are designed to help you validate the toolchain, understand the memory trade-offs, and replace the bundled model with your own.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you shall be dual-licensed as above, without any additional terms or conditions.

**Other languages:** [简体中文](README-zh-CN.md)