# Oneliner

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

## Quick Start

### 1. Install the host model toolchain

Python 3.10 or newer is required. A virtual environment keeps the compiler tools isolated from the rest of your system:

```sh
pip install "iree-base-compiler[onnx]" tosa-converter-for-tflite
```

Verify the installation:

```sh
iree-compile --version
tosa-converter-for-tflite --version
iree-import-onnx --help
```

The packages provide:

- `iree-base-compiler`: the IREE compiler used for every model
- `iree-base-compiler[onnx]`: ONNX import support
- `tosa-converter-for-tflite`: TFLite-to-TOSA import support

If you only use MLIR input, `iree-base-compiler` is sufficient.

### 2. Add Oneliner

Add the crate to your application's `Cargo.toml`:

```toml
[dependencies]
oneliner = "0.1.0"
```
### 3. Bind and run a model

Model paths are resolved relative to the application's `Cargo.toml`.

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

## Examples

Each example is an independent Cargo project. Run its commands from the example directory with the Python environment activated.

| Example | What it demonstrates | Active model |
| --- | --- | --- |
| [Desktop IREE](examples/std-iree/) | The shortest end-to-end validation path on a standard host | Quantized MCUNet visual wake word |
| [Ariel OS + IREE](examples/ariel-os-iree/) | `no_std`, Ariel OS threads, native-board validation, and inference timing | Quantized LeNet5 |
| [Embassy + IREE on Pico](examples/embassy-pico-iree/) | Bare-metal RP2040, shared model workspace, static input storage, and `defmt` logging | Quantized LeNet5 |

Start with the [desktop example](examples/std-iree/) to confirm the model toolchain, then move to the operating system or board example that matches your target.

## Supported Models

The built-in IREE backend currently accepts:

- TFLite
- ONNX
- MLIR accepted by IREE

The generated `ModelInference` API currently targets fixed-shape models with:

- exactly one input tensor
- exactly one output tensor
- up to four dimensions

Integer and floating-point tensor element types are inferred automatically.

## Memory Modes

The default `owned` mode gives each model instance an independent workspace:

```rust
#[model("models/model.tflite")]
struct MyModel;
```

This is the natural choice when model instances may run concurrently.

The `shared` mode keeps one synchronized static workspace for all instances of a model type:

```rust
#[model("models/model.tflite", arena = "shared")]
struct MyModel;
```

Use it when reducing duplicate RAM use matters more than concurrent inference. The Pico example demonstrates this configuration.


## Project Status

Oneliner is currently at version `0.1.0`. The project focuses on making fixed-shape, single-input, single-output inference straightforward across desktop Rust and memory-constrained `no_std` targets.

The examples are intentionally small and explicit. They are designed to help you validate the toolchain, understand the memory trade-offs, and replace the bundled model with your own.

## Testing

With the host model toolchain active, run the std end-to-end test suite from
the repository root:

```sh
cargo test
```

This runs end-to-end inference for every model in `examples/models`, using both
the `owned` and `shared` arena modes.
