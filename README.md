# OneLiner

> **Model inference with one-line code.**

OneLiner turns a model file into a callable Rust type with one attribute:

```rust
#[model("models/model.tflite")]
struct MyModel;
```

At build time, OneLiner imports the model, compiles it for the selected Rust target, generates the Rust binding, and links the native model code. At runtime, your application works with ordinary, strongly typed Rust tensors.

```rust
use OneLiner::model;
use OneLiner::runtime::ModelInference;

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

## Why OneLiner?

- **One-line model binding:** Replace conversion scripts, native linking setup, tensor declarations, and dispatch glue with `#[model(...)]`.
- **Typed inputs and outputs:** Tensor element types and shapes come from the model, so mismatches surface during the build instead of on the device.
- **Made for local inference:** The model is compiled into target-native code. Inference does not depend on a cloud service.
- **Embedded-ready:** The runtime supports `no_std` and is demonstrated with Ariel OS and Embassy on RP2040.
- **No Python on the target:** Python and IREE are host-side build tools only.
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

### 2. Add OneLiner

Add the local crate to your application's `Cargo.toml`:

```toml
[dependencies]
OneLiner = { path = "path/to/oneliner/oneliner" }
```

For an embedded `no_std` application:

```toml
[dependencies]
OneLiner = {
    path = "path/to/oneliner/oneliner",
    default-features = false,
    features = ["iree-runtime"]
}
```

### 3. Bind and run a model

Model paths are resolved relative to the application's `Cargo.toml`.

```rust
use OneLiner::model;
use OneLiner::runtime::ModelInference;

#[model("models/model.tflite")]
struct MyModel;

let mut model = MyModel::new();
let mut input = MyModel::create_input_tensor();
input.as_slice_mut().copy_from_slice(&input_data);

let output = model.run(&input);
let values = output.as_slice();
```

OneLiner generates the input and output tensor types directly from the model. The application does not need to repeat their data types or dimensions.

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

OneLiner is currently at version `0.1.0`. The project focuses on making fixed-shape, single-input, single-output inference straightforward across desktop Rust and memory-constrained `no_std` targets.

The examples are intentionally small and explicit. They are designed to help you validate the toolchain, understand the memory trade-offs, and replace the bundled model with your own.
