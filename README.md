# OneLiner

`OneLiner` provides a Rust attribute macro named `model`:

```rust
use OneLiner::model;

#[model("path/to/model.tflite")]
struct MyModel;

fn main() {
    let prediction = MyModel::predict(input_data);
    println!("{} output bytes", prediction.as_bytes().len());
}
```

The macro is backend-oriented. IREE is the built-in default backend. Microflow
is the pure Rust backend for runtimes that do not need compile-time compiler or
IREE dispatch generation.

## Rust Usage

Add the runtime crate to a Rust package:

```toml
[dependencies]
OneLiner = { path = "path/to/this/workspace/oneliner" }
```

For `no_std` targets, disable default features:

```toml
[dependencies]
OneLiner = { path = "path/to/this/workspace/oneliner", default-features = false }
```

If you want owned output bytes in `no_std` with an allocator, enable `alloc`:

```toml
[dependencies]
OneLiner = { path = "path/to/this/workspace/oneliner", default-features = false, features = ["alloc"] }
```

For an IREE-generated static flow on `no_std`, also enable `iree-runtime`:

```toml
[dependencies]
OneLiner = { path = "path/to/this/workspace/oneliner", default-features = false, features = ["iree-runtime"] }
```

Use the default IREE backend:

```rust
use OneLiner::model;

#[model("path/to/model.tflite")]
struct MyModel;
```

Select a backend explicitly:

```rust
#[model("path/to/model.microflow", backend = "microflow")]
struct MyModel;
```

For fallible code, use `try_predict`:

```rust
let prediction = MyModel::try_predict(input_data)?;
```

Generated artifact paths are available for debugging:

```rust
use OneLiner::runtime::ModelSource;

println!("{:#?}", MyModel::ARTIFACTS);
```

## Built-In IREE Backend

The IREE backend compiles the model at Rust compile time. For `.tflite` inputs
it first converts the FlatBuffer to TOSA MLIR with `tosa-converter-for-tflite`, then asks
IREE to emit the model object file, finds the stage-10 executable-targets IR,
runs `iree_stream_flow_to_rust_using_re.py`, and includes the generated Rust
flow.

Required tools:

- `tosa-converter-for-tflite` on `PATH`, or `ONELINER_TOSA_CONVERTER_FOR_TFLITE=/path/to/tosa-converter-for-tflite`
- `iree-compile` on `PATH`, or `ONELINER_IREE_COMPILE=/path/to/iree-compile`
- `python` with `iree-compiler` installed, or `ONELINER_PYTHON=/path/to/python`

Default IREE flags:

```powershell
tosa-converter-for-tflite model.tflite --text -o <OUT_DIR>/model.tosa.mlir

iree-compile <OUT_DIR>/model.tosa.mlir `
  --iree-hal-target-device=local `
  --iree-hal-local-target-device-backends=llvm-cpu `
  --iree-llvmcpu-target-triple=<TARGET> `
  --iree-llvmcpu-target-cpu=<target-cpu> `
  --iree-llvmcpu-target-cpu-features=<target-features> `
  --iree-stream-partitioning-favor=min-peak-memory `
  --iree-llvmcpu-link-embedded=false `
  --iree-llvmcpu-link-static `
  --iree-llvmcpu-static-library-output-path=<OUT_DIR>/model.o `
  --dump-compilation-phases-to=<OUT_DIR>/iree-ir-dumps `
  -o <OUT_DIR>/model.vmfb
```

IREE-specific environment overrides:

- `ONELINER_TOSA_CONVERTER_FOR_TFLITE`
- `ONELINER_IREE_TARGET_TRIPLE`
- `ONELINER_IREE_TARGET_CPU`
- `ONELINER_IREE_CPU_FEATURES`
- `ONELINER_IREE_COMPILE_FLAGS`
- `ONELINER_IREE_STREAM_FLOW_TO_RUST`

The older `IREE_*` / `IREE_MODEL_*` variable names are still accepted as
fallbacks where they existed before.

## Microflow Backend

Use `backend = "microflow"` and implement the Microflow runtime trait for the
generated model:

```rust
use OneLiner::model;

#[model("path/to/model.microflow", backend = "microflow")]
struct MyModel;

impl OneLiner::runtime::MicroflowModel for MyModel {
    type Error = MyError;
    type Output = MyOutput;

    fn try_predict_microflow(input: &[u8]) -> Result<Self::Output, Self::Error> {
        run_microflow(include_bytes!("path/to/model.microflow"), input)
    }
}
```

The macro still generates the familiar API:

```rust
let output = MyModel::predict(input);
let output = MyModel::try_predict(input)?;
```

Microflow does not use a compiler, generated Rust flow, IREE dispatch helpers,
or native linking.

## Linking Note

The built-in IREE backend links its generated native object with a
`#[link(..., modifiers = "+verbatim")]` extern block. Microflow does not use
this path. `cargo:rustc-link-arg=...` only works when printed by a
`build.rs`; proc-macro stdout is not interpreted as Cargo build-script
directives.

## Runtime And no_std

The `oneliner` runtime crate supports:

- `std` default feature: implements `std::error::Error` and enables `alloc`.
- `alloc` feature: keeps `Prediction::from_bytes` and `Prediction::into_bytes`.
- `iree-runtime` feature: enables IREE local executable ABI helpers such as
  `dispatch`, `try_dispatch`, and HAL dispatch-state structs.
- no default features: pure `no_std` runtime with borrowed output slices.

The public runtime surface is intentionally small:

```rust
pub trait Predict<Input: ?Sized = [u8]> {
    type Error;
    type Output;

    fn try_predict(input: &Input) -> Result<Self::Output, Self::Error>;
}

pub trait ModelSource {
    const MODEL_PATH: &'static str;
    const ARTIFACTS: ModelArtifacts;
}

pub trait MicroflowModel: ModelSource {
    type Error;
    type Output;

    fn try_predict_microflow(input: &[u8]) -> Result<Self::Output, Self::Error>;
}

```

Microflow can return typed outputs through `MicroflowModel`. OneLiner also
implements `Predict<[u8]>` for Microflow-backed models.

In pure `no_std`, `Prediction` is a borrowed view over the generated static
output buffer:

```rust
let prediction = MyModel::predict(input);
let bytes: &[u8] = prediction.as_bytes();
```

The proc-macro, and any built-in IREE compilation it performs, still runs on
the host during Cargo builds and uses `std`; only the target runtime is
`no_std`.

## Source Layout

- `oneliner/src/runtime/interface.rs`: public runtime traits and shared metadata.
- `oneliner/src/runtime/microflow.rs`: Microflow runtime trait.
- `oneliner/src/runtime/prediction.rs`: byte prediction value type.
- `oneliner/src/runtime/buffer.rs`: generic static-buffer helpers.
- `oneliner/src/runtime/iree.rs`: optional IREE local executable runtime helpers.
- `oneliner-macro/src/backend/microflow.rs`: Microflow backend expansion.
- `oneliner-macro/src/backend/iree.rs`: IREE expansion entry point.
- `oneliner-macro/src/backend/iree/`: artifact, metadata, toolchain, discovery, and codegen modules.

## Flow Converter CLI

The existing IREE flow converter is still available directly:

```powershell
python iree_stream_flow_to_rust_using_re.py .\stream.mlir -o .\stream_flow.rs
```

It parses IREE Stream/Flow/HAL MLIR and renders a Rust call-flow skeleton.
