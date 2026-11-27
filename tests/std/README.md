# Standard-host end-to-end tests

This standalone Cargo test package compiles selected models from
`examples/models` through the complete Oneliner/IREE pipeline and exercises
inference on the host through the generated model API.

## Test layers

Model targets are gated by Cargo features so `cargo test --workspace` never
needs the IREE host toolchain:

| Target | Model | Purpose | Feature |
| --- | --- | --- | --- |
| `abs2_mlir` | `abs2.mlir` | lightweight MLIR import; wraps a real inference with `Profiler` | `host-e2e` |
| `lenet5_onnx` | `lenet5_quantized.onnx` | ONNX import and numerical regression | `host-e2e` |
| `lenet5_pytorch` | `lenet5_pytorch.pt2` | PyTorch `ExportedProgram`/Turbine import | `host-e2e` |
| `lenet5_tensorflow` | `lenet5_tensorflow/` | TensorFlow SavedModel import | `host-e2e` |
| `mcunet_tflite` | `mcunet-10fps_vww.tflite` | large int8 model regression | `host-e2e` |

Install the host tools listed in the repository README, then run from the
repository root:

```sh
# Fast layer: unit, integration and doc tests; no model compilation.
cargo test --workspace

# One test per model import path (five targets); the MCUNet target keeps the
# large int8 regression.
cargo test -p oneliner-std-e2e-tests --release --features host-e2e
```

Because `default = []`, running `cargo test` inside this package without
`--features host-e2e` intentionally builds no model targets. Other model files
under `examples/models` are inputs for examples and are not part of the
automated matrix.

## Why one executable per model

IREE static libraries export model-level symbols (`*_library_query`), and each
compiled model pulls its own library into the test binary. A test executable
must therefore avoid compiling the same model twice — two `#[model]`
expansions of one model file cannot share a target. Keeping one model per
target preserves clear failure attribution and makes the constraint explicit.
A target that violates it fails at link time with `duplicate symbol:
<name>_library_query`.
