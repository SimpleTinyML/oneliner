# Standard-host end-to-end tests

This standalone Cargo test package compiles selected models from
`examples/models` through the complete Oneliner/IREE pipeline and runs one
inference on the host.

## Test layers

Model targets are gated by Cargo features so `cargo test --workspace` never
needs the IREE host toolchain:

| Target | Model | Purpose | Feature |
| --- | --- | --- | --- |
| `abs2_owned` | `abs2.mlir` | lightweight MLIR import; wraps a real inference with `Profiler` | `host-e2e` |
| `lenet5_onnx_owned` | `lenet5_quantized.onnx` | ONNX import and numerical regression | `host-e2e` |
| `lenet5_pytorch_owned` | `lenet5_pytorch.pt2` | PyTorch `ExportedProgram`/Turbine import | `host-e2e` |
| `lenet5_tensorflow_owned` | `lenet5_tensorflow/` | TensorFlow SavedModel import | `host-e2e` |
| `lenet5_tflite_shared` | `lenet5_quantized.tflite` | TFLite import and non-trivial shared workspace | `host-e2e` |
| `mcunet_tflite_owned` | `mcunet-10fps_vww.tflite` | large int8 model regression | `host-e2e` |

Install the host tools listed in the repository README, then run from the
repository root:

```sh
# Fast layer: unit, integration and doc tests; no model compilation.
cargo test --workspace

# One test per model import path; both arena modes are represented, including
# the large MCUNet int8 regression.
cargo test -p oneliner-std-e2e-tests --release --features host-e2e
```

Because `default = []`, running `cargo test` inside this package without
`--features host-e2e` intentionally builds no model targets. Other model files
under `examples/models` are inputs for examples and are not part of the
automated matrix.

## Why one executable per model

IREE static libraries export model-level symbols (`*_library_query`), and each
compiled model pulls its own library into the test binary. A test executable
must therefore avoid compiling the same model twice (for example, the `owned`
and `shared` variants of one model cannot share a target), and models whose
normalized input stem collides (the ONNX and TFLite variants of
`lenet5_quantized`) cannot share one either. Different models may coexist, but
keeping one model per target preserves clear failure attribution and makes the
constraint explicit. A target that violates it fails at link time with
`duplicate symbol: <name>_library_query`.
