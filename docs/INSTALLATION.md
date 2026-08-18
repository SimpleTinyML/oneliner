# Installing the host model toolchain

Oneliner compiles models on your host machine during `cargo build`, so you need
a small set of Python tools installed before the `#[model(...)]` attribute can
work.

## Prerequisites

- Python 3.10 or newer
- A working Rust toolchain (`rustc` 1.95 or newer)
- `cargo` on your `PATH`

## Install the compiler packages

A virtual environment keeps the compiler tools isolated from the rest of your
system. Activate it before every build that uses `#[model(...)]`:

```sh
python -m venv .venv
source .venv/bin/activate

pip install "iree-base-compiler[onnx]" tosa-converter-for-tflite
```

This covers TFLite, ONNX, and MLIR models.

### TensorFlow SavedModels

To compile TensorFlow SavedModels, install TensorFlow and the matching IREE
TensorFlow tools in the same environment:

```sh
pip install tensorflow iree-tools-tf
```

### PyTorch models

To compile PyTorch models, install the CPU build of PyTorch and IREE Turbine in
the same environment. PyTorch, Turbine, and IREE must be mutually compatible, so
pin a working combination together for reproducible builds:

```sh
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install iree-turbine
```

## Verify the installation

```sh
iree-compile --version
tosa-converter-for-tflite --version
iree-import-onnx --help
iree-import-tf --help
python -c "import torch, iree.turbine.aot"
python -c "import tensorflow"
```

## Package reference

| Package | Purpose |
| --- | --- |
| `iree-base-compiler` | The IREE compiler used for every model |
| `iree-base-compiler[onnx]` | ONNX import support |
| `tosa-converter-for-tflite` | TFLite-to-TOSA import support |
| `torch` | Exports and loads PyTorch `ExportedProgram` models |
| `iree-turbine` | Imports PyTorch programs into IREE-compatible MLIR |
| `tensorflow` | Loads and inspects SavedModel signatures |
| `iree-tools-tf` | Imports TensorFlow SavedModels into IREE-compatible MLIR |

If you only use MLIR input, `iree-base-compiler` is sufficient.