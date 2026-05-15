# OneLiner Ariel OS IREE Example

This example validates the OneLiner IREE backend from an Ariel OS app.

It builds a tiny IREE model at Rust compile time, links the generated static
object, runs the generated dispatch flow in a `no_std` Ariel OS task, and exits
successfully only when the output bytes match the expected result.

The model is `models/abs2.mlir`:

- input: `tensor<2xf32>` / 8 bytes
- operation: `math.absf`
- expected validation: `[-1.0, 3.0] -> [1.0, 3.0]`

## Build

From this directory:

```sh
conda run -n ariel_ml laze build -b native
```

To run the native board:

```sh
conda run -n ariel_ml laze build -b native run
```

This example tracks Ariel OS `v0.4.0`, the latest release tag available when it
was added. Ariel OS `v0.4.0` requires Rust 1.94 or newer. The OneLiner IREE
backend needs `iree-compile` on `PATH`. For `.tflite` models it also needs
`iree-import-tflite`, but this example uses MLIR directly.

## Hardware Boards

The example is intentionally small, but the IREE backend still compiles a native
object for the active Cargo target. For non-native boards, make sure your IREE
toolchain supports the target triple selected by Ariel OS, or set
`ONELINER_IREE_TARGET_TRIPLE`, `ONELINER_IREE_TARGET_CPU`, and
`ONELINER_IREE_CPU_FEATURES` as needed.
