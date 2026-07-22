# OneLiner Ariel OS IREE Example

This example validates the OneLiner IREE backend from an Ariel OS app with a
quantized LeNet-5 model.

It builds the model at Rust compile time, links the generated static
object, runs the generated dispatch flow in a `no_std` Ariel OS thread, and exits
successfully only when the output bytes match the expected result.

The example places a generated `ModelWorkspace` in a `ConstStaticCell` and
creates a `ModelSession` from its unique mutable reference. Mutable input,
output, and temporary buffers belong to that workspace instead of global
storage, so independent workspace cells can run predictions concurrently.

The active model is `models/lenet5_quantized.tflite`:

- input: 28 x 28 x 4 bytes
- output: 10 x 4 bytes
- expected validation: input bytes filled with `7` -> 40 zero bytes

## Build

From this directory:

```sh
conda run -n ariel_ml laze build -b native
```

To run the native board:

```sh
conda run -n ariel_ml laze build -b native run --bin oneliner-ariel-os-iree
```

This example tracks Ariel OS `v0.4.0`, the latest release tag available when it
was added. Ariel OS `v0.4.0` requires Rust 1.94 or newer. The OneLiner IREE
backend needs `iree-compile` on `PATH`. For `.tflite` models it also needs
`tosa-converter-for-tflite` on `PATH`.

## Hardware Boards

The example is intentionally small, but the IREE backend still compiles a native
object for the active Cargo target. For non-native boards, make sure your IREE
toolchain supports the target triple selected by Ariel OS, or set
`ONELINER_IREE_TARGET_TRIPLE`, `ONELINER_IREE_TARGET_CPU`, and
`ONELINER_IREE_CPU_FEATURES` as needed.
