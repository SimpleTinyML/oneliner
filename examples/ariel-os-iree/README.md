# OneLiner Ariel OS IREE Example

This example validates the OneLiner IREE backend from an Ariel OS app with a
quantized MCUNet visual wake-word model.

It builds the model at Rust compile time, links the generated static
object, runs the generated dispatch flow in a `no_std` Ariel OS thread, and exits
successfully only when the typed output tensor matches the expected result.

The model uses `arena = "shared"`, so the macro initializes one private static
arena and synchronizes access from every `Model` instance. The input and output
bindings point directly at tensor values created locally in `main`, while only
temporary buffers belong to the shared arena.

The active model is `models/mcunet-10fps_vww.tflite`:

- input: `Tensor<i8, 1, 64, 64, 3>`
- output: `Tensor<i8, 1, 1, 1, 2>`
- expected validation: input elements filled with `7` -> `[4, -5]`

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
