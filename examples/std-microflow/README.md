# OneLiner + MicroFlow on Desktop

This example runs the quantized DS-CNN-S speech-command model with OneLiner's
pure-Rust MicroFlow backend on a standard host. It does not require Python,
IREE, a native compiler toolchain, or native model libraries.

```rust
#[model("../models/ds_cnn_s_quantized.tflite", backend = "microflow")]
struct Model;
```

## What This Example Shows

- compile-time binding of a quantized TFLite model through MicroFlow;
- a native `Buffer2D<f32, 1, 490>` input and `Buffer2D<f32, 1, 12>` output;
- moving the input buffer directly into `ModelInference::run`;
- selecting the highest-scoring class from the 12 model outputs.

`create_input_tensor` initializes all 490 input features to zero. Replace its
contents with the preprocessed audio features when integrating real input.

## Run

From this directory:

```sh
cargo run --release
```

Expected output includes:

```text
MicroFlow inference completed: class=..., score=...
```

## Model Constraints

The MicroFlow backend accepts statically shaped, quantized TFLite models with
one input and one output. Model tensors must use `INT8` or `UINT8`, and their
ranks must be 1, 2, or 4. Operator support is determined by MicroFlow 0.1.3.
