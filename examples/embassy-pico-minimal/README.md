# Oneliner + IREE on Raspberry Pi Pico

This example runs a Oneliner model on an RP2040 with Embassy. It demonstrates bare-metal, `no_std` model inference with the model and its workspace in static storage.

```rust
use oneliner::model;
use oneliner::runtime::ModelInference;
use static_cell::ConstStaticCell;

#[model("../models/lenet5_quantized.tflite")]
struct Model;

static INPUT_CELL: ConstStaticCell<<Model as ModelInference>::InputTensor> =
    ConstStaticCell::new(<Model as ModelInference>::InputTensor::new(0.0));
static MODEL: ConstStaticCell<Model> = ConstStaticCell::new(Model::new());

let mut model = MODEL.take();
let mut input = INPUT_CELL.take();
let output = model.run(&input);
```

## Sample Model

The example currently uses `../models/lenet5_quantized.tflite`:

- input: `Tensor<f32, 1, 28, 28, 1>`
- output: `Tensor<f32, 1, 1, 1, 10>`
- input data: every element is filled with `7.0`

## Hardware and Tools

You need:

- a Raspberry Pi Pico or another compatible RP2040 board;
- a debug probe supported by `probe-rs`;
- `probe-rs` installed on the host;
- the stable Rust toolchain and `thumbv6m-none-eabi` target;
- the Python/IREE toolchain from [docs/installation.md](../../docs/installation.md).

The included Rust toolchain and Cargo configuration select the RP2040 target automatically. Keep the Python virtual environment active during the build.

## Build

From this directory:

```sh
cargo build --release
```

## Flash and Run

Connect the board and debug probe, then run:

```sh
cargo run --release
```

The configured Cargo runner uses:

```text
probe-rs run --chip RP2040
```


## Why Static Model and Input Storage?

The LeNet5 input tensor and the model workspace are significant on an RP2040 with 264 KiB of RAM. This example keeps both out of the async task's stack:

- `ConstStaticCell<Model>` places the whole model, including its inline workspace, in static storage; `Model::new()` is a const function, so it can build the initializer.
- `ConstStaticCell` also places the input tensor in static storage.

Both are zero-initialized, so they land in `.bss`: they consume RAM but no flash. This example is single-threaded. If another task or an interrupt also runs inference, give each its own `Model` instance, or share one instance only behind a lock or critical section (see the [memory model](../../docs/memory_model.md)).
