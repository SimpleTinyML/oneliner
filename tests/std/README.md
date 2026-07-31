# Standard-host end-to-end tests

This standalone Cargo test package compiles every model in
`examples/models` through the complete OneLiner/IREE pipeline and executes an
inference on the host. Every model is tested once with an instance-owned arena
and once with the synchronized shared arena.

Install the host tools listed in the repository README, then run from the
repository root:

```sh
cargo test --manifest-path tests/std/Cargo.toml --release
```

Each model/arena combination is a separate integration-test executable. IREE
static libraries expose model-level symbols, so this isolation prevents symbol
collisions while still allowing one command to run the full matrix.

`model_coverage.rs` scans the model directory. Adding a new `.mlir`, `.onnx`,
or `.tflite` file without adding both arena tests makes the suite fail with a
targeted error.
