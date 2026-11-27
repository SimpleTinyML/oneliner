# Memory model

Model inference needs a scratch workspace -- a memory block to place activations and intermediate tensors. Every `#[model]` macro generates a typed, zero-initialized workspace value attached to its model.

## Internal workspace contract

`Model::new()` creates the workspace inside the model:

```rust
#[model("models/model.tflite")]
struct MyModel;

// Generated Code by #[model]
struct MyModelWorkspace {
    ARENA1: [u8, N1],
    ARENA2: [u8, N2]
    //...
}

struct MyModel {
    workspace: MyModelWorkspace,
    //...
}

impl MyModel {
    pub const fn new() -> Self {
        Self {
            workspace: MyModelWorkspace::new(),
        }
    }
}
//...
// END

```

- Every `Model::new()` call creates a fresh workspace; `run()` uses only the model's own workspace.

- `Model::new()` is a `const fn`, so a complete model value can be initialized directly in a `static` or a `ConstStaticCell`. The [Embassy Pico examples](../examples/embassy-pico-minimal/) use this pattern to put the model and its inline workspace in static RAM.

- The workspace value is stored inline in the model struct, so it lives wherever the model instance lives.

- No synchronization is needed between instances: instances never touch each other's memory, so concurrent inference on different instances (e.g. from multiple threads or tasks) is safe with zero coordination overhead.

- `Default` is implemented as `new()`.


## Sharing across threads and interrupts

The workspace belongs to the model instance, so different instances can be used concurrently without synchronization. Sharing one model value between threads, tasks or interrupts would alias its workspace through a single `&mut`, which safe Rust does not allow; serializing such access is the caller's responsibility. If one model instance must be shared, wrap it in the synchronization primitive your environment provides (a mutex, a `critical_section::Mutex`, or an Ariel OS/Embassy lock) and create and run inference while holding the guard.

## Examples

- The [Embassy Pico examples](../examples/embassy-pico-minimal/) keep the whole const-initialized `Model` in a `ConstStaticCell`, so the model and its inline workspace live in static RAM.
- The [Ariel OS examples](../examples/ariel-os-minimal/) use `Model::new()` and run inference from an RTOS thread.
- The [desktop example](../examples/std-minimal/) is the fastest place to check the toolchain before moving to an embedded target.

See [Profiling and footprint reporting](profiling.md) for how `arena_size` is reported, and [Benchmark](benchmark.md) for measured workspace payloads.