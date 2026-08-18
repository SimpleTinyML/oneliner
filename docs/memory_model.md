# Memory model

Inference needs a scratch workspace: the arena holds the model's transient data (activations, intermediate tensors) for the duration of a `run()` call. Oneliner offers two memory modes, chosen per model type with the `arena` attribute option:

- `owned` — every model instance gets an independent workspace (the **default**).
- `shared` — all instances of a model type share one synchronized static workspace.

The workspace size (`ram_size`, reported as `arena` in the build-time footprint report) is fixed at compile time by the IREE backend; the choice below decides how many copies of it exist and how they are coordinated.

## owned mode

The default. Each model instance owns a private workspace:

```rust
#[model("models/model.tflite")]
struct MyModel;
```

- Each call to `MyModel::new()` creates a fresh workspace; `run()` uses only its own.
- Without the `alloc` feature the workspace is embedded inline in the model struct, so it lives wherever the instance lives (stack, static, or another object) and costs no heap and no runtime allocation.
- With the `alloc` feature the workspace is boxed on the heap, keeping the model struct small.
- No synchronization is needed: instances never touch each other's memory, so concurrent inference on different instances (e.g. from multiple threads/tasks) is safe with zero coordination overhead.
- Cost: N instances use N workspaces, so `owned` multiplies the RAM footprint by the instance count.

## shared mode

One static workspace per model type, shared and synchronized across all instances:

```rust
#[model("models/model.tflite", arena = "shared")]
struct MyModel;
```

- The macro emits a single zero-initialized static storage for the model type. Being all-zero, it lands in `.bss`: it consumes RAM but no flash.
- Every instance of the type references the same storage, so the workspace is paid once regardless of instance count.
- Access is serialized by a mutex — an Ariel OS `Lock` on Ariel OS targets, a `critical_section`-based guard elsewhere (with recursive-access detection). Only one `run()` call may use the workspace at a time.
- Choose `shared` when reducing duplicate RAM matters more than concurrent inference: typical cases are a single model instance, or several instances that are never executed concurrently.

## Which mode to choose

| | `owned` | `shared` |
| --- | --- | --- |
| RAM used | N × workspace (N instances) | 1 × workspace per model type |
| Flash used | none (workspace is inline or heap) | none (zeroed static lands in `.bss`) |
| Concurrent instances | safe, no coordination | serialized by a lock |
| Allocation | heap only with the `alloc` feature | always static, no heap |
| Overhead | none | lock / critical section per `run()` |

In short: use `owned` when instances may run concurrently or when the workspace should live inside each instance; use `shared` when memory is tight and inference is effectively single-threaded. The [Embassy Pico example](../examples/embassy-pico-minimal/) demonstrates `shared` on bare-metal RP2040, and the [Ariel OS profiler example](../examples/ariel-os-profiler/) shows the same mode under an OS.
