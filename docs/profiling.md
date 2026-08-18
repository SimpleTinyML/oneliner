# Profiling and footprint reporting

Oneliner offers two kinds of measurements:

- **Runtime latency profiling** through the optional `oneliner-profiler` crate.
- **Build-time memory footprint reporting**, printed automatically by the model macro for every `#[model(...)]` binding.

## Measuring inference latency

Add the optional profiler crate alongside `oneliner`:

```toml
[dependencies]
oneliner = { version = "0.2" }
oneliner-profiler = "0.1"
```

Wrap any inference call in a profiler scope:

```rust
use oneliner_profiler::Profiler;

let mut profiler = Profiler::new();
let output = profiler.profile(|| model.run(&input));
println!("{}", profiler.stats());
```

`profile` runs the closure, measures its wall-clock duration, records it into the accumulated statistics, and returns the closure's result unchanged. The model stays a plain local value — profiling is a scope around the call, mirroring `torch.profiler.profile`.

### Reading the statistics

`stats()` returns a `LatencyStats`:

| Field / method | Meaning |
| --- | --- |
| `samples` | Number of recorded inference calls |
| `total` | Sum of all recorded durations |
| `min` / `max` | Shortest / longest recorded duration (`Option<Duration>`) |
| `average()` | Mean duration, `None` until at least one sample exists |
| `min_micros()` / `max_micros()` / `average_micros()` | Convenience accessors in microseconds |
| `reset()` | Clears all accumulated statistics |

`LatencyStats` is `Copy` and keeps only a counter plus running total, min, and max — cheap enough for `no_std` firmware. It prints through `core::fmt::Display` (`samples=… avg=… min=… max=…`, with automatic us/ms/s units) and implements `defmt::Format` for embedded logging.

### Choosing the timer

The timer backend is selected by the profiler crate's features:

| Feature | Timer | Typical use |
| --- | --- | --- |
| `std` (default) | `StdTimer` (`std::time::Instant`) | Desktop / std targets |
| `ariel-os` | `ArielOsTimer` (Ariel OS monotonic clock) | `no_std` Ariel OS threads |
| `embassy` | `EmbassyTimer` (`embassy_time::Instant`) | `no_std` Embassy applications |

Only one of the three should be enabled. `Profiler::new()` automatically uses the `DefaultTimer` selected by the enabled feature, so on `no_std` targets you simply depend on `oneliner-profiler` with the matching feature:

```toml
[dependencies]
oneliner-profiler = { version = "0.1", features = ["embassy"], default-features = false }
```

To profile with a custom clock, implement the `Timer` trait (`now()` and `elapsed(start) -> Duration`) and construct the profiler with `Profiler::with_timer(my_timer)`.

## Automatic memory footprint report

Every `#[model(...)]` build prints a memory report to stderr, e.g.:

```
[oneliner-profiler] MyModel memory footprint:
  Flash Usage: params = 108080 B (105 KiB), text(code) = 97560 B (95 KiB), rodata = 460 B (0 KiB), total = 206100 B (201 KiB)
  RAM Usage: arena = 33548 B (32 KiB), input = 3136 B (3 KiB), output = 40 B (0 KiB)
```

The numbers come from two sources:

- **IREE flow metadata:** the model's constant (weights) and temporary (workspace) resources. Sizes are deduplicated — a blob shared by several IREE `cmd_execute` blocks is emitted once and counted once.
- **The compiled IREE object file:** parsed section-by-section to measure the machine code and read-only data that the linker places in flash.

## ModelArtifacts size fields

The same numbers are available to your program at runtime through the `ARTIFACTS` constant of the generated model type:

```rust
use oneliner::runtime::ModelSource;

let artifacts = <MyModel as ModelSource>::ARTIFACTS;
```

Each size field is a byte count:

| Field | Meaning |
| --- | --- |
| `params_size` | Model parameter/weight bytes placed in flash. |
| `code_size` | Executable machine code of the compiled model (`.text` sections of the IREE object file), placed in flash. |
| `rodata_size` | Read-only data embedded in the compiled object — lookup tables, library metadata, and `.ARM.exidx` unwind tables (`.rodata`, `.data.rel.ro`) — placed in flash. |
| `total_flash_size` | Total model flash footprint: `params_size + code_size + rodata_size`. |
| `ram_size` | Model's arena workspace held in RAM during inference. |
| `input_size` | Size in bytes of the model's single input tensor. |
| `output_size` | Size in bytes of the model's single output tensor. |

`ModelArtifacts` also carries build-related paths (`model_path`, `compile_input_path`, `object_path`, `link_path`, `ir_path`, `flow_rs_path`, `metadata_json_path`) and the `backend` name for tooling use.

## Examples

- [`ariel-os-profiler`](../examples/ariel-os-profiler/) — latency profiling and footprint logging on `no_std` Ariel OS.
- [`embassy-pico-profiler`](../examples/embassy-pico-profiler/) — the same on bare-metal RP2040 with Embassy.
