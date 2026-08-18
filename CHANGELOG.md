# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-08-18

### Highlights

- PyTorch model support: add support for PyTorch `ExportedProgram` (`.pt2`) format ([#1]).
- TensorFlow model support: add support for TensorFlow SavedModel v2 directory with `format = "tensorflow"` ([#9]).
- New `oneliner-profiler` crate with latency profiling (`Profiler`, `LatencyStats`, `Timer` with std / Ariel OS / Embassy backends) ([#10]).


### Added

- Build-time flash/RAM footprint report (parameters, machine code, read-only data, workspace) printed automatically during compilation ([#10]).
- Ariel OS and Embassy Pico profiler examples ([#10]).
- End-to-end std tests for PyTorch and TensorFlow models, in both `owned` and `shared` arena modes.

### Changed

- Simplified and improved the model parsing frontend with a more extensible design ([#3]).
- Degraded the Ariel OS executor to sequential execution on single-core MCUs ([#11]).

### Fixed

- Removed model source embedding from generated code. ([#8]).

## [0.1.0] - 2026-08-04

### Added

- Initial release with the IREE backend.
- `#[model(...)]` attribute binding for TFLite, ONNX, and MLIR model files.
- Build-time type generation from the model's input/output signature.
- `owned` and `shared` workspace memory modes.
- `no_std` runtime with Ariel OS and Embassy example projects.
- End-to-end std test suite across.

[#1]: https://github.com/SimpleTinyML/oneliner/pull/1
[#3]: https://github.com/SimpleTinyML/oneliner/pull/3
[#8]: https://github.com/SimpleTinyML/oneliner/pull/8
[#9]: https://github.com/SimpleTinyML/oneliner/pull/9
[#10]: https://github.com/SimpleTinyML/oneliner/pull/10
[#11]: https://github.com/SimpleTinyML/oneliner/pull/11

[0.2.0]: https://github.com/SimpleTinyML/oneliner/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/SimpleTinyML/oneliner/releases/tag/v0.1.0