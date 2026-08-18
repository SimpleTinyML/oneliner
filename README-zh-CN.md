# Oneliner

[![Current Crates.io Version](https://img.shields.io/crates/v/oneliner.svg)](https://crates.io/crates/oneliner)
[![Minimum Supported Rust Version](https://img.shields.io/crates/msrv/oneliner)](https://crates.io/crates/oneliner)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](#%E8%AE%B8%E5%8F%AF%E8%AF%81)

> **一行代码完成 TinyML 模型推理，支持 `no_std` 嵌入式目标。**

Oneliner 只需要一个属性就能把一个模型文件变成一个可直接调用的 Rust 类型：

```rust
#[model("models/model.tflite")]
struct MyModel;
```

在构建阶段，Oneliner 会导入模型、针对所选 Rust 目标编译、生成 Rust 绑定，并链接原生模型代码。在运行时，你的应用可以直接使用普通的、强类型的 Rust 张量。

```rust
use oneliner::model;
use oneliner::runtime::ModelInference;

#[model("models/model.tflite")]
struct MyModel;

fn main() {
    let mut model = MyModel::new();
    let mut input = MyModel::create_input_tensor();
    input.fill(1);

    let output = model.run(&input);
    println!("{:?}", output.as_slice());
}
```

## 为什么选择 Oneliner？

- **一行绑定模型：** 用 `#[model(...)]` 取代模型转换脚本、原生链接配置、张量声明和调度胶水代码。
- **类型化的输入输出：** 张量元素类型和形状直接来自模型，不匹配会在构建期暴露，而不是等到设备上运行时才报错。
- **专为设备端推理而生：** 模型被编译成目标原生代码，推理不依赖任何云服务。
- **为嵌入式准备：** 运行时支持 `no_std`，并已在 RP2040 上的 Ariel OS 和 Embassy 中验证。
- **内存可控：** 可选用每个实例独立的工作区，或一个同步共享的工作区。
- **内置性能剖析：** 可选的 `profiler` 特性提供推理延迟测量与构建期 flash/RAM 占用报告。

## 快速开始

1. [安装宿主端模型工具链](docs/INSTALLATION.md)。
2. 在 `Cargo.toml` 中添加依赖：

   ```toml
   [dependencies]
   oneliner = "0.2"
   ```

3. 绑定并运行模型：

   ```rust
   use oneliner::model;
   use oneliner::runtime::ModelInference;

   #[model("models/model.tflite")]
   struct MyModel;

   let mut model = MyModel::new();
   let mut input = MyModel::create_input_tensor();
   input.as_slice_mut().copy_from_slice(&input_data);

   let output = model.run(&input);
   let values = output.as_slice();
   ```

Oneliner 会直接从模型生成输入和输出张量类型，应用无需重复声明它们的数据类型和维度。

## 支持的模型

内置的 IREE 后端支持：

- TFLite
- ONNX
- PyTorch `ExportedProgram`（`.pt2`）
- TensorFlow SavedModel v2 目录
- IREE 接受的 MLIR

各格式的使用指南以及 `owned`/`shared` [内存模式](docs/MODEL_FORMATS.md#memory-modes) 详见[模型格式说明](docs/MODEL_FORMATS.md)。

## 性能剖析

启用可选的 `profiler` 特性即可测量推理延迟：

```toml
[dependencies]
oneliner = { version = "0.2", features = ["profiler"] }
```

用剖析器作用域包住任意一次推理调用：

```rust
use oneliner::profiler::Profiler;

let mut profiler = Profiler::new();
let output = profiler.profile(|| model.run(&input));
println!("{}", profiler.stats());
```

在 `no_std` 目标上，请直接依赖 `oneliner-profiler` 以选择合适的计时器后端（Ariel OS 或 Embassy）。此外，每次模型构建都会自动打印 flash/RAM 占用报告（参数、机器码、只读数据、工作区），详见两个 profiler 示例的实时输出。

## 示例

每个示例都是独立的 Cargo 工程。请在示例目录下、并激活 Python 虚拟环境后运行相应命令。

| 示例 | 演示内容 | 使用的模型 |
| --- | --- | --- |
| [Desktop IREE](examples/std-iree/) | 在标准宿主上的最短端到端验证路径 | 量化 MCUNet 视觉唤醒词 |
| [Ariel OS + IREE](examples/ariel-os-iree/) | `no_std`、Ariel OS 线程、板级原生验证与推理计时 | 量化 LeNet5 |
| [Embassy + IREE on Pico](examples/embassy-pico-iree/) | 裸机 RP2040、共享模型工作区、静态输入存储与 `defmt` 日志 | 量化 LeNet5 |
| [Ariel OS + Profiler](examples/ariel-os-profiler/) | `no_std` 延迟剖析（`Profiler`）与自动 flash/RAM 占用报告 | 量化 LeNet5 |
| [Embassy + Profiler on Pico](examples/embassy-pico-profiler/) | 裸机 RP2040 延迟剖析与占用报告 | 量化 LeNet5 |

建议先运行[桌面示例](examples/std-iree/)确认模型工具链，再选择与目标环境匹配的操作系统或开发板示例。

## 项目状态

Oneliner 当前版本为 `0.2.0`。项目专注于在桌面 Rust 和内存受限的 `no_std` 目标上，让固定形状、单输入单输出的推理变得简单直接。

示例刻意保持小而直观，目的是帮助你验证工具链、理解内存取舍，并把内置模型替换成你自己的模型。

## 测试

激活宿主模型工具链后，在仓库根目录运行 std 端到端测试套件：

```sh
cargo test
```

该命令会对 `examples/models` 中的每个模型执行端到端推理，分别使用 `owned` 和 `shared` 两种工作区模式。

## 许可证

本仓库基于 [Apache License, Version 2.0](LICENSE-APACHE) 或 [MIT license](LICENSE-MIT) 二选一授权。

除非你明确声明，任何有意提交的贡献都将按上述双重许可授权，不附加任何额外条款或条件。

**其他语言：** [English](README.md)