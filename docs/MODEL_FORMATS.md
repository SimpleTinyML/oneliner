# Model formats

The built-in IREE backend accepts TFLite, ONNX, PyTorch `ExportedProgram`, and
TensorFlow SavedModel v2 models. Model paths are resolved relative to the
application's `Cargo.toml`.

The generated `ModelInference` API currently targets fixed-shape models with:

- exactly one input tensor
- exactly one output tensor
- up to four dimensions

Integer and floating-point tensor element types are inferred automatically.

## PyTorch models

Oneliner accepts a PyTorch `ExportedProgram` saved with the conventional `.pt2`
extension. Export the inference model with fixed example input shapes:

```python
import torch

model = MyModel()
model.load_state_dict(torch.load("model.pth", weights_only=True))
model.eval()

example_input = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
exported = torch.export.export(model, (example_input,))
torch.export.save(exported, "model.pt2")
```

Then bind it like any other model:

```rust
#[model("models/model.pt2")]
struct MyModel;
```

`.pt` and `.pth` are checkpoint conventions rather than self-contained model
formats: a checkpoint may contain only a `state_dict`, with no forward graph or
input signature. Convert those checkpoints to `.pt2` before using them with
Oneliner. Only load models from trusted sources because PyTorch deserialization
uses pickle internally.

## TensorFlow SavedModels

Oneliner accepts TensorFlow SavedModel v2 directories. For the conventional
exported `main` method and `serving_default` signature, only the format is
needed:

```rust
#[model("models/my_saved_model", format = "tensorflow")]
struct MyModel;
```

The model must expose a `main` method with a `serving_default` signature, and
the directory must contain `saved_model.pb`. TensorFlow, `iree-tools-tf`, and
`iree-base-compiler` should be pinned to mutually compatible versions.

## Memory modes

The default `owned` mode gives each model instance an independent workspace:

```rust
#[model("models/model.tflite")]
struct MyModel;
```

This is the natural choice when model instances may run concurrently.

The `shared` mode keeps one synchronized static workspace for all instances of a
model type:

```rust
#[model("models/model.tflite", arena = "shared")]
struct MyModel;
```

Use it when reducing duplicate RAM use matters more than concurrent inference.
The [Pico example](../examples/embassy-pico-iree/) demonstrates this
configuration.