import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT = (
    Path(__file__).parents[1]
    / "oneliner-macro"
    / "python"
    / "inspect_tensorflow_saved_model.py"
)
SPEC = importlib.util.spec_from_file_location("oneliner_tensorflow_inspector", SCRIPT)
INSPECTOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = INSPECTOR
SPEC.loader.exec_module(INSPECTOR)


class Shape:
    def __init__(self, dimensions):
        self.dimensions = dimensions
        self.rank = None if dimensions is None else len(dimensions)

    def as_list(self):
        return self.dimensions


def tensor(dtype, shape):
    return SimpleNamespace(
        dtype=SimpleNamespace(name=dtype),
        shape=Shape(shape),
    )


class TensorflowInspectorTests(unittest.TestCase):
    def test_accepts_one_static_tensor_input_and_output(self):
        metadata = INSPECTOR.inspect_signature(
            [tensor("float32", [1, 2])], [tensor("int32", [2])]
        )

        self.assertEqual(metadata["input"]["element_type"], "f32")
        self.assertEqual(metadata["input"]["shape"], [1, 1, 1, 2])
        self.assertEqual(metadata["output"]["element_type"], "i32")

    def test_rejects_multiple_inputs(self):
        with self.assertRaisesRegex(ValueError, "exactly one TensorFlow signature input"):
            INSPECTOR.inspect_signature(
                [tensor("float32", [2]), tensor("float32", [2])],
                [tensor("float32", [2])],
            )

    def test_rejects_dynamic_shapes(self):
        with self.assertRaisesRegex(ValueError, "dynamic dimension"):
            INSPECTOR.inspect_signature(
                [tensor("float32", [None, 2])], [tensor("float32", [2])]
            )

    def test_rejects_unsupported_dtype(self):
        with self.assertRaisesRegex(TypeError, "does not support.*float16"):
            INSPECTOR.inspect_signature(
                [tensor("float16", [2])], [tensor("float32", [2])]
            )


if __name__ == "__main__":
    unittest.main()
