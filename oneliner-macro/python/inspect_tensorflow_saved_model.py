#!/usr/bin/env python3
"""Extracts OneLiner-compatible I/O metadata from a TensorFlow SavedModel."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a TensorFlow SavedModel v2 for OneLiner"
    )
    parser.add_argument("input", type=Path, help="SavedModel directory")
    parser.add_argument("--output", required=True, type=Path, help="output JSON file")
    return parser.parse_args()


def package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def tensor_metadata(tensor: Any, label: str) -> dict[str, Any]:
    dtype_name = tensor.dtype.name
    element_type = {
        "int8": "i8",
        "int16": "i16",
        "int32": "i32",
        "int64": "i64",
        "uint8": "u8",
        "uint16": "u16",
        "uint32": "u32",
        "uint64": "u64",
        "float32": "f32",
        "float64": "f64",
    }.get(dtype_name)
    if element_type is None:
        raise TypeError(
            f"OneLiner does not support TensorFlow {label} dtype {dtype_name}"
        )

    shape = tensor.shape
    if shape.rank is None:
        raise ValueError(f"TensorFlow {label} tensor has unknown rank")
    dimensions = shape.as_list()
    if len(dimensions) > 4:
        raise ValueError(
            f"TensorFlow {label} tensor rank {len(dimensions)} exceeds "
            "Tensor's four dimensions"
        )
    if any(dimension is None for dimension in dimensions):
        raise ValueError(f"TensorFlow {label} tensor has a dynamic dimension")
    return {
        "element_type": element_type,
        "shape": [1] * (4 - len(dimensions)) + dimensions,
    }


def inspect_signature(inputs: Any, outputs: Any) -> dict[str, Any]:
    inputs = list(inputs)
    outputs = list(outputs)
    if len(inputs) != 1:
        raise ValueError(
            "OneLiner ModelInference requires exactly one TensorFlow signature "
            f"input, but the signature declares {len(inputs)}"
        )
    if len(outputs) != 1:
        raise ValueError(
            "OneLiner ModelInference requires exactly one TensorFlow signature "
            f"output, but the signature declares {len(outputs)}"
        )
    return {
        "input": tensor_metadata(inputs[0], "input"),
        "output": tensor_metadata(outputs[0], "output"),
    }


def inspect_model(input_path: Path) -> dict[str, Any]:
    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError(
            "TensorFlow support requires the 'tensorflow' host package"
        ) from error

    model = tf.saved_model.load(input_path)
    if "serving_default" not in model.signatures:
        raise ValueError(
            "TensorFlow signature 'serving_default' was not found; available "
            f"signatures: {sorted(model.signatures)}"
        )
    signature = model.signatures["serving_default"]
    _, inputs = signature.structured_input_signature
    return inspect_signature(inputs.values(), signature.structured_outputs.values())


def main() -> None:
    args = parse_args()
    try:
        metadata = inspect_model(args.input.resolve())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    except Exception as error:
        versions = (
            f"tensorflow={package_version('tensorflow')}, "
            f"iree-tools-tf={package_version('iree-tools-tf')}, "
            f"iree-base-compiler={package_version('iree-base-compiler')}"
        )
        raise RuntimeError(f"{error} ({versions})") from error


if __name__ == "__main__":
    main()
