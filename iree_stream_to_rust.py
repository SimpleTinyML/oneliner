#!/usr/bin/env python3
"""Extract an IREE Stream-stage MLIR execution flow as Rust call skeletons.

This tool intentionally parses MLIR through ``iree.compiler.ir`` instead of
matching textual MLIR with regular expressions. It walks the MLIR operation
tree, keeps SSA value dependencies, and renders Stream/Flow/HAL dispatch-like
operations as Rust function calls.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence


RUST_KEYWORDS = {
    "as",
    "break",
    "const",
    "continue",
    "crate",
    "else",
    "enum",
    "extern",
    "false",
    "fn",
    "for",
    "if",
    "impl",
    "in",
    "let",
    "loop",
    "match",
    "mod",
    "move",
    "mut",
    "pub",
    "ref",
    "return",
    "self",
    "Self",
    "static",
    "struct",
    "super",
    "trait",
    "true",
    "type",
    "unsafe",
    "use",
    "where",
    "while",
    "async",
    "await",
    "dyn",
}

FLOW_PREFIXES = ("stream.", "flow.", "hal.")
STRUCTURAL_OPS = {
    "builtin.module",
    "func.func",
    "func.return",
    "stream.executable",
    "stream.executable.export",
    "hal.executable",
    "hal.executable.variant",
}
TARGET_ATTRS = (
    "callee",
    "entry_point",
    "function",
    "target",
    "export",
    "symbol",
)


@dataclass
class FlowCall:
    op_name: str
    call_name: str
    operands: list[str]
    results: list[str]
    attrs: dict[str, str] = field(default_factory=dict)
    location: str | None = None
    depth: int = 1


@dataclass
class FlowFunction:
    name: str
    args: list[str]
    calls: list[FlowCall] = field(default_factory=list)


@dataclass
class ExtractedModule:
    functions: list[FlowFunction]
    exports: list[str]


def import_iree_ir():
    """Imports IREE's MLIR Python bindings and registers common dialect modules."""

    try:
        from iree.compiler import ir
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Cannot import iree.compiler.ir. Install IREE's compiler package, "
            "for example: pip install iree-compiler"
        ) from exc

    # Importing generated dialect modules registers their custom assembly forms
    # with the active MLIR context in normal IREE Python installations. Missing
    # modules are harmless across IREE versions.
    for module_name in (
        "iree.compiler.dialects.arith",
        "iree.compiler.dialects.builtin",
        "iree.compiler.dialects.cf",
        "iree.compiler.dialects.func",
        "iree.compiler.dialects.hal",
        "iree.compiler.dialects.flow",
        "iree.compiler.dialects.stream",
        "iree.compiler.dialects.scf",
        "iree.compiler.dialects.tensor",
        "iree.compiler.dialects.util",
    ):
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            pass

    return ir


def unwrap_operation(op_like):
    return getattr(op_like, "operation", op_like)


def iter_child_ops(op) -> Iterable:
    op = unwrap_operation(op)
    for region in op.regions:
        for block in region.blocks:
            for child in block.operations:
                yield unwrap_operation(child)


def walk_ops(op, depth: int = 0) -> Iterable[tuple[object, int]]:
    op = unwrap_operation(op)
    yield op, depth
    for child in iter_child_ops(op):
        yield from walk_ops(child, depth + 1)


def get_attr_map(op) -> dict[str, str]:
    attrs = getattr(op, "attributes", None)
    if attrs is None:
        return {}

    items: list[tuple[str, object]] = []
    if hasattr(attrs, "items"):
        try:
            items = list(attrs.items())
        except Exception:
            items = []

    if not items:
        try:
            for name in attrs:
                items.append((str(name), attrs[name]))
        except Exception:
            pass

    return {str(name): attr_to_text(attr) for name, attr in items}


def attr_to_text(attr) -> str:
    value = getattr(attr, "value", None)
    if value is not None:
        text = str(value)
    else:
        text = str(attr)
    text = text.strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        return text[1:-1]
    if text.startswith("@"):
        return text[1:]
    return text


def get_symbol_name(op) -> str | None:
    attrs = get_attr_map(op)
    for key in ("sym_name", "name"):
        if key in attrs:
            return strip_symbol_prefix(attrs[key])
    return None


def strip_symbol_prefix(text: str) -> str:
    text = text.strip()
    while text.startswith("@") or text.startswith("^"):
        text = text[1:]
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1]
    return text


def rust_ident(text: str, fallback: str = "unnamed") -> str:
    text = strip_symbol_prefix(text)
    chars: list[str] = []
    previous_underscore = False
    for char in text:
        if char.isalnum() or char == "_":
            chars.append(char.lower())
            previous_underscore = False
        else:
            if not previous_underscore:
                chars.append("_")
                previous_underscore = True
    ident = "".join(chars).strip("_")
    if not ident:
        ident = fallback
    if ident[0].isdigit():
        ident = "_" + ident
    if ident in RUST_KEYWORDS:
        ident = ident + "_"
    return ident


def unique_name(base: str, used: set[str]) -> str:
    candidate = base
    index = 1
    while candidate in used:
        candidate = f"{base}_{index}"
        index += 1
    used.add(candidate)
    return candidate


def value_key(value) -> str:
    text = str(value).strip()
    return text if text else repr(value)


def op_results(op) -> list:
    try:
        return list(op.results)
    except Exception:
        return []


def op_operands(op) -> list:
    try:
        return list(op.operands)
    except Exception:
        return []


def should_emit_op(op_name: str, include_all: bool) -> bool:
    if op_name in STRUCTURAL_OPS:
        return False
    if include_all:
        return True
    return op_name.startswith(FLOW_PREFIXES) or op_name == "func.call"


def infer_call_name(op_name: str, attrs: Mapping[str, str]) -> str:
    target = None
    for attr_name in TARGET_ATTRS:
        if attr_name in attrs:
            target = strip_symbol_prefix(attrs[attr_name])
            break

    if target and (op_name.endswith(".dispatch") or op_name in {"func.call", "stream.call"}):
        return rust_ident(target, fallback="dispatch")
    if target and op_name.endswith(".call"):
        return rust_ident(target, fallback="call")
    return rust_ident(op_name, fallback="mlir_op")


class StreamFlowExtractor:
    def __init__(self, include_all_ops: bool = False):
        self.include_all_ops = include_all_ops

    def extract(self, module_op) -> ExtractedModule:
        module_op = unwrap_operation(module_op)
        exports = self._collect_exports(module_op)
        functions = self._collect_functions(module_op)
        if not functions:
            functions = [self._extract_region_as_function("module", module_op)]
        return ExtractedModule(functions=functions, exports=exports)

    def _collect_exports(self, module_op) -> list[str]:
        exports: list[str] = []
        used: set[str] = set()
        executable_stack: list[str] = []

        def visit(op):
            op_name = op.name
            pushed = False
            if op_name in {"stream.executable", "hal.executable", "hal.executable.variant"}:
                sym = get_symbol_name(op)
                if sym:
                    executable_stack.append(sym)
                    pushed = True
            if op_name in {"stream.executable.export", "hal.executable.export"}:
                sym = get_symbol_name(op)
                if sym:
                    qualified = "::".join([*executable_stack, sym]) if executable_stack else sym
                    exports.append(unique_name(qualified, used))
            for child in iter_child_ops(op):
                visit(child)
            if pushed:
                executable_stack.pop()

        visit(module_op)
        return exports

    def _collect_functions(self, module_op) -> list[FlowFunction]:
        functions: list[FlowFunction] = []
        used: set[str] = set()
        for op, _depth in walk_ops(module_op):
            if op.name == "func.func":
                sym_name = get_symbol_name(op) or "function"
                name = unique_name(rust_ident(sym_name, fallback="function"), used)
                functions.append(self._extract_region_as_function(name, op))
        return functions

    def _extract_region_as_function(self, name: str, container_op) -> FlowFunction:
        value_names: dict[str, str] = {}
        used_names: set[str] = set()
        args = self._name_block_arguments(container_op, value_names, used_names)
        calls: list[FlowCall] = []

        for op, depth in self._walk_executable_flow(container_op):
            op_name = op.name
            if not should_emit_op(op_name, self.include_all_ops):
                continue

            attrs = get_attr_map(op)
            operands = [self._name_value(value, value_names, used_names, "input") for value in op_operands(op)]
            results = [
                self._assign_result_name(value, op_name, value_names, used_names, index)
                for index, value in enumerate(op_results(op))
            ]
            location_text = self._location_text(op)
            calls.append(
                FlowCall(
                    op_name=op_name,
                    call_name=infer_call_name(op_name, attrs),
                    operands=operands,
                    results=results,
                    attrs=self._interesting_attrs(attrs),
                    location=location_text,
                    depth=max(1, depth),
                )
            )
        return FlowFunction(name=name, args=args, calls=calls)

    def _walk_executable_flow(self, container_op) -> Iterable[tuple[object, int]]:
        for child in iter_child_ops(container_op):
            yield from self._walk_executable_flow_from_child(child, depth=1)

    def _walk_executable_flow_from_child(self, op, depth: int) -> Iterable[tuple[object, int]]:
        if op.name == "func.func":
            return
        yield op, depth
        for child in iter_child_ops(op):
            yield from self._walk_executable_flow_from_child(child, depth + 1)

    def _name_block_arguments(
        self, container_op, value_names: dict[str, str], used_names: set[str]
    ) -> list[str]:
        args: list[str] = []
        for region in container_op.regions:
            for block in region.blocks:
                try:
                    block_args = list(block.arguments)
                except Exception:
                    block_args = []
                for index, block_arg in enumerate(block_args):
                    base = rust_ident(value_key(block_arg), fallback=f"arg{index}")
                    if base.startswith("_"):
                        base = f"arg{index}"
                    name = unique_name(base, used_names)
                    value_names[value_key(block_arg)] = name
                    args.append(name)
                return args
        return args

    def _name_value(
        self, value, value_names: dict[str, str], used_names: set[str], fallback_prefix: str
    ) -> str:
        key = value_key(value)
        existing = value_names.get(key)
        if existing:
            return existing
        name = unique_name(f"{fallback_prefix}{len(used_names)}", used_names)
        value_names[key] = name
        return name

    def _assign_result_name(
        self,
        value,
        op_name: str,
        value_names: dict[str, str],
        used_names: set[str],
        result_index: int,
    ) -> str:
        key = value_key(value)
        existing = value_names.get(key)
        if existing:
            return existing
        op_base = rust_ident(op_name, fallback="value")
        suffix = "" if result_index == 0 else f"_{result_index}"
        name = unique_name(f"{op_base}{suffix}", used_names)
        value_names[key] = name
        return name

    def _interesting_attrs(self, attrs: Mapping[str, str]) -> dict[str, str]:
        result: dict[str, str] = {}
        for key, value in attrs.items():
            if key in {"sym_name", "function_type"}:
                continue
            result[key] = value
        return result

    def _location_text(self, op) -> str | None:
        try:
            text = str(op.location).strip()
        except Exception:
            return None
        if text and text != "unknown":
            return text
        return None


class RustRenderer:
    def render(self, module: ExtractedModule) -> str:
        lines: list[str] = [
            "// Generated by iree_stream_to_rust.py.",
            "// The calls are an execution-flow skeleton, not direct IREE runtime API bindings.",
            "",
        ]
        if module.exports:
            lines.append("// Discovered executable exports:")
            for export in module.exports:
                lines.append(f"// - {export}")
            lines.append("")

        for function in module.functions:
            lines.extend(self._render_function(function))
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def _render_function(self, function: FlowFunction) -> list[str]:
        args = ", ".join(f"{arg}: Value" for arg in function.args)
        lines = [
            "#[allow(unused_variables)]",
            f"pub fn {function.name}_flow(ctx: &mut StreamContext, {args}) {{"
            if args
            else f"pub fn {function.name}_flow(ctx: &mut StreamContext) {{",
        ]
        if not function.calls:
            lines.append("    // No Stream/Flow/HAL execution operations were found in this region.")
        for call in function.calls:
            lines.extend(self._render_call(call))
        lines.append("}")
        return lines

    def _render_call(self, call: FlowCall) -> list[str]:
        lines: list[str] = []
        indent = "    " * call.depth
        comment_parts = [call.op_name]
        if call.location:
            comment_parts.append(call.location)
        if call.attrs:
            attrs = ", ".join(f"{key}={value}" for key, value in sorted(call.attrs.items()))
            comment_parts.append(attrs)
        lines.append(f"{indent}// {' | '.join(comment_parts)}")

        args = ", ".join(["ctx", *call.operands])
        invocation = f"{call.call_name}({args})"
        if not call.results:
            lines.append(f"{indent}{invocation};")
        elif len(call.results) == 1:
            lines.append(f"{indent}let {call.results[0]} = {invocation};")
        else:
            result_tuple = ", ".join(call.results)
            lines.append(f"{indent}let ({result_tuple}) = {invocation};")
        return lines


def extract_mlir(path: Path, include_all_ops: bool) -> ExtractedModule:
    ir = import_iree_ir()
    with ir.Context() as context:
        context.allow_unregistered_dialects = True
        text = path.read_text(encoding="utf-8")
        module = ir.Module.parse(text)
        return StreamFlowExtractor(include_all_ops=include_all_ops).extract(module.operation)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Parse IREE Stream-stage MLIR with iree.compiler.ir and render "
            "the execution flow as Rust-style function calls."
        )
    )
    parser.add_argument("input", type=Path, help="Input MLIR file produced around IREE's Stream stage.")
    parser.add_argument("-o", "--output", type=Path, help="Rust output file. Defaults to stdout.")
    parser.add_argument(
        "--include-all-ops",
        action="store_true",
        help="Emit every operation in function bodies instead of only stream/flow/hal/call operations.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    extracted = extract_mlir(args.input, include_all_ops=args.include_all_ops)
    rendered = RustRenderer().render(extracted)
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
