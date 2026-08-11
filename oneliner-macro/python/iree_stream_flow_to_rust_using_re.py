#!/usr/bin/env python3
"""
Extract IREE Stream command execution blocks and render Rust call flows.

This version intentionally uses regex-based MLIR text matching plus balanced
delimiter scanning. It does not require IREE Python bindings.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import struct
import sys
from pathlib import Path
from typing import Any


SSA_RE = r"%[A-Za-z_.$-][\w.$-]*(?:#\d+)?|%\d+"
RUST_KEYWORDS = {
    "as", "async", "await", "break", "const", "continue", "crate", "dyn",
    "else", "enum", "extern", "false", "fn", "for", "gen", "if", "impl",
    "in", "let", "loop", "match", "mod", "move", "mut", "pub", "ref",
    "return", "self", "Self", "static", "struct", "super", "trait", "true",
    "try", "type", "unsafe", "use", "where", "while", "yield",
}


@dataclasses.dataclass
class ConstantBlob:
    name: str
    size: int
    data: bytes
    source: str


@dataclasses.dataclass
class ResourceBinding:
    arg: str
    source: str
    kind: str
    size_expr: str
    size: int | None
    role: str
    constant_name: str | None = None


@dataclasses.dataclass
class TensorRange:
    access: str
    arg: str
    kind: str
    tensor_name: str
    offset_expr: str
    offset: int | None
    length_expr: str
    length: int | None


@dataclasses.dataclass
class DispatchCall:
    kind: str
    callee: str
    executable: str
    function: str
    ordinal: int
    params: list[str]
    param_values: list[int | None]
    ranges: list[TensorRange]
    workload: tuple[int | None, ...]


@dataclasses.dataclass
class FillCommand:
    kind: str
    value_expr: str
    value: int | None
    value_type: str
    target: TensorRange


@dataclasses.dataclass
class ConcurrentCommand:
    kind: str
    commands: list[Any]


@dataclasses.dataclass
class CmdExecute:
    name: str
    result: str | None
    line_no: int | None
    resources: list[ResourceBinding]
    commands: list[Any]


class StreamExtractionError(RuntimeError):
    pass


def rust_ident(raw: str) -> str:
    ident = re.sub(r"[^0-9A-Za-z_]", "_", raw).strip("_").lower()
    ident = re.sub(r"_+", "_", ident)
    if not ident:
        ident = "value"
    if ident[0].isdigit():
        ident = f"v_{ident}"
    if ident in RUST_KEYWORDS:
        ident += "_"
    return ident


def const_ident(raw: str) -> str:
    return rust_ident(raw).upper()


def find_matching(text: str, start: int, open_ch: str, close_ch: str) -> int:
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        ch = text[index]
        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_string:
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return index
    raise StreamExtractionError(f"unbalanced {open_ch}{close_ch}")


def split_balanced_items(text: str, separator: str = ",") -> list[str]:
    items: list[str] = []
    start = 0
    depth_angle = depth_square = depth_round = depth_brace = 0
    in_string = False
    escaped = False
    for index, ch in enumerate(text):
        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_string:
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "<":
            depth_angle += 1
        elif ch == ">":
            depth_angle -= 1
        elif ch == "[":
            depth_square += 1
        elif ch == "]":
            depth_square -= 1
        elif ch == "(":
            depth_round += 1
        elif ch == ")":
            depth_round -= 1
        elif ch == "{":
            depth_brace += 1
        elif ch == "}":
            depth_brace -= 1
        elif (
            ch == separator
            and depth_angle == 0
            and depth_square == 0
            and depth_round == 0
            and depth_brace == 0
        ):
            item = text[start:index].strip()
            if item:
                items.append(item)
            start = index + 1
    tail = text[start:].strip()
    if tail:
        items.append(tail)
    return items


def line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def parse_integer_constants(text: str) -> dict[str, int]:
    constants: dict[str, int] = {}
    pattern = re.compile(
        rf"(?P<name>{SSA_RE})\s*=\s*arith\.constant\s+(?P<value>-?\d+)\s*:\s*(?:index|i\d+)\b"
    )
    for match in pattern.finditer(text):
        constants[match.group("name")] = int(match.group("value"))
    return constants


def resolve_int(expr: str, constants: dict[str, int]) -> int | None:
    expr = expr.strip()
    if expr in constants:
        return constants[expr]
    if re.fullmatch(r"-?\d+", expr):
        return int(expr)
    return None


def parse_tensor_type(type_text: str) -> tuple[int, str] | None:
    match = re.search(r"(?:tensor|vector)<(?P<body>[^>]+)>", type_text.strip())
    if not match:
        return None
    parts = match.group("body").split("x")
    if not parts:
        return None
    element_type = parts[-1]
    count = 1
    for dim in parts[:-1]:
        if dim == "?":
            return None
        count *= int(dim)
    return count, element_type


def element_width(element_type: str) -> int:
    if element_type in {"i1", "i8", "ui8"}:
        return 1
    if element_type in {"i16", "ui16", "f16", "bf16"}:
        return 2
    if element_type in {"i32", "ui32", "f32"}:
        return 4
    if element_type in {"i64", "ui64", "f64"}:
        return 8
    raise StreamExtractionError(f"unsupported dense element type: {element_type}")


def pack_scalar(value: str, element_type: str) -> bytes:
    if element_type.startswith("i") or element_type.startswith("ui"):
        bits = element_width(element_type) * 8
        return (int(value) & ((1 << bits) - 1)).to_bytes(bits // 8, "little", signed=False)
    if element_type == "f32":
        return struct.pack("<f", float(value))
    if element_type == "f64":
        return struct.pack("<d", float(value))
    raise StreamExtractionError(f"unsupported dense element type: {element_type}")


def numeric_tokens(text: str) -> list[str]:
    return re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", text)


def dense_payload_to_bytes(payload: str, type_text: str) -> bytes | None:
    type_info = parse_tensor_type(type_text)
    if type_info is None:
        return None
    count, element_type = type_info
    payload = payload.strip()

    if payload.startswith('"0x') and payload.endswith('"'):
        return bytes.fromhex(payload[3:-1])

    tokens = numeric_tokens(payload)
    if len(tokens) == 1:
        return pack_scalar(tokens[0], element_type) * count
    if len(tokens) == count:
        return b"".join(pack_scalar(token, element_type) for token in tokens)
    return None


def dense_attr_to_bytes(item: str, dense_resources: dict[str, bytes]) -> bytes | None:
    resource_match = re.search(r"dense_resource<(?P<alias>[\w.$-]+)>", item)
    if resource_match:
        colon_pos = item.find(":", resource_match.end())
        if colon_pos < 0:
            return None
        type_info = parse_tensor_type(item[colon_pos + 1 :])
        if type_info is None:
            return None
        count, element_type = type_info
        data = dense_resources.get(resource_match.group("alias"))
        if data is None or len(data) != count * element_width(element_type):
            return None
        return data

    dense_pos = item.find("dense<")
    if dense_pos < 0:
        return None
    open_pos = dense_pos + len("dense")
    close_pos = find_matching(item, open_pos, "<", ">")
    payload = item[open_pos + 1 : close_pos]
    colon_pos = item.find(":", close_pos)
    if colon_pos < 0:
        return None
    return dense_payload_to_bytes(payload, item[colon_pos + 1 :].strip())


def parse_dense_resources(text: str) -> dict[str, bytes]:
    resources: dict[str, bytes] = {}
    pattern = re.compile(
        r'^[ \t]*(?P<alias>[\w.$-]+)[ \t]*:[ \t]*'
        r'"0x(?P<data>[0-9A-Fa-f]*)"',
        re.MULTILINE,
        )
    for match in pattern.finditer(text):
        encoded = bytes.fromhex(match.group("data"))
        if len(encoded) >= 4:
            # MLIR prefixes resource blobs with a little-endian alignment field.
            resources[match.group("alias")] = encoded[4:]
    return resources


def composite_to_bytes(
    text: str, dense_resources: dict[str, bytes]
) -> tuple[int | None, bytes | None]:
    marker = text.find("#util.composite")
    if marker < 0:
        return None, None
    open_pos = text.find("<", marker)
    if open_pos < 0:
        return None, None
    close_pos = find_matching(text, open_pos, "<", ">")
    body = text[open_pos + 1 : close_pos]
    header, _, rest = body.partition("[")
    declared_match = re.search(r"(?P<size>\d+)xi8", header)
    declared_size = int(declared_match.group("size")) if declared_match else None
    if not rest:
        return declared_size, None
    list_open = body.find("[")
    list_close = find_matching(body, list_open, "[", "]")
    data = bytearray()
    for item in split_balanced_items(body[list_open + 1 : list_close]):
        payload = dense_attr_to_bytes(item, dense_resources)
        if payload is None:
            return declared_size, None
        data.extend(payload)
    if declared_size is not None and len(data) != declared_size:
        return declared_size, None
    return declared_size, bytes(data)


def parse_composite_constants(text: str) -> dict[str, ConstantBlob]:
    constants: dict[str, ConstantBlob] = {}
    dense_resources = parse_dense_resources(text)
    pattern = re.compile(r"(?P<alias>#[\w.$-]+)\s*=\s*#util\.composite")
    for match in pattern.finditer(text):
        marker = text.find("#util.composite", match.start())
        open_pos = text.find("<", marker)
        close_pos = find_matching(text, open_pos, "<", ">")
        composite_text = text[marker : close_pos + 1]
        declared_size, data = composite_to_bytes(composite_text, dense_resources)
        if data is None:
            continue
        name = f"constant_{rust_ident(match.group('alias'))}"
        constants[name] = ConstantBlob(
            name=name,
            size=declared_size if declared_size is not None else len(data),
            data=data,
            source=match.group("alias"),
        )
    return constants


def parse_constant_sources(text: str, constants: dict[str, ConstantBlob]) -> dict[str, str]:
    alias_to_constant = {blob.source: name for name, blob in constants.items()}
    buffer_to_constant: dict[str, str] = {}
    resource_to_constant: dict[str, str] = {}
    global_to_constant: dict[str, str] = {}
    load_to_constant: dict[str, str] = {}

    buffer_pattern = re.compile(
        rf"(?P<value>{SSA_RE})\s*=\s*util\.buffer\.constant\b.*?=\s*(?P<alias>#[\w.$-]+)",
        re.DOTALL,
    )
    for match in buffer_pattern.finditer(text):
        const_name = alias_to_constant.get(match.group("alias"))
        if const_name:
            buffer_to_constant[match.group("value")] = const_name

    try_map_pattern = re.compile(
        rf"(?P<results>{SSA_RE}(?:\s*,\s*{SSA_RE})*)\s*=\s*stream\.resource\.try_map\b.*?"
        rf"(?P<buffer>{SSA_RE})\s*\[.*?\].*?!util\.buffer\s*->\s*i1\s*,\s*!stream\.resource<constant>",
        re.DOTALL,
    )
    for match in try_map_pattern.finditer(text):
        const_name = buffer_to_constant.get(match.group("buffer"))
        if not const_name:
            continue
        for result in re.findall(SSA_RE, match.group("results")):
            resource_to_constant[result] = const_name

    await_pattern = re.compile(
        rf"(?P<result>{SSA_RE})\s*=\s*stream\.timepoint\.await\b.*?=>\s*(?P<src>{SSA_RE})\s*:"
        r"\s*!stream\.resource<constant>",
        re.DOTALL,
    )
    for match in await_pattern.finditer(text):
        const_name = resource_to_constant.get(match.group("src"))
        if const_name:
            resource_to_constant[match.group("result")] = const_name

    store_pattern = re.compile(
        rf"util\.global\.store\s+(?P<src>{SSA_RE})\s*,\s*(?P<global>@[\w.$-]+)\s*:"
        r"\s*!stream\.resource<constant>"
    )
    for match in store_pattern.finditer(text):
        const_name = resource_to_constant.get(match.group("src"))
        if const_name:
            global_to_constant[match.group("global")] = const_name

    if len(constants) == 1:
        only_constant = next(iter(constants))
        for match in re.finditer(r"util\.global\s+private\s+(?P<global>@[\w.$-]+)\s*:\s*!stream\.resource<constant>", text):
            global_to_constant.setdefault(match.group("global"), only_constant)

    load_pattern = re.compile(
        rf"(?P<value>{SSA_RE})\s*=\s*util\.global\.load\s+immutable\s+(?P<global>@[\w.$-]+)\s*:"
        r"\s*!stream\.resource<constant>"
    )
    for match in load_pattern.finditer(text):
        const_name = global_to_constant.get(match.group("global"))
        if const_name:
            load_to_constant[match.group("value")] = const_name
    return load_to_constant


def parse_resource_roles(text: str) -> dict[str, str]:
    roles: dict[str, str] = {}
    for match in re.finditer(rf"(?P<value>{SSA_RE})\s*=\s*stream\.tensor\.import\b", text):
        roles[match.group("value")] = "input"
    alloca_pattern = re.compile(
        rf"(?P<results>{SSA_RE}(?:\s*,\s*{SSA_RE})*)\s*=\s*stream\.resource\.alloca\b.*?"
        r"!stream\.resource<(?P<kind>external|transient)>",
        re.DOTALL,
    )
    for match in alloca_pattern.finditer(text):
        first_result = re.findall(SSA_RE, match.group("results"))[0]
        roles[first_result] = "temporary" if match.group("kind") == "transient" else "output"
    return roles


def binding_name(binding: ResourceBinding) -> str:
    if binding.role == "input":
        return f"input_{rust_ident(binding.arg)}"
    if binding.role == "output":
        return f"output_{rust_ident(binding.arg)}"
    if binding.role == "inout":
        return f"inout_{rust_ident(binding.arg)}"
    if binding.role == "temporary":
        return f"temp_{rust_ident(binding.arg)}"
    if binding.role == "constant":
        return f"const_{rust_ident(binding.arg)}"
    return f"{rust_ident(binding.role)}_{rust_ident(binding.arg)}"


def parse_with_bindings(
    with_text: str,
    constants: dict[str, int],
    roles: dict[str, str],
    constant_by_value: dict[str, str],
) -> tuple[list[ResourceBinding], dict[str, ResourceBinding]]:
    bindings: list[ResourceBinding] = []
    by_arg: dict[str, ResourceBinding] = {}
    pattern = re.compile(
        rf"(?P<src>{SSA_RE})\s+as\s+(?P<arg>{SSA_RE})\s*:\s*"
        r"!stream\.resource<(?P<kind>[^>]+)>\{(?P<size>[^}]+)\}"
    )
    for item in split_balanced_items(with_text):
        match = pattern.search(item)
        if not match:
            continue
        source = match.group("src")
        kind = match.group("kind")
        role = roles.get(source, kind)
        if kind == "constant":
            role = "constant"
        elif kind == "transient":
            role = "temporary"
        binding = ResourceBinding(
            arg=match.group("arg"),
            source=source,
            kind=kind,
            size_expr=match.group("size").strip(),
            size=resolve_int(match.group("size"), constants),
            role=role,
            constant_name=constant_by_value.get(source),
        )
        bindings.append(binding)
        by_arg[binding.arg] = binding
    return bindings, by_arg


def split_symbol_ref(symbol: str) -> tuple[str, str]:
    parts = [part.lstrip("@") for part in symbol.split("::")]
    if len(parts) == 1:
        return "", parts[0]
    return parts[0], parts[-1]


def parse_params(param_text: str, constants: dict[str, int]) -> tuple[list[str], list[int | None]]:
    if not param_text.strip():
        return [], []
    value_part = param_text.rsplit(":", 1)[0] if ":" in param_text else param_text
    params = [item.strip() for item in split_balanced_items(value_part) if item.strip()]
    return params, [resolve_int(param, constants) for param in params]


def parse_ranges(range_text: str, constants: dict[str, int]) -> list[TensorRange]:
    ranges: list[TensorRange] = []
    range_pattern = re.compile(
        rf"(?P<access>ro|wo|rw)\s+(?P<arg>{SSA_RE})"
        r"\[(?P<offset>.*?)\s+for\s+(?P<length>.*?)\]\s*:\s*"
        r"!stream\.resource<(?P<kind>[^>]+)>\{(?P<size>[^}]+)\}",
        re.DOTALL,
    )
    for item in split_balanced_items(range_text):
        match = range_pattern.search(item)
        if not match:
            continue
        offset_expr = match.group("offset").strip()
        length_expr = match.group("length").strip()
        ranges.append(
            TensorRange(
                access=match.group("access"),
                arg=match.group("arg"),
                kind=match.group("kind"),
                tensor_name="",
                offset_expr=offset_expr,
                offset=resolve_int(offset_expr, constants),
                length_expr=length_expr,
                length=resolve_int(length_expr, constants),
            )
        )
    return ranges


def parse_dispatch(head: str, body: str, constants: dict[str, int], func_workloads) -> DispatchCall:
    match = re.search(r"stream\.cmd\.dispatch\s+(?P<callee>@[\w$]+(::@[\w$]+)*)(?P<tail>.*)", head, re.DOTALL)
    if not match:
        raise StreamExtractionError(f"could not parse dispatch head: {head.strip()}")
    callee = match.group("callee").strip()
    tail = match.group("tail").strip()
    params: list[str] = []
    param_values: list[int | None] = []
    if tail.startswith("("):
        close_pos = find_matching(tail, 0, "(", ")")
        params, param_values = parse_params(tail[1:close_pos], constants)
    executable, function = split_symbol_ref(callee)
    workload = tuple()
    ordinal = (
        int(re.sub(r".*_dispatch_", "", executable))
    )
    for item in func_workloads:
        if item["function"] == function:
            workload = item["workload"]
            break

    return DispatchCall(
        kind="dispatch",
        callee=callee,
        executable=executable,
        function=function,
        ordinal=ordinal,
        params=params,
        param_values=param_values,
        ranges=parse_ranges(body, constants),
        workload=workload,
    )


def parse_fill(fill_text: str, constants: dict[str, int]) -> FillCommand:
    pattern = re.compile(
        rf"stream\.cmd\.fill\s+(?P<value>{SSA_RE}|-?\d+)\s*,\s*(?P<arg>{SSA_RE})"
        r"\[(?P<offset>.*?)\s+for\s+(?P<length>.*?)\]\s*:\s*"
        r"(?P<value_type>.*?)\s*->\s*!stream\.resource<(?P<kind>[^>]+)>\{(?P<size>[^}]+)\}",
        re.DOTALL,
    )
    match = pattern.search(fill_text)
    if not match:
        raise StreamExtractionError(f"could not parse fill command: {fill_text.strip()[:120]}")
    value_expr = match.group("value").strip()
    offset_expr = match.group("offset").strip()
    length_expr = match.group("length").strip()
    return FillCommand(
        kind="fill",
        value_expr=value_expr,
        value=resolve_int(value_expr, constants),
        value_type=match.group("value_type").strip(),
        target=TensorRange(
            access="wo",
            arg=match.group("arg"),
            kind=match.group("kind"),
            tensor_name="",
            offset_expr=offset_expr,
            offset=resolve_int(offset_expr, constants),
            length_expr=length_expr,
            length=resolve_int(length_expr, constants),
        ),
    )


def parse_commands(body: str, constants: dict[str, int], func_workloads) -> list[Any]:
    commands: list[Any] = []
    pattern = re.compile(r"stream\.cmd\.(dispatch|fill|concurrent)\b")
    pos = 0
    while True:
        match = pattern.search(body, pos)
        if not match:
            break
        kind = match.group(1)
        if kind == "concurrent":
            open_pos = body.find("{", match.end())
            close_pos = find_matching(body, open_pos, "{", "}")
            commands.append(ConcurrentCommand(kind="concurrent", commands=parse_commands(body[open_pos + 1 : close_pos], constants, func_workloads)))
            pos = close_pos + 1
        elif kind == "dispatch":
            open_pos = body.find("{", match.end())
            close_pos = find_matching(body, open_pos, "{", "}")
            commands.append(parse_dispatch(body[match.start() : open_pos], body[open_pos + 1 : close_pos], constants, func_workloads))
            pos = close_pos + 1
        else:
            fill_match = re.compile(
                rf"stream\.cmd\.fill\s+.*?->\s*!stream\.resource<[^>]+>\{{[^}}]+\}}",
                re.DOTALL,
            ).match(body, match.start())
            if not fill_match:
                raise StreamExtractionError("could not find end of stream.cmd.fill")
            commands.append(parse_fill(fill_match.group(0), constants))
            pos = fill_match.end()
    return commands


def command_ranges(command: Any) -> list[TensorRange]:
    if isinstance(command, DispatchCall):
        return command.ranges
    if isinstance(command, FillCommand):
        return [command.target]
    if isinstance(command, ConcurrentCommand):
        ranges: list[TensorRange] = []
        for child in command.commands:
            ranges.extend(command_ranges(child))
        return ranges
    return []


def infer_external_roles(bindings: list[ResourceBinding], commands: list[Any]) -> None:
    access_by_arg: dict[str, set[str]] = {}
    for command in commands:
        for item in command_ranges(command):
            access_by_arg.setdefault(item.arg, set()).add(item.access)

    for binding in bindings:
        if binding.kind != "external":
            continue
        accesses = access_by_arg.get(binding.arg, set())
        has_read = bool(accesses & {"ro", "rw"})
        has_write = bool(accesses & {"wo", "rw"})
        if has_read and has_write:
            binding.role = "inout"
        elif has_write:
            binding.role = "output"
        elif has_read:
            binding.role = "input"


def apply_tensor_names(command: Any, bindings_by_arg: dict[str, ResourceBinding]) -> None:
    for item in command_ranges(command):
        binding = bindings_by_arg.get(item.arg)
        if binding is None:
            raise StreamExtractionError(f"resource binding for {item.arg} was not found")
        item.tensor_name = binding_name(binding)


def parse_func_workloads(mlir_text: str):
    results = []

    # Find all executable export blocks.
    export_pattern = re.compile(
        r'hal\.executable\.export\s+public\s+@([^\s(]+)(?P<head>[^{]*)\{',
        re.MULTILINE
    )

    for match in export_pattern.finditer(mlir_text):
        func_name = match.group(1)
        start_idx = match.end() - 1  # Points at the opening brace.

        # Use brace counting to find the full export body.
        brace_count = 0
        i = start_idx
        while i < len(mlir_text):
            if mlir_text[i] == '{':
                brace_count += 1
            elif mlir_text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    break
            i += 1

        func_body = mlir_text[start_idx:i+1]

        # Extract local integer constants.
        const_pattern = re.compile(
            rf"({SSA_RE})\s*=\s*arith\.constant\s+(-?\d+)\s*:\s*index"
        )
        const_map = {
            var: int(val)
            for var, val in const_pattern.findall(func_body)
        }

        # Extract the workload returned by the export.
        return_pattern = re.compile(
            r'hal\.return\s+([^:]+)\s*:\s*((?:index\s*,\s*)*index)'
        )
        return_match = return_pattern.search(func_body)

        workload = tuple()
        if return_match:
            vars_ = [v.strip() for v in return_match.group(1).split(',')]
            workload = tuple(resolve_int(v, const_map) for v in vars_)

        results.append({
            "function": func_name,
            "workload": workload
        })
    return results


def iter_dispatches(commands: list[Any]) -> Any:
    for command in commands:
        if isinstance(command, DispatchCall):
            yield command
        elif isinstance(command, ConcurrentCommand):
            yield from iter_dispatches(command.commands)

def normalize_dispatch_ordinals(commands: list[Any]) -> None:
    sorted_ordinals = sorted({command.ordinal for command in iter_dispatches(commands)})
    ordinal_map = {ordinal: index for index, ordinal in enumerate(sorted_ordinals)}
    for command in iter_dispatches(commands):
        command.ordinal = ordinal_map[command.ordinal]


def parse_cmd_executes(text: str) -> tuple[list[CmdExecute], dict[str, ConstantBlob]]:
    constants = parse_integer_constants(text)
    constant_blobs = parse_composite_constants(text)
    constant_by_value = parse_constant_sources(text, constant_blobs)
    roles = parse_resource_roles(text)
    func_workloads = parse_func_workloads(text)
    executes: list[CmdExecute] = []

    execute_pattern = re.compile(rf"(?:(?P<result>{SSA_RE})\s*=\s*)?stream\.cmd\.execute\b")
    search_from = 0
    while True:
        match = execute_pattern.search(text, search_from)
        if not match:
            break
        with_pos = text.find("with(", match.end())
        if with_pos < 0:
            search_from = match.end()
            continue
        with_open = with_pos + len("with")
        with_close = find_matching(text, with_open, "(", ")")
        body_open = text.find("{", with_close)
        if body_open < 0:
            search_from = with_close + 1
            continue
        body_close = find_matching(text, body_open, "{", "}")

        bindings, bindings_by_arg = parse_with_bindings(text[with_open + 1 : with_close], constants, roles, constant_by_value)
        commands = parse_commands(text[body_open + 1 : body_close], constants, func_workloads)
        infer_external_roles(bindings, commands)
        for command in commands:
            apply_tensor_names(command, bindings_by_arg)

        normalize_dispatch_ordinals(commands)

        executes.append(
            CmdExecute(
                name=f"cmd_execute_{len(executes)}",
                result=match.group("result"),
                line_no=line_number(text, match.start()),
                resources=bindings,
                commands=commands,
            )
        )
        search_from = body_close + 1

    return executes, constant_blobs


def bytes_to_rust_array(data: bytes, indent: str = "    ", per_line: int = 16) -> list[str]:
    lines: list[str] = []
    for start in range(0, len(data), per_line):
        chunk = data[start : start + per_line]
        lines.append(f"{indent}{', '.join(f'0x{byte:02X}' for byte in chunk)},")
    return lines


def render_resource_static(binding: ResourceBinding, constant_blobs: dict[str, ConstantBlob]) -> list[str]:
    name = const_ident(binding_name(binding))
    if binding.role != "constant":
        raise StreamExtractionError(f"mutable resource {binding.arg} must be stored in Workspace")
    if binding.constant_name and binding.constant_name in constant_blobs:
        blob = constant_blobs[binding.constant_name]
        lines = [f"pub static {name}: Aligned<AlignedType,[u8; {blob.size}]> = Aligned(["]
        lines.extend(bytes_to_rust_array(blob.data))
        lines.append("]);")
        return lines
    raise StreamExtractionError(f"constant {binding.arg} could not be materialized")


def render_workspace_field(binding: ResourceBinding) -> str:
    name = const_ident(binding_name(binding))
    if binding.role != "temporary":
        raise StreamExtractionError(
            f"{binding.role} resource {binding.arg} cannot be a Workspace field"
        )

    if binding.size is None:
        raise StreamExtractionError(
            f"resource {binding.arg} size expression {binding.size_expr} could not be resolved"
        )
    return f"pub(super) {name}: Aligned<AlignedType, [u8; {binding.size}]>,"


def render_workspace_initializer(binding: ResourceBinding) -> str:
    name = const_ident(binding_name(binding))
    if binding.role != "temporary":
        raise StreamExtractionError(
            f"{binding.role} resource {binding.arg} cannot initialize Workspace"
        )
    if binding.size is None:
        raise StreamExtractionError(
            f"resource {binding.arg} size expression {binding.size_expr} could not be resolved"
        )
    return f"{name}: Aligned([0; {binding.size}]),"


def render_tensor_range(
    item: TensorRange,
    workspace_names: frozenset[str] = frozenset(),
    external_roles: dict[str, str] | None = None,
) -> str:
    access = {"ro": "Ro", "wo": "Wo", "rw": "Rw"}.get(item.access, "Unknown")
    if item.offset is None or item.length is None:
        raise StreamExtractionError(
            f"unresolved tensor range {item.arg}: {item.offset_expr} for {item.length_expr}"
        )
    offset = item.offset
    length = item.length
    storage_name = const_ident(item.tensor_name)
    role = (external_roles or {}).get(item.tensor_name)
    if item.tensor_name in workspace_names:
        storage = f"(*workspace.{storage_name}).to_buffer_mut()"
    elif role == "input":
        storage = "input"
    elif role == "output":
        storage = "output"
    else:
        storage = f"(*{storage_name}).to_buffer_ref()"
    return (
        f"AnyBufferRange {{ buffer: {storage}.into(), access: Access::{access}, "
        f"offset: {offset}, length: {length} }}"
    )


def render_command(
    command: Any,
    indent: str,
    workspace_names: frozenset[str] = frozenset(),
    external_roles: dict[str, str] | None = None,
) -> list[str]:
    out: list[str] = []
    if isinstance(command, DispatchCall):
        if (
            any(value is None for value in command.param_values)
            or len(command.workload) != 3
            or any(value is None for value in command.workload)
        ):
            raise StreamExtractionError(f"unresolved dispatch values for {command.callee}")
        params = ", ".join(str(value) for value in command.param_values)
        workload = ", ".join(str(value) for value in command.workload)
        out.append(f"{indent}unsafe {{")
        out.append(
            f"{indent}    try_dispatch(dispatch_fn_from_library(QUERY_FN_PTR, {command.ordinal})?, &[{params}], &[{workload}], &["
        )
        for item in command.ranges:
            out.append(
                f"{indent}        {render_tensor_range(item, workspace_names, external_roles)},"
            )
        out.append(f"{indent}    ])?;")
        out.append(f"{indent}}}")
    elif isinstance(command, FillCommand):
        if command.value is None:
            raise StreamExtractionError(f"unresolved fill value {command.value_expr}")
        rendered = render_tensor_range(command.target, workspace_names, external_roles)
        out.append(f"{indent}unsafe {{ fill({rendered}, {command.value})?; }}")
    elif isinstance(command, ConcurrentCommand):
        out.append(f"{indent}concurrent(|| {{")
        for child in command.commands:
            out.extend(
                render_command(child, indent + "    ", workspace_names, external_roles)
            )
        out.append(f"{indent}    Ok(())")
        out.append(f"{indent}}})?;")
    return out

def render_rust(executes: list[CmdExecute], constant_blobs: dict[str, ConstantBlob]) -> str:
    out: list[str] = [
        "// Generated by iree_stream_flow_to_rust_using_re.py",
        "// MLIR was matched with regex plus balanced delimiter scanning.",
        "",
    ]

    emitted: set[str] = set()
    bindings: list[ResourceBinding] = []
    for execute in executes:
        for binding in execute.resources:
            name = binding_name(binding)
            if name in emitted:
                continue
            emitted.add(name)
            bindings.append(binding)

    constant_bindings = [binding for binding in bindings if binding.role == "constant"]
    workspace_bindings = [binding for binding in bindings if binding.role == "temporary"]
    external_bindings = [
        binding for binding in bindings if binding.role in {"input", "output"}
    ]
    unsupported_bindings = [
        binding
        for binding in bindings
        if binding.role not in {"constant", "temporary", "input", "output"}
    ]
    if unsupported_bindings:
        raise StreamExtractionError(
            "unsupported mutable resource roles: "
            + ", ".join(binding.role for binding in unsupported_bindings)
        )
    workspace_names = frozenset(binding_name(binding) for binding in workspace_bindings)
    external_roles = {
        binding_name(binding): binding.role for binding in external_bindings
    }

    for binding in constant_bindings:
        out.extend(render_resource_static(binding, constant_blobs))
        out.append("")

    out.append("pub struct Workspace {")
    for binding in workspace_bindings:
        out.append(f"    {render_workspace_field(binding)}")
    out.append("}")
    out.append("")
    out.append("impl Workspace {")
    out.append("    pub const fn new() -> Self {")
    out.append("        Self {")
    for binding in workspace_bindings:
        out.append(f"            {render_workspace_initializer(binding)}")
    out.append("        }")
    out.append("    }")
    out.append("}")
    out.append("")
    out.append("impl Default for Workspace {")
    out.append("    fn default() -> Self {")
    out.append("        Self::new()")
    out.append("    }")
    out.append("}")
    out.append("")

    for execute in executes:
        out.append(
            f"pub fn {rust_ident(execute.name)}("
            "workspace: &mut Workspace, input: Buffer, output: BufferMut) "
            "-> Result<(), Error> {"
        )
        if execute.line_no is not None:
            out.append(f"    // source MLIR line: {execute.line_no}")
        if execute.result:
            out.append(f"    // stream.cmd.execute result timepoint: {execute.result}")
        for command in execute.commands:
            out.extend(
                render_command(command, "    ", workspace_names, external_roles)
            )
        out.append("    Ok(())")
        out.append("}")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def dataclass_to_json(value: Any) -> Any:
    if isinstance(value, ResourceBinding):
        rendered = {
            field.name: dataclass_to_json(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
        rendered["static_ident"] = const_ident(binding_name(value))
        return rendered
    if dataclasses.is_dataclass(value):
        return {field.name: dataclass_to_json(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, list):
        return [dataclass_to_json(item) for item in value]
    if isinstance(value, dict):
        return {key: dataclass_to_json(item) for key, item in value.items()}
    if isinstance(value, bytes):
        return value.hex()
    return value


def render_metadata_json(executes: list[CmdExecute]) -> str:
    document = {
        "schema_version": 1,
        "cmd_executes": [
            {
                "name": execute.name,
                "resources": [
                    {
                        "static_ident": const_ident(binding_name(binding)),
                        "kind": binding.kind,
                        "size": binding.size,
                        "role": binding.role,
                    }
                    for binding in execute.resources
                ],
            }
            for execute in executes
        ],
    }
    return json.dumps(document, indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Regex-match IREE Stream MLIR and emit Rust dispatch flow.")
    parser.add_argument("input", type=Path, help="Input .mlir file")
    parser.add_argument("-o", "--output", type=Path, help="Output file, defaults to stdout")
    parser.add_argument("--format", choices=("rust", "json"), default="rust")
    parser.add_argument("--rust-output", type=Path, help="Write generated Rust to this file")
    parser.add_argument("--json-output", type=Path, help="Write generated metadata JSON to this file")
    args = parser.parse_args(argv)

    if args.output and (args.rust_output or args.json_output):
        parser.error("--output cannot be combined with --rust-output or --json-output")

    try:
        text = args.input.read_text(encoding="utf-8")
        executes, constant_blobs = parse_cmd_executes(text)
        rust_rendered = render_rust(executes, constant_blobs)

        if args.rust_output or args.json_output:
            if args.rust_output:
                args.rust_output.write_text(rust_rendered, encoding="utf-8")
            if args.json_output:
                args.json_output.write_text(render_metadata_json(executes), encoding="utf-8")
            return 0

        rendered = (
            json.dumps(
                {
                    "constants": dataclass_to_json(constant_blobs),
                    "cmd_executes": dataclass_to_json(executes),
                },
                indent=2,
            )
            + "\n"
            if args.format == "json"
            else rust_rendered
        )
        if args.output:
            args.output.write_text(rendered, encoding="utf-8")
        else:
            sys.stdout.write(rendered)
    except (OSError, StreamExtractionError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
