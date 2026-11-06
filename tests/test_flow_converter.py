import importlib.util
import json
import struct
import sys
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "oneliner-macro"
    / "python"
    / "iree_stream_flow_to_rust.py"
)
FIXTURE = Path(__file__).parent / "fixtures" / "abs2.10.executable-targets.mlir"
DISPATCH = """      stream.cmd.dispatch @main_dispatch_0::@static::@main_dispatch_0_elementwise_2_f32 {
        ro %arg1[%c0 for %c8] : !stream.resource<external>{%c8},
        wo %arg2[%c0 for %c8] : !stream.resource<external>{%c8}
      }"""
SPEC = importlib.util.spec_from_file_location("oneliner_flow_converter", SCRIPT)
CONVERTER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CONVERTER
SPEC.loader.exec_module(CONVERTER)


class FlowConverterTests(unittest.TestCase):
    def test_structured_parser_resolves_exports_and_ordinals_in_named_module(self):
        text = FIXTURE.read_text(encoding="utf-8").replace(
            "module attributes", "module @container attributes", 1
        )
        executable_start = text.index("  hal.executable private")
        executable_end = text.index("  util.func public", executable_start)
        executable = text[executable_start:executable_end]
        second_executable = executable.replace("main_dispatch_0", "main_dispatch_1")
        second_dispatch = DISPATCH.replace("main_dispatch_0", "main_dispatch_1")
        text = text[:executable_end] + second_executable + text[executable_end:]
        text = text.replace(DISPATCH, DISPATCH + "\n" + second_dispatch, 1)

        executes, _ = CONVERTER.parse_cmd_executes(text)

        self.assertEqual(
            [command.ordinal for command in executes[0].commands], [0, 1]
        )

    def test_structured_parser_tracks_external_resource_roles(self):
        # Read-only access to an external buffer does not change its logical role.
        text = FIXTURE.read_text(encoding="utf-8").replace(
            "wo %arg2[%c0 for %c8]", "ro %arg2[%c0 for %c8]", 1
        )
        executes, _ = CONVERTER.parse_cmd_executes(text)
        self.assertEqual(executes[0].resources[0].role, "input")
        self.assertEqual(executes[0].resources[1].role, "output")

        # Provenance forwarded through cf.br must still be classified as output.
        allocation = """    %result, %result_timepoint = stream.resource.alloca uninitialized on(#hal.device.affinity<@__device_0>) : !stream.resource<external>{%c8} => !stream.timepoint
    %1 = stream.cmd.execute"""
        forwarded = """    %result, %result_timepoint = stream.resource.alloca uninitialized on(#hal.device.affinity<@__device_0>) : !stream.resource<external>{%c8} => !stream.timepoint
    cf.br ^bb1(%result : !stream.resource<external>)
  ^bb1(%forwarded: !stream.resource<external>):
    %1 = stream.cmd.execute"""
        forwarded_text = (
            FIXTURE.read_text(encoding="utf-8")
            .replace(allocation, forwarded, 1)
            .replace("%result as %arg2", "%forwarded as %arg2", 1)
            .replace("wo %arg2[%c0 for %c8]", "ro %arg2[%c0 for %c8]", 1)
        )
        executes, _ = CONVERTER.parse_cmd_executes(forwarded_text)
        self.assertEqual(executes[0].resources[1].role, "output")

    def test_structured_parser_rejects_non_contiguous_local_ordinals(self):
        text = FIXTURE.read_text(encoding="utf-8").replace(
            "ordinal(0)", "ordinal(7)", 1
        )

        with self.assertRaisesRegex(
            CONVERTER.StreamExtractionError, "non-contiguous or duplicate ordinals"
        ):
            CONVERTER.parse_cmd_executes(text)

    def test_structured_parser_rejects_unknown_commands(self):
        text = FIXTURE.read_text(encoding="utf-8")
        copy = (
            '      "stream.cmd.copy"(%arg1, %c8, %c0, %arg2, %c8, %c0, %c8) '
            ": (!stream.resource<external>, index, index, "
            "!stream.resource<external>, index, index, index) -> ()"
        )
        self.assertIn(DISPATCH, text)

        with self.assertRaisesRegex(
            CONVERTER.StreamExtractionError,
            "unsupported command operation stream.cmd.copy",
        ):
            CONVERTER.parse_cmd_executes(text.replace(DISPATCH, copy, 1))

    def test_structured_parser_parses_concurrent_fill(self):
        text = FIXTURE.read_text(encoding="utf-8").replace(
            "    %c0 = arith.constant 0 : index",
            "    %c0 = arith.constant 0 : index\n"
            "    %c0_i8 = arith.constant 0 : i8",
            1,
        )
        concurrent_fill = """      stream.cmd.concurrent {
        "stream.cmd.fill"(%arg2, %c8, %c0, %c8, %c0_i8) : (!stream.resource<external>, index, index, index, i8) -> ()
      }"""

        executes, _ = CONVERTER.parse_cmd_executes(
            text.replace(DISPATCH, concurrent_fill, 1)
        )

        concurrent = executes[0].commands[0]
        self.assertIsInstance(concurrent, CONVERTER.ConcurrentCommand)
        self.assertIsInstance(concurrent.commands[0], CONVERTER.FillCommand)
        self.assertEqual(concurrent.commands[0].target.tensor_name, "output_arg2")

    def test_structured_parser_rejects_dynamic_ranges(self):
        text = (
            FIXTURE.read_text(encoding="utf-8")
            .replace(
                "@main(%arg0: !hal.buffer_view)",
                "@main(%arg0: !hal.buffer_view, %dynamic: index)",
                1,
            )
            .replace("ro %arg1[%c0 for %c8]", "ro %arg1[%c0 for %dynamic]", 1)
        )

        with self.assertRaisesRegex(
            CONVERTER.StreamExtractionError, "dynamic resource range"
        ):
            CONVERTER.parse_cmd_executes(text)

    def test_rust_ident_uses_rust_keywords(self):
        self.assertEqual(CONVERTER.rust_ident("struct"), "struct_")
        self.assertEqual(CONVERTER.rust_ident("gen"), "gen_")
        self.assertEqual(CONVERTER.rust_ident("9-model.value"), "v_9_model_value")

    def test_renders_workspace_and_metadata_contract(self):
        binding = CONVERTER.ResourceBinding(
            arg="%arg-0",
            source="%source",
            kind="transient",
            size_expr="16",
            size=16,
            role="temporary",
        )
        dispatch = CONVERTER.DispatchCall(
            kind="dispatch",
            callee="@main",
            executable="main",
            function="main_dispatch_0",
            ordinal=0,
            params=[],
            param_values=[],
            ranges=[
                CONVERTER.TensorRange(
                    access="rw",
                    arg="%arg-0",
                    kind="transient",
                    tensor_name="temp_arg_0",
                    offset_expr="0",
                    offset=0,
                    length_expr="16",
                    length=16,
                )
            ],
            workload=(1, 1, 1),
        )
        execute = CONVERTER.CmdExecute(
            name="cmd_execute_0",
            result=None,
            line_no=1,
            resources=[binding],
            commands=[dispatch],
        )

        rendered = CONVERTER.render_rust([execute], {})

        self.assertIn("-> Result<(), Error>", rendered)
        self.assertIn("pub struct Workspace", rendered)
        self.assertIn("workspace: &mut Workspace", rendered)
        self.assertIn("(*workspace.TEMP_ARG_0).to_buffer_mut()", rendered)
        self.assertNotIn("static mut", rendered)
        self.assertIn("dispatch_fn_from_library(QUERY_FN_PTR, 0)?", rendered)
        self.assertIn("])?;", rendered)

        metadata = json.loads(CONVERTER.render_metadata_json([execute]))
        self.assertEqual(metadata["schema_version"], 1)
        self.assertNotIn("commands", metadata["cmd_executes"][0])
        self.assertEqual(
            metadata["cmd_executes"][0]["resources"][0]["static_ident"],
            "TEMP_ARG_0",
        )

        # The rendered Rust and the metadata must agree on the workspace name.
        self.assertEqual(CONVERTER.dataclass_to_json(binding)["static_ident"], "TEMP_ARG_0")
        workspace_field = CONVERTER.render_workspace_field(binding)
        self.assertIn("TEMP_ARG_0", workspace_field)
        self.assertIn("Aligned<AlignedType", workspace_field)

    def test_unresolved_dispatch_is_an_error(self):
        dispatch = CONVERTER.DispatchCall(
            kind="dispatch",
            callee="@main",
            executable="main",
            function="main_dispatch_0",
            ordinal=0,
            params=["%unknown"],
            param_values=[None],
            ranges=[],
            workload=(1, 1, 1),
        )

        with self.assertRaises(CONVERTER.StreamExtractionError):
            CONVERTER.render_command(dispatch, "")

    def test_parses_composite_constants_and_provenance(self):
        text = """#a = #util.composite<1xi8, [dense<1> : tensor<1xi8>]>
#b = #util.composite<1xi8, [dense<2> : tensor<1xi8>]>
#weights = #util.composite<12xi8, [
  dense_resource<model_weights> : tensor<2xf32>,
  dense<0> : vector<4xi8>,
]>
module {
  util.func private @constants() {
    %a = util.buffer.constant : !util.buffer = #a
    %b = util.buffer.constant : !util.buffer = #b
    %weights = util.buffer.constant : !util.buffer = #weights
    util.return
  }
}
{-#
  dialect_resources: {
    builtin: {
      model_weights: "0x040000000000803F000000C0"
    }
  }
#-}
"""

        parser = CONVERTER.StructuredStreamParser(text)
        buffer_constants = CONVERTER.ir.get_ops_of_type(
            parser.module, CONVERTER.util.BufferConstantOp
        )

        self.assertEqual(
            [parser.constant_by_value[operation.result] for operation in buffer_constants],
            ["constant_a", "constant_b", "constant_weights"],
        )
        self.assertEqual(
            sorted(parser.constant_blobs),
            ["constant_a", "constant_b", "constant_weights"],
        )
        weights = parser.constant_blobs["constant_weights"]
        self.assertEqual(weights.size, 12)
        self.assertEqual(weights.data, struct.pack("<ff", 1.0, -2.0) + bytes(4))

    def test_constant_provenance_requires_all_cf_inputs(self):
        text = """#a = #util.composite<1xi8, [dense<1> : tensor<1xi8>]>
module {
  util.func private @constants(%condition: i1, %unknown: !util.buffer) {
    %a = util.buffer.constant : !util.buffer = #a
    cf.cond_br %condition, ^bb1(%a : !util.buffer), ^bb1(%unknown : !util.buffer)
  ^bb1(%joined: !util.buffer):
    util.return
  }
}
"""

        parser = CONVERTER.StructuredStreamParser(text)
        joined = next(iter(parser.block_argument_sources))

        self.assertNotIn(joined, parser.constant_by_value)


if __name__ == "__main__":
    unittest.main()
