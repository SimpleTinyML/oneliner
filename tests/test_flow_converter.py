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
    / "iree_stream_flow_to_rust_using_re.py"
)
SPEC = importlib.util.spec_from_file_location("oneliner_flow_converter", SCRIPT)
CONVERTER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CONVERTER
SPEC.loader.exec_module(CONVERTER)


class FlowConverterTests(unittest.TestCase):
    def test_rust_ident_uses_rust_keywords(self):
        self.assertEqual(CONVERTER.rust_ident("struct"), "struct_")
        self.assertEqual(CONVERTER.rust_ident("gen"), "gen_")
        self.assertEqual(CONVERTER.rust_ident("9-model.value"), "v_9_model_value")

    def test_metadata_contains_rendered_workspace_identifier(self):
        binding = CONVERTER.ResourceBinding(
            arg="%arg-0",
            source="%source",
            kind="transient",
            size_expr="16",
            size=16,
            role="temporary",
        )

        rendered = CONVERTER.dataclass_to_json(binding)

        self.assertEqual(rendered["static_ident"], "TEMP_ARG_0")
        self.assertIn("Aligned<AlignedType", CONVERTER.render_workspace_field(binding))

    def test_generated_execute_function_propagates_errors(self):
        binding = CONVERTER.ResourceBinding(
            arg="%arg0",
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
                    arg="%arg0",
                    kind="transient",
                    tensor_name="temp_arg0",
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
        self.assertIn("(*workspace.TEMP_ARG0).to_buffer_mut()", rendered)
        self.assertNotIn("static mut", rendered)
        self.assertIn("dispatch_fn_from_library(QUERY_FN_PTR, 0)?", rendered)
        self.assertIn("])?;", rendered)

        metadata = json.loads(CONVERTER.render_metadata_json([execute]))
        self.assertEqual(metadata["schema_version"], 1)
        self.assertNotIn("commands", metadata["cmd_executes"][0])

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

    def test_composite_constant_includes_dense_resources(self):
        text = """
            #weights = #util.composite<12xi8, [
                dense_resource<model_weights> : tensor<2xf32>,
                dense<0> : vector<4xi8>,
            ]>
            {-#
              dialect_resources: {
                builtin: {
                  model_weights: "0x040000000000803F000000C0"
                }
              }
            #-}
        """

        constants = CONVERTER.parse_composite_constants(text)

        self.assertEqual(len(constants), 1)
        blob = next(iter(constants.values()))
        self.assertEqual(blob.size, 12)
        self.assertEqual(blob.data, struct.pack("<ff", 1.0, -2.0) + bytes(4))


if __name__ == "__main__":
    unittest.main()
