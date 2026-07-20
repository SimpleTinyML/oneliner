import importlib.util
import json
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "iree_stream_flow_to_rust_using_re.py"
SPEC = importlib.util.spec_from_file_location("oneliner_flow_converter", SCRIPT)
CONVERTER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CONVERTER
SPEC.loader.exec_module(CONVERTER)


class FlowConverterTests(unittest.TestCase):
    def test_rust_ident_uses_rust_keywords(self):
        self.assertEqual(CONVERTER.rust_ident("struct"), "struct_")
        self.assertEqual(CONVERTER.rust_ident("gen"), "gen_")
        self.assertEqual(CONVERTER.rust_ident("9-model.value"), "v_9_model_value")

    def test_metadata_contains_rendered_static_identifier(self):
        binding = CONVERTER.ResourceBinding(
            arg="%arg-0",
            source="%source",
            kind="external",
            size_expr="16",
            size=16,
            role="input",
        )

        rendered = CONVERTER.dataclass_to_json(binding)

        self.assertEqual(rendered["static_ident"], "INPUT_ARG_0")
        self.assertIn("Aligned<AlignedType", CONVERTER.render_resource_static(binding, {})[0])

    def test_generated_execute_function_propagates_errors(self):
        dispatch = CONVERTER.DispatchCall(
            kind="dispatch",
            callee="@main",
            executable="main",
            function="main_dispatch_0",
            ordinal=0,
            params=[],
            param_values=[],
            ranges=[],
            workload=(1, 1, 1),
        )
        execute = CONVERTER.CmdExecute(
            name="cmd_execute_0",
            result=None,
            line_no=1,
            resources=[],
            commands=[dispatch],
        )

        rendered = CONVERTER.render_rust([execute], {})

        self.assertIn("-> ::OneLiner::runtime::Result<()>", rendered)
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


if __name__ == "__main__":
    unittest.main()
