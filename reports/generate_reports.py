# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Generate the sample diagnostic reports and capture test output.

Run with: python reports/generate_reports.py
Writes the three scenario reports plus the inference test-run log into reports/.
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import onnx  # noqa: E402
from onnx import TensorProto, helper  # noqa: E402
from onnx.diagnostic import (  # noqa: E402
    infer_shapes_with_diagnostics,
    render_report,
)
from onnx.parser import parse_graph  # noqa: E402

REPORTS = Path(__file__).resolve().parent


def if_conflict_model() -> onnx.ModelProto:
    graph = parse_graph("""
        agraph (bool[1] cond, float[1] current_value, float[1] add_value, float[5] sub_value) => (if_output)
        {
            if_output = If (cond) <
                then_branch = then_subgraph () => (then_output) { then_output = Add(current_value, add_value) },
                else_branch = else_subgraph () => (else_output) { else_output = Sub(current_value, sub_value) }
            >
        }
    """)
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 21)], producer_name="report"
    )


def custom_op_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [4, 8])
    custom = helper.make_node("MyCustomOp", ["X"], ["mid"], domain="com.example.custom")
    relu = helper.make_node("Relu", ["mid"], ["Y"])
    downstream = helper.make_node("Relu", ["Y"], ["Z"])
    graph = helper.make_graph(
        [custom, relu, downstream],
        "g",
        [x],
        [helper.make_empty_tensor_value_info("Z")],
    )
    return helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 21),
            helper.make_opsetid("com.example.custom", 1),
        ],
        producer_name="report",
    )


def loop_model() -> onnx.ModelProto:
    graph = parse_graph("""
        agraph (int64 max_trip, bool keep, float[2,3] state_in) => (state_out)
        {
            state_out = Loop (max_trip, keep, state_in) <
                body = loop_body (int64 i, bool cond_in, float[2,3] state)
                    => (bool cond_out, float[2,3] state_next) {
                    updated = Add(state, state)
                    cond_out = Identity(cond_in)
                    state_next = Identity(updated)
                }
            >
        }
    """)
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 21)], producer_name="report"
    )


SCENARIOS = (
    ("branch-conflict", if_conflict_model),
    ("custom-op", custom_op_model),
    ("control-flow-loop", loop_model),
)


def main() -> None:
    for name, builder in SCENARIOS:
        model = builder()
        result = infer_shapes_with_diagnostics(model)
        text = render_report(result.report)
        gate_line = (
            f"Gate: passed={result.gate.passed} exit_code={result.gate.exit_code} "
            f"reasons={result.gate.reasons}\n"
        )
        (REPORTS / f"{name}.txt").write_text(gate_line + text, encoding="utf-8")
        print(f"wrote reports/{name}.txt")  # noqa: T201

    # Capture the real test-suite output for the acceptance record.
    buf = io.StringIO()
    args = [
        "tests/python/diagnostic_test.py",
        "tests/python/shape_inference_test.py",
        "-q",
    ]
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        exit_code = pytest.main(args)
    log = f"pytest {' '.join(args)}\nexit_code={exit_code}\n\n{buf.getvalue()}"
    (REPORTS / "test-run.txt").write_text(log, encoding="utf-8")
    print(f"wrote reports/test-run.txt (pytest exit {exit_code})")  # noqa: T201


if __name__ == "__main__":
    main()
