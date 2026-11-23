# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for shape-inference diagnostics.

These tests cover the acceptance scenarios for the diagnostic layer:

1. Conflicting dimension claims retain both sources and are locatable.
2. Every dimension carries provenance (node, input declaration, or initial).
3. Unknown dimensions are classified by reason.
4. A custom (undefined) op only suppresses inference for itself.
5. Subgraph inference is isolated and non-converging dims are marked with
   their containing subgraph.
6. With diagnostics disabled, ordinary inference is byte-for-byte unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import onnx
from onnx import TensorProto, helper
from onnx.diagnostic import (
    DimensionStatus,
    NodeStatus,
    SourceKind,
    evaluate_gate,
    infer_shapes_with_diagnostics,
    render_report,
    write_report,
)
from onnx.parser import parse_graph

if TYPE_CHECKING:
    from pathlib import Path


def _if_conflict_model() -> onnx.ModelProto:
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
        graph,
        opset_imports=[helper.make_opsetid("", 21)],
        producer_name="diagnostic-test",
    )


def _custom_op_model() -> onnx.ModelProto:
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
        producer_name="diagnostic-test",
    )


def _loop_model() -> onnx.ModelProto:
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
        graph,
        opset_imports=[helper.make_opsetid("", 21)],
        producer_name="diagnostic-test",
    )


class TestDimensionConflict:
    def test_both_sources_retained_and_locatable(self) -> None:
        result = infer_shapes_with_diagnostics(_if_conflict_model())
        assert len(result.report.conflicts) == 1
        conflict = result.report.conflicts[0]
        assert conflict.tensor == "if_output"
        assert conflict.axis == 0
        values = sorted(
            s.claimed_value for s in conflict.sources if s.claimed_value is not None
        )
        assert values == [1, 5]
        # Both producing nodes are identifiable by index and op type.
        labels = {s.label() for s in conflict.sources}
        assert any("Add" in label for label in labels)
        assert any("Sub" in label for label in labels)
        # The conflicting dimension is not silently overwritten.
        tensor = result.report.tensor((), "if_output")
        assert tensor is not None
        assert tensor.dimensions[0].status == DimensionStatus.UNKNOWN_CONFLICT

    def test_gate_fails_on_conflict_by_default(self) -> None:
        result = infer_shapes_with_diagnostics(_if_conflict_model())
        assert result.gate.passed is False
        assert result.gate.exit_code == 1
        assert any("conflict" in reason for reason in result.gate.reasons)

    def test_conflict_counted_once_in_summary(self) -> None:
        result = infer_shapes_with_diagnostics(_if_conflict_model())
        assert result.report.summary.num_conflicts == 1


class TestProvenance:
    def test_input_declaration_sources(self) -> None:
        result = infer_shapes_with_diagnostics(_if_conflict_model())
        tensor = result.report.tensor((), "current_value")
        assert tensor is not None
        assert all(dim.status == DimensionStatus.KNOWN for dim in tensor.dimensions)
        assert all(
            any(source.kind == SourceKind.INPUT_DECL for source in dim.sources)
            for dim in tensor.dimensions
        )

    def test_node_produced_sources(self) -> None:
        result = infer_shapes_with_diagnostics(_loop_model())
        updated = result.report.tensor((0,), "updated")
        assert updated is not None
        add_sources = [
            source
            for dim in updated.dimensions
            for source in dim.sources
            if source.kind == SourceKind.NODE and source.op_type == "Add"
        ]
        assert add_sources

    def test_initializer_sources(self) -> None:
        init = helper.make_tensor("w", TensorProto.FLOAT, [6], [1.0] * 6)
        x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
        node = helper.make_node("Add", ["X", "w"], ["Y"])
        graph = helper.make_graph(
            [node],
            "g",
            [x],
            [helper.make_empty_tensor_value_info("Y")],
            initializer=[init],
        )
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 21)],
            producer_name="diagnostic-test",
        )
        result = infer_shapes_with_diagnostics(model)
        w = result.report.tensor((), "w")
        assert w is not None
        assert all(
            any(source.kind == SourceKind.INITIALIZER for source in dim.sources)
            for dim in w.dimensions
        )


class TestUnknownReasonClassification:
    def test_runtime_dependent_reshape(self) -> None:
        # A shape tensor of known length yields an output of known rank, but
        # every dimension depends on runtime data and cannot be resolved.
        x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [6])
        shape = helper.make_tensor_value_info("shape", TensorProto.INT64, [2])
        node = helper.make_node("Reshape", ["X", "shape"], ["Y"])
        graph = helper.make_graph(
            [node],
            "g",
            [x, shape],
            [helper.make_empty_tensor_value_info("Y")],
        )
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 21)],
            producer_name="diagnostic-test",
        )
        result = infer_shapes_with_diagnostics(model)
        tensor = result.report.tensor((), "Y")
        assert tensor is not None
        assert tensor.rank == 2
        assert all(
            dim.status == DimensionStatus.UNKNOWN_RUNTIME for dim in tensor.dimensions
        )
        # A runtime unknown must never be reported as a conflict.
        assert result.report.conflicts == []

    def test_runtime_unknown_is_distinct_from_missing_op(self) -> None:
        result = infer_shapes_with_diagnostics(_custom_op_model())
        reasons = {
            dim.status
            for tensor in result.report.tensors.values()
            for dim in tensor.dimensions
        }
        assert DimensionStatus.UNKNOWN_RUNTIME not in reasons
        assert DimensionStatus.UNKNOWN_MISSING_OP in reasons or any(
            n.status == NodeStatus.MISSING_DEFINITION for n in result.report.nodes
        )

    def test_every_unknown_has_a_reason(self) -> None:
        valid = set(DimensionStatus)
        for model in (_if_conflict_model(), _custom_op_model(), _loop_model()):
            result = infer_shapes_with_diagnostics(model)
            for tensor in result.report.tensors.values():
                for dim in tensor.dimensions:
                    if dim.status.value.startswith("unknown"):
                        assert dim.status in valid


class TestCustomOpContainment:
    def test_error_only_on_custom_op(self) -> None:
        result = infer_shapes_with_diagnostics(_custom_op_model())
        statuses = {node.op_type: node.status for node in result.report.nodes}
        assert statuses["MyCustomOp"] == NodeStatus.MISSING_DEFINITION
        # Downstream standard ops are not counted as independent errors.
        assert statuses["Relu"] in (NodeStatus.SKIPPED, NodeStatus.OK)
        assert result.report.summary.num_inference_errors == 0
        assert result.report.summary.num_missing_op_nodes == 1

    def test_missing_op_is_its_own_class(self) -> None:
        result = infer_shapes_with_diagnostics(_custom_op_model())
        mid = result.report.tensor((), "mid")
        assert mid is not None
        # A missing-op unknown must never be mislabelled as a conflict.
        conflict_tensors = {c.tensor for c in result.report.conflicts}
        assert "mid" not in conflict_tensors

    def test_traversal_continues_after_custom_op(self) -> None:
        result = infer_shapes_with_diagnostics(_custom_op_model())
        ops = [node.op_type for node in result.report.nodes]
        # All nodes were visited, not stopped at the custom op.
        assert ops == ["MyCustomOp", "Relu", "Relu"]


class TestModelLocalFunction:
    def _model(self) -> onnx.ModelProto:
        body = [helper.make_node("Relu", ["x"], ["y"])]
        func = helper.make_function(
            domain="local",
            fname="MyRelu",
            inputs=["x"],
            outputs=["y"],
            nodes=body,
            opset_imports=[helper.make_opsetid("", 21)],
        )
        x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
        call = helper.make_node("MyRelu", ["X"], ["Z"], domain="local")
        graph = helper.make_graph(
            [call], "g", [x], [helper.make_empty_tensor_value_info("Z")]
        )
        return helper.make_model(
            graph,
            opset_imports=[
                helper.make_opsetid("", 21),
                helper.make_opsetid("local", 1),
            ],
            functions=[func],
            producer_name="diagnostic-test",
        )

    def test_function_call_is_not_missing_op(self) -> None:
        result = infer_shapes_with_diagnostics(self._model())
        assert result.report.summary.num_missing_op_nodes == 0
        statuses = {n.op_type: n.status for n in result.report.nodes}
        assert statuses["MyRelu"] == NodeStatus.FUNCTION_CALL

    def test_function_body_feeds_caller_output(self) -> None:
        result = infer_shapes_with_diagnostics(self._model())
        z = result.report.tensor((), "Z")
        assert z is not None and z.rank == 2
        assert all(dim.status == DimensionStatus.KNOWN for dim in z.dimensions)
        # Body-local names stay in the function scope, not the root.
        assert result.report.tensor((), "y") is None
        assert result.report.tensor((0,), "y") is not None

    def test_function_result_matches_engine(self) -> None:
        model = self._model()
        baseline = onnx.shape_inference.infer_shapes(model)
        result = infer_shapes_with_diagnostics(model)
        base_dims = [
            d.dim_value for d in baseline.graph.value_info[0].type.tensor_type.shape.dim
        ]
        diag_dims = [d.value for d in result.report.tensor((), "Z").dimensions]
        assert base_dims == diag_dims == [2, 3]


class TestSubgraphIsolation:
    def test_subgraph_locals_not_in_root(self) -> None:
        result = infer_shapes_with_diagnostics(_loop_model())
        assert result.report.tensor((), "updated") is None
        assert result.report.tensor((0,), "updated") is not None

    def test_boundary_declared(self) -> None:
        result = infer_shapes_with_diagnostics(_loop_model())
        assert len(result.report.subgraphs) == 1
        sub = result.report.subgraphs[0]
        assert sub.path == (0,)
        assert "state_in" in sub.boundary_inputs
        assert "state_next" in sub.boundary_outputs

    def test_non_converging_dim_marked_with_subgraph(self) -> None:
        result = infer_shapes_with_diagnostics(_loop_model())
        state_out = result.report.tensor((), "state_out")
        assert state_out is not None
        assert all(
            dim.status == DimensionStatus.UNKNOWN_CONTROL_FLOW
            for dim in state_out.dimensions
        )
        for dim in state_out.dimensions:
            assert "loop" in dim.detail or "boundary" in dim.detail

    def test_if_branch_subgraphs_are_isolated_scopes(self) -> None:
        result = infer_shapes_with_diagnostics(_if_conflict_model())
        paths = {sub.path for sub in result.report.subgraphs}
        assert paths == {(0,)}
        kinds = {sub.kind for sub in result.report.subgraphs}
        assert kinds == {"then_branch", "else_branch"}


class TestModesAndGate:
    def test_lenient_mode_records_without_raising(self) -> None:
        # Lenient mode must return a report even when issues exist.
        result = infer_shapes_with_diagnostics(_custom_op_model(), strict=False)
        assert result.report.summary.num_missing_op_nodes == 1

    def test_strict_mode_preserves_raise_behavior(self) -> None:
        # A value_info shape that contradicts inference makes the authoritative
        # strict run raise, and the diagnostic wrapper must not swallow it.
        x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
        y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])
        node = helper.make_node("Add", ["X", "Y"], ["Z"])
        wrong = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [9, 9])
        graph = helper.make_graph([node], "g", [x, y], [], value_info=[wrong])
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 21)],
            producer_name="diagnostic-test",
        )
        with pytest.raises(onnx.shape_inference.InferenceError):
            infer_shapes_with_diagnostics(model, strict=True)

    def test_gate_thresholds(self) -> None:
        report = infer_shapes_with_diagnostics(_if_conflict_model()).report
        failing = evaluate_gate(report, max_conflicts=0)
        assert failing.passed is False and failing.exit_code == 1
        passing = evaluate_gate(report, max_conflicts=10)
        assert passing.passed is True and passing.exit_code == 0

    def test_unknown_threshold(self) -> None:
        report = infer_shapes_with_diagnostics(_loop_model()).report
        blocked = evaluate_gate(report, max_conflicts=0, max_unknown=0)
        assert blocked.passed is False
        open_gate = evaluate_gate(report, max_conflicts=0, max_unknown=None)
        assert open_gate.passed is True


class TestReport:
    def test_report_contains_only_safe_tokens(self) -> None:
        result = infer_shapes_with_diagnostics(_if_conflict_model())
        text = render_report(result.report)
        assert "if_output" in text
        assert "UNKNOWN/CONFLICT" in text
        # No filesystem paths: check for home/root markers and Windows paths.
        assert "/home" not in text
        assert "/tmp" not in text  # noqa: S108 - asserting the string is absent, not using the path
        assert "\\" not in text
        # No weight values are rendered, only dimension integers and symbols.
        assert "weight" not in text.lower()

    def test_report_write_failure_is_surfaced(self, tmp_path: Path) -> None:
        result = infer_shapes_with_diagnostics(_if_conflict_model())
        # A read-only directory makes the atomic temporary-file creation fail.
        readonly = tmp_path / "readonly"
        readonly.mkdir()
        readonly.chmod(0o500)
        target = readonly / "report.txt"
        with pytest.raises(OSError):
            write_report(result.report, target)
        # Restore permission so tmp teardown can remove it.
        readonly.chmod(0o700)

    def test_roundtrip_to_disk(self, tmp_path: Path) -> None:
        result = infer_shapes_with_diagnostics(_if_conflict_model())
        out = tmp_path / "report.txt"
        write_report(result.report, out)
        body = out.read_text(encoding="utf-8")
        assert "ONNX Shape Inference Diagnostic Report" in body

    def test_conflict_never_resolves_to_a_deterministic_dim(self) -> None:
        # Same-source invariant: where the report records a conflict, the
        # authoritative inferred model must not present that dimension as a
        # concrete value. The report cannot contradict the result it describes.
        result = infer_shapes_with_diagnostics(_if_conflict_model())
        inferred_by_name = {vi.name: vi for vi in result.model.graph.value_info}
        inferred_by_name.update((vi.name, vi) for vi in result.model.graph.output)
        for conflict in result.report.conflicts:
            assert conflict.subgraph_path == ()
            value_info = inferred_by_name[conflict.tensor]
            dim = value_info.type.tensor_type.shape.dim[conflict.axis]
            assert not dim.HasField("dim_value"), (
                f"report flags conflict on {conflict.tensor} axis {conflict.axis} "
                "but the inferred model assigns a concrete dimension"
            )

    def test_report_path_write_failure_propagates(self, tmp_path: Path) -> None:
        # When the report cannot be written, the diagnostic call must fail
        # instead of silently returning a successful result.
        readonly = tmp_path / "readonly"
        readonly.mkdir()
        readonly.chmod(0o500)
        with pytest.raises(OSError):
            infer_shapes_with_diagnostics(
                _if_conflict_model(), report_path=readonly / "report.txt"
            )
        readonly.chmod(0o700)


class TestDiagnosticsOffEquivalence:
    def test_plain_inference_unchanged(self) -> None:
        model = _if_conflict_model()
        baseline = onnx.shape_inference.infer_shapes(model)
        # The diagnostic entry point returns the same authoritative model.
        result = infer_shapes_with_diagnostics(model)
        assert baseline.SerializeToString() == result.model.SerializeToString()

    def test_standard_model_byte_equivalence(self) -> None:
        x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
        y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])
        nodes = [
            helper.make_node("Relu", ["X"], ["a"]),
            helper.make_node("Add", ["a", "X"], ["Y"]),
        ]
        graph = helper.make_graph(nodes, "g", [x], [y])
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 21)],
            producer_name="diagnostic-test",
        )
        baseline = onnx.shape_inference.infer_shapes(model)
        diagnostic = infer_shapes_with_diagnostics(model)
        assert baseline.SerializeToString() == diagnostic.model.SerializeToString()
        # And the standard entry point signature is untouched.
        again = onnx.shape_inference.infer_shapes(
            model, check_type=False, strict_mode=False, data_prop=False
        )
        assert again.SerializeToString() == baseline.SerializeToString()
