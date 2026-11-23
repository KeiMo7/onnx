# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import tempfile
import time
import unittest

import numpy as np
from onnx.onnx_pb import GraphProto, ModelProto

import onnx
from onnx import helper, numpy_helper
from onnx.graph_diff import (
    ComparisonStatus,
    compare_graphs,
    compare_models,
    compare_models_batch,
)
from onnx.graph_diff import (
    main as graph_diff_main,
)


def _value_names(graph: GraphProto) -> set[str]:
    names = {vi.name for vi in graph.input}
    names.update(vi.name for vi in graph.output)
    names.update(vi.name for vi in graph.value_info)
    names.update(t.name for t in graph.initializer)
    for node in graph.node:
        names.update(n for n in node.input if n)
        names.update(n for n in node.output if n)
    return names


def _rename_model(model: ModelProto, prefix: str) -> ModelProto:
    """Rename every node and tensor name in the main graph."""
    renamed = ModelProto()
    renamed.CopyFrom(model)
    graph = renamed.graph
    mapping = {name: f"{prefix}{name}" for name in sorted(_value_names(graph))}

    def rn(name: str) -> str:
        return mapping.get(name, name)

    for i, node in enumerate(graph.node):
        node.name = f"{prefix}node_{i}"
        del node.input[:]
        node.input.extend(rn(n) if n else "" for n in _node_inputs(model.graph, i))
        del node.output[:]
        node.output.extend(rn(n) for n in _node_outputs(model.graph, i))
    for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
        vi.name = rn(vi.name)
    for init in graph.initializer:
        init.name = rn(init.name)
    return renamed


def _node_inputs(graph: GraphProto, index: int) -> list[str]:
    return list(graph.node[index].input)


def _node_outputs(graph: GraphProto, index: int) -> list[str]:
    return list(graph.node[index].output)


def _make_conv_relu_model(
    weight: np.ndarray | None = None,
    *,
    include_relu: bool = True,
) -> ModelProto:
    if weight is None:
        weight = np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2) / 10.0
    w = numpy_helper.from_array(weight, name="W")
    x = helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 1, 3, 3])
    y = helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, None)
    conv = helper.make_node("Conv", ["x", "W"], ["c"], name="conv")
    nodes = [conv]
    if include_relu:
        nodes.append(helper.make_node("Relu", ["c"], ["y"], name="relu"))
        out_name = "y"
    else:
        out_name = "c"
    y.name = out_name
    graph = helper.make_graph(nodes, "g", [x], [y], [w])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


def _make_two_branch_model(*, swapped: bool) -> ModelProto:
    x1 = helper.make_tensor_value_info("x1", onnx.TensorProto.FLOAT, [2])
    x2 = helper.make_tensor_value_info("x2", onnx.TensorProto.FLOAT, [2])
    y = helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [2])
    neg = helper.make_node("Neg", ["x1"], ["a"], name="neg")
    exp = helper.make_node("Exp", ["x2"], ["b"], name="exp")
    add = helper.make_node("Add", ["a", "b"], ["y"], name="add")
    nodes = [exp, neg, add] if swapped else [neg, exp, add]
    graph = helper.make_graph(nodes, "g", [x1, x2], [y])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


def _make_gemm_model(*, alpha: float | None, attr_order: int = 0) -> ModelProto:
    x = helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [2, 2])
    y = helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [2, 2])
    a = numpy_helper.from_array(np.eye(2, dtype=np.float32), name="A")
    b = numpy_helper.from_array(np.eye(2, dtype=np.float32), name="B")
    attrs = {}
    if alpha is not None:
        attrs["alpha"] = alpha
    attrs["beta"] = 2.0
    attrs["transA"] = 1
    node = helper.make_node("Gemm", ["x", "A", "B"], ["y"], name="gemm", **attrs)
    if attr_order:
        # Shuffle the attribute declaration order.
        shuffled = list(node.attribute)[::-1]
        del node.attribute[:]
        node.attribute.extend(shuffled)
    graph = helper.make_graph([node], "g", [x], [y], [a, b])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


def _make_if_model(*, then_op: str = "Add") -> ModelProto:
    cond = helper.make_tensor_value_info("cond", onnx.TensorProto.BOOL, [])
    x = helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [2])
    y = helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [2])
    then_graph = helper.make_graph(
        [helper.make_node(then_op, ["x", "x"], ["t"], name="then_add")],
        "then",
        [],
        [helper.make_tensor_value_info("t", onnx.TensorProto.FLOAT, [2])],
    )
    else_graph = helper.make_graph(
        [helper.make_node("Sub", ["x", "x"], ["e"], name="else_sub")],
        "else",
        [],
        [helper.make_tensor_value_info("e", onnx.TensorProto.FLOAT, [2])],
    )
    if_node = helper.make_node(
        "If",
        ["cond"],
        ["y"],
        name="if_node",
        then_branch=then_graph,
        else_branch=else_graph,
    )
    graph = helper.make_graph([if_node], "g", [cond, x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


def _make_chain_model(length: int, prefix: str = "") -> ModelProto:
    x = helper.make_tensor_value_info(f"{prefix}x", onnx.TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info(f"{prefix}y", onnx.TensorProto.FLOAT, [1])
    nodes = []
    prev = f"{prefix}x"
    for i in range(length):
        out = f"{prefix}v{i}" if i < length - 1 else f"{prefix}y"
        nodes.append(helper.make_node("Identity", [prev], [out], name=f"{prefix}n{i}"))
        prev = out
    graph = helper.make_graph(nodes, "g", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


class TestEquivalence(unittest.TestCase):
    def test_identical_models_are_equivalent(self) -> None:
        model = _make_conv_relu_model()
        report = compare_models(model, model)
        self.assertEqual(report.status, ComparisonStatus.EQUIVALENT)
        self.assertTrue(report.equivalent)
        self.assertEqual(report.differences, [])

    def test_rename_and_reorder_are_equivalent(self) -> None:
        model_a = _make_conv_relu_model()
        model_b = _rename_model(model_a, "b_")
        # Shuffle initializer order and metadata.
        model_b.graph.initializer.reverse()
        model_b.producer_name = "some-other-exporter"
        model_b.doc_string = "unrelated documentation"
        model_b.metadata_props.add(key="source", value="elsewhere")
        report = compare_models(model_a, model_b)
        self.assertEqual(report.status, ComparisonStatus.EQUIVALENT)
        self.assertEqual(report.differences, [])

    def test_node_declaration_order_is_equivalent(self) -> None:
        report = compare_models(
            _make_two_branch_model(swapped=False),
            _make_two_branch_model(swapped=True),
        )
        self.assertEqual(report.status, ComparisonStatus.EQUIVALENT)

    def test_attribute_order_and_metadata_are_equivalent(self) -> None:
        model_a = _make_gemm_model(alpha=2.0, attr_order=0)
        model_b = _make_gemm_model(alpha=2.0, attr_order=1)
        model_b.graph.name = "another-name"
        model_b.graph.doc_string = "different docs"
        model_b.ir_version = model_a.ir_version - 1
        report = compare_models(model_a, model_b)
        self.assertEqual(report.status, ComparisonStatus.EQUIVALENT)
        self.assertTrue(any("ir_version" in note for note in report.notes))

    def test_explicit_default_attribute_is_equivalent(self) -> None:
        model_a = _make_gemm_model(alpha=None)  # alpha omitted: default 1.0
        model_b = _make_gemm_model(alpha=1.0)  # alpha explicit default
        report = compare_models(model_a, model_b)
        self.assertEqual(report.status, ComparisonStatus.EQUIVALENT)
        self.assertTrue(
            any("schema default" in note and "alpha" in note for note in report.notes),
            f"expected a note about the explicit default, got: {report.notes}",
        )

    def test_omitted_optional_input_placeholders(self) -> None:
        x = helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [2])
        y = helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [2])
        clip_plain = helper.make_node("Clip", ["x"], ["y"], name="clip")
        clip_placeholders = helper.make_node("Clip", ["x", "", ""], ["y"], name="clip")
        model_a = helper.make_model(
            helper.make_graph([clip_plain], "g", [x], [y]),
            opset_imports=[helper.make_opsetid("", 18)],
        )
        model_b = helper.make_model(
            helper.make_graph([clip_placeholders], "g", [x], [y]),
            opset_imports=[helper.make_opsetid("", 18)],
        )
        report = compare_models(model_a, model_b)
        self.assertEqual(report.status, ComparisonStatus.EQUIVALENT)

    def test_backend_fixture_model_rename(self) -> None:
        fixture = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "onnx",
            "backend",
            "test",
            "data",
            "simple",
            "test_gradient_of_add",
            "model.onnx",
        )
        if not os.path.exists(fixture):
            self.skipTest("backend test fixture not available")
        model_a = onnx.load(fixture)
        model_b = _rename_model(model_a, "z_")
        report = compare_models(model_a, model_b)
        self.assertEqual(report.status, ComparisonStatus.EQUIVALENT)

    def test_subgraph_rename_is_equivalent(self) -> None:
        model_a = _make_if_model()
        model_b = _make_if_model()
        # Rename nodes and values inside both branches of model B.
        if_node = model_b.graph.node[0]
        for attr in if_node.attribute:
            branch = attr.g
            for i, node in enumerate(branch.node):
                node.name = f"renamed_{i}"
                for j, out in enumerate(node.output):
                    node.output[j] = f"renamed_{out}"
            for out in branch.output:
                out.name = f"renamed_{out.name}"
        report = compare_models(model_a, model_b)
        self.assertEqual(report.status, ComparisonStatus.EQUIVALENT)


class TestDifferences(unittest.TestCase):
    def test_removed_node_is_located(self) -> None:
        model_a = _make_conv_relu_model(include_relu=True)
        model_b = _make_conv_relu_model(include_relu=False)
        report = compare_models(model_a, model_b)
        self.assertEqual(report.status, ComparisonStatus.DIFFERENT)
        self.assertFalse(report.equivalent)
        self.assertEqual(report.removed_nodes, ["relu (Relu)"])
        kinds = {d.kind for d in report.differences}
        self.assertIn("node-removed", kinds)
        text = report.to_text()
        self.assertIn("relu", text)

    def test_added_node_is_located(self) -> None:
        model_a = _make_conv_relu_model(include_relu=False)
        model_b = _make_conv_relu_model(include_relu=True)
        report = compare_models(model_a, model_b)
        self.assertEqual(report.status, ComparisonStatus.DIFFERENT)
        self.assertEqual(report.added_nodes, ["relu (Relu)"])
        kinds = {d.kind for d in report.differences}
        self.assertIn("node-added", kinds)

    def test_changed_attribute_is_located(self) -> None:
        model_a = _make_gemm_model(alpha=1.5)
        model_b = _make_gemm_model(alpha=2.5)
        report = compare_models(model_a, model_b)
        self.assertEqual(report.status, ComparisonStatus.DIFFERENT)
        attr_diffs = [d for d in report.differences if d.kind == "attribute"]
        self.assertEqual(len(attr_diffs), 1)
        self.assertIn("alpha", attr_diffs[0].detail)
        self.assertIn("gemm", attr_diffs[0].detail)

    def test_weight_value_change_reports_downstream(self) -> None:
        weight_a = np.full((1, 1, 2, 2), 0.25, dtype=np.float32)
        weight_b = np.full((1, 1, 2, 2), 0.5, dtype=np.float32)
        model_a = _make_conv_relu_model(weight_a)
        model_b = _make_conv_relu_model(weight_b)
        report = compare_models(model_a, model_b)
        self.assertEqual(report.status, ComparisonStatus.DIFFERENT)
        input_diffs = [d for d in report.differences if d.kind == "input"]
        self.assertEqual(len(input_diffs), 1)
        self.assertIn("values differ", input_diffs[0].detail)
        # The relu consumes the conv output and must be flagged as affected.
        self.assertTrue(any("relu" in n for n in report.affected_downstream))
        # Weight values themselves must never appear in the report.
        self.assertNotIn("0.25", report.to_text())
        self.assertNotIn("0.5", report.to_text())

    def test_dtype_change_reports_downstream(self) -> None:
        weight = np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2)
        model_a = _make_conv_relu_model(weight)
        model_b = _make_conv_relu_model(weight.astype(np.float16))
        report = compare_models(model_a, model_b)
        self.assertEqual(report.status, ComparisonStatus.DIFFERENT)
        input_diffs = [d for d in report.differences if d.kind == "input"]
        self.assertEqual(len(input_diffs), 1)
        self.assertIn("data type differs", input_diffs[0].detail)
        self.assertIn("FLOAT16", input_diffs[0].detail)
        self.assertTrue(any("relu" in n for n in report.affected_downstream))

    def test_subgraph_difference_is_located(self) -> None:
        model_a = _make_if_model(then_op="Add")
        model_b = _make_if_model(then_op="Mul")
        report = compare_models(model_a, model_b)
        self.assertEqual(report.status, ComparisonStatus.DIFFERENT)
        locations = {d.location for d in report.differences}
        self.assertTrue(
            any("then_branch" in loc for loc in locations),
            f"expected a difference located in then_branch, got {locations}",
        )

    def test_difference_set_and_verdict_are_consistent(self) -> None:
        model_a = _make_gemm_model(alpha=1.5)
        model_b = _make_gemm_model(alpha=2.5)
        report = compare_models(model_a, model_b)
        # The verdict must be derived from the reported difference set.
        self.assertEqual(
            report.status is ComparisonStatus.DIFFERENT,
            bool(report.differences),
        )


class TestOpsets(unittest.TestCase):
    def _make_unary_model(self, op: str, opset: int) -> ModelProto:
        x = helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [2])
        y = helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [2])
        node = helper.make_node(op, ["x"], ["y"], name="op")
        graph = helper.make_graph([node], "g", [x], [y])
        return helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])

    def test_compatible_opset_difference(self) -> None:
        # Sigmoid's schema signature is identical in opsets 13 and 18.
        report = compare_models(
            self._make_unary_model("Sigmoid", 13),
            self._make_unary_model("Sigmoid", 18),
        )
        self.assertEqual(report.status, ComparisonStatus.COMPATIBLE)
        self.assertTrue(report.compatible)
        self.assertFalse(report.equivalent)
        self.assertEqual(report.differences, [])
        self.assertTrue(any("opset" in note for note in report.notes))

    def test_incompatible_opset_difference(self) -> None:
        # ReduceMean changed its schema between opsets 13 and 18.
        report = compare_models(
            self._make_unary_model("ReduceMean", 13),
            self._make_unary_model("ReduceMean", 18),
        )
        self.assertEqual(report.status, ComparisonStatus.DIFFERENT)
        opset_diffs = [d for d in report.differences if d.kind == "opset"]
        self.assertTrue(opset_diffs, f"expected opset diffs, got {report}")
        detail = opset_diffs[0].detail
        self.assertIn("ReduceMean", detail)
        self.assertIn("13", detail)
        self.assertIn("18", detail)


class TestErrors(unittest.TestCase):
    def test_dangling_input_is_an_error_not_a_difference(self) -> None:
        x = helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [2])
        y = helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [2])
        bad = helper.make_node("Relu", ["ghost"], ["y"], name="broken")
        graph = helper.make_graph([bad], "g", [x], [y])
        model_a = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
        model_b = _make_chain_model(1)
        report = compare_models(model_a, model_b)
        self.assertEqual(report.status, ComparisonStatus.ERROR)
        self.assertEqual(report.differences, [])
        self.assertEqual(report.added_nodes, [])
        self.assertEqual(report.removed_nodes, [])
        self.assertTrue(report.errors)
        self.assertTrue(any("ghost" in e for e in report.errors))
        self.assertTrue(any("broken" in e for e in report.errors))

    def test_unparseable_bytes_are_an_error(self) -> None:
        report = compare_models(b"\x00\x01\x02garbage", _make_chain_model(1))
        self.assertEqual(report.status, ComparisonStatus.ERROR)
        self.assertEqual(report.differences, [])

    def test_compare_graphs_entry_point(self) -> None:
        model = _make_chain_model(3)
        report = compare_graphs(model.graph, model.graph, opset_imports={"": 18})
        self.assertEqual(report.status, ComparisonStatus.EQUIVALENT)


class TestBatchAndPerformance(unittest.TestCase):
    def test_batch_matches_serial(self) -> None:
        base = _make_conv_relu_model()
        renamed = _rename_model(base, "w_")
        changed_attr = _make_gemm_model(alpha=3.0)
        pairs = [
            (base, renamed),
            (base, _make_conv_relu_model(include_relu=False)),
            (_make_gemm_model(alpha=1.5), changed_attr),
            (
                _make_two_branch_model(swapped=False),
                _make_two_branch_model(swapped=True),
            ),
            (_make_if_model(), _make_if_model(then_op="Mul")),
            (_make_chain_model(20), _rename_model(_make_chain_model(20), "q_")),
        ]
        start = time.perf_counter()
        serial = compare_models_batch(pairs, max_workers=1)
        serial_time = time.perf_counter() - start
        start = time.perf_counter()
        parallel = compare_models_batch(pairs, max_workers=4)
        parallel_time = time.perf_counter() - start
        print(
            f"\nbatch of {len(pairs)} pairs: serial={serial_time:.3f}s "
            f"parallel(4 workers)={parallel_time:.3f}s"
        )
        self.assertEqual(serial, parallel)
        self.assertEqual(
            [r.status for r in serial],
            [
                ComparisonStatus.EQUIVALENT,
                ComparisonStatus.DIFFERENT,
                ComparisonStatus.DIFFERENT,
                ComparisonStatus.EQUIVALENT,
                ComparisonStatus.DIFFERENT,
                ComparisonStatus.EQUIVALENT,
            ],
        )

    def test_batch_parallel_timing(self) -> None:
        # Heavier workload so that the serial/parallel timing comparison is
        # meaningful; results must be identical either way.
        chains = [_make_chain_model(300, prefix=f"m{i}_") for i in range(8)]
        pairs = [(m, _rename_model(m, f"r{i}_")) for i, m in enumerate(chains)]
        start = time.perf_counter()
        serial = compare_models_batch(pairs, max_workers=1)
        serial_time = time.perf_counter() - start
        start = time.perf_counter()
        parallel = compare_models_batch(pairs, max_workers=4)
        parallel_time = time.perf_counter() - start
        print(
            f"\nbatch of {len(pairs)} 300-node pairs: serial={serial_time:.3f}s "
            f"parallel(4 workers)={parallel_time:.3f}s"
        )
        self.assertEqual(serial, parallel)
        self.assertTrue(all(r.equivalent for r in parallel))

    def test_thousand_node_model_performance(self) -> None:
        model_a = _make_chain_model(1000)
        model_b = _rename_model(model_a, "r_")
        start = time.perf_counter()
        report = compare_models(model_a, model_b)
        elapsed = time.perf_counter() - start
        print(f"\n1000-node chain comparison: {elapsed:.3f}s")
        self.assertEqual(report.status, ComparisonStatus.EQUIVALENT)
        self.assertLess(elapsed, 30.0)

    def test_thousand_node_diff_performance(self) -> None:
        model_a = _make_chain_model(1000)
        model_b = _rename_model(model_a, "r_")
        # Flip one operator in the middle of the chain.
        model_b.graph.node[500].op_type = "Neg"
        start = time.perf_counter()
        report = compare_models(model_a, model_b)
        elapsed = time.perf_counter() - start
        print(f"\n1000-node chain diff comparison: {elapsed:.3f}s")
        self.assertEqual(report.status, ComparisonStatus.DIFFERENT)
        self.assertLess(elapsed, 30.0)
        # Exactly one root cause (the swapped operator); the rest of the
        # chain is reported as affected downstream, not as differences.
        op_diffs = [d for d in report.differences if d.kind == "op"]
        self.assertEqual(len(op_diffs), 1)
        self.assertIn("Identity", op_diffs[0].detail)
        self.assertIn("Neg", op_diffs[0].detail)
        self.assertGreater(len(report.affected_downstream), 400)


class TestCli(unittest.TestCase):
    def test_main_exit_codes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path_a = os.path.join(tmp, "a.onnx")
            path_b = os.path.join(tmp, "b.onnx")
            onnx.save(_make_conv_relu_model(), path_a)
            onnx.save(_rename_model(_make_conv_relu_model(), "c_"), path_b)
            self.assertEqual(graph_diff_main([path_a, path_b]), 0)
            onnx.save(_make_conv_relu_model(include_relu=False), path_b)
            self.assertEqual(graph_diff_main([path_a, path_b]), 1)
            with open(path_b, "wb") as f:
                f.write(b"junk")
            self.assertEqual(graph_diff_main([path_a, path_b]), 2)


if __name__ == "__main__":
    unittest.main()
