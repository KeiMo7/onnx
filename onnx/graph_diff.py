# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Structural equivalence comparison of ONNX models.

This module answers the question "do these two models describe the same
computation?" without executing either model.  The comparison is purely
structural and is insensitive to superficial producer differences:

* node names, intermediate tensor names and node ordering,
* attribute ordering, initializer ordering and metadata
  (``doc_string``, ``producer_*``, ``metadata_props``, ``ir_version``),
* optional attributes spelled out explicitly with their schema default
  value versus omitted (a note is recorded when this rule is applied),
* trailing omitted optional inputs versus explicit ``""`` placeholders.

Differences are reported at node and input granularity together with a
minimal root-cause set: nodes that merely sit downstream of a change are
listed separately as "affected" instead of being reported as differences
themselves.  Weight *values* are never printed in reports; only node and
tensor names appear.

The module is self-contained and read-only with respect to the rest of
ONNX: importing or using it does not change model loading, checking or
any other existing behaviour.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import enum
import hashlib
import json
import os
import struct
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import IO, TYPE_CHECKING, Any, ClassVar

import onnx.defs
from onnx.onnx_pb import (
    AttributeProto,
    GraphProto,
    ModelProto,
    NodeProto,
    TensorProto,
    TypeProto,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "ComparisonReport",
    "ComparisonStatus",
    "Difference",
    "compare_graphs",
    "compare_models",
    "compare_models_batch",
    "main",
]


class ComparisonStatus(enum.Enum):
    """Verdict of a model/graph comparison."""

    #: Structurally identical computation (modulo tolerated cosmetics).
    EQUIVALENT = "equivalent"
    #: Structurally identical, and the declared opset versions differ but
    #: every operator used has an identical schema signature in both versions.
    COMPATIBLE = "compatible"
    #: Real structural differences were found; see ``differences``.
    DIFFERENT = "different"
    #: The comparison could not be completed (unparseable structure or an
    #: internal failure).  ``differences`` is empty; see ``errors``.
    ERROR = "error"


@dataclasses.dataclass(frozen=True)
class Difference:
    """A single located difference between the two compared graphs.

    ``location`` is a path to the graph the difference was found in: ``""``
    for the main graph, or something like ``/node 'if_1'.then_branch`` for a
    subgraph.  ``node_a``/``node_b`` are the display names of the involved
    node on each side (``None`` when not applicable).  ``tensor`` is the
    involved tensor name when applicable.  No tensor *values* are ever
    included.
    """

    kind: str
    location: str
    detail: str
    node_a: str | None = None
    node_b: str | None = None
    tensor: str | None = None


@dataclasses.dataclass
class ComparisonReport:
    """Result of comparing two models or graphs.

    The verdict is derived from the same difference set that is reported:
    ``status`` is :attr:`ComparisonStatus.DIFFERENT` exactly when
    ``differences`` is non-empty, so the conclusion and the minimal
    difference set can never disagree.
    """

    status: ComparisonStatus
    differences: list[Difference] = dataclasses.field(default_factory=list)
    notes: list[str] = dataclasses.field(default_factory=list)
    added_nodes: list[str] = dataclasses.field(default_factory=list)
    removed_nodes: list[str] = dataclasses.field(default_factory=list)
    affected_downstream: list[str] = dataclasses.field(default_factory=list)
    errors: list[str] = dataclasses.field(default_factory=list)

    @property
    def equivalent(self) -> bool:
        """True when the two models describe the same computation."""
        return self.status is ComparisonStatus.EQUIVALENT

    @property
    def compatible(self) -> bool:
        """True when equivalent, or equivalent up to opset versions."""
        return self.status in (ComparisonStatus.EQUIVALENT, ComparisonStatus.COMPATIBLE)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "equivalent": self.equivalent,
            "compatible": self.compatible,
            "differences": [dataclasses.asdict(d) for d in self.differences],
            "notes": list(self.notes),
            "added_nodes": list(self.added_nodes),
            "removed_nodes": list(self.removed_nodes),
            "affected_downstream": list(self.affected_downstream),
            "errors": list(self.errors),
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_text(self) -> str:
        lines = [f"Status: {self.status.value.upper()}"]
        if self.errors:
            lines.append(f"Errors ({len(self.errors)}):")
            lines.extend(f"  {e}" for e in self.errors)
        if self.differences:
            lines.append(f"Differences ({len(self.differences)}):")
            for d in self.differences:
                where = d.location or "<main graph>"
                lines.append(f"  [{d.kind}] {where}: {d.detail}")
        if self.added_nodes:
            lines.append(f"Nodes only in model B ({len(self.added_nodes)}):")
            lines.extend(f"  + {n}" for n in self.added_nodes)
        if self.removed_nodes:
            lines.append(f"Nodes only in model A ({len(self.removed_nodes)}):")
            lines.extend(f"  - {n}" for n in self.removed_nodes)
        if self.affected_downstream:
            lines.append(
                f"Downstream nodes affected by the differences "
                f"({len(self.affected_downstream)}):"
            )
            lines.extend(f"  ~ {n}" for n in self.affected_downstream)
        if self.notes:
            lines.append(f"Notes ({len(self.notes)}):")
            lines.extend(f"  {n}" for n in self.notes)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_text()


# ---------------------------------------------------------------------------
# Canonical fingerprints
# ---------------------------------------------------------------------------


def _hash(*parts: Any) -> str:
    """Deterministic structural hash of the given canonical parts."""
    h = hashlib.sha256()
    for part in parts:
        data = part if isinstance(part, bytes) else repr(part).encode("utf-8")
        h.update(struct.pack("<q", len(data)))
        h.update(data)
    return h.hexdigest()


def _tensor_fingerprint(
    tensor: TensorProto, notes: list[str], side: str, path: str
) -> str:
    """Content fingerprint of an initializer/constant tensor.

    The tensor *name* and documentation are excluded so that renaming a
    weight does not change its fingerprint.  Raw values are hashed, never
    displayed.
    """
    if tensor.HasField("raw_data"):
        return _hash(
            "tensor",
            tensor.data_type,
            tuple(tensor.dims),
            tensor.raw_data,
        )
    if tensor.data_location == TensorProto.EXTERNAL:
        notes.append(
            f"{side}: tensor '{tensor.name}'{path} uses external data that is "
            "not loaded; it is compared by type and shape only"
        )
        return _hash("tensor-external", tensor.data_type, tuple(tensor.dims))
    cleared = TensorProto()
    cleared.CopyFrom(tensor)
    cleared.ClearField("name")
    cleared.ClearField("doc_string")
    cleared.ClearField("metadata_props")
    return _hash("tensor", cleared.SerializeToString(deterministic=True))


def _type_fingerprint(type_proto: TypeProto) -> str:
    """Canonical string for a TypeProto (used for graph input/output types)."""
    which = type_proto.WhichOneof("value")
    if which in ("tensor_type", "sparse_tensor_type"):
        t = getattr(type_proto, which)
        dims = []
        if t.HasField("shape"):
            for dim in t.shape.dim:
                kind = dim.WhichOneof("value")
                if kind == "dim_value":
                    dims.append(f"v{dim.dim_value}")
                elif kind == "dim_param":
                    dims.append(f"p:{dim.dim_param}")
                else:
                    dims.append("?")
        prefix = "tensor" if which == "tensor_type" else "sparse"
        return f"{prefix}({t.elem_type};{','.join(dims)})"
    if which == "sequence_type":
        return f"seq({_type_fingerprint(type_proto.sequence_type.elem_type)})"
    if which == "map_type":
        m = type_proto.map_type
        return f"map({m.key_type},{_type_fingerprint(m.value_type)})"
    if which == "optional_type":
        return f"opt({_type_fingerprint(type_proto.optional_type.elem_type)})"
    if which == "opaque_type":
        o = type_proto.opaque_type
        return f"opaque({o.domain},{o.name})"
    return "unknown"


def _display_name(node: NodeProto) -> str:
    """Human-readable node identifier used in reports."""
    if node.name:
        return node.name
    for out in node.output:
        if out:
            return f"{node.op_type}({out})"
    return node.op_type


@dataclasses.dataclass
class _NodeInfo:
    """Canonical view of one node used for matching and diff reporting."""

    index: int  # position in graph.node (declaration order)
    topo_rank: int  # position in topological order
    domain: str
    op_type: str
    display: str
    inputs: tuple[str, ...]  # value colors, trailing omitted entries dropped
    input_names: tuple[str, ...]  # raw input names aligned with ``inputs``
    num_outputs: int  # number of (non-empty) outputs
    attrs: dict[str, Any]  # canonical attribute values (hashable)
    attr_graphs: dict[str, GraphProto]  # subgraph attributes, for recursion
    color: str


@dataclasses.dataclass
class _GraphContext:
    """Fingerprinted view of one graph."""

    path: str
    nodes: list[_NodeInfo]  # topological order
    value_colors: dict[str, str]
    init_colors: dict[str, str]  # initializer name -> content color
    init_meta: dict[str, tuple[int, tuple[int, ...]]]  # name -> (dtype, dims)
    input_names: list[str]
    output_names: list[str]
    input_types: list[str]
    output_types: list[str]
    output_colors: list[str]
    unused_init_colors: list[str]
    fingerprint: str


class _Fingerprinter:
    """Computes canonical structural fingerprints for all graphs of a model."""

    def __init__(
        self,
        side: str,
        opsets: dict[str, int],
        notes: list[str],
        errors: list[str],
    ) -> None:
        self._side = side
        self._opsets = opsets
        self._notes = notes
        self._errors = errors
        self._schema_cache: dict[tuple[str, str], Any] = {}
        self._no_schema_noted: set[tuple[str, str]] = set()

    # -- schema helpers ----------------------------------------------------

    def _schema(self, domain: str, op_type: str) -> Any | None:
        key = (domain, op_type)
        if key not in self._schema_cache:
            version = self._opsets.get(domain)
            try:
                if version is None:
                    schema = onnx.defs.get_schema(op_type, domain=domain)
                else:
                    schema = onnx.defs.get_schema(op_type, version, domain)
            except (onnx.defs.SchemaError, RuntimeError):
                schema = None
            self._schema_cache[key] = schema
            if schema is None and key not in self._no_schema_noted:
                self._no_schema_noted.add(key)
                self._notes.append(
                    f"{self._side}: no registered schema for operator "
                    f"'{domain or 'ai.onnx'}::{op_type}'; its attributes are "
                    "compared literally without default-value normalization"
                )
        return self._schema_cache[key]

    # -- attribute canonicalization ----------------------------------------

    _FIELD_TO_ATTR_TYPE: ClassVar[dict[str, Any]] = {
        "f": AttributeProto.FLOAT,
        "i": AttributeProto.INT,
        "s": AttributeProto.STRING,
        "t": AttributeProto.TENSOR,
        "g": AttributeProto.GRAPH,
        "tp": AttributeProto.TYPE_PROTO,
        "floats": AttributeProto.FLOATS,
        "ints": AttributeProto.INTS,
        "strings": AttributeProto.STRINGS,
        "tensors": AttributeProto.TENSORS,
        "graphs": AttributeProto.GRAPHS,
        "sparse_tensor": AttributeProto.SPARSE_TENSOR,
        "sparse_tensors": AttributeProto.SPARSE_TENSORS,
        "type_protos": AttributeProto.TYPE_PROTOS,
    }

    def _attr_canonical(  # noqa: PLR0911
        self,
        attr: AttributeProto,
        env: dict[str, str],
        path: str,
        attr_graphs: dict[str, GraphProto] | None = None,
    ) -> Any:
        atype = attr.type
        if atype == AttributeProto.UNDEFINED:
            # AttributeProto.type is not always populated by producers; fall
            # back to whichever value field is set.
            atype = self._FIELD_TO_ATTR_TYPE.get(
                attr.WhichOneof("value") or "", AttributeProto.UNDEFINED
            )
        if atype == AttributeProto.FLOAT:
            return ("f", attr.f)
        if atype == AttributeProto.INT:
            return ("i", attr.i)
        if atype == AttributeProto.STRING:
            return ("s", attr.s)
        if atype == AttributeProto.TENSOR:
            return ("t", _tensor_fingerprint(attr.t, self._notes, self._side, path))
        if atype == AttributeProto.SPARSE_TENSOR:
            return ("st", attr.sparse_tensor.SerializeToString(deterministic=True))
        if atype == AttributeProto.GRAPH:
            sub = self.fingerprint_graph(attr.g, dict(env), f"{path}.{attr.name}")
            if attr_graphs is not None:
                attr_graphs[attr.name] = attr.g
            return ("g", sub.fingerprint)
        if atype == AttributeProto.TYPE_PROTO:
            return ("tp", _type_fingerprint(attr.tp))
        if atype == AttributeProto.FLOATS:
            return ("fs", tuple(attr.floats))
        if atype == AttributeProto.INTS:
            return ("is", tuple(attr.ints))
        if atype == AttributeProto.STRINGS:
            return ("ss", tuple(attr.strings))
        if atype == AttributeProto.TENSORS:
            return (
                "ts",
                tuple(
                    _tensor_fingerprint(t, self._notes, self._side, path)
                    for t in attr.tensors
                ),
            )
        if atype == AttributeProto.SPARSE_TENSORS:
            return (
                "sts",
                tuple(
                    t.SerializeToString(deterministic=True) for t in attr.sparse_tensors
                ),
            )
        if atype == AttributeProto.GRAPHS:
            fps = []
            for i, g in enumerate(attr.graphs):
                sub = self.fingerprint_graph(g, dict(env), f"{path}.{attr.name}[{i}]")
                fps.append(sub.fingerprint)
            return ("gs", tuple(fps))
        if atype == AttributeProto.TYPE_PROTOS:
            return ("tps", tuple(_type_fingerprint(tp) for tp in attr.type_protos))
        return ("unknown", attr.SerializeToString(deterministic=True))

    def _node_attributes(
        self,
        node: NodeProto,
        env: dict[str, str],
        path: str,
        attr_graphs: dict[str, GraphProto],
    ) -> dict[str, Any]:
        schema = self._schema(node.domain, node.op_type)
        canonical: dict[str, Any] = {}
        explicit: dict[str, AttributeProto] = {a.name: a for a in node.attribute}
        for name, attr in explicit.items():
            canonical[name] = self._attr_canonical(attr, env, path, attr_graphs)
        if schema is not None:
            for name, attr_schema in schema.attributes.items():
                default = attr_schema.default_value
                has_default = default.type != AttributeProto.UNDEFINED
                if name in explicit:
                    if (
                        has_default
                        and name not in attr_graphs
                        and canonical.get(name)
                        == self._attr_canonical(default, env, path)
                    ):
                        self._notes.append(
                            f"{self._side}: node '{_display_name(node)}'{path} "
                            f"sets attribute '{name}' to its schema default "
                            "explicitly; treated as equivalent to omitting it"
                        )
                elif has_default:
                    canonical[name] = self._attr_canonical(default, env, path)
        return canonical

    # -- graph fingerprinting ------------------------------------------------

    def fingerprint_graph(
        self, graph: GraphProto, env: dict[str, str], path: str
    ) -> _GraphContext:
        """Fingerprint ``graph``; ``env`` maps outer-scope value colors."""
        value_colors: dict[str, str] = dict(env)
        init_colors: dict[str, str] = {}
        init_meta: dict[str, tuple[int, tuple[int, ...]]] = {}
        for init in graph.initializer:
            color = _tensor_fingerprint(init, self._notes, self._side, path)
            init_colors[init.name] = color
            init_meta[init.name] = (init.data_type, tuple(init.dims))
            value_colors[init.name] = color
        for sparse in graph.sparse_initializer:
            color = _hash("sparse", sparse.SerializeToString(deterministic=True))
            value_colors[sparse.values.name] = color
        input_names = [vi.name for vi in graph.input]
        output_names = [vi.name for vi in graph.output]
        input_types = [_type_fingerprint(vi.type) for vi in graph.input]
        output_types = [_type_fingerprint(vi.type) for vi in graph.output]
        for idx, name in enumerate(input_names):
            if name not in init_colors:
                value_colors[name] = f"IN:{idx}"

        # Producer table and topological order (Kahn's algorithm, stable in
        # declaration order so results are deterministic).
        producer: dict[str, int] = {}
        for i, node in enumerate(graph.node):
            for out in node.output:
                if out:
                    producer[out] = i
        indegree = [0] * len(graph.node)
        predecessors: list[set[int]] = [set() for i in range(len(graph.node))]
        missing: list[tuple[int, str]] = []
        for i, node in enumerate(graph.node):
            for inp in node.input:
                if not inp:
                    continue
                if inp in producer:
                    pred = producer[inp]
                    if pred not in predecessors[i]:
                        predecessors[i].add(pred)
                        indegree[i] += 1
                elif inp not in value_colors:
                    missing.append((i, inp))
        for i, inp in missing:
            node = graph.node[i]
            self._errors.append(
                f"{self._side}: node '{_display_name(node)}'{path} refers to "
                f"input '{inp}' which is neither a graph input, an "
                "initializer, nor the output of any node"
            )
            value_colors[inp] = f"MISSING:{inp}"
        ready = collections.deque(i for i in range(len(graph.node)) if indegree[i] == 0)
        topo: list[int] = []
        successors: list[list[int]] = [[] for _ in range(len(graph.node))]
        for i, preds in enumerate(predecessors):
            for p in preds:
                successors[p].append(i)
        while ready:
            i = ready.popleft()
            topo.append(i)
            for j in successors[i]:
                indegree[j] -= 1
                if indegree[j] == 0:
                    ready.append(j)
        if len(topo) != len(graph.node):
            in_topo = set(topo)
            remaining = [i for i in range(len(graph.node)) if i not in in_topo]
            names = ", ".join(
                f"'{_display_name(graph.node[i])}'" for i in remaining[:5]
            )
            self._errors.append(
                f"{self._side}: graph{path or ' <main graph>'} contains a "
                f"dependency cycle involving node(s) {names}"
            )
            topo.extend(remaining)  # keep going to collect more errors

        nodes: list[_NodeInfo] = []
        for rank, i in enumerate(topo):
            node = graph.node[i]
            input_colors: list[str] = []
            input_names_raw: list[str] = []
            for inp in node.input:
                input_names_raw.append(inp)
                input_colors.append(value_colors[inp] if inp else "OMITTED")
            while input_colors and input_colors[-1] == "OMITTED":
                input_colors.pop()
                input_names_raw.pop()
            if len(input_colors) != len(node.input):
                self._notes.append(
                    f"{self._side}: node '{_display_name(node)}'{path} has "
                    "trailing omitted optional input(s); normalized"
                )
            attr_graphs: dict[str, GraphProto] = {}
            attrs = self._node_attributes(node, value_colors, path, attr_graphs)
            color = _hash(
                "node",
                node.domain,
                node.op_type,
                sorted(attrs.items()),
                tuple(input_colors),
            )
            info = _NodeInfo(
                index=i,
                topo_rank=rank,
                domain=node.domain,
                op_type=node.op_type,
                display=_display_name(node),
                inputs=tuple(input_colors),
                input_names=tuple(input_names_raw),
                num_outputs=sum(1 for o in node.output if o),
                attrs=attrs,
                attr_graphs=attr_graphs,
                color=color,
            )
            nodes.append(info)
            for out_idx, out in enumerate(node.output):
                if out:
                    value_colors[out] = _hash("val", color, out_idx)

        output_colors = [
            value_colors.get(name, f"MISSING:{name}") for name in output_names
        ]
        consumed = {inp for node in graph.node for inp in node.input if inp}
        consumed.update(output_names)
        unused_init_colors = sorted(
            color for name, color in init_colors.items() if name not in consumed
        )
        fingerprint = _hash(
            "graph",
            tuple(input_types),
            tuple(sorted(n.color for n in nodes)),
            tuple(output_colors),
            tuple(unused_init_colors),
        )
        return _GraphContext(
            path=path,
            nodes=nodes,
            value_colors=value_colors,
            init_colors=init_colors,
            init_meta=init_meta,
            input_names=input_names,
            output_names=output_names,
            input_types=input_types,
            output_types=output_types,
            output_colors=output_colors,
            unused_init_colors=unused_init_colors,
            fingerprint=fingerprint,
        )


# ---------------------------------------------------------------------------
# Graph comparison
# ---------------------------------------------------------------------------


class _GraphComparator:
    """Compares two fingerprinted graphs and produces located differences."""

    def __init__(
        self,
        fa: _Fingerprinter,
        fb: _Fingerprinter,
        differences: list[Difference],
        notes: list[str],
        added: list[str],
        removed: list[str],
        affected: list[str],
    ) -> None:
        self._fa = fa
        self._fb = fb
        self._differences = differences
        self._notes = notes
        self._added = added
        self._removed = removed
        self._affected = affected
        self._current_correspondence: set[tuple[str, str]] = set()

    def compare(self, ca: _GraphContext, cb: _GraphContext, path: str) -> None:
        self._compare_signature(ca, cb, path)
        colors_a = [n.color for n in ca.nodes]
        colors_b = [n.color for n in cb.nodes]
        if sorted(colors_a) == sorted(colors_b):
            if ca.output_colors != cb.output_colors:
                self._differences.append(
                    Difference(
                        kind="graph-output",
                        location=path,
                        detail=(
                            "graph outputs are produced in a different order "
                            f"({ca.output_names} vs {cb.output_names})"
                        ),
                    )
                )
            self._compare_unused_initializers(ca, cb, path)
            return
        self._compare_nodes(ca, cb, path)
        self._compare_unused_initializers(ca, cb, path)

    # -- signature -----------------------------------------------------------

    def _compare_signature(
        self, ca: _GraphContext, cb: _GraphContext, path: str
    ) -> None:
        where = path or "<main graph>"
        for label, names_a, names_b, types_a, types_b in (
            (
                "graph-input",
                ca.input_names,
                cb.input_names,
                ca.input_types,
                cb.input_types,
            ),
            (
                "graph-output",
                ca.output_names,
                cb.output_names,
                ca.output_types,
                cb.output_types,
            ),
        ):
            if len(names_a) != len(names_b):
                self._differences.append(
                    Difference(
                        kind=label,
                        location=path,
                        detail=(
                            f"{where} has {len(names_a)} {label.split('-')[1]}(s) "
                            f"{names_a} but the other graph has {len(names_b)} "
                            f"{names_b}"
                        ),
                    )
                )
                continue
            for pos, (na, nb, ta, tb) in enumerate(
                zip(names_a, names_b, types_a, types_b, strict=True)
            ):
                if ta != tb:
                    self._differences.append(
                        Difference(
                            kind=label,
                            location=path,
                            detail=(
                                f"{label.replace('-', ' ')} {pos} type differs: "
                                f"'{na}' ({ta}) vs '{nb}' ({tb})"
                            ),
                            tensor=na,
                        )
                    )
                elif na != nb:
                    self._notes.append(
                        f"{label.replace('-', ' ')} {pos} renamed: "
                        f"'{na}' vs '{nb}' (same type and position)"
                    )

    # -- node matching ---------------------------------------------------------

    def _compare_nodes(self, ca: _GraphContext, cb: _GraphContext, path: str) -> None:
        unmatched_a = self._unmatched(ca.nodes, cb.nodes)
        unmatched_b = self._unmatched(cb.nodes, ca.nodes)

        # Value correspondence between the two graphs, seeded with the
        # exact structural matches and extended whenever two nodes are
        # paired.  A pair (color_a, color_b) means "these values play the
        # same role on both sides".
        common = collections.Counter(ca.value_colors.values()) & collections.Counter(
            cb.value_colors.values()
        )
        correspond: set[tuple[str, str]] = {(c, c) for c in common}
        correspond_a = {x for x, _ in correspond}
        correspond_b = {y for _, y in correspond}

        def inputs_correspond(na: _NodeInfo, nb: _NodeInfo) -> bool:
            return len(na.inputs) == len(nb.inputs) and all(
                (x, y) in correspond for x, y in zip(na.inputs, nb.inputs, strict=True)
            )

        def output_colors(n: _NodeInfo) -> list[str]:
            return [_hash("val", n.color, k) for k in range(n.num_outputs)]

        paired_a: set[int] = set()
        paired_b: set[int] = set()
        pairs: list[tuple[_NodeInfo, _NodeInfo]] = []

        def commit(na: _NodeInfo, nb: _NodeInfo) -> None:
            pairs.append((na, nb))
            paired_a.add(id(na))
            paired_b.add(id(nb))
            for x, y in zip(output_colors(na), output_colors(nb), strict=False):
                correspond.add((x, y))
                correspond_a.add(x)
                correspond_b.add(y)

        # Phase 1: pair nodes with the same operator by mutual-best score
        # (number of corresponding inputs, then topological proximity),
        # iterating to a fixpoint so that each pairing propagates value
        # correspondences to the neighbours of the paired nodes.
        groups_b: dict[tuple[str, str], list[_NodeInfo]] = {}
        for n in unmatched_b:
            groups_b.setdefault((n.domain, n.op_type), []).append(n)
        groups_a: dict[tuple[str, str], list[_NodeInfo]] = {}
        for n in unmatched_a:
            groups_a.setdefault((n.domain, n.op_type), []).append(n)

        def best_candidate(
            n: _NodeInfo, candidates: list[_NodeInfo], paired: set[int]
        ) -> _NodeInfo | None:
            best: _NodeInfo | None = None
            best_key: tuple | None = None
            for m in candidates:
                if id(m) in paired:
                    continue
                shared = sum(
                    1
                    for x, y in zip(n.inputs, m.inputs, strict=False)
                    if (x, y) in correspond
                )
                key = (shared, -abs(n.topo_rank - m.topo_rank), -m.topo_rank)
                if best_key is None or key > best_key:
                    best_key = key
                    best = m
            return best

        while True:
            progress = False
            for key, group_a in groups_a.items():
                group_b = groups_b.get(key, [])
                for na in group_a:
                    if id(na) in paired_a:
                        continue
                    nb = best_candidate(na, group_b, paired_b)
                    if nb is None:
                        continue
                    if best_candidate(nb, group_a, paired_a) is na:
                        commit(na, nb)
                        progress = True
            if not progress:
                break

        # Phase 2: pair remaining frontier nodes across operator types when
        # their inputs correspond — this catches nodes whose operator was
        # swapped for another one.  A node is a frontier (root) node when
        # every one of its inputs already has a counterpart on the other
        # side, i.e. the node is not downstream of an unpaired difference.
        def is_frontier_a(n: _NodeInfo) -> bool:
            return all(c in correspond_a for c in n.inputs)

        def is_frontier_b(n: _NodeInfo) -> bool:
            return all(c in correspond_b for c in n.inputs)

        leftover_b = [n for n in unmatched_b if id(n) not in paired_b]
        for na in unmatched_a:
            if id(na) in paired_a or not is_frontier_a(na):
                continue
            best: _NodeInfo | None = None
            best_key: tuple | None = None
            for nb in leftover_b:
                if id(nb) in paired_b or not is_frontier_b(nb):
                    continue
                if not inputs_correspond(na, nb):
                    continue
                cand_key = (na.domain == nb.domain, -nb.topo_rank)
                if best_key is None or cand_key > best_key:
                    best_key = cand_key
                    best = nb
            if best is not None:
                commit(na, best)

        # Classification happens only after the matching fixpoint, so pairs
        # are judged against the final value correspondence.
        self._current_correspondence = correspond
        for na, nb in pairs:
            self._classify_pair(ca, cb, na, nb, path, inputs_correspond(na, nb))

        for n in unmatched_a:
            if id(n) not in paired_a:
                if is_frontier_a(n):
                    self._removed.append(f"{n.display} ({n.op_type})")
                    self._differences.append(
                        Difference(
                            kind="node-removed",
                            location=path,
                            detail=(
                                f"node '{n.display}' ({n.op_type}) exists only "
                                "in model A"
                            ),
                            node_a=n.display,
                        )
                    )
                else:
                    self._affected.append(f"A:{n.display} ({n.op_type})")
        for n in unmatched_b:
            if id(n) not in paired_b:
                if is_frontier_b(n):
                    self._added.append(f"{n.display} ({n.op_type})")
                    self._differences.append(
                        Difference(
                            kind="node-added",
                            location=path,
                            detail=(
                                f"node '{n.display}' ({n.op_type}) exists only "
                                "in model B"
                            ),
                            node_b=n.display,
                        )
                    )
                else:
                    self._affected.append(f"B:{n.display} ({n.op_type})")

    @staticmethod
    def _unmatched(nodes: list[_NodeInfo], other: list[_NodeInfo]) -> list[_NodeInfo]:
        remaining = collections.Counter(n.color for n in other)
        out = []
        for n in nodes:
            if remaining[n.color] > 0:
                remaining[n.color] -= 1
            else:
                out.append(n)
        return out

    # -- per-node diff ---------------------------------------------------------

    def _classify_pair(
        self,
        ca: _GraphContext,
        cb: _GraphContext,
        na: _NodeInfo,
        nb: _NodeInfo,
        path: str,
        inputs_correspond: bool,
    ) -> None:
        same_op = (na.domain, na.op_type) == (nb.domain, nb.op_type)
        if inputs_correspond and same_op and na.attrs == nb.attrs:
            # Same computation; only its position in a changed subgraph
            # differs.  It is a victim of an upstream change, not a cause.
            self._affected.append(f"{na.display} <-> {nb.display} ({na.op_type})")
            return
        head = (
            f"node '{na.display}' <-> '{nb.display}'"
            if na.display != nb.display
            else f"node '{na.display}'"
        )
        if not same_op:
            self._differences.append(
                Difference(
                    kind="op",
                    location=path,
                    detail=(
                        f"{head}: operator differs "
                        f"({na.domain or 'ai.onnx'}::{na.op_type} vs "
                        f"{nb.domain or 'ai.onnx'}::{nb.op_type})"
                    ),
                    node_a=na.display,
                    node_b=nb.display,
                )
            )
        self._diff_attributes(ca, cb, na, nb, path, head)
        if inputs_correspond:
            return
        if len(na.inputs) != len(nb.inputs):
            self._differences.append(
                Difference(
                    kind="input",
                    location=path,
                    detail=(
                        f"{head} ({na.op_type}): input count differs "
                        f"({len(na.inputs)} vs {len(nb.inputs)})"
                    ),
                    node_a=na.display,
                    node_b=nb.display,
                )
            )
        for pos, (xa, xb) in enumerate(zip(na.inputs, nb.inputs, strict=False)):
            if (xa, xb) in self._current_correspondence or xa == xb:
                continue
            name_a = na.input_names[pos] if pos < len(na.input_names) else "?"
            name_b = nb.input_names[pos] if pos < len(nb.input_names) else "?"
            detail = self._describe_input_mismatch(ca, cb, name_a, name_b, xa, xb)
            self._differences.append(
                Difference(
                    kind="input",
                    location=path,
                    detail=f"{head} ({na.op_type}): input {pos}: {detail}",
                    node_a=na.display,
                    node_b=nb.display,
                    tensor=name_a or None,
                )
            )

    def _diff_attributes(
        self,
        ca: _GraphContext,
        cb: _GraphContext,
        na: _NodeInfo,
        nb: _NodeInfo,
        path: str,
        head: str,
    ) -> None:
        for name in sorted(set(na.attrs) | set(nb.attrs)):
            va, vb = na.attrs.get(name), nb.attrs.get(name)
            if va == vb:
                continue
            ga, gb = na.attr_graphs.get(name), nb.attr_graphs.get(name)
            if ga is not None and gb is not None:
                sub_path = f"{path}/node '{na.display}'.{name}"
                sub_a = self._fa.fingerprint_graph(ga, dict(ca.value_colors), sub_path)
                sub_b = self._fb.fingerprint_graph(gb, dict(cb.value_colors), sub_path)
                self.compare(sub_a, sub_b, sub_path)
            else:
                self._differences.append(
                    Difference(
                        kind="attribute",
                        location=path,
                        detail=f"{head} ({na.op_type}): attribute '{name}' differs",
                        node_a=na.display,
                        node_b=nb.display,
                    )
                )

    def _describe_input_mismatch(
        self,
        ca: _GraphContext,
        cb: _GraphContext,
        name_a: str,
        name_b: str,
        color_a: str,
        color_b: str,
    ) -> str:
        init_a = name_a in ca.init_colors
        init_b = name_b in cb.init_colors
        if init_a and init_b:
            meta_a = ca.init_meta[name_a]
            meta_b = cb.init_meta[name_b]
            if meta_a[0] != meta_b[0]:
                return (
                    f"initializer '{name_a}' vs '{name_b}': data type differs "
                    f"({TensorProto.DataType.Name(meta_a[0])} vs "
                    f"{TensorProto.DataType.Name(meta_b[0])})"
                )
            if meta_a[1] != meta_b[1]:
                return (
                    f"initializer '{name_a}' vs '{name_b}': shape differs "
                    f"({list(meta_a[1])} vs {list(meta_b[1])})"
                )
            return (
                f"initializer '{name_a}' vs '{name_b}': values differ "
                "(same data type and shape)"
            )
        if init_a != init_b:
            which = name_a if init_a else name_b
            side = "A" if init_a else "B"
            return (
                f"'{name_a}' vs '{name_b}': a constant initializer "
                f"('{which}') in model {side} but a computed value in the other"
            )
        if color_a == "OMITTED" or color_b == "OMITTED":
            return f"'{name_a or '<omitted>'}' vs '{name_b or '<omitted>'}'"
        return f"'{name_a}' vs '{name_b}': produced by different computations"

    # -- initializers ------------------------------------------------------------

    def _compare_unused_initializers(
        self, ca: _GraphContext, cb: _GraphContext, path: str
    ) -> None:
        if ca.unused_init_colors == cb.unused_init_colors:
            return
        self._differences.append(
            Difference(
                kind="initializer",
                location=path,
                detail=(
                    "the graphs carry different unused (dead) initializers: "
                    f"{len(ca.unused_init_colors)} vs "
                    f"{len(cb.unused_init_colors)}"
                ),
            )
        )


# ---------------------------------------------------------------------------
# Opset compatibility
# ---------------------------------------------------------------------------


def _schema_signature(schema: Any) -> Any:
    return (
        tuple((i.name, str(i.option)) for i in schema.inputs),
        tuple((o.name, str(o.option)) for o in schema.outputs),
        tuple(
            sorted((k, str(a.type), a.required) for k, a in schema.attributes.items())
        ),
        tuple(
            sorted(
                (tc.type_param_str, tuple(sorted(tc.allowed_type_strs)))
                for tc in schema.type_constraints
            )
        ),
    )


def _collect_used_ops(graph: GraphProto, into: set[tuple[str, str]]) -> None:
    for node in graph.node:
        into.add((node.domain, node.op_type))
        for attr in node.attribute:
            if attr.type == AttributeProto.GRAPH:
                _collect_used_ops(attr.g, into)
            elif attr.type == AttributeProto.GRAPHS:
                for g in attr.graphs:
                    _collect_used_ops(g, into)


def _compare_opsets(
    model_a: ModelProto,
    model_b: ModelProto,
    differences: list[Difference],
    notes: list[str],
) -> bool:
    """Compare opset imports; return True if versions differ but are compatible."""
    opsets_a = {op.domain or "": op.version for op in model_a.opset_import}
    opsets_b = {op.domain or "": op.version for op in model_b.opset_import}
    used_a: set[tuple[str, str]] = set()
    used_b: set[tuple[str, str]] = set()
    _collect_used_ops(model_a.graph, used_a)
    _collect_used_ops(model_b.graph, used_b)
    compatible_with_different_versions = False
    for domain in sorted(set(opsets_a) | set(opsets_b)):
        va, vb = opsets_a.get(domain), opsets_b.get(domain)
        if va == vb:
            continue
        if va is None or vb is None:
            differences.append(
                Difference(
                    kind="opset",
                    location="",
                    detail=(
                        f"opset import for domain '{domain or 'ai.onnx'}' is "
                        f"present in only one model "
                        f"(A: {va}, B: {vb})"
                    ),
                )
            )
            continue
        domain_compatible = True
        ops_in_domain = sorted(op for (d, op) in (used_a | used_b) if d == domain)
        for op_type in ops_in_domain:
            try:
                sa = onnx.defs.get_schema(op_type, va, domain)
            except (onnx.defs.SchemaError, RuntimeError):
                sa = None
            try:
                sb = onnx.defs.get_schema(op_type, vb, domain)
            except (onnx.defs.SchemaError, RuntimeError):
                sb = None
            if sa is None or sb is None:
                missing_in = f"opset {va}" if sa is None else f"opset {vb}"
                differences.append(
                    Difference(
                        kind="opset",
                        location="",
                        detail=(
                            f"operator '{domain or 'ai.onnx'}::{op_type}' has "
                            f"no schema in {missing_in} (domain opsets differ: "
                            f"{va} vs {vb})"
                        ),
                    )
                )
                domain_compatible = False
            elif _schema_signature(sa) != _schema_signature(sb):
                differences.append(
                    Difference(
                        kind="opset",
                        location="",
                        detail=(
                            f"operator '{domain or 'ai.onnx'}::{op_type}' has "
                            f"different schema signatures in opset {va} and "
                            f"opset {vb} of domain "
                            f"'{domain or 'ai.onnx'}'"
                        ),
                    )
                )
                domain_compatible = False
        if domain_compatible:
            compatible_with_different_versions = True
            notes.append(
                f"opset import for domain '{domain or 'ai.onnx'}' differs "
                f"({va} vs {vb}) but every operator used has an identical "
                "schema signature in both versions"
            )
    return compatible_with_different_versions


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_ModelLike = ModelProto | bytes | str | os.PathLike | IO[bytes]


def _load_model(source: _ModelLike, side: str) -> ModelProto:
    if isinstance(source, ModelProto):
        return source
    model = ModelProto()
    if isinstance(source, (bytes, bytearray)):
        model.ParseFromString(bytes(source))
        return model
    if isinstance(source, (str, os.PathLike)):
        with open(source, "rb") as f:
            model.ParseFromString(f.read())
        return model
    if hasattr(source, "read"):
        model.ParseFromString(source.read())
        return model
    raise TypeError(f"{side}: unsupported model source type {type(source)!r}")


def _model_opsets(model: ModelProto) -> dict[str, int]:
    return {op.domain or "": op.version for op in model.opset_import}


def _compare_models_impl(model_a: ModelProto, model_b: ModelProto) -> ComparisonReport:
    differences: list[Difference] = []
    notes: list[str] = []
    added: list[str] = []
    removed: list[str] = []
    affected: list[str] = []
    errors: list[str] = []

    opsets_a = _model_opsets(model_a)
    opsets_b = _model_opsets(model_b)
    fa = _Fingerprinter("model A", opsets_a, notes, errors)
    fb = _Fingerprinter("model B", opsets_b, notes, errors)
    ca = fa.fingerprint_graph(model_a.graph, {}, "")
    cb = fb.fingerprint_graph(model_b.graph, {}, "")

    if not errors:
        comparator = _GraphComparator(
            fa, fb, differences, notes, added, removed, affected
        )
        comparator.compare(ca, cb, "")

        # Function definitions (compared by structural fingerprint).
        funcs_a = {
            (f.domain, f.name): f.SerializeToString(deterministic=True)
            for f in model_a.functions
        }
        funcs_b = {
            (f.domain, f.name): f.SerializeToString(deterministic=True)
            for f in model_b.functions
        }
        for key in sorted(set(funcs_a) | set(funcs_b)):
            if key not in funcs_a:
                differences.append(
                    Difference(
                        kind="function",
                        location="",
                        detail=f"function '{key[0]}::{key[1]}' exists only in model B",
                    )
                )
            elif key not in funcs_b:
                differences.append(
                    Difference(
                        kind="function",
                        location="",
                        detail=f"function '{key[0]}::{key[1]}' exists only in model A",
                    )
                )
            elif funcs_a[key] != funcs_b[key]:
                differences.append(
                    Difference(
                        kind="function",
                        location="",
                        detail=f"function '{key[0]}::{key[1]}' differs between the models",
                    )
                )

        opset_compatible = _compare_opsets(model_a, model_b, differences, notes)
        if model_a.ir_version != model_b.ir_version:
            notes.append(
                f"ir_version differs ({model_a.ir_version} vs "
                f"{model_b.ir_version}); treated as metadata"
            )

    # The verdict is derived from the very same difference set that is
    # reported, so the conclusion and the minimal difference set are always
    # consistent.  The report object is assembled atomically here: a failed
    # comparison yields an ERROR report with empty difference lists rather
    # than a partially filled one.
    if errors:
        return ComparisonReport(
            status=ComparisonStatus.ERROR,
            notes=notes,
            errors=errors,
        )
    if differences:
        status = ComparisonStatus.DIFFERENT
    elif opset_compatible:
        status = ComparisonStatus.COMPATIBLE
    else:
        status = ComparisonStatus.EQUIVALENT
    return ComparisonReport(
        status=status,
        differences=differences,
        notes=notes,
        added_nodes=sorted(added),
        removed_nodes=sorted(removed),
        affected_downstream=sorted(affected),
    )


def compare_models(
    model_a: _ModelLike,
    model_b: _ModelLike,
) -> ComparisonReport:
    """Compare two ONNX models for structural equivalence.

    Accepts ``ModelProto`` objects, serialized bytes, file paths or binary
    file objects.  Never executes the models.  Returns a
    :class:`ComparisonReport`; unparseable input yields a report with
    ``status == ComparisonStatus.ERROR`` rather than an exception or a
    "different" verdict.
    """
    try:
        a = _load_model(model_a, "model A")
        b = _load_model(model_b, "model B")
        return _compare_models_impl(a, b)
    except Exception as exc:  # noqa: BLE001 - a failed comparison must not
        # propagate and must not produce a partial report.
        return ComparisonReport(
            status=ComparisonStatus.ERROR,
            errors=[f"comparison failed: {type(exc).__name__}: {exc}"],
        )


def compare_graphs(
    graph_a: GraphProto,
    graph_b: GraphProto,
    *,
    opset_imports: dict[str, int] | None = None,
) -> ComparisonReport:
    """Compare two graphs directly (opset context may be supplied)."""
    try:
        opsets = dict(opset_imports or {})
        model_a = ModelProto()
        model_a.graph.CopyFrom(graph_a)
        model_b = ModelProto()
        model_b.graph.CopyFrom(graph_b)
        for domain, version in opsets.items():
            for model in (model_a, model_b):
                model.opset_import.add(domain=domain, version=version)
        return _compare_models_impl(model_a, model_b)
    except Exception as exc:  # noqa: BLE001
        return ComparisonReport(
            status=ComparisonStatus.ERROR,
            errors=[f"comparison failed: {type(exc).__name__}: {exc}"],
        )


def _to_bytes(source: _ModelLike, side: str) -> bytes:
    if isinstance(source, ModelProto):
        return bytes(source.SerializeToString())
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    if isinstance(source, (str, os.PathLike)):
        with open(source, "rb") as f:
            return f.read()
    if hasattr(source, "read"):
        return source.read()
    raise TypeError(f"{side}: unsupported model source type {type(source)!r}")


def _compare_pair_bytes(payload: tuple[bytes, bytes]) -> ComparisonReport:
    a_bytes, b_bytes = payload
    return compare_models(a_bytes, b_bytes)


def compare_models_batch(
    pairs: Sequence[tuple[_ModelLike, _ModelLike]],
    *,
    max_workers: int | None = None,
) -> list[ComparisonReport]:
    """Compare many model pairs, optionally in parallel.

    Results are returned in input order and are identical to running
    :func:`compare_models` serially on each pair, regardless of
    ``max_workers``.
    """
    payloads = [(_to_bytes(a, "model A"), _to_bytes(b, "model B")) for a, b in pairs]
    if max_workers is None or max_workers <= 1 or len(payloads) <= 1:
        return [_compare_pair_bytes(p) for p in payloads]
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(_compare_pair_bytes, payloads))


# ---------------------------------------------------------------------------
# Command line interface
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Compare two ONNX model files and print a difference report."""
    parser = argparse.ArgumentParser(
        prog="python -m onnx.graph_diff",
        description=(
            "Structurally compare two ONNX models and report where they "
            "differ (node level and input level)."
        ),
    )
    parser.add_argument("model_a", help="path to the first .onnx model")
    parser.add_argument("model_b", help="path to the second .onnx model")
    parser.add_argument(
        "--json", action="store_true", help="emit the report as JSON instead of text"
    )
    args = parser.parse_args(argv)
    report = compare_models(args.model_a, args.model_b)
    print(report.to_json() if args.json else report.to_text())  # noqa: T201
    if report.status is ComparisonStatus.ERROR:
        return 2
    return 0 if report.compatible else 1


if __name__ == "__main__":
    sys.exit(main())
