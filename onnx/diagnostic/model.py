# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Data model for shape-inference diagnostics.

All records in this module are *descriptions* of inference results. They never
hold tensor (weight) values or filesystem paths: the most specific node
identity recorded is a node index within its containing (sub)graph, plus the
op type. See :mod:`onnx.diagnostic` for the public entry points.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

# A subgraph path is a tuple of node indices (index within the containing
# graph) leading from the model root to the graph a record belongs to.
# The empty tuple denotes the root graph.
SubgraphPath = tuple[int, ...]


class DimensionStatus(enum.StrEnum):
    """Lifecycle state of a single dimension.

    The four ``UNKNOWN_*`` members are the reason classes required for
    triage; a dimension is never left merely "unknown" without one of
    these reasons attached.
    """

    # A concrete dim_value was established and is not contested.
    KNOWN = "known"
    # A static symbolic dim_param was declared/inferred.
    SYMBOLIC = "symbolic"
    # Declared (graph input / value_info) without value or symbol.
    UNKNOWN_INITIAL = "unknown_initial"
    # The producing node has no op definition (schema or model-local function).
    UNKNOWN_MISSING_OP = "unknown_missing_op"
    # Two inference paths claimed different concrete/symbolic values.
    UNKNOWN_CONFLICT = "unknown_conflict"
    # The dimension can only be known from runtime data.
    UNKNOWN_RUNTIME = "unknown_runtime"
    # A control-flow boundary prevents static convergence.
    UNKNOWN_CONTROL_FLOW = "unknown_control_flow"


class SourceKind(enum.StrEnum):
    """Where a dimension claim came from."""

    INPUT_DECL = "input_decl"
    INITIALIZER = "initializer"
    VALUE_INFO = "value_info"
    NODE = "node"
    # No claim has been made yet; used only before the first producer runs.
    NONE = "none"


class NodeStatus(enum.StrEnum):
    OK = "ok"
    MISSING_DEFINITION = "missing_definition"
    INFERENCE_ERROR = "inference_error"
    # The node was not inferred because a required input type is unavailable;
    # downstream handling continues normally.
    SKIPPED = "skipped"
    # The node resolves to a model-local function whose body is traced directly.
    FUNCTION_CALL = "function_call"


@dataclass(frozen=True)
class DimensionSource:
    """One provenance claim for a dimension.

    Node identity is the node's index inside its containing graph (plus op
    type for readability). Free-form node names are not stored here.
    """

    kind: SourceKind
    node_index: int | None = None
    op_type: str | None = None
    subgraph_path: SubgraphPath = ()
    claimed_value: int | None = None
    claimed_param: str | None = None

    def label(self) -> str:
        if self.kind == SourceKind.NODE:
            return f"node#{self.node_index}({self.op_type})"
        return str(self.kind.value)


@dataclass
class DimensionRecord:
    """Per-dimension diagnostic: state plus every contributing source."""

    index: int
    status: DimensionStatus
    sources: list[DimensionSource] = field(default_factory=list)
    value: int | None = None
    dim_param: str | None = None
    detail: str = ""

    @property
    def is_conflict(self) -> bool:
        return self.status == DimensionStatus.UNKNOWN_CONFLICT

    def display(self) -> str:
        if self.status == DimensionStatus.KNOWN:
            return str(self.value)
        if self.status == DimensionStatus.SYMBOLIC:
            return self.dim_param or "?"
        return self.status.value


@dataclass
class TensorDiagnostic:
    """All dimension records for one value in one (sub)graph scope."""

    name: str
    subgraph_path: SubgraphPath
    elem_type: int | None
    # ``None`` rank means the type itself has no shape (unknown rank).
    rank: int | None
    dimensions: list[DimensionRecord] = field(default_factory=list)

    def qualified_name(self) -> str:
        if not self.subgraph_path:
            return self.name
        return f"{self.name}@graph{list(self.subgraph_path)}"


@dataclass
class NodeDiagnostic:
    index: int
    subgraph_path: SubgraphPath
    op_type: str
    domain: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    status: NodeStatus
    # Sanitized, short message. Never contains tensor values or paths.
    error: str | None = None


@dataclass
class SubgraphDiagnostic:
    path: SubgraphPath
    kind: str
    parent_node_index: int | None
    # Values crossing the boundary, in either direction.
    boundary_inputs: list[str] = field(default_factory=list)
    boundary_outputs: list[str] = field(default_factory=list)
    # "qualified-tensor#axis" entries that could not converge statically.
    undecidable_dims: list[str] = field(default_factory=list)


@dataclass
class ConflictRecord:
    """A retained conflict: both claims survive, nothing is overwritten."""

    tensor: str
    subgraph_path: SubgraphPath
    axis: int
    sources: list[DimensionSource]
    detail: str


@dataclass
class ReportSummary:
    num_tensors: int = 0
    num_nodes: int = 0
    num_subgraphs: int = 0
    num_conflicts: int = 0
    num_missing_op_nodes: int = 0
    num_inference_errors: int = 0
    unknown_breakdown: dict[str, int] = field(default_factory=dict)


@dataclass
class GateResult:
    passed: bool
    exit_code: int
    reasons: list[str] = field(default_factory=list)


@dataclass
class DiagnosticReport:
    tensors: dict[tuple[SubgraphPath, str], TensorDiagnostic] = field(
        default_factory=dict
    )
    nodes: list[NodeDiagnostic] = field(default_factory=list)
    subgraphs: list[SubgraphDiagnostic] = field(default_factory=list)
    conflicts: list[ConflictRecord] = field(default_factory=list)
    summary: ReportSummary = field(default_factory=ReportSummary)
    strict: bool = False

    def tensor(self, path: SubgraphPath, name: str) -> TensorDiagnostic | None:
        return self.tensors.get((path, name))

    def conflicts_for(self, path: SubgraphPath, name: str) -> Sequence[ConflictRecord]:
        return [
            c for c in self.conflicts if c.subgraph_path == path and c.tensor == name
        ]
