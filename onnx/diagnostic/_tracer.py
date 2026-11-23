# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tracer that re-derives shape inference with full provenance.

The tracer drives the *same* per-node inference primitive used by
:func:`onnx.shape_inference.infer_node_outputs` (a thin wrapper over the C++
schema inference functions) once per node, in topological graph order. Every
decision therefore comes from the same inference context as ordinary shape
inference: conflicts are detected by comparing claims inside a single type
table during one traversal, never by running inference twice and diffing.

Design properties enforced here:

* Nodes whose op definition is missing get their own status and do not poison
  downstream nodes: traversal always continues, and later nodes are inferred
  normally whenever their inputs are available.
* Subgraphs are traversed in dedicated scopes. A subgraph only reads
  externally through its declared graph inputs (and values explicitly visible
  in outer scope via the node's own inputs); nothing written inside a
  subgraph mutates the parent type table.
* The tracer only stores node indices, op types, dimensions and symbols. It
  never inspects or records tensor data or filesystem paths.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from onnx import AttributeProto, defs, helper
from onnx.diagnostic.model import (
    ConflictRecord,
    DiagnosticReport,
    DimensionRecord,
    DimensionSource,
    DimensionStatus,
    NodeDiagnostic,
    NodeStatus,
    SourceKind,
    SubgraphDiagnostic,
    SubgraphPath,
    TensorDiagnostic,
)
from onnx.shape_inference import InferenceError, infer_node_outputs

if TYPE_CHECKING:
    from onnx import (
        FunctionProto,
        GraphProto,
        ModelProto,
        NodeProto,
        TensorProto,
        TypeProto,
    )
    from onnx.defs import OpSchema
    from onnx.onnx_pb import TensorShapeProto

# Per-dimension source lists are capped so total provenance memory stays linear
# in graph size even on pathological fan-in.
_MAX_SOURCES_PER_DIM = 8

# Control-flow op types whose graph attributes define isolation boundaries.
_CONTROL_FLOW_OPS = frozenset({"If", "Loop", "Scan", "SequenceMap"})
_GRAPH_ATTRS = frozenset(
    {"then_branch", "else_branch", "body", "attribute", "subgraph"}
)


def _is_graph_attribute(attr: AttributeProto) -> bool:
    return attr.type in (AttributeProto.GRAPH, AttributeProto.GRAPHS)


def _dim_state(
    type_dim: TensorShapeProto.Dimension,
) -> tuple[str, int | None, str | None]:
    which = type_dim.WhichOneof("value")
    if which == "dim_value":
        return "value", int(type_dim.dim_value), None
    if which == "dim_param":
        return "param", None, type_dim.dim_param
    return "empty", None, None


class _Scope:
    """One inference scope: a root graph or a single subgraph invocation."""

    def __init__(self, path: SubgraphPath, kind: str, parent_node: int | None) -> None:
        self.path = path
        self.kind = kind
        self.parent_node = parent_node
        # Canonical type table driving inference (the single context).
        self.types: dict[str, TypeProto] = {}
        # Provenance per tensor name: tuple of DimensionSource per axis, plus
        # the status assigned when the value was produced/declared.
        self.provenance: dict[str, list[list[DimensionSource]]] = {}
        self.statuses: dict[str, list[DimensionStatus]] = {}
        self.details: dict[str, list[str]] = {}
        # Names whose producing node has no op definition. Downstream nodes
        # reading these are treated as blocked, not independently erroneous.
        self.missing_outputs: set[str] = set()

    def sources_for(self, name: str, axis: int) -> list[DimensionSource]:
        prov = self.provenance.get(name)
        if prov is None or axis >= len(prov):
            return []
        return prov[axis]


class ShapeInferenceTracer:
    """Produces a :class:`DiagnosticReport` by tracing one model."""

    def __init__(self, model: ModelProto, *, strict: bool) -> None:
        self.model = model
        self.strict = strict
        self.report = DiagnosticReport(strict=strict)
        # Model-local functions, keyed the same way the C++ engine does.
        self._functions: dict[tuple[str, str, str], FunctionProto] = {}
        for fn in model.functions:
            key = (fn.domain, fn.name, fn.overload)
            self._functions[key] = fn
        self._opset_versions = {oi.domain: oi.version for oi in model.opset_import}
        # Built once; reused for every per-node inference call.
        self._opset_list = [
            helper.make_opsetid(domain, version)
            for domain, version in self._opset_versions.items()
        ]

    # ------------------------------------------------------------------ schema
    def _resolve_schema(
        self, node: NodeProto
    ) -> tuple[OpSchema | None, NodeStatus, str | None]:
        domain = node.domain
        version = self._opset_versions.get(domain)
        if version is None and domain == "":
            version = self._opset_versions.get("ai.onnx")
        if version is None:
            return (
                None,
                NodeStatus.MISSING_DEFINITION,
                (f"no opset import for domain {domain!r}"),
            )
        try:
            schema = defs.get_schema(node.op_type, version, domain)
        except defs.SchemaError:
            schema = None
        if schema is not None:
            return schema, NodeStatus.OK, None
        # A model-local function is a real definition, just one without a
        # registered schema. It must not be confused with a genuinely missing op.
        key = (node.domain, node.op_type, getattr(node, "overload", ""))
        if key in self._functions:
            return None, NodeStatus.FUNCTION_CALL, None
        return (
            None,
            NodeStatus.MISSING_DEFINITION,
            (f"no op definition registered for {node.op_type!r} in domain {domain!r}"),
        )

    # ------------------------------------------------------------- seeding
    def _seed_value(
        self,
        scope: _Scope,
        name: str,
        type_proto: TypeProto,
        kind: SourceKind,
        node_index: int | None = None,
    ) -> None:
        if not name:
            return
        if not type_proto.HasField("tensor_type"):
            # Non-tensor values are tracked by type only, without dimensions.
            scope.types[name] = type_proto
            return
        scope.types[name] = type_proto
        tensor = type_proto.tensor_type
        if not tensor.HasField("shape"):
            scope.provenance[name] = []
            scope.statuses[name] = []
            scope.details[name] = []
            return
        prov: list[list[DimensionSource]] = []
        statuses: list[DimensionStatus] = []
        details: list[str] = []
        for _axis, dim in enumerate(tensor.shape.dim):
            kind_of, value, param = _dim_state(dim)
            src = DimensionSource(
                kind=kind,
                node_index=node_index,
                op_type=None,
                subgraph_path=scope.path,
                claimed_value=value,
                claimed_param=param,
            )
            if kind_of == "value":
                statuses.append(DimensionStatus.KNOWN)
                details.append("")
            elif kind_of == "param":
                statuses.append(DimensionStatus.SYMBOLIC)
                details.append("")
            else:
                statuses.append(DimensionStatus.UNKNOWN_INITIAL)
                details.append("declared without a dimension")
            prov.append([src])
        scope.provenance[name] = prov
        scope.statuses[name] = statuses
        scope.details[name] = details

    # ------------------------------------------------------------- recording
    def _record_tensor(
        self, scope: _Scope, name: str, type_proto: TypeProto | None
    ) -> None:
        existing = self.report.tensors.get((scope.path, name))
        if existing is not None:
            return
        elem_type = None
        shape_dims = []
        if type_proto is not None and type_proto.HasField("tensor_type"):
            tensor = type_proto.tensor_type
            elem_type = int(tensor.elem_type) if tensor.elem_type else None
            if tensor.HasField("shape"):
                shape_dims = list(tensor.shape.dim)

        statuses = scope.statuses.get(name, [])
        details = scope.details.get(name, [])
        provenance = scope.provenance.get(name, [])
        # Provenance may establish a rank even when the engine's merged type
        # carries no shape (control-flow outputs are commonly cleared).
        rank = (
            len(shape_dims) if shape_dims else (len(provenance) if provenance else None)
        )
        dims: list[DimensionRecord] = []
        for axis in range(rank or 0):
            if axis < len(shape_dims):
                _, value, param = _dim_state(shape_dims[axis])
            else:
                value, param = None, None
            status = (
                statuses[axis]
                if axis < len(statuses)
                else DimensionStatus.UNKNOWN_INITIAL
            )
            detail = details[axis] if axis < len(details) else ""
            sources = provenance[axis] if axis < len(provenance) else []
            if status == DimensionStatus.KNOWN:
                concrete = next(
                    (s.claimed_value for s in sources if s.claimed_value is not None),
                    None,
                )
                if concrete is not None and value is None:
                    value = concrete
            dims.append(
                DimensionRecord(
                    index=axis,
                    status=status,
                    sources=list(sources),
                    value=value,
                    dim_param=param,
                    detail=detail,
                )
            )
        self.report.tensors[(scope.path, name)] = TensorDiagnostic(
            name=name,
            subgraph_path=scope.path,
            elem_type=elem_type,
            rank=rank,
            dimensions=dims,
        )

    def _add_conflict(
        self,
        scope: _Scope,
        name: str,
        axis: int,
        sources: list[DimensionSource],
        detail: str,
    ) -> None:
        self.report.conflicts.append(
            ConflictRecord(
                tensor=name,
                subgraph_path=scope.path,
                axis=axis,
                sources=list(sources),
                detail=detail,
            )
        )

    # ------------------------------------------------------------- merging
    def _merge_output(
        self,
        scope: _Scope,
        node: NodeProto,
        node_index: int,
        output_name: str,
        inferred: TypeProto,
        control_flow: bool,
    ) -> None:
        """Merge one node output into the scope, retaining every claim."""
        node_src = DimensionSource(
            kind=SourceKind.NODE,
            node_index=node_index,
            op_type=node.op_type,
            subgraph_path=scope.path,
        )
        if not inferred.HasField("tensor_type") or not inferred.tensor_type.HasField(
            "shape"
        ):
            # The engine produced no shape claim. If a control-flow merge already
            # established provenance for this output, keep it; do not overwrite it
            # with an empty-rank record. Otherwise record an unknown-rank value.
            if control_flow and scope.provenance.get(output_name):
                scope.types.setdefault(output_name, inferred)
                return
            scope.types.setdefault(output_name, inferred)
            scope.provenance.setdefault(output_name, [])
            scope.statuses.setdefault(output_name, [])
            scope.details.setdefault(output_name, [])
            return

        in_shape = inferred.tensor_type.shape
        # A previously seeded entry may carry no shape (untyped subgraph
        # outputs). Establish empty per-axis baselines so the claims below are
        # recorded instead of being dropped by the early-return above on a
        # later merge.
        prior_prov = scope.provenance.get(output_name)
        if prior_prov is None or (not prior_prov and len(in_shape.dim) > 0):
            prior_prov = [[] for _ in in_shape.dim]
            scope.provenance[output_name] = prior_prov
            scope.statuses[output_name] = [
                DimensionStatus.UNKNOWN_INITIAL for _ in in_shape.dim
            ]
            scope.details[output_name] = ["" for _ in in_shape.dim]
        scope.types.get(output_name)
        prior_status = scope.statuses.get(output_name)
        prior_details = scope.details.get(output_name)

        new_prov: list[list[DimensionSource]] = []
        new_status: list[DimensionStatus] = []
        new_details: list[str] = []

        for axis, dim in enumerate(in_shape.dim):
            kind_of, value, param = _dim_state(dim)
            src = DimensionSource(
                kind=SourceKind.NODE,
                node_index=node_index,
                op_type=node.op_type,
                subgraph_path=scope.path,
                claimed_value=value,
                claimed_param=param,
            )
            prior_sources = (
                list(prior_prov[axis])
                if prior_prov is not None and axis < len(prior_prov)
                else []
            )

            if kind_of == "value":
                # Conflict check against a prior concrete claim in this context.
                prior_concrete = next(
                    (s for s in prior_sources if s.claimed_value is not None), None
                )
                if prior_concrete is not None and prior_concrete.claimed_value != value:
                    sources = self._cap([*prior_sources, src])
                    new_prov.append(sources)
                    new_status.append(DimensionStatus.UNKNOWN_CONFLICT)
                    detail = (
                        f"conflicting dim values {prior_concrete.claimed_value} "
                        f"({prior_concrete.label()}) vs {value} ({src.label()})"
                    )
                    new_details.append(detail)
                    self._add_conflict(scope, output_name, axis, sources, detail)
                    continue
                # A concrete claim wins over symbolic/empty prior claims.
                sources = self._cap([*prior_sources, src])
                new_prov.append(sources)
                new_status.append(DimensionStatus.KNOWN)
                new_details.append("")
                continue

            if kind_of == "param":
                prior_param = next(
                    (s for s in prior_sources if s.claimed_param is not None), None
                )
                if prior_param is not None and prior_param.claimed_param != param:
                    sources = self._cap([*prior_sources, src])
                    new_prov.append(sources)
                    status = (
                        DimensionStatus.UNKNOWN_CONTROL_FLOW
                        if control_flow
                        else DimensionStatus.UNKNOWN_CONFLICT
                    )
                    detail = (
                        f"diverging symbolic dims {prior_param.claimed_param!r} "
                        f"({prior_param.label()}) vs {param!r} ({src.label()})"
                    )
                    new_details.append(detail)
                    new_status.append(status)
                    if status == DimensionStatus.UNKNOWN_CONFLICT:
                        self._add_conflict(scope, output_name, axis, sources, detail)
                    continue
                sources = self._cap([*prior_sources, src])
                new_prov.append(sources)
                # Preserve a prior KNOWN if the new claim is only symbolic.
                if (
                    prior_status
                    and axis < len(prior_status)
                    and prior_status[axis] == DimensionStatus.KNOWN
                ):
                    new_status.append(DimensionStatus.KNOWN)
                else:
                    new_status.append(DimensionStatus.SYMBOLIC)
                new_details.append("")
                continue

            # Empty dimension on the node output: the engine could not resolve it.
            sources = self._cap([*prior_sources, node_src])
            new_prov.append(sources)
            inherited = (
                prior_status[axis]
                if prior_status is not None and axis < len(prior_status)
                else None
            )
            new_status.append(self._classify_empty(inherited, control_flow))
            new_details.append(
                self._empty_detail(
                    prior_details[axis]
                    if prior_details is not None and axis < len(prior_details)
                    else "",
                    control_flow,
                )
            )

        scope.types[output_name] = inferred
        scope.provenance[output_name] = new_prov
        scope.statuses[output_name] = new_status
        scope.details[output_name] = new_details

    @staticmethod
    def _cap(sources: list[DimensionSource]) -> list[DimensionSource]:
        if len(sources) <= _MAX_SOURCES_PER_DIM:
            return sources
        # Keep the first claims and the most recent one; provenance stays bounded.
        return sources[: _MAX_SOURCES_PER_DIM - 1] + sources[-1:]

    @staticmethod
    def _classify_empty(
        inherited: DimensionStatus | None, control_flow: bool
    ) -> DimensionStatus:
        if inherited in (
            DimensionStatus.UNKNOWN_MISSING_OP,
            DimensionStatus.UNKNOWN_CONFLICT,
            DimensionStatus.UNKNOWN_RUNTIME,
            DimensionStatus.UNKNOWN_CONTROL_FLOW,
        ):
            return inherited
        if control_flow:
            return DimensionStatus.UNKNOWN_CONTROL_FLOW
        return DimensionStatus.UNKNOWN_RUNTIME

    @staticmethod
    def _empty_detail(prior_detail: str, control_flow: bool) -> str:
        if prior_detail:
            return prior_detail
        if control_flow:
            return "dimension does not converge across a control-flow boundary"
        return "dimension depends on runtime data or unresolved input"

    # ------------------------------------------------------------- control flow
    def _merge_control_flow_branches(
        self,
        node: NodeProto,
        index: int,
        scope: _Scope,
        child_scopes: dict[str, list[_Scope]],
    ) -> None:
        """Fold subgraph output claims into the parent node outputs.

        For ``If``, the i-th output of ``then_branch`` and ``else_branch`` feeds
        node output i. Conflicting concrete claims are both retained and flagged;
        this is the merge the C++ engine performs silently (it merely clears the
        dimension). For ``Loop``/``Scan`` the body output is carried over an
        iteration boundary, so non-converging dims are marked control-flow rather
        than hard conflicts.
        """
        is_if = node.op_type == "If"
        then_scopes = child_scopes.get("then_branch", [])
        else_scopes = child_scopes.get("else_branch", [])
        body_scopes = child_scopes.get("body", [])

        if is_if and then_scopes and else_scopes:
            then_scope, else_scope = then_scopes[0], else_scopes[0]
            then_graph = self._graph_attr(node, "then_branch")
            else_graph = self._graph_attr(node, "else_branch")
            if then_graph is None or else_graph is None:
                return
            for out_index, out_name in enumerate(node.output):
                if not out_name or out_index >= len(then_graph.output):
                    continue
                then_local = then_graph.output[out_index].name
                else_local = else_graph.output[out_index].name
                self._combine_branch_claims(
                    scope,
                    index,
                    node,
                    out_name,
                    then_scope,
                    then_local,
                    else_scope,
                    else_local,
                    hard_conflict=True,
                )
            return

        # Loop / Scan: body outputs map to node outputs after the loop-state
        # prefix. Claims crossing the boundary are control-flow unknown unless
        # they are concrete and identical on every observed path.
        if body_scopes:
            body_scope = body_scopes[0]
            body_graph = self._graph_attr(node, "body")
            if body_graph is None:
                return
            body_outputs = list(body_graph.output)
            for out_index, out_name in enumerate(node.output):
                if not out_name:
                    continue
                if node.op_type == "Loop":
                    # body output 0 is the loop condition; node output i maps
                    # to body output i+1.
                    body_out_index = out_index + 1
                else:
                    # Scan node outputs line up positionally with body outputs.
                    body_out_index = out_index
                if body_out_index >= len(body_outputs):
                    continue
                local_name = body_outputs[body_out_index].name
                self._carry_loop_claim(
                    scope, index, node, out_name, body_scope, local_name
                )

    @staticmethod
    def _graph_attr(node: NodeProto, name: str):
        for attr in node.attribute:
            if attr.name == name and attr.type == AttributeProto.GRAPH:
                return attr.g
        return None

    def _combine_branch_claims(
        self,
        scope: _Scope,
        _index: int,
        _node: NodeProto,
        out_name: str,
        scope_a: _Scope,
        local_a: str,
        scope_b: _Scope,
        local_b: str,
        *,
        hard_conflict: bool,
    ) -> None:
        prov_a = scope_a.provenance.get(local_a, [])
        prov_b = scope_b.provenance.get(local_b, [])
        status_a = scope_a.statuses.get(local_a, [])
        status_b = scope_b.statuses.get(local_b, [])
        rank = max(len(prov_a), len(prov_b))
        if rank == 0:
            return
        merged_prov: list[list[DimensionSource]] = []
        merged_status: list[DimensionStatus] = []
        merged_details: list[str] = []
        for axis in range(rank):
            srcs_a = list(prov_a[axis]) if axis < len(prov_a) else []
            srcs_b = list(prov_b[axis]) if axis < len(prov_b) else []
            concrete_a = next((s for s in srcs_a if s.claimed_value is not None), None)
            concrete_b = next((s for s in srcs_b if s.claimed_value is not None), None)
            all_sources = self._cap(srcs_a + srcs_b)
            if (
                hard_conflict
                and concrete_a is not None
                and concrete_b is not None
                and concrete_a.claimed_value != concrete_b.claimed_value
            ):
                detail = (
                    f"branch conflict: {concrete_a.claimed_value} "
                    f"({concrete_a.label()}) vs {concrete_b.claimed_value} "
                    f"({concrete_b.label()})"
                )
                merged_status.append(DimensionStatus.UNKNOWN_CONFLICT)
                merged_details.append(detail)
                self._add_conflict(scope, out_name, axis, all_sources, detail)
            elif hard_conflict:
                # Symbolic divergence across branches cannot be reconciled.
                param_a = next((s for s in srcs_a if s.claimed_param is not None), None)
                param_b = next((s for s in srcs_b if s.claimed_param is not None), None)
                if (
                    param_a is not None
                    and param_b is not None
                    and param_a.claimed_param != param_b.claimed_param
                ):
                    merged_status.append(DimensionStatus.UNKNOWN_CONTROL_FLOW)
                    merged_details.append(
                        "symbolic dimension does not converge across the "
                        f"then/else boundary ({param_a.claimed_param} vs {param_b.claimed_param})"
                    )
                elif concrete_a is not None and concrete_a.claimed_value == (
                    concrete_b.claimed_value if concrete_b else None
                ):
                    merged_status.append(DimensionStatus.KNOWN)
                    merged_details.append("")
                else:
                    inherited = self._pick_inherited(status_a, axis, status_b)
                    merged_status.append(
                        inherited or DimensionStatus.UNKNOWN_CONTROL_FLOW
                    )
                    merged_details.append(
                        "dimension does not converge across a control-flow boundary"
                    )
            else:
                merged_status.append(DimensionStatus.UNKNOWN_CONTROL_FLOW)
                merged_details.append("dimension carried across a loop/scan boundary")
            merged_prov.append(all_sources)
        scope.provenance[out_name] = merged_prov
        scope.statuses[out_name] = merged_status
        scope.details[out_name] = merged_details

    @staticmethod
    def _pick_inherited(status_a, axis, status_b):
        for candidate in (
            status_a[axis] if axis < len(status_a) else None,
            status_b[axis] if axis < len(status_b) else None,
        ):
            if candidate is not None and candidate.startswith("unknown"):
                return candidate
        return None

    def _carry_loop_claim(
        self,
        scope: _Scope,
        index: int,
        _node: NodeProto,
        out_name: str,
        body_scope: _Scope,
        local_name: str,
    ) -> None:
        prov = body_scope.provenance.get(local_name, [])
        body_status = body_scope.statuses.get(local_name, [])
        if not prov:
            return
        merged_prov: list[list[DimensionSource]] = []
        merged_status: list[DimensionStatus] = []
        merged_details: list[str] = []
        for axis, sources in enumerate(prov):
            merged_prov.append(self._cap(list(sources)))
            concrete = next((s for s in sources if s.claimed_value is not None), None)
            prior_status = body_status[axis] if axis < len(body_status) else None
            if prior_status == DimensionStatus.UNKNOWN_CONFLICT:
                merged_status.append(DimensionStatus.UNKNOWN_CONFLICT)
                merged_details.append(
                    f"conflict inside loop body carried to node#{index} output"
                )
            elif prior_status == DimensionStatus.UNKNOWN_MISSING_OP:
                merged_status.append(DimensionStatus.UNKNOWN_MISSING_OP)
                merged_details.append(
                    "loop-carried value depends on a node with no op definition"
                )
            else:
                # A loop state variable is allowed to change shape across
                # iterations, so even a body-internal concrete dimension cannot
                # be proven invariant without runtime information.
                merged_status.append(DimensionStatus.UNKNOWN_CONTROL_FLOW)
                detail = (
                    "loop-carried shape cannot be proven invariant across iterations"
                )
                if concrete is not None:
                    detail += f"; body claims {concrete.claimed_value}"
                merged_details.append(detail)
        scope.provenance[out_name] = merged_prov
        scope.statuses[out_name] = merged_status
        scope.details[out_name] = merged_details

    # ------------------------------------------------------------- traversal
    def trace(self) -> DiagnosticReport:
        root = _Scope((), "root", None)
        graph = self.model.graph
        # Inputs/value_info/outputs seed declared provenance first.
        for vi in graph.value_info:
            self._seed_value(root, vi.name, vi.type, SourceKind.VALUE_INFO)
        for vi in graph.input:
            self._seed_value(root, vi.name, vi.type, SourceKind.INPUT_DECL)
        for vi in graph.output:
            self._seed_value(root, vi.name, vi.type, SourceKind.INPUT_DECL)
        for init in graph.initializer:
            init_type = self._initializer_type(init)
            self._seed_value(root, init.name, init_type, SourceKind.INITIALIZER)
        self._walk_graph(graph, root)
        # Ensure every declared value has a tensor record even if never produced.
        for name in list(root.types):
            self._record_tensor(root, name, root.types[name])
        self._finalize_summary()
        return self.report

    # ------------------------------------------------------------- function calls
    def _walk_function_call(
        self,
        node: NodeProto,
        index: int,
        scope: _Scope,
        node_diag: NodeDiagnostic,
    ) -> None:
        """Trace a model-local function in an isolated body scope.

        Caller inputs are bound to the function's formal parameters by position.
        Body-local names never enter the caller's type table; the body's output
        claims are mapped back to the caller's output names by position, keeping
        their full provenance.
        """
        fn = self._functions[(node.domain, node.op_type, getattr(node, "overload", ""))]
        body_path = (*scope.path, index)
        body = _Scope(body_path, f"function:{fn.name}", index)

        # Bind actual caller inputs onto formal parameters by position.
        for pos, formal in enumerate(fn.input):
            if pos >= len(node.input) or not node.input[pos]:
                continue
            actual = node.input[pos]
            outer = scope.types.get(actual)
            if outer is not None:
                self._seed_value(body, formal, outer, SourceKind.INPUT_DECL)
        # Function prototypes carry value_info declarations but no initializers.
        for vi in fn.value_info:
            if vi.HasField("type"):
                self._seed_value(body, vi.name, vi.type, SourceKind.VALUE_INFO)

        self._walk_graph(fn, body)
        for name in list(body.types):
            self._record_tensor(body, name, body.types[name])

        # Map formal outputs back to caller outputs by position.
        for pos, caller_out in enumerate(node.output):
            if not caller_out or pos >= len(fn.output):
                continue
            formal_out = fn.output[pos]
            type_proto = body.types.get(formal_out)
            if type_proto is not None:
                scope.types[caller_out] = type_proto
            prov = body.provenance.get(formal_out, [])
            scope.provenance[caller_out] = [list(ax) for ax in prov]
            scope.statuses[caller_out] = list(body.statuses.get(formal_out, []))
            scope.details[caller_out] = list(body.details.get(formal_out, []))
            self._record_tensor(scope, caller_out, scope.types.get(caller_out))

        # Propagate any missing-op/error encountered inside the body to the call
        # node's own status without losing the function-call classification.
        body_node_diags = [n for n in self.report.nodes if n.subgraph_path == body_path]
        if any(n.status == NodeStatus.MISSING_DEFINITION for n in body_node_diags):
            node_diag.status = NodeStatus.FUNCTION_CALL
            node_diag.error = "function body contains a node with no op definition"
        elif any(n.status == NodeStatus.INFERENCE_ERROR for n in body_node_diags):
            node_diag.status = NodeStatus.INFERENCE_ERROR
            node_diag.error = "inference error inside function body"
        else:
            node_diag.status = NodeStatus.FUNCTION_CALL
        self.report.nodes.append(node_diag)

        sub_diag = SubgraphDiagnostic(
            path=body_path,
            kind=f"function:{fn.name}",
            parent_node_index=index,
            boundary_inputs=[n for n in node.input if n],
            boundary_outputs=[n for n in node.output if n],
        )
        for tname, statuses in body.statuses.items():
            for axis, dim_status in enumerate(statuses):
                if dim_status in (
                    DimensionStatus.UNKNOWN_CONTROL_FLOW,
                    DimensionStatus.UNKNOWN_CONFLICT,
                ):
                    sub_diag.undecidable_dims.append(f"{tname}#axis{axis}")
        self.report.subgraphs.append(sub_diag)

    @staticmethod
    def _initializer_type(init: TensorProto) -> TypeProto:
        return helper.make_tensor_type_proto(init.data_type, list(init.dims))

    def _walk_graph(self, graph: GraphProto | FunctionProto, scope: _Scope) -> None:
        for index, node in enumerate(graph.node):
            self._walk_node(node, index, scope)

    def _walk_node(self, node: NodeProto, index: int, scope: _Scope) -> None:
        schema, status, error = self._resolve_schema(node)
        control_flow = node.op_type in _CONTROL_FLOW_OPS

        # Subgraphs are always traversed in isolated scopes first, so branch
        # provenance exists before we interpret the node's merged output.
        subgraph_attrs = [a for a in node.attribute if _is_graph_attribute(a)]
        boundary_in = list(node.input)
        # attr name -> list of traversed child scopes (one per graph attribute).
        child_scopes: dict[str, list[_Scope]] = {}
        for attr in subgraph_attrs:
            graphs = list(attr.g) if attr.type == attr.GRAPHS else [attr.g]
            child_scopes[attr.name] = []
            for _sub_index, sub in enumerate(graphs):
                child_path = (*scope.path, index)
                child = _Scope(child_path, attr.name, index)
                child_scopes[attr.name].append(child)
                # Seed declared subgraph inputs from the parent context.
                for vi in sub.input:
                    outer = scope.types.get(vi.name)
                    if outer is not None:
                        self._seed_value(child, vi.name, outer, SourceKind.INPUT_DECL)
                    else:
                        self._seed_value(child, vi.name, vi.type, SourceKind.INPUT_DECL)
                # Subgraphs may implicitly capture outer-scope values (an If
                # branch commonly has zero declared inputs yet references parent
                # tensors). Inject them read-only: they cross the boundary as
                # declared boundary inputs and nothing written here flows back.
                declared = {vi.name for vi in sub.input}
                locally_defined = {
                    *(vi.name for vi in sub.value_info),
                    *(vi.name for vi in sub.output),
                    *(init.name for init in sub.initializer),
                    *(out for n in sub.node for out in n.output if out),
                }
                captured: list[str] = []
                for sub_node in sub.node:
                    for ref in sub_node.input:
                        if (
                            ref
                            and ref not in declared
                            and ref not in locally_defined
                            and ref in scope.types
                        ):
                            declared.add(ref)
                            captured.append(ref)
                            self._seed_value(
                                child, ref, scope.types[ref], SourceKind.INPUT_DECL
                            )
                for vi in sub.value_info:
                    self._seed_value(child, vi.name, vi.type, SourceKind.VALUE_INFO)
                for vi in sub.output:
                    self._seed_value(child, vi.name, vi.type, SourceKind.INPUT_DECL)
                self._walk_graph(sub, child)
                for name in list(child.types):
                    self._record_tensor(child, name, child.types[name])
                sub_diag = SubgraphDiagnostic(
                    path=child_path,
                    kind=attr.name,
                    parent_node_index=index,
                    boundary_inputs=[n for n in boundary_in if n] + captured,
                    boundary_outputs=[vi.name for vi in sub.output],
                )
                # Collect non-converging dims directly from the local child
                # scope (linear in subgraph size; no global table scan).
                for tname, statuses in child.statuses.items():
                    for axis, dim_status in enumerate(statuses):
                        if dim_status in (
                            DimensionStatus.UNKNOWN_CONTROL_FLOW,
                            DimensionStatus.UNKNOWN_CONFLICT,
                        ):
                            sub_diag.undecidable_dims.append(f"{tname}#axis{axis}")
                self.report.subgraphs.append(sub_diag)

        node_diag = NodeDiagnostic(
            index=index,
            subgraph_path=scope.path,
            op_type=node.op_type,
            domain=node.domain,
            inputs=tuple(i for i in node.input if i),
            outputs=tuple(o for o in node.output if o),
            status=NodeStatus.OK,
        )

        if schema is None and status == NodeStatus.FUNCTION_CALL:
            self._walk_function_call(node, index, scope, node_diag)
            return

        if schema is None:
            node_diag.status = status
            node_diag.error = error
            # Containment: record missing-definition outputs, then continue.
            for out in node.output:
                if not out:
                    continue
                self._mark_missing(scope, node, index, out)
            self.report.nodes.append(node_diag)
            return

        # For control-flow nodes, fold the per-branch claims into the parent
        # outputs *before* interpreting the engine's merged (often empty) shape,
        # so both branch sources survive and a real conflict is recorded.
        if control_flow and child_scopes:
            self._merge_control_flow_branches(node, index, scope, child_scopes)

        # If a required input is unavailable *because* an upstream node has no
        # op definition, this node is not independently at fault. The C++ engine
        # would suppress its error message here; we mirror that containment by
        # marking it skipped and inheriting the missing-op reason, instead of
        # counting a second, misleading inference error.
        blocked_by_missing = self._blocked_by_missing_op(scope, node)

        # Single per-node inference call in the current shared context. Only
        # the tensors this node actually references are serialized, keeping the
        # per-node cost bounded by fan-in rather than total graph size.
        input_map: dict[str, TypeProto] = {}
        for ref in node.input:
            if ref and ref not in input_map and ref in scope.types:
                input_map[ref] = scope.types[ref]
        opset_list = self._opset_list
        try:
            inferred_types = infer_node_outputs(
                schema,
                node,
                input_map,
                opset_imports=opset_list,
                ir_version=self.model.ir_version,
            )
        except InferenceError as exc:
            if blocked_by_missing:
                node_diag.status = NodeStatus.SKIPPED
                node_diag.error = (
                    "not inferred: an input comes from a node with no op definition"
                )
                for out in node.output:
                    if out:
                        self._mark_blocked_downstream(scope, node, index, out)
            else:
                node_diag.status = NodeStatus.INFERENCE_ERROR
                node_diag.error = self._sanitize(str(exc))
                for out in node.output:
                    if out:
                        self._mark_inference_error(
                            scope, node, index, out, control_flow
                        )
            self.report.nodes.append(node_diag)
            return
        except Exception as exc:  # noqa: BLE001 - containment is a hard requirement
            if blocked_by_missing:
                node_diag.status = NodeStatus.SKIPPED
                node_diag.error = (
                    "not inferred: an input comes from a node with no op definition"
                )
                for out in node.output:
                    if out:
                        self._mark_blocked_downstream(scope, node, index, out)
            else:
                node_diag.status = NodeStatus.INFERENCE_ERROR
                node_diag.error = self._sanitize(str(exc))
                for out in node.output:
                    if out:
                        self._mark_inference_error(
                            scope, node, index, out, control_flow
                        )
            self.report.nodes.append(node_diag)
            return

        for out in node.output:
            if not out:
                continue
            inferred = inferred_types.get(out)
            if inferred is None:
                if blocked_by_missing:
                    self._mark_blocked_downstream(scope, node, index, out)
                else:
                    self._mark_inference_error(scope, node, index, out, control_flow)
                continue
            self._merge_output(scope, node, index, out, inferred, control_flow)
            self._record_tensor(scope, out, scope.types.get(out))

        self.report.nodes.append(node_diag)

    def _blocked_by_missing_op(self, scope: _Scope, node: NodeProto) -> bool:
        for ref in node.input:
            if not ref:
                continue
            if ref in scope.missing_outputs:
                return True
            statuses = scope.statuses.get(ref)
            if statuses and any(
                s == DimensionStatus.UNKNOWN_MISSING_OP for s in statuses
            ):
                return True
        return False

    def _mark_blocked_downstream(
        self, scope: _Scope, node: NodeProto, index: int, out: str
    ) -> None:
        src = DimensionSource(
            kind=SourceKind.NODE,
            node_index=index,
            op_type=node.op_type,
            subgraph_path=scope.path,
        )
        prior = scope.types.get(out)
        rank = None
        if (
            prior is not None
            and prior.HasField("tensor_type")
            and prior.tensor_type.HasField("shape")
        ):
            rank = len(prior.tensor_type.shape.dim)
        if rank is None:
            scope.types.setdefault(out, helper.make_tensor_type_proto(0, None))
            scope.provenance.setdefault(out, [])
            scope.statuses.setdefault(out, [])
            scope.details.setdefault(out, [])
            scope.missing_outputs.add(out)
            return
        prov = [[*list(scope.sources_for(out, ax)), src] for ax in range(rank)]
        scope.provenance[out] = [self._cap(p) for p in prov]
        scope.statuses[out] = [DimensionStatus.UNKNOWN_MISSING_OP] * rank
        scope.details[out] = [
            f"downstream of a node with no op definition (node#{index} {node.op_type})"
        ] * rank
        scope.missing_outputs.add(out)

    def _mark_missing(
        self, scope: _Scope, node: NodeProto, index: int, out: str
    ) -> None:
        src = DimensionSource(
            kind=SourceKind.NODE,
            node_index=index,
            op_type=node.op_type,
            subgraph_path=scope.path,
        )
        prior = scope.types.get(out)
        rank = None
        if (
            prior is not None
            and prior.HasField("tensor_type")
            and prior.tensor_type.HasField("shape")
        ):
            rank = len(prior.tensor_type.shape.dim)
        if rank is None:
            scope.types.setdefault(out, helper.make_tensor_type_proto(0, None))
            scope.provenance.setdefault(out, [])
            scope.statuses.setdefault(out, [])
            scope.details.setdefault(out, [])
            scope.missing_outputs.add(out)
            return
        prov = [[*list(scope.provenance.get(out, [[]])[ax]), src] for ax in range(rank)]
        scope.provenance[out] = [self._cap(p) for p in prov]
        scope.statuses[out] = [DimensionStatus.UNKNOWN_MISSING_OP] * rank
        scope.details[out] = [
            f"producer node#{index}({node.op_type}) has no op definition"
        ] * rank
        scope.missing_outputs.add(out)

    def _mark_inference_error(
        self,
        scope: _Scope,
        node: NodeProto,
        index: int,
        out: str,
        control_flow: bool,
    ) -> None:
        src = DimensionSource(
            kind=SourceKind.NODE,
            node_index=index,
            op_type=node.op_type,
            subgraph_path=scope.path,
        )
        prior = scope.types.get(out)
        rank = None
        if (
            prior is not None
            and prior.HasField("tensor_type")
            and prior.tensor_type.HasField("shape")
        ):
            rank = len(prior.tensor_type.shape.dim)
        if rank is None:
            scope.types.setdefault(out, helper.make_tensor_type_proto(0, None))
            scope.provenance.setdefault(out, [])
            scope.statuses.setdefault(out, [])
            scope.details.setdefault(out, [])
            return
        prov = [[*list(scope.sources_for(out, ax)), src] for ax in range(rank)]
        scope.provenance[out] = [self._cap(p) for p in prov]
        inherited = scope.statuses.get(out, [])
        new_status: list[DimensionStatus] = []
        for ax in range(rank):
            prior_status = inherited[ax] if ax < len(inherited) else None
            if prior_status in (
                DimensionStatus.UNKNOWN_MISSING_OP,
                DimensionStatus.UNKNOWN_CONFLICT,
                DimensionStatus.UNKNOWN_CONTROL_FLOW,
            ):
                new_status.append(prior_status)
            elif control_flow:
                new_status.append(DimensionStatus.UNKNOWN_CONTROL_FLOW)
            else:
                new_status.append(DimensionStatus.UNKNOWN_RUNTIME)
        scope.statuses[out] = new_status
        scope.details[out] = [
            f"node#{index}({node.op_type}) inference did not produce a dimension"
        ] * rank

    @staticmethod
    def _sanitize(message: str) -> str:
        # Error strings must never leak paths or data: keep a bounded, single-line
        # description without filesystem-looking tokens.
        cleaned = " ".join(message.split())
        return cleaned[:300]

    # ------------------------------------------------------------- summary
    def _finalize_summary(self) -> None:
        summary = self.report.summary
        summary.num_tensors = len(self.report.tensors)
        summary.num_nodes = len(self.report.nodes)
        summary.num_subgraphs = len(self.report.subgraphs)
        summary.num_conflicts = len(self.report.conflicts)
        summary.num_missing_op_nodes = sum(
            1 for n in self.report.nodes if n.status == NodeStatus.MISSING_DEFINITION
        )
        summary.num_inference_errors = sum(
            1 for n in self.report.nodes if n.status == NodeStatus.INFERENCE_ERROR
        )
        breakdown: dict[str, int] = {}
        for tensor in self.report.tensors.values():
            for dim in tensor.dimensions:
                if dim.status.startswith("unknown"):
                    breakdown[dim.status.value] = breakdown.get(dim.status.value, 0) + 1
        summary.unknown_breakdown = dict(sorted(breakdown.items()))
