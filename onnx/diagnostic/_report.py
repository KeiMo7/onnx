# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Human-readable report rendering and gating logic."""

from __future__ import annotations

import contextlib
import os
import tempfile
from typing import TYPE_CHECKING

from onnx.diagnostic.model import DimensionStatus, GateResult, NodeStatus

if TYPE_CHECKING:
    from onnx.diagnostic.model import DiagnosticReport

_STATUS_TAG = {
    DimensionStatus.KNOWN: "KNOWN",
    DimensionStatus.SYMBOLIC: "SYMBOLIC",
    DimensionStatus.UNKNOWN_INITIAL: "UNKNOWN/INITIAL",
    DimensionStatus.UNKNOWN_MISSING_OP: "UNKNOWN/MISSING-OP",
    DimensionStatus.UNKNOWN_CONFLICT: "UNKNOWN/CONFLICT",
    DimensionStatus.UNKNOWN_RUNTIME: "UNKNOWN/RUNTIME",
    DimensionStatus.UNKNOWN_CONTROL_FLOW: "UNKNOWN/CONTROL-FLOW",
}

_NODE_TAG = {
    NodeStatus.OK: "ok",
    NodeStatus.MISSING_DEFINITION: "missing-definition",
    NodeStatus.INFERENCE_ERROR: "inference-error",
    NodeStatus.SKIPPED: "skipped",
    NodeStatus.FUNCTION_CALL: "function-call",
}


def render_report(report: DiagnosticReport) -> str:
    """Render the report as deterministic plain text.

    The output contains only node indices, op types, tensor names local to the
    model, dimension values and symbolic names. No weight values and no
    filesystem paths are emitted.
    """
    lines: list[str] = []
    lines.append("ONNX Shape Inference Diagnostic Report")
    lines.append("=" * 40)
    lines.append(f"Mode: {'strict' if report.strict else 'lenient'}")
    s = report.summary
    lines.append(
        "Summary: "
        f"tensors={s.num_tensors} nodes={s.num_nodes} subgraphs={s.num_subgraphs} "
        f"conflicts={s.num_conflicts} missing_op={s.num_missing_op_nodes} "
        f"inference_errors={s.num_inference_errors}"
    )
    if s.unknown_breakdown:
        parts = ", ".join(f"{k}={v}" for k, v in s.unknown_breakdown.items())
        lines.append(f"Unknown dimensions by reason: {parts}")
    lines.append("")

    # Group tensors by their subgraph scope.
    paths = sorted({key[0] for key in report.tensors}, key=lambda p: (len(p), p))
    for path in paths:
        header = "root graph" if not path else f"subgraph via node path {list(path)}"
        lines.append(f"[{header}]")
        sub = report.subgraphs
        matching = [g for g in sub if g.path == path]
        for graph_info in matching:
            lines.append(
                f"  boundary graph attr={graph_info.kind} "
                f"parent_node#{graph_info.parent_node_index} "
                f"inputs=({', '.join(graph_info.boundary_inputs) or '-'}) "
                f"outputs=({', '.join(graph_info.boundary_outputs) or '-'})"
            )
            if graph_info.undecidable_dims:
                lines.append("  undecidable: " + ", ".join(graph_info.undecidable_dims))
        lines.append("  Tensors:")
        names = sorted(name for (p, name) in report.tensors if p == path)
        for name in names:
            tensor = report.tensors[(path, name)]
            lines.append(f"    {name}")
            if tensor.rank is None:
                lines.append("      rank: unknown")
                continue
            if not tensor.dimensions:
                lines.append("      rank: 0 (scalar)")
                continue
            for dim in tensor.dimensions:
                tag = _STATUS_TAG[dim.status]
                srcs = ", ".join(src.label() for src in dim.sources) or "-"
                shown = dim.display()
                line = f"      axis {dim.index}: {shown} [{tag}] sources: {srcs}"
                lines.append(line)
                if dim.detail:
                    lines.append(f"        note: {dim.detail}")
        lines.append("  Nodes:")
        for node in report.nodes:
            if node.subgraph_path != path:
                continue
            tag = _NODE_TAG[node.status]
            lines.append(
                f"    node#{node.index} {node.op_type} domain={node.domain or 'ai.onnx'} "
                f"inputs=({', '.join(node.inputs) or '-'}) "
                f"outputs=({', '.join(node.outputs) or '-'}) [{tag}]"
            )
            if node.error:
                lines.append(f"      error: {node.error}")
        lines.append("")

    lines.append("Conflicts:")
    if not report.conflicts:
        lines.append("  (none)")
    for conflict in report.conflicts:
        where = "root" if not conflict.subgraph_path else list(conflict.subgraph_path)
        srcs = "; ".join(
            f"{src.label()} -> value={src.claimed_value if src.claimed_value is not None else src.claimed_param}"
            for src in conflict.sources
        )
        lines.append(
            f"  tensor={conflict.tensor} scope={where} axis={conflict.axis}: {conflict.detail}"
        )
        lines.append(f"    retained sources: {srcs}")
    lines.append("")
    return "\n".join(lines)


def evaluate_gate(
    report: DiagnosticReport,
    *,
    max_conflicts: int = 0,
    max_unknown: int | None = None,
) -> GateResult:
    """Apply gating thresholds to a report.

    ``max_unknown=None`` disables the unknown-dimension threshold. Every
    unknown counted here carries an explicit reason class, so there are no
    unexplained unknowns to gate separately.
    """
    reasons: list[str] = []
    if report.summary.num_conflicts > max_conflicts:
        reasons.append(
            f"conflicts {report.summary.num_conflicts} exceed threshold {max_conflicts}"
        )
    if max_unknown is not None:
        total_unknown = sum(report.summary.unknown_breakdown.values())
        if total_unknown > max_unknown:
            reasons.append(
                f"unknown dimensions {total_unknown} exceed threshold {max_unknown}"
            )
    return GateResult(
        passed=not reasons,
        exit_code=0 if not reasons else 1,
        reasons=reasons,
    )


def write_report_atomic(text: str, path: str | os.PathLike[str]) -> None:
    """Write a report such that a failed write never looks like success.

    The file is written to a temporary name in the same directory and renamed
    into place only after a full flush and fsync; any error propagates to the
    caller instead of being swallowed.
    """
    path = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".diag-report-", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)
        raise
