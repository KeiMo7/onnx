# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Shape-inference diagnostics.

This package adds provenance, conflict detection and gating on top of ONNX
shape inference without changing the existing inference entry points. It is
pure Python, depends only on the standard library and the already-installed
:mod:`onnx` package, and is opt-in:

* :func:`onnx.shape_inference.infer_shapes` keeps its existing signature,
  return type and bit-for-bit results.
* Diagnostics run only through :func:`infer_shapes_with_diagnostics`.
* With diagnostics enabled, the inferred model returned to the caller is the
  ordinary inference result; the :class:`DiagnosticReport` describes exactly
  how each dimension of each tensor was reached. The report and the inferred
  model are produced from the same per-node inference context.

Diagnostic records contain node indices, op types, tensor names local to the
model, dimension values and symbolic names. They never contain weight values
or filesystem paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import onnx
from onnx import ModelProto
from onnx.diagnostic._report import (
    evaluate_gate,
    render_report,
    write_report_atomic,
)
from onnx.diagnostic._tracer import ShapeInferenceTracer
from onnx.diagnostic.model import (
    ConflictRecord,
    DiagnosticReport,
    DimensionRecord,
    DimensionSource,
    DimensionStatus,
    GateResult,
    NodeDiagnostic,
    NodeStatus,
    SourceKind,
    SubgraphDiagnostic,
    TensorDiagnostic,
)

if TYPE_CHECKING:
    import os

__all__ = [
    "ConflictRecord",
    "DiagnosticReport",
    "DimensionRecord",
    "DimensionSource",
    "DimensionStatus",
    "GateResult",
    "NodeDiagnostic",
    "NodeStatus",
    "SourceKind",
    "SubgraphDiagnostic",
    "TensorDiagnostic",
    "infer_shapes_with_diagnostics",
    "render_report",
    "evaluate_gate",
    "write_report",
]


@dataclass
class DiagnosticResult:
    """The inferred model plus its diagnostic report and gate outcome."""

    model: ModelProto
    report: DiagnosticReport
    gate: GateResult


def infer_shapes_with_diagnostics(
    model: ModelProto,
    *,
    strict: bool = False,
    check_type: bool = False,
    data_prop: bool = False,
    max_conflicts: int = 0,
    max_unknown: int | None = None,
    report_path: str | os.PathLike[str] | None = None,
) -> DiagnosticResult:
    """Run shape inference and produce a provenance report.

    Arguments:
        model: a :class:`ModelProto`. The input model is not modified.
        strict: when ``True``, inference errors retain the ordinary strict
            behavior (the underlying inference call raises). When ``False``
            (lenient), issues are recorded in the report and traversal
            continues.
        check_type: forwarded to ordinary shape inference.
        data_prop: forwarded to ordinary shape inference.
        max_conflicts: gate threshold for retained dimension conflicts.
        max_unknown: optional gate threshold for unknown dimensions of any
            reason class. ``None`` disables this threshold.
        report_path: when given, the rendered report is written here,
            atomically, before this function returns. A write failure
            propagates and this function does not return a successful result.

    Returns:
        :class:`DiagnosticResult` bundling the inferred model, the report and
        the gate outcome.
    """
    if not isinstance(model, ModelProto):
        raise TypeError("infer_shapes_with_diagnostics expects a ModelProto")

    # Authoritative inference result. In strict mode this keeps the existing
    # raise behavior; in lenient mode the engine stops at failing nodes while
    # still returning the partially inferred model.
    inferred_model = onnx.shape_inference.infer_shapes(
        model,
        check_type=check_type,
        strict_mode=strict,
        data_prop=data_prop,
    )

    # Provenance is traced independently of the authoritative run but uses the
    # same per-node schema inference primitive, driven once per node in graph
    # order inside a single shared type context.
    tracer = ShapeInferenceTracer(model, strict=strict)
    report = tracer.trace()

    gate = evaluate_gate(
        report,
        max_conflicts=max_conflicts,
        max_unknown=max_unknown,
    )

    if report_path is not None:
        # A failed write must propagate: callers must not mistake a missing
        # report for a successful diagnostic run.
        write_report_atomic(render_report(report), report_path)

    return DiagnosticResult(model=inferred_model, report=report, gate=gate)


def write_report(report: DiagnosticReport, path: str | os.PathLike[str]) -> None:
    """Render ``report`` and write it atomically to ``path``."""
    write_report_atomic(render_report(report), path)
