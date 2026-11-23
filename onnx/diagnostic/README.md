# Shape Inference Diagnostics

Opt-in provenance, conflict detection and gating for ONNX shape inference.

## Why

Ordinary shape inference has two blind spots that this package makes visible:

1. When two inference paths claim different values for the same dimension, the
   engine keeps the model valid by clearing that dimension to unknown, with no
   record of either claim. A control-flow merge (`If` branches) does this
   silently.
2. A node whose operator has no registered definition suppresses error
   messages for downstream nodes as well, so the originating node is hard to
   locate — and the suppression does not carry across subgraph boundaries.

This layer observes the *same* per-node inference primitive
(`onnx.shape_inference.infer_node_outputs`) once per node in graph order, inside
a single shared type context, and records how every dimension of every tensor
was reached.

## Entry point

```python
from onnx.diagnostic import infer_shapes_with_diagnostics

result = infer_shapes_with_diagnostics(
    model,
    strict=False,          # lenient: record issues, keep traversing
    max_conflicts=0,       # gate threshold
    max_unknown=None,      # optional unknown-dimension threshold
    report_path="diag.txt",  # atomic write; a failure raises
)
result.model          # ordinary inferred ModelProto
result.report         # DiagnosticReport with per-dimension provenance
result.gate           # GateResult(passed, exit_code, reasons)
```

`onnx.shape_inference.infer_shapes` is unchanged. With diagnostics disabled (the
default), nothing in this package runs and inference results are byte-for-byte
identical.

## Dimension states

Every dimension is classified into exactly one state:

| State | Meaning |
|---|---|
| `known` | a concrete `dim_value` was established and is not contested |
| `symbolic` | a static `dim_param` was declared or inferred |
| `unknown_initial` | declared (input/value_info) without value or symbol |
| `unknown_missing_op` | the producing node has no op definition |
| `unknown_conflict` | two inference paths claimed different concrete/symbolic values |
| `unknown_runtime` | only resolvable from runtime data |
| `unknown_control_flow` | a control-flow boundary prevents static convergence |

A conflict never overwrites either claim: both contributing sources (node index
+ op type, input declaration, or initializer) are retained, and the gate fails.

## Guarantees

- **Single context.** Conflicts are detected by comparing claims in one type
  table during one traversal, not by running inference twice and diffing.
- **Same source of truth.** Where the report records a conflict, the returned
  model does not present that dimension as a concrete value.
- **Privacy.** Records contain node indices, op types, local tensor names and
  dimension values/symbols only — never weight values or filesystem paths.
- **Isolation.** Each subgraph gets its own scope; the parent type table is
  read through declared boundary inputs and never mutated from inside.
- **Linear cost.** Per-node work is bounded by the node's fan-in, not the total
  graph size. Provenance per dimension is capped, so total memory stays linear
  in graph size.
- **No new dependencies.** Standard library plus the installed `onnx` package.

See `tests/python/diagnostic_test.py` for executable specifications of each
guarantee and `reports/` for sample rendered reports.
