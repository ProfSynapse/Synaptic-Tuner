"""
MechInterp: generic mechanistic-interpretation cells for the Synaptic Tuner.

This package gives any researcher a project-agnostic toolkit for reading and
writing model internals during generation:

  - intervention: forward-hook edits to the residual stream (additive push and
    erase-and-write setpoint laws), with per-row selection, per-batch-element
    strength, position policies, and readback instrumentation.
  - extraction: generation with hidden-state capture at configurable token
    positions and layer ranges, written to safetensors with a manifest.
  - probe: linear readout fitting (configurable dim-reduction + classifier),
    out-of-fold scoring, and direction freezing to JSON.
  - stats: gate primitives (flip counts, kill-difference bootstrap CI,
    permutation p-value, AUROC floor) and a declarative gates.yaml evaluator.
  - grading: a thin interface for pluggable, project-supplied graders.

The public surface is the recipe-driven cell model (see cell.py) plus the
individual libraries, all reachable from the tuner CLI verbs.

Vocabulary here is deliberately neutral: direction, readout, selection score,
intervention, cell. Nothing in this package is tied to a particular research
question.
"""

__all__ = [
    "config",
]
