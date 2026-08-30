"""M15 — whether the model's projections are allowed to reach the board.

Promotion is a decision made by held-out evidence and carried with the evidence that
justified it. Nothing here infers permission from the model merely existing: an unproven
model contributes exactly nothing, and a promoted one *substitutes* for the
calibrated-ECR series rather than being averaged into it.

Substituting rather than blending is the point. The model earns its place by beating that
series on held-out error; averaging the two would dilute a measured win with the thing it
beat, at a weight nobody has measured.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from ffdraft.contracts import assert_columns
from ffdraft.models.evaluation import GateResult

CALIBRATED_ECR = "calibrated_ecr"
MODEL = "model"


@dataclass(frozen=True)
class ProjectionSource:
    """Which projection source is live, and the evidence that decided it."""

    name: str
    promoted: bool
    evidence: str
    soft_gate: str | None = None

    def describe(self) -> str:
        """One line naming the live source and why — for whoever is looking at a board."""
        headline = (
            "projections: MODEL (promoted)"
            if self.promoted
            else "projections: calibrated ECR board (model not promoted)"
        )
        lines = [f"{headline} — {self.evidence}"]
        if self.soft_gate:
            lines.append(f"  third-party comparison: {self.soft_gate}")
        return "\n".join(lines)


def decide_source(hard: GateResult, soft: GateResult | None = None) -> ProjectionSource:
    """The model is promoted only if the hard gate passed. A skipped soft gate cannot promote."""
    return ProjectionSource(
        name=MODEL if hard.passed else CALIBRATED_ECR,
        promoted=hard.passed,
        evidence=hard.reason,
        soft_gate=soft.reason if soft is not None else None,
    )


def apply_promotion(
    board: pl.DataFrame, model_projections: pl.DataFrame, source: ProjectionSource
) -> pl.DataFrame:
    """Substitute the model's projections into the board, if and only if it was promoted.

    Players the model does not cover keep the board's own projection: substituting a null
    would drop them off the board entirely, which is the silent-disappearance failure R4
    exists to prevent.
    """
    assert_columns(board, {"gsis_id", "projected_points"}, "promotion.apply_promotion")
    if not source.promoted:
        return board

    assert_columns(
        model_projections, {"gsis_id", "projected_points"}, "promotion.apply_promotion"
    )
    replacement = model_projections.select(
        "gsis_id", pl.col("projected_points").alias("_model_points")
    )
    return (
        board.join(replacement, on="gsis_id", how="left")
        .with_columns(
            pl.coalesce("_model_points", "projected_points").alias("projected_points")
        )
        .drop("_model_points")
    )
