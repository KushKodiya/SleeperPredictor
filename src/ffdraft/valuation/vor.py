"""M8 — value over replacement.

VOR is calibrated points minus the positional replacement level. It is the whole reason
a 260-point running back can be worth less than a 240-point tight end: what matters is
the gap to what you could have had for free at that position.
"""

from __future__ import annotations

import polars as pl

from ffdraft.contracts import assert_columns


def value_over_replacement(projections: pl.DataFrame, replacement: pl.DataFrame) -> pl.DataFrame:
    """Attach `replacement_points` and `vor` to each projected player.

    Ordering is invariant to a positive affine rescaling of the input points, because
    replacement is measured in those same units and rescales with them. (A general
    monotone transform does not preserve cross-position ordering — only the affine
    family does, since VOR is a difference.)
    """
    assert_columns(projections, {"gsis_id", "position", "projected_points"}, "vor.projections")
    assert_columns(replacement, {"position", "replacement_points"}, "vor.replacement")

    missing = set(projections["position"].unique()) - set(replacement["position"].unique())
    if missing:
        raise ValueError(
            f"no replacement level for position(s) {sorted(missing)}; "
            f"every projected position needs one or its players cannot be valued"
        )
    return (
        projections.join(replacement.select("position", "replacement_points"), on="position")
        .with_columns((pl.col("projected_points") - pl.col("replacement_points")).alias("vor"))
        .sort("vor", descending=True)
    )
