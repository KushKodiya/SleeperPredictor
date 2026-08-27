"""M8 — tiers.

A tier break is a real drop, not an arbitrary group of five. Players are cut where the
gap to the next player exceeds that player's own standard error, so a cluster the
experts disagree about stays one tier while a genuine cliff splits.

The standard error is not invented: FantasyPros publishes `sd`, the spread of expert
ranks for each player. Pushing rank ± sd back through the calibration converts that
disagreement into points, which is the unit the gap is measured in.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import polars as pl

from ffdraft.contracts import assert_columns
from ffdraft.valuation.calibration import Calibration


def points_standard_error(ranked: pl.DataFrame, calibration: Calibration) -> pl.DataFrame:
    """Convert each player's expert-rank disagreement into a points standard error."""
    assert_columns(ranked, {"gsis_id", "position", "ecr", "sd"}, "tiers.points_standard_error")
    frames = []
    for position in sorted(set(ranked["position"].unique()) & set(calibration.positions())):
        rows = ranked.filter(pl.col("position") == position)
        ecr = rows["ecr"].to_numpy()
        sd = np.nan_to_num(rows["sd"].to_numpy(), nan=0.0)
        # A better rank is worth more, so the optimistic edge is ecr - sd.
        optimistic = calibration.predict(position, np.clip(ecr - sd, 1.0, None))
        pessimistic = calibration.predict(position, ecr + sd)
        frames.append(
            rows.with_columns(
                pl.Series("points_se", np.maximum((optimistic - pessimistic) / 2.0, 0.0))
            )
        )
    if not frames:
        return ranked.head(0).with_columns(pl.lit(0.0).alias("points_se"))
    return pl.concat(frames)


def assign_tiers(board: pl.DataFrame, *, value_column: str = "vor") -> pl.DataFrame:
    """Number the board into tiers, breaking where a gap exceeds the standard error.

    Walks the board in value order; a player starts a new tier when the drop from the
    player above exceeds that player's `points_se`.
    """
    assert_columns(board, {value_column, "points_se"}, "tiers.assign_tiers")
    # The caller owns the final order, because ties are broken on a column tiers knows
    # nothing about. Re-sorting here would undo that, so verify instead of assuming.
    ordered = board
    values = ordered[value_column].to_list()
    if any(b > a for a, b in pairwise(values)):
        raise ValueError(
            f"assign_tiers expects the board already sorted by {value_column} descending; "
            f"tiers would otherwise be assigned against the wrong neighbours"
        )
    errors = ordered["points_se"].to_list()

    tier, tiers = 1, []
    for i, value in enumerate(values):
        if i:
            gap = values[i - 1] - value
            if gap > (errors[i - 1] or 0.0):
                tier += 1
        tiers.append(tier)
    return ordered.with_columns(pl.Series("tier", tiers, dtype=pl.Int32))
