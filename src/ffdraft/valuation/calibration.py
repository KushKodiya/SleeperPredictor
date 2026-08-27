"""M7 — turning expert consensus rank into calibrated points.

The PRD assumed FantasyPros would supply projected points to shrink. It does not:
`load_ff_rankings` publishes ranks only (see CLAUDE.md's R2 log). What it *does* have is
seven seasons of preseason ranks, and Phase 2 can score what actually happened. So the
calibration is fit directly from rank to outcome:

    isotonic( preseason ECR within position )  ->  actual season points

fit per position on seasons strictly before the target. The mapping is decreasing —
a better (lower) rank earns more points — and monotone by construction, so it cannot
invent a player who out-scores someone ranked above him.

This preserves the property the PRD cared about. A slope below 1.0 in a
projection→actual fit means projections are too spread out: the top is over-projected
and the middle under-projected. Fitting on realised outcomes reproduces that shrinkage
directly, and `shrinkage_ratio` measures it — the fitted spread is narrower than the
spread of the outcomes it was fit on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

from ffdraft.contracts import assert_columns

TRAINING_REQUIRED = {"season", "gsis_id", "position", "ecr", "actual_points"}


@dataclass(frozen=True)
class Calibration:
    """Per-position rank→points mappings, fit only on seasons before `target_season`."""

    target_season: int
    training_seasons: tuple[int, ...]
    models: dict[str, IsotonicRegression]
    _spread: dict[str, tuple[float, float]]  # position -> (sd of fitted, sd of actual)

    def positions(self) -> tuple[str, ...]:
        return tuple(sorted(self.models))

    def predict(self, position: str, ecr: np.ndarray | list[float]) -> np.ndarray:
        """Calibrated points for a position's ranks. Ranks outside the fit range clip."""
        if position not in self.models:
            raise KeyError(
                f"no calibration fit for position {position!r}; "
                f"fitted positions are {list(self.positions())}"
            )
        return self.models[position].predict(np.asarray(ecr, dtype=float))

    def shrinkage_ratio(self, position: str) -> float:
        """sd(fitted) / sd(actual) over the fit pool — the analogue of the <1.0 slope.

        Below 1.0 means the fit compresses the outcome spread rather than chasing the
        best season each rank ever produced, which is the effect the PRD's slope test
        was checking for.
        """
        fitted_sd, actual_sd = self._spread[position]
        return fitted_sd / actual_sd if actual_sd else 0.0


def actual_season_points(scored_weeks: pl.DataFrame, *, id_column: str = "player_id") -> pl.DataFrame:
    """Sum a season of scored weekly points into one row per player.

    `scored_weeks` is the output of `scoring.engine.score_players` or `score_defenses`,
    so the actuals carry the owner's own league scoring rather than a generic PPR total.
    """
    assert_columns(scored_weeks, {"season", id_column, "points"}, "calibration.actual_season_points")
    return (
        scored_weeks.group_by(["season", id_column])
        .agg(pl.col("points").sum().alias("actual_points"))
        .rename({id_column: "gsis_id"})
    )


def fit_calibration(
    training: pl.DataFrame,
    *,
    target_season: int,
    fit_pool: dict[str, int],
    min_training_seasons: int = 1,
) -> Calibration:
    """Fit the per-position rank→points mapping for `target_season`.

    `fit_pool` caps how deep the training set goes for the positions that name a cap
    (matching FFA's evaluation frame); a position with no configured cap trains on every
    ranked player it has.
    """
    assert_columns(training, TRAINING_REQUIRED, "calibration.fit_calibration")

    # Temporal leakage guard. Not a convention — the fit fails rather than quietly
    # learning from the season it is about to predict.
    leaked = training.filter(pl.col("season") >= target_season)
    if not leaked.is_empty():
        seasons = sorted(leaked["season"].unique().to_list())
        raise ValueError(
            f"calibration for {target_season} would leak: training contains "
            f"{leaked.height} row(s) from season(s) {seasons}. Every training row must "
            f"have season < {target_season}."
        )

    seasons = tuple(sorted(training["season"].unique().to_list()))
    if len(seasons) < min_training_seasons:
        raise ValueError(
            f"calibration needs at least {min_training_seasons} prior seasons, "
            f"got {len(seasons)}: {list(seasons)}"
        )

    models: dict[str, IsotonicRegression] = {}
    spread: dict[str, tuple[float, float]] = {}
    for position in sorted(training["position"].unique().to_list()):
        rows = training.filter(pl.col("position") == position)
        cap = fit_pool.get(position)
        if cap is not None:
            rows = rows.filter(pl.col("ecr") <= cap)
        rows = rows.filter(pl.col("ecr").is_not_null() & pl.col("actual_points").is_not_null())
        if rows.height < 2:
            continue

        x = rows["ecr"].to_numpy()
        y = rows["actual_points"].to_numpy()
        # increasing=False: rank 1 is the best player, so points fall as rank rises.
        model = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(x, y)
        models[position] = model
        spread[position] = (float(np.std(model.predict(x))), float(np.std(y)))

    if not models:
        raise ValueError("calibration produced no fitted positions; training data is too thin")
    return Calibration(
        target_season=target_season, training_seasons=seasons, models=models, _spread=spread
    )


def calibrated_points(ranked: pl.DataFrame, calibration: Calibration) -> pl.DataFrame:
    """Apply the fit to a season's ECR, returning one projection row per player."""
    assert_columns(ranked, {"gsis_id", "position", "ecr"}, "calibration.calibrated_points")
    frames = []
    for position in calibration.positions():
        rows = ranked.filter(pl.col("position") == position)
        if rows.is_empty():
            continue
        frames.append(
            rows.with_columns(
                pl.Series("projected_points", calibration.predict(position, rows["ecr"].to_numpy()))
            )
        )
    if not frames:
        return ranked.head(0).with_columns(pl.lit(0.0).alias("projected_points"))
    return pl.concat(frames).sort("projected_points", descending=True)
