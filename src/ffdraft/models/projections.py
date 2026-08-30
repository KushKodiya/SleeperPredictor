"""M15 — the projection model.

Three parts, fit separately because each has different predictability and a single
season-total model conflates them: a durability miss, a usage miss and an efficiency
miss all look identical in season points.

    projection = games_played x opportunity_per_game x points_per_opportunity

Games played is **not** modelled here. M16 already returns a calibrated distribution over
games, and the PRD is explicit that a per-player injury classifier is not worth building;
this module consumes that distribution's mean rather than re-learning it badly.

Opportunity is where the constraint lives. Shares are fit in log space and turned into
shares by a softmax **within each team**, so a team's projected shares sum to one by
construction rather than by a cleanup pass. Without it the model happily projects four
pass-catchers on one team past the volume that team actually throws — an arithmetically
impossible roster that no per-player error metric would ever flag.

Every stochastic step takes an explicit seed and LightGBM runs deterministically (R7).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import polars as pl

from ffdraft.contracts import assert_columns
from ffdraft.models.training import FEATURE_COLUMNS

# The positions a fantasy roster actually starts. Punters and offensive linemen appear in
# the stat feed and would otherwise be fit as if anyone might draft them.
MODELLED_POSITIONS = ("QB", "RB", "WR", "TE")

QUANTILES = (0.1, 0.5, 0.9)

# Share of a team's opportunity below which a player is treated as having none. Log-space
# fitting needs a floor, and zero has no logarithm.
MIN_SHARE = 1e-6


@dataclass(frozen=True)
class ProjectionModel:
    """A fitted three-part model plus its quantile heads."""

    share_models: dict[str, lgb.Booster]
    points_per_opportunity: lgb.Booster
    quantiles: dict[float, lgb.Booster]
    feature_names: tuple[str, ...]
    training_seasons: tuple[int, ...] = ()
    positions: tuple[str, ...] = MODELLED_POSITIONS


def _matrix(frame: pl.DataFrame, features: Sequence[str]) -> np.ndarray:
    """Feature matrix as float64, with nulls left as NaN for LightGBM to route."""
    return (
        frame.select(
            [pl.col(name).cast(pl.Float64, strict=False) for name in features]
        )
        .to_numpy()
        .astype(np.float64)
    )


def _booster(
    x: np.ndarray, y: np.ndarray, *, seed: int, objective: str, alpha: float | None = None
) -> lgb.Booster:
    """One small, deterministic LightGBM model.

    The training frames here are a few thousand rows, so the model is deliberately small:
    a deeper one memorises a handful of seasons rather than learning a situation.
    """
    params = {
        "objective": objective,
        "num_leaves": 15,
        "min_data_in_leaf": 40,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "seed": seed,
        "deterministic": True,
        "force_row_wise": True,
        "verbosity": -1,
    }
    if alpha is not None:
        params["alpha"] = alpha
    return lgb.train(params, lgb.Dataset(x, label=y), num_boost_round=200)


def fit(
    frame: pl.DataFrame,
    *,
    seed: int,
    positions: Sequence[str] = MODELLED_POSITIONS,
) -> ProjectionModel:
    """Fit the three parts and the quantile heads on a point-in-time training frame."""
    assert_columns(
        frame,
        {"season", "gsis_id", "position", "team", "actual_points", *FEATURE_COLUMNS},
        "projections.fit",
    )
    usable = frame.filter(
        pl.col("position").is_in(list(positions)) & pl.col("actual_points").is_not_null()
    )
    if usable.is_empty():
        raise ValueError(
            f"no training rows for positions {sorted(positions)}; "
            f"frame has {sorted(set(frame['position'].to_list()))}"
        )

    x = _matrix(usable, FEATURE_COLUMNS)

    # Opportunity shares, fit in log space so the softmax that follows is a
    # renormalisation of the model's own scale rather than a rescue of it.
    share_models: dict[str, lgb.Booster] = {}
    for kind, column in (
        ("attempts", "actual_attempts"),
        ("targets", "actual_targets"),
        ("carries", "actual_carries"),
    ):
        realised = _realised_share(usable, column)
        share_models[kind] = _booster(
            x, np.log(np.clip(realised, MIN_SHARE, None)), seed=seed, objective="regression"
        )

    ppo = _booster(
        x,
        usable["actual_points_per_opportunity"].fill_null(0.0).to_numpy(),
        seed=seed,
        objective="regression",
    )
    quantiles = {
        q: _booster(
            x, usable["actual_points"].to_numpy(), seed=seed, objective="quantile", alpha=q
        )
        for q in QUANTILES
    }

    return ProjectionModel(
        share_models=share_models,
        points_per_opportunity=ppo,
        quantiles=quantiles,
        feature_names=tuple(FEATURE_COLUMNS),
        training_seasons=tuple(sorted(set(usable["season"].to_list()))),
        positions=tuple(positions),
    )


def _realised_share(frame: pl.DataFrame, column: str) -> np.ndarray:
    """The share of its team's `column` that a player actually took, in that season.

    Targets and carries get their own denominator. Sharing one would make the two share
    models identical, and the spec requires each to reconcile to its own team volume.
    """
    shares = frame.with_columns(
        pl.when(pl.col(column).fill_null(0).sum().over("team", "season") > 0)
        .then(
            pl.col(column).fill_null(0)
            / pl.col(column).fill_null(0).sum().over("team", "season")
        )
        .otherwise(MIN_SHARE)
        .alias("_share")
    )
    return shares["_share"].to_numpy()


def project(
    model: ProjectionModel,
    frame: pl.DataFrame,
    *,
    expected_games: dict[str, float],
    team_attempts: dict[str, float],
    team_targets: dict[str, float],
    team_carries: dict[str, float],
) -> pl.DataFrame:
    """Project every player in `frame`, with the three parts kept inspectable.

    `expected_games` is the availability model's mean games per player — this module does
    not model durability. `team_targets` and `team_carries` are each team's projected
    volume, which the constrained shares are multiplied by.

    Targets and carries are softmaxed **separately within each team**, so each share
    column sums to 1.0 per team by construction and each projected volume reconciles to
    that team's own total. The alternative — normalising after the fact — silently
    rescales every player on a team to fix one outlier.
    """
    assert_columns(
        frame, {"gsis_id", "team", "position", *model.feature_names}, "projections.project"
    )
    usable = frame.filter(pl.col("position").is_in(list(model.positions)))
    if usable.is_empty():
        return usable

    x = _matrix(usable, model.feature_names)
    out = usable.select("gsis_id", "team", "position").with_columns(
        pl.Series(
            "points_per_opportunity",
            np.clip(model.points_per_opportunity.predict(x), 0.0, None),
        ),
        pl.Series(
            "games", [float(expected_games.get(pid, np.nan)) for pid in usable["gsis_id"]]
        ),
    )

    for kind, share, projected, volume in (
        ("attempts", "attempt_share", "projected_attempts", team_attempts),
        ("targets", "target_share", "projected_targets", team_targets),
        ("carries", "carry_share", "projected_carries", team_carries),
    ):
        raw = model.share_models[kind].predict(x)
        out = (
            out.with_columns(pl.Series("_raw", raw))
            # Subtracting the team max before exponentiating is the standard numerically
            # stable softmax; it cancels in the ratio.
            .with_columns(
                (pl.col("_raw") - pl.col("_raw").max().over("team")).exp().alias("_e")
            )
            .with_columns((pl.col("_e") / pl.col("_e").sum().over("team")).alias(share))
            .with_columns(
                (
                    pl.col(share)
                    * pl.col("team").replace_strict(
                        volume, default=0.0, return_dtype=pl.Float64
                    )
                ).alias(projected)
            )
            .drop("_raw", "_e")
        )

    out = out.with_columns(
        (
            pl.col("projected_attempts")
            + pl.col("projected_targets")
            + pl.col("projected_carries")
        ).alias("opportunity")
    ).with_columns(
        pl.when(pl.col("games") > 0)
        .then(pl.col("opportunity") / pl.col("games"))
        .otherwise(0.0)
        .alias("opportunity_per_game")
    ).with_columns(
        (
            pl.col("games")
            * pl.col("opportunity_per_game")
            * pl.col("points_per_opportunity")
        ).alias("projected_points")
    )

    for q, booster in sorted(model.quantiles.items()):
        out = out.with_columns(pl.Series(f"p{int(q * 100)}", booster.predict(x)))

    # A quantile head can cross its neighbour on a thin training set. The ordering is a
    # property of what quantiles mean, not something the fit is entitled to violate, so
    # it is enforced rather than reported as a p10 above a p90 nobody can interpret.
    return (
        out.with_columns(
            pl.min_horizontal("p10", "p50", "p90").alias("p10"),
            pl.max_horizontal("p10", "p50", "p90").alias("p90"),
        )
        .with_columns(pl.col("p50").clip(pl.col("p10"), pl.col("p90")).alias("p50"))
        .sort("projected_points", descending=True)
    )
