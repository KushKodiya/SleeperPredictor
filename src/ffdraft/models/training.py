"""M15 — the point-in-time training frame behind the projection model.

One row per `(gsis_id, season)`: what was knowable before that season kicked off, and
what the player actually did. The features are the prior season's opportunity plus the
situation-change derivations from M17; the targets are the three parts the model
projects separately.

The `season` column here means **the season being projected**, and every feature in the
row is drawn from `season - 1` or earlier. That convention is what lets the Phase 6
leakage guard be pointed straight at this frame: a row whose season is the target season
is exactly the row that must not exist in a training set for it.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from ffdraft.contracts import assert_columns
from ffdraft.features import situation

# What the model needs to see and to predict. `season` is the projected season.
TRAINING_REQUIRED = {
    "season",
    "gsis_id",
    "position",
    "team",
    "prior_target_share",
    "prior_carry_share",
    "prior_points_per_game",
    "prior_games",
    "vacated_target_share",
    "vacated_carry_share",
    "team_implied_total",
    "actual_games",
    "actual_opportunity_per_game",
    "actual_points_per_opportunity",
    "actual_attempts",
    "actual_targets",
    "actual_carries",
    "actual_points",
}

FEATURE_COLUMNS = (
    "prior_target_share",
    "prior_carry_share",
    "prior_points_per_game",
    "prior_games",
    "vacated_target_share",
    "vacated_carry_share",
    "team_implied_total",
    "draft_round",
    "guaranteed_money",
    "head_coach_change",
    "qb_change",
)

# The three parts of the decomposition, kept separate on purpose: a season total
# conflates a durability miss with a usage miss with an efficiency miss.
TARGET_COLUMNS = (
    "actual_games",
    "actual_opportunity_per_game",
    "actual_points_per_opportunity",
)


def _season_outcomes(scored_weeks: pl.DataFrame, *, season: int) -> pl.DataFrame:
    """Per player for one season: games, opportunity, points, and the derived parts.

    Opportunity is pass attempts plus targets plus carries — every play on which the
    player could have scored. Pass attempts are not optional: a quarterback has almost no
    targets or carries, so leaving them out gives him a tiny denominator and an efficiency
    of eight points per "opportunity", which pools into the efficiency model and inflates
    everyone. It is the quantity the team-share constraint applies to.
    """
    assert_columns(
        scored_weeks,
        {"season", "player_id", "team", "position", "points",
         "attempts", "targets", "carries"},
        "training._season_outcomes",
    )
    weeks = scored_weeks.filter(pl.col("season") == season)
    return (
        weeks.group_by("player_id")
        .agg(
            pl.col("team").last().alias("team"),
            pl.col("position").last().alias("position"),
            pl.len().alias("games"),
            pl.col("points").fill_null(0).sum().alias("points"),
            pl.col("attempts").fill_null(0).sum().alias("attempts"),
            pl.col("targets").fill_null(0).sum().alias("targets"),
            pl.col("carries").fill_null(0).sum().alias("carries"),
        )
        .with_columns(
            (pl.col("attempts") + pl.col("targets") + pl.col("carries")).alias("opportunity")
        )
        .with_columns(
            (pl.col("points") / pl.col("games")).alias("points_per_game"),
            (pl.col("opportunity") / pl.col("games")).alias("opportunity_per_game"),
            pl.when(pl.col("opportunity") > 0)
            .then(pl.col("points") / pl.col("opportunity"))
            .otherwise(0.0)
            .alias("points_per_opportunity"),
        )
    )


def training_rows(
    scored_weeks: pl.DataFrame,
    player_stats: pl.DataFrame,
    rosters: pl.DataFrame,
    schedules: pl.DataFrame,
    *,
    season: int,
    draft_capital: pl.DataFrame | None = None,
    contracts: pl.DataFrame | None = None,
    qb_change: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """One row per player for `season`, features from `season - 1`, targets from `season`.

    Used for training (where the targets are known) and for projection (where they are
    not yet, and come back null).
    """
    prior = _season_outcomes(scored_weeks, season=season - 1)
    actual = _season_outcomes(scored_weeks, season=season)

    shares = situation.opportunity_shares(player_stats, season=season - 1)
    vacated = situation.vacated_opportunity(player_stats, rosters, prior_season=season - 1)
    implied = situation.team_implied_total(
        schedules, season=season, fallback_season=season - 1
    ).select("team", "team_implied_total")
    coaches = situation.head_coach_change(schedules, season=season).select(
        "team", "head_coach_change"
    )

    frame = (
        prior.rename({"player_id": "gsis_id"})
        .join(
            shares.rename({"player_id": "gsis_id"}).select(
                "gsis_id",
                pl.col("target_share").alias("prior_target_share"),
                pl.col("carry_share").alias("prior_carry_share"),
            ),
            on="gsis_id",
            how="left",
        )
        .rename(
            {"points_per_game": "prior_points_per_game", "games": "prior_games"}
        )
        .select("gsis_id", "team", "position", "prior_target_share", "prior_carry_share",
                "prior_points_per_game", "prior_games")
        .join(vacated.select("team", "vacated_target_share", "vacated_carry_share"),
              on="team", how="left")
        .join(implied, on="team", how="left")
        .join(coaches, on="team", how="left")
        .join(
            (qb_change.select("team", "qb_change") if qb_change is not None
             else pl.DataFrame(schema={"team": pl.String, "qb_change": pl.Boolean})),
            on="team", how="left",
        )
        .join(
            actual.select(
                pl.col("player_id").alias("gsis_id"),
                pl.col("games").alias("actual_games"),
                pl.col("opportunity_per_game").alias("actual_opportunity_per_game"),
                pl.col("attempts").alias("actual_attempts"),
                pl.col("targets").alias("actual_targets"),
                pl.col("carries").alias("actual_carries"),
                pl.col("points_per_opportunity").alias("actual_points_per_opportunity"),
                pl.col("points").alias("actual_points"),
            ),
            on="gsis_id",
            how="left",
        )
        .with_columns(pl.lit(season).alias("season"))
    )

    # Optional per-player capital. Absent is null, never zero: a player with no contract
    # on record has unknown guaranteed money, which is not the same as none (R4).
    if draft_capital is not None:
        frame = frame.join(
            draft_capital.select("gsis_id", "draft_round", "draft_pick"),
            on="gsis_id", how="left",
        )
    if contracts is not None:
        frame = frame.join(
            contracts.select("gsis_id", "guaranteed_money"), on="gsis_id", how="left"
        )

    for column in FEATURE_COLUMNS:
        if column not in frame.columns:
            frame = frame.with_columns(pl.lit(None, dtype=pl.Float64).alias(column))

    assert_columns(frame, TRAINING_REQUIRED, "training.training_rows")
    return frame.with_columns(
        pl.col("vacated_target_share").fill_null(0.0),
        pl.col("vacated_carry_share").fill_null(0.0),
    ).sort("gsis_id")


def training_frame(
    scored_weeks: pl.DataFrame,
    player_stats: pl.DataFrame,
    rosters: pl.DataFrame,
    schedules: pl.DataFrame,
    *,
    seasons: Sequence[int],
    draft_picks: pl.DataFrame | None = None,
    contracts: pl.DataFrame | None = None,
    snap_counts: pl.DataFrame | None = None,
    crosswalk_ids: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Stack `training_rows` over several projected seasons, keeping only complete rows.

    The optional sources are taken **raw** rather than pre-derived, because every one of
    them is season-dependent: which quarterback left is a fact about one transition, and
    "most recent contract" means something different for 2022 than for 2025. Deriving
    them once outside the loop and reusing the result across seasons would quietly hand
    the model information from the future.

    A row with no realized outcome cannot train anything, so it is dropped here rather
    than being imputed into the fit.
    """
    if not list(seasons):
        raise ValueError("no seasons given; a training frame needs at least one")

    frames = []
    for season in seasons:
        capital = (
            situation.draft_capital(draft_picks, before_season=season)
            if draft_picks is not None
            else None
        )
        money = (
            situation.guaranteed_money(contracts, before_season=season)
            if contracts is not None
            else None
        )
        quarterbacks = (
            situation.qb_change(
                snap_counts, rosters, crosswalk_ids, prior_season=season - 1
            )
            if snap_counts is not None and crosswalk_ids is not None
            else None
        )
        frames.append(
            training_rows(
                scored_weeks, player_stats, rosters, schedules,
                season=season, draft_capital=capital, contracts=money,
                qb_change=quarterbacks,
            )
        )
    return (
        pl.concat(frames, how="diagonal")
        .filter(pl.col("actual_points").is_not_null() & (pl.col("prior_games") > 0))
        .sort("season", "gsis_id")
    )
