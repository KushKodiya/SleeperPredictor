"""M15 — assembling the projection model's evidence from real data.

`projections.py` and `evaluation.py` are the pure parts; this is the part that fetches.
For each held-out season it fits the model on strictly earlier seasons, projects that
season, builds the incumbent calibrated-ECR board the same way Phase 6 does, and scores
both against what actually happened.

Everything fitted passes the leakage guard before it is used. The evaluation pool is the
`calibration.fit_pool` frame — players inside the top N at their position by preseason
expert rank — which is knowable before kickoff and so restricts nothing unfairly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl

from ffdraft.backtest.harness import assert_point_in_time
from ffdraft.config import Config
from ffdraft.data import nflverse
from ffdraft.models import evaluation, projections
from ffdraft.models.training import training_frame
from ffdraft.scoring import statlines
from ffdraft.scoring.engine import ScoringRules, score_players
from ffdraft.sim.availability import (
    PlayerSeason,
    availability_history,
    build_availability,
)
from ffdraft.valuation.board import build_board

# Opportunity columns the training frame needs but `score_players` does not carry through.
_OPPORTUNITY = ("attempts", "targets", "carries")


@dataclass(frozen=True)
class SeasonEvidence:
    """One held-out season's model projections, incumbent board, and realized points."""

    season: int
    frame: pl.DataFrame
    training_seasons: tuple[int, ...]
    pool_size: int


def scored_weeks(rules: ScoringRules, seasons: Sequence[int], *, refresh: bool = False):
    """Weekly scored lines with the opportunity columns joined back on."""
    stats = nflverse.player_stats(list(seasons), refresh=refresh)
    scored = score_players(
        statlines.player_week_stats(stats, nflverse.pbp(list(seasons), refresh=refresh)), rules
    )
    return (
        scored.join(
            stats.select("season", "week", "player_id", "team", *_OPPORTUNITY),
            on=["season", "week", "player_id"],
            how="left",
        ),
        stats,
    )


def _team_volumes(scored: pl.DataFrame, *, season: int) -> dict[str, dict[str, float]]:
    """Each team's opportunity in `season`, used as next season's projected volume."""
    totals = (
        scored.filter(pl.col("season") == season)
        .group_by("team")
        .agg([pl.col(c).fill_null(0).sum().cast(pl.Float64).alias(c) for c in _OPPORTUNITY])
    )
    return {
        column: dict(zip(totals["team"].to_list(), totals[column].to_list(), strict=True))
        for column in _OPPORTUNITY
    }


def _expected_games(cfg: Config, seasons: Sequence[int], *, refresh: bool = False):
    """Mean games played per position from M16 — durability is not re-modelled here."""
    history = availability_history(
        nflverse.player_stats(list(seasons), refresh=refresh),
        nflverse.players(refresh=refresh),
        nflverse.snap_counts(list(seasons), refresh=refresh),
        nflverse.ff_playerids(refresh=refresh),
        positions=set(projections.MODELLED_POSITIONS),
        rosterable_percentile=cfg.availability.rosterable_percentile,
    )
    model = build_availability(
        history,
        games_per_season=cfg.availability.games_per_season,
        age_bin_width=cfg.availability.age_bin_width,
        min_bin_count=cfg.availability.min_bin_count,
        workload_percentile=cfg.availability.workload_percentile_flag,
    )
    return model


def season_evidence(
    cfg: Config,
    rules: ScoringRules,
    *,
    season: int,
    seed: int,
    min_training_seasons: int = 2,
    refresh: bool = False,
) -> SeasonEvidence:
    """Fit on seasons before `season`, project it, and line the model up with the board."""
    history = sorted(s for s in cfg.data.history_seasons if s < season)
    if len(history) < 2:
        raise ValueError(
            f"projecting {season} needs at least two prior configured seasons, got {history}"
        )
    all_seasons = [*history, season]
    scored, stats = scored_weeks(rules, all_seasons, refresh=refresh)
    rosters = nflverse.rosters(all_seasons, refresh=refresh)
    schedules = nflverse.schedules(all_seasons, refresh=refresh)
    # Train on every season we can build a full feature row for, all strictly earlier.
    train_seasons = [s for s in history if s - 1 in history]
    draft_picks = nflverse.draft_picks(refresh=refresh)
    contracts = nflverse.contracts(refresh=refresh)
    snaps = nflverse.snap_counts(all_seasons, refresh=refresh)
    ids = nflverse.ff_playerids(refresh=refresh)
    sources = {
        "draft_picks": draft_picks,
        "contracts": contracts,
        "snap_counts": snaps,
        "crosswalk_ids": ids,
    }
    train = training_frame(
        scored, stats, rosters, schedules, seasons=train_seasons, **sources
    )
    assert_point_in_time(train, target_season=season, source=f"M15 training frame for {season}")
    model = projections.fit(train, seed=seed)

    frame = training_frame(
        scored, stats, rosters, schedules, seasons=[season], **sources
    )
    availability = _expected_games(cfg, history, refresh=refresh)

    # Expected games is the mean of M16's distribution, not a constant and not a
    # re-fit here. Age and prior workload are not carried on the training frame, so
    # every player pools to his position's distribution — the same simplification the
    # Phase 8 rollout already makes, and a known place this could be sharpened.
    by_position = {
        position: float((availability.pmf(PlayerSeason(position)) * availability.games).sum())
        for position in projections.MODELLED_POSITIONS
    }
    games = {
        row["gsis_id"]: by_position.get(
            row["position"], float(cfg.availability.games_per_season)
        )
        for row in frame.iter_rows(named=True)
    }
    volumes = _team_volumes(scored, season=season - 1)
    projected = projections.project(
        model,
        frame,
        expected_games=games,
        team_attempts=volumes["attempts"],
        team_targets=volumes["targets"],
        team_carries=volumes["carries"],
    )

    board, _ = build_board(
        cfg, rules, season=season, overrides=[], refresh=refresh,
        min_training_seasons=min_training_seasons,
    )
    pool = _fit_pool(board, cfg)
    actuals = frame.select("gsis_id", "actual_points").drop_nulls()

    joined = (
        pool.join(
            projected.select(
                "gsis_id",
                pl.col("projected_points").alias("model_points"),
                # The quantile head predicts season points directly rather than through
                # the decomposition. Carrying it lets the gate say *which* part is wrong
                # when the product loses: if p50 wins where the product does not, the
                # fault is in opportunity or team volume, not in the features.
                pl.col("p50").alias("model_median_points"),
            ),
            on="gsis_id", how="inner",
        )
        .join(actuals, on="gsis_id", how="inner")
        .with_columns(pl.lit(season).alias("season"))
    )
    return SeasonEvidence(
        season=season,
        frame=joined,
        training_seasons=model.training_seasons,
        pool_size=pool.height,
    )


def _fit_pool(board: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    """The evaluation frame: top-N by preseason expert rank within each position."""
    keep = []
    for position, cap in cfg.calibration.fit_pool.items():
        keep.append(
            board.filter((pl.col("position") == position) & (pl.col("ecr") <= cap)).select(
                "gsis_id",
                "position",
                pl.col("projected_points").alias("board_points"),
            )
        )
    return pl.concat(keep) if keep else board.head(0)


def run_gates(
    cfg: Config,
    rules: ScoringRules,
    *,
    seasons: Sequence[int],
    seed: int,
    min_training_seasons: int = 2,
    refresh: bool = False,
):
    """Both gates over every held-out season, with the per-season evidence retained."""
    evidence = []
    for season in seasons:
        evidence.append(
            season_evidence(
                cfg, rules, season=season, seed=seed,
                min_training_seasons=min_training_seasons, refresh=refresh,
            )
        )
    frame = pl.concat([e.frame for e in evidence]) if evidence else pl.DataFrame()
    hard = evaluation.hard_gate(frame)
    soft = evaluation.soft_gate(frame, list(seasons))
    return hard, soft, evidence
