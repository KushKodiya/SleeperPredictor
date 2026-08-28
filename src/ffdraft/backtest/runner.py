"""M14 — assembling a point-in-time backtest from real data.

`harness.py` is the pure part; this is the part that fetches. It builds one board per
backtested season using only what was knowable before that season kicked off, runs each
strategy from every draft slot, and scores the resulting rosters on what actually
happened.

Two honesty constraints are enforced here rather than assumed:

- Every decision input passes `assert_point_in_time` before a draft runs.
- Each season reports how many training seasons its board actually had. FantasyPros ECR
  begins in 2020, so a 2024 board is fit on four seasons where production demands six;
  the number travels with the result instead of being averaged away.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from ffdraft.backtest.harness import (
    BASELINES,
    BacktestResult,
    Candidate,
    assert_point_in_time,
    run_draft,
    score_roster_actuals,
)
from ffdraft.config import Config
from ffdraft.data import nflverse, projections
from ffdraft.data.adp import adp_format_from_scoring, fetch_adp, join_adp_to_crosswalk
from ffdraft.data.crosswalk import build_crosswalk
from ffdraft.lineup.slots import SlotConfig
from ffdraft.lineup.value import Player
from ffdraft.scoring import statlines
from ffdraft.scoring.engine import ScoringRules, score_players
from ffdraft.valuation.board import build_board


@dataclass(frozen=True)
class SeasonInputs:
    """One season's point-in-time board and the realized outcomes to score it against."""

    season: int
    candidates: tuple[Candidate, ...]
    weekly_actuals: pl.DataFrame
    training_seasons: int
    unresolved_adp: int


def season_inputs(
    cfg: Config,
    rules: ScoringRules,
    *,
    season: int,
    min_training_seasons: int,
    refresh: bool = False,
) -> SeasonInputs:
    """Everything a backtest of `season` needs, with the leakage guard applied."""
    board, diagnostics = build_board(
        cfg, rules, season=season, overrides=[], refresh=refresh,
        min_training_seasons=min_training_seasons,
    )
    # The board's own training set is the thing most likely to leak; check it explicitly.
    assert_point_in_time(
        pl.DataFrame({"season": list(diagnostics.calibration.training_seasons)}),
        target_season=season,
        source=f"calibration training seasons for {season}",
    )

    crosswalk = build_crosswalk(nflverse.ff_playerids(refresh=refresh))
    adp = fetch_adp(adp_format_from_scoring(rules.weights), cfg.league.teams, season)
    matched, unmatched = join_adp_to_crosswalk(
        adp, crosswalk, fuzzy_threshold=cfg.crosswalk.fuzzy_threshold
    )
    adp_by_player = dict(zip(matched["gsis_id"].to_list(), matched["adp"].to_list(), strict=True))

    candidates = tuple(
        Candidate(
            player_id=row["gsis_id"], name=row["name"], position=row["position"],
            adp=adp_by_player.get(row["gsis_id"]), ecr=row["ecr"], vor=row["vor"],
        )
        for row in board.iter_rows(named=True)
    )

    pbp = nflverse.pbp([season], refresh=refresh)
    weekly = score_players(
        statlines.player_week_stats(nflverse.player_stats([season], refresh=refresh), pbp), rules
    ).select("week", pl.col("player_id"), "points")

    return SeasonInputs(
        season=season,
        candidates=candidates,
        weekly_actuals=weekly,
        training_seasons=len(diagnostics.calibration.training_seasons),
        unresolved_adp=unmatched.height,
    )


def backtest_season(
    inputs: SeasonInputs, cfg: Config, slots: SlotConfig, result: BacktestResult
) -> None:
    """Every strategy from every draft slot for one season."""
    for name, strategy in sorted(BASELINES.items()):
        for slot in range(1, cfg.league.teams + 1):
            rosters = run_draft(
                inputs.candidates,
                teams=cfg.league.teams,
                rounds=cfg.league.rounds,
                my_slot=slot,
                strategy=strategy,
                roster_positions=cfg.league.fallback_roster_positions,
                flex_eligibility=cfg.flex_eligibility,
            )
            roster = [Player(c.player_id, c.position, 0.0) for c in rosters[slot]]
            points = score_roster_actuals(roster, inputs.weekly_actuals, slots)
            result.add(season=inputs.season, slot=slot, strategy=name, points=points)


def backtestable_seasons(cfg: Config, *, refresh: bool = False) -> list[int]:
    """Seasons with both preseason ECR and finished outcomes to score against.

    A season needs at least one prior ECR season to have a board at all, and must be
    over for its realized points to exist.
    """
    rankings = nflverse.ff_rankings(refresh=refresh)
    schedules = nflverse.schedules(
        [cfg.league.season, *range(cfg.league.season - 10, cfg.league.season)], refresh=refresh
    )
    available = sorted(projections.seasons_with_preseason_ecr(rankings, schedules))
    return [s for s in available if s < cfg.league.season and any(x < s for x in available)]
