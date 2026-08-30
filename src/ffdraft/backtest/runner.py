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

from dataclasses import dataclass, replace

import numpy as np
import polars as pl

from ffdraft.backtest.harness import (
    BASELINES,
    BacktestResult,
    Candidate,
    DraftContext,
    Strategy,
    assert_point_in_time,
    run_draft,
    score_roster_actuals,
    snake_order,
    static_vor,
)
from ffdraft.config import Config
from ffdraft.data import nflverse, projections
from ffdraft.data.adp import adp_format_from_scoring, fetch_adp, join_adp_to_crosswalk
from ffdraft.data.crosswalk import build_crosswalk
from ffdraft.lineup.slots import SlotConfig
from ffdraft.lineup.value import Player
from ffdraft.scoring import statlines
from ffdraft.scoring.engine import ScoringRules, score_players
from ffdraft.sim.availability import AvailabilityModel, availability_history, build_availability
from ffdraft.sim.opponent import FEATURE_NAMES, OpponentModel
from ffdraft.sim.outcomes import SimPlayer, bye_weeks, weekly_dispersion
from ffdraft.sim.rollout import LeagueContext, build_shortlist, rollout
from ffdraft.valuation.board import build_board
from ffdraft.valuation.replacement import rostered_depth


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
    high_by_player = (
        dict(zip(matched["gsis_id"].to_list(), matched["high"].to_list(), strict=True))
        if "high" in matched.columns
        else {}
    )

    candidates = tuple(
        Candidate(
            player_id=row["gsis_id"], name=row["name"], position=row["position"],
            adp=adp_by_player.get(row["gsis_id"]),
            adp_high=high_by_player.get(row["gsis_id"]),
            ecr=row["ecr"], vor=row["vor"],
            projected_points=row["projected_points"], team=row["team"],
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


def backtestable_seasons(
    cfg: Config, *, min_training_seasons: int, refresh: bool = False
) -> list[int]:
    """Seasons with both preseason ECR and finished outcomes to score against.

    A season must be over for its realized points to exist, and must have at least
    `min_training_seasons` prior ECR seasons behind it — the same floor `build_board`
    enforces. Returning a season the board cannot be built for only moves the failure
    from here to a crash several minutes into the run.
    """
    rankings = nflverse.ff_rankings(refresh=refresh)
    schedules = nflverse.schedules(
        [cfg.league.season, *range(cfg.league.season - 10, cfg.league.season)], refresh=refresh
    )
    available = sorted(projections.seasons_with_preseason_ecr(rankings, schedules))
    return [
        s
        for s in available
        if s < cfg.league.season
        and sum(1 for x in available if x < s) >= min_training_seasons
    ]


@dataclass(frozen=True)
class SimSeason:
    """Point-in-time simulation inputs for the Phase 8 rollout in one backtested season.

    Byes are a preseason fact about the season being drafted, like its ADP and its board,
    so they come from that season's own schedule. Everything *fitted* — the availability
    distribution, the weekly dispersion — is fit on prior seasons only and passes the
    leakage guard before it is used.
    """

    byes: dict[str, int]
    dispersion: dict[str, float]
    availability: AvailabilityModel


def sim_season(cfg: Config, rules: ScoringRules, *, season: int, refresh: bool = False) -> SimSeason:
    """Fit the season simulation's inputs on what was knowable before `season` kicked off."""
    prior = sorted(s for s in cfg.data.history_seasons if s < season)
    if not prior:
        raise ValueError(
            f"no configured history season precedes {season}; "
            f"history_seasons is {cfg.data.history_seasons}"
        )
    lookback = prior[-cfg.availability.lookback_seasons :]

    history = availability_history(
        nflverse.player_stats(lookback, refresh=refresh),
        nflverse.players(refresh=refresh),
        nflverse.snap_counts(lookback, refresh=refresh),
        nflverse.ff_playerids(refresh=refresh),
        positions=set(
            rostered_depth(
                cfg.league.fallback_roster_positions, cfg.flex_eligibility, teams=cfg.league.teams
            )
        ),
        rosterable_percentile=cfg.availability.rosterable_percentile,
    )
    assert_point_in_time(history, target_season=season, source=f"availability history for {season}")
    availability = build_availability(
        history,
        games_per_season=cfg.availability.games_per_season,
        age_bin_width=cfg.availability.age_bin_width,
        min_bin_count=cfg.availability.min_bin_count,
        workload_percentile=cfg.availability.workload_percentile_flag,
    )

    # Weekly spread comes from the most recently completed season — the same rule the live
    # path uses, so the backtest measures dispersion the way production will.
    last = prior[-1]
    weeks = score_players(
        statlines.player_week_stats(
            nflverse.player_stats([last], refresh=refresh), nflverse.pbp([last], refresh=refresh)
        ),
        rules,
    )
    assert_point_in_time(weeks, target_season=season, source=f"weekly dispersion for {season}")

    return SimSeason(
        byes=bye_weeks(nflverse.schedules([season], refresh=refresh), season=season),
        dispersion=weekly_dispersion(weeks.filter(pl.col("points") > 0)),
        availability=availability,
    )


def backtest_opponent(cfg: Config) -> OpponentModel:
    """The opponent the rollout assumes while backtesting: ADP order with jitter.

    The Phase 7 fit cannot be used here. It is fit on the owner's leaguemates' drafts,
    which are drafts from 2023 onward — for every backtested season that is either the
    season itself or later, so importing the fit would leak. The `adp_noise` rung needs
    no fit, and it is also the honest match: the backtest's opponents literally follow
    ADP, so a jitter around ADP is the family they are drawn from.
    """
    return OpponentModel(
        rung="adp_noise",
        tau=cfg.opponent_model.temperature_init,
        league_beta=np.zeros(len(FEATURE_NAMES)),
        adp_noise_sigma_rounds=cfg.opponent_model.adp_noise_sigma_rounds,
    )


def _as_sim_player(candidate: Candidate) -> SimPlayer:
    return SimPlayer(
        candidate.player_id, candidate.position, candidate.projected_points, team=candidate.team
    )


def rollout_strategy(
    sim: SimSeason,
    cfg: Config,
    slots: SlotConfig,
    opponent: OpponentModel,
    *,
    my_slot: int,
    seed: int,
    league: LeagueContext | None = None,
) -> Strategy:
    """The Phase 8 rollout, wired as a backtest strategy.

    Candidates are drawn from `context.legal()` so roster bookkeeping matches every
    baseline and the comparison measures the ranking method. The draft is replayed over
    the *whole* pool, though: opponents are not bound by the owner's open slots.
    """
    mine = [
        slot == my_slot
        for slot in snake_order(teams=cfg.league.teams, rounds=cfg.league.rounds)
    ]

    def strategy(context: DraftContext) -> Candidate:
        by_id = {c.player_id: c for c in context.available}
        pool = [_as_sim_player(c) for c in context.available]
        legal = {c.player_id for c in context.legal()}
        roster = [_as_sim_player(c) for c in context.roster]

        shortlist = build_shortlist(
            [p for p in pool if p.player_id in legal] or pool,
            roster,
            slots,
            top_n=cfg.simulation.shortlist_size,
            force_best_at_each_position=cfg.simulation.force_best_at_each_position,
        )
        recommendation = rollout(
            shortlist,
            pool,
            roster,
            slots,
            opponent,
            sim.availability,
            sim.byes,
            # Players without a published ADP keep the replay's own default rather than a
            # number invented here.
            adp_rounds={
                c.player_id: c.adp / cfg.league.teams
                for c in context.available
                if c.adp is not None
            },
            picks_remaining=mine[context.pick_number - 1 :],
            dispersion=sim.dispersion,
            seed=seed,
            n_sims=(
                cfg.simulation.n_sims_equity if league else cfg.simulation.n_sims_backtest
            ),
            n_scenarios=(
                cfg.simulation.n_scenarios_equity
                if league
                else cfg.simulation.n_scenarios_backtest
            ),
            league=league,
            n_weeks=cfg.availability.games_per_season + 1,
            time_budget_seconds=cfg.simulation.time_budget_seconds,
            static_fallback=_as_sim_player(static_vor(context)),
        )
        return by_id[recommendation.player_id]

    return strategy


def _slot_context(league: LeagueContext | None, slot: int) -> LeagueContext | None:
    """Re-point the league context at the draft slot currently being played.

    The bracket, schedule and seeding are league facts and do not move; which team is
    "the owner" does, once per slot.
    """
    if league is None:
        return None
    return replace(league, my_slot=slot - 1)  # run_draft slots are 1-based


def backtest_season_rollout(
    inputs: SeasonInputs,
    cfg: Config,
    slots: SlotConfig,
    result: BacktestResult,
    sim: SimSeason,
    opponent: OpponentModel,
    *,
    seed: int,
    league: LeagueContext | None = None,
    strategy_name: str = "rollout",
) -> dict[int, list[Candidate]]:
    """The rollout arm of one season, from every draft slot.

    Returns each slot's roster in pick order, so a caller can audit what was taken and
    when — the QB sanity gate is a claim about round 3, not about the final score.
    """
    drafted: dict[int, list[Candidate]] = {}
    for slot in range(1, cfg.league.teams + 1):
        rosters = run_draft(
            inputs.candidates,
            teams=cfg.league.teams,
            rounds=cfg.league.rounds,
            my_slot=slot,
            strategy=rollout_strategy(
                sim, cfg, slots, opponent, my_slot=slot, seed=seed,
                league=_slot_context(league, slot),
            ),
            roster_positions=cfg.league.fallback_roster_positions,
            flex_eligibility=cfg.flex_eligibility,
        )
        roster = [Player(c.player_id, c.position, 0.0) for c in rosters[slot]]
        points = score_roster_actuals(roster, inputs.weekly_actuals, slots)
        result.add(
            season=inputs.season, slot=slot, strategy=strategy_name, points=points
        )
        drafted[slot] = rosters[slot]
    return drafted
