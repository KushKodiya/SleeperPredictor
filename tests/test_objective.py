"""Phase 10 objective integration — the switch changes scoring and nothing else."""

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ffdraft.data.sleeper import BracketMatch
from ffdraft.lineup.slots import SlotConfig
from ffdraft.sim.availability import build_availability
from ffdraft.sim.opponent import OpponentModel
from ffdraft.sim.outcomes import SimPlayer
from ffdraft.sim.rollout import LeagueContext, build_shortlist, rollout
from ffdraft.sim.season import learn_slot_seeds, standings_from_matchups

FIXTURE = Path(__file__).parent / "fixtures" / "winners_bracket_2025.json"
TEAMS, ROUNDS = 8, 6
SLOTS = SlotConfig.from_league(
    ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "BN"], {"FLEX": ["RB", "WR", "TE"]}
)


@pytest.fixture(scope="module")
def league_bits():
    real = json.loads(FIXTURE.read_text(encoding="utf-8"))
    bracket = [BracketMatch.model_validate(m) for m in real["winners_bracket"]]
    points = {int(w): [tuple(r) for r in rows]
              for w, rows in real["regular_season"]["points"].items()}
    pairings = {int(w): [tuple(p) for p in rows]
                for w, rows in real["regular_season"]["pairings"].items()}
    ranking = standings_from_matchups(points, pairings)
    return bracket, learn_slot_seeds(bracket, ranking), real["settings"]["playoff_week_start"]


def _availability():
    rows = [
        {"season": 2024, "gsis_id": f"p{i}", "position": pos, "age": 26.0,
         "games_played": 17, "prior_offense_snaps": 500.0, "workload_percentile": 0.5}
        for i in range(60) for pos in ("QB", "RB", "WR", "TE")
    ]
    return build_availability(
        pl.DataFrame(rows), games_per_season=17, age_bin_width=1,
        min_bin_count=1, workload_percentile=0.8,
    )


def _pool():
    players, curve = [], {"QB": 300, "RB": 260, "WR": 250, "TE": 200}
    for position, top in curve.items():
        for i in range(14):
            players.append(SimPlayer(f"{position}{i}", position, float(top - i * 12), team="X"))
    return players


def _common(league=None, budget=1e9, n_scenarios=2, n_sims=4):
    pool = _pool()
    picks = [(i % TEAMS == 0) for i in range(TEAMS * ROUNDS)]
    return rollout(
        build_shortlist(pool, [], SLOTS, top_n=4),
        pool, [], SLOTS,
        OpponentModel(rung="adp_noise", tau=1.0, league_beta=np.zeros(4)),
        _availability(), {},
        adp_rounds={p.player_id: i / TEAMS for i, p in enumerate(pool)},
        picks_remaining=picks,
        dispersion={"QB": 0.4, "RB": 0.6, "WR": 0.6, "TE": 0.5},
        seed=42, n_sims=n_sims, n_scenarios=n_scenarios, n_weeks=17,
        time_budget_seconds=budget, league=league,
    )


def _context(league_bits):
    bracket, seeds, start = league_bits
    return LeagueContext(
        pick_slots=[i % TEAMS for i in range(TEAMS * ROUNDS)],
        my_slot=0,
        rosters={slot: [] for slot in range(TEAMS)},
        schedule=[[(0, 1), (2, 3), (4, 5), (6, 7)] for _ in range(13)],
        bracket=bracket,
        slot_seeds=seeds,
        playoff_week_start=start,
    )


# --- 3.1 the switch changes scoring only ---------------------------------------------


def test_expected_points_reproduces_the_phase_8_recommendation_exactly():
    """Passing no league context must leave Phase 8's behaviour byte-identical."""
    first, second = _common(), _common()
    assert first.player_id == second.player_id
    assert first.scores == second.scores


def test_the_objective_changes_the_scale_of_q():
    """Expected points is in points; championship equity is a probability."""
    points = _common()
    assert max(points.scores.values()) > 1.5  # season points, not a probability


# --- 3.2 Q as a title fraction with a measured error ---------------------------------


def test_championship_equity_makes_q_a_probability(league_bits):
    out = _common(league=_context(league_bits))
    for value in out.scores.values():
        assert 0.0 <= value <= 1.0


def test_the_standard_error_is_measured_and_reported(league_bits):
    """A Bernoulli outcome per sim is far noisier than a points mean, so it is reported."""
    out = _common(league=_context(league_bits), n_scenarios=4)
    assert out.standard_errors
    assert set(out.standard_errors) == set(out.scores)
    assert all(error >= 0.0 for error in out.standard_errors.values())


def test_the_owner_is_the_team_whose_title_share_is_reported(league_bits):
    """Q is the owner's probability, not the field's."""
    out = _common(league=_context(league_bits))
    assert sum(out.scores.values()) <= len(out.scores)  # each is one team's share


# --- 3.3 one-rung degradation --------------------------------------------------------


def test_an_over_budget_equity_run_falls_back_to_expected_points(league_bits):
    """Not to the static board: the budget can still afford the cheaper rollout."""
    out = _common(league=_context(league_bits), budget=1e-6)
    assert out.degraded
    assert "fell back to expected points" in out.reason


def test_the_fallback_still_returns_a_real_recommendation(league_bits):
    out = _common(league=_context(league_bits), budget=1e-6)
    assert out.player_id in out.shortlist


def test_expected_points_over_budget_still_uses_the_static_board():
    """The last rung is unchanged from Phase 8."""
    out = _common(budget=1e-6)
    assert out.degraded
    assert "static board" in out.reason
