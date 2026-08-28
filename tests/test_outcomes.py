"""M10 tests — season sampling, byes, convergence, and the conditional value of depth."""

import numpy as np
import polars as pl
import pytest

from ffdraft.lineup.slots import SlotConfig
from ffdraft.sim.availability import build_availability
from ffdraft.sim.outcomes import (
    SimPlayer,
    bye_weeks,
    roster_value,
    season_totals,
    simulate_players,
    simulate_roster,
    simulated_marginal_value,
    weekly_dispersion,
)

WEEKS = 18
FLEX = {"FLEX": ["RB", "WR", "TE"]}
SLOTS = SlotConfig.from_league(["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DEF", "BN"], FLEX)
NO_SPREAD = {}  # cv defaults to 1.0; tests that need a fixed score pass cv 0


def _model(games_by_position, *, n=60):
    """An availability model whose every draw for a position is a fixed games count."""
    rows = [
        {"season": 2024, "gsis_id": f"{pos}{i}", "position": pos, "games_played": g,
         "age": 26.0, "workload_percentile": 0.5}
        for pos, games in games_by_position.items()
        for i, g in enumerate(games * n)
    ]
    return build_availability(pl.DataFrame(rows), games_per_season=17, age_bin_width=1,
                              min_bin_count=10, workload_percentile=0.8)


def _fixed_scores():
    """cv 0 for every position: each active week scores exactly the player's rate."""
    return dict.fromkeys(("QB", "RB", "WR", "TE", "K", "DEF"), 0.0)


# --- 3.1 byes and weekly sampling --------------------------------------------------


def test_bye_weeks_are_read_from_the_schedule():
    games = pl.DataFrame(
        [{"season": 2026, "week": w, "game_type": "REG", "home_team": "KC", "away_team": "DEN"}
         for w in (1, 2, 4)]
        + [{"season": 2026, "week": 3, "game_type": "REG", "home_team": "SF", "away_team": "SEA"}]
    )
    byes = bye_weeks(games, season=2026)
    assert byes["KC"] == 3 and byes["DEN"] == 3


def test_a_player_on_bye_contributes_zero_that_week():
    model = _model({"RB": [17]})
    roster = [SimPlayer("rb1", "RB", 170.0, team="KC", age=26.0, workload_percentile=0.5)]
    weekly = simulate_players(roster, model, {"KC": 7}, np.random.default_rng(1),
                              n_sims=50, n_weeks=WEEKS, dispersion=_fixed_scores())
    assert weekly[:, 0, 6].sum() == 0.0          # week 7 is the bye, always zero
    assert (weekly[:, 0, :] > 0).sum(axis=1).max() == 17  # and 17 weeks are played


def test_a_player_who_misses_games_has_zero_weeks():
    model = _model({"RB": [10]})
    roster = [SimPlayer("rb1", "RB", 100.0, team="KC", age=26.0, workload_percentile=0.5)]
    weekly = simulate_players(roster, model, {}, np.random.default_rng(2),
                              n_sims=40, n_weeks=WEEKS, dispersion=_fixed_scores())
    assert (weekly[:, 0, :] > 0).sum(axis=1).tolist() == [10] * 40


def test_the_same_seed_reproduces_the_season():
    model = _model({"RB": [12, 15, 17]})
    roster = [SimPlayer("rb1", "RB", 200.0, team="KC", age=26.0, workload_percentile=0.5)]
    args = {"n_sims": 30, "n_weeks": WEEKS, "dispersion": {"RB": 0.6}}
    first = simulate_players(roster, model, {}, np.random.default_rng(5), **args)
    second = simulate_players(roster, model, {}, np.random.default_rng(5), **args)
    assert np.allclose(first, second)


# --- 3.2 convergence to the projection ---------------------------------------------


@pytest.mark.parametrize("games", [[17], [10, 12, 14, 17], [6, 17]])
def test_expected_season_total_converges_to_the_projection(games):
    """The weekly rate is projection / expected games, so the mean lands on the projection."""
    model = _model({"RB": games})
    roster = [SimPlayer("rb1", "RB", 240.0, team="KC", age=26.0, workload_percentile=0.5)]
    weekly = simulate_players(roster, model, {}, np.random.default_rng(9),
                              n_sims=6000, n_weeks=WEEKS, dispersion={"RB": 0.7})
    assert season_totals(weekly)[:, 0].mean() == pytest.approx(240.0, rel=0.03)


def test_a_bye_does_not_cost_the_player_his_projection():
    """The bye is already inside the historical games count; it must not double-charge."""
    model = _model({"RB": [16]})
    roster = [SimPlayer("rb1", "RB", 240.0, team="KC", age=26.0, workload_percentile=0.5)]
    weekly = simulate_players(roster, model, {"KC": 5}, np.random.default_rng(4),
                              n_sims=4000, n_weeks=WEEKS, dispersion={"RB": 0.7})
    assert season_totals(weekly)[:, 0].mean() == pytest.approx(240.0, rel=0.03)


def test_weekly_dispersion_is_measured_from_history():
    scored = pl.DataFrame(
        [{"position": "RB", "points": p} for p in (5.0, 10.0, 15.0, 20.0)]
        + [{"position": "K", "points": p} for p in (8.0, 8.0, 8.0, 8.0)]
    )
    cv = weekly_dispersion(scored)
    assert cv["RB"] > 0.3          # running backs swing week to week
    assert cv["K"] == pytest.approx(0.0)  # this kicker never varied


# --- 3.3 depth is worth something, conditionally -----------------------------------


def _starter_and_backup(starter_games):
    """Everyone shares a bye so absences line up; only injury drives who is available.

    With staggered byes the starters would already miss different weeks and the backup
    would cover those gaps — true, and exactly what M10 is for, but it would stop this
    pair of tests from isolating the injury effect they are about.
    """
    model = _model({"RB": starter_games, "QB": [17], "WR": [17], "TE": [17],
                    "K": [17], "DEF": [17]})
    roster = [
        SimPlayer("rb1", "RB", 255.0, team="KC", age=26.0, workload_percentile=0.5),
        SimPlayer("rb2", "RB", 200.0, team="KC", age=26.0, workload_percentile=0.5),
    ]
    backup = SimPlayer("rb3", "RB", 120.0, team="KC", age=26.0, workload_percentile=0.5)
    return model, roster, backup


def test_a_backup_is_worth_nothing_when_the_starters_never_miss_time():
    model, roster, backup = _starter_and_backup([17])
    slots = SlotConfig.from_league(["RB", "RB", "BN"], FLEX)
    value = simulated_marginal_value(backup, roster, slots, model, {"KC": 7}, seed=3,
                                     n_sims=120, n_weeks=WEEKS, dispersion=_fixed_scores())
    assert value == pytest.approx(0.0, abs=1e-6)


def test_the_same_backup_is_worth_something_once_starters_get_hurt():
    model, roster, backup = _starter_and_backup([6])
    slots = SlotConfig.from_league(["RB", "RB", "BN"], FLEX)
    value = simulated_marginal_value(backup, roster, slots, model, {"KC": 7}, seed=3,
                                     n_sims=120, n_weeks=WEEKS, dispersion=_fixed_scores())
    assert value > 0.0


def test_depth_is_worth_more_the_more_the_starters_miss():
    slots = SlotConfig.from_league(["RB", "RB", "BN"], FLEX)
    values = []
    for games in ([17], [12], [6]):
        model, roster, backup = _starter_and_backup(games)
        values.append(
            simulated_marginal_value(backup, roster, slots, model, {"KC": 7}, seed=8,
                                     n_sims=120, n_weeks=WEEKS, dispersion=_fixed_scores())
        )
    assert values == sorted(values)  # never-miss <= sometimes <= often


def test_a_benched_player_adds_nothing_to_the_lineup_this_week():
    """roster_value counts started points only — the whole reason M10 exists.

    All three share a bye and never get hurt, so they are available in identical weeks;
    only one can start, and the other two are worth nothing.
    """
    model = _model({"RB": [17]})
    roster = [SimPlayer(f"rb{i}", "RB", 170.0, team="KC", age=26.0, workload_percentile=0.5)
              for i in range(3)]
    slots = SlotConfig.from_league(["RB", "BN", "BN"], FLEX)
    weekly = simulate_players(roster, model, {"KC": 7}, np.random.default_rng(6), n_sims=5,
                              n_weeks=WEEKS, dispersion=_fixed_scores())
    started = roster_value(roster, weekly, slots)
    assert started.mean() == pytest.approx(170.0, rel=0.01)  # one RB slot, not three


def test_uncorrelated_absences_are_what_make_depth_valuable():
    """Staggered byes mean a second back really does cover weeks the first cannot."""
    model = _model({"RB": [17]})
    roster = [SimPlayer("rb1", "RB", 170.0, team="KC", age=26.0, workload_percentile=0.5),
              SimPlayer("rb2", "RB", 170.0, team="SF", age=26.0, workload_percentile=0.5)]
    slots = SlotConfig.from_league(["RB", "BN"], FLEX)
    one = roster_value(roster[:1], simulate_players(roster[:1], model, {"KC": 7},
                       np.random.default_rng(6), n_sims=5, n_weeks=WEEKS,
                       dispersion=_fixed_scores()), slots)
    both = roster_value(roster, simulate_players(roster, model, {"KC": 7, "SF": 9},
                        np.random.default_rng(6), n_sims=5, n_weeks=WEEKS,
                        dispersion=_fixed_scores()), slots)
    assert both.mean() > one.mean()


def test_simulate_roster_matches_roster_value_on_the_same_seed():
    model = _model({"RB": [14, 17]})
    roster = [SimPlayer("rb1", "RB", 200.0, team="KC", age=26.0, workload_percentile=0.5)]
    slots = SlotConfig.from_league(["RB"], FLEX)
    args = {"n_sims": 20, "n_weeks": WEEKS, "dispersion": {"RB": 0.5}}
    direct = simulate_roster(roster, slots, model, {}, np.random.default_rng(2), **args)
    weekly = simulate_players(roster, model, {}, np.random.default_rng(2), **args)
    assert np.allclose(direct, roster_value(roster, weekly, slots))
