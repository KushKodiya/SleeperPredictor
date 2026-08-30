"""M18 tests — standings, bracket resolution, and championship equity.

The bracket tests run against the owner's real, completed 2025 season, frozen as a
fixture with league identifiers scrubbed. That is the only way to know the resolver
reproduces a bracket rather than merely producing a plausible one.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from ffdraft.data.sleeper import BracketMatch
from ffdraft.sim.season import (
    LeagueSeason,
    SeedingUnlearnable,
    champion,
    championship_probabilities,
    entry_slots,
    learn_slot_seeds,
    play_schedule,
    resolve_bracket,
    seed_order,
    standings_from_matchups,
)

FIXTURE = Path(__file__).parent / "fixtures" / "winners_bracket_2025.json"


@pytest.fixture(scope="module")
def real():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def bracket(real):
    return [BracketMatch.model_validate(m) for m in real["winners_bracket"]]


@pytest.fixture(scope="module")
def ranking(real):
    points = {int(w): [tuple(r) for r in rows]
              for w, rows in real["regular_season"]["points"].items()}
    pairings = {int(w): [tuple(p) for p in rows]
                for w, rows in real["regular_season"]["pairings"].items()}
    return standings_from_matchups(points, pairings)


# --- learning the seeding convention -------------------------------------------------


def test_standings_are_recovered_from_the_real_regular_season(real, ranking):
    assert ranking == real["standings"]
    assert len(ranking) == real["settings"]["num_teams"]


def test_the_seeding_convention_is_learned_not_assumed(bracket, ranking):
    """No ordering rule recovers this: t1 is the higher seed in one match, lower in the other."""
    seeds = learn_slot_seeds(bracket, ranking)
    assert sorted(seeds) == [1, 2, 3, 4, 5, 6]
    # The two byes take the top seeds.
    assert seeds[:2] == [1, 2]
    # First-round matches pair 4v5 and 3v6 — the standard six-team layout.
    assert sorted(seeds[2:4]) == [4, 5]
    assert sorted(seeds[4:]) == [3, 6]


def test_an_unplayed_bracket_cannot_teach_seeding():
    unplayed = [BracketMatch.model_validate({"r": 1, "m": 1, "t1": None, "t2": None})]
    with pytest.raises(SeedingUnlearnable, match="unplayed bracket"):
        learn_slot_seeds(unplayed, [1, 2])


def test_a_bracket_from_another_season_is_rejected(bracket):
    with pytest.raises(SeedingUnlearnable, match="different seasons"):
        learn_slot_seeds(bracket, [99, 98, 97])


# --- 2.3 reproducing the real bracket ------------------------------------------------


def _score_from(real):
    start = real["settings"]["playoff_week_start"]
    points = real["points_by_week"]

    def score(team: int, rnd: int) -> float:
        return float(points[str(start + rnd - 1)][str(team)])

    return score


def test_the_real_bracket_reproduces_its_winners_and_placements_exactly(real, bracket):
    """Given the season's realized weekly points, every match must fall the way it did."""
    assignment = {}
    for match_id, side in entry_slots(bracket):
        match = next(m for m in bracket if int(m.m) == match_id)
        assignment[(match_id, side)] = int(match.t1 if side == "t1" else match.t2)

    placements = resolve_bracket(bracket, assignment, _score_from(real))
    published = {m.p: int(m.w) for m in bracket if m.p is not None}
    for placement, roster in published.items():
        assert placements[placement] == roster, f"placement {placement} differs"
    assert placements[1] == published[1]


def test_the_champion_is_reached_through_the_learned_seeding(real, bracket, ranking):
    """Same answer by the route simulation uses: standings -> seeds -> slots."""
    seeds = learn_slot_seeds(bracket, ranking)
    winner = champion(bracket, ranking, seeds, _score_from(real))
    assert winner == int(next(m.w for m in bracket if m.p == 1))


# --- 2.4 a different playoff size ----------------------------------------------------


def _four_team_bracket():
    """A published four-team bracket: two semis, a final and a third-place game."""
    return [
        BracketMatch.model_validate({"r": 1, "m": 1, "t1": 1, "t2": 4}),
        BracketMatch.model_validate({"r": 1, "m": 2, "t1": 2, "t2": 3}),
        BracketMatch.model_validate(
            {"r": 2, "m": 3, "p": 1, "t1_from": {"w": 1}, "t2_from": {"w": 2}}
        ),
        BracketMatch.model_validate(
            {"r": 2, "m": 4, "p": 3, "t1_from": {"l": 1}, "t2_from": {"l": 2}}
        ),
    ]


def test_a_four_team_bracket_runs_from_its_own_structure():
    """No code change: the resolver reads rounds and references, not a hardcoded shape."""
    bracket = _four_team_bracket()
    assert len(entry_slots(bracket)) == 4

    points = {1: 100.0, 2: 90.0, 3: 80.0, 4: 70.0}
    placements = resolve_bracket(
        bracket,
        {(1, "t1"): 1, (1, "t2"): 4, (2, "t1"): 2, (2, "t2"): 3},
        lambda team, _rnd: points[team],
    )
    assert placements[1] == 1  # best team wins
    assert placements[2] == 2
    assert placements[3] == 3  # third-place game resolved from the losers


def test_slot_references_resolve_from_their_own_keys():
    """A row whose t1/t2 is null is filled from t1_from/t2_from, not mistaken for empty."""
    bracket = _four_team_bracket()
    final = next(m for m in bracket if m.p == 1)
    assert final.t1 is None and final.t1_from == {"w": 1}


# --- 2.2 / 2.5 simulation ------------------------------------------------------------


def _league(n_sims=200, n_weeks=16, teams=6, *, seed=0, dominant=None):
    rng = np.random.default_rng(seed)
    weekly = {}
    for team in range(1, teams + 1):
        mean = 400.0 if team == dominant else 100.0
        weekly[team] = rng.normal(mean, 10.0, size=(n_sims, n_weeks))
    schedule = [
        [(1, 2), (3, 4), (5, 6)][: teams // 2] for _ in range(13)
    ]
    return play_schedule(weekly, schedule)


def test_the_same_seed_reproduces_identical_standings():
    """R7: no global randomness anywhere in the season sim."""
    first, second = _league(seed=7), _league(seed=7)
    assert seed_order(first, 0) == seed_order(second, 0)
    assert np.array_equal(first.wins[1], second.wins[1])


def test_a_different_seed_gives_a_different_season():
    assert not np.array_equal(_league(seed=1).wins[1], _league(seed=2).wins[1])


def test_standings_rank_on_wins_then_points():
    weekly = {
        1: np.array([[10.0, 10.0]]),
        2: np.array([[30.0, 1.0]]),
    }
    season = play_schedule(weekly, [[(1, 2)], [(1, 2)]])
    # team 2 wins week 1 by a lot, loses week 2 -> 1-1 each, team 2 has more points
    assert season.wins[1][0] == 1.0 and season.wins[2][0] == 1.0
    assert seed_order(season, 0) == [2, 1]


def test_championship_probabilities_sum_to_one(bracket, ranking):
    season = _league(n_sims=200, teams=8, seed=3)
    seeds = learn_slot_seeds(bracket, ranking)
    probabilities = _probabilities(season, bracket, seeds)
    assert sum(probabilities.values()) == pytest.approx(1.0, abs=0.001)


def test_a_dominant_roster_wins_more_often_than_any_other(bracket, ranking):
    season = _league(n_sims=300, teams=8, seed=5, dominant=1)
    seeds = learn_slot_seeds(bracket, ranking)
    probabilities = _probabilities(season, bracket, seeds)
    best = max(probabilities, key=lambda team: probabilities[team])
    assert best == 1
    assert probabilities[1] > 0.5


def _probabilities(season, bracket, seeds):
    """championship_probabilities with the learned seeding threaded through."""
    return championship_probabilities(
        season, bracket, playoff_week_start=14, slot_seeds=seeds
    )


def test_probabilities_are_reproducible_under_a_seed(bracket, ranking):
    seeds = learn_slot_seeds(bracket, ranking)
    first = _probabilities(_league(n_sims=100, teams=8, seed=11), bracket, seeds)
    second = _probabilities(_league(n_sims=100, teams=8, seed=11), bracket, seeds)
    assert first == second


def test_every_team_has_a_probability_even_if_it_never_wins(bracket, ranking):
    """A team that misses the playoffs must appear with 0.0, not be absent (R4)."""
    season = _league(n_sims=100, teams=8, seed=5, dominant=1)
    probabilities = _probabilities(season, bracket, learn_slot_seeds(bracket, ranking))
    assert len(probabilities) == 8
    assert all(0.0 <= p <= 1.0 for p in probabilities.values())


def test_a_league_season_carries_points_for_alongside_wins():
    season: LeagueSeason = _league(n_sims=10, teams=6, seed=1)
    assert set(season.points_for) == set(season.wins)
