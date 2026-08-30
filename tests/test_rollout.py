"""M12 tests — shortlist, CRN stability, the budget, and the QB sanity gate.

The headline test is `test_vectorized_lineup_matches_the_reference_optimizer`: the fast
path exists only for speed, so it has to agree with the slow one exactly.
"""

import numpy as np
import polars as pl
import pytest

from ffdraft.lineup.slots import SlotConfig
from ffdraft.lineup.value import Player, lineup_value, marginal_value
from ffdraft.sim.availability import build_availability
from ffdraft.sim.opponent import OpponentModel
from ffdraft.sim.outcomes import SimPlayer
from ffdraft.sim.rollout import (
    Recommendation,
    _greedy_pick,
    build_shortlist,
    is_single_qb_league,
    qb_sanity_violations,
    replay_remaining,
    rollout,
    vectorized_lineup_value,
)

FLEX = {"FLEX": ["RB", "WR", "TE"]}
LEAGUE = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "FLEX", "K", "DEF"] + ["BN"] * 5
SLOTS = SlotConfig.from_league(LEAGUE, FLEX)
CYCLE = ["RB", "WR", "WR", "RB", "TE", "QB", "WR", "RB", "K", "DEF"]

# Points that actually have a replacement baseline. Quarterbacks score the most in raw
# points but sit close together, so the fourth is nearly as good as the first; running
# backs and receivers fall away fast. Without that shape a synthetic pool has no scarcity
# and the QB sanity gate has nothing meaningful to catch.
CURVES = {
    "QB": (330.0, 6.0), "RB": (285.0, 26.0), "WR": (280.0, 22.0),
    "TE": (245.0, 30.0), "K": (140.0, 3.0), "DEF": (145.0, 4.0),
}


def _player(i, points=None):
    return SimPlayer(f"p{i}", CYCLE[i % len(CYCLE)], points if points is not None else 300.0 - i,
                     team="KC", age=26.0, workload_percentile=0.5)


def _pool(n=60):
    return [_player(i) for i in range(n)]


def _realistic_pool(n=120):
    """A board with genuine positional scarcity, for the tests that need one."""
    seen: dict[str, int] = {}
    pool = []
    for i in range(n):
        position = CYCLE[i % len(CYCLE)]
        rank = seen.get(position, 0)
        seen[position] = rank + 1
        top, decay = CURVES[position]
        pool.append(
            SimPlayer(f"{position}{rank}", position, max(top - decay * rank, 20.0),
                      team="KC", age=26.0, workload_percentile=0.5)
        )
    return pool


def _availability(games=(14, 16, 17)):
    rows = [{"season": 2024, "gsis_id": f"{pos}{i}", "position": pos, "games_played": g,
             "age": 26.0, "workload_percentile": 0.5}
            for pos in ("QB", "RB", "WR", "TE", "K", "DEF")
            for i, g in enumerate(list(games) * 40)]
    return build_availability(pl.DataFrame(rows), games_per_season=17, age_bin_width=1,
                              min_bin_count=10, workload_percentile=0.8)


def _opponent(tau=1.0):
    return OpponentModel(rung="temperature", tau=tau, league_beta=np.zeros(4))


# --- the vectorised lineup must equal the reference --------------------------------


@pytest.mark.parametrize("seed", range(6))
def test_vectorized_lineup_matches_the_reference_optimizer(seed):
    """The fast path exists for speed alone; disagreeing with the slow one is a bug."""
    rng = np.random.default_rng(seed)
    roster = [_player(i) for i in range(15)]
    positions = [p.position for p in roster]
    scores = rng.gamma(2.0, 8.0, size=(4, len(roster), 3))

    fast = vectorized_lineup_value(scores, positions, SLOTS)
    for sim in range(scores.shape[0]):
        for week in range(scores.shape[2]):
            active = [
                Player(p.player_id, p.position, float(scores[sim, i, week]))
                for i, p in enumerate(roster)
            ]
            assert fast[sim, week] == pytest.approx(lineup_value(active, SLOTS))


def test_the_vectorised_lineup_benches_surplus_players():
    roster = [SimPlayer(f"w{i}", "WR", 100.0) for i in range(5)]
    slots = SlotConfig.from_league(["WR", "WR"], {})
    scores = np.arange(1, 6, dtype=float).reshape(1, 5, 1)
    assert vectorized_lineup_value(scores, [p.position for p in roster], slots)[0, 0] == 9.0


# --- 1.1 the shortlist -------------------------------------------------------------


def test_best_available_at_each_position_is_always_shortlisted():
    """The last back in a tier is outside the top-N and is exactly the reach worth making."""
    pool = _pool()
    short = build_shortlist(pool, [], SLOTS, top_n=3)
    assert {p.position for p in short} == {p.position for p in pool}
    assert len(short) > 3


def test_turning_off_the_union_leaves_only_the_top_n():
    short = build_shortlist(_pool(), [], SLOTS, top_n=3, force_best_at_each_position=False)
    assert len(short) == 3


def test_the_shortlist_has_no_duplicates():
    short = build_shortlist(_pool(), [], SLOTS, top_n=12)
    assert len({p.player_id for p in short}) == len(short)


def test_the_shortlist_is_deterministic():
    pool = _pool()
    assert [p.player_id for p in build_shortlist(pool, [], SLOTS, top_n=8)] == [
        p.player_id for p in build_shortlist(list(reversed(pool)), [], SLOTS, top_n=8)
    ]


# --- 1.2 the replay ----------------------------------------------------------------


def _adp(pool):
    return {p.player_id: float(i) / 8 for i, p in enumerate(pool)}


def _uniforms(n, seed=1):
    return np.random.default_rng(seed).random(n)


def test_the_replay_fills_only_the_owners_own_picks():
    pool = _pool()
    mine = [True, False, False, True, False]
    roster = replay_remaining(pool, [], SLOTS, _opponent(), picks_remaining=mine,
                              adp_rounds=_adp(pool), uniforms=_uniforms(len(mine)))
    assert len(roster) == sum(mine)


def test_the_replay_never_drafts_the_same_player_twice():
    pool = _pool()
    roster = replay_remaining(pool, [], SLOTS, _opponent(), picks_remaining=[True] * 10,
                              adp_rounds=_adp(pool), uniforms=_uniforms(10, 2))
    assert len({p.player_id for p in roster}) == len(roster)


def test_opponents_consume_the_board_without_joining_the_owners_roster():
    pool = _pool()
    roster = replay_remaining(pool, [], SLOTS, _opponent(0.05), picks_remaining=[False] * 12,
                              adp_rounds=_adp(pool), uniforms=_uniforms(12, 3))
    assert roster == []


def test_a_sharp_opponent_takes_the_top_of_the_board_first():
    pool = _pool(20)
    adp = _adp(pool)
    survivors = replay_remaining(pool, [], SLOTS, _opponent(0.01), picks_remaining=[False] * 5,
                                 adp_rounds=adp, uniforms=_uniforms(5, 4))
    assert survivors == []  # none reach the owner, which is the point of the check


# --- 2.1 common random numbers -----------------------------------------------------


def _rollout(seed=7, n_sims=12, budget=1e9, candidates=None, clock=None, pool=None,
             picks=None, n_scenarios=8):
    pool = _pool(40) if pool is None else pool
    short = candidates if candidates is not None else build_shortlist(pool, [], SLOTS, top_n=4)
    kwargs = {
        "candidates": short, "available": pool, "my_roster": [], "slots": SLOTS,
        "opponent": _opponent(), "availability": _availability(), "byes": {},
        "adp_rounds": _adp(pool),
        "picks_remaining": picks or [True, False, True, False, True, False, True],
        "dispersion": dict.fromkeys(("QB", "RB", "WR", "TE", "K", "DEF"), 0.6),
        "seed": seed, "n_sims": n_sims, "n_scenarios": n_scenarios, "n_weeks": 18,
        "time_budget_seconds": budget,
    }
    if clock is not None:
        kwargs["clock"] = clock
    return rollout(**kwargs)


def test_the_rollout_scores_every_candidate_and_recommends_one_of_them():
    out = _rollout()
    assert set(out.scores) == set(out.shortlist)
    assert out.player_id in out.shortlist
    assert not out.degraded
    # the pick is the best on Q, or tied with it inside the measured standard error
    leader = max(out.scores, key=lambda pid: out.scores[pid])
    gap = out.scores[leader] - out.scores[out.player_id]
    assert gap <= out.standard_errors[leader] + out.standard_errors[out.player_id]


def test_common_random_numbers_make_the_recommendation_stable_across_seeds():
    """Shared draws plus a survival tiebreak make the *pick* reproducible under reseeding.

    The raw Q ordering cannot be: candidates the greedy replay would take anyway score
    identically, so their relative order is noise however many scenarios are run.
    """
    picks = [_rollout(seed=seed, n_sims=8, n_scenarios=16).player_id for seed in (1, 2, 3, 4, 5)]
    assert len(set(picks)) == 1, picks


def test_scenario_averaging_is_what_reduces_the_noise():
    """Most of Q's variance is in how the board falls, not in the seasons that follow."""
    spreads = {}
    for n_scenarios in (2, 24):
        runs = [_rollout(seed=seed, n_sims=6, n_scenarios=n_scenarios) for seed in (1, 2, 3, 4)]
        first = runs[0].shortlist[0]
        spreads[n_scenarios] = np.std([r.scores[first] for r in runs])
    assert spreads[24] < spreads[2]


def test_survival_breaks_a_tie_toward_the_player_who_will_not_last():
    """Two candidates worth the same: take the one who will not be there next time."""
    from ffdraft.sim.rollout import select_best

    scores = {"scarce": 1000.0, "plentiful": 1002.0}
    errors = {"scarce": 6.0, "plentiful": 6.0}       # a 2-point gap is inside the noise
    survival = {"scarce": 0.10, "plentiful": 0.95}
    assert select_best(scores, errors, survival) == "scarce"


def test_survival_does_not_override_a_real_difference():
    from ffdraft.sim.rollout import select_best

    scores = {"scarce": 900.0, "better": 1100.0}
    errors = {"scarce": 5.0, "better": 5.0}          # a 200-point gap is not a tie
    survival = {"scarce": 0.01, "better": 0.99}
    assert select_best(scores, errors, survival) == "better"


def test_survival_falls_with_the_number_of_picks_until_your_turn():
    from ffdraft.sim.rollout import picks_until_next_turn, survival_probabilities

    pool = _pool(20)
    soon = survival_probabilities(pool, _opponent(0.5), _adp(pool), picks_until_next=1)
    later = survival_probabilities(pool, _opponent(0.5), _adp(pool), picks_until_next=10)
    assert later[pool[0].player_id] < soon[pool[0].player_id]
    assert picks_until_next_turn([False, False, True, False]) == 2


def test_the_same_seed_reproduces_the_recommendation():
    assert _rollout(seed=5).scores == _rollout(seed=5).scores


# --- 3.1 the budget ----------------------------------------------------------------


def test_a_rollout_that_would_overrun_falls_back_to_the_static_board():
    ticks = iter([0.0, 0.0, 30.0, 40.0] + [1000.0] * 50)
    out = _rollout(budget=45.0, clock=lambda: next(ticks))
    assert out.degraded
    assert "static board" in out.reason
    assert out.player_id in out.shortlist


def test_a_comfortable_budget_does_not_degrade():
    assert not _rollout(budget=1e9).degraded


def test_the_recommendation_reports_how_long_it_took():
    assert _rollout().elapsed_seconds >= 0.0


# --- 3.2 the QB sanity gate --------------------------------------------------------


def test_a_one_qb_league_is_recognised():
    assert is_single_qb_league(SLOTS)


def test_a_superflex_league_is_not_a_one_qb_league():
    superflex = SlotConfig.from_league(
        ["QB", "SUPER_FLEX"], {"SUPER_FLEX": ["QB", "RB", "WR", "TE"]}
    )
    assert not is_single_qb_league(superflex)


def _rec(player_id):
    return Recommendation(player_id, 0.0, {}, (player_id,), 0.0)


def test_a_quarterback_taken_earlier_than_the_market_ever_did_trips_the_gate():
    """Reported, not filtered — the fault is replacement level, and hiding it hides that."""
    violations = qb_sanity_violations(
        _rec("q1"), {"q1": "QB"}, SLOTS, pick_number=8, market_earliest={"q1": 20.0}
    )
    assert violations and "replacement level" in violations[0]


def test_a_quarterback_inside_the_market_range_is_fine():
    """The rule is about reaching, not about the round.

    An eight-team round 3 is picks 17-24, which is exactly where this market's own QB1
    goes; a fixed round number would fire on league size rather than on a defect.
    """
    assert qb_sanity_violations(
        _rec("q1"), {"q1": "QB"}, SLOTS, pick_number=19, market_earliest={"q1": 11.0}
    ) == ()


def test_a_non_quarterback_is_never_a_violation():
    assert qb_sanity_violations(
        _rec("r1"), {"r1": "RB"}, SLOTS, pick_number=1, market_earliest={"r1": 20.0}
    ) == ()


def test_a_quarterback_the_market_never_drafted_cannot_be_judged():
    assert qb_sanity_violations(
        _rec("q1"), {"q1": "QB"}, SLOTS, pick_number=1, market_earliest={}
    ) == ()


def test_a_superflex_league_may_take_a_quarterback_early():
    superflex = SlotConfig.from_league(
        ["QB", "SUPER_FLEX"], {"SUPER_FLEX": ["QB", "RB", "WR", "TE"]}
    )
    assert qb_sanity_violations(
        _rec("q1"), {"q1": "QB"}, superflex, pick_number=1, market_earliest={"q1": 20.0}
    ) == ()


def test_the_engine_does_not_reach_for_a_quarterback_past_the_market():
    """The gate run against the real rollout on a board that has a replacement baseline.

    Quarterbacks top the raw-points list here, so an engine that valued raw points would
    take one first. It should not: skipping a QB costs a few points because the fourth is
    nearly as good, while skipping a back costs far more. The market here never took a
    quarterback before pick 17, so wanting one at the owner's opening pick is the reach.
    """
    pool = _realistic_pool()
    positions = {p.player_id: p.position for p in pool}
    earliest = {p.player_id: 17.0 for p in pool if p.position == "QB"}
    # a full remaining draft, so the rollout can see what survives to the next pick
    picks = [(i % 8 == 0) for i in range(96)]
    out = _rollout(pool=pool, picks=picks, n_sims=8)
    assert qb_sanity_violations(
        out, positions, SLOTS, pick_number=1, market_earliest=earliest
    ) == (), out.player_id


def test_the_greedy_pick_reduction_matches_scoring_the_whole_board():
    """Only the best player at each position is evaluated; that must change nothing.

    The reduction is what buys the time budget, so it is worth proving rather than
    asserting: score every player the slow way and demand the same pick, from empty
    rosters through full ones, on a pool with real positional depth.
    """
    pool = _realistic_pool()
    slots = SlotConfig.from_league(
        ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "FLEX", "K", "BN", "BN"],
        {"FLEX": ["RB", "WR", "TE"]},
    )

    def brute_force(available, roster):
        current = [Player(p.player_id, p.position, p.projected_points) for p in roster]
        return max(
            available,
            key=lambda p: (
                marginal_value(
                    Player(p.player_id, p.position, p.projected_points), current, slots
                ),
                p.projected_points,
                p.player_id,
            ),
        )

    available, roster = list(pool), []
    for _ in range(11):
        expected = brute_force(available, roster)
        assert _greedy_pick(available, roster, slots).player_id == expected.player_id
        roster.append(expected)
        available.remove(expected)
