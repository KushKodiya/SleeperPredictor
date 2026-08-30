"""M12 — recommending a pick by simulating the rest of the draft.

For each candidate the engine replays the remaining draft — opponents from the Phase 7
model, itself by greedy marginal value — samples the season that follows, and takes the
candidate with the highest mean final team value.

Three things make this work inside a 45-second live budget:

- **Common random numbers.** Every candidate is evaluated against the same sampled
  seasons and the same opponent draws, so the comparison measures the candidate rather
  than the noise between two independent runs. The PRD calls this the single
  highest-leverage optimisation and it cuts required sims by roughly an order of
  magnitude; it is also what makes the ordering stable, which is the test.
- **A vectorised lineup.** The reference optimiser in `lineup/value.py` solves one week
  for one roster in Python. The hot path here needs millions of those, so
  `vectorized_lineup_value` computes every sim and week at once with numpy sorts. It is
  tested to agree with the reference exactly.
- **`force_best_at_each_position`.** Without it the shortlist is the top-N by marginal
  value, and the engine can never find the reach-for-the-last-back-in-the-tier move,
  because that player is not in the top ten.

A recommendation that would blow the budget degrades to the static board. During a live
draft an overrun is worse than a less clever pick.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from ffdraft.lineup.slots import BENCH, SlotConfig
from ffdraft.lineup.value import Player, marginal_value, vectorized_lineup_value
from ffdraft.sim.availability import AvailabilityModel
from ffdraft.sim.opponent import OpponentModel
from ffdraft.sim.outcomes import SimPlayer, simulate_players
from ffdraft.sim.season import (
    championship_probabilities,
    play_schedule,
    simulate_rosters,
)

QB = "QB"


@dataclass(frozen=True)
class LeagueContext:
    """What championship equity needs beyond what expected points already has.

    Every field is a fact about the league rather than a tuning knob: who picks when, the
    other teams' rosters so far, the weekly schedule, and the published bracket with the
    seeding learned from a completed season (`sim.season.learn_slot_seeds` — the bracket
    does not encode seeds).
    """

    pick_slots: Sequence[int]
    my_slot: int
    rosters: dict[int, list[SimPlayer]]
    schedule: Sequence[Sequence[tuple[int, int]]]
    bracket: Sequence[Any]
    slot_seeds: Sequence[int]
    playoff_week_start: int


@dataclass(frozen=True)
class Recommendation:
    """The pick, why it won, and whether the engine had time to think."""

    player_id: str
    q_value: float
    scores: dict[str, float]
    shortlist: tuple[str, ...]
    elapsed_seconds: float
    degraded: bool = False
    reason: str | None = None
    sanity_violations: tuple[str, ...] = ()
    survival: dict[str, float] = field(default_factory=dict)
    standard_errors: dict[str, float] = field(default_factory=dict)


def survival_probabilities(
    available: Sequence[SimPlayer],
    opponent: OpponentModel,
    adp_rounds: dict[str, float],
    *,
    picks_until_next: int,
) -> dict[str, float]:
    """P(each player is still there at the owner's next pick).

    Q alone cannot separate two candidates the greedy replay would take anyway: pick the
    quarterback now and the back next, or the other way round, and the final roster is
    identical, so both score exactly the same. What actually distinguishes them is which
    one survives. Among equals, take the one who will not.

    One softmax over the current board, compounded over the intervening picks — cheap
    enough to recompute at every pick, and a property of the board rather than of any
    candidate, so it is shared across the shortlist.
    """
    if not available:
        return {}
    rounds = np.array([adp_rounds.get(p.player_id, 99.0) for p in available])
    taken = opponent.probabilities(
        rounds, np.zeros((len(available), len(opponent.feature_names))), user_id="rollout"
    )
    survive = np.power(np.clip(1.0 - taken, 0.0, 1.0), max(picks_until_next, 0))
    return {p.player_id: float(s) for p, s in zip(available, survive, strict=True)}


def picks_until_next_turn(picks_remaining: Sequence[bool]) -> int:
    """How many opponent picks come before the owner is on the clock again."""
    for index, mine in enumerate(picks_remaining):
        if mine:
            return index
    return len(picks_remaining)


# --- the shortlist -----------------------------------------------------------------


def build_shortlist(
    available: Sequence[SimPlayer],
    roster: Sequence[SimPlayer],
    slots: SlotConfig,
    *,
    top_n: int,
    force_best_at_each_position: bool = True,
) -> list[SimPlayer]:
    """Top-N by marginal value, plus the best available at each position.

    The union is not optional. A tier's last surviving running back is often outside the
    top ten by marginal value and is exactly the pick worth reaching for; a shortlist
    that cannot see him cannot recommend him.
    """
    current = [Player(p.player_id, p.position, p.projected_points) for p in roster]
    ranked = sorted(
        available,
        key=lambda p: (
            -marginal_value(Player(p.player_id, p.position, p.projected_points), current, slots),
            -p.projected_points,
            p.player_id,
        ),
    )
    shortlist = list(ranked[:top_n])
    if force_best_at_each_position:
        chosen = {p.player_id for p in shortlist}
        best_by_position: dict[str, SimPlayer] = {}
        for player in sorted(available, key=lambda p: (-p.projected_points, p.player_id)):
            best_by_position.setdefault(player.position, player)
        for player in best_by_position.values():
            if player.player_id not in chosen:
                shortlist.append(player)
                chosen.add(player.player_id)
    return shortlist


# --- the draft replay --------------------------------------------------------------


def _greedy_pick(
    available: list[SimPlayer], roster: list[SimPlayer], slots: SlotConfig
) -> SimPlayer:
    """The owner's own pick inside a replay: the best marginal value on the board.

    Only the top player at each position can win this argmax, so only they are evaluated.
    Marginal value is monotone in projected points at a fixed position — a lineup built
    with a strictly better player at the same position dominates the old one slot for
    slot — which makes the reduction exact rather than a shortcut.

    It is also the whole performance story. Profiled on a 2024 backtest board, scoring
    every one of 848 players at each of fifteen picks put `optimal_lineup` at 94% of a
    recommendation's runtime; six evaluations per pick is what makes the budget.
    """
    current = [Player(p.player_id, p.position, p.projected_points) for p in roster]
    best_at_position: dict[str, SimPlayer] = {}
    for player in available:
        incumbent = best_at_position.get(player.position)
        if incumbent is None or (player.projected_points, player.player_id) > (
            incumbent.projected_points,
            incumbent.player_id,
        ):
            best_at_position[player.position] = player
    return max(
        best_at_position.values(),
        key=lambda p: (
            marginal_value(Player(p.player_id, p.position, p.projected_points), current, slots),
            p.projected_points,
            p.player_id,
        ),
    )


def replay_league(
    available: list[SimPlayer],
    rosters: dict[int, list[SimPlayer]],
    slots: SlotConfig,
    opponent: OpponentModel,
    *,
    pick_slots: Sequence[int],
    my_slot: int,
    adp_rounds: dict[str, float],
    uniforms: np.ndarray,
) -> dict[int, list[SimPlayer]]:
    """Play the draft out and return **every** team's roster, not only the owner's.

    `pick_slots[i]` names the team on the clock for pick `i`. The owner takes the greedy
    marginal-value pick at his own turns; every other team is sampled from the opponent
    model's distribution over who is left.

    `uniforms` is one pre-drawn number per pick, shared across every candidate. Drawing
    from a generator instead would break common random numbers: two candidate rosters
    differ by one player, the streams desynchronise at the first opponent pick, and every
    pick after that diverges for reasons that have nothing to do with the candidate.

    The other teams' rosters are the input championship equity needs, and they cost
    nothing extra — the replay was already computing those picks and discarding them.
    """
    pool = list(available)
    drafted = {slot: list(roster) for slot, roster in rosters.items()}

    for index, slot in enumerate(pick_slots):
        if not pool:
            break
        current = drafted.setdefault(slot, [])
        if slot == my_slot:
            chosen = _greedy_pick(pool, current, slots)
        else:
            rounds = np.array([adp_rounds.get(p.player_id, 99.0) for p in pool])
            probabilities = opponent.probabilities(
                rounds, np.zeros((len(pool), len(opponent.feature_names))), user_id="rollout"
            )
            draw = float(uniforms[index % len(uniforms)])
            chosen = pool[int(np.searchsorted(np.cumsum(probabilities), draw, side="right"))
                          if draw < 1.0 else len(pool) - 1]
        current.append(chosen)
        pool.remove(chosen)
    return drafted


# Slot standing in for "some opponent" when the caller only tracks the owner's roster.
_ANY_OPPONENT = -1
_OWNER = 0


def replay_remaining(
    available: list[SimPlayer],
    my_roster: list[SimPlayer],
    slots: SlotConfig,
    opponent: OpponentModel,
    *,
    picks_remaining: Sequence[bool],
    adp_rounds: dict[str, float],
    uniforms: np.ndarray,
) -> list[SimPlayer]:
    """Play the draft out and return the owner's roster. `picks_remaining[i]` is his turn.

    A thin view over `replay_league`: when only the owner's roster is wanted, the other
    teams collapse into one bucket, because the opponent model treats every opponent
    identically anyway. One loop serves both so the two cannot drift apart.
    """
    drafted = replay_league(
        available,
        {_OWNER: list(my_roster)},
        slots,
        opponent,
        pick_slots=[_OWNER if mine else _ANY_OPPONENT for mine in picks_remaining],
        my_slot=_OWNER,
        adp_rounds=adp_rounds,
        uniforms=uniforms,
    )
    return drafted[_OWNER]


# --- the rollout -------------------------------------------------------------------


def _title_probability(
    available: list[SimPlayer],
    my_roster: list[SimPlayer],
    slots: SlotConfig,
    opponent: OpponentModel,
    availability: AvailabilityModel,
    byes: dict[str, int],
    league: LeagueContext,
    *,
    adp_rounds: dict[str, float],
    dispersion: dict[str, float],
    uniforms: np.ndarray,
    rng: np.random.Generator,
    crn_seed: int,
    n_sims: int,
    n_weeks: int,
) -> float:
    """One scenario's championship equity: the owner's share of simulated titles.

    The other teams' rosters cost nothing extra — the replay was already making those
    picks and throwing them away.
    """
    rosters = replay_league(
        available,
        {**league.rosters, league.my_slot: list(my_roster)},
        slots,
        opponent,
        pick_slots=league.pick_slots,
        my_slot=league.my_slot,
        adp_rounds=adp_rounds,
        uniforms=uniforms,
    )
    weekly = simulate_rosters(
        rosters, availability, byes, slots,
        rng=rng, n_sims=n_sims, n_weeks=n_weeks, dispersion=dispersion, crn_seed=crn_seed,
    )
    season = play_schedule(weekly, league.schedule)
    probabilities = championship_probabilities(
        season, league.bracket,
        playoff_week_start=league.playoff_week_start, slot_seeds=league.slot_seeds,
    )
    return float(probabilities.get(league.my_slot, 0.0))


def rollout(
    candidates: Sequence[SimPlayer],
    available: Sequence[SimPlayer],
    my_roster: Sequence[SimPlayer],
    slots: SlotConfig,
    opponent: OpponentModel,
    availability: AvailabilityModel,
    byes: dict[str, int],
    *,
    adp_rounds: dict[str, float],
    picks_remaining: Sequence[bool],
    dispersion: dict[str, float],
    seed: int,
    n_sims: int,
    n_scenarios: int,
    n_weeks: int,
    time_budget_seconds: float,
    league: LeagueContext | None = None,
    static_fallback: SimPlayer | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> Recommendation:
    """Q(p) for every candidate under common random numbers; the best one wins.

    `league` selects the objective, and it is the *only* thing it changes. Pass nothing
    and Q is mean final team value, exactly as Phase 8 computed it; pass a `LeagueContext`
    and Q becomes the owner's probability of winning the title. The shortlist, the replay,
    the common random numbers and the survival tie-break are untouched either way —
    the objective changes how a finished rollout is scored, not how it is played out.

    Returns early with the static-board pick if the remaining budget cannot cover another
    candidate. Blowing the clock at a live draft is worse than a less clever pick.
    """
    started = clock()
    scores: dict[str, float] = {}
    errors: dict[str, float] = {}
    per_candidate: list[float] = []
    # One shared block of opponent draws for every candidate — the other half of CRN.
    # A row per scenario, because a single draft future is not enough to average over:
    # most of the variance in Q lives in how the board falls, not in the seasons that
    # follow, so raising the season count alone leaves the ranking unstable.
    uniforms = np.random.default_rng(seed).random(
        (max(n_scenarios, 1), max(len(picks_remaining), 1))
    )

    for candidate in candidates:
        elapsed = clock() - started
        projected = max(per_candidate, default=0.0)
        if per_candidate and elapsed + projected > time_budget_seconds:
            # Degrade one rung at a time. Championship equity multiplies the season work
            # by the number of teams, so it is the rung most likely to be abandoned —
            # dropping straight from it to the static board would throw away a rollout the
            # remaining budget can still afford, which is a worse pick than necessary at
            # exactly the moment it matters.
            if league is not None:
                cheaper = rollout(
                    candidates, available, my_roster, slots, opponent, availability, byes,
                    adp_rounds=adp_rounds, picks_remaining=picks_remaining,
                    dispersion=dispersion, seed=seed, n_sims=n_sims,
                    n_scenarios=n_scenarios, n_weeks=n_weeks,
                    time_budget_seconds=max(time_budget_seconds - elapsed, 0.0),
                    league=None, static_fallback=static_fallback, clock=clock,
                )
                return replace(
                    cheaper,
                    degraded=True,
                    elapsed_seconds=clock() - started,
                    reason=(
                        f"championship equity would exceed the {time_budget_seconds:g}s "
                        f"budget after {len(scores)} of {len(candidates)} candidates; fell "
                        f"back to expected points"
                        + (f" — then {cheaper.reason}" if cheaper.degraded else "")
                    ),
                )

            fallback = static_fallback or candidates[0]
            return Recommendation(
                player_id=fallback.player_id,
                q_value=scores.get(fallback.player_id, float("nan")),
                scores=scores,
                shortlist=tuple(c.player_id for c in candidates),
                elapsed_seconds=clock() - started,
                degraded=True,
                reason=(
                    f"stopped after {len(scores)} of {len(candidates)} candidates to stay "
                    f"inside the {time_budget_seconds:g}s budget; using the static board"
                ),
            )

        candidate_started = clock()
        # Common random numbers: every candidate sees the same draft futures and the same
        # player seasons, so a difference between two Q values is the candidate, not the draws.
        remaining = [p for p in available if p.player_id != candidate.player_id]
        totals = []
        for scenario in range(max(n_scenarios, 1)):
            draws = np.random.default_rng([seed, scenario])
            crn = seed * 1000 + scenario
            if league is None:
                roster = replay_remaining(
                    remaining, [*my_roster, candidate], slots, opponent,
                    picks_remaining=picks_remaining, adp_rounds=adp_rounds,
                    uniforms=uniforms[scenario],
                )
                weekly = simulate_players(
                    roster, availability, byes, draws,
                    n_sims=n_sims, n_weeks=n_weeks, dispersion=dispersion, crn_seed=crn,
                )
                value = vectorized_lineup_value(weekly, [p.position for p in roster], slots)
                totals.append(float(value.sum(axis=1).mean()))
            else:
                totals.append(
                    _title_probability(
                        remaining, [*my_roster, candidate], slots, opponent, availability,
                        byes, league, adp_rounds=adp_rounds, dispersion=dispersion,
                        uniforms=uniforms[scenario], rng=draws, crn_seed=crn,
                        n_sims=n_sims, n_weeks=n_weeks,
                    )
                )
        scores[candidate.player_id] = float(np.mean(totals))
        errors[candidate.player_id] = float(
            np.std(totals) / np.sqrt(len(totals)) if len(totals) > 1 else 0.0
        )
        per_candidate.append(clock() - candidate_started)

    survival = survival_probabilities(
        available, opponent, adp_rounds,
        picks_until_next=picks_until_next_turn(picks_remaining[1:]) + 1,
    )
    best = select_best(scores, errors, survival)
    return Recommendation(
        player_id=best,
        q_value=scores[best],
        scores=scores,
        shortlist=tuple(c.player_id for c in candidates),
        elapsed_seconds=clock() - started,
        survival=survival,
        standard_errors=errors,
    )


def select_best(
    scores: dict[str, float], errors: dict[str, float], survival: dict[str, float]
) -> str:
    """The best candidate, with survival breaking ties Q cannot resolve.

    Two candidates count as tied when their Q values sit inside their combined standard
    error — a tolerance measured from the scenarios actually run, not a number chosen to
    look reasonable (R5). Among those, the one least likely to last until the next pick
    wins, because the other can still be had later.
    """
    ranked = sorted(scores, key=lambda pid: (-scores[pid], pid))
    leader = ranked[0]
    tolerance = errors.get(leader, 0.0)
    tied = [
        pid for pid in ranked
        if scores[leader] - scores[pid] <= tolerance + errors.get(pid, 0.0)
    ]
    return min(tied, key=lambda pid: (survival.get(pid, 1.0), -scores[pid], pid))


# --- sanity gates ------------------------------------------------------------------


def is_single_qb_league(slots: SlotConfig) -> bool:
    """One QB slot and no flex a quarterback could fill."""
    dedicated = sum(1 for s in slots.slots if slots.eligibility[s] == frozenset({QB}))
    flexible = any(QB in slots.eligibility[s] and len(slots.eligibility[s]) > 1 for s in slots.slots)
    return dedicated <= 1 and not flexible


def qb_sanity_violations(
    recommendation: Recommendation,
    positions: dict[str, str],
    slots: SlotConfig,
    *,
    pick_number: int,
    market_earliest: dict[str, float],
) -> tuple[str, ...]:
    """Catch the failure that means replacement level is wrong.

    In a 1-QB league, wanting a quarterback earlier than the market has ever taken him
    is the signature of a broken replacement baseline: set replacement too low and the
    engine sees phantom value at the one position where only one starter is needed. The
    fault would be upstream in replacement level, not in the rollout, so this reports
    the violation loudly rather than filtering quarterbacks out and hiding the cause.

    `market_earliest` is FFC's `high` — the earliest pick at which that player was
    actually drafted across the sampled drafts. Comparing against it rather than a fixed
    round is what makes the gate mean the same thing in a league of any size. An earlier
    draft of this rule read "never a QB in round 3", which is a twelve-team heuristic:
    round 3 of this owner's eight-team league is picks 17-24, and the market's own QB1
    goes at pick 20-25 there, so the rule fired on league size rather than on a defect.

    A quarterback the market never drafted has no `high` to compare against and cannot
    be judged here; the drafted-outside-ADP case belongs to the shortlist, not to this
    gate.
    """
    if not is_single_qb_league(slots) or positions.get(recommendation.player_id) != QB:
        return ()
    earliest = market_earliest.get(recommendation.player_id)
    if earliest is None or pick_number >= earliest:
        return ()
    return (
        (
            f"recommended {recommendation.player_id} (QB) at pick {pick_number} of a 1-QB "
            f"league, earlier than the market ever took him (earliest observed "
            f"{earliest:g}); replacement level is almost certainly wrong"
        ),
    )


def bench_positions(slots: SlotConfig, roster_positions: Sequence[str]) -> int:
    """Bench slots, which the starting-slot config deliberately excludes."""
    return sum(1 for slot in roster_positions if slot == BENCH)
