"""M9 — what a roster is worth, and what one more player adds to it.

`lineup_value` is the best legal starting lineup a roster can field. `marginal_value` is
the difference one player makes to that, which is the number a draft board actually needs:
a fourth wide receiver in a two-WR league is worth nothing this week no matter how good
he is, and a board that values him on projected points alone will waste picks on him.

Slots are filled most-restrictive-first. That is optimal whenever slot eligibility is
nested — a kicker slot takes only kickers, a flex takes a superset — and `SlotConfig`
refuses to build a league where it would not be.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from ffdraft.lineup.slots import SlotConfig


@dataclass(frozen=True)
class Player:
    """The three things a lineup decision needs."""

    player_id: str
    position: str
    points: float


def optimal_lineup(roster: Iterable[Player], slots: SlotConfig) -> dict[int, Player]:
    """Best legal assignment of players to starting slots, keyed by slot index.

    Walks slots most-restrictive-first, giving each the highest scorer still available
    that it can legally take.
    """
    available = sorted(roster, key=lambda p: (-p.points, p.player_id))
    used: set[int] = set()
    lineup: dict[int, Player] = {}

    for index, slot in enumerate(slots.slots):
        for position, player in enumerate(available):
            if position in used or not slots.eligible(slot, player.position):
                continue
            lineup[index] = player
            used.add(position)
            break
    return lineup


def lineup_value(roster: Sequence[Player], slots: SlotConfig) -> float:
    """Points the best legal starting lineup scores. Benched players contribute nothing."""
    return sum(player.points for player in optimal_lineup(roster, slots).values())


def marginal_value(player: Player, roster: Sequence[Player], slots: SlotConfig) -> float:
    """What adding `player` is worth: the lift in best-lineup value, never negative."""
    return lineup_value([*roster, player], slots) - lineup_value(roster, slots)


# --- the vectorised form ------------------------------------------------------------
#
# Lives here rather than in the rollout because it is a lineup optimiser, and because the
# season simulation needs it too — importing it from the rollout would make sim.season and
# sim.rollout circular.

def vectorized_lineup_value(
    scores: np.ndarray, positions: Sequence[str], slots: SlotConfig
) -> np.ndarray:
    """Best legal lineup value for every (sim, week) at once.

    `scores` is (n_sims, n_players, n_weeks). Returns (n_sims, n_weeks).

    Same greedy rule as the reference optimiser — dedicated slots first, then flex from
    whoever is left — expressed as sorts along the player axis so numpy does all of it.
    """
    dedicated: dict[str, int] = {}
    flex_slots: list[str] = []
    for slot in slots.slots:
        eligible = slots.eligibility[slot]
        if len(eligible) == 1:
            dedicated[next(iter(eligible))] = dedicated.get(next(iter(eligible)), 0) + 1
        else:
            flex_slots.append(slot)

    by_position: dict[str, list[int]] = {}
    for index, position in enumerate(positions):
        by_position.setdefault(position, []).append(index)

    total = np.zeros((scores.shape[0], scores.shape[2]))
    leftovers: dict[str, np.ndarray] = {}

    for position, indices in by_position.items():
        block = -np.sort(-scores[:, indices, :], axis=1)  # descending along players
        take = dedicated.get(position, 0)
        if take:
            total += block[:, :take, :].sum(axis=1)
        leftovers[position] = block[:, take:, :]

    for slot in flex_slots:
        # sorted: eligibility is a frozenset, and which key carries the leftovers
        # must not depend on iteration order (R7)
        eligible = sorted(p for p in slots.eligibility[slot] if p in leftovers)
        pool = [leftovers[p] for p in eligible if leftovers[p].shape[1]]
        if not pool:
            continue
        stacked = -np.sort(-np.concatenate(pool, axis=1), axis=1)
        total += stacked[:, :1, :].sum(axis=1)
        # the player just used is no longer available to a later flex slot
        remaining = stacked[:, 1:, :]
        for position in eligible:
            leftovers[position] = np.zeros((scores.shape[0], 0, scores.shape[2]))
        if eligible:
            leftovers[eligible[0]] = remaining
    return total


# --- survival ----------------------------------------------------------------------
