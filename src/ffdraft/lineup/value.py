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
