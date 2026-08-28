"""M13 — draft state reconciled from the full set of seen picks.

Reconciliation is a pure function of every pick seen so far, never an incremental
mutation. Sleeper can hand back the same pick twice, or hand back picks out of order
after a reconnect, and a loop that applied deltas would double-count or corrupt itself.
Recomputing from the whole set converges no matter what order the picks arrive in.

Positions come from each pick's own `metadata`, not from the crosswalk. A rookie the
crosswalk has never heard of still has a position in the pick, so the roster and needs
stay correct even when the board cannot price him (PRD §8 M13).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from ffdraft.data.sleeper import Pick

BENCH = "BN"


@dataclass(frozen=True)
class DraftedPlayer:
    """One player off the board, described only by what the pick itself carries."""

    player_id: str  # Sleeper id; a team abbreviation for a defense
    position: str | None
    name: str
    picked_by: str
    pick_no: int
    is_mine: bool


@dataclass(frozen=True)
class DraftState:
    drafted: dict[str, DraftedPlayer] = field(default_factory=dict)
    my_players: tuple[DraftedPlayer, ...] = ()
    filled: dict[str, int] = field(default_factory=dict)
    needs: dict[str, int] = field(default_factory=dict)
    overflow: tuple[DraftedPlayer, ...] = ()

    @property
    def drafted_ids(self) -> set[str]:
        return set(self.drafted)

    @property
    def pick_count(self) -> int:
        return len(self.drafted)


def _player_name(pick: Pick) -> str:
    meta = pick.metadata or {}
    first, last = meta.get("first_name", ""), meta.get("last_name", "")
    return f"{first} {last}".strip() or pick.player_id


def assign_slots(
    positions: list[str], roster_positions: list[str], flex_eligibility: dict[str, list[str]]
) -> tuple[list[str | None], list[int]]:
    """Fit drafted positions into roster slots, dedicated first, then flex, then bench.

    Returns (slot_for_each_player, indices_that_did_not_fit). A running back takes the
    RB slot before a FLEX slot, so that the flex stays open for whoever comes next.
    """
    open_slots = list(roster_positions)
    taken = [False] * len(open_slots)
    assigned: list[str | None] = []
    overflow: list[int] = []

    def matches(slot: str, position: str, tier: int) -> bool:
        if tier == 0:
            return slot == position
        if tier == 1:
            return slot in flex_eligibility and position in flex_eligibility[slot]
        return slot == BENCH

    for index, position in enumerate(positions):
        chosen: int | None = None
        for tier in (0, 1, 2):  # dedicated, then flex, then bench
            for slot_index, slot in enumerate(open_slots):
                if not taken[slot_index] and matches(slot, position, tier):
                    chosen = slot_index
                    break
            if chosen is not None:
                break
        if chosen is None:
            assigned.append(None)
            overflow.append(index)
        else:
            taken[chosen] = True
            assigned.append(open_slots[chosen])

    return assigned, overflow


def reconcile(
    picks: list[Pick],
    *,
    my_user_id: str,
    roster_positions: list[str],
    flex_eligibility: dict[str, list[str]],
) -> DraftState:
    """Derive the whole draft state from every pick seen so far.

    Idempotent: passing the same picks again, in any order, with any duplicates, yields
    the identical state.
    """
    # Keyed by player, so a pick seen twice collapses; sorted by pick number, so the
    # order Sleeper happened to return them in cannot change the roster fit.
    unique: dict[str, Pick] = {}
    for pick in picks:
        existing = unique.get(pick.player_id)
        if existing is None or pick.pick_no < existing.pick_no:
            unique[pick.player_id] = pick

    drafted: dict[str, DraftedPlayer] = {}
    for pick in sorted(unique.values(), key=lambda p: p.pick_no):
        drafted[pick.player_id] = DraftedPlayer(
            player_id=pick.player_id,
            position=(pick.metadata or {}).get("position"),
            name=_player_name(pick),
            picked_by=pick.picked_by,
            pick_no=pick.pick_no,
            is_mine=bool(pick.picked_by) and pick.picked_by == my_user_id,
        )

    mine = tuple(p for p in drafted.values() if p.is_mine)
    # A pick whose metadata carries no position matches no dedicated or flex slot, so it
    # lands on the bench — it really is on the roster taking up room. The runtime logs it
    # as an anomaly, which is where the owner sees that something was unrecognised.
    slots, overflow_indexes = assign_slots(
        [p.position or "" for p in mine], roster_positions, flex_eligibility
    )
    filled = Counter(slot for slot in slots if slot is not None)
    needs = Counter(roster_positions)
    needs.subtract(filled)

    return DraftState(
        drafted=drafted,
        my_players=mine,
        filled=dict(filled),
        needs={slot: count for slot, count in sorted(needs.items()) if count > 0},
        overflow=tuple(mine[i] for i in overflow_indexes),
    )
