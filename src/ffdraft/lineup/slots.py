"""M9 — the league's starting slots and what may fill them.

Slots come from the live league object, never from a hardcoded lineup. Bench slots are
excluded: a benched player scores nothing in a given week, which is precisely why depth
needs M10's season simulation to be worth anything at all.
"""

from __future__ import annotations

from dataclasses import dataclass

BENCH = "BN"
# Slots that hold no starter and therefore contribute nothing to a weekly lineup.
NON_STARTING = frozenset({BENCH, "IR", "TAXI"})


class NonNestedSlotsError(ValueError):
    """Raised when greedy slot filling is not provably optimal for this league.

    Filling the most restrictive slot first is optimal only when every pair of slots has
    nested eligibility — one set contains the other. Two overlapping-but-incomparable
    flex types (say WR/TE and RB/WR) can be filled in an order that costs points, so
    rather than return a quietly suboptimal lineup this refuses to run.
    """


@dataclass(frozen=True)
class SlotConfig:
    """Starting slots in fill order, most restrictive first."""

    slots: tuple[str, ...]
    eligibility: dict[str, frozenset[str]]

    @classmethod
    def from_league(
        cls, roster_positions: list[str], flex_eligibility: dict[str, list[str]]
    ) -> SlotConfig:
        starting = [slot for slot in roster_positions if slot not in NON_STARTING]
        eligibility = {
            slot: frozenset(flex_eligibility.get(slot, [slot])) for slot in set(starting)
        }
        _require_nested(eligibility)
        # Fewest eligible positions first, name as a tiebreak so the order is stable.
        ordered = tuple(sorted(starting, key=lambda s: (len(eligibility[s]), s)))
        return cls(slots=ordered, eligibility=eligibility)

    def eligible(self, slot: str, position: str) -> bool:
        return position in self.eligibility.get(slot, frozenset())

    def __len__(self) -> int:
        return len(self.slots)


def _require_nested(eligibility: dict[str, frozenset[str]]) -> None:
    sets = sorted(set(eligibility.values()), key=len)
    for i, smaller in enumerate(sets):
        for larger in sets[i + 1:]:
            if not smaller <= larger and smaller & larger:
                raise NonNestedSlotsError(
                    f"slot eligibility {sorted(smaller)} and {sorted(larger)} overlap without "
                    f"nesting; greedy filling is not optimal for this league and would "
                    f"silently under-count lineup value"
                )
