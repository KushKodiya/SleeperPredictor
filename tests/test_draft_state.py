"""M13 state tests — idempotent reconciliation and positional needs."""

import json
from pathlib import Path

import pytest

from ffdraft.data.sleeper import Pick
from ffdraft.draft.state import assign_slots, reconcile

FIXTURES = Path(__file__).parent / "fixtures"
ROSTER = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "FLEX", "K", "DEF"] + ["BN"] * 5
FLEX = {"FLEX": ["RB", "WR", "TE"]}
ME = "841478247987380224"


def _pick(player_id, position, pick_no, *, picked_by=ME, name="A Player") -> Pick:
    return Pick.model_validate(
        {"player_id": player_id, "picked_by": picked_by, "roster_id": "1", "round": 1,
         "draft_slot": 1, "pick_no": pick_no,
         "metadata": {"position": position, "first_name": name, "last_name": "X"}}
    )


def _state(picks):
    return reconcile(picks, my_user_id=ME, roster_positions=ROSTER, flex_eligibility=FLEX)


# --- 1.1 idempotent reconciliation ------------------------------------------------


def test_a_pick_marks_the_player_drafted():
    state = _state([_pick("100", "RB", 1)])
    assert state.drafted_ids == {"100"}
    assert state.pick_count == 1


def test_duplicate_picks_collapse():
    state = _state([_pick("100", "RB", 1), _pick("100", "RB", 1), _pick("100", "RB", 1)])
    assert state.drafted_ids == {"100"}
    assert len(state.my_players) == 1


def test_out_of_order_picks_converge_to_the_same_state():
    picks = [_pick("100", "RB", 1), _pick("200", "WR", 2), _pick("300", "TE", 3)]
    forward = _state(picks)
    backward = _state(list(reversed(picks)))
    shuffled = _state([picks[1], picks[2], picks[0]])
    assert forward.drafted_ids == backward.drafted_ids == shuffled.drafted_ids
    assert forward.needs == backward.needs == shuffled.needs
    assert [p.player_id for p in forward.my_players] == [p.player_id for p in backward.my_players]


def test_reconciling_twice_is_idempotent():
    picks = [_pick("100", "RB", 1), _pick("200", "WR", 2)]
    assert _state(picks) == _state(picks + picks)


def test_other_managers_picks_leave_the_board_but_not_my_roster():
    state = _state([_pick("100", "RB", 1, picked_by="someone-else")])
    assert state.drafted_ids == {"100"}
    assert state.my_players == ()


def test_empty_picked_by_is_not_mine():
    """Sleeper sends an empty string when a slot has no user (PRD §6.2)."""
    state = _state([_pick("100", "RB", 1, picked_by="")])
    assert state.drafted_ids == {"100"}
    assert state.my_players == ()


def test_defense_pick_reconciles():
    state = _state([_pick("DEN", "DEF", 1)])
    assert state.drafted_ids == {"DEN"}
    assert state.filled.get("DEF") == 1


# --- 1.2 roster and needs ---------------------------------------------------------


def test_needs_start_as_the_full_roster():
    needs = _state([]).needs
    assert needs["QB"] == 1 and needs["RB"] == 2 and needs["BN"] == 5
    assert sum(needs.values()) == len(ROSTER)


def test_needs_recompute_after_a_pick():
    before = _state([]).needs
    after = _state([_pick("100", "RB", 1)]).needs
    assert before["RB"] == 2
    assert after["RB"] == 1


def test_dedicated_slots_fill_before_flex():
    state = _state([_pick(str(i), "RB", i) for i in range(1, 3)])
    assert state.filled["RB"] == 2
    assert "FLEX" not in state.filled
    assert state.needs["FLEX"] == 2


def test_third_running_back_takes_a_flex_slot():
    state = _state([_pick(str(i), "RB", i) for i in range(1, 4)])
    assert state.filled["RB"] == 2
    assert state.filled["FLEX"] == 1
    assert state.needs["FLEX"] == 1


def test_flex_ineligible_position_goes_to_the_bench():
    """A second quarterback cannot use a FLEX slot in this league."""
    state = _state([_pick("1", "QB", 1), _pick("2", "QB", 2)])
    assert state.filled["QB"] == 1
    assert state.filled["BN"] == 1
    assert "FLEX" not in state.filled


def test_a_full_roster_reports_no_needs():
    positions = ["QB", "RB", "RB", "WR", "WR", "TE", "RB", "WR", "K", "DEF",
                 "QB", "RB", "WR", "TE", "K"]
    state = _state([_pick(str(i), p, i) for i, p in enumerate(positions, start=1)])
    assert state.needs == {}
    assert state.overflow == ()


def test_a_pick_beyond_the_roster_is_overflow_not_a_crash():
    full = ["QB", "RB", "RB", "WR", "WR", "TE", "RB", "WR", "K", "DEF",
            "QB", "RB", "WR", "TE", "K"]
    state = _state([_pick(str(i), p, i) for i, p in enumerate([*full, "WR", "RB"], start=1)])
    assert state.needs == {}
    assert len(state.overflow) == 2  # the roster was already full


def test_a_position_with_no_slot_left_goes_to_the_bench_then_overflows():
    """Only one QB slot and no flex eligibility, so extra quarterbacks fill the bench."""
    state = _state([_pick(str(i), "QB", i) for i in range(1, 9)])
    assert state.filled["QB"] == 1
    assert state.filled["BN"] == 5
    assert len(state.overflow) == 2


def test_a_pick_with_no_position_metadata_lands_on_the_bench():
    """It is genuinely on the roster, so it takes the least-committal slot, not a starter."""
    state = _state([_pick("100", None, 1)])
    assert state.drafted_ids == {"100"}
    assert state.filled == {"BN": 1}
    assert state.needs["RB"] == 2  # no starting slot was consumed


def test_assign_slots_prefers_dedicated_then_flex_then_bench():
    slots, overflow = assign_slots(["RB", "RB", "RB", "QB", "QB"], ROSTER, FLEX)
    assert slots == ["RB", "RB", "FLEX", "QB", "BN"]
    assert overflow == []


# --- 4.1 the real draft, replayed -------------------------------------------------


def test_a_real_completed_draft_reconciles_every_pick():
    """Every pick of the owner's actual 2025 draft, replayed through reconciliation."""
    raw = json.loads((FIXTURES / "draft_picks_2025.json").read_text(encoding="utf-8"))
    picks = [Pick.model_validate(p) for p in raw]
    assert len(picks) == 120

    state = reconcile(picks, my_user_id=ME, roster_positions=ROSTER, flex_eligibility=FLEX)
    assert state.pick_count == 120  # every pick landed, none collapsed or lost
    assert state.drafted_ids == {p.player_id for p in picks}
    assert len(state.my_players) == 15  # a full roster for one manager
    assert state.needs == {}

    # and the same picks in reverse, with every one duplicated, converge identically
    scrambled = list(reversed(picks)) + picks
    assert reconcile(
        scrambled, my_user_id=ME, roster_positions=ROSTER, flex_eligibility=FLEX
    ) == state


def test_real_draft_includes_defenses_and_they_reconcile():
    raw = json.loads((FIXTURES / "draft_picks_2025.json").read_text(encoding="utf-8"))
    picks = [Pick.model_validate(p) for p in raw]
    defenses = {p.player_id for p in picks if p.is_defense}
    assert len(defenses) == 8  # one per team in an 8-team league
    state = reconcile(picks, my_user_id=ME, roster_positions=ROSTER, flex_eligibility=FLEX)
    assert defenses <= state.drafted_ids


@pytest.mark.parametrize("seed", range(5))
def test_real_draft_converges_from_any_arrival_order(seed):
    import random

    raw = json.loads((FIXTURES / "draft_picks_2025.json").read_text(encoding="utf-8"))
    picks = [Pick.model_validate(p) for p in raw]
    baseline = reconcile(picks, my_user_id=ME, roster_positions=ROSTER, flex_eligibility=FLEX)

    rng = random.Random(seed)  # R7: seeded, so a failure is reproducible
    shuffled = picks[:]
    rng.shuffle(shuffled)
    assert reconcile(
        shuffled, my_user_id=ME, roster_positions=ROSTER, flex_eligibility=FLEX
    ) == baseline
