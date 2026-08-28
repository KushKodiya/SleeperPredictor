"""M9 tests — slot resolution and the hand-verified marginal value cases from the PRD."""

import pytest

from ffdraft.lineup.slots import NonNestedSlotsError, SlotConfig
from ffdraft.lineup.value import Player, lineup_value, marginal_value, optimal_lineup

FLEX = {"FLEX": ["RB", "WR", "TE"]}
LEAGUE = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "FLEX", "K", "DEF"] + ["BN"] * 5
# The PRD's hand cases are written against a 2WR/1FLEX league.
TWO_WR_ONE_FLEX = SlotConfig.from_league(["WR", "WR", "FLEX"], FLEX)


def wr(points, name="w"):
    return Player(f"{name}{points}", "WR", points)


# --- 2.1 slot and FLEX resolution --------------------------------------------------


def test_bench_slots_are_not_startable():
    slots = SlotConfig.from_league(LEAGUE, FLEX)
    assert "BN" not in slots.slots
    assert len(slots) == 10


def test_flex_eligibility_comes_from_the_league():
    slots = SlotConfig.from_league(LEAGUE, FLEX)
    assert slots.eligible("FLEX", "RB") and slots.eligible("FLEX", "TE")
    assert not slots.eligible("FLEX", "QB")
    assert slots.eligible("QB", "QB") and not slots.eligible("QB", "RB")


def test_restrictive_slots_are_filled_before_flexible_ones():
    slots = SlotConfig.from_league(LEAGUE, FLEX)
    positions = [slots.eligibility[s] for s in slots.slots]
    assert [len(p) for p in positions] == sorted(len(p) for p in positions)
    assert slots.slots[-1] == "FLEX"  # the loosest slot fills last


def test_a_running_back_takes_the_rb_slot_before_the_flex():
    slots = SlotConfig.from_league(["RB", "FLEX"], FLEX)
    lineup = optimal_lineup([Player("a", "RB", 100.0)], slots)
    assert slots.slots[next(iter(lineup))] == "RB"


def test_the_flex_takes_the_best_remaining_player_not_the_leftover():
    slots = SlotConfig.from_league(["RB", "FLEX"], FLEX)
    roster = [Player("rb1", "RB", 100.0), Player("rb2", "RB", 90.0), Player("wr", "WR", 95.0)]
    assert lineup_value(roster, slots) == pytest.approx(195.0)  # RB 100 + WR 95, not RB 90


def test_a_superflex_league_lets_a_quarterback_flex():
    slots = SlotConfig.from_league(
        ["QB", "SUPER_FLEX"], {"SUPER_FLEX": ["QB", "RB", "WR", "TE"]}
    )
    roster = [Player("q1", "QB", 300.0), Player("q2", "QB", 280.0)]
    assert lineup_value(roster, slots) == pytest.approx(580.0)


def test_non_nested_flex_types_are_refused_rather_than_filled_wrongly():
    """WR/TE and RB/WR overlap without nesting, so greedy filling is not optimal."""
    with pytest.raises(NonNestedSlotsError, match="nesting"):
        SlotConfig.from_league(
            ["REC_FLEX", "WRRB_FLEX"], {"REC_FLEX": ["WR", "TE"], "WRRB_FLEX": ["RB", "WR"]}
        )


# --- 2.2 the hand-verified value cases --------------------------------------------


def test_empty_roster_values_a_250_point_receiver_at_250():
    assert marginal_value(wr(250.0), [], TWO_WR_ONE_FLEX) == pytest.approx(250.0)


def test_third_receiver_fills_the_flex_and_the_fourth_is_worth_nothing():
    """The PRD's case: 2WR/1FLEX, two 250s already rostered."""
    roster = [wr(250.0, "a"), wr(250.0, "b")]
    third = wr(240.0, "c")
    assert marginal_value(third, roster, TWO_WR_ONE_FLEX) == pytest.approx(240.0)
    assert marginal_value(wr(240.0, "d"), [*roster, third], TWO_WR_ONE_FLEX) == pytest.approx(0.0)


def test_a_kicker_adds_nothing_when_the_kicker_slot_is_filled():
    slots = SlotConfig.from_league(LEAGUE, FLEX)
    roster = [Player("k1", "K", 130.0)]
    assert marginal_value(Player("k2", "K", 125.0), roster, slots) == pytest.approx(0.0)


def test_marginal_value_is_the_difference_in_lineup_value():
    slots = SlotConfig.from_league(LEAGUE, FLEX)
    roster = [Player("rb1", "RB", 200.0), Player("wr1", "WR", 180.0)]
    player = Player("te1", "TE", 150.0)
    assert marginal_value(player, roster, slots) == pytest.approx(
        lineup_value([*roster, player], slots) - lineup_value(roster, slots)
    )


def test_a_better_player_at_a_full_position_still_adds_value():
    """Upgrading a starter is worth the difference, not zero."""
    roster = [wr(200.0, "a"), wr(190.0, "b"), wr(180.0, "c")]  # WR, WR, FLEX all full
    assert marginal_value(wr(250.0, "new"), roster, TWO_WR_ONE_FLEX) == pytest.approx(70.0)


def test_marginal_value_is_never_negative():
    roster = [wr(200.0, "a"), wr(190.0, "b")]
    assert marginal_value(wr(10.0, "bad"), roster, TWO_WR_ONE_FLEX) >= 0.0


def test_an_empty_roster_is_worth_nothing():
    assert lineup_value([], SlotConfig.from_league(LEAGUE, FLEX)) == pytest.approx(0.0)


def test_an_ineligible_position_cannot_fill_a_slot():
    slots = SlotConfig.from_league(["QB"], {})
    assert lineup_value([Player("rb", "RB", 300.0)], slots) == pytest.approx(0.0)


def test_the_lineup_is_deterministic_when_players_tie_on_points():
    roster = [wr(100.0, "b"), wr(100.0, "a"), wr(100.0, "c")]
    first = optimal_lineup(roster, TWO_WR_ONE_FLEX)
    second = optimal_lineup(list(reversed(roster)), TWO_WR_ONE_FLEX)
    assert [p.player_id for p in first.values()] == [p.player_id for p in second.values()]
