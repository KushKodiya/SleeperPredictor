"""M14 tests. The leakage guard matters more than the harness around it (PRD §8 M14)."""

import json
from pathlib import Path

import polars as pl
import pytest

from ffdraft.backtest.harness import (
    BASELINES,
    BacktestResult,
    Candidate,
    DraftContext,
    LeakageError,
    adp_follow,
    assert_point_in_time,
    head_to_head,
    raw_consensus,
    reconstruct_rosters,
    run_draft,
    score_roster_actuals,
    snake_order,
    static_vor,
    summarize,
)
from ffdraft.lineup.slots import SlotConfig
from ffdraft.lineup.value import Player

FIXTURES = Path(__file__).parent / "fixtures"
FLEX = {"FLEX": ["RB", "WR", "TE"]}
ROSTER = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "FLEX", "K", "DEF"] + ["BN"] * 5
SLOTS = SlotConfig.from_league(ROSTER, FLEX)


def _cand(pid, position, adp, ecr=None, vor=None):
    return Candidate(pid, f"Player {pid}", position, adp=adp, ecr=ecr or adp, vor=vor)


def _board(n=200):
    """A board deep enough to fill a whole draft, with every position represented."""
    cycle = ["RB", "WR", "WR", "RB", "TE", "QB", "WR", "RB", "K", "DEF"]
    return [
        _cand(f"p{i}", cycle[i % len(cycle)], adp=float(i + 1), ecr=float(i + 1),
              vor=float(n - i))
        for i in range(n)
    ]


# --- 2.1 the leakage guard ---------------------------------------------------------


def test_future_season_rows_are_rejected():
    frame = pl.DataFrame([{"season": 2023, "x": 1}, {"season": 2024, "x": 2}])
    with pytest.raises(LeakageError, match="would leak into the 2024 draft"):
        assert_point_in_time(frame, target_season=2024, source="board")


def test_the_target_season_itself_is_leakage():
    """`>=`, not `>`: a decision may not see the season it is drafting for."""
    with pytest.raises(LeakageError, match=r"\[2024\]"):
        assert_point_in_time(
            pl.DataFrame([{"season": 2024}]), target_season=2024, source="board"
        )


def test_strictly_prior_seasons_pass():
    frame = pl.DataFrame([{"season": s} for s in (2020, 2021, 2022, 2023)])
    assert_point_in_time(frame, target_season=2024, source="board")  # no raise


def test_a_frame_without_a_season_column_cannot_be_vouched_for():
    """Unable to check is not the same as safe."""
    with pytest.raises(LeakageError, match="no season column"):
        assert_point_in_time(pl.DataFrame([{"x": 1}]), target_season=2024, source="board")


def test_the_error_names_the_offending_seasons():
    frame = pl.DataFrame([{"season": s} for s in (2019, 2025, 2026)])
    with pytest.raises(LeakageError) as exc:
        assert_point_in_time(frame, target_season=2024, source="projections")
    assert "2025" in str(exc.value) and "2026" in str(exc.value)
    assert "projections" in str(exc.value)


# --- 1.1 point-in-time replay ------------------------------------------------------


def test_snake_order_reverses_every_other_round():
    assert snake_order(teams=4, rounds=3) == [1, 2, 3, 4, 4, 3, 2, 1, 1, 2, 3, 4]


def test_every_team_gets_the_same_number_of_picks():
    order = snake_order(teams=8, rounds=15)
    assert len(order) == 120
    assert {order.count(slot) for slot in range(1, 9)} == {15}


def test_a_draft_fills_every_roster_without_duplicates():
    rosters = run_draft(_board(), teams=8, rounds=15, my_slot=3, strategy=static_vor,
                        roster_positions=ROSTER, flex_eligibility=FLEX)
    assert set(rosters) == set(range(1, 9))
    assert all(len(r) == 15 for r in rosters.values())
    picked = [p.player_id for roster in rosters.values() for p in roster]
    assert len(picked) == len(set(picked))  # nobody drafted twice


def test_the_draft_is_deterministic():
    args = {"teams": 8, "rounds": 15, "my_slot": 3, "strategy": static_vor,
            "roster_positions": ROSTER, "flex_eligibility": FLEX}
    first = run_draft(_board(), **args)
    second = run_draft(_board(), **args)
    assert {k: [p.player_id for p in v] for k, v in first.items()} == {
        k: [p.player_id for p in v] for k, v in second.items()
    }


def test_opponents_follow_adp_so_the_first_picks_are_the_top_of_the_board():
    rosters = run_draft(_board(), teams=8, rounds=15, my_slot=8, strategy=static_vor,
                        roster_positions=ROSTER, flex_eligibility=FLEX)
    assert rosters[1][0].adp == 1.0  # slot 1 takes ADP 1


def test_a_real_draft_reconstructs_into_rosters():
    """The owner's actual 2025 draft, replayed into the rosters it produced."""
    raw = json.loads((FIXTURES / "draft_picks_2025.json").read_text(encoding="utf-8"))
    picks = pl.DataFrame(
        [{"roster_id": str(p["roster_id"]), "player_id": p["player_id"], "pick_no": p["pick_no"]}
         for p in raw]
    )
    rosters = reconstruct_rosters(picks)
    assert len(rosters) == 8
    assert all(len(r) == 15 for r in rosters.values())
    assert sum(len(r) for r in rosters.values()) == 120
    everyone = [pid for r in rosters.values() for pid in r]
    assert len(everyone) == len(set(everyone))


# --- 1.2 scoring on realized points ------------------------------------------------


def test_a_roster_is_scored_on_what_actually_happened():
    roster = [Player("a", "QB", 0.0), Player("b", "RB", 0.0)]
    weekly = pl.DataFrame(
        [{"week": w, "player_id": "a", "points": 20.0} for w in (1, 2)]
        + [{"week": w, "player_id": "b", "points": 10.0} for w in (1, 2)]
    )
    assert score_roster_actuals(roster, weekly, SLOTS) == pytest.approx(60.0)


def test_only_the_best_legal_lineup_counts_each_week():
    """Three receivers, two WR slots and a FLEX: the fourth is on the bench."""
    slots = SlotConfig.from_league(["WR", "WR"], {})
    roster = [Player(p, "WR", 0.0) for p in ("a", "b", "c")]
    weekly = pl.DataFrame(
        [{"week": 1, "player_id": "a", "points": 30.0},
         {"week": 1, "player_id": "b", "points": 20.0},
         {"week": 1, "player_id": "c", "points": 10.0}]
    )
    assert score_roster_actuals(roster, weekly, slots) == pytest.approx(50.0)  # not 60


def test_players_not_on_the_roster_are_ignored():
    roster = [Player("a", "QB", 0.0)]
    weekly = pl.DataFrame(
        [{"week": 1, "player_id": "a", "points": 20.0},
         {"week": 1, "player_id": "z", "points": 99.0}]
    )
    assert score_roster_actuals(roster, weekly, SLOTS) == pytest.approx(20.0)


def test_a_week_a_player_did_not_play_scores_nothing():
    roster = [Player("a", "QB", 0.0)]
    weekly = pl.DataFrame([{"week": 1, "player_id": "a", "points": 20.0}])  # no week 2 row
    assert score_roster_actuals(roster, weekly, SLOTS) == pytest.approx(20.0)


# --- 3.1 the baselines -------------------------------------------------------------


def _context(available, roster=()):
    return DraftContext(tuple(available), tuple(roster), tuple(ROSTER), FLEX, pick_number=1)


def test_adp_follow_takes_the_market_next():
    board = [_cand("late", "RB", adp=40.0), _cand("early", "RB", adp=2.0)]
    assert adp_follow(_context(board)).player_id == "early"


def test_raw_consensus_takes_the_best_expert_rank():
    board = [_cand("a", "RB", adp=1.0, ecr=9.0), _cand("b", "RB", adp=50.0, ecr=2.0)]
    assert raw_consensus(_context(board)).player_id == "b"


def test_static_vor_takes_the_most_value_over_replacement():
    board = [_cand("a", "RB", adp=1.0, vor=10.0), _cand("b", "TE", adp=50.0, vor=90.0)]
    assert static_vor(_context(board)).player_id == "b"


def test_every_baseline_respects_open_starting_slots():
    """With only a kicker slot left to fill, no strategy may take a sixth receiver."""
    roster = [Candidate(f"r{i}", "n", pos, adp=1.0, ecr=1.0, vor=1.0)
              for i, pos in enumerate(["QB", "RB", "RB", "WR", "WR", "TE", "RB", "WR", "DEF"])]
    board = [_cand("wr_stud", "WR", adp=1.0, ecr=1.0, vor=999.0),
             _cand("kicker", "K", adp=200.0, ecr=200.0, vor=1.0)]
    for name, strategy in BASELINES.items():
        assert strategy(_context(board, roster)).player_id == "kicker", name


def test_strategies_fall_back_to_anyone_once_the_starters_are_set():
    roster = [Candidate(f"r{i}", "n", pos, adp=1.0, ecr=1.0, vor=1.0)
              for i, pos in enumerate(ROSTER[:10])]
    board = [_cand("bench_wr", "WR", adp=5.0, vor=50.0)]
    assert static_vor(_context(board, roster)).player_id == "bench_wr"


def test_a_baseline_with_no_metric_still_returns_a_pick():
    """A player missing an ADP must not stall the draft."""
    board = [Candidate("x", "X", "RB")]
    assert adp_follow(_context(board)).player_id == "x"


@pytest.mark.parametrize("name", sorted(BASELINES))
def test_each_baseline_drafts_a_sane_full_roster(name):
    rosters = run_draft(_board(), teams=8, rounds=15, my_slot=4, strategy=BASELINES[name],
                        roster_positions=ROSTER, flex_eligibility=FLEX)
    mine = rosters[4]
    assert len(mine) == 15
    positions = {p.position for p in mine}
    # a sane roster fills its mandatory singleton slots rather than hoarding one position
    assert {"QB", "K", "DEF"} <= positions


# --- 3.2 distributional reporting --------------------------------------------------


def _results():
    result = BacktestResult()
    for season in (2022, 2023):
        for slot in (1, 2, 3):
            result.add(season=season, slot=slot, strategy="adp_follow", points=1000.0 + slot)
            result.add(season=season, slot=slot, strategy="static_vor",
                       points=1000.0 + slot + (20.0 if slot < 3 else -60.0))
    return result.frame()


def test_the_report_is_a_distribution_not_a_single_number():
    summary = summarize(_results())
    assert set(summary.columns) >= {"min", "p25", "median", "p75", "max", "sd", "seasons", "slots"}
    assert summary.height == 2
    assert summary["seasons"].to_list() == [2, 2]
    assert summary["slots"].to_list() == [3, 3]


def test_the_report_keeps_every_season_and_slot():
    raw = _results()
    assert raw.height == 12  # 2 seasons x 3 slots x 2 strategies, nothing collapsed
    assert raw.group_by(["season", "slot"]).len()["len"].to_list() == [2] * 6


def test_head_to_head_shows_a_strategy_that_wins_on_average_but_not_everywhere():
    """static_vor beats ADP at two slots and loses badly at the third — the mean lies."""
    paired = head_to_head(_results(), baseline="adp_follow")
    row = paired.filter(pl.col("strategy") == "static_vor").to_dicts()[0]
    assert row["drafts"] == 6
    assert row["wins"] == 4          # not 6 — it loses at slot 3 in both seasons
    assert row["worst"] == pytest.approx(-60.0)
    assert row["best"] == pytest.approx(20.0)


def test_head_to_head_needs_the_baseline_present():
    with pytest.raises(ValueError, match="no rows for baseline"):
        head_to_head(_results(), baseline="nonexistent")
