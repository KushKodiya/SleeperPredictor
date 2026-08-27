"""M8 tests — replacement level, VOR ordering, and tier breaks."""

import numpy as np
import polars as pl
import pytest

from ffdraft.valuation.calibration import fit_calibration
from ffdraft.valuation.replacement import (
    replacement_levels,
    rostered_depth,
    starters_per_team,
)
from ffdraft.valuation.tiers import assign_tiers, points_standard_error
from ffdraft.valuation.vor import value_over_replacement

ROSTER = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "FLEX", "K", "DEF"] + ["BN"] * 5
FLEX = {"FLEX": ["RB", "WR", "TE"]}


# --- 3.1 replacement level --------------------------------------------------------


def test_flex_slots_spread_across_the_eligible_positions():
    starters = starters_per_team(ROSTER, FLEX)
    assert starters["QB"] == pytest.approx(1.0)
    assert starters["RB"] == pytest.approx(2.8)  # 2 dedicated + 40% of 2 flex
    assert starters["WR"] == pytest.approx(2.8)
    assert starters["TE"] == pytest.approx(1.4)  # 1 dedicated + 20% of 2 flex
    assert "BN" not in starters and "FLEX" not in starters


def test_rostered_depth_scales_with_league_size():
    small = rostered_depth(ROSTER, FLEX, teams=8)
    large = rostered_depth(ROSTER, FLEX, teams=12)
    assert large["RB"] > small["RB"]
    assert small["QB"] == 8 and large["QB"] == 12


def _weekly(position="RB", players=40, weeks=17, seasons=(2024,)) -> pl.DataFrame:
    rows = []
    for season in seasons:
        for week in range(1, weeks + 1):
            for i in range(1, players + 1):
                rows.append({"season": season, "week": week, "gsis_id": f"p{i}",
                             "position": position, "points": 30.0 - 0.5 * i})
    return pl.DataFrame(rows)


def test_replacement_is_strictly_positive():
    levels = replacement_levels(
        _weekly(), depth={"RB": 22}, percentile=0.5, games_per_season=17
    )
    assert levels["weekly_replacement"].item() > 0
    assert levels["replacement_points"].item() > 0


def test_replacement_is_the_best_player_nobody_rostered():
    """With 22 RBs rostered, replacement is what the 23rd-best RB scored on the season."""
    levels = replacement_levels(
        _weekly(), depth={"RB": 22}, percentile=0.5, games_per_season=17
    )
    per_week = 30.0 - 0.5 * 23
    assert levels["replacement_points"].item() == pytest.approx(per_week * 17)
    assert levels["weekly_replacement"].item() == pytest.approx(per_week)


def test_deeper_rosters_lower_replacement_level():
    shallow = replacement_levels(_weekly(), depth={"RB": 10}, percentile=0.5, games_per_season=17)
    deep = replacement_levels(_weekly(), depth={"RB": 30}, percentile=0.5, games_per_season=17)
    assert deep["weekly_replacement"].item() < shallow["weekly_replacement"].item()


def test_replacement_is_measured_in_season_points_not_a_healthy_week():
    """A replacement who missed games must not be priced as if he played every week.

    Projections are season totals that already carry their player's missed games, so
    replacement has to be in the same unit. Scaling this player's excellent median week
    up to 17 games would price replacement at 510 instead of the 270 he actually scored.
    """
    rows = []
    for i in range(1, 23):  # 22 rostered players, 20/week all season
        rows += [{"season": 2024, "week": w, "gsis_id": f"p{i}", "position": "RB",
                  "points": 20.0} for w in range(1, 18)]
    rows += [{"season": 2024, "week": w, "gsis_id": "hurt", "position": "RB",
              "points": 30.0} for w in range(1, 10)]      # 9 games at 30 -> 270
    rows += [{"season": 2024, "week": w, "gsis_id": "steady", "position": "RB",
              "points": 15.0} for w in range(1, 18)]      # 17 games at 15 -> 255

    levels = replacement_levels(
        pl.DataFrame(rows), depth={"RB": 22}, percentile=0.5, games_per_season=17
    )
    assert levels["replacement_points"].item() == pytest.approx(270.0)  # not 30 * 17


def test_lookback_limits_the_seasons_considered():
    weekly = pl.concat([_weekly(seasons=(2020,)), _weekly(seasons=(2024,))])
    levels = replacement_levels(
        weekly, depth={"RB": 22}, percentile=0.5, games_per_season=17, lookback_seasons=1
    )
    assert levels["weekly_replacement"].item() > 0


# --- 3.2 value over replacement ---------------------------------------------------


_PROJECTIONS = pl.DataFrame(
    [{"gsis_id": "rb1", "position": "RB", "projected_points": 260.0},
     {"gsis_id": "te1", "position": "TE", "projected_points": 240.0},
     {"gsis_id": "rb2", "position": "RB", "projected_points": 200.0}]
)
_REPLACEMENT = pl.DataFrame(
    [{"position": "RB", "replacement_points": 150.0},
     {"position": "TE", "replacement_points": 90.0}]
)


def test_vor_is_points_minus_positional_replacement():
    out = {r["gsis_id"]: r for r in
           value_over_replacement(_PROJECTIONS, _REPLACEMENT).iter_rows(named=True)}
    assert out["rb1"]["vor"] == pytest.approx(110.0)
    assert out["te1"]["vor"] == pytest.approx(150.0)


def test_scarcity_can_outrank_raw_points():
    """The whole point of VOR: a 240-point TE beats a 260-point RB when TEs are thin."""
    ranked = value_over_replacement(_PROJECTIONS, _REPLACEMENT)["gsis_id"].to_list()
    assert ranked[0] == "te1"


@pytest.mark.parametrize(("scale", "shift"), [(2.0, 0.0), (0.5, 0.0), (1.0, 25.0), (3.0, -10.0)])
def test_vor_ordering_is_stable_under_affine_rescaling(scale, shift):
    """Replacement is measured in the same units, so it rescales with the points."""
    base = value_over_replacement(_PROJECTIONS, _REPLACEMENT)["gsis_id"].to_list()
    moved = value_over_replacement(
        _PROJECTIONS.with_columns(pl.col("projected_points") * scale + shift),
        _REPLACEMENT.with_columns(pl.col("replacement_points") * scale + shift),
    )["gsis_id"].to_list()
    assert moved == base


def test_position_without_replacement_level_raises():
    with pytest.raises(ValueError, match="TE"):
        value_over_replacement(
            _PROJECTIONS, _REPLACEMENT.filter(pl.col("position") == "RB")
        )


# --- 3.3 tiers --------------------------------------------------------------------


def _calibration():
    rows = [
        {"season": s, "gsis_id": f"r{r}-{s}", "position": "RB", "ecr": float(r),
         "actual_points": 300.0 - 5.0 * r}
        for s in (2022, 2023, 2024) for r in range(1, 41)
    ]
    return fit_calibration(pl.DataFrame(rows), target_season=2025, fit_pool={"RB": 40})


def test_expert_disagreement_becomes_a_points_standard_error():
    cal = _calibration()
    ranked = pl.DataFrame(
        [{"gsis_id": "a", "position": "RB", "ecr": 10.0, "sd": 0.0},
         {"gsis_id": "b", "position": "RB", "ecr": 10.0, "sd": 5.0}]
    )
    out = {r["gsis_id"]: r["points_se"] for r in
           points_standard_error(ranked, cal).iter_rows(named=True)}
    assert out["a"] == pytest.approx(0.0)  # experts agree -> no uncertainty
    assert out["b"] > 0  # experts disagree -> real spread in points
    assert np.isfinite(out["b"])


def test_a_gap_bigger_than_the_standard_error_starts_a_new_tier():
    board = pl.DataFrame(
        [{"gsis_id": "a", "vor": 100.0, "points_se": 5.0},
         {"gsis_id": "b", "vor": 98.0, "points_se": 5.0},   # gap 2 < se 5 -> same tier
         {"gsis_id": "c", "vor": 60.0, "points_se": 5.0}]   # gap 38 > se 5 -> new tier
    )
    tiers = {r["gsis_id"]: r["tier"] for r in assign_tiers(board).iter_rows(named=True)}
    assert tiers["a"] == tiers["b"] == 1
    assert tiers["c"] == 2


def test_a_flat_board_is_one_tier():
    board = pl.DataFrame(
        [{"gsis_id": f"p{i}", "vor": 100.0 - i, "points_se": 10.0} for i in range(6)]
    )
    assert assign_tiers(board)["tier"].unique().to_list() == [1]


def test_tiers_never_decrease_down_the_board():
    board = pl.DataFrame(
        [{"gsis_id": f"p{i}", "vor": 100.0 - i * 7.0, "points_se": 3.0} for i in range(10)]
    )
    tiers = assign_tiers(board)["tier"].to_list()
    assert tiers == sorted(tiers)


# --- board assembly ---------------------------------------------------------------


def test_team_defenses_get_a_synthetic_identity():
    """Defenses never join the crosswalk, so the team abbreviation is their id."""
    from ffdraft.valuation.board import defense_rankings

    unresolved = pl.DataFrame(
        [{"season": 2026, "gsis_id": None, "name": "Houston Texans", "position": "DST",
          "team": "HOU", "ecr": 1.0, "sd": 0.5, "scrape_date": "2026-08-21"},
         {"season": 2026, "gsis_id": None, "name": "Ghost Rookie", "position": "WR",
          "team": "KC", "ecr": 90.0, "sd": 4.0, "scrape_date": "2026-08-21"}]
    )
    out = defense_rankings(unresolved)
    assert out.height == 1
    assert out["gsis_id"].item() == "DEF_HOU"
    assert out["position"].item() == "DEF"


def test_assign_tiers_rejects_an_unsorted_board():
    """Tiers are assigned against neighbours, so order is a precondition, not a hint."""
    board = pl.DataFrame(
        [{"gsis_id": "a", "vor": 10.0, "points_se": 1.0},
         {"gsis_id": "b", "vor": 50.0, "points_se": 1.0}]
    )
    with pytest.raises(ValueError, match="already sorted"):
        assign_tiers(board)


def test_bench_slots_count_toward_roster_depth():
    """A player stashed on a bench is not on the waiver wire."""
    with_bench = rostered_depth(ROSTER, FLEX, teams=8)
    without_bench = rostered_depth([p for p in ROSTER if p != "BN"], FLEX, teams=8)
    assert with_bench["RB"] > without_bench["RB"]
    assert with_bench["WR"] > without_bench["WR"]
    assert with_bench["QB"] == without_bench["QB"]  # nobody hoards backup quarterbacks here


def test_replacement_level_is_independent_of_row_order():
    """R7: two players tied at the boundary rank must not be separated by row order."""
    tied = pl.DataFrame(
        [{"season": 2024, "week": w, "gsis_id": g, "position": "RB", "points": p}
         for g, p in [("a", 20.0), ("b", 12.0), ("c", 12.0), ("d", 4.0)]
         for w in range(1, 18)]
    )
    levels = {
        replacement_levels(
            tied.sample(fraction=1.0, shuffle=True, seed=seed),
            depth={"RB": 1}, percentile=0.5, games_per_season=17,
        )["replacement_points"].item()
        for seed in range(8)
    }
    assert len(levels) == 1, levels


def test_only_flex_slots_the_league_fields_make_a_position_hoardable():
    """SUPER_FLEX is defined in config but this roster has none, so QB gets no bench."""
    config_flex = {
        "FLEX": ["RB", "WR", "TE"],
        "SUPER_FLEX": ["QB", "RB", "WR", "TE"],  # defined, but not in ROSTER
        "REC_FLEX": ["WR", "TE"],
    }
    depth = rostered_depth(ROSTER, config_flex, teams=8)
    assert depth["QB"] == 8  # 1 starter x 8 teams, no bench share

    superflex_roster = [*ROSTER, "SUPER_FLEX"]
    assert rostered_depth(superflex_roster, config_flex, teams=8)["QB"] > 8
