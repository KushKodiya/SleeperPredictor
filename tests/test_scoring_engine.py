"""M6 tests. Hand-computed stat lines against the owner's real scoring settings."""

import json
from pathlib import Path

import polars as pl
import pytest

from ffdraft.scoring.engine import (
    DEFENSE_RULES,
    PLAYER_RULES,
    parse_settings,
    score_defenses,
    score_players,
)

GOLDEN = Path(__file__).parent / "fixtures" / "golden"
LEAGUE_SETTINGS = json.loads((GOLDEN / "league.json").read_text(encoding="utf-8"))[
    "scoring_settings"
]


def _columns(rules: dict[str, pl.Expr]) -> set[str]:
    """Every stat column the rule table reads, taken from the expressions themselves."""
    return {name for expr in rules.values() for name in expr.meta.root_names()}


def _row(rules: dict[str, pl.Expr], **stats) -> pl.DataFrame:
    """A one-row stat frame: every column the rules read is 0 unless named in `stats`."""
    base = {name: 0.0 for name in _columns(rules)}
    base.update({"season": 2025, "week": 1, "player_id": "00-0000001", "position": "QB", "team": "KC"})
    base.update(stats)
    return pl.DataFrame([base])


# --- 1.1 / 1.2 parsing ------------------------------------------------------------


def test_parse_settings_keeps_the_leagues_own_weights():
    rules = parse_settings({"rec": 1.0, "pass_yd": 0.04, "pass_td": 4.0, "qb_hit": 0.0})
    assert rules.weight("rec") == 1.0
    assert rules.weight("pass_yd") == 0.04
    assert rules.weight("pass_td") == 4.0
    assert rules.weight("rush_td") == 0.0  # absent key is worth nothing, not an error


def test_the_live_league_settings_parse():
    rules = parse_settings(LEAGUE_SETTINGS)
    assert rules.weight("rec") == pytest.approx(1.0)  # full PPR
    assert rules.weight("pass_td") == pytest.approx(6.0)
    assert rules.weight("pts_allow_0") == pytest.approx(10.0)


def test_unrecognized_key_raises_naming_that_key():
    with pytest.raises(ValueError, match="wizard_bonus"):
        parse_settings({"rec": 1.0, "wizard_bonus": 3.0})


def test_recognized_key_with_no_stat_source_raises_only_when_weighted():
    parse_settings({"tkl_solo": 0.0})  # present but unused: fine
    with pytest.raises(ValueError, match="tkl_solo"):
        parse_settings({"tkl_solo": 1.0})


def test_folded_defense_key_weighted_differently_raises():
    with pytest.raises(ValueError, match="def_st_td"):
        parse_settings({"st_td": 6.0, "def_st_td": 3.0})


# --- 2.1 skill players and kickers ------------------------------------------------


def test_hand_computed_quarterback_week():
    rules = parse_settings(LEAGUE_SETTINGS)
    stats = _row(
        PLAYER_RULES,
        passing_yards=305.0,      # .05  -> 15.25
        passing_tds=2.0,          # 6    -> 12
        passing_interceptions=1.0,  # -2 -> -2
        passing_40=1.0,           # 1    -> 1
        pass_td_40p=1.0,          # 1    -> 1
        rushing_yards=30.0,       # .1   -> 3
        rushing_tds=1.0,          # 6    -> 6
        fumbles_total=1.0,        # -1   -> -1
        fumbles_lost_total=1.0,   # -2   -> -2
    )
    assert score_players(stats, rules)["points"].item() == pytest.approx(33.25)


def test_hand_computed_receiver_week():
    rules = parse_settings(LEAGUE_SETTINGS)
    stats = _row(
        PLAYER_RULES,
        position="WR",
        receptions=8.0,        # 1   -> 8
        receiving_yards=124.0,  # .1  -> 12.4
        receiving_tds=1.0,     # 6   -> 6
        receiving_2pt_conversions=1.0,  # 2 -> 2
    )
    assert score_players(stats, rules)["points"].item() == pytest.approx(28.4)


def test_blocked_kicks_count_as_misses():
    """Sleeper charges a blocked FG/PAT to the kicker; nflverse tracks blocks separately."""
    rules = parse_settings(LEAGUE_SETTINGS)
    stats = _row(
        PLAYER_RULES,
        position="K",
        fg_made_40_49=1.0,   # 4
        fg_made_50_59=1.0,   # 5
        fg_missed=1.0,       # -1
        fg_blocked=1.0,      # -1
        pat_made=3.0,        # 3
        pat_blocked=1.0,     # -1
    )
    assert score_players(stats, rules)["points"].item() == pytest.approx(9.0)


# --- 2.2 team defense -------------------------------------------------------------


def _defense(**stats) -> pl.DataFrame:
    base = {name: 0.0 for name in _columns(DEFENSE_RULES)}
    base.update({"season": 2025, "week": 1, "team": "DEN", "pts_allow": 0.0, "yds_allow": 0.0})
    base.update(stats)
    return pl.DataFrame([base])


def test_hand_computed_defense_week():
    rules = parse_settings(LEAGUE_SETTINGS)
    stats = _defense(
        sack=3.0,        # 1 -> 3
        int=2.0,         # 2 -> 4
        ff=1.0,          # 1 -> 1
        fum_rec=1.0,     # 2 -> 2
        st_fum_rec=1.0,  # 1 -> 1
        def_td=1.0,      # 6 -> 6
        blk_kick=1.0,    # 2 -> 2
        safe=1.0,        # 2 -> 2
        pts_allow=10.0,  # pts_allow_7_13  -> 4
        yds_allow=275.0,  # yds_allow_200_299 -> 2
    )
    assert score_defenses(stats, rules)["points"].item() == pytest.approx(27.0)


def test_special_teams_fumble_recovery_pays_less_than_a_defensive_one():
    rules = parse_settings(LEAGUE_SETTINGS)
    on_defense = score_defenses(_defense(fum_rec=1.0, pts_allow=24.0, yds_allow=320.0), rules)
    on_st = score_defenses(_defense(st_fum_rec=1.0, pts_allow=24.0, yds_allow=320.0), rules)
    assert on_defense["points"].item() == pytest.approx(2.0)
    assert on_st["points"].item() == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("points_allowed", "expected"),
    [(0, 10.0), (6, 7.0), (7, 4.0), (13, 4.0), (14, 1.0), (21, 0.0), (28, -1.0), (48, -4.0)],
)
def test_points_allowed_tier_boundaries(points_allowed, expected):
    rules = parse_settings(LEAGUE_SETTINGS)
    stats = _defense(pts_allow=float(points_allowed), yds_allow=320.0)  # yards tier is 0
    assert score_defenses(stats, rules)["points"].item() == pytest.approx(expected)


@pytest.mark.parametrize(
    ("yards_allowed", "expected"),
    [(75, 5.0), (150, 3.0), (275, 2.0), (320, 0.0), (375, -1.0), (425, -3.0), (700, -7.0)],
)
def test_yards_allowed_tier_boundaries(yards_allowed, expected):
    rules = parse_settings(LEAGUE_SETTINGS)
    stats = _defense(pts_allow=24.0, yds_allow=float(yards_allowed))  # points tier is 0
    assert score_defenses(stats, rules)["points"].item() == pytest.approx(expected)


def test_defense_touchdown_is_paid_once():
    """def_td already pays the score; fum_rec_td must not stack on top of it."""
    rules = parse_settings(LEAGUE_SETTINGS)
    stats = _defense(def_td=1.0, fum_rec_td=1.0, pts_allow=24.0, yds_allow=320.0)
    assert score_defenses(stats, rules)["points"].item() == pytest.approx(6.0)
