"""Tests for the raw stat assembly the scoring engine consumes.

These cover the derivations that are not visible in any single nflverse column and
that the 2025 reproduction pinned down: who owns a return touchdown, which fumble
credits are special-teams ones, and what actually counts as points allowed.
"""

import polars as pl
import pytest

from ffdraft.scoring.statlines import defense_week_stats, long_td_counts

PBP_SCHEMA = {
    "game_id": pl.String,
    "season": pl.Int64,
    "week": pl.Int64,
    "season_type": pl.String,
    "posteam": pl.String,
    "defteam": pl.String,
    "play_type": pl.String,
    "special": pl.Float64,
    "touchdown": pl.Float64,
    "td_team": pl.String,
    "pass_touchdown": pl.Float64,
    "rush_touchdown": pl.Float64,
    "yards_gained": pl.Float64,
    "passer_player_id": pl.String,
    "rusher_player_id": pl.String,
    "receiver_player_id": pl.String,
    "sack": pl.Float64,
    "interception": pl.Float64,
    "safety": pl.Float64,
    "punt_blocked": pl.Float64,
    "field_goal_result": pl.String,
    "extra_point_result": pl.String,
    "fumble": pl.Float64,
    "aborted_play": pl.Float64,
    "forced_fumble_player_1_team": pl.String,
    "forced_fumble_player_2_team": pl.String,
    "fumble_recovery_1_team": pl.String,
    "fumble_recovery_2_team": pl.String,
    "fumbled_1_team": pl.String,
    "fumbled_2_team": pl.String,
}
_NUMERIC_DEFAULT = {name: 0.0 for name, dt in PBP_SCHEMA.items() if dt is pl.Float64}


def _play(**kw) -> dict:
    row = {name: None for name in PBP_SCHEMA}
    row.update(_NUMERIC_DEFAULT)
    row.update(
        {"game_id": "2025_01_KC_DEN", "season": 2025, "week": 1, "season_type": "REG",
         "play_type": "pass", "posteam": "KC", "defteam": "DEN"}
    )
    row.update(kw)
    return row


def _pbp(*plays: dict) -> pl.DataFrame:
    return pl.DataFrame(list(plays), schema=PBP_SCHEMA)


def _team_stats(**yards) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {"team": team, "season": 2025, "week": 1, "season_type": "REG",
             "passing_yards": y, "rushing_yards": 0, "sack_yards_lost": 0}
            for team, y in yards.items()
        ]
    )


def _schedules(home_score: int, away_score: int) -> pl.DataFrame:
    return pl.DataFrame(
        [{"season": 2025, "week": 1, "home_team": "DEN", "away_team": "KC",
          "home_score": home_score, "away_score": away_score}]
    )


def _den(pbp: pl.DataFrame, *, kc_yards: int = 250, kc_points: int = 20) -> dict:
    frame = defense_week_stats(pbp, _team_stats(DEN=300, KC=kc_yards), _schedules(24, kc_points))
    return frame.filter(pl.col("team") == "DEN").to_dicts()[0]


def test_return_touchdowns_belong_to_the_returning_teams_defense():
    """On kickoffs `posteam` is the receiving team, so possession cannot classify the score."""
    row = _den(
        _pbp(
            _play(play_type="kickoff", special=1.0, posteam="DEN", defteam="KC",
                  touchdown=1.0, td_team="DEN"),
            _play(interception=1.0, touchdown=1.0, td_team="DEN"),
            _play(posteam="DEN", defteam="KC", touchdown=1.0, td_team="DEN", play_type="run"),
        )
    )
    assert row["st_td"] == 1  # kick return
    assert row["def_td"] == 1  # pick six
    # DEN's own offensive touchdown is not a defensive score.


def test_points_allowed_drops_the_opponents_defensive_score_but_not_its_return_score():
    pick_six_against_den = _play(
        posteam="DEN", defteam="KC", interception=1.0, touchdown=1.0, td_team="KC"
    )
    kick_return_against_den = _play(
        play_type="kickoff", special=1.0, posteam="KC", defteam="DEN", touchdown=1.0, td_team="KC"
    )
    assert _den(_pbp(pick_six_against_den), kc_points=20)["pts_allow"] == 14
    assert _den(_pbp(kick_return_against_den), kc_points=20)["pts_allow"] == 20


def test_yards_allowed_is_the_opponents_net_yardage():
    pbp = _pbp(_play(yards_gained=60.0))
    frame = defense_week_stats(
        pbp,
        pl.DataFrame(
            [
                {"team": "KC", "season": 2025, "week": 1, "season_type": "REG",
                 "passing_yards": 300, "rushing_yards": 120, "sack_yards_lost": -45},
                {"team": "DEN", "season": 2025, "week": 1, "season_type": "REG",
                 "passing_yards": 200, "rushing_yards": 100, "sack_yards_lost": -10},
            ]
        ),
        _schedules(24, 20),
    )
    assert frame.filter(pl.col("team") == "DEN")["yds_allow"].item() == 375


def test_special_teams_fumbles_use_the_st_credits():
    row = _den(
        _pbp(
            _play(play_type="punt", special=1.0, posteam="DEN", defteam="KC", fumble=1.0,
                  forced_fumble_player_1_team="DEN", fumble_recovery_1_team="DEN",
                  fumbled_1_team="KC"),
        )
    )
    assert (row["st_ff"], row["st_fum_rec"]) == (1, 1)
    assert (row["ff"], row["fum_rec"]) == (0, 0)


def test_recovering_your_own_fumble_is_not_a_takeaway():
    row = _den(
        _pbp(
            _play(posteam="DEN", defteam="KC", fumble=1.0, fumble_recovery_1_team="DEN",
                  fumbled_1_team="DEN"),
        )
    )
    assert (row["fum_rec"], row["st_fum_rec"]) == (0, 0)


def test_stripping_a_ball_carrier_forces_a_fumble_but_a_sack_strip_does_not():
    """Sleeper names no forcer on either, yet only pays a forced fumble on the first."""
    stripped = _play(play_type="run", fumble=1.0, fumble_recovery_1_team="DEN", fumbled_1_team="KC")
    sacked = _play(sack=1.0, fumble=1.0, fumble_recovery_1_team="DEN", fumbled_1_team="KC")
    aborted = _play(play_type="run", fumble=1.0, aborted_play=1.0,
                    fumble_recovery_1_team="DEN", fumbled_1_team="KC")

    assert _den(_pbp(stripped))["ff"] == 1
    assert _den(_pbp(sacked))["ff"] == 0
    assert _den(_pbp(aborted))["ff"] == 0
    for play in (stripped, sacked, aborted):
        assert _den(_pbp(play))["fum_rec"] == 1


def test_long_td_counts_credit_passer_receiver_and_rusher():
    counts = long_td_counts(
        _pbp(
            _play(pass_touchdown=1.0, touchdown=1.0, yards_gained=55.0,
                  passer_player_id="QB1", receiver_player_id="WR1"),
            _play(rush_touchdown=1.0, touchdown=1.0, yards_gained=41.0, rusher_player_id="RB1"),
            _play(rush_touchdown=1.0, touchdown=1.0, yards_gained=39.0, rusher_player_id="RB1"),
        )
    )
    by_player = {r["player_id"]: r for r in counts.iter_rows(named=True)}
    assert by_player["QB1"]["pass_td_40p"] == 1
    assert by_player["WR1"]["rec_td_40p"] == 1
    assert by_player["RB1"]["rush_td_40p"] == 1  # the 39-yard score does not qualify


def test_missing_contract_column_fails_loudly():
    with pytest.raises(ValueError, match="aborted_play"):
        defense_week_stats(
            _pbp(_play()).drop("aborted_play"), _team_stats(DEN=300, KC=250), _schedules(24, 20)
        )
