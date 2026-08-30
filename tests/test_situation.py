"""M17 tests — vacated opportunity and the four edge cases that silently corrupt it.

Every fixture here is hand-built, so each expected share is a number you can check by
eye. That matters more than usual: vacated opportunity is the kind of derivation that
looks plausible while being wrong, and a plausible-looking wrong share would flow
straight into the projection model as if it were evidence.
"""

import polars as pl
import pytest

from ffdraft.features import situation
from ffdraft.features.situation import (
    PRESENT_STATUSES,
    arrivals,
    opportunity_shares,
    vacated_opportunity,
)


def _stats(rows):
    """Weekly player stats. `rows` is (player_id, team, targets, carries)."""
    return pl.DataFrame(
        [
            {
                "season": 2024,
                "week": 1,
                "season_type": "REG",
                "player_id": pid,
                "position": "WR",
                "team": team,
                "targets": targets,
                "carries": carries,
            }
            for pid, team, targets, carries in rows
        ]
    )


def _rosters(rows, *, season=2025):
    """Season rosters. `rows` is (player_id, team, status).

    The schema is declared so that an empty roster still carries its columns, the way a
    real nflverse frame does — otherwise the empty case would fail its contract check for
    a reason the fixture invented.
    """
    return pl.DataFrame(
        [
            {
                "season": season,
                "team": team,
                "gsis_id": pid,
                "position": "WR",
                "status": status,
            }
            for pid, team, status in rows
        ],
        schema={
            "season": pl.Int64,
            "team": pl.String,
            "gsis_id": pl.String,
            "position": pl.String,
            "status": pl.String,
        },
    )


# --- 2.1 vacated opportunity ---------------------------------------------------------


def test_a_known_departure_vacates_exactly_its_own_share():
    """One departure of a known share must produce that share, not something near it."""
    stats = _stats([("a", "DET", 75, 0), ("b", "DET", 25, 0)])
    rosters = _rosters([("b", "DET", "ACT")])  # a is gone

    row = vacated_opportunity(stats, rosters, prior_season=2024).row(0, named=True)
    assert row["team"] == "DET"
    assert row["vacated_target_share"] == pytest.approx(0.75)


def test_a_team_that_lost_nobody_vacates_exactly_zero_not_null():
    """Null would reach the model as a missing feature rather than as 'nothing left'."""
    stats = _stats([("a", "DET", 75, 10), ("b", "DET", 25, 30)])
    rosters = _rosters([("a", "DET", "ACT"), ("b", "DET", "ACT")])

    row = vacated_opportunity(stats, rosters, prior_season=2024).row(0, named=True)
    assert row["vacated_target_share"] == 0.0
    assert row["vacated_carry_share"] == 0.0
    assert row["vacated_target_share"] is not None


def test_shares_stay_within_zero_and_one():
    stats = _stats(
        [("a", "DET", 40, 5), ("b", "DET", 60, 15), ("c", "GB", 10, 0), ("d", "GB", 90, 100)]
    )
    rosters = _rosters([("b", "DET", "ACT")])  # everyone else gone

    frame = vacated_opportunity(stats, rosters, prior_season=2024)
    for column in ("vacated_target_share", "vacated_carry_share"):
        assert frame[column].min() >= 0.0
        assert frame[column].max() <= 1.0


def test_a_team_with_no_carries_gets_a_zero_share_not_a_division_by_zero():
    stats = _stats([("a", "DET", 100, 0), ("b", "DET", 50, 0)])
    shares = opportunity_shares(stats, season=2024)
    assert shares["carry_share"].to_list() == [0.0, 0.0]
    assert shares["target_share"].sum() == pytest.approx(1.0)


def test_every_team_with_prior_opportunity_appears():
    stats = _stats([("a", "DET", 10, 0), ("b", "GB", 10, 0), ("c", "CHI", 10, 0)])
    frame = vacated_opportunity(stats, _rosters([]), prior_season=2024)
    assert sorted(frame["team"].to_list()) == ["CHI", "DET", "GB"]


# --- 2.2 the four edge cases ---------------------------------------------------------


def test_a_season_long_ir_player_is_present_not_departed():
    """IR still occupies the role; counting it as vacated invents opportunity."""
    assert "RES" in PRESENT_STATUSES
    stats = _stats([("a", "DET", 80, 0), ("b", "DET", 20, 0)])
    rosters = _rosters([("a", "DET", "RES"), ("b", "DET", "ACT")])

    row = vacated_opportunity(stats, rosters, prior_season=2024).row(0, named=True)
    assert row["vacated_target_share"] == 0.0


def test_a_team_change_vacates_from_the_old_team_and_arrives_at_the_new():
    stats = _stats([("mover", "DET", 60, 0), ("stay", "DET", 40, 0), ("gb1", "GB", 100, 0)])
    rosters = _rosters(
        [("mover", "GB", "ACT"), ("stay", "DET", "ACT"), ("gb1", "GB", "ACT")]
    )

    vacated = vacated_opportunity(stats, rosters, prior_season=2024)
    by_team = dict(zip(vacated["team"], vacated["vacated_target_share"], strict=True))
    assert by_team["DET"] == pytest.approx(0.60)  # departed from the old team
    assert by_team["GB"] == 0.0

    incoming = arrivals(stats, rosters, prior_season=2024)
    assert ("GB", "mover") in list(zip(incoming["team"], incoming["player_id"], strict=True))


def test_a_rookie_vacates_nothing_but_counts_as_arriving_competition():
    stats = _stats([("vet", "DET", 100, 0)])
    rosters = _rosters([("vet", "DET", "ACT"), ("rookie", "DET", "ACT")])

    row = vacated_opportunity(stats, rosters, prior_season=2024).row(0, named=True)
    assert row["vacated_target_share"] == 0.0

    incoming = arrivals(stats, rosters, prior_season=2024)
    assert "rookie" in incoming["player_id"].to_list()


def test_a_practice_squad_player_does_not_hold_his_prior_opportunity():
    """`DEV` is the practice squad — not a roster spot, so his prior share is vacated."""
    assert "DEV" not in PRESENT_STATUSES
    stats = _stats([("a", "DET", 30, 0), ("b", "DET", 70, 0)])
    rosters = _rosters([("a", "DET", "DEV"), ("b", "DET", "ACT")])

    row = vacated_opportunity(stats, rosters, prior_season=2024).row(0, named=True)
    assert row["vacated_target_share"] == pytest.approx(0.30)


def test_a_cut_player_is_a_departure():
    stats = _stats([("a", "DET", 30, 0), ("b", "DET", 70, 0)])
    rosters = _rosters([("a", "DET", "CUT"), ("b", "DET", "ACT")])
    row = vacated_opportunity(stats, rosters, prior_season=2024).row(0, named=True)
    assert row["vacated_target_share"] == pytest.approx(0.30)


def test_departures_are_matched_on_id_never_on_name():
    """R4: two different players sharing a display name must not collapse into one."""
    stats = _stats([("00-0001", "DET", 50, 0), ("00-0002", "DET", 50, 0)])
    rosters = _rosters([("00-0002", "DET", "ACT")])
    row = vacated_opportunity(stats, rosters, prior_season=2024).row(0, named=True)
    assert row["vacated_target_share"] == pytest.approx(0.50)


def test_a_missing_column_fails_loudly():
    with pytest.raises(ValueError, match="targets"):
        opportunity_shares(
            pl.DataFrame([{"season": 2024, "week": 1, "player_id": "a", "team": "DET"}]),
            season=2024,
        )


# --- 2.3 team context and player capital ---------------------------------------------


def _schedules(rows, *, season=2025):
    """Games. `rows` is (home, away, spread_line, total_line, home_coach, away_coach)."""
    return pl.DataFrame(
        [
            {
                "season": season,
                "week": i + 1,
                "game_type": "REG",
                "home_team": home,
                "away_team": away,
                "spread_line": spread,
                "total_line": total,
                "home_coach": hc,
                "away_coach": ac,
            }
            for i, (home, away, spread, total, hc, ac) in enumerate(rows)
        ],
        schema={
            "season": pl.Int64, "week": pl.Int64, "game_type": pl.String,
            "home_team": pl.String, "away_team": pl.String,
            "spread_line": pl.Float64, "total_line": pl.Float64,
            "home_coach": pl.String, "away_coach": pl.String,
        },
    )


def test_implied_total_splits_the_line_by_the_spread():
    """Home favoured by 3 in a 47-point game: 25 and 22, per the PRD §6.1 formula."""
    frame = situation.team_implied_total(
        _schedules([("DET", "GB", 3.0, 47.0, "Campbell", "LaFleur")]), season=2025
    )
    by_team = dict(zip(frame["team"], frame["team_implied_total"], strict=True))
    assert by_team["DET"] == pytest.approx(25.0)
    assert by_team["GB"] == pytest.approx(22.0)


def test_an_unposted_line_falls_back_and_is_flagged_never_zero():
    """§11.11: imputing zero would silently claim a team scores nothing."""
    prior = _schedules([("DET", "GB", 0.0, 44.0, "Campbell", "LaFleur")], season=2024)
    current = _schedules([("DET", "GB", None, None, "Campbell", "LaFleur")], season=2025)

    frame = situation.team_implied_total(
        pl.concat([prior, current]), season=2025, fallback_season=2024
    )
    assert frame["implied_total_is_fallback"].all()
    assert frame["lines_posted"].to_list() == [0, 0]
    assert frame["team_implied_total"].to_list() == [pytest.approx(22.0)] * 2
    assert 0.0 not in frame["team_implied_total"].to_list()


def test_a_coaching_change_is_flagged_and_a_stay_is_not():
    prior = _schedules([("DET", "GB", 0.0, 44.0, "Campbell", "Old")], season=2024)
    current = _schedules([("DET", "GB", 0.0, 44.0, "Campbell", "New")], season=2025)

    frame = situation.head_coach_change(pl.concat([prior, current]), season=2025)
    by_team = dict(zip(frame["team"], frame["head_coach_change"], strict=True))
    assert by_team["GB"] is True
    assert by_team["DET"] is False


def test_a_team_with_no_prior_season_is_unknown_not_unchanged():
    """Null says "we cannot tell"; False would claim the coach stayed."""
    frame = situation.head_coach_change(
        _schedules([("DET", "GB", 0.0, 44.0, "Campbell", "LaFleur")], season=2025),
        season=2025,
    )
    assert frame["head_coach_change"].null_count() == 2


def test_guaranteed_money_takes_the_most_recent_contract():
    contracts = pl.DataFrame(
        [
            {"gsis_id": "a", "guaranteed": 10.0, "year_signed": 2021, "years": 3,
             "is_active": False},
            {"gsis_id": "a", "guaranteed": 50.0, "year_signed": 2024, "years": 4,
             "is_active": True},
        ]
    )
    row = situation.guaranteed_money(contracts).row(0, named=True)
    assert row["guaranteed_money"] == 50.0
    assert row["year_signed"] == 2024


def test_draft_capital_reads_round_and_pick_not_the_prd_names():
    picks = pl.DataFrame(
        [{"season": 2023, "team": "DET", "gsis_id": "a", "round": 2, "pick": 34}]
    )
    row = situation.draft_capital(picks).row(0, named=True)
    assert (row["draft_round"], row["draft_pick"]) == (2, 34)
