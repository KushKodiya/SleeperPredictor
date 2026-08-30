"""M15 tests — the training frame, the simplex constraint, and the quantile heads.

All synthetic: no test may hit the network (PRD §10). The frames are small but shaped
like the real ones, which is enough to prove the properties the spec actually requires —
that a team's shares sum to one, that the parts multiply back to the projection, and
that the leakage guard bites.
"""

import numpy as np
import polars as pl
import pytest

from ffdraft.backtest.harness import LeakageError, assert_point_in_time
from ffdraft.models import projections as P
from ffdraft.models.training import training_frame, training_rows

TEAMS = ("DET", "GB")
POSITIONS = ("QB", "RB", "WR", "TE")


def _scored_weeks(seasons=(2022, 2023, 2024), *, seed=0):
    """Weekly scored lines for two full teams over several seasons."""
    rng = np.random.default_rng(seed)
    rows = []
    for season in seasons:
        for team in TEAMS:
            for slot, position in enumerate(POSITIONS * 3):
                pid = f"{team}-{position}-{slot}"
                for week in range(1, 18):
                    passing = position == "QB"
                    rows.append(
                        {
                            "season": season,
                            "week": week,
                            "player_id": pid,
                            "team": team,
                            "position": position,
                            "points": float(rng.uniform(2, 25)),
                            "attempts": int(rng.integers(25, 40)) if passing else 0,
                            "targets": 0 if passing else int(rng.integers(0, 12)),
                            "carries": int(rng.integers(0, 20)) if position in ("QB", "RB") else 0,
                        }
                    )
    return pl.DataFrame(rows)


def _player_stats(scored):
    return scored.select(
        "season", "week", "player_id", "team", "position", "attempts", "targets", "carries"
    )


def _rosters(scored, seasons):
    players = scored.select("player_id", "team").unique()
    return pl.DataFrame(
        [
            {
                "season": season,
                "team": row["team"],
                "gsis_id": row["player_id"],
                "position": "WR",
                "status": "ACT",
            }
            for season in seasons
            for row in players.iter_rows(named=True)
        ]
    )


def _schedules(seasons):
    return pl.DataFrame(
        [
            {
                "season": season,
                "week": week,
                "game_type": "REG",
                "home_team": "DET",
                "away_team": "GB",
                "spread_line": 2.0,
                "total_line": 45.0,
                "home_coach": "Campbell",
                "away_coach": "LaFleur",
            }
            for season in seasons
            for week in range(1, 18)
        ]
    )


def _inputs(seasons=(2021, 2022, 2023, 2024)):
    scored = _scored_weeks(seasons)
    return scored, _player_stats(scored), _rosters(scored, seasons), _schedules(seasons)


# --- 3.1 the point-in-time training frame --------------------------------------------


def test_the_training_frame_carries_the_projected_season_and_passes_the_guard():
    scored, stats, rosters, schedules = _inputs()
    frame = training_frame(scored, stats, rosters, schedules, seasons=[2022, 2023, 2024])

    assert frame.height > 0
    assert sorted(frame["season"].unique().to_list()) == [2022, 2023, 2024]
    # Every feature came from season - 1, so this is a legal training set for 2025.
    assert_point_in_time(frame, target_season=2025, source="M15 training frame")


def test_a_target_season_row_trips_the_leakage_guard():
    """The guard must bite on the frame itself, not only on its inputs."""
    scored, stats, rosters, schedules = _inputs()
    frame = training_frame(scored, stats, rosters, schedules, seasons=[2022, 2023, 2024])

    with pytest.raises(LeakageError, match="would leak into the 2024 draft"):
        assert_point_in_time(frame, target_season=2024, source="M15 training frame")


def test_features_come_from_the_prior_season_not_the_projected_one():
    scored, stats, rosters, schedules = _inputs()
    rows = training_rows(scored, stats, rosters, schedules, season=2024)
    prior = training_rows(scored, stats, rosters, schedules, season=2023)
    # 2024's features are 2023's outcomes, so they cannot equal 2024's own outcomes.
    assert rows["prior_points_per_game"].to_list() != rows["actual_points"].to_list()
    assert rows.height == prior.height


def test_rows_without_a_realized_outcome_are_dropped_not_imputed():
    scored, stats, rosters, schedules = _inputs()
    frame = training_frame(scored, stats, rosters, schedules, seasons=[2022, 2023])
    assert frame["actual_points"].null_count() == 0


# --- 3.2 / 3.3 / 3.4 the model -------------------------------------------------------


def _fitted():
    scored, stats, rosters, schedules = _inputs()
    train = training_frame(scored, stats, rosters, schedules, seasons=[2022, 2023])
    model = P.fit(train, seed=42)
    project_frame = training_frame(scored, stats, rosters, schedules, seasons=[2024])
    volumes = {t: 600.0 for t in TEAMS}
    games = {pid: 17.0 for pid in project_frame["gsis_id"]}
    out = P.project(
        model,
        project_frame,
        expected_games=games,
        team_attempts=volumes,
        team_targets=volumes,
        team_carries=volumes,
    )
    return model, out


def test_target_shares_sum_to_one_within_each_team():
    _, out = _fitted()
    sums = out.group_by("team").agg(pl.col("target_share").sum().alias("total"))
    for total in sums["total"]:
        assert total == pytest.approx(1.0, abs=0.001)


def test_carries_reconcile_to_the_projected_team_volume():
    _, out = _fitted()
    sums = out.group_by("team").agg(pl.col("projected_carries").sum().alias("total"))
    for total in sums["total"]:
        assert total == pytest.approx(600.0, abs=0.001)


def test_attempt_share_is_constrained_too():
    _, out = _fitted()
    sums = out.group_by("team").agg(pl.col("attempt_share").sum().alias("total"))
    for total in sums["total"]:
        assert total == pytest.approx(1.0, abs=0.001)


def test_the_three_parts_are_inspectable_and_multiply_back():
    """A season total that cannot be decomposed hides which part is wrong."""
    _, out = _fitted()
    for column in ("games", "opportunity_per_game", "points_per_opportunity"):
        assert column in out.columns
    product = out["games"] * out["opportunity_per_game"] * out["points_per_opportunity"]
    for expected, actual in zip(product, out["projected_points"], strict=True):
        assert actual == pytest.approx(expected)


def test_games_come_from_the_caller_not_from_a_model_here():
    """M16 owns durability; this module must not silently re-model it."""
    scored, stats, rosters, schedules = _inputs()
    train = training_frame(scored, stats, rosters, schedules, seasons=[2022, 2023])
    model = P.fit(train, seed=42)
    frame = training_frame(scored, stats, rosters, schedules, seasons=[2024])
    volumes = {t: 600.0 for t in TEAMS}

    halved = P.project(
        model, frame,
        expected_games={pid: 8.5 for pid in frame["gsis_id"]},
        team_attempts=volumes, team_targets=volumes, team_carries=volumes,
    )
    assert halved["games"].unique().to_list() == [8.5]


def test_quantiles_are_ordered():
    _, out = _fitted()
    assert ((out["p10"] <= out["p50"]) & (out["p50"] <= out["p90"])).all()


def test_the_fit_is_deterministic_under_a_seed():
    """R7: two fits with the same seed must give identical projections."""
    first = _fitted()[1]
    second = _fitted()[1]
    assert first["projected_points"].to_list() == second["projected_points"].to_list()
    assert first["p50"].to_list() == second["p50"].to_list()


def test_only_startable_positions_are_modelled():
    """Punters appear in the stat feed; nobody drafts them."""
    _, out = _fitted()
    assert set(out["position"].unique().to_list()) <= set(P.MODELLED_POSITIONS)


def test_fitting_with_no_usable_rows_raises_rather_than_returning_a_null_model():
    scored, stats, rosters, schedules = _inputs()
    train = training_frame(scored, stats, rosters, schedules, seasons=[2022, 2023])
    with pytest.raises(ValueError, match="no training rows"):
        P.fit(train, seed=42, positions=("K",))
