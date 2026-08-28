"""M16 tests — the games-played distribution, its pooling, and its conditioning.

Every draw goes through a seeded `np.random.Generator` (R7), so a failure here is
reproducible rather than a coin flip.
"""

import numpy as np
import polars as pl
import pytest

from ffdraft.sim.availability import (
    PlayerSeason,
    availability_history,
    build_availability,
)

GAMES = 17


def _history(rows) -> pl.DataFrame:
    return pl.DataFrame(
        [{"season": s, "gsis_id": g, "position": p, "games_played": gp, "age": a,
          "workload_percentile": w} for s, g, p, gp, a, w in rows],
        schema={"season": pl.Int64, "gsis_id": pl.String, "position": pl.String,
                "games_played": pl.Int64, "age": pl.Float64, "workload_percentile": pl.Float64},
    )


def _cell(position, age, workload, games, n, season=2024):
    """`n` player-seasons in one cell, cycling through `games`."""
    return [(season, f"{position}{age}{workload}{i}", position, games[i % len(games)],
             float(age), workload) for i in range(n)]


def _model(rows, *, min_bin_count=30):
    return build_availability(
        _history(rows), games_per_season=GAMES, age_bin_width=1,
        min_bin_count=min_bin_count, workload_percentile=0.80,
    )


# --- 1.1 proper empirical distributions -------------------------------------------


def test_every_cell_distribution_sums_to_one():
    model = _model(_cell("RB", 24, 0.5, [10, 12, 14, 17], 40))
    for pmf in (*model.cells.values(), *model.by_position.values(), model.overall):
        assert pmf.sum() == pytest.approx(1.0)
        assert (pmf >= 0).all()


def test_sampled_mean_recovers_the_historical_mean():
    games = [8, 10, 12, 14, 16, 17]
    model = _model(_cell("RB", 24, 0.5, games, 60))
    rng = np.random.default_rng(11)
    draws = model.games_played_distribution(PlayerSeason("RB", 24, 0.5), rng, n_sims=40_000)
    assert draws.mean() == pytest.approx(float(np.mean(games)), abs=0.1)


def test_expected_games_matches_the_pmf():
    model = _model(_cell("RB", 24, 0.5, [10, 14], 40))
    assert model.expected_games(PlayerSeason("RB", 24, 0.5)) == pytest.approx(12.0)


# --- 1.2 thin cells pool up --------------------------------------------------------


def test_a_thin_cell_pools_instead_of_degenerating():
    """Four observations would make a spike; the position-wide shape is the honest answer."""
    rows = _cell("RB", 24, 0.5, [9, 11, 13, 15, 17], 60) + _cell("RB", 33, 0.5, [17], 4)
    model = _model(rows)

    thin = model.pmf(PlayerSeason("RB", 33, 0.5))
    assert ("RB", 33, False) not in model.cells      # never built
    assert model.counts[("RB", 33, False)] == 4      # but its thinness is recorded
    assert np.count_nonzero(thin) > 1                # pooled, not a spike at 17
    assert thin is model.by_position_tier[("RB", False)]


def test_a_dense_cell_uses_its_own_distribution():
    rows = _cell("RB", 24, 0.5, [17], 40) + _cell("RB", 30, 0.5, [8], 40)
    model = _model(rows)
    assert model.expected_games(PlayerSeason("RB", 24, 0.5)) == pytest.approx(17.0)
    assert model.expected_games(PlayerSeason("RB", 30, 0.5)) == pytest.approx(8.0)


def test_unknown_age_pools_to_the_position():
    model = _model(_cell("RB", 24, 0.5, [10, 12], 40))
    assert model.pmf(PlayerSeason("RB", None, 0.5)) is model.by_position_tier[("RB", False)]


def test_an_unseen_position_falls_all_the_way_back():
    model = _model(_cell("RB", 24, 0.5, [10, 12], 40))
    assert model.pmf(PlayerSeason("LS", 24, 0.5)) is model.overall


def test_building_from_no_history_raises():
    with pytest.raises(ValueError, match="empty"):
        build_availability(_history([]), games_per_season=GAMES, age_bin_width=1,
                           min_bin_count=30, workload_percentile=0.8)


# --- 1.3 conditioning on age and workload ------------------------------------------


def test_older_players_have_fewer_expected_games_than_younger_ones():
    """Age behaves as the PRD expects once the population is rosterable players."""
    rows = _cell("RB", 24, 0.5, [13, 14, 15], 40) + _cell("RB", 30, 0.5, [9, 10, 11], 40)
    model = _model(rows)
    assert model.expected_games(PlayerSeason("RB", 30, 0.5)) < model.expected_games(
        PlayerSeason("RB", 24, 0.5)
    )


def test_workload_conditioning_follows_the_data_not_an_assumed_direction():
    """High prior workload is a durability marker; the model must reproduce whatever
    direction the history shows, not a sign fixed in advance."""
    durable = _cell("RB", 26, 0.95, [15, 16, 17], 40)   # high workload, played a lot
    fragile = _cell("RB", 26, 0.50, [8, 9, 10], 40)     # median workload, played less
    model = _model(durable + fragile)

    high = model.expected_games(PlayerSeason("RB", 26, 0.95))
    mid = model.expected_games(PlayerSeason("RB", 26, 0.50))
    assert high > mid  # exactly the direction 2015-2025 shows

    # and with the history inverted, the model inverts with it
    flipped = _model(_cell("RB", 26, 0.95, [8, 9, 10], 40) + _cell("RB", 26, 0.50, [15, 16, 17], 40))
    assert flipped.expected_games(PlayerSeason("RB", 26, 0.95)) < flipped.expected_games(
        PlayerSeason("RB", 26, 0.50)
    )


def test_the_workload_threshold_comes_from_config():
    model = _model(_cell("RB", 26, 0.95, [17], 40) + _cell("RB", 26, 0.50, [8], 40))
    assert model.is_high_workload(0.95) and not model.is_high_workload(0.50)
    assert model.is_high_workload(None) is False  # no prior snaps is not high workload


# --- 1.4 the sampling interface ----------------------------------------------------


def test_draws_have_the_requested_shape_and_stay_in_range():
    model = _model(_cell("RB", 24, 0.5, [0, 5, 12, 17], 40))
    draws = model.games_played_distribution(PlayerSeason("RB", 24, 0.5),
                                            np.random.default_rng(3), n_sims=500)
    assert draws.shape == (500,)
    assert draws.dtype.kind in "iu"
    assert draws.min() >= 0 and draws.max() <= GAMES


def test_draws_are_deterministic_under_a_seeded_generator():
    model = _model(_cell("RB", 24, 0.5, [6, 9, 13, 17], 40))
    player = PlayerSeason("RB", 24, 0.5)
    first = model.games_played_distribution(player, np.random.default_rng(7), n_sims=200)
    second = model.games_played_distribution(player, np.random.default_rng(7), n_sims=200)
    assert np.array_equal(first, second)


def test_a_different_seed_gives_a_different_draw():
    model = _model(_cell("RB", 24, 0.5, [6, 9, 13, 17], 40))
    player = PlayerSeason("RB", 24, 0.5)
    a = model.games_played_distribution(player, np.random.default_rng(1), n_sims=200)
    b = model.games_played_distribution(player, np.random.default_rng(2), n_sims=200)
    assert not np.array_equal(a, b)


def test_zero_sims_raises():
    model = _model(_cell("RB", 24, 0.5, [10], 40))
    with pytest.raises(ValueError, match="at least 1"):
        model.games_played_distribution(PlayerSeason("RB", 24, 0.5), np.random.default_rng(0), 0)


# --- history assembly --------------------------------------------------------------


def test_history_keeps_only_rosterable_players_and_shifts_workload_a_season():
    stats = pl.DataFrame(
        [{"season": 2024, "player_id": f"p{i}", "position": "RB", "season_type": "REG",
          "week": w} for i in range(1, 5) for w in range(1, 13)]
    )
    players = pl.DataFrame(
        [{"gsis_id": f"p{i}", "birth_date": "1998-03-01"} for i in range(1, 5)]
    )
    snaps = pl.DataFrame(
        [{"season": 2023, "week": 1, "game_type": "REG", "pfr_player_id": f"P{i}",
          "offense_snaps": i * 100, "position": "RB", "team": "KC"} for i in range(1, 5)]
    )
    ids = pl.DataFrame([{"pfr_id": f"P{i}", "gsis_id": f"p{i}"} for i in range(1, 5)])

    out = availability_history(
        stats, players, snaps, ids, positions={"RB"}, rosterable_percentile=0.5
    )
    assert set(out["gsis_id"]) == {"p2", "p3", "p4"}  # p1 is bottom-quartile workload
    assert out["games_played"].unique().to_list() == [12]
    assert out["age"].unique().to_list() == [26.0]  # born 1998, season starts Sep 2024
    assert out["season"].unique().to_list() == [2024]
