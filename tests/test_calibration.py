"""M7 tests — monotonicity, the leakage guard, and the shrinkage sanity check."""

import numpy as np
import polars as pl
import pytest

from ffdraft.valuation.calibration import (
    actual_season_points,
    calibrated_points,
    fit_calibration,
)

FIT_POOL = {"QB": 20, "RB": 40}


def _training(seasons=(2022, 2023, 2024), *, position="RB", depth=40, noise=0.0) -> pl.DataFrame:
    """Ranks 1..depth whose actual points fall off with rank, plus optional noise."""
    rng = np.random.default_rng(17)  # R7: explicit generator, never global randomness
    rows = []
    for season in seasons:
        for rank in range(1, depth + 1):
            actual = 320.0 - 6.0 * rank + (rng.normal(0.0, noise) if noise else 0.0)
            rows.append(
                {"season": season, "gsis_id": f"{position}{rank}-{season}",
                 "position": position, "ecr": float(rank), "actual_points": actual}
            )
    return pl.DataFrame(rows)


def test_mapping_is_monotone_in_rank():
    """A worse rank can never be worth more points than a better one."""
    cal = fit_calibration(_training(noise=45.0), target_season=2025, fit_pool=FIT_POOL)
    fitted = cal.predict("RB", np.arange(1, 41, dtype=float))
    assert np.all(np.diff(fitted) <= 1e-9), fitted


def test_fit_rejects_a_training_row_from_the_target_season():
    training = _training(seasons=(2023, 2024, 2025))
    with pytest.raises(ValueError, match="would leak"):
        fit_calibration(training, target_season=2025, fit_pool=FIT_POOL)


def test_fit_rejects_a_training_row_from_after_the_target_season():
    training = _training(seasons=(2023, 2026))
    with pytest.raises(ValueError, match=r"2026"):
        fit_calibration(training, target_season=2025, fit_pool=FIT_POOL)


def test_fit_accepts_strictly_prior_seasons():
    cal = fit_calibration(_training(seasons=(2022, 2023, 2024)), target_season=2025,
                          fit_pool=FIT_POOL)
    assert cal.training_seasons == (2022, 2023, 2024)
    assert max(cal.training_seasons) < cal.target_season


def test_too_few_training_seasons_raises():
    with pytest.raises(ValueError, match="at least 6 prior seasons"):
        fit_calibration(_training(seasons=(2023, 2024)), target_season=2025,
                        fit_pool=FIT_POOL, min_training_seasons=6)


def test_fit_compresses_the_outcome_spread():
    """The rank->points analogue of the PRD's slope < 1.0 check.

    A slope below 1.0 says projections are too spread out. Fitting on realised outcomes
    reproduces that: the fitted curve must be narrower than the outcomes it saw, not
    chase the best season each rank ever produced.
    """
    cal = fit_calibration(_training(noise=60.0), target_season=2025, fit_pool=FIT_POOL)
    assert cal.shrinkage_ratio("RB") < 1.0


def test_fit_pool_caps_training_depth():
    """Ranks beyond the configured pool must not enter the fit."""
    deep = _training(depth=80)
    cal = fit_calibration(deep, target_season=2025, fit_pool={"RB": 40})
    # rank 80 is outside the pool, so it clips to the rank-40 fitted value
    assert cal.predict("RB", [80.0])[0] == pytest.approx(cal.predict("RB", [40.0])[0])


def test_position_without_a_configured_cap_trains_on_everything():
    cal = fit_calibration(_training(position="K", depth=30), target_season=2025, fit_pool={})
    assert "K" in cal.positions()
    assert cal.predict("K", [30.0])[0] < cal.predict("K", [1.0])[0]


def test_predict_on_an_unfitted_position_raises():
    cal = fit_calibration(_training(), target_season=2025, fit_pool=FIT_POOL)
    with pytest.raises(KeyError, match="WR"):
        cal.predict("WR", [1.0])


def test_calibrated_points_attaches_a_projection_per_player():
    cal = fit_calibration(_training(), target_season=2025, fit_pool=FIT_POOL)
    ranked = pl.DataFrame(
        [{"gsis_id": "a", "position": "RB", "ecr": 1.0},
         {"gsis_id": "b", "position": "RB", "ecr": 25.0}]
    )
    out = calibrated_points(ranked, cal)
    assert out.height == 2
    best, worse = out["projected_points"].to_list()
    assert best > worse  # sorted descending, and rank 1 outscores rank 25


def test_actual_season_points_sums_the_scored_weeks():
    weeks = pl.DataFrame(
        [{"season": 2024, "player_id": "a", "week": w, "points": 10.0} for w in range(1, 18)]
        + [{"season": 2024, "player_id": "b", "week": 1, "points": 3.5}]
    )
    out = {r["gsis_id"]: r["actual_points"] for r in actual_season_points(weeks).iter_rows(named=True)}
    assert out["a"] == pytest.approx(170.0)
    assert out["b"] == pytest.approx(3.5)
