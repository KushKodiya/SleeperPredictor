"""End-to-end board assembly, with the data loaders injected so no test hits the network.

The pipeline order is the thing worth pinning down here: aggregate, calibrate, *then*
override, then value against replacement. Getting that order wrong would silently shrink
a manual number the owner meant to be taken at face value.
"""

from datetime import UTC, datetime

import numpy as np
import polars as pl
import pytest

from ffdraft.config import load_config
from ffdraft.data.overrides import Override
from ffdraft.scoring.engine import ScoringRules
from ffdraft.valuation import board as board_module
from ffdraft.valuation.board import build_board

TRAINING_SEASONS = (2020, 2021, 2022, 2023, 2024, 2025)  # matches min_training_seasons: 6
TARGET = 2026
POSITIONS = {"RB": 40, "WR": 40}


def _rankings() -> pl.DataFrame:
    """Preseason redraft-positional ranks, plus in-season noise that must be ignored."""
    rows = []
    for season in (*TRAINING_SEASONS, TARGET):
        for position, depth in POSITIONS.items():
            for rank in range(1, depth + 1):
                rows.append(
                    {"id": f"{position}{rank}", "player": f"{position} Player {rank}",
                     "pos": position, "team": "KC", "ecr": float(rank), "sd": 2.0,
                     "best": rank, "worst": rank + 3, "scrape_date": f"{season}-08-20",
                     "ecr_type": "rp"}
                )
                rows.append(  # week 3, after kickoff: must never be selected
                    {"id": f"{position}{rank}", "player": f"{position} Player {rank}",
                     "pos": position, "team": "KC", "ecr": float(depth + 1 - rank), "sd": 2.0,
                     "best": 1, "worst": 99, "scrape_date": f"{season}-09-25",
                     "ecr_type": "rp"}
                )
    return pl.DataFrame(rows)


def _schedules() -> pl.DataFrame:
    return pl.DataFrame(
        [{"season": s, "week": 1, "gameday": f"{s}-09-05"} for s in (*TRAINING_SEASONS, TARGET)]
    )


def _ids() -> pl.DataFrame:
    return pl.DataFrame(
        [{"fantasypros_id": f"{p}{r}", "gsis_id": f"gsis-{p}{r}", "name": f"{p} Player {r}",
          "merge_name": f"{p} player {r}", "position": p, "team": "KC", "sleeper_id": None}
         for p, depth in POSITIONS.items() for r in range(1, depth + 1)]
    )


def _actuals(season: int, rules, *, refresh: bool = False) -> pl.DataFrame:
    """Outcomes that fall off with rank, plus the season-to-season noise a real one has.

    The noise matters: a noiseless outcome leaves the fit nothing to shrink, and the
    shrinkage check would pass or fail for the wrong reason.
    """
    rng = np.random.default_rng(season)  # R7: seeded, so the fit is reproducible
    return pl.DataFrame(
        [{"season": season, "gsis_id": f"gsis-{p}{r}",
          "actual_points": 300.0 - 4.0 * r + rng.normal(0.0, 40.0)}
         for p, depth in POSITIONS.items() for r in range(1, depth + 1)]
    )


def _weekly(cfg, rules, season: int, *, refresh: bool = False) -> pl.DataFrame:
    return pl.DataFrame(
        [{"season": season - 1, "week": w, "gsis_id": f"gsis-{p}{r}", "position": p,
          "points": 20.0 - 0.3 * r}
         for p, depth in POSITIONS.items() for r in range(1, depth + 1) for w in range(1, 18)]
    )


@pytest.fixture
def wired(monkeypatch):
    """Point the board at synthetic frames instead of nflverse."""
    monkeypatch.setattr(board_module.nflverse, "ff_rankings", lambda **_: _rankings())
    monkeypatch.setattr(board_module.nflverse, "schedules", lambda *_a, **_k: _schedules())
    monkeypatch.setattr(board_module.nflverse, "ff_playerids", lambda **_: _ids())
    monkeypatch.setattr(board_module, "season_actuals", _actuals)
    monkeypatch.setattr(board_module, "_historical_weekly", _weekly)
    monkeypatch.setattr(board_module.projections, "load_csv_sources",
                        lambda *_a, **_k: (pl.DataFrame(schema={
                            "gsis_id": pl.String, "source": pl.String,
                            "projected_points": pl.Float64}), pl.DataFrame()))
    cfg = load_config("config.example.yaml")
    return cfg, ScoringRules(weights={})


def test_board_renders_ranked_and_tiered(wired):
    cfg, rules = wired
    board, _ = build_board(cfg, rules, season=TARGET)

    assert board.height == sum(POSITIONS.values())
    assert board["rank"].to_list() == list(range(1, board.height + 1))
    assert board["vor"].to_list() == sorted(board["vor"].to_list(), reverse=True)
    assert board["tier"].min() == 1
    assert set(board.columns) >= {"rank", "tier", "name", "position", "vor", "n_sources"}


def test_calibration_trains_only_on_prior_seasons(wired):
    cfg, rules = wired
    _, diagnostics = build_board(cfg, rules, season=TARGET)
    assert diagnostics.calibration.target_season == TARGET
    assert max(diagnostics.calibration.training_seasons) < TARGET
    assert len(diagnostics.calibration.training_seasons) >= cfg.calibration.min_training_seasons


def test_every_position_shrinks_the_outcome_spread(wired):
    cfg, rules = wired
    _, diagnostics = build_board(cfg, rules, season=TARGET)
    shrinkage = diagnostics.shrinkage()
    assert shrinkage  # the gate is vacuous if nothing was fit
    assert all(ratio < 1.0 for ratio in shrinkage.values()), shrinkage


def test_ecr_counts_as_one_source(wired):
    cfg, rules = wired
    board, _ = build_board(cfg, rules, season=TARGET)
    assert board["n_sources"].unique().to_list() == [1]


def test_override_is_taken_at_face_value_not_recalibrated(wired):
    """The manual number must survive to the board exactly as typed."""
    cfg, rules = wired
    stamp = datetime(2026, 8, 27, tzinfo=UTC)
    override = Override(gsis_id="gsis-RB20", field="projected_points", value=999.0,
                        reason="breakout report", line=2, applied_at=stamp)
    board, _ = build_board(cfg, rules, season=TARGET, overrides=[override])

    row = board.filter(pl.col("gsis_id") == "gsis-RB20").to_dicts()[0]
    assert row["projected_points"] == pytest.approx(999.0)
    assert row["override_reason"] == "breakout report"
    assert row["rank"] == 1  # and it re-ranks off the overridden value


def test_excluded_player_is_absent_from_the_board(wired):
    cfg, rules = wired
    stamp = datetime(2026, 8, 27, tzinfo=UTC)
    override = Override(gsis_id="gsis-RB1", field="exclude", value=True,
                        reason="ruled out", line=2, applied_at=stamp)
    board, _ = build_board(cfg, rules, season=TARGET, overrides=[override])
    assert "gsis-RB1" not in board["gsis_id"].to_list()


def test_override_for_a_player_off_the_board_is_reported(wired):
    cfg, rules = wired
    stamp = datetime(2026, 8, 27, tzinfo=UTC)
    override = Override(gsis_id="gsis-NOBODY", field="projected_points", value=100.0,
                        reason="typo", line=2, applied_at=stamp)
    _, diagnostics = build_board(cfg, rules, season=TARGET, overrides=[override])
    assert [o.gsis_id for o in diagnostics.unmatched_overrides] == ["gsis-NOBODY"]


def test_in_season_rankings_never_reach_the_board(wired):
    """The week-3 scrape reverses the ranking; if it leaked, the board would invert."""
    cfg, rules = wired
    board, _ = build_board(cfg, rules, season=TARGET)
    top_rb = board.filter(pl.col("position") == "RB").head(1).to_dicts()[0]
    assert top_rb["ecr"] == pytest.approx(1.0)


def test_target_season_without_prior_ecr_raises(wired):
    cfg, rules = wired
    with pytest.raises(ValueError, match="no season before"):
        build_board(cfg, rules, season=2019)


def test_board_is_reproducible(wired):
    """R7: two runs over the same data must produce the identical board.

    Players tied on both value and expert rank have no natural order, so without a
    stable last-resort key they swap between runs and can move a tier boundary.
    """
    cfg, rules = wired
    first, _ = build_board(cfg, rules, season=TARGET)
    second, _ = build_board(cfg, rules, season=TARGET)
    assert first.equals(second)
