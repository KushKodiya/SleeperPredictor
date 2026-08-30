"""M15 gate tests — the hard gate's standard, and that a skipped soft gate is not a pass."""

from pathlib import Path

import polars as pl
import pytest

from ffdraft.models import evaluation as E


def _frame(model, board, actual, *, season=2024, third_party=None):
    rows = {
        "season": [season] * len(actual),
        "gsis_id": [f"p{i}" for i in range(len(actual))],
        "model_points": model,
        "board_points": board,
        "actual_points": actual,
    }
    if third_party is not None:
        rows["third_party_points"] = third_party
    return pl.DataFrame(rows)


def test_a_model_that_beats_the_board_on_both_metrics_passes():
    actual = [100.0, 200.0, 300.0, 400.0]
    result = E.hard_gate(_frame(model=[105.0, 195.0, 305.0, 395.0],
                                board=[150.0, 150.0, 350.0, 350.0], actual=actual))
    assert result.passed and "beats" in result.reason


def test_a_model_that_wins_on_error_but_loses_on_rank_does_not_pass():
    """Both metrics, not either: good calibration with bad ordering drafts wrong."""
    actual = [100.0, 200.0, 300.0, 400.0]
    result = E.hard_gate(_frame(model=[260.0, 240.0, 260.0, 240.0],
                                board=[100.0, 200.0, 300.0, 900.0], actual=actual))
    assert not result.passed


def test_the_gate_reports_per_season_as_well_as_pooled():
    """A win driven by one season must be visible, not averaged into a headline."""
    frame = pl.concat([
        _frame([105.0, 195.0], [150.0, 150.0], [100.0, 200.0], season=2023),
        _frame([105.0, 195.0], [150.0, 150.0], [100.0, 200.0], season=2024),
    ])
    result = E.hard_gate(frame)
    assert sorted(result.by_season["season"].unique().to_list()) == [2023, 2024]
    assert result.pooled.height == 2


def test_an_absent_csv_makes_the_soft_gate_skipped_not_passed(tmp_path: Path):
    frame = _frame([100.0], [100.0], [100.0])
    result = E.soft_gate(frame, [2024], csv_dir=tmp_path)
    assert result.skipped
    assert not result.passed
    assert "SKIPPED" in result.reason
    assert result.skipped_seasons == (2024,)


def test_the_soft_gate_names_the_seasons_it_could_not_cover(tmp_path: Path):
    result = E.soft_gate(_frame([100.0], [100.0], [100.0]), [2022, 2023], csv_dir=tmp_path)
    assert "2022" in result.reason and "2023" in result.reason


def test_the_soft_gate_runs_when_a_csv_is_present(tmp_path: Path):
    (tmp_path / "espn_2024.csv").write_text("player_name\nx\n", encoding="utf-8")
    frame = _frame([105.0, 195.0], [150.0, 150.0], [100.0, 200.0],
                   third_party=[130.0, 170.0])
    result = E.soft_gate(frame, [2024], csv_dir=tmp_path)
    assert not result.skipped
    assert result.passed  # the model is closer than the third party here


def test_a_skipped_soft_gate_does_not_block_the_hard_gate(tmp_path: Path):
    frame = _frame([105.0, 195.0], [150.0, 150.0], [100.0, 200.0])
    assert E.hard_gate(frame).passed
    assert E.soft_gate(frame, [2024], csv_dir=tmp_path).skipped


def test_third_party_seasons_are_split_into_covered_and_missing(tmp_path: Path):
    (tmp_path / "sleeper_2023.csv").write_text("player_name\nx\n", encoding="utf-8")
    covered, missing = E.third_party_sources([2022, 2023], csv_dir=tmp_path)
    assert covered == [2023] and missing == [2022]


def test_spearman_is_one_for_a_perfectly_ordered_projection():
    frame = _frame([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0], [10.0, 20.0, 30.0, 40.0])
    scores = {r["source"]: r for r in E.score_sources(
        frame, ["model_points", "board_points"]).iter_rows(named=True)}
    assert scores["model_points"]["spearman"] == pytest.approx(1.0)
    assert scores["board_points"]["spearman"] == pytest.approx(-1.0)


def test_an_incumbent_with_no_ordering_does_not_block_a_model_that_has_one():
    """A constant projection has an undefined rank correlation, not a bad one.

    Comparing against NaN returns False silently, which would let a degenerate incumbent
    veto a model that orders players perfectly.
    """
    result = E.hard_gate(
        _frame(model=[105.0, 195.0], board=[150.0, 150.0], actual=[100.0, 200.0])
    )
    assert result.passed


def test_a_model_with_no_ordering_fails_even_against_a_flat_incumbent():
    result = E.hard_gate(
        _frame(model=[150.0, 150.0], board=[150.0, 150.0], actual=[100.0, 200.0])
    )
    assert not result.passed
