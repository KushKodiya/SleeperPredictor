"""M3a tests — the draft-morning override path."""

from datetime import UTC, datetime

import polars as pl
import pytest

from ffdraft.data.overrides import OverrideError, apply_overrides, load_overrides

NOW = datetime(2026, 8, 27, 9, 30, tzinfo=UTC)  # fixed so applied_at is deterministic

CROSSWALK = pl.DataFrame(
    [
        {"gsis_id": "00-0000001", "normalized_name": "aj brown"},
        {"gsis_id": "00-0000002", "normalized_name": "mike williams"},
        {"gsis_id": "00-0000003", "normalized_name": "mike williams"},  # two of them exist
        {"gsis_id": "00-0000004", "normalized_name": "travis kelce"},
    ]
)

BOARD = pl.DataFrame(
    [
        {"gsis_id": "00-0000001", "position": "WR", "projected_points": 240.0},
        {"gsis_id": "00-0000002", "position": "WR", "projected_points": 150.0},
        {"gsis_id": "00-0000004", "position": "TE", "projected_points": 200.0},
    ]
)


def _write(tmp_path, body: str):
    path = tmp_path / "projection_overrides.csv"
    path.write_text(body, encoding="utf-8")
    return path


def test_missing_file_is_not_an_error(tmp_path):
    """Most drafts need no overrides."""
    assert load_overrides(tmp_path / "absent.csv", CROSSWALK, now=NOW) == []


def test_name_resolves_through_the_crosswalk(tmp_path):
    path = _write(tmp_path, "player_name,field,value,reason\nA.J. Brown,projected_points,275,hot camp\n")
    (override,) = load_overrides(path, CROSSWALK, now=NOW)
    assert override.gsis_id == "00-0000001"
    assert override.value == pytest.approx(275.0)
    assert override.reason == "hot camp"
    assert override.applied_at == NOW


def test_ambiguous_player_name_raises_rather_than_guessing(tmp_path):
    path = _write(tmp_path, "player_name,field,value,reason\nMike Williams,projected_points,120,news\n")
    with pytest.raises(OverrideError, match="ambiguous"):
        load_overrides(path, CROSSWALK, now=NOW)


def test_unknown_player_name_raises(tmp_path):
    path = _write(tmp_path, "player_name,field,value,reason\nNobody Here,projected_points,120,news\n")
    with pytest.raises(OverrideError, match="matches no player"):
        load_overrides(path, CROSSWALK, now=NOW)


def test_gsis_id_outside_the_crosswalk_raises(tmp_path):
    path = _write(tmp_path, "gsis_id,field,value,reason\n00-9999999,projected_points,120,news\n")
    with pytest.raises(OverrideError, match="not in the crosswalk"):
        load_overrides(path, CROSSWALK, now=NOW)


def test_malformed_row_fails_the_whole_load_naming_the_line(tmp_path):
    path = _write(
        tmp_path,
        "player_name,field,value,reason\n"
        "A.J. Brown,projected_points,275,fine\n"      # line 2, valid
        "Travis Kelce,projected_points,not-a-number,typo\n",  # line 3, broken
    )
    with pytest.raises(OverrideError, match="line 3") as exc:
        load_overrides(path, CROSSWALK, now=NOW)
    assert "no overrides were applied" in str(exc.value)


def test_every_bad_line_is_reported_not_just_the_first(tmp_path):
    path = _write(
        tmp_path,
        "player_name,field,value,reason\n"
        "A.J. Brown,projected_points,nope,typo\n"
        "Mike Williams,projected_points,120,news\n",
    )
    with pytest.raises(OverrideError) as exc:
        load_overrides(path, CROSSWALK, now=NOW)
    assert "line 2" in str(exc.value) and "line 3" in str(exc.value)


def test_unknown_field_raises(tmp_path):
    path = _write(tmp_path, "player_name,field,value,reason\nA.J. Brown,vibes,9,hunch\n")
    with pytest.raises(OverrideError, match="unknown field"):
        load_overrides(path, CROSSWALK, now=NOW)


def test_games_played_override_is_refused_until_the_availability_model_exists(tmp_path):
    """Recognised by the PRD schema but not actionable yet - refuse loudly, never no-op."""
    path = _write(tmp_path, "player_name,field,value,reason\nA.J. Brown,games_played,14,soft tissue\n")
    with pytest.raises(OverrideError, match="Phase 5"):
        load_overrides(path, CROSSWALK, now=NOW)


def test_reason_is_required(tmp_path):
    path = _write(tmp_path, "player_name,field,value,reason\nA.J. Brown,projected_points,275,\n")
    with pytest.raises(OverrideError, match="reason is required"):
        load_overrides(path, CROSSWALK, now=NOW)


# --- application ------------------------------------------------------------------


def _load(tmp_path, body):
    return load_overrides(_write(tmp_path, body), CROSSWALK, now=NOW)


def test_manual_value_is_used_directly_not_recalibrated(tmp_path):
    overrides = _load(tmp_path, "player_name,field,value,reason\nA.J. Brown,projected_points,275,hot camp\n")
    board, unmatched = apply_overrides(BOARD, overrides)
    row = board.filter(pl.col("gsis_id") == "00-0000001").to_dicts()[0]
    assert row["projected_points"] == pytest.approx(275.0)  # exactly what was typed
    assert row["override_reason"] == "hot camp"
    assert unmatched == []


def test_excluded_player_disappears_from_the_board(tmp_path):
    overrides = _load(tmp_path, "player_name,field,value,reason\nTravis Kelce,exclude,true,ruled out\n")
    board, _ = apply_overrides(BOARD, overrides)
    assert "00-0000004" not in board["gsis_id"].to_list()
    assert board.height == BOARD.height - 1


def test_exclude_false_leaves_the_player_on_the_board(tmp_path):
    overrides = _load(tmp_path, "player_name,field,value,reason\nTravis Kelce,exclude,false,cleared\n")
    board, _ = apply_overrides(BOARD, overrides)
    assert "00-0000004" in board["gsis_id"].to_list()


def test_non_override_rows_are_untouched(tmp_path):
    overrides = _load(tmp_path, "player_name,field,value,reason\nA.J. Brown,projected_points,275,hot camp\n")
    board, _ = apply_overrides(BOARD, overrides)
    other = board.filter(pl.col("gsis_id") == "00-0000002").to_dicts()[0]
    assert other["projected_points"] == pytest.approx(150.0)
    assert other["override_reason"] is None


def test_override_for_a_player_off_the_board_is_surfaced(tmp_path):
    """An override that quietly does nothing is the trap this module exists to avoid."""
    # 00-0000003 is in the crosswalk but not on the board; name would be ambiguous, so use the id
    overrides = load_overrides(
        _write(tmp_path, "gsis_id,field,value,reason\n00-0000003,projected_points,99,news\n"),
        CROSSWALK, now=NOW,
    )
    board, unmatched = apply_overrides(BOARD, overrides)
    assert board.height == BOARD.height
    assert [o.gsis_id for o in unmatched] == ["00-0000003"]


def test_no_overrides_still_yields_the_marker_column():
    board, unmatched = apply_overrides(BOARD, [])
    assert "override_reason" in board.columns
    assert board["override_reason"].null_count() == board.height
    assert unmatched == []
