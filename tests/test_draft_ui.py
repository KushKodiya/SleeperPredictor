"""M13 UI tests — rendered to a string and read back, so the assertions are what a
person would actually see on screen."""

from datetime import UTC, datetime

import polars as pl
import pytest
from rich.console import Console

from ffdraft.data.sleeper import Pick
from ffdraft.draft.runtime import Snapshot
from ffdraft.draft.state import reconcile
from ffdraft.draft.ui import board_is_renderable, render

ME = "841478247987380224"
ROSTER = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "FLEX", "K", "DEF"] + ["BN"] * 5
FLEX = {"FLEX": ["RB", "WR", "TE"]}
NOW = datetime(2026, 8, 27, 12, 30, tzinfo=UTC)

BOARD = pl.DataFrame(
    [
        {"rank": 1, "tier": 1, "name": "Alpha Back", "position": "RB", "vor": 140.0,
         "gsis_id": "gsis-1", "override_reason": None, "override_games": None},
        {"rank": 2, "tier": 1, "name": "Beta Wide", "position": "WR", "vor": 90.5,
         "gsis_id": "gsis-2", "override_reason": "hamstring, per beat writer",
         "override_games": None},
    ]
)


def _snapshot(board=BOARD, *, picks=(), warning=None) -> Snapshot:
    state = reconcile(list(picks), my_user_id=ME, roster_positions=ROSTER, flex_eligibility=FLEX)
    return Snapshot(board=board, state=state, updated_at=NOW, warning=warning,
                    picks_seen=state.pick_count)


def _screen(snapshot, *, width=140) -> str:
    console = Console(width=width, record=True, force_terminal=False)
    console.print(render(snapshot))
    return console.export_text()


def _pick(player_id, position, pick_no, name="Alpha") -> Pick:
    return Pick.model_validate(
        {"player_id": player_id, "picked_by": ME, "roster_id": "1", "round": 1,
         "draft_slot": 1, "pick_no": pick_no,
         "metadata": {"position": position, "first_name": name, "last_name": "Back"}}
    )


# --- 3.1 the board renders --------------------------------------------------------


def test_available_players_render_with_vor_and_tier():
    screen = _screen(_snapshot())
    assert "Alpha Back" in screen
    assert "140.0" in screen
    assert "available" in screen


def test_roster_and_needs_render():
    screen = _screen(_snapshot(picks=[_pick("111", "RB", 1, name="Alpha")]))
    assert "my roster" in screen
    assert "needs" in screen
    assert "RB×1" in screen  # one RB slot left of the two
    assert "bench ×5" in screen


def test_empty_roster_says_so_rather_than_rendering_blank():
    screen = _screen(_snapshot())
    assert "nothing drafted yet" in screen


def test_a_complete_roster_reports_it():
    positions = ["QB", "RB", "RB", "WR", "WR", "TE", "RB", "WR", "K", "DEF",
                 "QB", "RB", "WR", "TE", "K"]
    picks = [_pick(str(i), p, i) for i, p in enumerate(positions, start=1)]
    assert "roster complete" in _screen(_snapshot(picks=picks))


def test_an_empty_board_still_renders():
    """Late in a draft the board can genuinely empty out; it must not blow up."""
    screen = _screen(_snapshot(board=BOARD.head(0)))
    assert "no players available" in screen


def test_future_phase_columns_render_as_visible_placeholders():
    """Marginal value and Q(p) land in Phases 5 and 8; the gap should be obvious."""
    screen = _screen(_snapshot())
    assert "marg" in screen and "Q(p)" in screen
    assert "—" in screen


# --- 3.2 override visibility ------------------------------------------------------


def test_an_overridden_player_is_marked_and_explained():
    """An invisible override is a trap (PRD §8 M3a)."""
    screen = _screen(_snapshot())
    assert "*Beta Wide" in screen
    assert "hamstring, per beat writer" in screen


def test_a_games_played_override_shows_the_games_alongside_the_reason():
    """Shown, never folded into the number — the owner makes that call by eye."""
    board = BOARD.with_columns(
        pl.when(pl.col("gsis_id") == "gsis-1").then(13.0)
        .otherwise(pl.col("override_games")).alias("override_games"),
        pl.when(pl.col("gsis_id") == "gsis-1").then(pl.lit("knee, expects to miss 4"))
        .otherwise(pl.col("override_reason")).alias("override_reason"),
    )
    screen = _screen(_snapshot(board=board))
    assert "13g" in screen
    assert "knee, expects to miss 4" in screen
    assert "140.0" in screen  # the projection-derived VOR is untouched


def test_a_player_without_an_override_is_not_marked():
    screen = _screen(_snapshot())
    assert "*Alpha Back" not in screen


# --- the warning banner -----------------------------------------------------------


def test_a_warning_renders_a_banner_with_the_last_update_time():
    screen = _screen(_snapshot(warning="draft feed unreachable; showing last known picks"))
    assert "stale board" in screen
    assert "draft feed unreachable" in screen
    assert "12:30:00" in screen


def test_no_banner_when_the_board_is_current():
    assert "stale board" not in _screen(_snapshot())


def test_the_board_still_lists_players_while_stale():
    """Degrading means showing the last good board, not showing nothing."""
    screen = _screen(_snapshot(warning="draft feed unreachable"))
    assert "Alpha Back" in screen


@pytest.mark.parametrize("width", [80, 100, 140, 200])
def test_renders_at_any_terminal_width(width):
    assert "Alpha Back" in _screen(_snapshot(), width=width)


def test_board_is_renderable_checks_the_columns_the_ui_reads():
    assert board_is_renderable(BOARD)
    assert not board_is_renderable(BOARD.drop("vor"))
