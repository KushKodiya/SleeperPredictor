"""M13 runtime tests — every induced failure must leave a usable board.

No test here touches the network: the Sleeper client is a stub whose behaviour each test
chooses, including the failure modes the PRD names (500, timeout, malformed payload,
duplicate and out-of-order picks, a player absent from the crosswalk).
"""

import threading
from datetime import UTC, datetime

import httpx
import polars as pl
import pytest

from ffdraft.data.sleeper import Pick, SleeperError
from ffdraft.draft.runtime import DraftSession, run_poller, sleeper_to_board_id

ME = "841478247987380224"
ROSTER = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "FLEX", "K", "DEF"] + ["BN"] * 5
FLEX = {"FLEX": ["RB", "WR", "TE"]}
NOW = datetime(2026, 8, 27, 12, 0, tzinfo=UTC)

BOARD = pl.DataFrame(
    [
        {"rank": 1, "tier": 1, "name": "Alpha Back", "position": "RB", "team": "KC",
         "gsis_id": "gsis-1", "ecr": 1.0, "projected_points": 260.0,
         "replacement_points": 120.0, "vor": 140.0, "points_se": 5.0, "n_sources": 1,
         "override_reason": None},
        {"rank": 2, "tier": 1, "name": "Beta Wide", "position": "WR", "team": "SF",
         "gsis_id": "gsis-2", "ecr": 2.0, "projected_points": 240.0,
         "replacement_points": 150.0, "vor": 90.0, "points_se": 5.0, "n_sources": 1,
         "override_reason": None},
        {"rank": 3, "tier": 2, "name": "Denver Defense", "position": "DEF", "team": "DEN",
         "gsis_id": "DEF_DEN", "ecr": 1.0, "projected_points": 150.0,
         "replacement_points": 120.0, "vor": 30.0, "points_se": 3.0, "n_sources": 1,
         "override_reason": None},
    ]
)
CROSSWALK = pl.DataFrame(
    [{"gsis_id": "gsis-1", "sleeper_id": "111", "normalized_name": "alpha back"},
     {"gsis_id": "gsis-2", "sleeper_id": "222", "normalized_name": "beta wide"}]
)


def _pick(player_id, position, pick_no, *, picked_by=ME, name="Alpha") -> Pick:
    return Pick.model_validate(
        {"player_id": player_id, "picked_by": picked_by, "roster_id": "1", "round": 1,
         "draft_slot": 1, "pick_no": pick_no,
         "metadata": {"position": position, "first_name": name, "last_name": "Back"}}
    )


class StubClient:
    """A Sleeper client the test drives: returns picks, or raises whatever it is given."""

    def __init__(self, picks=None, error=None, block=None):
        self.picks = picks or []
        self.error = error
        self.block = block
        self.calls = 0

    def get_draft_picks(self, draft_id):
        self.calls += 1
        if self.block is not None:
            self.block.wait(timeout=5)
        if self.error is not None:
            raise self.error
        return self.picks


def _session(tmp_path, client, *, reload_overrides=True, overrides="") -> DraftSession:
    path = tmp_path / "projection_overrides.csv"
    if overrides:
        path.write_text(overrides, encoding="utf-8")
    return DraftSession(
        client=client, draft_id="d1", base_board=BOARD, crosswalk=CROSSWALK, my_user_id=ME,
        roster_positions=ROSTER, flex_eligibility=FLEX, overrides_path=path,
        reload_overrides=reload_overrides, anomaly_log=tmp_path / "anomalies.log",
        clock=lambda: NOW,
    )


# --- 2.1 polling and reconciliation -----------------------------------------------


def test_a_pick_removes_the_player_from_the_board(tmp_path):
    session = _session(tmp_path, StubClient([_pick("111", "RB", 1)]))
    snap = session.poll_once()
    assert "gsis-1" not in snap.board["gsis_id"].to_list()
    assert snap.picks_seen == 1
    assert snap.warning is None


def test_a_drafted_defense_is_removed_by_team_abbreviation(tmp_path):
    session = _session(tmp_path, StubClient([_pick("DEN", "DEF", 1)]))
    snap = session.poll_once()
    assert "DEF_DEN" not in snap.board["gsis_id"].to_list()
    assert snap.unknown_players == ()


def test_duplicate_and_out_of_order_picks_converge(tmp_path):
    picks = [_pick("222", "WR", 2), _pick("111", "RB", 1), _pick("111", "RB", 1)]
    session = _session(tmp_path, StubClient(picks))
    snap = session.poll_once()
    assert snap.picks_seen == 2
    assert snap.board["gsis_id"].to_list() == ["DEF_DEN"]


def test_owner_roster_and_needs_track_picks(tmp_path):
    session = _session(tmp_path, StubClient([_pick("111", "RB", 1)]))
    snap = session.poll_once()
    assert [p.player_id for p in snap.state.my_players] == ["111"]
    assert snap.state.needs["RB"] == 1


# --- 2.2 degrade, never crash -----------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [
        SleeperError("500 after retries"),
        httpx.TimeoutException("timed out"),
        httpx.HTTPStatusError("404", request=None, response=None),
        ValueError("malformed response"),
        KeyError("unexpected shape"),
    ],
)
def test_every_fetch_failure_keeps_the_last_good_board(tmp_path, error):
    session = _session(tmp_path, StubClient([_pick("111", "RB", 1)]))
    good = session.poll_once()
    assert good.warning is None

    session._client.error = error
    degraded = session.poll_once()

    assert degraded.warning is not None
    assert "unreachable" in degraded.warning
    assert degraded.board.equals(good.board)  # the last good board, unchanged
    assert degraded.picks_seen == good.picks_seen


def test_failure_on_the_very_first_poll_still_renders(tmp_path):
    """No good board has ever existed; the full board is the right fallback."""
    session = _session(tmp_path, StubClient(error=SleeperError("down")))
    snap = session.poll_once()
    assert snap.warning is not None
    assert snap.board.height == BOARD.height
    assert snap.state.pick_count == 0


def test_recovery_after_a_failure(tmp_path):
    client = StubClient([_pick("111", "RB", 1)], error=SleeperError("down"))
    session = _session(tmp_path, client)
    assert session.poll_once().warning is not None
    client.error = None
    recovered = session.poll_once()
    assert recovered.warning is None
    assert "gsis-1" not in recovered.board["gsis_id"].to_list()


# --- 2.3 anomaly logging ----------------------------------------------------------


def test_uncrosswalked_pick_is_logged_and_the_board_still_renders(tmp_path):
    session = _session(tmp_path, StubClient([_pick("99999", "WR", 1, name="Ghost")]))
    snap = session.poll_once()

    assert snap.unknown_players == ("99999",)
    assert snap.board.height == BOARD.height  # nothing wrongly removed
    assert "crosswalk" in snap.warning
    log = (tmp_path / "anomalies.log").read_text(encoding="utf-8")
    assert "99999" in log and "Ghost" in log


def test_an_anomaly_is_logged_once_not_every_poll(tmp_path):
    """A 2-second poll would otherwise write the same line 1800 times an hour."""
    session = _session(tmp_path, StubClient([_pick("99999", "WR", 1)]))
    for _ in range(5):
        session.poll_once()
    log = (tmp_path / "anomalies.log").read_text(encoding="utf-8")
    assert log.count("99999") == 1


def test_an_unwritable_log_does_not_break_the_poll(tmp_path):
    session = DraftSession(
        client=StubClient([_pick("99999", "WR", 1)]), draft_id="d1", base_board=BOARD,
        crosswalk=CROSSWALK, my_user_id=ME, roster_positions=ROSTER, flex_eligibility=FLEX,
        overrides_path=tmp_path / "none.csv", reload_overrides=False,
        anomaly_log=tmp_path / "nested" / "\0bad", clock=lambda: NOW,
    )
    assert session.poll_once().board.height == BOARD.height  # no crash


# --- 2.4 live override reload -----------------------------------------------------


def test_mid_draft_override_applies_on_the_next_poll(tmp_path):
    session = _session(tmp_path, StubClient([]))
    assert session.poll_once().board.filter(pl.col("gsis_id") == "gsis-2")["vor"].item() == 90.0

    (tmp_path / "projection_overrides.csv").write_text(
        "gsis_id,field,value,reason\ngsis-2,projected_points,400,camp buzz\n", encoding="utf-8"
    )
    snap = session.poll_once()
    row = snap.board.filter(pl.col("gsis_id") == "gsis-2").to_dicts()[0]
    assert row["projected_points"] == pytest.approx(400.0)
    assert row["vor"] == pytest.approx(250.0)  # revalued, not just relabelled
    assert row["rank"] == 1  # and re-ranked to the top
    assert row["override_reason"] == "camp buzz"


def test_a_malformed_override_edit_keeps_the_last_good_state(tmp_path):
    session = _session(
        tmp_path, StubClient([]),
        overrides="gsis_id,field,value,reason\ngsis-2,projected_points,400,camp buzz\n",
    )
    good = session.poll_once()
    assert good.board.filter(pl.col("gsis_id") == "gsis-2")["projected_points"].item() == 400.0

    (tmp_path / "projection_overrides.csv").write_text(
        "gsis_id,field,value,reason\ngsis-2,projected_points,not-a-number,typo\n", encoding="utf-8"
    )
    snap = session.poll_once()
    assert "override file invalid" in snap.warning
    assert snap.board.filter(pl.col("gsis_id") == "gsis-2")["projected_points"].item() == 400.0
    assert "override reload failed" in (tmp_path / "anomalies.log").read_text(encoding="utf-8")


def test_removing_an_override_reverts_the_board(tmp_path):
    """Re-applied from scratch each poll, so a deleted line actually takes effect."""
    session = _session(
        tmp_path, StubClient([]),
        overrides="gsis_id,field,value,reason\ngsis-2,projected_points,400,camp buzz\n",
    )
    assert session.poll_once().board.filter(pl.col("gsis_id") == "gsis-2")["vor"].item() == 250.0
    (tmp_path / "projection_overrides.csv").write_text(
        "gsis_id,field,value,reason\n", encoding="utf-8"
    )
    snap = session.poll_once()
    assert snap.board.filter(pl.col("gsis_id") == "gsis-2")["vor"].item() == pytest.approx(90.0)


def test_an_excluded_player_disappears_mid_draft(tmp_path):
    session = _session(
        tmp_path, StubClient([]),
        overrides="gsis_id,field,value,reason\ngsis-1,exclude,true,ruled out\n",
    )
    assert "gsis-1" not in session.poll_once().board["gsis_id"].to_list()


def test_reload_disabled_leaves_overrides_alone(tmp_path):
    session = _session(tmp_path, StubClient([]), reload_overrides=False)
    (tmp_path / "projection_overrides.csv").write_text(
        "gsis_id,field,value,reason\ngsis-2,projected_points,400,late\n", encoding="utf-8"
    )
    assert session.poll_once().board.filter(pl.col("gsis_id") == "gsis-2")["vor"].item() == 90.0


# --- the poller thread ------------------------------------------------------------


def test_a_slow_poll_never_blocks_the_snapshot(tmp_path):
    """The UI reads snapshots; a hung network call must not stall a render."""
    gate = threading.Event()
    client = StubClient([_pick("111", "RB", 1)], block=gate)
    session = _session(tmp_path, client)
    stop = threading.Event()
    run_poller(session, interval=0.01, stop=stop, on_error=None)
    try:
        # The poll is parked inside the client; snapshots still return immediately.
        for _ in range(50):
            assert session.snapshot() is not None
        assert client.calls >= 1
    finally:
        gate.set()
        stop.set()


def test_the_poller_survives_a_raising_session(tmp_path):
    session = _session(tmp_path, StubClient([]))
    seen: list[BaseException] = []
    session.poll_once = lambda: (_ for _ in ()).throw(RuntimeError("boom"))  # type: ignore[method-assign]

    stop = threading.Event()
    thread = run_poller(session, interval=0.01, stop=stop, on_error=seen.append)
    assert stop.wait(0.2) is False  # still running, not dead
    stop.set()
    thread.join(timeout=2)
    assert seen and isinstance(seen[0], RuntimeError)


def test_sleeper_to_board_id_maps_players_and_skips_blanks():
    mapping = sleeper_to_board_id(
        pl.DataFrame([{"gsis_id": "g1", "sleeper_id": "1"},
                      {"gsis_id": None, "sleeper_id": "2"},
                      {"gsis_id": "g3", "sleeper_id": None}])
    )
    assert mapping == {"1": "g1"}


# --- 4.1 a real draft, arriving one pick at a time --------------------------------


class ReplayClient:
    """Serves a real draft the way a live one arrives: a few more picks each poll."""

    def __init__(self, picks, *, per_poll=7):
        self.picks = picks
        self.per_poll = per_poll
        self.served = 0

    def get_draft_picks(self, draft_id):
        self.served = min(self.served + self.per_poll, len(self.picks))
        return self.picks[: self.served]


def _real_picks():
    import json
    from pathlib import Path

    raw = json.loads(
        (Path(__file__).parent / "fixtures" / "draft_picks_2025.json").read_text(encoding="utf-8")
    )
    return [Pick.model_validate(p) for p in raw]


def _replay_session(tmp_path, client, picks) -> DraftSession:
    """A board and crosswalk covering every player in the real draft."""
    board = pl.DataFrame(
        [{"rank": i + 1, "tier": 1, "name": p.player_id, "position": "RB", "team": "KC",
          "gsis_id": f"gsis-{p.player_id}", "ecr": float(i + 1), "projected_points": 100.0,
          "replacement_points": 50.0, "vor": 50.0, "points_se": 1.0, "n_sources": 1,
          "override_reason": None}
         for i, p in enumerate(picks)]
    )
    crosswalk = pl.DataFrame(
        [{"gsis_id": f"gsis-{p.player_id}", "sleeper_id": p.player_id,
          "normalized_name": p.player_id} for p in picks]
    )
    return DraftSession(
        client=client, draft_id="d1", base_board=board, crosswalk=crosswalk, my_user_id=ME,
        roster_positions=ROSTER, flex_eligibility=FLEX,
        overrides_path=tmp_path / "none.csv", reload_overrides=False,
        anomaly_log=tmp_path / "anomalies.log", clock=lambda: NOW,
    )


def test_every_pick_of_a_real_draft_is_tracked_as_it_arrives(tmp_path):
    """The live-draft shape: picks trickle in and the board shrinks to match."""
    picks = _real_picks()
    client = ReplayClient(picks)
    session = _replay_session(tmp_path, client, picks)

    seen = []
    while client.served < len(picks):
        snap = session.poll_once()
        seen.append(snap.picks_seen)
        # the board always holds exactly the players not yet taken
        assert snap.board.height == len(picks) - snap.picks_seen
        assert snap.warning is None

    final = session.poll_once()
    assert final.picks_seen == 120  # every pick tracked
    assert final.board.is_empty()
    assert len(final.state.my_players) == 15
    assert final.state.needs == {}
    assert seen == sorted(seen)  # picks only ever accumulate


def test_a_dropped_poll_mid_draft_recovers_every_pick(tmp_path):
    """A reconnect re-sends the whole draft; nothing may be lost or double-counted."""
    picks = _real_picks()
    client = ReplayClient(picks, per_poll=40)
    session = _replay_session(tmp_path, client, picks)

    assert session.poll_once().picks_seen == 40
    healthy = client.get_draft_picks

    def boom(draft_id):
        raise SleeperError("connection reset")

    client.get_draft_picks = boom  # type: ignore[method-assign]
    degraded = session.poll_once()
    assert degraded.warning is not None
    assert degraded.picks_seen == 40  # held, not lost

    # the reconnect re-sends the whole draft at once
    client.get_draft_picks = healthy  # type: ignore[method-assign]
    client.served, client.per_poll = 0, len(picks)
    recovered = session.poll_once()
    assert recovered.warning is None
    assert recovered.picks_seen == 120
    assert recovered.board.is_empty()
