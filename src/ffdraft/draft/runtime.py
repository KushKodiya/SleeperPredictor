"""M13 — the live-draft loop.

Resilience is the whole point of this module. The owner is on the clock with forty-five
seconds left; a crash is worse than a stale number. So every path through `poll_once`
catches, degrades to the last good board, raises a banner, and writes the reason to a
log for after the draft. It is the one function in this project that must never raise.

Network work never happens on the render path: a poller thread owns the I/O and swaps a
finished snapshot into place, and the UI only ever reads the current snapshot. A poll
that hangs for thirty seconds leaves the board on screen and responsive.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from ffdraft.data.overrides import Override, load_overrides
from ffdraft.data.sleeper import Pick
from ffdraft.draft.state import DraftState, reconcile
from ffdraft.valuation.board import revalue

DEFENSE_PREFIX = "DEF_"


@dataclass(frozen=True)
class Snapshot:
    """Everything the UI needs, already computed. Reading one never touches the network."""

    board: pl.DataFrame
    state: DraftState
    updated_at: datetime
    warning: str | None = None
    picks_seen: int = 0
    unknown_players: tuple[str, ...] = ()

    @property
    def is_stale(self) -> bool:
        return self.warning is not None


def _anomaly_logger(path: str | Path) -> Callable[[str], None]:
    """Append-only anomaly log. Logging must never be the thing that breaks the draft."""
    target = Path(path)

    def log(message: str) -> None:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as handle:
                handle.write(f"{datetime.now(UTC).isoformat()}\t{message}\n")
        except Exception:  # noqa: BLE001, S110 - a failed log must not interrupt the draft
            pass

    return log


def sleeper_to_board_id(crosswalk: pl.DataFrame) -> dict[str, str]:
    """Sleeper player id -> the id the board is keyed on.

    Defenses never appear in the crosswalk; their Sleeper id is the team abbreviation,
    which maps onto the board's synthetic `DEF_{team}` row.
    """
    resolved = crosswalk.filter(
        pl.col("sleeper_id").is_not_null() & pl.col("gsis_id").is_not_null()
    ).unique(subset=["sleeper_id"])
    return {
        str(sleeper): gsis
        for sleeper, gsis in zip(
            resolved["sleeper_id"].to_list(), resolved["gsis_id"].to_list(), strict=True
        )
    }


class DraftSession:
    """Owns live draft state. `poll_once` is the only thing that touches the network."""

    def __init__(
        self,
        *,
        client,
        draft_id: str,
        base_board: pl.DataFrame,
        crosswalk: pl.DataFrame,
        my_user_id: str,
        roster_positions: list[str],
        flex_eligibility: dict[str, list[str]],
        overrides_path: str | Path,
        reload_overrides: bool,
        anomaly_log: str | Path,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._client = client
        self._draft_id = draft_id
        # Held without overrides applied, so each poll can re-apply the current file from
        # scratch and a *removed* override reverts instead of sticking.
        self._base_board = base_board
        self._ids = sleeper_to_board_id(crosswalk)
        self._my_user_id = my_user_id
        self._roster_positions = roster_positions
        self._flex_eligibility = flex_eligibility
        self._overrides_path = overrides_path
        self._reload_overrides = reload_overrides
        self._log = _anomaly_logger(anomaly_log)
        self._clock = clock
        self._crosswalk = crosswalk

        self._lock = threading.Lock()
        self._picks: list[Pick] = []
        self._overrides: list[Override] = []
        self._reported_unknown: set[str] = set()
        self._snapshot = Snapshot(
            board=base_board,
            state=reconcile([], my_user_id=my_user_id, roster_positions=roster_positions,
                            flex_eligibility=flex_eligibility),
            updated_at=clock(),
            warning="waiting for the first poll",
        )

    def snapshot(self) -> Snapshot:
        """The current snapshot. Cheap, lock-guarded, and never blocked by a poll."""
        with self._lock:
            return self._snapshot

    def poll_once(self) -> Snapshot:
        """Fetch, reload, reconcile and revalue. Catches everything; never raises."""
        warnings: list[str] = []

        picks = self._fetch_picks(warnings)
        overrides = self._reload(warnings)

        try:
            state = reconcile(
                picks,
                my_user_id=self._my_user_id,
                roster_positions=self._roster_positions,
                flex_eligibility=self._flex_eligibility,
            )
        except Exception as exc:  # noqa: BLE001 - keep the last good state on screen
            self._log(f"reconcile failed: {exc!r}")
            warnings.append("could not reconcile picks; showing last good board")
            with self._lock:
                self._snapshot = replace(
                    self._snapshot, warning="; ".join(warnings), updated_at=self._clock()
                )
                return self._snapshot

        board, unknown = self._available_board(state, overrides, warnings)
        snapshot = Snapshot(
            board=board,
            state=state,
            updated_at=self._clock(),
            warning="; ".join(warnings) if warnings else None,
            picks_seen=state.pick_count,
            unknown_players=unknown,
        )
        with self._lock:
            self._snapshot = snapshot
        return snapshot

    # --- the pieces, each of which degrades on its own -----------------------------

    def _fetch_picks(self, warnings: list[str]) -> list[Pick]:
        try:
            picks = self._client.get_draft_picks(self._draft_id)
        except Exception as exc:  # noqa: BLE001 - timeouts, 500s, malformed payloads
            self._log(f"pick fetch failed: {exc!r}")
            warnings.append("draft feed unreachable; showing last known picks")
            return self._picks
        # Only replace known-good picks once the new set actually parsed.
        self._picks = picks
        return picks

    def _reload(self, warnings: list[str]) -> list[Override]:
        if not self._reload_overrides:
            return self._overrides
        try:
            self._overrides = load_overrides(self._overrides_path, self._crosswalk)
        except Exception as exc:  # noqa: BLE001 - a typo mid-draft must not end the draft
            self._log(f"override reload failed: {exc!r}")
            warnings.append("override file invalid; keeping last good overrides")
        return self._overrides

    def _available_board(
        self, state: DraftState, overrides: list[Override], warnings: list[str]
    ) -> tuple[pl.DataFrame, tuple[str, ...]]:
        try:
            board, unmatched = revalue(self._base_board, overrides)
        except Exception as exc:  # noqa: BLE001
            self._log(f"revalue failed: {exc!r}")
            warnings.append("could not apply overrides; showing unadjusted board")
            board, unmatched = self._base_board, []
        for override in unmatched:
            self._note_once(f"override:{override.gsis_id}",
                            f"override on line {override.line} targets {override.gsis_id}, "
                            f"who is not on the board")

        drafted_board_ids, unknown = [], []
        for player_id in state.drafted_ids:
            board_id = self._ids.get(player_id)
            if board_id is None:
                board_id = (
                    f"{DEFENSE_PREFIX}{player_id}"
                    if player_id.isalpha() and player_id.isupper()
                    else None
                )
            if board_id is None:
                unknown.append(player_id)
                drafted = state.drafted[player_id]
                self._note_once(
                    f"unknown:{player_id}",
                    f"drafted player {player_id} ({drafted.name}, {drafted.position}) "
                    f"is not in the crosswalk; board cannot mark him taken",
                )
                continue
            drafted_board_ids.append(board_id)

        if unknown:
            warnings.append(f"{len(unknown)} drafted player(s) not in the crosswalk; see the log")
        try:
            board = board.filter(~pl.col("gsis_id").is_in(drafted_board_ids))
        except Exception as exc:  # noqa: BLE001
            self._log(f"could not remove drafted players: {exc!r}")
            warnings.append("board may still show drafted players")
        return board, tuple(sorted(unknown))

    def _note_once(self, key: str, message: str) -> None:
        """Log an anomaly the first time only — a 2-second poll would flood the file."""
        if key not in self._reported_unknown:
            self._reported_unknown.add(key)
            self._log(message)


def run_poller(
    session: DraftSession,
    *,
    interval: float,
    stop: threading.Event,
    on_error: Callable[[BaseException], None] | None = None,
) -> threading.Thread:
    """Start the background poller. The UI thread never waits on it.

    The thread is a daemon so a hung network call can never keep the process alive after
    the owner quits.
    """

    def loop() -> None:
        while not stop.is_set():
            try:
                session.poll_once()
            except BaseException as exc:  # noqa: BLE001 - poll_once should not raise; belt and braces
                if on_error is not None:
                    on_error(exc)
            stop.wait(interval)

    thread = threading.Thread(target=loop, name="ffdraft-poller", daemon=True)
    thread.start()
    return thread
