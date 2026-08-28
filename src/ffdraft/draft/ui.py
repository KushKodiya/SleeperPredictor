"""M13 — the terminal board.

Read under time pressure, so the layout answers three questions in order: who should I
take, what do I still need, and can I trust what I am looking at. The warning banner is
first because a stale board that looks fresh is the dangerous failure.

Columns for values later phases produce — marginal value (Phase 5) and Q(p) (Phase 8) —
render as a visible placeholder rather than being hidden, so the gap is obvious.
"""

from __future__ import annotations

import polars as pl
from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ffdraft.draft.runtime import Snapshot

PLACEHOLDER = "—"
OVERRIDE_MARKER = "*"


def warning_banner(snapshot: Snapshot) -> RenderableType | None:
    """A loud banner whenever the board is not known to be current."""
    if snapshot.warning is None:
        return None
    return Panel(
        Text(f"{snapshot.warning}\nlast updated {snapshot.updated_at:%H:%M:%S}", style="bold"),
        title="[bold]stale board[/bold]",
        border_style="yellow",
    )


def available_table(snapshot: Snapshot, *, top: int = 15) -> Table:
    """The best players still on the board."""
    table = Table(title=f"available — {snapshot.picks_seen} picks in", expand=False)
    for column, justify in (
        ("#", "right"), ("tier", "right"), ("player", "left"), ("pos", "left"),
        ("vor", "right"), ("marg", "right"), ("Q(p)", "right"), ("note", "left"),
    ):
        table.add_column(column, justify=justify, no_wrap=(column != "note"))

    if snapshot.board.is_empty():
        table.add_row(PLACEHOLDER, PLACEHOLDER, "no players available", *[PLACEHOLDER] * 5)
        return table

    for row in snapshot.board.head(top).iter_rows(named=True):
        reason = row.get("override_reason")
        games = row.get("override_games")
        # games_played is shown, never folded into the number — the projection already
        # carries typical missed games, so the owner makes that call by eye until Phase 5.
        note = f"{games:g}g · {reason}" if reason and games is not None else (reason or "")
        name = row.get("name") or row.get("gsis_id", "")
        table.add_row(
            str(row.get("rank", "")),
            str(row.get("tier", "")),
            f"{OVERRIDE_MARKER}{name}" if reason else str(name),
            str(row.get("position", "")),
            f"{row['vor']:.1f}" if row.get("vor") is not None else PLACEHOLDER,
            PLACEHOLDER,  # marginal value arrives with the lineup model in Phase 5
            PLACEHOLDER,  # Q(p) arrives with the simulation in Phase 8
            Text(note, style="yellow") if note else "",
            style="yellow" if reason else None,
        )
    return table


def roster_panel(snapshot: Snapshot) -> Panel:
    """What the owner already has, in the slot each player fills."""
    table = Table.grid(padding=(0, 2))
    table.add_column("pick", justify="right")
    table.add_column("player")
    table.add_column("pos")
    for player in sorted(snapshot.state.my_players, key=lambda p: p.pick_no):
        table.add_row(str(player.pick_no), player.name, player.position or PLACEHOLDER)
    if not snapshot.state.my_players:
        table.add_row(PLACEHOLDER, "nothing drafted yet", PLACEHOLDER)
    return Panel(table, title="my roster", border_style="cyan")


def needs_panel(snapshot: Snapshot) -> Panel:
    """Slots still to fill, starters before bench."""
    needs = snapshot.state.needs
    if not needs:
        return Panel(Text("roster complete"), title="needs", border_style="green")
    starters = {slot: n for slot, n in needs.items() if slot != "BN"}
    bench = needs.get("BN", 0)
    body = "  ".join(f"{slot}×{count}" for slot, count in sorted(starters.items())) or "starters set"
    if bench:
        body += f"    (bench ×{bench})"
    return Panel(Text(body), title="needs", border_style="cyan")


def render(snapshot: Snapshot, *, top: int = 15) -> RenderableType:
    """The whole screen for one snapshot."""
    parts: list[RenderableType] = []
    banner = warning_banner(snapshot)
    if banner is not None:
        parts.append(banner)
    parts.extend([available_table(snapshot, top=top), roster_panel(snapshot), needs_panel(snapshot)])
    return Group(*parts)


def board_is_renderable(board: pl.DataFrame) -> bool:
    """The columns the UI reads. Checked before a draft, not during one."""
    return {"rank", "tier", "name", "position", "vor"} <= set(board.columns)
