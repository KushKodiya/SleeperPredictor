"""ffdraft CLI. Phase 1: `rebuild` reconstructs the full data cache in one command."""

from __future__ import annotations

import threading
from pathlib import Path

import polars as pl
import typer
from rich.console import Console
from rich.live import Live

from ffdraft.backtest.harness import BacktestResult, head_to_head, summarize
from ffdraft.backtest.runner import backtest_season, backtestable_seasons, season_inputs
from ffdraft.config import load_config
from ffdraft.data import nflverse
from ffdraft.data.adp import adp_format_from_scoring, fetch_adp, join_adp_to_crosswalk
from ffdraft.data.crosswalk import build_crosswalk, load_id_overrides, write_unmatched_report
from ffdraft.data.overrides import load_overrides
from ffdraft.data.sleeper import SleeperClient
from ffdraft.draft.runtime import DraftSession, run_poller
from ffdraft.draft.ui import board_is_renderable, render
from ffdraft.lineup.slots import SlotConfig
from ffdraft.scoring import golden
from ffdraft.scoring.engine import parse_settings
from ffdraft.valuation.board import build_board

app = typer.Typer(add_completion=False, help="Fantasy football draft engine")
console = Console()


@app.callback()
def main() -> None:
    """Fantasy football draft engine (keeps `rebuild` an explicit subcommand)."""


@app.command()
def rebuild(
    refresh: bool = typer.Option(False, "--refresh", help="Bypass caches and re-fetch"),
    season: int = typer.Option(None, help="Override the ADP/season year"),
) -> None:
    """Rebuild the full data cache and report crosswalk coverage (Phase 1 gate)."""
    cfg = load_config()
    year = season or cfg.league.season
    console.print("[bold]Rebuilding data cache...[/bold]")

    # 1. nflverse sources
    ff = nflverse.ff_playerids(refresh=refresh)
    nflverse.players(refresh=refresh)

    # 2. Sleeper global player map (no league_id needed); non-fatal on failure
    try:
        SleeperClient(
            timeout=cfg.runtime.http_timeout_seconds, max_retries=cfg.runtime.http_max_retries
        ).players_nfl(
            Path(cfg.data.cache_dir) / "sleeper" / "players_nfl.json",
            cfg.data.sleeper_players_ttl_hours,
        )
    except Exception as exc:  # noqa: BLE001 - offline prep, warn and continue
        console.print(f"[yellow]warn: Sleeper players fetch failed: {exc}[/yellow]")

    # 3. Crosswalk
    crosswalk = build_crosswalk(ff)

    # 4. ADP format from the live league scoring; default ppr until league_id is set
    if cfg.league.league_id == "REPLACE_ME":
        fmt = "ppr"
        console.print("[yellow]league_id not set — defaulting ADP format to 'ppr'[/yellow]")
    else:
        league = SleeperClient(
            timeout=cfg.runtime.http_timeout_seconds, max_retries=cfg.runtime.http_max_retries
        ).get_league(cfg.league.league_id)
        fmt = adp_format_from_scoring(league.scoring_settings)
    adp = fetch_adp(fmt, cfg.league.teams, year, ttl_hours=cfg.data.adp_ttl_hours)

    # 5. Resolve ADP -> crosswalk, write the unmatched report
    overrides = load_id_overrides(cfg.overrides.player_id_overrides)
    _, unmatched = join_adp_to_crosswalk(
        adp, crosswalk, fuzzy_threshold=cfg.crosswalk.fuzzy_threshold, overrides=overrides
    )
    write_unmatched_report(unmatched, cfg.crosswalk.unmatched_report)

    # 6. Top-300 coverage gate
    top = adp.filter(pl.col("rank") <= 300)
    top_matched, top_unmatched = join_adp_to_crosswalk(
        top, crosswalk, fuzzy_threshold=cfg.crosswalk.fuzzy_threshold, overrides=overrides
    )
    coverage = top_matched.height / max(top.height, 1)

    if top_unmatched.height:  # R4 startup warning
        console.print(
            f"[yellow]WARNING: {top_unmatched.height} of top-{top.height} ADP players "
            f"unmatched — review {cfg.crosswalk.unmatched_report}[/yellow]"
        )
    style = "green" if coverage >= 0.98 else "red"
    console.print(
        f"top-{top.height} ADP crosswalk coverage: "
        f"[bold {style}]{coverage:.1%}[/bold {style}]  (gate: >=98%)"
    )


@app.command("freeze-golden")
def freeze_golden(
    season: int = typer.Option(None, help="Season to freeze; defaults to the prior season"),
    refresh: bool = typer.Option(False, "--refresh", help="Bypass nflverse caches"),
) -> None:
    """Freeze last season of the owner's league as the scoring canary fixture (Phase 2 gate)."""
    cfg = load_config()
    client = SleeperClient(
        timeout=cfg.runtime.http_timeout_seconds, max_retries=cfg.runtime.http_max_retries
    )
    league = client.get_league(cfg.league.league_id)
    if league.previous_league_id is None:
        raise typer.BadParameter(
            f"league {cfg.league.league_id} has no previous_league_id; "
            f"there is no prior season to reproduce"
        )
    prior = client.get_league(league.previous_league_id)
    year = season or int(prior.season or cfg.league.season - 1)

    out = golden.build_fixture(prior.league_id, year, client=client, refresh=refresh)
    console.print(f"froze {year} league {prior.league_id} -> [bold]{out}[/bold]")

    teams, _ = golden.reproduce(golden.load_fixture(out))
    worst = teams.select(pl.col("diff").abs().max()).item()
    failures = teams.filter(pl.col("diff").abs() > golden.TOLERANCE)
    style = "green" if failures.height == 0 else "red"
    console.print(
        f"team-weeks reproduced: [bold {style}]{teams.height - failures.height}/{teams.height}"
        f"[/bold {style}]  worst diff {worst:.2f}  (gate: <= {golden.TOLERANCE})"
    )
    if failures.height:
        console.print(failures)


@app.command()
def board(
    season: int = typer.Option(None, help="Season to build the board for"),
    top: int = typer.Option(20, help="How many players to print"),
    refresh: bool = typer.Option(False, "--refresh", help="Bypass nflverse caches"),
) -> None:
    """Build the ranked draft board for the owner's league (Phase 3 gate)."""
    cfg = load_config()
    year = season or cfg.league.season
    client = SleeperClient(
        timeout=cfg.runtime.http_timeout_seconds, max_retries=cfg.runtime.http_max_retries
    )
    rules = parse_settings(client.get_league(cfg.league.league_id).scoring_settings)

    crosswalk = build_crosswalk(nflverse.ff_playerids(refresh=refresh))
    overrides = load_overrides(cfg.overrides.projection_overrides, crosswalk)
    ranked, diagnostics = build_board(
        cfg, rules, season=year, overrides=overrides, refresh=refresh
    )

    console.print(ranked.head(top))
    shrinkage = diagnostics.shrinkage()
    worst = max(shrinkage.values())
    style = "green" if worst < 1.0 else "red"
    console.print(
        "calibration shrinkage (fitted spread / actual spread, gate: < 1.0): "
        + ", ".join(f"{p}=[bold {style}]{v:.2f}[/bold {style}]" for p, v in sorted(shrinkage.items()))
    )
    console.print(f"board: {ranked.height} players over {ranked['tier'].max()} tiers")
    if diagnostics.unresolved_rankings.height:  # R4
        console.print(
            f"[yellow]WARNING: {diagnostics.unresolved_rankings.height} ranked players "
            f"did not resolve to a gsis_id and are absent from the board[/yellow]"
        )
    for override in diagnostics.unmatched_overrides:
        console.print(
            f"[yellow]WARNING: override on line {override.line} targets "
            f"{override.gsis_id}, who is not on the board[/yellow]"
        )


@app.command()
def draft(
    season: int = typer.Option(None, help="Season the board is built for"),
    top: int = typer.Option(15, help="How many available players to show"),
    once: bool = typer.Option(False, "--once", help="Poll once, print the board, exit"),
    draft_id: str = typer.Option(None, help="Override the draft id from config"),
    refresh: bool = typer.Option(False, "--refresh", help="Bypass nflverse caches"),
) -> None:
    """Track a live Sleeper draft against the ranked board (Phase 4 gate)."""
    cfg = load_config()
    year = season or cfg.league.season
    client = SleeperClient(
        timeout=cfg.runtime.http_timeout_seconds, max_retries=cfg.runtime.http_max_retries
    )

    # Roster slots come from the live league; config only backs it up (config.yaml §league).
    try:
        league = client.get_league(cfg.league.league_id)
        rules = parse_settings(league.scoring_settings)
        roster_positions = league.roster_positions or cfg.league.fallback_roster_positions
    except Exception as exc:
        console.print(f"[yellow]league fetch failed ({exc}); using configured fallbacks[/yellow]")
        raise typer.Exit(1) from exc

    console.print("[bold]building the board...[/bold]")
    board, diagnostics = build_board(cfg, rules, season=year, overrides=[], refresh=refresh)
    if not board_is_renderable(board):
        raise typer.BadParameter(f"board is missing UI columns: {sorted(board.columns)}")
    if diagnostics.unresolved_rankings.height:
        console.print(
            f"[yellow]{diagnostics.unresolved_rankings.height} ranked players are not on "
            f"the board (see the unmatched report)[/yellow]"
        )

    session = DraftSession(
        client=client,
        draft_id=draft_id or cfg.league.draft_id,
        base_board=board,
        crosswalk=build_crosswalk(nflverse.ff_playerids()),
        my_user_id=cfg.league.my_user_id,
        roster_positions=roster_positions,
        flex_eligibility=cfg.flex_eligibility,
        overrides_path=cfg.overrides.projection_overrides,
        reload_overrides=cfg.overrides.reload_on_every_pick,
        anomaly_log=cfg.runtime.anomaly_log,
    )

    if once:
        console.print(render(session.poll_once(), top=top))
        return

    stop = threading.Event()
    run_poller(session, interval=cfg.runtime.poll_interval_seconds, stop=stop)
    try:
        with Live(render(session.snapshot(), top=top), console=console, screen=False) as live:
            while not stop.is_set():
                live.update(render(session.snapshot(), top=top))
                stop.wait(0.25)  # render cadence, independent of the poll interval
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        console.print(f"anomalies logged to [bold]{cfg.runtime.anomaly_log}[/bold]")


@app.command()
def backtest(
    seasons: str = typer.Option(None, help="Comma-separated seasons; default all backtestable"),
    min_training_seasons: int = typer.Option(
        2, help="Training-season floor for point-in-time boards (see the phase 6 gate)"
    ),
    refresh: bool = typer.Option(False, "--refresh", help="Bypass nflverse caches"),
) -> None:
    """Replay historical drafts under point-in-time data and compare baselines (Phase 6 gate)."""
    cfg = load_config()
    client = SleeperClient(
        timeout=cfg.runtime.http_timeout_seconds, max_retries=cfg.runtime.http_max_retries
    )
    rules = parse_settings(client.get_league(cfg.league.league_id).scoring_settings)
    slots = SlotConfig.from_league(cfg.league.fallback_roster_positions, cfg.flex_eligibility)

    targets = (
        [int(s) for s in seasons.split(",")] if seasons else backtestable_seasons(cfg, refresh=refresh)
    )
    if not targets:
        raise typer.BadParameter("no season has both preseason ECR and finished outcomes")

    result, notes = BacktestResult(), []
    for season in targets:
        console.print(f"[bold]backtesting {season}...[/bold]")
        inputs = season_inputs(
            cfg, rules, season=season, min_training_seasons=min_training_seasons, refresh=refresh
        )
        backtest_season(inputs, cfg, slots, result)
        notes.append((season, inputs.training_seasons, inputs.unresolved_adp))

    frame = result.frame()
    console.print("\n[bold]distribution across seasons x slots[/bold]")
    console.print(summarize(frame))
    console.print("\n[bold]paired against adp_follow, per draft[/bold]")
    console.print(head_to_head(frame, baseline="adp_follow"))
    console.print("\n[yellow]point-in-time caveats — the engine differs by season:[/yellow]")
    for season, trained, unresolved in notes:
        console.print(
            f"  {season}: board fit on {trained} training season(s) "
            f"(production floor is {cfg.calibration.min_training_seasons}); "
            f"{unresolved} ADP players unresolved"
        )


if __name__ == "__main__":
    app()
