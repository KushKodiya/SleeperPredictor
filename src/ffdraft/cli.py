"""ffdraft CLI. Phase 1: `rebuild` reconstructs the full data cache in one command."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import typer
from rich.console import Console

from ffdraft.config import load_config
from ffdraft.data import nflverse
from ffdraft.data.adp import adp_format_from_scoring, fetch_adp, join_adp_to_crosswalk
from ffdraft.data.crosswalk import build_crosswalk, load_id_overrides, write_unmatched_report
from ffdraft.data.sleeper import SleeperClient
from ffdraft.scoring import golden

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


if __name__ == "__main__":
    app()
