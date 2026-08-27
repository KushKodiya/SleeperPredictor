"""M1 — thin cached wrapper over nflreadpy.

Every loader caches to `data/raw/nflverse/{key}.parquet` and calls `assert_columns`
before returning, so column drift fails loudly at load time (R1). `refresh=True`
bypasses the cache. All functions return `polars.DataFrame` (never pandas).

Only the loaders Phase 1 needs are wrapped here; later phases add more using the
same `_cached` mechanism. Column names are verified against the live schema, not
assumed (R1/R2).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import nflreadpy as nfl
import polars as pl

from ffdraft.contracts import (
    FF_PLAYERIDS_REQUIRED,
    PBP_REQUIRED,
    PLAYER_STATS_REQUIRED,
    PLAYERS_REQUIRED,
    SCHEDULES_REQUIRED,
    TEAM_STATS_REQUIRED,
    assert_columns,
)

CACHE_DIR = Path("data/raw/nflverse")


def _cached(
    key: str,
    loader: Callable[[], pl.DataFrame],
    required: set[str],
    source: str,
    *,
    refresh: bool = False,
    cache_dir: Path = CACHE_DIR,
) -> pl.DataFrame:
    """Return a cached frame, loading + writing it on a miss or `refresh`.

    `assert_columns` runs on every return path, so a cached frame with a stale
    schema fails just as loudly as a fresh one.
    """
    path = cache_dir / f"{key}.parquet"
    if path.exists() and not refresh:
        df = pl.read_parquet(path)
    else:
        df = loader()
        path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(path)
    assert_columns(df, required, source)
    return df


def ff_playerids(*, refresh: bool = False, cache_dir: Path = CACHE_DIR) -> pl.DataFrame:
    """DynastyProcess crosswalk with Sleeper<->GSIS mapping (PRD §6.1)."""
    return _cached(
        "ff_playerids",
        nfl.load_ff_playerids,
        FF_PLAYERIDS_REQUIRED,
        "nflverse.load_ff_playerids",
        refresh=refresh,
        cache_dir=cache_dir,
    )


def players(*, refresh: bool = False, cache_dir: Path = CACHE_DIR) -> pl.DataFrame:
    """Canonical player table keyed on gsis_id (PRD §6.1)."""
    return _cached(
        "players",
        nfl.load_players,
        PLAYERS_REQUIRED,
        "nflverse.load_players",
        refresh=refresh,
        cache_dir=cache_dir,
    )


def _season_key(name: str, seasons: list[int]) -> str:
    return f"{name}_{'_'.join(str(s) for s in sorted(seasons))}"


def player_stats(
    seasons: list[int], *, refresh: bool = False, cache_dir: Path = CACHE_DIR
) -> pl.DataFrame:
    """Weekly raw stat lines — the only input the scoring engine scores (PRD §8 M6)."""
    return _cached(
        _season_key("player_stats", seasons),
        lambda: nfl.load_player_stats(seasons=seasons, summary_level="week"),
        PLAYER_STATS_REQUIRED,
        "nflverse.load_player_stats",
        refresh=refresh,
        cache_dir=cache_dir,
    )


def team_stats(
    seasons: list[int], *, refresh: bool = False, cache_dir: Path = CACHE_DIR
) -> pl.DataFrame:
    """Weekly team totals; supplies the opponent yardage behind the yds_allow tiers."""
    return _cached(
        _season_key("team_stats", seasons),
        lambda: nfl.load_team_stats(seasons=seasons, summary_level="week"),
        TEAM_STATS_REQUIRED,
        "nflverse.load_team_stats",
        refresh=refresh,
        cache_dir=cache_dir,
    )


def schedules(
    seasons: list[int], *, refresh: bool = False, cache_dir: Path = CACHE_DIR
) -> pl.DataFrame:
    """Game results; supplies final scores behind the pts_allow tiers."""
    return _cached(
        _season_key("schedules", seasons),
        lambda: nfl.load_schedules(seasons=seasons),
        SCHEDULES_REQUIRED,
        "nflverse.load_schedules",
        refresh=refresh,
        cache_dir=cache_dir,
    )


def pbp(seasons: list[int], *, refresh: bool = False, cache_dir: Path = CACHE_DIR) -> pl.DataFrame:
    """Play-by-play. Only the columns in `PBP_REQUIRED` are used, but the loader is all-or-nothing."""
    return _cached(
        _season_key("pbp", seasons),
        lambda: nfl.load_pbp(seasons=seasons),
        PBP_REQUIRED,
        "nflverse.load_pbp",
        refresh=refresh,
        cache_dir=cache_dir,
    )
