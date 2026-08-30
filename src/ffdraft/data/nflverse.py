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
    CONTRACTS_REQUIRED,
    DEPTH_CHARTS_REQUIRED,
    DRAFT_PICKS_REQUIRED,
    FF_OPPORTUNITY_REQUIRED,
    FF_PLAYERIDS_REQUIRED,
    FF_RANKINGS_REQUIRED,
    PBP_REQUIRED,
    PLAYER_STATS_REQUIRED,
    PLAYERS_REQUIRED,
    ROSTERS_REQUIRED,
    SCHEDULES_REQUIRED,
    SNAP_COUNTS_REQUIRED,
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


def ff_rankings(*, refresh: bool = False, cache_dir: Path = CACHE_DIR) -> pl.DataFrame:
    """FantasyPros expert consensus ranks, full scrape history (PRD §6.1).

    `type="all"` is the only variant carrying past seasons, which the calibration fit
    needs. Note this feed has no projected-points column — see CLAUDE.md's R2 log.
    """
    return _cached(
        "ff_rankings_all",
        lambda: nfl.load_ff_rankings(type="all"),
        FF_RANKINGS_REQUIRED,
        "nflverse.load_ff_rankings",
        refresh=refresh,
        cache_dir=cache_dir,
    )


def snap_counts(
    seasons: list[int], *, refresh: bool = False, cache_dir: Path = CACHE_DIR
) -> pl.DataFrame:
    """Per-game snap counts, 2012+ — the prior-season workload behind M16's tiers."""
    return _cached(
        _season_key("snap_counts", seasons),
        lambda: nfl.load_snap_counts(seasons=seasons),
        SNAP_COUNTS_REQUIRED,
        "nflverse.load_snap_counts",
        refresh=refresh,
        cache_dir=cache_dir,
    )


def rosters(
    seasons: list[int], *, refresh: bool = False, cache_dir: Path = CACHE_DIR
) -> pl.DataFrame:
    """Season rosters — who was actually on each team, for the vacated-opportunity calc.

    One row per player-season despite the `week` column, which holds the last week the
    player appears rather than making this a weekly frame. `status` carries the roster
    code that the M17 edge cases turn on: `RES` is reserve/IR, `DEV` the practice squad.
    """
    return _cached(
        _season_key("rosters", seasons),
        lambda: nfl.load_rosters(seasons=seasons),
        ROSTERS_REQUIRED,
        "nflverse.load_rosters",
        refresh=refresh,
        cache_dir=cache_dir,
    )


def draft_picks(*, refresh: bool = False, cache_dir: Path = CACHE_DIR) -> pl.DataFrame:
    """Draft capital, 1980+. The columns are `round` and `pick` (PRD M17 names them wrong)."""
    return _cached(
        "draft_picks",
        nfl.load_draft_picks,
        DRAFT_PICKS_REQUIRED,
        "nflverse.load_draft_picks",
        refresh=refresh,
        cache_dir=cache_dir,
    )


def contracts(*, refresh: bool = False, cache_dir: Path = CACHE_DIR) -> pl.DataFrame:
    """Player contracts — guaranteed money as a proxy for team intent.

    The guarantee column is `guaranteed`, not `guaranteed_money`. This frame carries no
    season column at all: a contract is placed in time by `year_signed` and `years`, so
    anything season-scoped must derive the season rather than filter on one.
    """
    return _cached(
        "contracts",
        nfl.load_contracts,
        CONTRACTS_REQUIRED,
        "nflverse.load_contracts",
        refresh=refresh,
        cache_dir=cache_dir,
    )


def depth_charts(
    seasons: list[int], *, refresh: bool = False, cache_dir: Path = CACHE_DIR
) -> pl.DataFrame:
    """Positional competition, 2001+. Team is `club_code` here, not `team`."""
    return _cached(
        _season_key("depth_charts", seasons),
        lambda: nfl.load_depth_charts(seasons=seasons),
        DEPTH_CHARTS_REQUIRED,
        "nflverse.load_depth_charts",
        refresh=refresh,
        cache_dir=cache_dir,
    )


def ff_opportunity(
    seasons: list[int], *, refresh: bool = False, cache_dir: Path = CACHE_DIR
) -> pl.DataFrame:
    """Pre-computed expected fantasy points from the ffverse model, 2006+.

    PRD §6.1 is explicit that this is consumed, never rebuilt. `player_id` holds a
    gsis_id, the team is `posteam`, and — alone among nflverse frames — `season` arrives
    as a String and `week` as a Float, so both are cast here to match everything else.
    """

    def load() -> pl.DataFrame:
        frame = nfl.load_ff_opportunity(seasons=seasons)
        return frame.with_columns(
            pl.col("season").cast(pl.Int64), pl.col("week").cast(pl.Int64)
        )

    return _cached(
        _season_key("ff_opportunity", seasons),
        load,
        FF_OPPORTUNITY_REQUIRED,
        "nflverse.load_ff_opportunity",
        refresh=refresh,
        cache_dir=cache_dir,
    )
