"""M3 — Fantasy Football Calculator ADP.

Fetches and caches FFC ADP for the format implied by the league's scoring, then
joins it to the crosswalk (reusing M5), reporting ADP players that do not
resolve rather than dropping them (R4).
"""

from __future__ import annotations

import time
from pathlib import Path

import httpx
import polars as pl

from ffdraft.contracts import assert_columns
from ffdraft.data.crosswalk import resolve_frame

FFC_BASE = "https://fantasyfootballcalculator.com/api/v1/adp"
_ADP_REQUIRED = {"name", "position", "team", "adp"}


def adp_format_from_scoring(scoring: dict) -> str:
    """Map league PPR scoring to an FFC format (PRD §6.3 / M3)."""
    rec = scoring.get("rec", 0.0) or 0.0
    if rec >= 1.0:
        return "ppr"
    if rec >= 0.5:
        return "half-ppr"
    return "standard"


def _get(fmt: str, teams: int, year: int, timeout: float, client: httpx.Client | None) -> dict:
    c = client or httpx.Client(timeout=timeout)
    try:
        resp = c.get(f"{FFC_BASE}/{fmt}", params={"teams": teams, "year": year})
        resp.raise_for_status()
        return resp.json()
    finally:
        if client is None:
            c.close()


def fetch_adp(
    fmt: str,
    teams: int,
    year: int,
    *,
    cache_dir: str | Path = "data/raw/adp",
    ttl_hours: float = 12.0,
    timeout: float = 15.0,
    client: httpx.Client | None = None,
) -> pl.DataFrame:
    """Return FFC ADP, cached to parquet and refreshed at most once per TTL."""
    cache = Path(cache_dir) / f"adp_{fmt}_{teams}_{year}.parquet"
    if cache.exists() and (time.time() - cache.stat().st_mtime) / 3600.0 < ttl_hours:
        return pl.read_parquet(cache)

    data = _get(fmt, teams, year, timeout, client)
    df = pl.DataFrame(data["players"])
    assert_columns(df, _ADP_REQUIRED, "ffc.adp")
    df = (
        df.with_columns(pl.col("position").replace({"PK": "K"}))  # FFC kickers are "PK"
        .sort("adp")
        .with_row_index("rank", offset=1)
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(cache)
    return df


def join_adp_to_crosswalk(
    adp_df: pl.DataFrame,
    crosswalk: pl.DataFrame,
    *,
    fuzzy_threshold: int,
    overrides: dict[str, str] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Resolve ADP rows to gsis_id; return (matched, unmatched). Unmatched reported (R4)."""
    # `high` is the earliest pick at which the market actually took the player. The QB
    # sanity gate needs it: a reach is only a reach relative to what real drafters did.
    keep = [
        c for c in ("name", "position", "team", "rank", "adp", "high") if c in adp_df.columns
    ]
    return resolve_frame(
        adp_df.select(keep), crosswalk, fuzzy_threshold=fuzzy_threshold, overrides=overrides
    )
