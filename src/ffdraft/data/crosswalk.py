"""M5 — player crosswalk. Canonical identity is `gsis_id` (PRD §7.1/§7.2).

Builds the crosswalk universe from the DynastyProcess ff_playerids table, and
resolves external source rows (ADP, projections) to a gsis_id using an ordered
match strategy that never matches below the fuzzy threshold silently (R4).
Team defenses use synthetic `DEF_{TEAM}` ids on a separate path.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from rapidfuzz import fuzz, process

from ffdraft.contracts import assert_columns

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
_APOSTROPHES = ("'", "’")

# Position taxonomy: DynastyProcess uses "PK" for kickers; Sleeper/FFC use "K".
# Normalize to the fantasy convention so a kicker matches across sources.
_POSITION_ALIASES = {"PK": "K"}


def normalize_name(name: str) -> str:
    """Normalize a player name per PRD §7.2, in this exact order."""
    s = name.lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))  # drop accents
    s = s.replace(".", "")
    for apo in _APOSTROPHES:
        s = s.replace(apo, "")
    tokens = [t for t in re.split(r"\s+", s) if t and t not in _SUFFIXES]
    return " ".join(tokens)


@dataclass(frozen=True)
class Match:
    gsis_id: str
    method: str  # dynastyprocess | exact | fuzzy | override | def
    score: float | None


def build_crosswalk(ff_playerids: pl.DataFrame) -> pl.DataFrame:
    """Crosswalk universe seeded from DynastyProcess (PRD §7.2 match order step 1)."""
    assert_columns(
        ff_playerids,
        {"gsis_id", "sleeper_id", "name", "position", "team"},
        "crosswalk.ff_playerids",
    )
    df = ff_playerids.filter(pl.col("gsis_id").is_not_null()).select(
        pl.col("gsis_id"),
        pl.col("sleeper_id"),
        pl.col("name").alias("full_name"),
        pl.col("name")
        .map_elements(normalize_name, return_dtype=pl.String)
        .alias("normalized_name"),
        pl.col("position").replace(_POSITION_ALIASES).alias("position"),
        pl.col("team"),
        pl.lit("dynastyprocess").alias("match_method"),
    )
    return df


class _Index:
    """Lookup structures over the crosswalk for the ordered match strategy."""

    def __init__(self, crosswalk: pl.DataFrame) -> None:
        self.by_npt: dict[tuple[str, str, str], str] = {}
        self.by_np: dict[tuple[str, str], str] = {}
        self.by_pos: dict[str, list[tuple[str, str]]] = {}
        for row in crosswalk.iter_rows(named=True):
            nm, pos, team, gsis = (
                row["normalized_name"],
                row["position"],
                row["team"],
                row["gsis_id"],
            )
            if nm is None or pos is None:
                continue
            if team is not None:
                self.by_npt.setdefault((nm, pos, team), gsis)
            self.by_np.setdefault((nm, pos), gsis)
            self.by_pos.setdefault(pos, []).append((nm, gsis))


def resolve_one(
    name: str,
    position: str,
    team: str | None,
    index: _Index,
    *,
    fuzzy_threshold: int,
    overrides: dict[str, str] | None = None,
) -> Match | None:
    """Resolve one external row to a gsis_id, or None (never a silent bad match)."""
    nm = normalize_name(name)

    # Overrides win over every method (PRD §7.2).
    if overrides:
        for key in (name, nm):
            if key in overrides:
                return Match(overrides[key], "override", None)

    # Team defenses: synthetic id, separate path — never name/fuzzy matched.
    if position == "DEF":
        if team is None:
            return None
        return Match(f"DEF_{team}", "def", None)

    if team is not None and (nm, position, team) in index.by_npt:
        return Match(index.by_npt[(nm, position, team)], "exact", 100.0)
    if (nm, position) in index.by_np:
        return Match(index.by_np[(nm, position)], "exact", 100.0)

    candidates = index.by_pos.get(position, [])
    if candidates:
        choices = [c[0] for c in candidates]
        best = process.extractOne(nm, choices, scorer=fuzz.WRatio)
        if best is not None and best[1] >= fuzzy_threshold:
            return Match(candidates[best[2]][1], "fuzzy", float(best[1]))

    return None


def resolve_frame(
    queries: pl.DataFrame,
    crosswalk: pl.DataFrame,
    *,
    fuzzy_threshold: int,
    overrides: dict[str, str] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Resolve a frame of {name, position, team[, rank]} rows.

    Returns (matched, unmatched). Unmatched rows are reported, never dropped (R4).
    """
    assert_columns(queries, {"name", "position"}, "crosswalk.resolve_frame")
    index = _Index(crosswalk)
    has_team = "team" in queries.columns

    matched_rows, unmatched_rows = [], []
    for row in queries.iter_rows(named=True):
        team = row["team"] if has_team else None
        m = resolve_one(
            row["name"], row["position"], team, index,
            fuzzy_threshold=fuzzy_threshold, overrides=overrides,
        )
        if m is None:
            unmatched_rows.append(row)
        else:
            matched_rows.append({**row, "gsis_id": m.gsis_id,
                                 "match_method": m.method, "match_score": m.score})

    matched = pl.DataFrame(matched_rows) if matched_rows else queries.head(0)
    unmatched = pl.DataFrame(unmatched_rows) if unmatched_rows else queries.head(0)
    return matched, unmatched


def write_unmatched_report(unmatched: pl.DataFrame, path: str | Path) -> None:
    """Write the unmatched report (R4). Callers also warn at startup."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    unmatched.write_csv(p)


def load_id_overrides(path: str | Path) -> dict[str, str]:
    """Load name->gsis_id overrides that win over all match methods (§7.2).

    Returns an empty mapping when the file is absent or has only a header, so a
    fresh checkout with no hand-maintained overrides works out of the box.
    """
    p = Path(path)
    if not p.exists():
        return {}
    df = pl.read_csv(p)
    if df.height == 0 or not {"name", "gsis_id"} <= set(df.columns):
        return {}
    return {
        n: g
        for n, g in zip(df["name"].to_list(), df["gsis_id"].to_list(), strict=True)
        if g is not None
    }
