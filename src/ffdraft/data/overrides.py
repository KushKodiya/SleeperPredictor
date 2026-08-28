"""M3a — the draft-morning override path.

A starter gets ruled out ninety minutes before the draft. This is how the owner acts on
that without editing code: a hand-edited CSV at `overrides/projection_overrides.csv`.

Three properties matter more than the module's size:

- **Applied after calibration.** A manual number is a statement of fact, not a
  projection to shrink, so it is taken at face value and never passed back through the
  fit.
- **`games_played` is recorded and shown, not arithmetic.** The projections are fit on
  actual season totals, which already carry the games those players missed, so scaling
  one by `games_played / 17` would discount availability twice by an unknown amount.
  Until Phase 5's availability model can say how many games a projection already
  assumes, the override rides along as a visible note and the number is left alone.
- **Unique resolution or nothing.** A `player_name` matching two players fails the load
  rather than guessing which one the owner meant.
- **All or nothing.** One malformed row fails the whole file, with every offending line
  named. A half-applied override file is worse than none — you would not know which half.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from ffdraft.contracts import assert_columns
from ffdraft.data.crosswalk import normalize_name

REQUIRED_COLUMNS = {"field", "value", "reason"}
PROJECTED_POINTS = "projected_points"
EXCLUDE = "exclude"
GAMES_PLAYED = "games_played"
SUPPORTED_FIELDS = (PROJECTED_POINTS, GAMES_PLAYED, EXCLUDE)

_TRUE = {"true", "yes", "1", "t", "y"}
_FALSE = {"false", "no", "0", "f", "n"}


@dataclass(frozen=True)
class Override:
    gsis_id: str
    field: str
    value: float | bool
    reason: str
    line: int
    applied_at: datetime


class OverrideError(ValueError):
    """Raised when an override file cannot be applied in full."""


def _name_index(crosswalk: pl.DataFrame) -> dict[str, list[str]]:
    """Normalized name -> every gsis_id carrying it, so ambiguity is detectable."""
    index: dict[str, list[str]] = {}
    for name, gsis in zip(
        crosswalk["normalized_name"].to_list(), crosswalk["gsis_id"].to_list(), strict=True
    ):
        if name and gsis and gsis not in index.setdefault(name, []):
            index[name].append(gsis)
    return index


def _parse_value(field: str, raw: object, line: int) -> float | bool:
    text = str(raw).strip()
    if field == GAMES_PLAYED:
        try:
            games = float(text)
        except ValueError:
            raise OverrideError(f"line {line}: games_played {text!r} is not a number") from None
        if games < 0:
            raise OverrideError(f"line {line}: games_played {text!r} cannot be negative")
        return games
    if field == EXCLUDE:
        lowered = text.lower()
        if lowered in _TRUE:
            return True
        if lowered in _FALSE:
            return False
        raise OverrideError(f"line {line}: exclude value {text!r} is not true or false")
    try:
        return float(text)
    except ValueError:
        raise OverrideError(f"line {line}: {field} value {text!r} is not a number") from None


def _resolve(row: dict, index: dict[str, list[str]], known: set[str], line: int) -> str:
    gsis = str(row.get("gsis_id") or "").strip()
    name = str(row.get("player_name") or "").strip()
    if gsis:
        if known and gsis not in known:
            raise OverrideError(f"line {line}: gsis_id {gsis!r} is not in the crosswalk")
        return gsis
    if not name:
        raise OverrideError(f"line {line}: needs either a gsis_id or a player_name")

    candidates = index.get(normalize_name(name), [])
    if not candidates:
        raise OverrideError(f"line {line}: player_name {name!r} matches no player")
    if len(candidates) > 1:
        raise OverrideError(
            f"line {line}: player_name {name!r} is ambiguous, matching {len(candidates)} "
            f"players ({', '.join(sorted(candidates))}). Use gsis_id instead."
        )
    return candidates[0]


def load_overrides(
    path: str | Path, crosswalk: pl.DataFrame, *, now: datetime | None = None
) -> list[Override]:
    """Read and fully validate the override file, or raise naming every bad line.

    An absent file is normal — most drafts need no overrides — and yields no overrides.
    """
    p = Path(path)
    if not p.exists():
        return []
    frame = pl.read_csv(p, infer_schema_length=0)  # every column as text; we parse per field
    if frame.is_empty():
        return []
    assert_columns(frame, REQUIRED_COLUMNS, f"overrides[{p.name}]")
    if not {"gsis_id", "player_name"} & set(frame.columns):
        raise OverrideError(f"{p.name} needs a gsis_id or player_name column to identify players")

    assert_columns(crosswalk, {"gsis_id", "normalized_name"}, "overrides.crosswalk")
    index = _name_index(crosswalk)
    known = set(crosswalk["gsis_id"].to_list())
    stamp = now or datetime.now(UTC)

    overrides, problems = [], []
    for offset, row in enumerate(frame.iter_rows(named=True)):
        line = offset + 2  # header occupies line 1
        try:
            field = str(row["field"] or "").strip()
            if field not in SUPPORTED_FIELDS:
                raise OverrideError(
                    f"line {line}: unknown field {field!r}; expected one of "
                    f"{', '.join(SUPPORTED_FIELDS)}"
                )
            reason = str(row["reason"] or "").strip()
            if not reason:
                raise OverrideError(f"line {line}: reason is required, an unexplained override is a trap")
            overrides.append(
                Override(
                    gsis_id=_resolve(row, index, known, line),
                    field=field,
                    value=_parse_value(field, row["value"], line),
                    reason=reason,
                    line=line,
                    applied_at=stamp,
                )
            )
        except OverrideError as exc:
            problems.append(str(exc))

    if problems:
        raise OverrideError(
            f"{p.name} has {len(problems)} bad row(s); no overrides were applied:\n  "
            + "\n  ".join(problems)
        )
    return overrides


def apply_overrides(
    board: pl.DataFrame, overrides: list[Override]
) -> tuple[pl.DataFrame, list[Override]]:
    """Apply overrides to a calibrated board, after aggregation and calibration.

    Returns (board, unmatched). `unmatched` are overrides naming a player the board does
    not carry — surfaced rather than silently doing nothing, since an override that
    quietly fails is exactly the trap this module exists to avoid.
    """
    assert_columns(board, {"gsis_id", PROJECTED_POINTS}, "overrides.apply_overrides")
    if not overrides:
        return board.with_columns(
            pl.lit(None, dtype=pl.String).alias("override_reason"),
            pl.lit(None, dtype=pl.Float64).alias("override_games"),
        ), []

    present = set(board["gsis_id"].to_list())
    unmatched = [o for o in overrides if o.gsis_id not in present]
    active = [o for o in overrides if o.gsis_id in present]

    excluded = {o.gsis_id for o in active if o.field == EXCLUDE and o.value is True}
    points = {o.gsis_id: float(o.value) for o in active if o.field == PROJECTED_POINTS}
    games = {o.gsis_id: float(o.value) for o in active if o.field == GAMES_PLAYED}

    # A player can carry more than one override row; every reason has to reach the board,
    # because the one that is hidden is the one that misleads.
    collected: dict[str, list[str]] = {}
    for override in active:
        collected.setdefault(override.gsis_id, []).append(override.reason)
    reasons = {gsis: "; ".join(dict.fromkeys(rs)) for gsis, rs in collected.items()}

    out = board.with_columns(
        pl.col("gsis_id").replace_strict(points, default=None).alias("_override_points"),
        pl.col("gsis_id").replace_strict(games, default=None).alias("override_games"),
        pl.col("gsis_id").replace_strict(reasons, default=None).alias("override_reason"),
    )
    return (
        out.with_columns(
            # Taken at face value: the manual number replaces the calibrated one outright.
            pl.coalesce("_override_points", PROJECTED_POINTS).alias(PROJECTED_POINTS)
        )
        .drop("_override_points")
        .filter(~pl.col("gsis_id").is_in(list(excluded)))
    ), unmatched
