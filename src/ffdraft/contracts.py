"""Schema contracts — the R1 anti-hallucination lynchpin.

Every module that consumes an external frame calls `assert_columns` immediately
after loading, so a renamed or missing column fails loudly at load time instead
of silently producing wrong numbers at compute time. See CLAUDE.md R1/R2.

Required-column constants live here too (PRD §4) and are added as each loader
that needs them is built.
"""

from __future__ import annotations

import polars as pl


def assert_columns(df: pl.DataFrame, required: set[str], source: str) -> None:
    """Raise if `df` is missing any of `required`, naming both missing and available.

    Fail loudly at load time, never silently at compute time (R1).
    """
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{source} is missing required columns: {sorted(missing)}. "
            f"Available columns: {sorted(df.columns)}"
        )


# --- Required-column constants, verified against the live schema on 2026-08-27 ---
# nflreadpy 0.1.5. Update here (and CLAUDE.md) if a future release renames a column (R2).

# load_ff_playerids() — DynastyProcess crosswalk; the Sleeper<->GSIS seed (PRD §6.1).
FF_PLAYERIDS_REQUIRED = {"gsis_id", "sleeper_id", "name", "merge_name", "position", "team"}

# load_players() — canonical player table keyed on gsis_id.
PLAYERS_REQUIRED = {
    "gsis_id",
    "display_name",
    "first_name",
    "last_name",
    "position",
    "latest_team",
}
