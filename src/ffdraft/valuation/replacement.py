"""M8 — replacement level from the waiver wire.

Neither zero nor the last starter: both misrank scarcity. Zero treats a startable
free-agent tight end as worth nothing; last-starter assumes you would field an empty
slot rather than stream. What you would actually do is start the best player still
available, so that is what a player is worth *more than*.

Replacement level here is the season production of the best free agent at each
position, taken at the configured percentile across the lookback.

It is measured as a **season total**, not as a median week scaled up. Both sides of
VOR have to be in the same unit: projections are season totals that already carry the
games their player missed, so pricing them against a replacement who never gets hurt
overstates replacement most at the positions that miss the most time.
"""

from __future__ import annotations

from collections import Counter

import polars as pl

from ffdraft.contracts import assert_columns

WEEKLY_REQUIRED = {"season", "week", "gsis_id", "position", "points"}


def starters_per_team(
    roster_positions: list[str], flex_eligibility: dict[str, list[str]]
) -> dict[str, float]:
    """Starting slots each team fields at each position, with flex spread across it.

    A flex slot is shared, so it is allocated to the eligible positions in proportion to
    their dedicated slots: a lineup with RB2/WR2/TE1 puts 40% of each flex on RB, 40% on
    WR and 20% on TE. Bench slots are not counted.
    """
    counts = Counter(p for p in roster_positions if p != "BN")
    dedicated = {p: float(n) for p, n in counts.items() if p not in flex_eligibility}

    for flex, eligible in flex_eligibility.items():
        slots = counts.get(flex, 0)
        if not slots:
            continue
        weights = {p: dedicated.get(p, 0.0) for p in eligible}
        total = sum(weights.values())
        # With no dedicated slots to weight by, split the flex evenly among the eligible.
        share = {p: (w / total if total else 1.0 / len(eligible)) for p, w in weights.items()}
        for position, fraction in share.items():
            dedicated[position] = dedicated.get(position, 0.0) + slots * fraction
    return dedicated


def rostered_depth(
    roster_positions: list[str], flex_eligibility: dict[str, list[str]], *, teams: int
) -> dict[str, int]:
    """How many players at each position are off the waiver wire league-wide.

    Bench slots count: a player stashed on someone's bench is not available to you, and
    ignoring them makes replacement look far stronger than it is. Benches are attributed
    to the flex-eligible positions in proportion to their starting demand — that is where
    depth actually gets hoarded, since nobody carries a backup kicker.

    ponytail: proportional to starting demand, not to observed hoarding. Swap in the real
    positional split from `load_rosters` if the late rounds ever look wrong.
    """
    counts = Counter(roster_positions)
    starters = starters_per_team(roster_positions, flex_eligibility)
    bench = float(counts.get("BN", 0))
    # Only the flex slots this league actually fields make a position hoardable. The
    # config defines SUPER_FLEX (QB-eligible) among others, but a roster with no such
    # slot gives nobody a reason to carry a second quarterback.
    hoardable = sorted(
        {p for flex, eligible in flex_eligibility.items() if counts.get(flex, 0) for p in eligible}
    )

    demand = {p: starters.get(p, 0.0) for p in hoardable}
    total = sum(demand.values())
    depth = dict(starters)
    if bench and total:
        for position, share in demand.items():
            depth[position] = depth.get(position, 0.0) + bench * (share / total)
    return {position: max(1, round(slots * teams)) for position, slots in depth.items()}


def replacement_levels(
    weekly: pl.DataFrame,
    *,
    depth: dict[str, int],
    percentile: float,
    games_per_season: int,
    lookback_seasons: int | None = None,
) -> pl.DataFrame:
    """Seasonal replacement level per position, with the weekly rate reported alongside.

    The best available free agent is the player who finished just past `depth[position]`
    in a season; `percentile` (0.5 = median) is taken across the seasons in the lookback.
    """
    assert_columns(weekly, WEEKLY_REQUIRED, "replacement.replacement_levels")
    frame = weekly.filter(pl.col("points").is_not_null())
    if lookback_seasons is not None:
        newest = frame.select(pl.col("season").max()).item()
        frame = frame.filter(pl.col("season") > newest - lookback_seasons)

    rows = []
    for position, roster_count in sorted(depth.items()):
        at_position = frame.filter(pl.col("position") == position)
        if at_position.is_empty():
            continue

        # The replacement player is identified by how he finished the season, not by
        # whoever happened to explode in a given week. Taking the weekly maximum among
        # free agents would price in perfect hindsight streaming and put replacement
        # above most starters.
        totals = (
            at_position.group_by(["season", "gsis_id"])
            .agg(pl.col("points").sum().alias("season_points"))
            # `gsis_id` makes the ordering total. Two players tied on season points at
            # exactly the boundary rank would otherwise be separated by row order, and
            # whichever one landed there sets the whole position's replacement level (R7).
            .sort(["season", "season_points", "gsis_id"], descending=[False, True, False])
            .with_columns(pl.int_range(pl.len()).over("season").alias("position_rank"))
            .filter(pl.col("position_rank") == roster_count)
        )
        if totals.is_empty():
            continue

        # Measured as a season total, because that is the unit VOR subtracts it from.
        # Taking his median week and multiplying by 17 would price a replacement who
        # never gets hurt against projections that already carry missed games — worth
        # ~16 points at running back, where the replacement-level player misses most.
        level = totals["season_points"].quantile(percentile)
        if level is None:
            continue
        weekly_level = level / games_per_season
        rows.append(
            {
                "position": position,
                "rostered": roster_count,
                "weekly_replacement": weekly_level,
                "replacement_points": weekly_level * games_per_season,
            }
        )
    if not rows:
        raise ValueError("no position had enough weekly rows to establish replacement level")
    return pl.DataFrame(rows)
