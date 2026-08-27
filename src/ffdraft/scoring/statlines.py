"""Raw stat lines the scoring engine consumes.

Nothing here scores anything; it only assembles the *raw* inputs a Sleeper scoring
rule can reference (R1: every source frame is contract-checked on the way in). Two
frames come out:

- `player_week_stats` — nflverse weekly lines plus the three 40+ yard touchdown
  counts, which only exist at play level.
- `defense_week_stats` — one row per team-week with the team-defense event counts,
  points allowed and yards allowed. Team defenses have no `gsis_id` and a different
  stat structure, so they stay on their own path (PRD §11.2).

The rules encoded here were verified week by week against the owner's 2025 league:

- Yards allowed is the opponent's *official net* yardage (gross passing + rushing +
  sack yards lost, the last being negative). Summing play-level `yards_gained`
  overcounts it.
- Points allowed drops 6 points per touchdown the opponent's **defense** scored (the
  fantasy defense was not on the field for it) and 2 per safety, but keeps their
  special-teams return touchdowns, and keeps the extra point in either case.
- Fumble credits split on `special`: a forced fumble or recovery on a special-teams
  play pays the `st_*` key, not the scrimmage one.
"""

from __future__ import annotations

import polars as pl

from ffdraft.contracts import (
    PBP_REQUIRED,
    PLAYER_STATS_REQUIRED,
    SCHEDULES_REQUIRED,
    TEAM_STATS_REQUIRED,
    assert_columns,
)

# nflverse spells the Rams "LA"; Sleeper uses "LAR". Relocated-franchise codes from
# earlier seasons (OAK/SD/STL) are deliberately absent — verify against a season that
# contains them before scoring those years rather than guessing (R2).
TEAM_ALIASES = {"LA": "LAR"}

DEF_COUNT_KEYS = (
    "sack",
    "int",
    "safe",
    "blk_kick",
    "ff",
    "st_ff",
    "fum_rec",
    "st_fum_rec",
    "def_td",
    "st_td",
)


def _norm_team(expr: pl.Expr) -> pl.Expr:
    return expr.replace(TEAM_ALIASES)


def long_td_counts(pbp: pl.DataFrame) -> pl.DataFrame:
    """Per player-week counts of touchdowns scored on plays of 40+ yards.

    Sleeper pays `pass_td_40p`/`rush_td_40p`/`rec_td_40p` off the play's total yardage,
    which the weekly stat lines do not carry.
    """
    assert_columns(pbp, PBP_REQUIRED, "statlines.long_td_counts")
    long = pbp.filter((pl.col("touchdown") == 1) & (pl.col("yards_gained") >= 40))

    def counts(mask: str, id_col: str, name: str) -> pl.DataFrame:
        return (
            long.filter((pl.col(mask) == 1) & pl.col(id_col).is_not_null())
            .group_by(["season", "week", id_col])
            .len(name)
            .rename({id_col: "player_id"})
        )

    out = counts("pass_touchdown", "passer_player_id", "pass_td_40p")
    for frame in (
        counts("rush_touchdown", "rusher_player_id", "rush_td_40p"),
        counts("pass_touchdown", "receiver_player_id", "rec_td_40p"),
    ):
        out = out.join(frame, on=["season", "week", "player_id"], how="full", coalesce=True)
    return out.fill_null(0)


def player_week_stats(
    stats: pl.DataFrame, pbp: pl.DataFrame, *, season_type: str = "REG"
) -> pl.DataFrame:
    """Weekly raw lines for every non-defense player, with the 40+ yard TD counts joined on."""
    assert_columns(stats, PLAYER_STATS_REQUIRED, "statlines.player_week_stats")
    stats = stats.filter(pl.col("season_type") == season_type)
    bonuses = long_td_counts(pbp.filter(pl.col("season_type") == season_type))
    return stats.join(bonuses, on=["season", "week", "player_id"], how="left").with_columns(
        pl.col("pass_td_40p", "rush_td_40p", "rec_td_40p").fill_null(0)
    )


def _team_event_counts(pbp: pl.DataFrame) -> pl.DataFrame:
    """One row per team-week holding every counted team-defense event."""
    idx = ["season", "week", "team"]

    def tally(frame: pl.DataFrame, team: pl.Expr, key: pl.Expr) -> pl.DataFrame:
        return (
            frame.select("season", "week", team.alias("team"), key.alias("key"))
            .filter(pl.col("team").is_not_null())
            .group_by([*idx, "key"])
            .len("n")
        )

    st = pl.col("special") == 1
    parts = [
        # Credited to whoever was on defense for the play.
        tally(pbp.filter(pl.col("sack") == 1), _norm_team(pl.col("defteam")), pl.lit("sack")),
        tally(
            pbp.filter(pl.col("interception") == 1), _norm_team(pl.col("defteam")), pl.lit("int")
        ),
        tally(pbp.filter(pl.col("safety") == 1), _norm_team(pl.col("defteam")), pl.lit("safe")),
        tally(
            pbp.filter(
                (pl.col("punt_blocked") == 1)
                | (pl.col("field_goal_result") == "blocked")
                | (pl.col("extra_point_result") == "blocked")
            ),
            _norm_team(pl.col("defteam")),
            pl.lit("blk_kick"),
        ),
    ]
    for slot in (1, 2):
        forced = pl.col(f"forced_fumble_player_{slot}_team")
        rec, fum = pl.col(f"fumble_recovery_{slot}_team"), pl.col(f"fumbled_{slot}_team")
        # Recovering your own fumble is not a defensive recovery.
        takeaway = rec.is_not_null() & fum.is_not_null() & (rec != fum)
        ff_key = pl.when(st).then(pl.lit("st_ff")).otherwise(pl.lit("ff"))
        parts.append(tally(pbp.filter(forced.is_not_null()), _norm_team(forced), ff_key))
        parts.append(
            tally(
                # Sleeper credits a forced fumble when a ball carrier is stripped even
                # where nflverse names no forcing defender. The three kinds of unforced
                # takeaway it does *not* pay for: a muff or blocked kick on special
                # teams, an aborted snap, and a sack fumble (whose forcer is the sacker).
                # All 14 unforced takeaways of the 2025 season agree with this split.
                pbp.filter(
                    takeaway
                    & forced.is_null()
                    & ~st
                    & (pl.col("aborted_play") != 1)
                    & (pl.col("sack") != 1)
                ),
                _norm_team(rec),
                ff_key,
            )
        )
        parts.append(
            tally(
                pbp.filter(takeaway),
                _norm_team(rec),
                pl.when(st).then(pl.lit("st_fum_rec")).otherwise(pl.lit("fum_rec")),
            )
        )
    # A touchdown belongs to the defense/special-teams unit unless the scoring team's own
    # offense was on the field. On kickoffs `posteam` is the receiving team, so the
    # `special` flag — not possession — separates a return score from an offensive one.
    offensive = (pl.col("td_team") == pl.col("posteam")) & ~st
    parts.append(
        tally(
            pbp.filter((pl.col("touchdown") == 1) & pl.col("td_team").is_not_null() & ~offensive),
            _norm_team(pl.col("td_team")),
            pl.when(st).then(pl.lit("st_td")).otherwise(pl.lit("def_td")),
        )
    )

    wide = (
        pl.concat(parts).pivot(on="key", index=idx, values="n", aggregate_function="sum").fill_null(0)
    )
    missing = [k for k in DEF_COUNT_KEYS if k not in wide.columns]
    return wide.with_columns([pl.lit(0, dtype=pl.UInt32).alias(k) for k in missing])


def defense_week_stats(
    pbp: pl.DataFrame,
    team_stats: pl.DataFrame,
    schedules: pl.DataFrame,
    *,
    season_type: str = "REG",
) -> pl.DataFrame:
    """One row per team-week of raw team-defense stats, keyed on the Sleeper team code."""
    assert_columns(pbp, PBP_REQUIRED, "statlines.defense_week_stats.pbp")
    assert_columns(team_stats, TEAM_STATS_REQUIRED, "statlines.defense_week_stats.team_stats")
    assert_columns(schedules, SCHEDULES_REQUIRED, "statlines.defense_week_stats.schedules")

    counts = _team_event_counts(pbp.filter(pl.col("season_type") == season_type))

    net_yards = team_stats.filter(pl.col("season_type") == season_type).select(
        "season",
        "week",
        _norm_team(pl.col("team")).alias("opponent"),
        (
            pl.col("passing_yards").fill_null(0)
            + pl.col("rushing_yards").fill_null(0)
            + pl.col("sack_yards_lost").fill_null(0)  # already negative
        ).alias("yds_allow"),
    )

    games = schedules.filter(pl.col("home_score").is_not_null())
    sides = pl.concat(
        [
            games.select(
                "season",
                "week",
                _norm_team(pl.col("home_team")).alias("team"),
                _norm_team(pl.col("away_team")).alias("opponent"),
                pl.col("away_score").alias("opp_score"),
            ),
            games.select(
                "season",
                "week",
                _norm_team(pl.col("away_team")).alias("team"),
                _norm_team(pl.col("home_team")).alias("opponent"),
                pl.col("home_score").alias("opp_score"),
            ),
        ]
    )

    opp_def_pts = counts.select(
        "season",
        "week",
        pl.col("team").alias("opponent"),
        (pl.col("def_td") * 6 + pl.col("safe") * 2).alias("opp_def_points"),
    )

    return (
        sides.join(counts, on=["season", "week", "team"], how="left")
        .join(opp_def_pts, on=["season", "week", "opponent"], how="left")
        .join(net_yards, on=["season", "week", "opponent"], how="left")
        .with_columns(
            [pl.col(k).fill_null(0) for k in (*DEF_COUNT_KEYS, "opp_def_points", "yds_allow")]
        )
        .with_columns((pl.col("opp_score") - pl.col("opp_def_points")).alias("pts_allow"))
        .drop("opp_def_points")
    )
