"""M17 — situation-change features, vacated opportunity above all.

The consensus handles roster turnover by hand. A receiver inheriting 28% of a team's
vacated target share is a different asset from one inheriting 4%, and nothing upstream of
this module knows the difference. The derivation is purely mechanical, which is exactly
why it is worth automating properly.

Everything here matches on `gsis_id` and never on name (R4), and every external frame
passes a column contract before it is read (R1).
"""

from __future__ import annotations

import polars as pl

from ffdraft.contracts import (
    CONTRACTS_REQUIRED,
    DRAFT_PICKS_REQUIRED,
    ROSTERS_REQUIRED,
    SNAP_COUNTS_REQUIRED,
    assert_columns,
)

# Only what this module actually reads. Demanding the scoring engine's full column set
# would reject a frame that is perfectly adequate for counting opportunity.
OPPORTUNITY_REQUIRED = {"season", "player_id", "team", "targets", "carries"}

# Roster statuses that count as still holding a place on the team.
#
# `RES` (reserve/IR) counts as **present**: a player who spends the season on injured
# reserve still occupies his role, and treating his prior workload as vacated would
# invent opportunity for teammates that nobody actually gave up. `INA` (game-day
# inactive) is present for the same reason.
#
# `DEV` is the practice squad and is deliberately **excluded** — a practice-squad player
# is not holding last season's snaps. `CUT`, `RET` and the trade codes are departures on
# their face.
PRESENT_STATUSES = frozenset({"ACT", "RES", "INA"})

# (raw column, share column) — the two kinds of opportunity a departure can vacate.
OPPORTUNITIES = (("targets", "target_share"), ("carries", "carry_share"))


def opportunity_shares(player_stats: pl.DataFrame, *, season: int) -> pl.DataFrame:
    """Per `(team, gsis_id)` for one season: targets, carries, and each as a team share.

    A player who changed teams mid-season appears once per team, holding the share he
    actually earned at each — which is what a team losing him should count as vacated.
    """
    assert_columns(player_stats, OPPORTUNITY_REQUIRED, "situation.opportunity_shares")
    totals = (
        player_stats.filter(pl.col("season") == season)
        .group_by("team", "player_id")
        .agg(
            pl.col("targets").fill_null(0).sum().alias("targets"),
            pl.col("carries").fill_null(0).sum().alias("carries"),
        )
    )
    if totals.is_empty():
        return totals.with_columns(
            pl.lit(0.0).alias("target_share"), pl.lit(0.0).alias("carry_share")
        )

    # A team with no opportunity of one kind gets a 0.0 share, never a null or a NaN.
    return totals.with_columns(
        [
            pl.when(pl.col(column).sum().over("team") > 0)
            .then(pl.col(column) / pl.col(column).sum().over("team"))
            .otherwise(0.0)
            .alias(share)
            for column, share in OPPORTUNITIES
        ]
    )


def _present(rosters: pl.DataFrame, *, season: int) -> pl.DataFrame:
    """`(team, player_id)` pairs still holding a place on that team in `season`."""
    assert_columns(rosters, ROSTERS_REQUIRED, "situation._present")
    return (
        rosters.filter(
            (pl.col("season") == season)
            & pl.col("status").is_in(list(PRESENT_STATUSES))
            & pl.col("gsis_id").is_not_null()
        )
        .select(pl.col("team"), pl.col("gsis_id").alias("player_id"))
        .unique()
    )


def vacated_opportunity(
    player_stats: pl.DataFrame, rosters: pl.DataFrame, *, prior_season: int
) -> pl.DataFrame:
    """Per team: the share of `prior_season` opportunity held by players who have left.

    Departure means "held opportunity for this team last season and does not hold a place
    on it now". A player who moved to another team is a departure *here* and an arrival
    there; `arrivals` reports the other side.

    Every team that had opportunity in `prior_season` appears in the result. A team that
    lost nobody gets exactly `0.0`, never null — a null would silently propagate into the
    model as a missing feature rather than as the true "nothing was vacated".
    """
    shares = opportunity_shares(player_stats, season=prior_season)
    present = _present(rosters, season=prior_season + 1)

    departed = shares.join(present, on=["team", "player_id"], how="anti")
    vacated = departed.group_by("team").agg(
        pl.col("target_share").sum().alias("vacated_target_share"),
        pl.col("carry_share").sum().alias("vacated_carry_share"),
    )
    # Left-join off the full team list so a team with no departures is 0.0, not absent.
    return (
        shares.select("team")
        .unique()
        .join(vacated, on="team", how="left")
        .with_columns(
            pl.col("vacated_target_share").fill_null(0.0),
            pl.col("vacated_carry_share").fill_null(0.0),
        )
        .with_columns(pl.lit(prior_season + 1).alias("season"))
        .sort("team")
    )


def arrivals(
    player_stats: pl.DataFrame, rosters: pl.DataFrame, *, prior_season: int
) -> pl.DataFrame:
    """Per `(team, player_id)`: who is on the team now but held no opportunity there before.

    This is the other half of a team change, and it is where rookies count. A rookie
    vacates nothing — he had no prior share to give up — but he is arriving competition
    for the opportunity somebody else vacated, which is the whole point of the feature.
    """
    prior = opportunity_shares(player_stats, season=prior_season).select("team", "player_id")
    return (
        _present(rosters, season=prior_season + 1)
        .join(prior, on=["team", "player_id"], how="anti")
        .with_columns(pl.lit(prior_season + 1).alias("season"))
        .sort("team", "player_id")
    )


# --- team context and player capital -------------------------------------------------

SCHEDULE_LINES_REQUIRED = {
    "season", "week", "game_type", "home_team", "away_team",
    "spread_line", "total_line", "home_coach", "away_coach",
}


def _team_games(schedules: pl.DataFrame, *, season: int) -> pl.DataFrame:
    """One row per team-game with that team's implied total and its coach.

    `spread_line` is positive when the home team is favoured (PRD §6.1), so the home
    side gets `(total + spread) / 2` and the away side `(total - spread) / 2`.
    """
    assert_columns(schedules, SCHEDULE_LINES_REQUIRED, "situation._team_games")
    games = schedules.filter((pl.col("season") == season) & (pl.col("game_type") == "REG"))
    home = games.select(
        pl.col("home_team").alias("team"),
        ((pl.col("total_line") + pl.col("spread_line")) / 2).alias("implied_total"),
        pl.col("home_coach").alias("coach"),
    )
    away = games.select(
        pl.col("away_team").alias("team"),
        ((pl.col("total_line") - pl.col("spread_line")) / 2).alias("implied_total"),
        pl.col("away_coach").alias("coach"),
    )
    return pl.concat([home, away])


def team_implied_total(
    schedules: pl.DataFrame, *, season: int, fallback_season: int | None = None
) -> pl.DataFrame:
    """Per team: mean implied total over the regular season, with an explicit null flag.

    Preseason rows carry no line at all — `spread_line` and `total_line` are null until
    the market posts them, which during draft season is most of the schedule (PRD §11.11).
    Imputing zero would silently claim a team is projected to score nothing, so a team
    with no posted lines falls back to `fallback_season`'s mean and is flagged.

    `lines_posted` is the count of games that actually had a line, so a consumer can tell
    a fully-priced season from one resting on a single posted game.
    """
    current = _team_games(schedules, season=season).drop("coach")
    priced = current.filter(pl.col("implied_total").is_not_null())

    per_team = priced.group_by("team").agg(
        pl.col("implied_total").mean().alias("team_implied_total"),
        pl.len().alias("lines_posted"),
    )
    teams = current.select("team").unique()
    joined = teams.join(per_team, on="team", how="left").with_columns(
        pl.col("lines_posted").fill_null(0)
    )

    if fallback_season is not None:
        prior = _team_games(schedules, season=fallback_season).drop("coach")
        prior_mean = prior["implied_total"].mean()
        joined = joined.with_columns(
            pl.col("team_implied_total").fill_null(prior_mean).alias("team_implied_total")
        )

    return joined.with_columns(
        (pl.col("lines_posted") == 0).alias("implied_total_is_fallback")
    ).sort("team")


def head_coach_change(schedules: pl.DataFrame, *, season: int) -> pl.DataFrame:
    """Per team: whether the head coach differs from the prior season's.

    A team with no prior season on record is reported with a null flag rather than
    `False` — "we do not know" is not the same claim as "the coach stayed".
    """
    now = (
        _team_games(schedules, season=season)
        .group_by("team")
        .agg(pl.col("coach").drop_nulls().first().alias("coach"))
    )
    before = (
        _team_games(schedules, season=season - 1)
        .group_by("team")
        .agg(pl.col("coach").drop_nulls().first().alias("prior_coach"))
    )
    return (
        now.join(before, on="team", how="left")
        .with_columns(
            pl.when(pl.col("prior_coach").is_null())
            .then(None)
            .otherwise(pl.col("coach") != pl.col("prior_coach"))
            .alias("head_coach_change")
        )
        .select("team", "coach", "prior_coach", "head_coach_change")
        .sort("team")
    )


def qb_change(
    snap_counts: pl.DataFrame,
    rosters: pl.DataFrame,
    crosswalk_ids: pl.DataFrame,
    *,
    prior_season: int,
) -> pl.DataFrame:
    """Per team: whether last season's QB snap leader is gone.

    Snap counts are keyed on `pfr_player_id`, not `gsis_id`, so the leader is resolved
    through the crosswalk before being looked for on this season's roster (R4 — an
    unresolvable id is reported as unknown, never silently treated as departed).
    """
    assert_columns(snap_counts, SNAP_COUNTS_REQUIRED, "situation.qb_change")
    assert_columns(crosswalk_ids, {"pfr_id", "gsis_id"}, "situation.qb_change")

    leaders = (
        snap_counts.filter(
            (pl.col("season") == prior_season)
            & (pl.col("position") == "QB")
            & (pl.col("game_type") == "REG")
        )
        .group_by("team", "pfr_player_id")
        .agg(pl.col("offense_snaps").fill_null(0).sum().alias("snaps"))
        .sort(["team", "snaps", "pfr_player_id"], descending=[False, True, False])
        .group_by("team")
        .first()
    )
    resolved = leaders.join(
        crosswalk_ids.select(
            pl.col("pfr_id").cast(pl.String).alias("pfr_player_id"),
            pl.col("gsis_id").cast(pl.String),
        ).drop_nulls(),
        on="pfr_player_id",
        how="left",
    )
    present = _present(rosters, season=prior_season + 1).rename({"player_id": "gsis_id"})

    return (
        resolved.join(present.with_columns(pl.lit(True).alias("stayed")),
                      on=["team", "gsis_id"], how="left")
        .with_columns(
            pl.when(pl.col("gsis_id").is_null())
            .then(None)  # leader could not be resolved; unknown, not "departed"
            .otherwise(pl.col("stayed").is_null())
            .alias("qb_change")
        )
        .select("team", pl.col("gsis_id").alias("prior_qb"), "qb_change")
        .sort("team")
    )


def guaranteed_money(
    contracts: pl.DataFrame, *, before_season: int | None = None
) -> pl.DataFrame:
    """Per player: guaranteed money on their most recent contract signed before a season.

    The column is `guaranteed`, not `guaranteed_money` (PRD M17 names it wrong). This
    frame carries no season column at all, so recency comes from `year_signed` — and that
    is exactly why `before_season` matters. "Most recent contract" evaluated globally
    would hand a 2024 signing to a model projecting 2022, which is future knowledge
    wearing a player attribute's clothes. Callers projecting a past season MUST pass it.
    """
    assert_columns(contracts, CONTRACTS_REQUIRED, "situation.guaranteed_money")
    scoped = (
        contracts
        if before_season is None
        else contracts.filter(pl.col("year_signed") < before_season)
    )
    return (
        scoped.filter(pl.col("gsis_id").is_not_null())
        .sort(["gsis_id", "year_signed"], descending=[False, True])
        .group_by("gsis_id")
        .first()
        .select(
            "gsis_id",
            pl.col("guaranteed").alias("guaranteed_money"),
            "year_signed",
        )
        .sort("gsis_id")
    )


def draft_capital(
    draft_picks: pl.DataFrame, *, before_season: int | None = None
) -> pl.DataFrame:
    """Per player: draft round and pick. The columns are `round` and `pick`.

    `before_season` drops drafts that had not happened yet — harmless for a player who
    could not have played earlier, but it keeps the frame honestly point-in-time.
    """
    assert_columns(draft_picks, DRAFT_PICKS_REQUIRED, "situation.draft_capital")
    scoped = (
        draft_picks
        if before_season is None
        else draft_picks.filter(pl.col("season") < before_season)
    )
    return (
        scoped.filter(pl.col("gsis_id").is_not_null())
        .select(
            "gsis_id",
            pl.col("round").alias("draft_round"),
            pl.col("pick").alias("draft_pick"),
            pl.col("season").alias("draft_season"),
        )
        .unique(subset=["gsis_id"])
        .sort("gsis_id")
    )
