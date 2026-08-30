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

# load_player_stats(summary_level="week") — weekly raw stat lines scored by M6.
PLAYER_STATS_REQUIRED = {
    "player_id", "player_name", "position", "team", "season", "week", "season_type",
    "passing_yards", "passing_tds", "passing_interceptions", "passing_2pt_conversions",
    "passing_40", "completions", "attempts", "passing_first_downs", "sacks_suffered",
    "rushing_yards", "rushing_tds", "rushing_2pt_conversions", "rushing_40", "carries",
    "rushing_first_downs", "receptions", "receiving_yards", "receiving_tds",
    "receiving_2pt_conversions", "receiving_40", "receiving_first_downs",
    "fumbles_total", "fumbles_lost_total", "fumble_recovery_tds", "special_teams_tds",
    "fg_made_0_19", "fg_made_20_29", "fg_made_30_39", "fg_made_40_49", "fg_made_50_59",
    "fg_made_60_", "fg_made", "fg_missed", "fg_missed_0_19", "fg_missed_20_29",
    "fg_missed_30_39", "fg_missed_40_49", "fg_missed_50_59", "fg_missed_60_",
    "fg_made_distance", "fg_blocked", "pat_made", "pat_missed", "pat_blocked",
}

# load_team_stats(summary_level="week") — only the offensive totals; yards allowed is the
# opponent's official net yardage (gross passing + rushing + sack yards lost, the last negative).
TEAM_STATS_REQUIRED = {
    "team", "season", "week", "season_type", "passing_yards", "rushing_yards", "sack_yards_lost",
}

# load_schedules() — final scores drive the points-allowed tier.
SCHEDULES_REQUIRED = {"season", "week", "home_team", "away_team", "home_score", "away_score"}

# load_pbp() — play level, needed for the 40+ yard TD bonuses and every team-defense event.
PBP_REQUIRED = {
    "game_id", "season", "week", "season_type", "posteam", "defteam", "play_type",
    "special", "touchdown", "td_team", "pass_touchdown", "rush_touchdown", "yards_gained",
    "passer_player_id", "rusher_player_id", "receiver_player_id",
    "sack", "interception", "safety", "punt_blocked", "field_goal_result", "extra_point_result",
    "fumble", "aborted_play", "forced_fumble_player_1_team", "forced_fumble_player_2_team",
    "fumble_recovery_1_team", "fumble_recovery_2_team", "fumbled_1_team", "fumbled_2_team",
}

# load_ff_rankings(type="all") — FantasyPros expert consensus RANK. Verified 2026-08-27:
# this feed carries no projected-points column at any `type`; `ecr` is an average of
# expert ranks. PRD §6.1 calls it "ECR / consensus projections" — it is ranks only (R2).
FF_RANKINGS_REQUIRED = {
    "id", "player", "pos", "team", "ecr", "sd", "best", "worst", "scrape_date", "ecr_type",
}

# ff_playerids join key for the rankings feed.
FF_PLAYERIDS_FANTASYPROS = {"fantasypros_id", "gsis_id"}

# Manual projection CSV drops (PRD §6.4), one file per source per season.
PROJECTION_CSV_REQUIRED = {"player_name", "team", "position", "source", "projected_points"}

# load_player_stats() — the subset M16 reads to count games played. Deliberately narrower
# than PLAYER_STATS_REQUIRED: availability never touches a scoring column, and demanding
# them would make the contract lie about what this consumer needs.
PLAYER_GAMES_REQUIRED = {"season", "player_id", "position", "season_type", "week"}

# load_snap_counts() — prior-season workload. Keyed on `pfr_player_id`, NOT gsis_id;
# join through ff_playerids.pfr_id. Verified 2026-08-27, coverage 2012+.
SNAP_COUNTS_REQUIRED = {
    "season", "week", "game_type", "pfr_player_id", "offense_snaps", "position", "team",
}

# load_players() — birth date drives age at season start. Kept separate from
# PLAYERS_REQUIRED so the Phase 1 contract stays untouched.
PLAYERS_BIRTH_REQUIRED = {"gsis_id", "birth_date"}

# ff_playerids join key for the snap-count feed.
FF_PLAYERIDS_PFR = {"pfr_id", "gsis_id"}

# --- Phase 9 sources, verified against the live schema on 2026-08-28 (nflreadpy 0.1.5) ---
# Several of these differ from the names PRD M17 uses; see the R2 log in CLAUDE.md.

# load_rosters() — ONE ROW PER PLAYER-SEASON, not per week, despite carrying a `week`
# column. `status` is the roster-status code: RES is reserve/IR, DEV is the practice
# squad. Both matter to the vacated-opportunity edge cases.
ROSTERS_REQUIRED = {"season", "team", "gsis_id", "position", "status"}

# load_draft_picks() — draft capital. The columns are `round` and `pick`, NOT the
# `draft_round` / `draft_pick` that PRD M17 names.
DRAFT_PICKS_REQUIRED = {"season", "team", "gsis_id", "round", "pick"}

# load_contracts() — the guarantee column is `guaranteed`, NOT `guaranteed_money`.
# This frame has NO season column; a contract is dated by `year_signed` and `years`.
CONTRACTS_REQUIRED = {"gsis_id", "guaranteed", "year_signed", "years", "is_active"}

# load_depth_charts() — team is `club_code`, not `team`.
DEPTH_CHARTS_REQUIRED = {"season", "club_code", "gsis_id", "position", "depth_team"}

# load_ff_opportunity() — pre-computed expected fantasy points (do not rebuild, §6.1).
# Keyed on `player_id`, which holds a gsis_id. `season` is a String and `week` a Float
# here, unlike every other nflverse frame; cast before joining.
FF_OPPORTUNITY_REQUIRED = {
    "season",
    "week",
    "player_id",
    "posteam",
    "position",
    "rec_attempt",
    "rush_attempt",
    "total_fantasy_points",
    "total_fantasy_points_exp",
}
