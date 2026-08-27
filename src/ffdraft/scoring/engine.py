"""M6 — the scoring engine.

Turns a league's own `scoring_settings` into a function over **raw stat lines**. No
scoring value is hardcoded: every coefficient comes from the live league object (R5).
Any key the league sets that this engine cannot honour raises at parse time rather
than being quietly scored around (R4) — including keys that are recognised but have
no stat source, which raise only when the league actually gives them weight.

Skill players/kickers and team defenses take separate paths: a team defense has no
`gsis_id` and an entirely different stat structure (PRD §11.2).

Sleeper semantics confirmed against the owner's 2025 league (all 18 weeks):

- `fgmiss` and `xpmiss` count blocked kicks; nflverse's `fg_missed`/`pat_missed` do not,
  so the blocked columns are added in.
- `fum` is charged on every fumble and `fum_lost` again on the ones lost.
- The `*_40p` bonuses stack with the base yardage and touchdown keys.
- A defensive/special-teams touchdown pays once, through `def_td`/`st_td`; `fum_rec_td`
  does not stack on top of it for a team defense.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


def _c(name: str) -> pl.Expr:
    """A stat column as a number, with nulls (player did not record the stat) as zero."""
    return pl.col(name).fill_null(0)


def _flag(cond: pl.Expr) -> pl.Expr:
    return cond.cast(pl.Int32)


# --- key registry -----------------------------------------------------------------
# Every key below is one Sleeper may put in `scoring_settings`. A key mapped to an
# expression is scored; a key in UNSUPPORTED_KEYS is recognised but has no stat source
# in this project's data, and raises if the league gives it a non-zero weight.

PLAYER_RULES: dict[str, pl.Expr] = {
    # passing
    "pass_yd": _c("passing_yards"),
    "pass_td": _c("passing_tds"),
    "pass_int": _c("passing_interceptions"),
    "pass_2pt": _c("passing_2pt_conversions"),
    "pass_att": _c("attempts"),
    "pass_cmp": _c("completions"),
    "pass_inc": _c("attempts") - _c("completions"),
    "pass_sack": _c("sacks_suffered"),
    "pass_fd": _c("passing_first_downs"),
    "pass_cmp_40p": _c("passing_40"),
    "pass_td_40p": _c("pass_td_40p"),
    "bonus_pass_yd_300": _flag(_c("passing_yards") >= 300),
    "bonus_pass_yd_400": _flag(_c("passing_yards") >= 400),
    "bonus_pass_cmp_25": _flag(_c("completions") >= 25),
    # rushing
    "rush_yd": _c("rushing_yards"),
    "rush_td": _c("rushing_tds"),
    "rush_2pt": _c("rushing_2pt_conversions"),
    "rush_att": _c("carries"),
    "rush_fd": _c("rushing_first_downs"),
    "rush_40p": _c("rushing_40"),
    "rush_td_40p": _c("rush_td_40p"),
    "bonus_rush_yd_100": _flag(_c("rushing_yards") >= 100),
    "bonus_rush_yd_200": _flag(_c("rushing_yards") >= 200),
    "bonus_rush_att_20": _flag(_c("carries") >= 20),
    # receiving
    "rec": _c("receptions"),
    "rec_yd": _c("receiving_yards"),
    "rec_td": _c("receiving_tds"),
    "rec_2pt": _c("receiving_2pt_conversions"),
    "rec_fd": _c("receiving_first_downs"),
    "rec_40p": _c("receiving_40"),
    "rec_td_40p": _c("rec_td_40p"),
    "bonus_rec_yd_100": _flag(_c("receiving_yards") >= 100),
    "bonus_rec_yd_200": _flag(_c("receiving_yards") >= 200),
    "bonus_rush_rec_yd_100": _flag(_c("rushing_yards") + _c("receiving_yards") >= 100),
    "bonus_rush_rec_yd_200": _flag(_c("rushing_yards") + _c("receiving_yards") >= 200),
    "bonus_rec_rb": pl.when(pl.col("position") == "RB").then(_c("receptions")).otherwise(0),
    "bonus_rec_wr": pl.when(pl.col("position") == "WR").then(_c("receptions")).otherwise(0),
    "bonus_rec_te": pl.when(pl.col("position") == "TE").then(_c("receptions")).otherwise(0),
    # turnovers and returns
    "fum": _c("fumbles_total"),
    "fum_lost": _c("fumbles_lost_total"),
    "fum_rec_td": _c("fumble_recovery_tds"),
    "st_td": _c("special_teams_tds"),
    # kicking. Sleeper counts a blocked kick as a miss; nflverse tracks blocks separately.
    "fgm": _c("fg_made"),
    "fgm_0_19": _c("fg_made_0_19"),
    "fgm_20_29": _c("fg_made_20_29"),
    "fgm_30_39": _c("fg_made_30_39"),
    "fgm_40_49": _c("fg_made_40_49"),
    "fgm_50p": _c("fg_made_50_59") + _c("fg_made_60_"),
    "fgm_yds": _c("fg_made_distance"),
    "fgmiss": _c("fg_missed") + _c("fg_blocked"),
    "xpm": _c("pat_made"),
    "xpmiss": _c("pat_missed") + _c("pat_blocked"),
}

DEFENSE_RULES: dict[str, pl.Expr] = {
    "sack": _c("sack"),
    "int": _c("int"),
    "safe": _c("safe"),
    "blk_kick": _c("blk_kick"),
    "ff": _c("ff"),
    "st_ff": _c("st_ff"),
    "fum_rec": _c("fum_rec"),
    "st_fum_rec": _c("st_fum_rec"),
    "def_td": _c("def_td"),
    "st_td": _c("st_td"),
    "pts_allow": _c("pts_allow"),
    "yds_allow": _c("yds_allow"),
    # The touchdown itself is already paid by def_td/st_td; a team defense is not paid
    # twice for a fumble returned for a score (verified across the 2025 season).
    "fum_rec_td": pl.lit(0),
}

# Keys whose count this engine folds into another key. If the league weights them
# differently from their counterpart the fold would be wrong, so parse_settings checks.
ALIASED_KEYS = {"def_st_td": "st_td", "def_st_ff": "st_ff", "def_st_fum_rec": "st_fum_rec"}

# (low, high, key) — inclusive, first match wins. Verified against the 2025 season.
PTS_ALLOW_TIERS = (
    (0, 0, "pts_allow_0"),
    (1, 6, "pts_allow_1_6"),
    (7, 13, "pts_allow_7_13"),
    (14, 20, "pts_allow_14_20"),
    (21, 27, "pts_allow_21_27"),
    (28, 34, "pts_allow_28_34"),
    (35, None, "pts_allow_35p"),
)
YDS_ALLOW_TIERS = (
    (None, 99, "yds_allow_0_100"),
    (100, 199, "yds_allow_100_199"),
    (200, 299, "yds_allow_200_299"),
    (300, 349, "yds_allow_300_349"),
    (350, 399, "yds_allow_350_399"),
    (400, 449, "yds_allow_400_449"),
    (450, 499, "yds_allow_450_499"),
    (500, 549, "yds_allow_500_549"),
    (550, None, "yds_allow_550p"),
)
TIER_KEYS = {k for _, _, k in (*PTS_ALLOW_TIERS, *YDS_ALLOW_TIERS)}

# Recognised, but nothing in this project's data supplies them. Scoring one of these
# would need play-level reception/return distances or individual-defender lines, which
# v1 has no roster slot for. Weighting any of them raises rather than being ignored.
UNSUPPORTED_KEYS = frozenset(
    {
        # reception-distance buckets (need per-reception yardage)
        "rec_0_4", "rec_5_9", "rec_10_19", "rec_20_29", "rec_30_39",
        # per-distance misses (nflverse buckets exclude blocks, Sleeper's do not)
        "fgmiss_0_19", "fgmiss_20_29", "fgmiss_30_39", "fgmiss_40_49", "fgmiss_50p",
        "fgm_yds_over_30",
        # return and turnover yardage
        "int_ret_yd", "fum_ret_yd", "sack_yd", "kr_yd", "pr_yd", "def_kr_yd", "def_pr_yd",
        "blk_kick_ret_yd", "fg_ret_yd",
        # individual defenders (no IDP slots in v1)
        "tkl", "tkl_ast", "tkl_solo", "tkl_loss", "qb_hit", "def_pass_def",
        "def_st_tkl_solo", "st_tkl_solo",
        # misc
        "def_2pt", "pass_int_td",
    }
)

RECOGNIZED_KEYS = (
    set(PLAYER_RULES) | set(DEFENSE_RULES) | set(ALIASED_KEYS) | TIER_KEYS | set(UNSUPPORTED_KEYS)
)


@dataclass(frozen=True)
class ScoringRules:
    """The league's scoring settings, validated. `weights` is the league's own dict."""

    weights: dict[str, float]

    def weight(self, key: str) -> float:
        return float(self.weights.get(key, 0.0))


def parse_settings(scoring_settings: dict) -> ScoringRules:
    """Validate a live league's `scoring_settings` and return the rule set.

    Raises on any key this engine does not recognise, and on any recognised key that it
    cannot compute from raw stats but the league has actually given weight to.
    """
    unknown = sorted(set(scoring_settings) - RECOGNIZED_KEYS)
    if unknown:
        raise ValueError(
            f"scoring_settings contains keys the scoring engine does not recognize: {unknown}. "
            f"Add a rule for each in ffdraft.scoring.engine before scoring this league."
        )

    weighted_unsupported = sorted(
        k for k in scoring_settings if k in UNSUPPORTED_KEYS and scoring_settings[k]
    )
    if weighted_unsupported:
        raise ValueError(
            f"scoring_settings gives weight to keys with no raw-stat source: "
            f"{weighted_unsupported}. Scoring this league would silently under-count."
        )

    mismatched = sorted(
        f"{alias}={scoring_settings[alias]} vs {target}={scoring_settings.get(target, 0.0)}"
        for alias, target in ALIASED_KEYS.items()
        if scoring_settings.get(alias) and scoring_settings[alias] != scoring_settings.get(target)
    )
    if mismatched:
        raise ValueError(
            f"scoring_settings weights a defense/special-teams key differently from the key "
            f"it is folded into: {mismatched}. Split the counts before scoring this league."
        )

    return ScoringRules(weights=dict(scoring_settings))


def _tier_expr(value: pl.Expr, tiers: tuple, rules: ScoringRules) -> pl.Expr:
    """Points for the first bucket `value` falls into. Inclusive bounds, first match wins."""
    out: pl.Expr | None = None
    for low, high, key in tiers:
        cond = pl.lit(True)
        if low is not None:
            cond = cond & (value >= low)
        if high is not None:
            cond = cond & (value <= high)
        branch = pl.when(cond).then(pl.lit(rules.weight(key)))
        out = branch if out is None else out.when(cond).then(pl.lit(rules.weight(key)))
    assert out is not None
    return out.otherwise(pl.lit(0.0))


def _points(rules: ScoringRules, table: dict[str, pl.Expr]) -> pl.Expr:
    """Sum of weight * stat over every rule the league actually weights."""
    terms = [expr * rules.weight(key) for key, expr in table.items() if rules.weight(key)]
    if not terms:
        return pl.lit(0.0)
    total = terms[0]
    for term in terms[1:]:
        total = total + term
    return total.cast(pl.Float64)


def score_players(stats: pl.DataFrame, rules: ScoringRules) -> pl.DataFrame:
    """Score raw weekly stat lines for skill players and kickers."""
    return stats.select(
        "season",
        "week",
        "player_id",
        "position",
        _points(rules, PLAYER_RULES).round(2).alias("points"),
    )


def score_defenses(defense_stats: pl.DataFrame, rules: ScoringRules) -> pl.DataFrame:
    """Score raw weekly team-defense lines, including the points- and yards-allowed tiers."""
    tiers = _tier_expr(_c("pts_allow"), PTS_ALLOW_TIERS, rules) + _tier_expr(
        _c("yds_allow"), YDS_ALLOW_TIERS, rules
    )
    return defense_stats.select(
        "season",
        "week",
        "team",
        (_points(rules, DEFENSE_RULES) + tiers).round(2).alias("points"),
    )
