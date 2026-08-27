"""M8 — assembling the ranked draft board.

The order matters and is fixed by the specs: aggregate sources, calibrate, *then* apply
overrides, then value against replacement. Overriding before calibration would shrink a
manual number that was meant to be taken at face value.

Team defenses ride a parallel identity. They have no `gsis_id`, so they use the
crosswalk's synthetic `DEF_{team}` convention, and their actuals come from the scoring
engine's separate defense path (PRD §11.2).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from ffdraft.config import Config
from ffdraft.data import nflverse, projections
from ffdraft.data.overrides import Override, apply_overrides
from ffdraft.scoring import statlines
from ffdraft.scoring.engine import ScoringRules, score_defenses, score_players
from ffdraft.valuation.calibration import (
    Calibration,
    actual_season_points,
    calibrated_points,
    fit_calibration,
)
from ffdraft.valuation.replacement import replacement_levels, rostered_depth
from ffdraft.valuation.tiers import assign_tiers, points_standard_error
from ffdraft.valuation.vor import value_over_replacement

# FantasyPros says DST, Sleeper says DEF; the roster slots are Sleeper's.
POSITION_ALIASES = {"DST": "DEF", "PK": "K"}

BOARD_COLUMNS = [
    "rank", "tier", "name", "position", "team", "gsis_id", "ecr", "projected_points",
    "replacement_points", "vor", "points_se", "n_sources", "override_reason",
]


@dataclass
class BoardDiagnostics:
    """What the board could not do, surfaced rather than swallowed (R4)."""

    calibration: Calibration
    unresolved_rankings: pl.DataFrame
    unmatched_csv_rows: pl.DataFrame
    unmatched_overrides: list[Override] = field(default_factory=list)

    def shrinkage(self) -> dict[str, float]:
        return {p: self.calibration.shrinkage_ratio(p) for p in self.calibration.positions()}


def _normalize_positions(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(pl.col("position").replace(POSITION_ALIASES))


def defense_rankings(unresolved: pl.DataFrame) -> pl.DataFrame:
    """Rebuild team defenses from the rows that could not resolve to a `gsis_id`.

    A defense has no player id, so it never joins the crosswalk; the team abbreviation
    is its identity.
    """
    return (
        _normalize_positions(unresolved)
        .filter((pl.col("position") == "DEF") & pl.col("team").is_not_null())
        .with_columns(("DEF_" + pl.col("team")).alias("gsis_id"))
    )


def season_actuals(season: int, rules: ScoringRules, *, refresh: bool = False) -> pl.DataFrame:
    """Points every player and defense actually scored in `season`, in league scoring."""
    pbp = nflverse.pbp([season], refresh=refresh)
    players = score_players(
        statlines.player_week_stats(nflverse.player_stats([season], refresh=refresh), pbp), rules
    )
    defenses = score_defenses(
        statlines.defense_week_stats(
            pbp,
            nflverse.team_stats([season], refresh=refresh),
            nflverse.schedules([season], refresh=refresh),
        ),
        rules,
    )
    return pl.concat(
        [
            actual_season_points(players),
            actual_season_points(
                defenses.with_columns(("DEF_" + pl.col("team")).alias("team")), id_column="team"
            ),
        ]
    )


def training_frame(
    seasons: list[int],
    rules: ScoringRules,
    rankings: pl.DataFrame,
    schedules: pl.DataFrame,
    crosswalk_ids: pl.DataFrame,
    *,
    positions: set[str],
    refresh: bool = False,
) -> pl.DataFrame:
    """(preseason rank, actual points) pairs for every prior season.

    `positions` restricts the fit to what the league actually starts — the ECR feed also
    ranks IDP, which this roster has no slot for.
    """
    frames = []
    for season in seasons:
        ranked, unresolved = projections.preseason_ecr(
            rankings, schedules, crosswalk_ids, season=season
        )
        ranked = pl.concat(
            [_normalize_positions(ranked), defense_rankings(unresolved).select(ranked.columns)]
        )
        ranked = ranked.filter(pl.col("position").is_in(list(positions)))
        actuals = season_actuals(season, rules, refresh=refresh)
        frames.append(
            ranked.join(actuals.drop("season"), on="gsis_id", how="inner").select(
                "season", "gsis_id", "position", "ecr", "actual_points"
            )
        )
    return pl.concat(frames)


def build_board(
    cfg: Config,
    rules: ScoringRules,
    *,
    season: int,
    overrides: list[Override] | None = None,
    refresh: bool = False,
) -> tuple[pl.DataFrame, BoardDiagnostics]:
    """The full pipeline, from expert ranks to a ranked, tiered board."""
    rankings = nflverse.ff_rankings(refresh=refresh)
    schedules = nflverse.schedules([season, *range(season - 10, season)], refresh=refresh)
    ids = nflverse.ff_playerids(refresh=refresh)

    # The league's own roster shape decides which positions are draftable at all.
    depth = rostered_depth(
        cfg.league.fallback_roster_positions, cfg.flex_eligibility, teams=cfg.league.teams
    )
    draftable = set(depth)

    # Bounded by both the owner's configured history and what FantasyPros actually
    # published before kickoff — the ECR feed starts later than the nflverse stats do.
    available = set(projections.seasons_with_preseason_ecr(rankings, schedules))
    prior = sorted(s for s in cfg.data.history_seasons if s < season and s in available)
    if not prior:
        raise ValueError(
            f"no season before {season} has both configured history and preseason ECR; "
            f"ECR covers {sorted(available)}"
        )
    training = training_frame(
        prior, rules, rankings, schedules, ids, positions=draftable, refresh=refresh
    )
    calibration = fit_calibration(
        training,
        target_season=season,
        fit_pool=cfg.calibration.fit_pool,
        min_training_seasons=cfg.calibration.min_training_seasons,
    )

    ranked, unresolved = projections.preseason_ecr(rankings, schedules, ids, season=season)
    ranked = pl.concat(
        [_normalize_positions(ranked), defense_rankings(unresolved).select(ranked.columns)]
    )
    ranked = ranked.filter(pl.col("position").is_in(list(draftable)))
    projected = calibrated_points(ranked, calibration)

    # M4: the calibrated ECR is one source; manual CSV drops are the others.
    crosswalk = _crosswalk(ids)
    csv_long, unmatched_csv = projections.load_csv_sources(
        crosswalk, season=season, fuzzy_threshold=cfg.crosswalk.fuzzy_threshold
    )
    combined = projections.aggregate(
        pl.concat(
            [
                projected.select(
                    "gsis_id",
                    pl.lit(projections.ECR_SOURCE).alias("source"),
                    pl.col("projected_points"),
                ),
                csv_long,
            ]
        )
    )
    board = projected.drop("projected_points").join(combined, on="gsis_id", how="inner")

    board, unmatched_overrides = apply_overrides(board, overrides or [])

    weekly = _historical_weekly(cfg, rules, season, refresh=refresh)
    replacement = replacement_levels(
        weekly,
        depth=depth,
        percentile=cfg.replacement.waiver_percentile,
        games_per_season=cfg.availability.games_per_season,
        lookback_seasons=cfg.replacement.waiver_lookback_seasons,
    )

    valued = value_over_replacement(board, replacement)
    valued = points_standard_error(valued, calibration)
    # Isotonic regression is a step function, so a run of players shares one fitted
    # value. Expert rank breaks those ties: among players the fit cannot separate, the
    # better-ranked one goes first. `gsis_id` is the last resort — without it, players
    # tied on both value and rank order arbitrarily and the board changes between runs,
    # which can move a tier boundary (R7).
    valued = assign_tiers(
        valued.sort(["vor", "ecr", "gsis_id"], descending=[True, False, False])
    )
    valued = valued.with_columns(pl.int_range(1, pl.len() + 1).alias("rank"))
    return valued.select([c for c in BOARD_COLUMNS if c in valued.columns]), BoardDiagnostics(
        calibration=calibration,
        # DST rows resolve through the defense path, and IDP is not draftable here, so
        # neither is a gap worth warning about.
        unresolved_rankings=_normalize_positions(unresolved).filter(
            pl.col("position").is_in(list(draftable - {"DEF"}))
        ),
        unmatched_csv_rows=unmatched_csv,
        unmatched_overrides=unmatched_overrides,
    )


def _crosswalk(ff_playerids: pl.DataFrame) -> pl.DataFrame:
    from ffdraft.data.crosswalk import build_crosswalk

    return build_crosswalk(ff_playerids)


def _historical_weekly(
    cfg: Config, rules: ScoringRules, season: int, *, refresh: bool = False
) -> pl.DataFrame:
    """Weekly scored points for the replacement-level lookback, players and defenses."""
    lookback = [
        s
        for s in cfg.data.history_seasons
        if season - cfg.replacement.waiver_lookback_seasons <= s < season
    ]
    frames = []
    for year in lookback:
        pbp = nflverse.pbp([year], refresh=refresh)
        players = statlines.player_week_stats(
            nflverse.player_stats([year], refresh=refresh), pbp
        )
        scored = score_players(players, rules).join(
            players.select("season", "week", "player_id", "position"),
            on=["season", "week", "player_id", "position"],
            how="left",
        )
        frames.append(
            scored.select("season", "week", pl.col("player_id").alias("gsis_id"), "position", "points")
        )
        defenses = score_defenses(
            statlines.defense_week_stats(
                pbp,
                nflverse.team_stats([year], refresh=refresh),
                nflverse.schedules([year], refresh=refresh),
            ),
            rules,
        )
        frames.append(
            defenses.select(
                "season", "week", ("DEF_" + pl.col("team")).alias("gsis_id"),
                pl.lit("DEF").alias("position"), "points",
            )
        )
    return pl.concat(frames)
