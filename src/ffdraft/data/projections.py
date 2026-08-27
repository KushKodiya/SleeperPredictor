"""M4 — projection sources and their equal-weighted aggregation.

Two kinds of source feed the board:

- **FantasyPros expert consensus rank.** `load_ff_rankings` carries no projected
  points at any `type` — only `ecr`, an average of expert ranks (see CLAUDE.md's R2
  log). `preseason_ecr` extracts the last redraft-positional ranking published before
  a season's opening kickoff; `valuation.calibration` is what turns that rank into
  points.
- **Manual CSV drops** at `data/raw/projections/*.csv` (PRD §6.4), which carry
  `projected_points` directly. Scraping is out of scope for v1.

`aggregate` averages whatever sources cover a player with **equal weights**. Twelve
seasons show source accuracy does not persist year to year, so inverse-MAE weighting
underperforms equal weighting (PRD §8 M4). A source that lacks a player contributes
nothing — never a zero, which would silently drag the mean down.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from ffdraft.contracts import (
    FF_PLAYERIDS_FANTASYPROS,
    FF_RANKINGS_REQUIRED,
    PROJECTION_CSV_REQUIRED,
    assert_columns,
)
from ffdraft.data.crosswalk import resolve_frame

CSV_DIR = Path("data/raw/projections")

# FantasyPros publishes several ranking families; "rp" is redraft, ranked within
# position, which is the one a season-long snake draft is played from. Best-ball ("bp"),
# dynasty ("dp"), superflex ("rsf") and weekly ("wp") ranks are deliberately excluded.
REDRAFT_POSITIONAL = "rp"

# The name every ECR-derived projection carries in the source column.
ECR_SOURCE = "fantasypros_ecr"


def seasons_with_preseason_ecr(rankings: pl.DataFrame, schedules: pl.DataFrame) -> list[int]:
    """Seasons whose redraft-positional ECR was published before kickoff.

    FantasyPros history starts partway through the nflverse era, so the calibration's
    training window is bounded by what was actually scraped, not by `history_seasons`.
    Derived from the feed rather than pinned to a year, so it extends itself.
    """
    assert_columns(rankings, FF_RANKINGS_REQUIRED, "projections.seasons_with_preseason_ecr")
    openers = (
        schedules.filter(pl.col("week") == 1)
        .group_by("season")
        .agg(pl.col("gameday").min().alias("opener"))
    )
    return sorted(
        rankings.filter(pl.col("ecr_type") == REDRAFT_POSITIONAL)
        .with_columns(pl.col("scrape_date").str.slice(0, 4).cast(pl.Int64).alias("season"))
        .join(openers, on="season")
        .filter(pl.col("scrape_date") < pl.col("opener"))["season"]
        .unique()
        .to_list()
    )


def preseason_ecr(
    rankings: pl.DataFrame, schedules: pl.DataFrame, crosswalk_ids: pl.DataFrame, *, season: int
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The last redraft-positional ECR published before `season` kicked off.

    Rankings keep updating through the season, so the cutoff is the week 1 kickoff
    date — without it the "latest preseason scrape" silently lands on a week-3 in-season
    ranking that already knows how the season started.

    Returns (ranked, unresolved). Rows whose FantasyPros id is not in the crosswalk are
    returned separately rather than dropped (R4).
    """
    assert_columns(rankings, FF_RANKINGS_REQUIRED, "projections.preseason_ecr.rankings")
    assert_columns(schedules, {"season", "week", "gameday"}, "projections.preseason_ecr.schedules")
    assert_columns(
        crosswalk_ids, FF_PLAYERIDS_FANTASYPROS, "projections.preseason_ecr.crosswalk_ids"
    )

    opener = (
        schedules.filter((pl.col("season") == season) & (pl.col("week") == 1))
        .select(pl.col("gameday").min())
        .item()
    )
    if opener is None:
        raise ValueError(f"no week 1 kickoff date in schedules for season {season}")

    scrapes = rankings.filter(
        (pl.col("ecr_type") == REDRAFT_POSITIONAL)
        & (pl.col("scrape_date").str.slice(0, 4).cast(pl.Int64) == season)
        & (pl.col("scrape_date") < opener)
    )
    if scrapes.is_empty():
        raise ValueError(
            f"no redraft-positional ECR scraped before the {season} opener ({opener}); "
            f"FantasyPros history starts in 2019"
        )
    latest = scrapes.select(pl.col("scrape_date").max()).item()

    ids = (
        crosswalk_ids.filter(pl.col("fantasypros_id").is_not_null() & pl.col("gsis_id").is_not_null())
        .with_columns(pl.col("fantasypros_id").cast(pl.String))
        .unique(subset=["fantasypros_id"])
        .select("fantasypros_id", "gsis_id")
    )
    joined = (
        scrapes.filter(pl.col("scrape_date") == latest)
        .with_columns(pl.col("id").cast(pl.String))
        .join(ids, left_on="id", right_on="fantasypros_id", how="left")
        .select(
            pl.lit(season).alias("season"),
            pl.col("gsis_id"),
            pl.col("player").alias("name"),
            pl.col("pos").alias("position"),
            pl.col("team"),
            pl.col("ecr"),
            pl.col("sd"),
            pl.lit(latest).alias("scrape_date"),
        )
    )
    resolved = joined.filter(pl.col("gsis_id").is_not_null())
    return resolved, joined.filter(pl.col("gsis_id").is_null())


def load_csv_sources(
    crosswalk: pl.DataFrame, *, season: int, fuzzy_threshold: int, csv_dir: Path = CSV_DIR
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Read every manual projection CSV for `season` and resolve its rows to `gsis_id`.

    Returns (long, unmatched) where `long` has one row per (source, player). An empty
    frame with the right columns comes back when no CSVs have been dropped in.
    """
    empty = pl.DataFrame(
        schema={"gsis_id": pl.String, "source": pl.String, "projected_points": pl.Float64}
    )
    files = sorted(csv_dir.glob(f"*_{season}.csv")) if csv_dir.exists() else []
    if not files:
        return empty, empty.head(0)

    frames, unmatched = [], []
    for path in files:
        raw = pl.read_csv(path)
        assert_columns(raw, PROJECTION_CSV_REQUIRED, f"projections.load_csv_sources[{path.name}]")
        queries = raw.rename({"player_name": "name"})
        matched, missed = resolve_frame(queries, crosswalk, fuzzy_threshold=fuzzy_threshold)
        frames.append(
            matched.select(
                "gsis_id",
                pl.col("source").cast(pl.String),
                pl.col("projected_points").cast(pl.Float64),
            )
        )
        unmatched.append(missed.with_columns(pl.lit(path.name).alias("file")))

    return pl.concat(frames) if frames else empty, (
        pl.concat(unmatched, how="diagonal") if unmatched else empty.head(0)
    )


def aggregate(sources: pl.DataFrame) -> pl.DataFrame:
    """Equal-weighted mean of `projected_points` over the sources covering each player.

    `sources` is long: one row per (gsis_id, source). The mean is taken over the rows
    that exist, so a source missing a player simply is not in the average — it does not
    enter as a zero. `n_sources` travels with the projection so a one-source player is
    visibly less trustworthy than a five-source one.
    """
    assert_columns(sources, {"gsis_id", "source", "projected_points"}, "projections.aggregate")
    return (
        sources.filter(pl.col("projected_points").is_not_null())
        .group_by("gsis_id")
        .agg(
            pl.col("projected_points").mean().alias("projected_points"),
            pl.col("source").n_unique().alias("n_sources"),
        )
        .sort("projected_points", descending=True)
    )
