"""M16 — how many games a player is available for, as a distribution.

The honest purpose of this module is **variance, not prediction**. Injury timing is close
to unpredictable, and the PRD is explicit that a per-player injury classifier is out of
scope. What matters is that the spread is right, because M10 samples from it and every
simulation downstream inherits its variance (PRD §11.12).

So the distribution is empirical: for each `(position, age band, workload tier)` cell,
the observed history of games played, with thin cells pooling up rather than collapsing
onto a spike.

**Workload runs the opposite way to the PRD's expectation, and this module follows the
data.** The PRD assumed high prior-season snap volume marks a worn-down player who will
miss more time. Across 2015-2025 the reverse holds at every position — age-matched,
top-decile-workload players average 2.8 (RB) to 5.5 (QB) *more* games than median-workload
ones. Prior snaps are a durability marker, not a mileage penalty: you only accumulate
them by staying on the field. Encoding the assumed direction would have made the model
knowingly miscalibrated, invisibly, in the one place variance is set.

Age behaves as expected once the population is right: among running backs with a real
prior-season role, mean games slides from 13.3 at 23 to 11.1 at 29.

The population is **rosterable players** — those at or above
`availability.rosterable_percentile` of prior-season snaps within their position. Training
on everyone who ever logged a stat line drags the running-back mean down to 7.6 games,
because it is dominated by fringe players who never had a role; that is not the
distribution a drafted player is drawn from.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from ffdraft.contracts import (
    FF_PLAYERIDS_PFR,
    PLAYER_GAMES_REQUIRED,
    PLAYERS_BIRTH_REQUIRED,
    SNAP_COUNTS_REQUIRED,
    assert_columns,
)

HISTORY_REQUIRED = {"season", "gsis_id", "position", "games_played", "age", "workload_percentile"}


@dataclass(frozen=True)
class PlayerSeason:
    """The three things availability conditions on. Any of them may be unknown."""

    position: str
    age: float | None = None
    workload_percentile: float | None = None


@dataclass(frozen=True)
class AvailabilityModel:
    """Empirical games-played distributions with a pooling ladder for thin cells."""

    games_per_season: int
    age_bin_width: int
    workload_threshold: float
    cells: dict[tuple[str, int, bool], np.ndarray] = field(default_factory=dict)
    by_position_tier: dict[tuple[str, bool], np.ndarray] = field(default_factory=dict)
    by_position: dict[str, np.ndarray] = field(default_factory=dict)
    overall: np.ndarray | None = None
    counts: dict[tuple[str, int, bool], int] = field(default_factory=dict)

    @property
    def games(self) -> np.ndarray:
        return np.arange(self.games_per_season + 1)

    def age_bin(self, age: float | None) -> int | None:
        if age is None or not np.isfinite(age):
            return None
        return int(np.floor(age / self.age_bin_width) * self.age_bin_width)

    def is_high_workload(self, percentile: float | None) -> bool:
        return percentile is not None and percentile >= self.workload_threshold

    def pmf(self, player: PlayerSeason) -> np.ndarray:
        """The distribution for a player, falling back down the pooling ladder.

        Cell -> position+tier -> position -> overall. A cell that never met the minimum
        count is absent from `cells` entirely, so this walk is what "pooling up" means.
        """
        tier = self.is_high_workload(player.workload_percentile)
        band = self.age_bin(player.age)
        if band is not None:
            cell = self.cells.get((player.position, band, tier))
            if cell is not None:
                return cell
        pooled = self.by_position_tier.get((player.position, tier))
        if pooled is not None:
            return pooled
        pooled = self.by_position.get(player.position)
        if pooled is not None:
            return pooled
        if self.overall is None:
            raise ValueError("availability model is empty; it was built from no history")
        return self.overall

    def games_played_distribution(
        self, player: PlayerSeason, rng: np.random.Generator, n_sims: int = 1
    ) -> np.ndarray:
        """`n_sims` draws of integer games played in [0, games_per_season] (R7: explicit rng)."""
        if n_sims < 1:
            raise ValueError(f"n_sims must be at least 1, got {n_sims}")
        return rng.choice(self.games, size=n_sims, p=self.pmf(player))

    def expected_games(self, player: PlayerSeason) -> float:
        return float(np.dot(self.games, self.pmf(player)))


def _pmf(values: np.ndarray, games_per_season: int) -> np.ndarray:
    counts = np.bincount(
        np.clip(values.astype(int), 0, games_per_season), minlength=games_per_season + 1
    ).astype(float)
    return counts / counts.sum()


def build_availability(
    history: pl.DataFrame,
    *,
    games_per_season: int,
    age_bin_width: int,
    min_bin_count: int,
    workload_percentile: float,
) -> AvailabilityModel:
    """Fit the empirical distributions. `history` is one row per rosterable player-season."""
    assert_columns(history, HISTORY_REQUIRED, "availability.build_availability")
    frame = history.filter(pl.col("games_played").is_not_null())
    if frame.is_empty():
        raise ValueError("availability history is empty; nothing to fit")

    frame = frame.with_columns(
        (pl.col("workload_percentile") >= workload_percentile)
        .fill_null(False)  # no prior-season snaps is not high workload
        .alias("high_workload"),
        (pl.col("age") / age_bin_width).floor().mul(age_bin_width).alias("age_band"),
    )

    overall = _pmf(frame["games_played"].to_numpy(), games_per_season)
    by_position, by_position_tier, cells, counts = {}, {}, {}, {}

    for (position,), rows in frame.group_by(["position"]):
        by_position[position] = _pmf(rows["games_played"].to_numpy(), games_per_season)
    for (position, tier), rows in frame.group_by(["position", "high_workload"]):
        if rows.height >= min_bin_count:
            by_position_tier[(position, tier)] = _pmf(
                rows["games_played"].to_numpy(), games_per_season
            )
    for (position, band, tier), rows in frame.drop_nulls("age_band").group_by(
        ["position", "age_band", "high_workload"]
    ):
        key = (position, int(band), tier)
        counts[key] = rows.height
        # Below the minimum the cell is simply not created; `pmf` then pools up. A cell
        # built from four observations would be a spike, not a distribution.
        if rows.height >= min_bin_count:
            cells[key] = _pmf(rows["games_played"].to_numpy(), games_per_season)

    return AvailabilityModel(
        games_per_season=games_per_season,
        age_bin_width=age_bin_width,
        workload_threshold=workload_percentile,
        cells=cells,
        by_position_tier=by_position_tier,
        by_position=by_position,
        overall=overall,
        counts=counts,
    )


def availability_history(
    player_stats: pl.DataFrame,
    players: pl.DataFrame,
    snap_counts: pl.DataFrame,
    crosswalk_ids: pl.DataFrame,
    *,
    positions: set[str],
    rosterable_percentile: float,
    season_start_month: int = 9,
) -> pl.DataFrame:
    """One row per rosterable player-season: games played, age, prior-season workload.

    Games played comes from the count of weekly stat rows; age from birth date at the
    start of the season; workload from the *prior* season's offensive snaps, ranked
    within position so the threshold means the same thing at every position.
    """
    assert_columns(player_stats, PLAYER_GAMES_REQUIRED, "availability.history.player_stats")
    assert_columns(players, PLAYERS_BIRTH_REQUIRED, "availability.history.players")
    assert_columns(snap_counts, SNAP_COUNTS_REQUIRED, "availability.history.snap_counts")
    assert_columns(crosswalk_ids, FF_PLAYERIDS_PFR, "availability.history.crosswalk_ids")

    games = (
        player_stats.filter(
            (pl.col("season_type") == "REG") & pl.col("position").is_in(list(positions))
        )
        .group_by(["season", "player_id", "position"])
        .agg(pl.col("week").n_unique().alias("games_played"))
    )

    pfr = (
        crosswalk_ids.filter(pl.col("pfr_id").is_not_null() & pl.col("gsis_id").is_not_null())
        .unique(subset=["pfr_id"])
        .select("pfr_id", "gsis_id")
    )
    # Shifted forward a season: what a player did last year is what we know at draft time.
    prior_snaps = (
        snap_counts.filter(pl.col("game_type") == "REG")
        .group_by(["season", "pfr_player_id"])
        .agg(pl.col("offense_snaps").sum().alias("prior_snaps"))
        .join(pfr, left_on="pfr_player_id", right_on="pfr_id", how="inner")
        .select((pl.col("season") + 1).alias("season"), "gsis_id", "prior_snaps")
    )

    ages = players.select(
        "gsis_id", pl.col("birth_date").str.to_date(strict=False).alias("birth_date")
    )

    frame = (
        games.join(ages, left_on="player_id", right_on="gsis_id", how="left")
        .join(prior_snaps, left_on=["season", "player_id"], right_on=["season", "gsis_id"],
              how="left")
        .with_columns(
            (
                (pl.date(pl.col("season"), season_start_month, 1) - pl.col("birth_date"))
                .dt.total_days() / 365.25
            ).floor().alias("age")
        )
        .with_columns(
            (
                pl.col("prior_snaps").rank("average") / pl.col("prior_snaps").count()
            ).over(["season", "position"]).alias("workload_percentile")
        )
        .rename({"player_id": "gsis_id"})
    )
    # Rosterable only: below this the population is fringe players who never had a role,
    # and their availability is not what a drafted player's is drawn from.
    return frame.filter(pl.col("workload_percentile") >= rosterable_percentile).select(
        "season", "gsis_id", "position", "games_played", "age", "workload_percentile"
    )
