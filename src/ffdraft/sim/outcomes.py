"""M10 — sampling a season for a roster, week by week.

This is what makes bench players worth anything. Scoring a roster on summed season
projections makes the engine conclude depth is worthless and waste its last six rounds;
in reality a backup is worth exactly what he scores in the weeks the starter ahead of him
is out, which only a weekly simulation can see.

Each sim draws games played from the availability model, blanks the team's bye week,
draws a score for every week the player is active, then starts the best legal lineup each
week and sums. Every stochastic step takes an explicit `rng` (R7).

**The weekly rate is the projection divided by expected games, not by 17.** The Phase 3
projections are fit on actual season totals, which already carry the games those players
missed. Dividing by a full season and then sampling games missed on top would discount
availability twice, and the mean would land below the projection instead of on it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl

from ffdraft.contracts import assert_columns
from ffdraft.lineup.slots import SlotConfig
from ffdraft.lineup.value import Player, lineup_value
from ffdraft.sim.availability import AvailabilityModel, PlayerSeason

NO_BYE = 0


@dataclass(frozen=True)
class SimPlayer:
    """A rosterable player, with everything the season sim needs to draw his year."""

    player_id: str
    position: str
    projected_points: float
    team: str | None = None
    age: float | None = None
    workload_percentile: float | None = None

    @property
    def availability(self) -> PlayerSeason:
        return PlayerSeason(self.position, self.age, self.workload_percentile)


def bye_weeks(schedules: pl.DataFrame, *, season: int) -> dict[str, int]:
    """Each team's bye: the regular-season week in which it has no game."""
    assert_columns(schedules, {"season", "week", "home_team", "away_team", "game_type"},
                   "outcomes.bye_weeks")
    games = schedules.filter((pl.col("season") == season) & (pl.col("game_type") == "REG"))
    if games.is_empty():
        return {}
    weeks = set(games["week"].unique().to_list())
    played = (
        pl.concat([games.select(pl.col("home_team").alias("team"), "week"),
                   games.select(pl.col("away_team").alias("team"), "week")])
        .group_by("team")
        .agg(pl.col("week").unique().alias("weeks"))
    )
    byes = {}
    for row in played.iter_rows(named=True):
        missing = sorted(weeks - set(row["weeks"]))
        if len(missing) == 1:
            byes[row["team"]] = missing[0]
    return byes


def weekly_dispersion(scored_weeks: pl.DataFrame, *, default: float = 1.0) -> dict[str, float]:
    """Coefficient of variation of weekly points, per position, measured from history.

    Not a chosen constant: the spread of a real weekly score is what it is, and the
    simulation's variance should come from the same place its means do (R5).
    """
    assert_columns(scored_weeks, {"position", "points"}, "outcomes.weekly_dispersion")
    stats = (
        scored_weeks.filter(pl.col("points").is_not_null())
        .group_by("position")
        .agg(pl.col("points").mean().alias("mean"), pl.col("points").std().alias("sd"))
    )
    out = {}
    for row in stats.iter_rows(named=True):
        mean, sd = row["mean"], row["sd"]
        # sd of exactly 0 is a measurement, not a missing one — do not fall back on it.
        out[row["position"]] = float(sd / mean) if mean and mean > 0 and sd is not None else default
    return out


def _active_weeks(
    games_played: np.ndarray, *, bye: int, n_weeks: int, rng: np.random.Generator
) -> np.ndarray:
    """Boolean (n_sims, n_weeks): which weeks the player is on the field.

    Injury *timing* is close to unpredictable (PRD §8 M16), so which weeks are missed is
    drawn uniformly rather than modelled. Only the count carries information.
    """
    n_sims = games_played.shape[0]
    playable = np.ones((n_sims, n_weeks), dtype=bool)
    if bye != NO_BYE and 1 <= bye <= n_weeks:
        playable[:, bye - 1] = False

    active = np.zeros((n_sims, n_weeks), dtype=bool)
    order = rng.random((n_sims, n_weeks))
    order[~playable] = np.inf  # a bye week can never be one of the weeks he plays
    ranked = np.argsort(order, axis=1)
    for sim in range(n_sims):
        played = min(int(games_played[sim]), int(playable[sim].sum()))
        active[sim, ranked[sim, :played]] = True
    return active


def simulate_players(
    roster: Sequence[SimPlayer],
    model: AvailabilityModel,
    byes: dict[str, int],
    rng: np.random.Generator,
    *,
    n_sims: int,
    n_weeks: int,
    dispersion: dict[str, float],
) -> np.ndarray:
    """Weekly points for every player in every sim: shape (n_sims, n_players, n_weeks).

    Inactive and bye weeks are exactly zero.
    """
    scores = np.zeros((n_sims, len(roster), n_weeks))
    for index, player in enumerate(roster):
        expected = model.expected_games(player.availability)
        if expected <= 0:
            continue
        # The rate that makes the season mean land on the projection.
        rate = player.projected_points / expected
        games = model.games_played_distribution(player.availability, rng, n_sims=n_sims)
        active = _active_weeks(games, bye=byes.get(player.team or "", NO_BYE),
                               n_weeks=n_weeks, rng=rng)

        cv = dispersion.get(player.position, 1.0)
        if rate > 0 and cv > 0:
            shape = 1.0 / (cv * cv)
            drawn = rng.gamma(shape, rate / shape, size=(n_sims, n_weeks))
        else:
            drawn = np.full((n_sims, n_weeks), rate)
        scores[:, index, :] = np.where(active, drawn, 0.0)
    return scores


def season_totals(weekly: np.ndarray) -> np.ndarray:
    """Per-sim, per-player season points: shape (n_sims, n_players)."""
    return weekly.sum(axis=2)


def roster_value(
    roster: Sequence[SimPlayer], weekly: np.ndarray, slots: SlotConfig
) -> np.ndarray:
    """Points the roster actually starts, summed over the season: shape (n_sims,).

    A player who scores 30 on the bench scores nothing for you, which is the entire
    reason this is not just `season_totals(...).sum(axis=1)`.
    """
    n_sims, _, n_weeks = weekly.shape
    totals = np.zeros(n_sims)
    for sim in range(n_sims):
        for week in range(n_weeks):
            points = weekly[sim, :, week]
            active = [
                Player(p.player_id, p.position, float(points[i]))
                for i, p in enumerate(roster)
                if points[i] > 0
            ]
            if active:
                totals[sim] += lineup_value(active, slots)
    return totals


def simulate_roster(
    roster: Sequence[SimPlayer],
    slots: SlotConfig,
    model: AvailabilityModel,
    byes: dict[str, int],
    rng: np.random.Generator,
    *,
    n_sims: int,
    n_weeks: int,
    dispersion: dict[str, float],
) -> np.ndarray:
    """Season points the roster starts, per sim."""
    weekly = simulate_players(roster, model, byes, rng, n_sims=n_sims, n_weeks=n_weeks,
                              dispersion=dispersion)
    return roster_value(roster, weekly, slots)


def simulated_marginal_value(
    player: SimPlayer,
    roster: Sequence[SimPlayer],
    slots: SlotConfig,
    model: AvailabilityModel,
    byes: dict[str, int],
    *,
    seed: int,
    n_sims: int,
    n_weeks: int,
    dispersion: dict[str, float],
) -> float:
    """What `player` adds to a roster across simulated seasons.

    Both rosters are simulated from the same seed — common random numbers — so the
    difference measures the player rather than the noise between two independent runs.
    """
    kwargs = {"n_sims": n_sims, "n_weeks": n_weeks, "dispersion": dispersion}
    without = simulate_roster(roster, slots, model, byes, np.random.default_rng(seed), **kwargs)
    with_him = simulate_roster([*roster, player], slots, model, byes,
                               np.random.default_rng(seed), **kwargs)
    return float((with_him - without).mean())
