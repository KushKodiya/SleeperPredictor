"""M18 — a drafted league played out to a champion.

Finishing fifth pays nothing. Championship probability is a convex function of team
strength, so the right strategy is somewhat variance-seeking, and expected points cannot
express that. This module turns the rosters a draft rollout already produces into each
team's probability of winning the league.

The playoff structure is **reproduced from what the league publishes**, never assumed. A
bracket row's slot references live in `t1_from` / `t2_from`, not inside `t1` / `t2`, and
`t1` / `t2` hold roster ids rather than seeds — see the R2 log; PRD M18 describes this
wrongly and code written from it looks for a reference that is never there.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from ffdraft.data.sleeper import BracketMatch
from ffdraft.lineup.slots import SlotConfig
from ffdraft.lineup.value import vectorized_lineup_value
from ffdraft.sim.availability import AvailabilityModel
from ffdraft.sim.outcomes import SimPlayer, simulate_players


@dataclass(frozen=True)
class LeagueSeason:
    """One simulated season: weekly points per team, and what they added up to."""

    weekly: dict[int, np.ndarray]  # roster id -> (n_sims, n_weeks)
    wins: dict[int, np.ndarray]  # roster id -> (n_sims,)
    points_for: dict[int, np.ndarray]  # roster id -> (n_sims,)


def simulate_rosters(
    rosters: dict[int, list[SimPlayer]],
    availability: AvailabilityModel,
    byes: dict[str, int],
    slots: SlotConfig,
    *,
    rng: np.random.Generator,
    n_sims: int,
    n_weeks: int,
    dispersion: dict[str, float],
    crn_seed: int | None = None,
) -> dict[int, np.ndarray]:
    """Weekly starting-lineup points for every team: roster id -> `(n_sims, n_weeks)`."""
    weekly: dict[int, np.ndarray] = {}
    for roster_id, players in sorted(rosters.items()):
        if not players:
            weekly[roster_id] = np.zeros((n_sims, n_weeks))
            continue
        scores = simulate_players(
            players, availability, byes, rng,
            n_sims=n_sims, n_weeks=n_weeks, dispersion=dispersion, crn_seed=crn_seed,
        )
        weekly[roster_id] = vectorized_lineup_value(
            scores, [p.position for p in players], slots
        )
    return weekly


def play_schedule(
    weekly: dict[int, np.ndarray], schedule: Sequence[Sequence[tuple[int, int]]]
) -> LeagueSeason:
    """Run the league's own weekly matchups into wins and points for.

    `schedule[w]` is the pairings for week `w`. A tie counts half a win to each side,
    which is how Sleeper standings treat it.
    """
    any_team = next(iter(weekly.values()))
    n_sims = any_team.shape[0]
    wins = {team: np.zeros(n_sims) for team in weekly}
    for week, pairings in enumerate(schedule):
        for home, away in pairings:
            home_points, away_points = weekly[home][:, week], weekly[away][:, week]
            wins[home] += np.where(
                home_points > away_points, 1.0, np.where(home_points == away_points, 0.5, 0.0)
            )
            wins[away] += np.where(
                away_points > home_points, 1.0, np.where(home_points == away_points, 0.5, 0.0)
            )
    weeks = len(schedule)
    return LeagueSeason(
        weekly=weekly,
        wins=wins,
        points_for={team: points[:, :weeks].sum(axis=1) for team, points in weekly.items()},
    )


def seed_order(season: LeagueSeason, sim: int) -> list[int]:
    """Teams ranked for one simulation: wins first, total points breaking the tie.

    Points-for is Sleeper's default tiebreaker and it is what the owner's league uses.
    Roster id breaks a remaining exact tie so the ordering is total and reproducible.
    """
    return sorted(
        season.wins,
        key=lambda team: (-season.wins[team][sim], -season.points_for[team][sim], team),
    )


def entry_slots(bracket: Sequence[BracketMatch]) -> list[tuple[int, str]]:
    """The bracket's entry slots — the ones a team enters at rather than advances into.

    Returns `(match_id, side)` pairs ordered by entry round (later rounds first, since
    entering later means a bye) then by match id, `t1` before `t2`. That ordering is
    topology only: it says who had a bye, **not** what anyone's seed was.

    The published bracket does not encode seeds. Measured against the owner's real 2025
    bracket, `t1` is the higher seed in one first-round match (seed 4 vs 5) and the lower
    in the other (seed 6 vs 3), so no reading of `t1`/`t2` order recovers seeding. Mapping
    seeds onto these slots needs a league's seeding convention, which is why it is a
    separate, explicit step rather than something inferred here.
    """
    entries: list[tuple[int, int, int, str]] = []
    for match in bracket:
        for side in ("t1", "t2"):
            comes_from = match.t1_from if side == "t1" else match.t2_from
            team = match.t1 if side == "t1" else match.t2
            if comes_from is None and team is not None:
                entries.append((-match.r, int(match.m), 0 if side == "t1" else 1, side))
    entries.sort()
    return [(match_id, side) for _, match_id, _, side in entries]


def resolve_bracket(
    bracket: Sequence[BracketMatch],
    assignment: dict[tuple[int, str], int],
    score: Callable[[int, int], float],
) -> dict[int, int]:
    """Play a published bracket out. Returns `placement -> roster id`.

    `assignment` fills the entry slots (`(match_id, side) -> roster id`); `score(team,
    round)` is that team's points in the week that round is played. Slots not filled
    directly are resolved through `t1_from` / `t2_from`, which is where the references
    actually live.
    """
    winners: dict[int, int] = {}
    losers: dict[int, int] = {}
    placements: dict[int, int] = {}

    for match in sorted(bracket, key=lambda m: (m.r, int(m.m))):
        match_id = int(match.m)

        def side_team(side: str, *, _match: BracketMatch = match, _id: int = match_id):
            direct = assignment.get((_id, side))
            if direct is not None:
                return direct
            comes_from = _match.t1_from if side == "t1" else _match.t2_from
            if comes_from is None:
                return None
            if "w" in comes_from:
                return winners.get(int(comes_from["w"]))
            return losers.get(int(comes_from["l"]))

        home, away = side_team("t1"), side_team("t2")
        if home is None or away is None:
            continue
        home_points, away_points = score(home, match.r), score(away, match.r)
        winner, loser = (home, away) if home_points >= away_points else (away, home)
        winners[match_id], losers[match_id] = winner, loser
        if match.p is not None:
            placements[match.p] = winner
            placements[match.p + 1] = loser
    return placements


def champion(
    bracket: Sequence[BracketMatch],
    ranking: Sequence[int],
    slot_seeds: Sequence[int],
    score: Callable[[int, int], float],
) -> int | None:
    """Who wins the title, given a standings ranking and the seed each entry slot holds.

    `slot_seeds[i]` is the 1-based seed belonging in `entry_slots(bracket)[i]`. It is an
    input rather than a derivation because the published bracket does not encode seeds
    (see `entry_slots`).
    """
    slots = entry_slots(bracket)
    assignment = {
        slot: ranking[seed - 1]
        for slot, seed in zip(slots, slot_seeds, strict=False)
        if seed - 1 < len(ranking)
    }
    return resolve_bracket(bracket, assignment, score).get(1)


def championship_probabilities(
    season: LeagueSeason,
    bracket: Sequence[BracketMatch],
    *,
    playoff_week_start: int,
    slot_seeds: Sequence[int],
) -> dict[int, float]:
    """Each team's share of simulated titles. Sums to 1.0 across the league.

    `slot_seeds` is the learned mapping from `learn_slot_seeds` — which seed belongs in
    which bracket slot. Round `r` is played in week `playoff_week_start + r - 1`, and
    Sleeper's weeks are 1-indexed where the simulated arrays are 0-indexed.

    Every team appears in the result, including those that never make the playoffs: a
    missing key would read as "no data" where the truth is a probability of zero (R4).
    """
    n_sims = next(iter(season.wins.values())).shape[0]
    titles = dict.fromkeys(season.wins, 0)

    for sim in range(n_sims):
        ranking = seed_order(season, sim)

        def score(team: int, rnd: int, _sim: int = sim) -> float:
            week = playoff_week_start + rnd - 2  # -1 for the round, -1 for 0-indexing
            column = season.weekly[team]
            return float(column[_sim, week]) if 0 <= week < column.shape[1] else 0.0

        winner = champion(bracket, ranking, slot_seeds, score)
        if winner is not None:
            titles[winner] += 1

    return {team: count / n_sims for team, count in titles.items()}


# --- learning the seeding convention from a season that was actually played ----------


class SeedingUnlearnable(ValueError):
    """Raised when a completed season cannot teach the slot-to-seed mapping."""


def standings_from_matchups(
    matchups_by_week: dict[int, Sequence[tuple[int, float]]],
    pairings_by_week: dict[int, Sequence[tuple[int, int]]],
) -> list[int]:
    """Regular-season ranking from real results: wins, then points for, then roster id.

    `matchups_by_week[w]` is `(roster_id, points)`; `pairings_by_week[w]` is who played
    whom. Ties count half a win each, as Sleeper standings do.
    """
    wins: dict[int, float] = {}
    points: dict[int, float] = {}
    for week, rows in matchups_by_week.items():
        for roster_id, scored in rows:
            points[roster_id] = points.get(roster_id, 0.0) + scored
            wins.setdefault(roster_id, 0.0)
        scores = dict(rows)
        for home, away in pairings_by_week.get(week, ()):
            if home not in scores or away not in scores:
                continue
            if scores[home] > scores[away]:
                wins[home] += 1.0
            elif scores[away] > scores[home]:
                wins[away] += 1.0
            else:
                wins[home] += 0.5
                wins[away] += 0.5
    return sorted(wins, key=lambda team: (-wins[team], -points.get(team, 0.0), team))


def learn_slot_seeds(bracket: Sequence[BracketMatch], ranking: Sequence[int]) -> list[int]:
    """The seed that belongs in each entry slot, learned from a season that was played.

    The published bracket records roster ids, and a completed season's standings say what
    seed each of those rosters held — so the mapping is *observed* rather than assumed. It
    is a property of the league's playoff settings, so it carries forward to seasons that
    have not been played yet.

    Measured on the owner's league, this recovers a layout no ordering rule reproduces:
    `t1` is the higher seed in one first-round match and the lower in the other.
    """
    # An entry position with no team is an unplayed bracket. `entry_slots` skips those
    # by design — an unplayed bracket is undecided, not malformed — but you cannot learn
    # a seeding convention from one, so that must fail here rather than return nothing.
    unfilled = [
        (int(m.m), side)
        for m in bracket
        for side in ("t1", "t2")
        if (m.t1_from if side == "t1" else m.t2_from) is None
        and (m.t1 if side == "t1" else m.t2) is None
    ]
    if unfilled:
        raise SeedingUnlearnable(
            f"entry slots {unfilled} are empty; an unplayed bracket cannot teach the "
            f"seeding convention"
        )

    position = {team: index + 1 for index, team in enumerate(ranking)}
    seeds: list[int] = []
    for match_id, side in entry_slots(bracket):
        match = next(m for m in bracket if int(m.m) == match_id)
        team = match.t1 if side == "t1" else match.t2
        if int(team) not in position:
            raise SeedingUnlearnable(
                f"roster {team} holds a playoff slot but is absent from the standings "
                f"{list(ranking)}; the bracket and the standings are from different seasons"
            )
        seeds.append(position[int(team)])

    if len(set(seeds)) != len(seeds):
        raise SeedingUnlearnable(f"two slots resolved to the same seed: {seeds}")
    return seeds
