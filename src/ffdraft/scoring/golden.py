"""The golden reproduction: score the owner's prior season from raw stats and compare.

This is the project's canary (PRD §10). If the engine cannot reproduce every team's
reported `points` for every week of last season to within 0.01, every valuation and
simulation downstream is quietly wrong.

`build_fixture` reaches the network once and freezes everything the check needs into
`tests/fixtures/golden/`; `reproduce` is pure and is what the CI test runs, so no test
ever hits the network.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from ffdraft.data import nflverse
from ffdraft.data.sleeper import SleeperClient
from ffdraft.scoring import statlines
from ffdraft.scoring.engine import parse_settings, score_defenses, score_players

FIXTURE_DIR = Path("tests/fixtures/golden")
EMPTY_SLOT = "0"
TOLERANCE = 0.01


@dataclass(frozen=True)
class GoldenFixture:
    season: int
    scoring_settings: dict
    matchups: list[dict]  # one entry per team-week
    player_stats: pl.DataFrame
    defense_stats: pl.DataFrame
    sleeper_to_gsis: dict[str, str]


def _is_defense(sleeper_id: str) -> bool:
    """Team defenses use the team abbreviation as their player id (PRD §6.2)."""
    return sleeper_id.isalpha() and sleeper_id.isupper()


def build_fixture(
    league_id: str,
    season: int,
    *,
    client: SleeperClient,
    weeks: range = range(1, 19),
    out_dir: Path = FIXTURE_DIR,
    refresh: bool = False,
) -> Path:
    """Freeze one season of the owner's league plus the raw stats needed to re-score it."""
    league = client.get_league(league_id)
    matchups: list[dict] = []
    for week in weeks:
        for m in client.get_matchups(league_id, week):
            matchups.append(
                {
                    "week": week,
                    "roster_id": m.roster_id,
                    "points": m.points,
                    "starters": m.starters,
                    "starters_points": m.starters_points,
                }
            )

    pbp = nflverse.pbp([season], refresh=refresh)
    stats = nflverse.player_stats([season], refresh=refresh)
    players = statlines.player_week_stats(stats, pbp)
    defenses = statlines.defense_week_stats(
        pbp, nflverse.team_stats([season], refresh=refresh), nflverse.schedules([season], refresh=refresh)
    )

    ff = nflverse.ff_playerids(refresh=refresh)
    crosswalk = (
        ff.filter(pl.col("sleeper_id").is_not_null() & pl.col("gsis_id").is_not_null())
        .with_columns(pl.col("sleeper_id").cast(pl.String))
        .unique(subset=["sleeper_id"])
    )
    started = {s for m in matchups for s in m["starters"] if s != EMPTY_SLOT}
    sleeper_to_gsis = {
        sid: gid
        for sid, gid in zip(
            crosswalk["sleeper_id"].to_list(), crosswalk["gsis_id"].to_list(), strict=True
        )
        if sid in started
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "league.json").write_text(
        json.dumps(
            # The league id is deliberately not written out: it is unused by the
            # reproduction and resolves, through the open Sleeper API, to every
            # leaguemate's display name. This fixture is committed to a public repo.
            {"season": season, "scoring_settings": league.scoring_settings},
            indent=1,
        ),
        encoding="utf-8",
    )
    (out_dir / "matchups.json").write_text(json.dumps(matchups, indent=1), encoding="utf-8")
    (out_dir / "sleeper_to_gsis.json").write_text(
        json.dumps(sleeper_to_gsis, indent=1), encoding="utf-8"
    )
    players.filter(pl.col("player_id").is_in(list(sleeper_to_gsis.values()))).write_parquet(
        out_dir / "player_stats.parquet"
    )
    defenses.write_parquet(out_dir / "defense_stats.parquet")
    return out_dir


def load_fixture(path: Path = FIXTURE_DIR) -> GoldenFixture:
    league = json.loads((path / "league.json").read_text(encoding="utf-8"))
    return GoldenFixture(
        season=league["season"],
        scoring_settings=league["scoring_settings"],
        matchups=json.loads((path / "matchups.json").read_text(encoding="utf-8")),
        player_stats=pl.read_parquet(path / "player_stats.parquet"),
        defense_stats=pl.read_parquet(path / "defense_stats.parquet"),
        sleeper_to_gsis=json.loads((path / "sleeper_to_gsis.json").read_text(encoding="utf-8")),
    )


def reproduce(fixture: GoldenFixture) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Re-score every team-week from raw stats.

    Returns (team_weeks, starters): one row per team-week with reported vs computed
    points, and one row per started player with the same comparison — the per-player
    frame is what tells you *which* rule is wrong when a team total is off.
    """
    rules = parse_settings(fixture.scoring_settings)
    players = {
        (r["week"], r["player_id"]): r["points"]
        for r in score_players(fixture.player_stats, rules).iter_rows(named=True)
    }
    defenses = {
        (r["week"], r["team"]): r["points"]
        for r in score_defenses(fixture.defense_stats, rules).iter_rows(named=True)
    }
    # A team on bye has no row that week; that is a real 0, not an unresolved id.
    known_teams = set(fixture.defense_stats["team"].to_list())

    starter_rows, team_rows = [], []
    for m in fixture.matchups:
        total = 0.0
        reported_by_slot = m["starters_points"]
        for slot, sleeper_id in enumerate(m["starters"]):
            reported = reported_by_slot[slot] if slot < len(reported_by_slot) else 0.0
            if sleeper_id == EMPTY_SLOT:
                computed, resolved = 0.0, True
            elif _is_defense(sleeper_id):
                computed = defenses.get((m["week"], sleeper_id), 0.0)
                resolved = sleeper_id in known_teams
            else:
                gsis = fixture.sleeper_to_gsis.get(sleeper_id)
                # No stat row means the player did not play that week, which scores 0.
                # An id absent from the crosswalk is a different failure and is reported.
                computed = players.get((m["week"], gsis), 0.0) if gsis else 0.0
                resolved = gsis is not None
            total += computed
            starter_rows.append(
                {
                    "week": m["week"],
                    "roster_id": m["roster_id"],
                    "sleeper_id": sleeper_id,
                    "resolved": resolved,
                    "reported": reported,
                    "computed": computed,
                    "diff": round(reported - computed, 2),
                }
            )
        team_rows.append(
            {
                "week": m["week"],
                "roster_id": m["roster_id"],
                "reported": m["points"],
                "computed": round(total, 2),
                "diff": round(m["points"] - total, 2),
            }
        )
    return pl.DataFrame(team_rows), pl.DataFrame(starter_rows)
