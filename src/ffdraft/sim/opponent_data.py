"""M11 — assembling opponent-model training data from real drafts.

`opponent.py` is the pure part; this is the part that fetches. Every draft here is
reached through a `user_id` or `league_id` already known from the owner's own league —
Sleeper publishes no way to enumerate drafts, so there is no crawl to perform and none
is attempted (PRD §11.9).

Each draft is scored against the ADP for *its own* season and scoring format, and
positions are converted to rounds so drafts of different sizes are comparable. A draft
whose settings cannot be resolved is excluded and reported, never pooled into a market
that means something else.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

from ffdraft.data import nflverse
from ffdraft.data.adp import fetch_adp
from ffdraft.data.crosswalk import build_crosswalk, resolve_frame
from ffdraft.data.sleeper import SleeperClient
from ffdraft.sim.opponent import (
    Choice,
    ExcludedDraft,
    ObservedPick,
    TrainingDraft,
    features,
    open_positions,
    resolve_draft_settings,
    to_round,
)
from ffdraft.sim.outcomes import bye_weeks


def collect_training_drafts(
    client: SleeperClient, *, league_id: str, seasons: Sequence[int]
) -> tuple[list[TrainingDraft], list[ExcludedDraft]]:
    """Every completed draft the leaguemates took part in, with settings resolved.

    The reachability chain is the whole point: league -> users -> that user's drafts.
    There is no step that discovers a draft belonging to a stranger.
    """
    usable: list[TrainingDraft] = []
    excluded: list[ExcludedDraft] = []
    seen: set[str] = set()

    for user in client.get_users(league_id):
        for season in seasons:
            try:
                drafts = client.get_user_drafts(user.user_id, season)
            except Exception as exc:  # noqa: BLE001 - one unreachable user must not stop training
                excluded.append(ExcludedDraft(f"user:{user.user_id}", season, repr(exc)))
                continue

            for draft in drafts:
                if draft.draft_id in seen:
                    continue
                seen.add(draft.draft_id)
                draft_season = int(draft.season) if draft.season else season

                if draft.status != "complete":
                    excluded.append(
                        ExcludedDraft(draft.draft_id, draft_season, f"status={draft.status}")
                    )
                    continue
                settings, reason = resolve_draft_settings(draft)
                if reason is not None:
                    excluded.append(ExcludedDraft(draft.draft_id, draft_season, reason))
                    continue

                picks = tuple(
                    ObservedPick(
                        draft_id=draft.draft_id, pick_no=p.pick_no, user_id=p.picked_by,
                        player_id=p.player_id,
                        position=(p.metadata or {}).get("position") or "",
                        nfl_team=(p.metadata or {}).get("team"),
                    )
                    for p in client.get_draft_picks(draft.draft_id)
                    if p.picked_by  # an empty slot has no manager to learn from
                )
                if not picks:
                    excluded.append(ExcludedDraft(draft.draft_id, draft_season, "no attributed picks"))
                    continue
                usable.append(
                    TrainingDraft(
                        draft_id=draft.draft_id, season=draft_season, teams=settings["teams"],
                        rounds=settings["rounds"], scoring_type=settings["scoring_type"],
                        slots=settings["slots"], picks=picks,
                    )
                )
    return usable, excluded


def adp_baseline(
    draft: TrainingDraft, crosswalk: pl.DataFrame, *, fuzzy_threshold: int
) -> dict[str, float]:
    """That draft's own ADP market, keyed on `gsis_id`.

    Season and scoring format are honoured. Team count cannot be — FFC serves only
    12-team ADP for past seasons — which is why positions are compared in rounds.
    """
    adp = fetch_adp(draft.ffc_format, draft.teams, draft.season)
    matched, _ = resolve_frame(adp, crosswalk, fuzzy_threshold=fuzzy_threshold)
    return dict(zip(matched["gsis_id"].to_list(), matched["adp"].to_list(), strict=True))


def build_choices(
    drafts: Sequence[TrainingDraft],
    crosswalk: pl.DataFrame,
    sleeper_to_gsis: dict[str, str],
    *,
    flex_eligibility: dict[str, list[str]],
    fuzzy_threshold: int,
    byes_by_season: dict[int, dict[str, int]] | None = None,
) -> tuple[list[Choice], int]:
    """Turn observed picks into conditional-logit training examples.

    Returns (choices, skipped). A pick of a player outside that season's published ADP
    has no prior to deviate from, so it is skipped and counted rather than guessed at.
    """
    byes_by_season = byes_by_season or {}
    choices: list[Choice] = []
    skipped = 0

    for draft in drafts:
        adp = adp_baseline(draft, crosswalk, fuzzy_threshold=fuzzy_threshold)
        if not adp:
            continue
        byes = byes_by_season.get(draft.season, {})
        universe = sorted(adp, key=lambda g: adp[g])

        available = list(universe)
        positions: dict[str, str] = {}
        teams: dict[str, str] = {}
        taken_by: dict[str, list[str]] = {}
        team_counts: dict[str, dict[str, int]] = {}
        recent: list[str] = []

        for pick in sorted(draft.picks, key=lambda p: p.pick_no):
            gsis = sleeper_to_gsis.get(pick.player_id)
            positions.setdefault(gsis or "", pick.position)
            teams.setdefault(gsis or "", pick.nfl_team or "")

            if gsis is None or gsis not in available:
                skipped += 1
            else:
                index = available.index(gsis)
                candidate_positions = np.array(
                    [positions.get(g, _position_of(g, crosswalk)) for g in available]
                )
                candidate_teams = np.array([teams.get(g, "") for g in available])
                candidate_byes = np.array([byes.get(teams.get(g, ""), 0) for g in available])
                mine = taken_by.setdefault(pick.user_id, [])
                counts = team_counts.setdefault(pick.user_id, {})
                total = sum(counts.values()) or 1

                choices.append(
                    Choice(
                        user_id=pick.user_id,
                        adp_rounds=np.array([to_round(adp[g], draft.teams) for g in available]),
                        features=features(
                            candidate_positions=candidate_positions,
                            candidate_teams=candidate_teams,
                            open_positions=open_positions(
                                [positions.get(g, "") for g in mine], draft.slots, flex_eligibility
                            ),
                            recent_positions=recent,
                            manager_team_share={t: c / total for t, c in counts.items()},
                            roster_byes={byes.get(teams.get(g, ""), 0) for g in mine} - {0},
                            candidate_byes=candidate_byes,
                        ),
                        chosen=index,
                    )
                )
                available.pop(index)
                mine.append(gsis)
                counts[pick.nfl_team or ""] = counts.get(pick.nfl_team or "", 0) + 1
            recent.append(pick.position)
    return choices, skipped


_POSITION_CACHE: dict[int, dict[str, str]] = {}


def _position_of(gsis: str, crosswalk: pl.DataFrame) -> str:
    """Position for a player nobody in this draft picked, from the crosswalk."""
    key = id(crosswalk)
    if key not in _POSITION_CACHE:
        _POSITION_CACHE[key] = dict(
            zip(crosswalk["gsis_id"].to_list(), crosswalk["position"].to_list(), strict=True)
        )
    return _POSITION_CACHE[key].get(gsis, "")


def training_inputs(
    client: SleeperClient, cfg, *, seasons: Sequence[int], refresh: bool = False
):
    """Everything needed to fit the model, straight from the live endpoints."""
    ids = nflverse.ff_playerids(refresh=refresh)
    crosswalk = build_crosswalk(ids)
    sleeper_to_gsis = {
        str(s): g
        for s, g in zip(ids["sleeper_id"].to_list(), ids["gsis_id"].to_list(), strict=True)
        if s is not None and g is not None
    }
    drafts, excluded = collect_training_drafts(
        client, league_id=cfg.league.league_id, seasons=seasons
    )
    byes = {
        season: bye_weeks(nflverse.schedules([season], refresh=refresh), season=season)
        for season in sorted({d.season for d in drafts})
    }
    choices, skipped = build_choices(
        drafts, crosswalk, sleeper_to_gsis,
        flex_eligibility=cfg.flex_eligibility,
        fuzzy_threshold=cfg.crosswalk.fuzzy_threshold,
        byes_by_season=byes,
    )
    return choices, drafts, excluded, skipped
