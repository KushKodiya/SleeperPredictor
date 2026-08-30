"""M2 — read-only Sleeper API client.

Calls only the endpoints verified in PRD §6.2. No auth, no writes. Adds timeout,
exponential-backoff retry on transient failures, and a token-bucket limiter kept
well under 1000/min. All IDs stay strings (they exceed 32-bit range); `roster_id`
is normalized to str (it arrives as a string in picks, an int in rosters).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, field_validator

BASE_URL = "https://api.sleeper.app/v1"
DEFAULT_RATE_PER_MIN = 600  # well under Sleeper's 1000/min ceiling


class SleeperError(RuntimeError):
    """Raised when a request fails after exhausting retries."""


# --- models -----------------------------------------------------------------
# Sleeper objects carry many fields; we model the ones we use and ignore the rest.


def _as_str(v: Any) -> Any:
    return v if v is None else str(v)


class _SleeperModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class Pick(_SleeperModel):
    player_id: str  # team abbreviation (e.g. "DET") for defenses
    picked_by: str = ""  # can be empty string when a slot has no user
    roster_id: str | None = None
    round: int
    draft_slot: int
    pick_no: int
    metadata: dict = {}
    is_keeper: bool | None = None
    draft_id: str | None = None

    @field_validator("player_id", "picked_by", "roster_id", "draft_id", mode="before")
    @classmethod
    def _stringify(cls, v: Any) -> Any:
        return _as_str(v)

    @property
    def has_picker(self) -> bool:
        return bool(self.picked_by)

    @property
    def is_defense(self) -> bool:
        return self.player_id.isalpha() and self.player_id.isupper()


class Roster(_SleeperModel):
    roster_id: str  # int in the API; normalized to str
    owner_id: str | None = None
    players: list[str] = []
    starters: list[str] = []

    @field_validator("roster_id", "owner_id", mode="before")
    @classmethod
    def _stringify(cls, v: Any) -> Any:
        return _as_str(v)


class User(_SleeperModel):
    user_id: str
    display_name: str | None = None

    @field_validator("user_id", mode="before")
    @classmethod
    def _stringify(cls, v: Any) -> Any:
        return _as_str(v)


class Matchup(_SleeperModel):
    roster_id: str  # int in the API; normalized to str
    matchup_id: str | None = None
    points: float = 0.0
    starters: list[str] = []          # "0" marks an empty slot
    starters_points: list[float] = []
    players_points: dict[str, float] = {}

    @field_validator("roster_id", "matchup_id", mode="before")
    @classmethod
    def _stringify(cls, v: Any) -> Any:
        return _as_str(v)


class BracketMatch(_SleeperModel):
    """One row of a playoff bracket.

    Verified live on 2026-08-28 against the owner's league. PRD M18 describes this shape
    wrongly: it claims `t1`/`t2` hold "roster ids, or `{w: n}` / `{l: n}` referencing
    another match". They do not. `t1`/`t2` are a roster id or null, and the references
    live in their own `t1_from` / `t2_from` keys. `p` — the placement the match decides —
    is not mentioned in the PRD at all.
    """

    r: int                              # round
    m: str                              # match id
    t1: str | None = None               # roster id, or null until the slot is filled
    t2: str | None = None
    t1_from: dict | None = None         # {"w": match} or {"l": match}
    t2_from: dict | None = None
    w: str | None = None                # winner roster id, null until played
    l: str | None = None
    p: int | None = None                # placement decided here: 1 final, 3 third, 5 fifth

    @field_validator("m", "t1", "t2", "w", "l", mode="before")
    @classmethod
    def _stringify(cls, v: Any) -> Any:
        return _as_str(v)

    @property
    def decided(self) -> bool:
        """Whether this match has been played. An unplayed bracket is not a malformed one."""
        return self.w is not None


class League(_SleeperModel):
    league_id: str
    previous_league_id: str | None = None  # prior season's league; the golden-test source
    season: str | None = None              # string in the API, e.g. "2025"
    scoring_settings: dict = {}
    roster_positions: list[str] = []
    total_rosters: int | None = None
    draft_id: str | None = None
    settings: dict = {}
    status: str | None = None

    @field_validator("league_id", "draft_id", "previous_league_id", mode="before")
    @classmethod
    def _stringify(cls, v: Any) -> Any:
        return _as_str(v)


class Draft(_SleeperModel):
    draft_id: str
    status: str | None = None
    type: str | None = None
    season: str | None = None          # string in the API, e.g. "2025"
    league_id: str | None = None
    metadata: dict = {}                # carries scoring_type
    settings: dict = {}
    draft_order: dict | None = None
    slot_to_roster_id: dict | None = None

    @field_validator("draft_id", "league_id", mode="before")
    @classmethod
    def _stringify(cls, v: Any) -> Any:
        return _as_str(v)


# --- rate limiter -----------------------------------------------------------


class _RateLimiter:
    """Token bucket, wall-clock.

    ponytail: single-threaded poller well under 1000/min; a global bucket is
    plenty. Swap for per-endpoint buckets only if we ever parallelize requests.
    """

    def __init__(
        self,
        per_minute: int,
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._capacity = float(per_minute)
        self._tokens = float(per_minute)
        self._per_sec = per_minute / 60.0
        self._last = clock()
        self._sleep = sleep
        self._clock = clock

    def acquire(self) -> None:
        now = self._clock()
        self._tokens = min(self._capacity, self._tokens + (now - self._last) * self._per_sec)
        self._last = now
        if self._tokens < 1.0:
            self._sleep((1.0 - self._tokens) / self._per_sec)
            self._tokens = 0.0
        else:
            self._tokens -= 1.0


# --- client -----------------------------------------------------------------


class SleeperClient:
    def __init__(
        self,
        *,
        timeout: float = 10.0,
        max_retries: int = 3,
        rate_per_min: int = DEFAULT_RATE_PER_MIN,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        backoff_base: float = 0.5,
    ) -> None:
        self._client = httpx.Client(base_url=BASE_URL, timeout=timeout, transport=transport)
        self._max_retries = max_retries
        self._sleep = sleep
        self._backoff_base = backoff_base
        self._limiter = _RateLimiter(rate_per_min, sleep=sleep)

    def _get_json(self, path: str) -> Any:
        last: Exception | None = None
        for attempt in range(self._max_retries + 1):
            self._limiter.acquire()
            try:
                resp = self._client.get(path)
            except httpx.TimeoutException as exc:  # transient → retry
                last = exc
            else:
                if resp.status_code < 500:
                    resp.raise_for_status()  # 4xx raises immediately, no retry
                    return resp.json()
                last = httpx.HTTPStatusError(
                    f"{resp.status_code} from {path}", request=resp.request, response=resp
                )
            if attempt < self._max_retries:
                self._sleep(self._backoff_base * (2**attempt))
        raise SleeperError(
            f"GET {path} failed after {self._max_retries + 1} attempts"
        ) from last

    # typed getters (all §6.2)
    def get_draft_picks(self, draft_id: str) -> list[Pick]:
        return [Pick.model_validate(p) for p in self._get_json(f"/draft/{draft_id}/picks")]

    def get_league(self, league_id: str) -> League:
        return League.model_validate(self._get_json(f"/league/{league_id}"))

    def get_draft(self, draft_id: str) -> Draft:
        return Draft.model_validate(self._get_json(f"/draft/{draft_id}"))

    def get_rosters(self, league_id: str) -> list[Roster]:
        return [Roster.model_validate(r) for r in self._get_json(f"/league/{league_id}/rosters")]

    def get_matchups(self, league_id: str, week: int) -> list[Matchup]:
        return [
            Matchup.model_validate(m)
            for m in self._get_json(f"/league/{league_id}/matchups/{week}")
        ]

    def get_winners_bracket(self, league_id: str) -> list[BracketMatch]:
        """The league's playoff bracket, as it actually publishes it.

        Sleeper serves a fully-formed bracket even for a league in `pre_draft` status,
        with every `w` and `l` null — so the presence of a bracket says nothing about
        whether the season has been played. Use `BracketMatch.decided` for that.
        """
        return [
            BracketMatch.model_validate(m)
            for m in self._get_json(f"/league/{league_id}/winners_bracket")
        ]

    def get_losers_bracket(self, league_id: str) -> list[BracketMatch]:
        """The consolation bracket, same row shape as the winners bracket."""
        return [
            BracketMatch.model_validate(m)
            for m in self._get_json(f"/league/{league_id}/losers_bracket")
        ]

    def get_user_drafts(self, user_id: str, season: int) -> list[Draft]:
        """Every draft a user took part in that season — the opponent model's corpus.

        This is the only way to reach a draft: through a known user or league id. Sleeper
        has no endpoint that enumerates public drafts, so there is nothing to crawl
        (PRD §11.9).
        """
        return [
            Draft.model_validate(d)
            for d in self._get_json(f"/user/{user_id}/drafts/nfl/{season}")
        ]

    def get_users(self, league_id: str) -> list[User]:
        return [User.model_validate(u) for u in self._get_json(f"/league/{league_id}/users")]

    def players_nfl(self, cache_path: Path, ttl_hours: float) -> dict:
        """The ~5MB player map, cached to disk and refreshed at most once per TTL."""
        if cache_path.exists():
            age_hours = (time.time() - cache_path.stat().st_mtime) / 3600.0
            if age_hours < ttl_hours:
                return json.loads(cache_path.read_text(encoding="utf-8"))
        data = self._get_json("/players/nfl")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(data), encoding="utf-8")
        return data

    def close(self) -> None:
        self._client.close()
