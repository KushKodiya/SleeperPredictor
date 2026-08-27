"""M2 tests. No network: httpx.MockTransport + frozen fixtures (PRD §10)."""

import json
from pathlib import Path

import httpx
import pytest

from ffdraft.data.sleeper import Pick, Roster, SleeperClient, SleeperError

FIXTURES = Path(__file__).parent / "fixtures"


def _noop_sleep(_seconds):
    return None


def test_retries_then_raises_on_repeated_5xx():
    calls = []

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(500)

    client = SleeperClient(
        max_retries=2, transport=httpx.MockTransport(handler), sleep=_noop_sleep
    )
    with pytest.raises(SleeperError):
        client.get_draft_picks("x")
    assert len(calls) == 3  # initial + 2 retries


def test_4xx_raises_without_retrying():
    calls = []

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(404)

    client = SleeperClient(
        max_retries=3, transport=httpx.MockTransport(handler), sleep=_noop_sleep
    )
    with pytest.raises(httpx.HTTPStatusError):
        client.get_league("x")
    assert len(calls) == 1  # no retry on client error


def test_parses_picks_with_empty_picked_by_and_defense():
    picks = [Pick.model_validate(p) for p in json.loads((FIXTURES / "sleeper_picks.json").read_text())]
    by_player = {p.player_id: p for p in picks}

    # empty picked_by is handled, attributed to no user
    mccaffrey = by_player["4034"]
    assert mccaffrey.picked_by == ""
    assert mccaffrey.has_picker is False

    # defense pick parses; player_id is a team abbreviation
    lions = by_player["DET"]
    assert lions.is_defense is True

    # roster_id stays a string
    assert isinstance(mccaffrey.roster_id, str)


def test_roster_id_int_is_normalized_to_str():
    rosters = [Roster.model_validate(r) for r in json.loads((FIXTURES / "sleeper_rosters.json").read_text())]
    assert rosters[0].roster_id == "1"  # int 1 in source -> "1"
    assert isinstance(rosters[0].roster_id, str)
    assert rosters[1].owner_id is None


def test_players_nfl_respects_ttl(tmp_path):
    calls = []

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(200, json={"2391": {"position": "RB"}})

    client = SleeperClient(transport=httpx.MockTransport(handler), sleep=_noop_sleep)
    cache = tmp_path / "players.json"

    first = client.players_nfl(cache, ttl_hours=24)
    second = client.players_nfl(cache, ttl_hours=24)  # within TTL -> disk, no refetch

    assert first == second
    assert len(calls) == 1
