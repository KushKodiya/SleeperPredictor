"""Playoff bracket parsing — against the owner's real, frozen 2025 bracket.

PRD M18 describes this endpoint's shape wrongly, so these tests pin the shape that
actually comes back rather than the one the document claims.
"""

import json
from pathlib import Path

import pytest

from ffdraft.data.sleeper import BracketMatch, SleeperClient

FIXTURE = Path(__file__).parent / "fixtures" / "winners_bracket_2025.json"


@pytest.fixture(scope="module")
def bracket_json():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def bracket(bracket_json):
    return [BracketMatch.model_validate(m) for m in bracket_json["winners_bracket"]]


# --- 1.1 parsing ---------------------------------------------------------------------


def test_the_fixture_contains_no_league_id():
    """The repo is public; a league id would let anyone look up the owner (§11)."""
    blob = FIXTURE.read_text(encoding="utf-8")
    assert "league_id" not in blob
    # Sleeper ids are 18-19 digit strings; no bare token in the fixture may look like one.
    for token in blob.replace('"', " ").replace(":", " ").split():
        assert not (token.strip(",").isdigit() and len(token.strip(",")) > 12)


def test_slot_references_come_from_their_own_keys_not_from_t1_t2(bracket):
    """M18 claims t1/t2 hold {w: n}; they never do. Code written to the PRD finds nothing."""
    referencing = [m for m in bracket if m.t1_from or m.t2_from]
    assert referencing, "fixture should contain advancement matches"
    for match in referencing:
        for side in (match.t1, match.t2):
            assert side is None or isinstance(side, str)
        for ref in (match.t1_from, match.t2_from):
            assert ref is None or set(ref) <= {"w", "l"}


def test_t1_and_t2_are_roster_ids_never_seeds(bracket):
    """Seeding must be read from the standings; these are roster ids."""
    rosters = {m.t1 for m in bracket} | {m.t2 for m in bracket}
    assert all(r is None or r.isdigit() for r in rosters)


def test_the_placement_key_is_retained(bracket):
    """`p` is absent from the PRD entirely, and it is how the final is identified."""
    placements = sorted(m.p for m in bracket if m.p is not None)
    assert 1 in placements  # the championship game
    assert placements == [1, 3, 5]


def test_a_six_team_bracket_has_seven_winners_rows(bracket):
    assert len(bracket) == 7
    assert sorted({m.r for m in bracket}) == [1, 2, 3]


def test_every_match_in_a_completed_bracket_is_decided(bracket):
    assert all(m.decided for m in bracket)
    assert all(m.w is not None and m.l is not None for m in bracket)


# --- 1.2 settings and the unplayed case ----------------------------------------------


def test_playoff_structure_is_read_from_settings_not_assumed(bracket_json):
    settings = bracket_json["settings"]
    assert settings["playoff_teams"] == 6
    assert settings["playoff_week_start"] == 14
    assert settings["num_teams"] == 8


def test_an_unplayed_bracket_is_undecided_not_malformed():
    """A pre_draft league returns a full bracket with every w/l null."""
    row = BracketMatch.model_validate({"r": 1, "m": 1, "t1": 4, "t2": 7, "w": None, "l": None})
    assert not row.decided
    assert row.t1 == "4" and row.t2 == "7"


def test_ids_are_strings_never_integers():
    row = BracketMatch.model_validate({"r": 1, "m": 1, "t1": 4, "t2": 7, "w": 4, "l": 7})
    assert (row.m, row.t1, row.w) == ("1", "4", "4")


# --- 1.3 the no-enumeration guarantee ------------------------------------------------


def test_the_client_exposes_no_draft_enumeration_operation():
    """The opponent model's no-crawl guarantee rests on this being true (PRD §11.9)."""
    operations = {n for n in dir(SleeperClient) if not n.startswith("_")}
    forbidden = {"search_drafts", "list_drafts", "all_drafts", "public_drafts",
                 "browse_drafts", "sample_drafts", "find_drafts"}
    assert not (operations & forbidden)


def test_every_draft_getter_requires_an_identifier_the_owner_already_has():
    """league -> users -> that user's drafts. No step discovers a stranger's draft."""
    import inspect

    for name, expected in (
        ("get_draft", "draft_id"),
        ("get_draft_picks", "draft_id"),
        ("get_user_drafts", "user_id"),
        ("get_users", "league_id"),
        ("get_winners_bracket", "league_id"),
    ):
        params = inspect.signature(getattr(SleeperClient, name)).parameters
        assert expected in params, f"{name} must be reached via {expected}"
