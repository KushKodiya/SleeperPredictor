"""The golden canary: reproduce the owner's real league scores from raw stats.

PRD §9 Phase 2 gate — "reproduces the owner's actual league scores for all of last
season to within 0.01. No exceptions." This runs off the frozen fixture in
`tests/fixtures/golden/`, so it never touches the network; rebuild that fixture with
`ffdraft freeze-golden` when the season or the league's settings change.
"""

from pathlib import Path

import polars as pl
import pytest

from ffdraft.scoring.golden import TOLERANCE, load_fixture, reproduce

GOLDEN = Path(__file__).parent / "fixtures" / "golden"


@pytest.fixture(scope="module")
def scored():
    fixture = load_fixture(GOLDEN)
    return fixture, *reproduce(fixture)


def test_fixture_covers_the_whole_season(scored):
    """A truncated fixture must not let the gate pass by scoring nothing."""
    _, teams, _ = scored
    assert teams.height == 144  # 8 teams x 18 weeks
    assert sorted(teams["week"].unique().to_list()) == list(range(1, 19))
    assert teams["reported"].sum() > 0


def test_every_starter_resolves_to_a_stat_line(scored):
    """A player silently missing from the crosswalk scores zero and is never noticed (R4)."""
    _, _, starters = scored
    unresolved = starters.filter(~pl.col("resolved"))
    assert unresolved.is_empty(), unresolved


def test_every_team_week_reproduces_within_a_hundredth(scored):
    _, teams, _ = scored
    failures = teams.filter(pl.col("diff").abs() > TOLERANCE)
    assert failures.is_empty(), failures


def test_every_started_player_reproduces_within_a_hundredth(scored):
    """Stricter than the gate, and it names the rule that broke when the gate fails."""
    _, _, starters = scored
    failures = starters.filter(pl.col("diff").abs() > TOLERANCE)
    assert failures.is_empty(), failures
