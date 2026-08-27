"""M3 tests. Format mapping + crosswalk join are unit-tested; live fetch is
exercised by the gate, not here (PRD §10)."""

import polars as pl

from ffdraft.data.adp import adp_format_from_scoring, join_adp_to_crosswalk
from ffdraft.data.crosswalk import build_crosswalk


def test_format_mapping():
    assert adp_format_from_scoring({"rec": 1.0}) == "ppr"
    assert adp_format_from_scoring({"rec": 0.5}) == "half-ppr"
    assert adp_format_from_scoring({"rec": 0.0}) == "standard"
    assert adp_format_from_scoring({}) == "standard"  # missing -> standard, not zero-impute crash


def _crosswalk():
    ff = pl.DataFrame(
        {
            "gsis_id": ["00-1", "00-2"],
            "sleeper_id": ["s1", "s2"],
            "name": ["Ja'Marr Chase", "Patrick Mahomes"],
            "position": ["WR", "QB"],
            "team": ["CIN", "KC"],
        }
    )
    return build_crosswalk(ff)


def test_join_reports_unmatched_not_dropped():
    adp = pl.DataFrame(
        {
            "name": ["Ja'Marr Chase", "Obscure Rookie"],
            "position": ["WR", "RB"],
            "team": ["CIN", "XXX"],
            "rank": [1, 200],
            "adp": [1.5, 200.0],
        }
    )
    matched, unmatched = join_adp_to_crosswalk(adp, _crosswalk(), fuzzy_threshold=92)
    assert matched.height == 1
    assert matched["gsis_id"].to_list() == ["00-1"]
    assert unmatched.height == 1  # surfaced, not silently dropped
    assert unmatched["name"].to_list() == ["Obscure Rookie"]
