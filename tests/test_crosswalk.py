"""M5 crosswalk tests — synthetic data only, no network (PRD §10)."""

from pathlib import Path

import polars as pl

from ffdraft.data.crosswalk import (
    _Index,
    build_crosswalk,
    normalize_name,
    resolve_frame,
    resolve_one,
    write_unmatched_report,
)


def _crosswalk():
    ff = pl.DataFrame(
        {
            "gsis_id": ["00-1", "00-2", "00-3"],
            "sleeper_id": ["s1", "s2", "s3"],
            "name": ["A.J. Brown", "Patrick Mahomes", "Amon-Ra St. Brown"],
            "position": ["WR", "QB", "WR"],
            "team": ["PHI", "KC", "DET"],
        }
    )
    return build_crosswalk(ff)


def test_normalize_name_cases():
    assert normalize_name("A.J. Brown") == "aj brown"
    assert normalize_name("Le'Veon Bell") == "leveon bell"
    assert normalize_name("Marvin Harrison Jr.") == "marvin harrison"
    assert normalize_name("Amon-Ra St. Brown") == "amon-ra st brown"


def test_build_crosswalk_seeds_from_dynastyprocess():
    cw = _crosswalk()
    assert set(cw.columns) >= {"gsis_id", "sleeper_id", "normalized_name", "match_method"}
    assert cw["match_method"].unique().to_list() == ["dynastyprocess"]


def test_exact_match():
    idx = _Index(_crosswalk())
    m = resolve_one("A.J. Brown", "WR", "PHI", idx, fuzzy_threshold=92)
    assert m is not None and m.gsis_id == "00-1" and m.method == "exact"


def test_below_threshold_goes_unmatched():
    idx = _Index(_crosswalk())
    m = resolve_one("Nonexistent Playerman", "WR", "XXX", idx, fuzzy_threshold=92)
    assert m is None  # reported unmatched, never a silent bad match


def test_high_similarity_fuzzy_matches():
    idx = _Index(_crosswalk())
    m = resolve_one("Patric Mahomes", "QB", None, idx, fuzzy_threshold=92)  # 1-char typo
    assert m is not None and m.gsis_id == "00-2" and m.method == "fuzzy"


def test_override_wins_over_exact():
    idx = _Index(_crosswalk())
    m = resolve_one(
        "A.J. Brown", "WR", "PHI", idx, fuzzy_threshold=92,
        overrides={"A.J. Brown": "00-OVERRIDE"},
    )
    assert m is not None and m.gsis_id == "00-OVERRIDE" and m.method == "override"


def test_defense_gets_synthetic_id():
    idx = _Index(_crosswalk())
    m = resolve_one("Detroit Lions", "DEF", "DET", idx, fuzzy_threshold=92)
    assert m is not None and m.gsis_id == "DEF_DET" and m.method == "def"


def test_unmatched_reported_not_dropped(tmp_path: Path):
    cw = _crosswalk()
    queries = pl.DataFrame(
        {
            "name": ["A.J. Brown", "Nonexistent Playerman"],
            "position": ["WR", "WR"],
            "team": ["PHI", "XXX"],
            "rank": [1, 2],
        }
    )
    matched, unmatched = resolve_frame(queries, cw, fuzzy_threshold=92)
    assert matched.height == 1
    assert unmatched.height == 1

    report = tmp_path / "interim" / "unmatched.csv"
    write_unmatched_report(unmatched, report)
    assert report.exists()
    assert pl.read_csv(report).height == 1
