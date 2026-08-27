"""M4 tests — equal-weighted aggregation and preseason ECR extraction."""

import polars as pl
import pytest

from ffdraft.data.projections import ECR_SOURCE, aggregate, load_csv_sources, preseason_ecr


def _long(*rows) -> pl.DataFrame:
    return pl.DataFrame(
        [{"gsis_id": g, "source": s, "projected_points": p} for g, s, p in rows],
        schema={"gsis_id": pl.String, "source": pl.String, "projected_points": pl.Float64},
    )


def test_player_in_three_of_five_sources_averages_over_three():
    """The divisor is the number of sources covering the player, not the source count."""
    sources = _long(
        ("A", "s1", 300.0), ("A", "s2", 240.0), ("A", "s3", 270.0),
        ("B", "s1", 100.0), ("B", "s2", 100.0), ("B", "s3", 100.0),
        ("B", "s4", 100.0), ("B", "s5", 100.0),
    )
    out = {r["gsis_id"]: r for r in aggregate(sources).iter_rows(named=True)}
    assert out["A"]["projected_points"] == pytest.approx(270.0)  # 810/3, not 810/5
    assert out["A"]["n_sources"] == 3
    assert out["B"]["n_sources"] == 5


def test_missing_source_is_not_zero_filled():
    """A zero-filled miss would drag the mean toward zero; absence must cost nothing."""
    covered = aggregate(_long(("A", "s1", 200.0), ("A", "s2", 100.0)))
    zero_filled = aggregate(_long(("A", "s1", 200.0), ("A", "s2", 100.0), ("A", "s3", 0.0)))
    assert covered["projected_points"].item() == pytest.approx(150.0)
    assert zero_filled["projected_points"].item() == pytest.approx(100.0)  # what we avoid


def test_null_projection_does_not_count_as_a_source():
    sources = _long(("A", "s1", 200.0), ("A", "s2", None))
    out = aggregate(sources)
    assert out["projected_points"].item() == pytest.approx(200.0)
    assert out["n_sources"].item() == 1


def test_source_count_travels_with_every_player():
    out = aggregate(_long(("A", "s1", 10.0), ("B", "s1", 5.0), ("B", "s2", 7.0)))
    assert set(out.columns) >= {"gsis_id", "projected_points", "n_sources"}
    assert out.filter(pl.col("gsis_id") == "A")["n_sources"].item() == 1


def test_no_csv_drops_yields_an_empty_frame_not_an_error(tmp_path):
    """A fresh checkout has no manual projections; that is a valid state, not a failure."""
    long, unmatched = load_csv_sources(
        pl.DataFrame(), season=2026, fuzzy_threshold=92, csv_dir=tmp_path
    )
    assert long.is_empty() and unmatched.is_empty()
    assert set(long.columns) == {"gsis_id", "source", "projected_points"}


# --- preseason ECR extraction -----------------------------------------------------


def _rankings(*rows) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {"id": i, "player": n, "pos": p, "team": "KC", "ecr": e, "sd": 1.0, "best": 1,
             "worst": 5, "scrape_date": d, "ecr_type": t}
            for i, n, p, e, d, t in rows
        ]
    )


_SCHEDULES = pl.DataFrame(
    [{"season": 2025, "week": 1, "gameday": "2025-09-04"}]
)
_IDS = pl.DataFrame([{"fantasypros_id": 111, "gsis_id": "00-0000111"}])


def test_preseason_ecr_takes_the_last_scrape_before_kickoff():
    """In-season scrapes already know how the season started and must not be used."""
    ranked, _ = preseason_ecr(
        _rankings(
            ("111", "A Back", "RB", 3.0, "2025-08-01", "rp"),
            ("111", "A Back", "RB", 2.0, "2025-08-29", "rp"),  # last before kickoff
            ("111", "A Back", "RB", 9.0, "2025-09-26", "rp"),  # week 3, in-season
        ),
        _SCHEDULES, _IDS, season=2025,
    )
    assert ranked.height == 1
    assert ranked["ecr"].item() == pytest.approx(2.0)
    assert ranked["scrape_date"].item() == "2025-08-29"


def test_preseason_ecr_uses_only_redraft_positional_ranks():
    ranked, _ = preseason_ecr(
        _rankings(
            ("111", "A Back", "RB", 2.0, "2025-08-29", "rp"),
            ("111", "A Back", "RB", 7.0, "2025-08-29", "bp"),   # best ball
            ("111", "A Back", "RB", 8.0, "2025-08-29", "rsf"),  # superflex
        ),
        _SCHEDULES, _IDS, season=2025,
    )
    assert ranked.height == 1 and ranked["ecr"].item() == pytest.approx(2.0)


def test_unresolvable_ranking_rows_are_reported_not_dropped():
    """A player missing from the crosswalk is a board the engine will never recommend (R4)."""
    ranked, unresolved = preseason_ecr(
        _rankings(
            ("111", "A Back", "RB", 2.0, "2025-08-29", "rp"),
            ("999", "Ghost Rookie", "WR", 40.0, "2025-08-29", "rp"),
        ),
        _SCHEDULES, _IDS, season=2025,
    )
    assert ranked.height == 1
    assert unresolved.height == 1
    assert unresolved["name"].item() == "Ghost Rookie"


def test_season_with_no_preseason_scrape_raises():
    with pytest.raises(ValueError, match="no redraft-positional ECR"):
        preseason_ecr(
            _rankings(("111", "A Back", "RB", 9.0, "2025-09-26", "rp")),
            _SCHEDULES, _IDS, season=2025,
        )


def test_ecr_source_name_is_stable():
    """The board and the override path both key on this string."""
    assert ECR_SOURCE == "fantasypros_ecr"
