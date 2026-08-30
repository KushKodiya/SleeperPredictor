"""M15 — the evidence a projection model must produce before anyone trusts it.

Two gates with deliberately different standing.

The **hard gate** compares the model against the incumbent calibrated-ECR board on
held-out seasons. It depends on nothing the project does not already load, so it is
always runnable, and it is the gate that governs promotion.

The **soft gate** compares against third-party projections dropped in by hand as CSVs
(PRD §6.4 puts scraping out of scope). Those files may never exist. When they do not, the
gate reports itself **skipped and names the seasons it could not cover** — a skip is never
a pass, and it never blocks the hard gate.

Both are measured on the `calibration.fit_pool` frame: the players whose preseason expert
rank puts them inside the top N at their position. That selection is preseason-knowable,
so restricting to it leaks nothing.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from ffdraft.contracts import assert_columns

CSV_DIR = Path("data/raw/projections")


@dataclass(frozen=True)
class GateResult:
    """One gate's verdict, with the per-season evidence that produced it."""

    name: str
    by_season: pl.DataFrame
    pooled: pl.DataFrame
    passed: bool
    reason: str
    skipped_seasons: tuple[int, ...] = ()

    @property
    def skipped(self) -> bool:
        """A gate that could not run. Never the same thing as one that passed."""
        return self.by_season.is_empty()


def _spearman(a: pl.Series, b: pl.Series) -> float:
    """Rank correlation, as Pearson over ranks. Avoids a scipy dependency for one number."""
    if a.len() < 2:
        return float("nan")
    ranks = pl.DataFrame({"a": a.rank(), "b": b.rank()})
    return float(ranks.select(pl.corr("a", "b")).item() or float("nan"))


def score_sources(frame: pl.DataFrame, sources: Sequence[str]) -> pl.DataFrame:
    """MAE and rank correlation for each named projection column against `actual_points`.

    `frame` is one row per player per season, already restricted to the evaluation pool.
    """
    assert_columns(frame, {"season", "gsis_id", "actual_points", *sources}, "evaluation.score")
    rows = []
    for season in sorted(frame["season"].unique().to_list()):
        block = frame.filter(pl.col("season") == season)
        for source in sources:
            usable = block.filter(
                pl.col(source).is_not_null() & pl.col("actual_points").is_not_null()
            )
            if usable.is_empty():
                continue
            rows.append(
                {
                    "season": season,
                    "source": source,
                    "n": usable.height,
                    "mae": float(
                        (usable[source] - usable["actual_points"]).abs().mean()
                    ),
                    "spearman": _spearman(usable[source], usable["actual_points"]),
                }
            )
    return pl.DataFrame(rows)


def _pool(by_season: pl.DataFrame) -> pl.DataFrame:
    """Pooled figures, weighted by how many players each season contributed."""
    if by_season.is_empty():
        return by_season
    return (
        by_season.group_by("source")
        .agg(
            pl.col("n").sum().alias("n"),
            ((pl.col("mae") * pl.col("n")).sum() / pl.col("n").sum()).alias("mae"),
            pl.col("spearman").mean().alias("spearman"),
            pl.len().alias("seasons"),
        )
        .sort("mae")
    )


def hard_gate(
    frame: pl.DataFrame, *, model: str = "model_points", incumbent: str = "board_points"
) -> GateResult:
    """The model must beat the incumbent board on held-out MAE **and** rank correlation.

    Both, not either: a model that ranks better while being wildly miscalibrated would
    poison replacement level and VOR, and one that is closer on average while ranking
    worse would draft the wrong players in the right range.
    """
    by_season = score_sources(frame, [model, incumbent])
    pooled = _pool(by_season)
    if pooled.is_empty():
        return GateResult("hard", by_season, pooled, False, "no rows to evaluate")

    scores = {row["source"]: row for row in pooled.iter_rows(named=True)}
    if model not in scores or incumbent not in scores:
        return GateResult("hard", by_season, pooled, False, "a source produced no rows")

    model_rank, incumbent_rank = scores[model]["spearman"], scores[incumbent]["spearman"]
    # A constant projection has no ordering, so its rank correlation is undefined rather
    # than bad. Comparing against NaN silently returns False and would let a degenerate
    # incumbent block a model that orders players perfectly well — so decide it here.
    if math.isnan(model_rank):  # the model itself has no ordering
        better_rank = False
    elif math.isnan(incumbent_rank):
        better_rank = True
    else:
        better_rank = model_rank > incumbent_rank

    better_mae = scores[model]["mae"] < scores[incumbent]["mae"]
    passed = better_mae and better_rank
    verdict = "beats" if passed else "does not beat"
    return GateResult(
        "hard",
        by_season,
        pooled,
        passed,
        (
            f"model {verdict} the incumbent board: MAE {scores[model]['mae']:.1f} vs "
            f"{scores[incumbent]['mae']:.1f}, rank correlation "
            f"{scores[model]['spearman']:.3f} vs {scores[incumbent]['spearman']:.3f} "
            f"over {scores[model]['seasons']} season(s)"
        ),
    )


def third_party_sources(
    seasons: Sequence[int], *, csv_dir: Path = CSV_DIR
) -> tuple[list[int], list[int]]:
    """Which of `seasons` have a manual third-party CSV, and which do not."""
    covered, missing = [], []
    for season in seasons:
        files = sorted(csv_dir.glob(f"*_{season}.csv")) if csv_dir.exists() else []
        (covered if files else missing).append(season)
    return covered, missing


def soft_gate(
    frame: pl.DataFrame,
    seasons: Sequence[int],
    *,
    model: str = "model_points",
    third_party: str = "third_party_points",
    csv_dir: Path = CSV_DIR,
) -> GateResult:
    """Compare against hand-supplied third-party projections, where they exist.

    Returns a result whose `skipped` is True when no season could be covered. That state
    is reported as a skip, never as a pass: a comparison nobody could run is not evidence
    that the model won it.
    """
    covered, missing = third_party_sources(seasons, csv_dir=csv_dir)
    if not covered or third_party not in frame.columns:
        return GateResult(
            "soft",
            pl.DataFrame(),
            pl.DataFrame(),
            False,
            (
                "SKIPPED — no third-party projection CSVs under "
                f"{csv_dir} for season(s) {sorted(seasons)}. §6.4 puts scraping out of "
                "scope, so these arrive by hand; this is not a pass."
            ),
            skipped_seasons=tuple(sorted(seasons)),
        )

    usable = frame.filter(pl.col("season").is_in(covered))
    by_season = score_sources(usable, [model, third_party])
    pooled = _pool(by_season)
    scores = {row["source"]: row for row in pooled.iter_rows(named=True)}
    passed = (
        model in scores
        and third_party in scores
        and scores[model]["mae"] < scores[third_party]["mae"]
    )
    return GateResult(
        "soft",
        by_season,
        pooled,
        passed,
        (
            f"model {'beats' if passed else 'does not beat'} third-party projections on "
            f"season(s) {covered}"
            + (f"; no CSVs for {missing}" if missing else "")
        ),
        skipped_seasons=tuple(missing),
    )
