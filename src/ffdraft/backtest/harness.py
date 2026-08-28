"""M14 — replaying historical drafts under strict point-in-time data.

Everything from Phase 7 onward is only worth adding if it beats something simpler, and
there is no way to know that without this. Which makes the leakage guard the most
important thing in the module — more important than the harness around it. A backtest
that quietly reads the season it is predicting will bless changes that do nothing, and it
will do so with confident numbers.

So `assert_point_in_time` is a hard assertion on every frame a decision reads, not a
convention. Realized outcomes are used for *scoring* a roster and never for choosing one;
the two paths are kept separate on purpose.

Reporting is distributional. A single mean across seasons and slots hides exactly the
variance that decides whether a change is real, so `summarize` refuses to collapse to one.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import polars as pl

from ffdraft.contracts import assert_columns
from ffdraft.draft.state import assign_slots
from ffdraft.lineup.slots import SlotConfig
from ffdraft.lineup.value import Player, lineup_value


class LeakageError(AssertionError):
    """Raised when a decision would read data from the target season or later."""


def assert_point_in_time(frame: pl.DataFrame, *, target_season: int, source: str) -> None:
    """Fail if any row could tell a decision what happened in the season it is drafting for.

    This is the phase's reason for existing. It is an assertion rather than a review
    convention because leakage is invisible in the output: the backtest still runs, still
    reports, and is simply wrong.
    """
    if "season" not in frame.columns:
        raise LeakageError(
            f"{source} has no season column, so it cannot be checked for leakage; "
            f"a decision input must carry the season it came from"
        )
    future = frame.filter(pl.col("season") >= target_season)
    if not future.is_empty():
        seasons = sorted(future["season"].unique().to_list())
        raise LeakageError(
            f"{source} would leak into the {target_season} draft: {future.height} row(s) "
            f"from season(s) {seasons}. Every decision input must have season < {target_season}."
        )


@dataclass(frozen=True)
class Candidate:
    """One draftable player, as the point-in-time board saw him."""

    player_id: str
    name: str
    position: str
    adp: float | None = None
    ecr: float | None = None
    vor: float | None = None


@dataclass(frozen=True)
class DraftContext:
    """What a strategy is allowed to look at when it is on the clock."""

    available: tuple[Candidate, ...]
    roster: tuple[Candidate, ...]
    roster_positions: tuple[str, ...]
    flex_eligibility: dict[str, list[str]]
    pick_number: int

    def legal(self) -> tuple[Candidate, ...]:
        """Candidates that can still go somewhere on this roster.

        Every strategy shares this rule, so a comparison between them measures the
        ranking metric rather than differences in roster bookkeeping. While a starting
        slot is open, only players who can fill one are considered; after that, anybody.
        """
        slots, _ = assign_slots(
            [p.position for p in self.roster], list(self.roster_positions), self.flex_eligibility
        )
        filled = [s for s in slots if s is not None]
        open_slots = list(self.roster_positions)
        for slot in filled:
            if slot in open_slots:
                open_slots.remove(slot)
        starting = [s for s in open_slots if s != "BN"]
        if not starting:
            return self.available

        fillable = {
            position
            for slot in starting
            for position in self.flex_eligibility.get(slot, [slot])
        }
        eligible = tuple(p for p in self.available if p.position in fillable)
        return eligible or self.available


Strategy = Callable[[DraftContext], Candidate]


def _best(context: DraftContext, key, *, descending: bool) -> Candidate:
    pool = [p for p in context.legal() if key(p) is not None]
    if not pool:
        pool = list(context.legal())
        if not pool:
            raise ValueError("no candidate available to draft")
        return pool[0]
    return min(pool, key=lambda p: (-key(p) if descending else key(p), p.player_id))


def adp_follow(context: DraftContext) -> Candidate:
    """Take the player the market drafts next. The baseline everything must beat."""
    return _best(context, lambda p: p.adp, descending=False)


def raw_consensus(context: DraftContext) -> Candidate:
    """Take the best expert rank, uncalibrated and unadjusted for scarcity."""
    return _best(context, lambda p: p.ecr, descending=False)


def static_vor(context: DraftContext) -> Candidate:
    """Take the highest value over replacement from the point-in-time board."""
    return _best(context, lambda p: p.vor, descending=True)


BASELINES: dict[str, Strategy] = {
    "adp_follow": adp_follow,
    "raw_consensus": raw_consensus,
    "static_vor": static_vor,
}


def snake_order(*, teams: int, rounds: int) -> list[int]:
    """Draft slots in pick order, reversing every other round."""
    order: list[int] = []
    for rnd in range(rounds):
        slots = range(1, teams + 1)
        order.extend(slots if rnd % 2 == 0 else reversed(list(slots)))
    return order


def run_draft(
    board: Sequence[Candidate],
    *,
    teams: int,
    rounds: int,
    my_slot: int,
    strategy: Strategy,
    opponent: Strategy = adp_follow,
    roster_positions: Sequence[str],
    flex_eligibility: dict[str, list[str]],
) -> dict[int, list[Candidate]]:
    """Snake-draft the board: `strategy` on `my_slot`, `opponent` everywhere else."""
    available = {p.player_id: p for p in board}
    rosters: dict[int, list[Candidate]] = {slot: [] for slot in range(1, teams + 1)}

    for pick_number, slot in enumerate(snake_order(teams=teams, rounds=rounds), start=1):
        if not available:
            break
        context = DraftContext(
            available=tuple(available.values()),
            roster=tuple(rosters[slot]),
            roster_positions=tuple(roster_positions),
            flex_eligibility=flex_eligibility,
            pick_number=pick_number,
        )
        chosen = (strategy if slot == my_slot else opponent)(context)
        rosters[slot].append(chosen)
        del available[chosen.player_id]
    return rosters


def reconstruct_rosters(picks: pl.DataFrame) -> dict[str, list[str]]:
    """Rosters as a real draft actually produced them, keyed by roster id."""
    assert_columns(picks, {"roster_id", "player_id", "pick_no"}, "backtest.reconstruct_rosters")
    rosters: dict[str, list[str]] = {}
    for row in picks.sort("pick_no").iter_rows(named=True):
        rosters.setdefault(str(row["roster_id"]), []).append(str(row["player_id"]))
    return rosters


def score_roster_actuals(
    roster: Sequence[Player],
    weekly: pl.DataFrame,
    slots: SlotConfig,
    *,
    id_column: str = "player_id",
) -> float:
    """Points a roster would have scored starting its best legal lineup every week.

    `weekly` is realized scoring — what actually happened. This is the only place the
    target season's outcomes are allowed, and it never feeds a draft decision.
    """
    assert_columns(weekly, {"week", id_column, "points"}, "backtest.score_roster_actuals")
    positions = {p.player_id: p.position for p in roster}
    rostered = weekly.filter(pl.col(id_column).is_in(list(positions)))

    total = 0.0
    for (_week,), rows in rostered.group_by(["week"]):
        active = [
            Player(r[id_column], positions[r[id_column]], float(r["points"]))
            for r in rows.iter_rows(named=True)
            if r["points"] is not None
        ]
        if active:
            total += lineup_value(active, slots)
    return total


@dataclass
class BacktestResult:
    """One row per (season, slot, strategy) — never pre-aggregated."""

    rows: list[dict] = field(default_factory=list)

    def add(self, *, season: int, slot: int, strategy: str, points: float) -> None:
        self.rows.append(
            {"season": season, "slot": slot, "strategy": strategy, "points": points}
        )

    def frame(self) -> pl.DataFrame:
        return pl.DataFrame(self.rows)


def summarize(results: pl.DataFrame) -> pl.DataFrame:
    """The distribution across seasons x slots, per strategy.

    Deliberately not a mean. A strategy that wins on average while losing at half the
    draft slots is not an improvement, and one number cannot tell you which you have.
    """
    assert_columns(results, {"season", "slot", "strategy", "points"}, "backtest.summarize")
    return (
        results.group_by("strategy")
        .agg(
            pl.len().alias("n"),
            pl.col("points").min().round(1).alias("min"),
            pl.col("points").quantile(0.25).round(1).alias("p25"),
            pl.col("points").median().round(1).alias("median"),
            pl.col("points").quantile(0.75).round(1).alias("p75"),
            pl.col("points").max().round(1).alias("max"),
            pl.col("points").mean().round(1).alias("mean"),
            pl.col("points").std().round(1).alias("sd"),
            pl.col("season").n_unique().alias("seasons"),
            pl.col("slot").n_unique().alias("slots"),
        )
        .sort("median", descending=True)
    )


def head_to_head(results: pl.DataFrame, *, baseline: str) -> pl.DataFrame:
    """Per (season, slot), how each strategy did against one baseline.

    The paired view is what tells you whether a strategy actually wins, rather than
    winning on average because it got lucky at one slot in one season.
    """
    assert_columns(results, {"season", "slot", "strategy", "points"}, "backtest.head_to_head")
    base = results.filter(pl.col("strategy") == baseline).select(
        "season", "slot", pl.col("points").alias("baseline_points")
    )
    if base.is_empty():
        raise ValueError(f"no rows for baseline {baseline!r}; cannot compare against it")
    return (
        results.filter(pl.col("strategy") != baseline)
        .join(base, on=["season", "slot"], how="inner")
        .with_columns((pl.col("points") - pl.col("baseline_points")).alias("edge"))
        .group_by("strategy")
        .agg(
            pl.len().alias("drafts"),
            (pl.col("edge") > 0).sum().alias("wins"),
            pl.col("edge").median().round(1).alias("median_edge"),
            pl.col("edge").min().round(1).alias("worst"),
            pl.col("edge").max().round(1).alias("best"),
        )
        .sort("median_edge", descending=True)
    )
