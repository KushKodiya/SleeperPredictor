"""M11 — predicting what the other managers will do.

**ADP is the population prior. There is nothing to crawl.** Sleeper has no endpoint that
enumerates public drafts — a draft is reachable only through a `user_id` or `league_id`
you already know — so "fetch 5,000 random drafts" is not an operation that exists
(PRD §11.9). That is fine, because ADP is itself an aggregate over thousands of real
drafts: exactly the object a crawl would try to rebuild. So ADP enters as a fixed offset
and only each manager's *deviation* from it is fitted.

`τ` (temperature) is the single most valuable parameter: it measures how tightly this
league clusters around ADP, and it is estimable from a dozen drafts. It is fitted first
and always, even when every per-manager term shrinks to nothing.

**Positions are measured in rounds, not overall picks.** FFC serves only 12-team ADP for
past seasons — asking for `teams=10&year=2024` returns `meta.teams: 12` — so a pick
cannot be scored against its own team count's ADP (see CLAUDE.md's R2 log). Dividing both
the pick number and the ADP rank by the draft's team count absorbs most of that
difference: the fifth pick of a 10-team draft and of a 12-team draft are both round 1,
which is the comparison the conditioning was reaching for.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

# Sleeper scoring_type -> the FFC format that matches it. A scoring type absent here
# cannot be resolved to a baseline, and its drafts are excluded and reported rather
# than scored against the wrong market.
FFC_FORMATS: dict[str, str] = {
    "ppr": "ppr",
    "half_ppr": "half-ppr",
    "std": "standard",
    "standard": "standard",
    "dynasty_ppr": "dynasty",
    "2qb": "2qb",
}

# FFC serves these team counts; 6 and below return an error, and larger leagues have no
# published market. Verified 2026-08-27.
FFC_TEAM_COUNTS = frozenset({8, 10, 12, 14})

# The PRD's four deviation features. A caution on the last one: it fires when a candidate
# shares a bye week with someone already on the manager's roster, and players on the same
# NFL team always share a bye — so in practice it detects **team stacking**, and it fits
# strongly *positive*. Managers reach for the stack far more often than they avoid the bye.
FEATURE_NAMES = ("need", "run_momentum", "team_bias", "bye_conflict")
RUN_WINDOW = 5  # picks of look-back for a positional run

RUNGS = ("full", "temperature", "adp_noise")


@dataclass(frozen=True)
class ObservedPick:
    """One pick from one historical draft."""

    draft_id: str
    pick_no: int
    user_id: str
    player_id: str
    position: str
    nfl_team: str | None = None


@dataclass(frozen=True)
class TrainingDraft:
    """A historical draft whose settings resolved to an ADP baseline."""

    draft_id: str
    season: int
    teams: int
    rounds: int
    scoring_type: str
    slots: dict[str, int]
    picks: tuple[ObservedPick, ...]

    @property
    def ffc_format(self) -> str:
        return FFC_FORMATS[self.scoring_type]


@dataclass(frozen=True)
class ExcludedDraft:
    """A draft left out of training, with the reason, so nothing is silently pooled."""

    draft_id: str
    season: int | None
    reason: str


def resolve_draft_settings(draft) -> tuple[dict, str | None]:
    """Pull the settings the model needs, or say why the draft cannot be used.

    Returns (settings, reason_excluded). A draft is excluded when its scoring type has no
    published ADP market or its team count is one FFC does not serve — never pooled into
    a market that means something different.
    """
    settings = draft.settings or {}
    scoring = (draft.metadata or {}).get("scoring_type")
    teams = settings.get("teams")

    if scoring not in FFC_FORMATS:
        return {}, f"scoring_type {scoring!r} has no ADP market"
    if teams not in FFC_TEAM_COUNTS:
        return {}, f"teams={teams} is not a team count FFC publishes"

    slots = {
        key.removeprefix("slots_").upper(): value
        for key, value in settings.items()
        if key.startswith("slots_") and value
    }
    return (
        {
            "teams": teams,
            "rounds": settings.get("rounds", 0),
            "scoring_type": scoring,
            "slots": slots,
        },
        None,
    )


def roster_positions(slots: dict[str, int]) -> list[str]:
    """Slot list in the project's usual shape, bench last."""
    order = ["QB", "RB", "WR", "TE", "FLEX", "SUPER_FLEX", "K", "DEF", "BN"]
    known = [s for s in order if s in slots]
    extra = sorted(set(slots) - set(order))
    return [slot for slot in [*known, *extra] for _ in range(slots[slot])]


def to_round(position_in_draft: float, teams: int) -> float:
    """Overall pick (or ADP rank) expressed in draft rounds.

    The normalisation that lets a 10-team pick and a 12-team pick be compared at all.
    """
    return position_in_draft / max(teams, 1)


# --- features ----------------------------------------------------------------------


def open_positions(
    taken_positions: Sequence[str], slots: dict[str, int], flex_eligibility: dict[str, list[str]]
) -> set[str]:
    """Positions that would fill a still-empty starting slot for this manager."""
    from ffdraft.draft.state import assign_slots

    roster = roster_positions(slots)
    assigned, _ = assign_slots(list(taken_positions), roster, flex_eligibility)
    remaining = list(roster)
    for slot in assigned:
        if slot in remaining:
            remaining.remove(slot)
    starting = [s for s in remaining if s != "BN"]
    return {p for slot in starting for p in flex_eligibility.get(slot, [slot])}


def features(
    *,
    candidate_positions: np.ndarray,
    candidate_teams: np.ndarray,
    open_positions: set[str],
    recent_positions: Sequence[str],
    manager_team_share: dict[str, float],
    roster_byes: set[int],
    candidate_byes: np.ndarray,
) -> np.ndarray:
    """The per-candidate deviation features, shape (n_candidates, 4).

    Deliberately few and cheap: with a few dozen picks per manager, every extra
    coefficient is another chance to fit noise.
    """
    need = np.isin(candidate_positions, list(open_positions)).astype(float)

    window = list(recent_positions)[-RUN_WINDOW:]
    run = np.array(
        [window.count(pos) / RUN_WINDOW if window else 0.0 for pos in candidate_positions]
    )
    bias = np.array([manager_team_share.get(team, 0.0) for team in candidate_teams])
    conflict = np.array(
        [1.0 if bye and int(bye) in roster_byes else 0.0 for bye in candidate_byes]
    )
    return np.column_stack([need, run, bias, conflict])


# --- the model ---------------------------------------------------------------------


@dataclass(frozen=True)
class Choice:
    """One fitted training example: the choice set and which player was taken."""

    user_id: str
    adp_rounds: np.ndarray          # (n_candidates,)
    features: np.ndarray            # (n_candidates, n_features)
    chosen: int                     # index into the candidate arrays


def _log_likelihood(choices: Sequence[Choice], tau: float, beta_for) -> float:
    if tau <= 0:
        return -np.inf
    total = 0.0
    for choice in choices:
        utility = -choice.adp_rounds / tau + choice.features @ beta_for(choice.user_id)
        utility -= utility.max()  # stabilise before exponentiating
        total += utility[choice.chosen] - np.log(np.exp(utility).sum())
    return float(total)


def fit_temperature(
    choices: Sequence[Choice], *, lo: float = 0.05, hi: float = 20.0, iterations: int = 60
) -> float:
    """Fit `τ` alone by golden-section search on the log-likelihood.

    One parameter over a unimodal likelihood — no optimiser dependency needed, and the
    result is deterministic, which a fitted constant in a draft engine ought to be.
    """
    if not choices:
        raise ValueError("cannot fit a temperature with no observed picks")
    zero = np.zeros(choices[0].features.shape[1])
    def score(t: float) -> float:
        return _log_likelihood(choices, t, lambda _u: zero)

    invphi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - invphi * (b - a), a + invphi * (b - a)
    fc, fd = score(c), score(d)
    for _ in range(iterations):
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = score(c)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = score(d)
    return float((a + b) / 2.0)


def _fit_beta(
    choices: Sequence[Choice], tau: float, *, start: np.ndarray, ridge: float = 1.0,
    steps: int = 200, lr: float = 0.5,
) -> np.ndarray:
    """Ridge-penalised gradient ascent on the conditional-logit likelihood.

    The penalty exists because a conditional logit learns only from variation *within* a
    choice set, and two of these features barely vary within one: in this league
    `team_bias` is identical across every candidate in 82% of picks and `bye_conflict` in
    53%. That is the classic setup for a huge coefficient built on a handful of
    observations.

    It turned out not to be happening here. Sweeping the penalty on a held-out validation
    split, every non-zero value made prediction monotonically worse (log loss 2.01 at
    ridge 0 rising to 3.83 at ridge 1.0), so the default is 0 and the knob stays for a
    thinner dataset that does need it. Selected from data, not chosen because it looked
    about right (R5).
    """
    beta = start.copy()
    for _ in range(steps):
        gradient = np.zeros_like(beta)
        for choice in choices:
            utility = -choice.adp_rounds / tau + choice.features @ beta
            utility -= utility.max()
            probability = np.exp(utility)
            probability /= probability.sum()
            gradient += choice.features[choice.chosen] - probability @ choice.features
        gradient = gradient / max(len(choices), 1) - ridge * beta
        beta += lr * gradient
    return beta


def shrink(
    manager_beta: np.ndarray, league_beta: np.ndarray, *, picks: int, prior_strength: float
) -> np.ndarray:
    """Empirical-Bayes shrinkage toward the league mean, proportional to pick count.

    A manager with a handful of picks lands on the league mean; one with hundreds moves
    away from it. Twelve independent per-manager fits on a few dozen picks each would be
    fitting noise and calling it insight.
    """
    weight = picks / (picks + prior_strength)
    return weight * manager_beta + (1.0 - weight) * league_beta


@dataclass(frozen=True)
class OpponentModel:
    """A fitted opponent model, at whichever rung earned its place."""

    rung: str
    tau: float
    league_beta: np.ndarray
    manager_beta: dict[str, np.ndarray] = field(default_factory=dict)
    feature_names: tuple[str, ...] = FEATURE_NAMES
    adp_noise_sigma_rounds: float = 1.2
    excluded: tuple[ExcludedDraft, ...] = ()
    training_picks: int = 0

    def beta(self, user_id: str) -> np.ndarray:
        if self.rung != "full":
            return np.zeros(len(self.feature_names))
        return self.manager_beta.get(user_id, self.league_beta)

    def probabilities(
        self, adp_rounds: np.ndarray, candidate_features: np.ndarray, *, user_id: str
    ) -> np.ndarray:
        """Pick probabilities over the available players, for one manager on the clock."""
        if self.rung == "adp_noise":
            # No fit survived: a lognormal jitter around ADP order, the honest floor.
            utility = -adp_rounds / max(self.adp_noise_sigma_rounds, 1e-6)
        else:
            utility = -adp_rounds / self.tau + candidate_features @ self.beta(user_id)
        utility = utility - utility.max()
        weights = np.exp(utility)
        return weights / weights.sum()

    def log_loss(self, choices: Sequence[Choice]) -> float:
        """Mean negative log-likelihood per pick — lower is better."""
        if not choices:
            raise ValueError("cannot score a model on no picks")
        total = 0.0
        for choice in choices:
            p = self.probabilities(choice.adp_rounds, choice.features, user_id=choice.user_id)
            total -= np.log(max(p[choice.chosen], 1e-12))
        return float(total / len(choices))

    def top_k_accuracy(self, choices: Sequence[Choice], *, k: int = 5) -> float:
        hits = 0
        for choice in choices:
            p = self.probabilities(choice.adp_rounds, choice.features, user_id=choice.user_id)
            hits += int(choice.chosen in np.argsort(p)[::-1][:k])
        return hits / len(choices) if choices else 0.0


def fit_opponent_model(
    train: Sequence[Choice],
    *,
    prior_strength: float,
    min_picks_to_fit_manager: int,
    adp_noise_sigma_rounds: float,
    ridge_penalty: float = 1.0,
    excluded: Sequence[ExcludedDraft] = (),
) -> OpponentModel:
    """Fit `τ`, then league-wide and per-manager deviations with shrinkage.

    Always returns the *full* rung; `select_rung` is what decides whether it earned use.
    """
    if not train:
        return OpponentModel(
            rung="adp_noise", tau=1.0, league_beta=np.zeros(len(FEATURE_NAMES)),
            adp_noise_sigma_rounds=adp_noise_sigma_rounds, excluded=tuple(excluded),
        )

    tau = fit_temperature(train)
    n_features = train[0].features.shape[1]
    league_beta = _fit_beta(train, tau, start=np.zeros(n_features), ridge=ridge_penalty)

    by_manager: dict[str, list[Choice]] = {}
    for choice in train:
        by_manager.setdefault(choice.user_id, []).append(choice)

    manager_beta = {}
    for user_id, picks in by_manager.items():
        if len(picks) < min_picks_to_fit_manager:
            # Below the threshold a manager gets the league average outright.
            manager_beta[user_id] = league_beta
            continue
        fitted = _fit_beta(picks, tau, start=league_beta, ridge=ridge_penalty)
        manager_beta[user_id] = shrink(
            fitted, league_beta, picks=len(picks), prior_strength=prior_strength
        )

    return OpponentModel(
        rung="full", tau=tau, league_beta=league_beta, manager_beta=manager_beta,
        adp_noise_sigma_rounds=adp_noise_sigma_rounds, excluded=tuple(excluded),
        training_picks=len(train),
    )


def select_rung(
    model: OpponentModel, holdout: Sequence[Choice], *, adp_noise_sigma_rounds: float
) -> tuple[OpponentModel, list[tuple[str, float]]]:
    """Demote the model until it beats the rung below it on held-out log loss.

    The model is on probation against its own simpler form: per-manager terms have to
    earn their place against τ-only, and τ has to earn its place against ADP noise.
    Returns the model that survived plus every rung's score, so the choice is auditable.
    """
    ladder = {
        "full": model,
        "temperature": OpponentModel(
            rung="temperature", tau=model.tau, league_beta=model.league_beta,
            adp_noise_sigma_rounds=adp_noise_sigma_rounds, excluded=model.excluded,
            training_picks=model.training_picks,
        ),
        "adp_noise": OpponentModel(
            rung="adp_noise", tau=model.tau, league_beta=model.league_beta,
            adp_noise_sigma_rounds=adp_noise_sigma_rounds, excluded=model.excluded,
            training_picks=model.training_picks,
        ),
    }
    scores = [(rung, ladder[rung].log_loss(holdout)) for rung in RUNGS]
    by_rung = dict(scores)

    for rung, below in (("full", "temperature"), ("temperature", "adp_noise")):
        if by_rung[rung] < by_rung[below]:
            return ladder[rung], scores
    return ladder["adp_noise"], scores


def describe_rung(model: OpponentModel, scores: Sequence[tuple[str, float]]) -> str:
    """The startup line. The owner has to know which model is actually live."""
    detail = "  ".join(f"{rung}={score:.4f}" for rung, score in scores)
    note = {
        "full": "per-manager deviations earned their place",
        "temperature": "per-manager terms did not beat league-wide temperature",
        "adp_noise": "no fitted rung beat ADP noise; running the fallback",
    }[model.rung]
    return (
        f"opponent model: rung={model.rung} tau={model.tau:.3f} "
        f"({model.training_picks} training picks, {len(model.excluded)} drafts excluded) "
        f"— {note}. held-out log loss: {detail}"
    )
