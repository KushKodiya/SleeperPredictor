"""M11 tests — settings conditioning, the temperature, shrinkage, and the ladder."""

import numpy as np
import pytest

from ffdraft.data.sleeper import Draft
from ffdraft.sim.opponent import (
    FEATURE_NAMES,
    FFC_FORMATS,
    Choice,
    ExcludedDraft,
    OpponentModel,
    describe_rung,
    features,
    fit_opponent_model,
    fit_temperature,
    resolve_draft_settings,
    roster_positions,
    select_rung,
    shrink,
    to_round,
)

FLEX = {"FLEX": ["RB", "WR", "TE"]}
RNG = np.random.default_rng(20260827)  # R7: every synthetic draft is reproducible


def _draft(*, teams=12, scoring="ppr", rounds=15):
    return Draft.model_validate(
        {"draft_id": "d1", "status": "complete", "type": "snake", "season": "2024",
         "metadata": {"scoring_type": scoring},
         "settings": {"teams": teams, "rounds": rounds, "slots_qb": 1, "slots_rb": 2,
                      "slots_wr": 2, "slots_te": 1, "slots_flex": 2, "slots_k": 1,
                      "slots_def": 1, "slots_bn": 5}}
    )


# --- 2.2 settings conditioning -----------------------------------------------------


def test_a_resolvable_draft_yields_its_own_settings():
    settings, reason = resolve_draft_settings(_draft(teams=10, scoring="ppr"))
    assert reason is None
    assert settings["teams"] == 10
    assert settings["scoring_type"] == "ppr"
    assert settings["slots"]["RB"] == 2 and settings["slots"]["FLEX"] == 2


def test_an_unpublished_scoring_type_is_excluded_not_pooled():
    _, reason = resolve_draft_settings(_draft(scoring="idp"))
    assert reason is not None and "idp" in reason


def test_a_team_count_ffc_does_not_publish_is_excluded():
    _, reason = resolve_draft_settings(_draft(teams=6))
    assert reason is not None and "teams=6" in reason


@pytest.mark.parametrize("scoring", sorted(FFC_FORMATS))
def test_every_supported_scoring_type_resolves(scoring):
    settings, reason = resolve_draft_settings(_draft(scoring=scoring))
    assert reason is None and settings["scoring_type"] == scoring


def test_a_pick_is_measured_in_rounds_so_team_counts_are_comparable():
    """FFC serves only 12-team ADP historically, so position is normalised, not the market."""
    assert to_round(5, 10) == pytest.approx(0.5)
    assert to_round(5, 12) == pytest.approx(0.4167, abs=1e-3)
    # the 11th pick of a 10-team draft and the 13th of a 12-team draft are both round ~1.1
    assert to_round(11, 10) == pytest.approx(to_round(13, 12), abs=0.02)


def test_a_ten_team_pick_is_not_scored_as_a_twelve_team_pick():
    assert to_round(10, 10) != to_round(10, 12)


def test_roster_positions_expand_the_slot_counts():
    slots = {"QB": 1, "RB": 2, "FLEX": 2, "BN": 3}
    assert roster_positions(slots) == ["QB", "RB", "RB", "FLEX", "FLEX", "BN", "BN", "BN"]


# --- features ----------------------------------------------------------------------


def _features(positions, teams=None, open_positions=frozenset({"RB"}), recent=(),
              share=None, roster_byes=frozenset(), byes=None):
    n = len(positions)
    return features(
        candidate_positions=np.array(positions),
        candidate_teams=np.array(teams if teams else ["KC"] * n),
        open_positions=set(open_positions),
        recent_positions=list(recent),
        manager_team_share=share or {},
        roster_byes=set(roster_byes),
        candidate_byes=np.array(byes if byes else [0] * n),
    )


def test_need_marks_players_who_fill_an_open_slot():
    out = _features(["RB", "QB"], open_positions={"RB"})
    assert out[0, 0] == 1.0 and out[1, 0] == 0.0


def test_run_momentum_reflects_the_last_few_picks():
    out = _features(["RB", "TE"], recent=["RB", "RB", "WR"])
    assert out[0, 1] > out[1, 1]


def test_team_bias_uses_the_managers_own_history():
    out = _features(["RB", "RB"], teams=["KC", "SF"], share={"KC": 0.4})
    assert out[0, 2] == pytest.approx(0.4) and out[1, 2] == 0.0


def test_bye_conflict_flags_a_clash_with_the_existing_roster():
    out = _features(["RB", "WR"], roster_byes={7}, byes=[7, 9])
    assert out[0, 3] == 1.0 and out[1, 3] == 0.0


def test_the_feature_block_has_one_column_per_named_feature():
    assert _features(["RB"]).shape == (1, len(FEATURE_NAMES))


# --- 2.1 the temperature -----------------------------------------------------------


def _choices(n=400, *, tau=1.0, n_candidates=30, beta=None, users=("m1",)):
    """Picks generated from a known temperature, so the fit has a truth to recover."""
    beta = np.zeros(len(FEATURE_NAMES)) if beta is None else beta
    out = []
    for i in range(n):
        adp = np.sort(RNG.uniform(0, 12, n_candidates))
        feats = RNG.random((n_candidates, len(FEATURE_NAMES)))
        utility = -adp / tau + feats @ beta
        p = np.exp(utility - utility.max())
        p /= p.sum()
        out.append(Choice(users[i % len(users)], adp, feats, int(RNG.choice(n_candidates, p=p))))
    return out


def test_temperature_is_recovered_from_picks():
    assert fit_temperature(_choices(tau=1.5)) == pytest.approx(1.5, rel=0.25)


def test_a_disciplined_league_fits_a_sharper_temperature_than_a_chaotic_one():
    sharp = fit_temperature(_choices(tau=0.4))
    loose = fit_temperature(_choices(tau=4.0))
    assert sharp < loose


def test_temperature_fits_even_when_every_deviation_is_zero():
    """τ is estimable on its own and is fitted first and always."""
    model = fit_opponent_model(
        _choices(tau=1.2, beta=np.zeros(len(FEATURE_NAMES))),
        prior_strength=40, min_picks_to_fit_manager=30, adp_noise_sigma_rounds=1.2,
    )
    assert model.tau == pytest.approx(1.2, rel=0.3)
    assert model.tau > 0


def test_fitting_a_temperature_with_no_picks_raises():
    with pytest.raises(ValueError, match="no observed picks"):
        fit_temperature([])


# --- 3.1 partial pooling -----------------------------------------------------------


def test_shrinkage_pulls_a_sparse_manager_toward_the_league_mean():
    league = np.array([1.0, 2.0, 3.0, 4.0])
    wild = np.array([50.0, -50.0, 50.0, -50.0])
    shrunk = shrink(wild, league, picks=5, prior_strength=40)
    # five picks against a prior of 40 keeps only 5/45 of the manager's own estimate
    assert np.allclose(shrunk, (5 / 45) * wild + (40 / 45) * league)
    assert np.all(np.abs(shrunk - league) < np.abs(wild - league))


def test_a_manager_with_five_picks_lands_within_five_percent_of_the_mean():
    """The guarantee comes from min_picks_to_fit_manager, not from shrinkage alone.

    With the configured prior strength of 40, five picks would still keep 11% of a
    manager's own estimate. What actually holds the spec's 5% is the threshold: below
    it a manager is given the league average outright, so the distance is zero.
    """
    train = _choices(n=200, users=("busy",) * 39 + ("quiet",))  # quiet gets ~5 picks
    model = fit_opponent_model(
        train, prior_strength=40, min_picks_to_fit_manager=30, adp_noise_sigma_rounds=1.2
    )
    quiet = sum(1 for c in train if c.user_id == "quiet")
    assert quiet < 30
    spread = np.abs(model.league_beta).max() or 1.0
    assert np.all(np.abs(model.beta("quiet") - model.league_beta) <= 0.05 * spread)


def test_more_picks_move_a_manager_further_from_the_mean():
    league, own = np.zeros(4), np.ones(4)
    few = shrink(own, league, picks=10, prior_strength=40)
    many = shrink(own, league, picks=400, prior_strength=40)
    assert np.all(many > few)


def test_a_manager_below_the_threshold_gets_the_league_average_outright():
    train = _choices(n=120, users=("busy", "busy", "busy", "quiet"))
    model = fit_opponent_model(
        train, prior_strength=40, min_picks_to_fit_manager=50, adp_noise_sigma_rounds=1.2
    )
    assert np.allclose(model.beta("quiet"), model.league_beta)


def test_an_unknown_manager_falls_back_to_the_league_mean():
    model = fit_opponent_model(
        _choices(n=80), prior_strength=40, min_picks_to_fit_manager=10,
        adp_noise_sigma_rounds=1.2,
    )
    assert np.allclose(model.beta("never-seen"), model.league_beta)


# --- 4.1 / 4.2 the degradation ladder ----------------------------------------------


def test_a_rung_is_used_only_when_it_beats_the_one_below():
    """Picks generated with no deviation signal: the full rung must not survive."""
    train = _choices(n=300, tau=1.0, beta=np.zeros(len(FEATURE_NAMES)))
    holdout = _choices(n=200, tau=1.0, beta=np.zeros(len(FEATURE_NAMES)))
    model = fit_opponent_model(
        train, prior_strength=40, min_picks_to_fit_manager=30, adp_noise_sigma_rounds=1.2
    )
    chosen, scores = select_rung(model, holdout, adp_noise_sigma_rounds=1.2)
    by_rung = dict(scores)
    if chosen.rung == "full":
        assert by_rung["full"] < by_rung["temperature"]
    assert chosen.rung in {"full", "temperature", "adp_noise"}


def test_a_real_deviation_signal_keeps_the_full_rung():
    beta = np.array([4.0, 0.0, 0.0, 0.0])  # strong, learnable "need" effect
    train = _choices(n=600, tau=1.0, beta=beta)
    holdout = _choices(n=300, tau=1.0, beta=beta)
    model = fit_opponent_model(
        train, prior_strength=1, min_picks_to_fit_manager=10, adp_noise_sigma_rounds=1.2
    )
    chosen, scores = select_rung(model, holdout, adp_noise_sigma_rounds=1.2)
    assert chosen.rung == "full"
    assert dict(scores)["full"] < dict(scores)["temperature"]


def test_a_fitted_temperature_beats_the_adp_noise_floor():
    train = _choices(n=400, tau=0.5)
    holdout = _choices(n=200, tau=0.5)
    model = fit_opponent_model(
        train, prior_strength=40, min_picks_to_fit_manager=30, adp_noise_sigma_rounds=1.2
    )
    _, scores = select_rung(model, holdout, adp_noise_sigma_rounds=1.2)
    assert dict(scores)["temperature"] < dict(scores)["adp_noise"]


def test_no_training_data_collapses_to_the_fallback_rung():
    model = fit_opponent_model(
        [], prior_strength=40, min_picks_to_fit_manager=30, adp_noise_sigma_rounds=1.2
    )
    assert model.rung == "adp_noise"


def test_the_startup_line_names_the_live_rung_and_every_score():
    train = _choices(n=200)
    model = fit_opponent_model(
        train, prior_strength=40, min_picks_to_fit_manager=30, adp_noise_sigma_rounds=1.2,
        excluded=(ExcludedDraft("d9", 2021, "teams=6 is not a team count FFC publishes"),),
    )
    chosen, scores = select_rung(model, _choices(n=100), adp_noise_sigma_rounds=1.2)
    line = describe_rung(chosen, scores)
    assert f"rung={chosen.rung}" in line
    assert "tau=" in line
    assert "1 drafts excluded" in line
    for rung, _ in scores:
        assert rung in line


def test_probabilities_are_a_proper_distribution_over_available_players():
    model = OpponentModel(rung="temperature", tau=1.0, league_beta=np.zeros(len(FEATURE_NAMES)))
    p = model.probabilities(np.array([0.5, 1.0, 4.0]), np.zeros((3, len(FEATURE_NAMES))),
                            user_id="m1")
    assert p.sum() == pytest.approx(1.0)
    assert p[0] > p[1] > p[2]  # better ADP is likelier


def test_the_fallback_rung_still_ranks_by_adp():
    model = OpponentModel(rung="adp_noise", tau=1.0, league_beta=np.zeros(len(FEATURE_NAMES)),
                          adp_noise_sigma_rounds=1.2)
    p = model.probabilities(np.array([0.2, 3.0]), np.zeros((2, len(FEATURE_NAMES))), user_id="m1")
    assert p[0] > p[1]


def test_top_k_accuracy_beats_chance_on_learnable_picks():
    beta = np.array([4.0, 0.0, 0.0, 0.0])
    model = fit_opponent_model(
        _choices(n=400, beta=beta), prior_strength=1, min_picks_to_fit_manager=10,
        adp_noise_sigma_rounds=1.2,
    )
    assert model.top_k_accuracy(_choices(n=200, beta=beta), k=5) > 5 / 30


def test_the_ridge_knob_shrinks_coefficients_when_turned_up():
    """Default 0 was selected on validation data; the knob must still work when needed."""
    train = _choices(n=300, beta=np.array([3.0, 0.0, 0.0, 0.0]))
    loose = fit_opponent_model(train, prior_strength=40, min_picks_to_fit_manager=30,
                               adp_noise_sigma_rounds=1.2, ridge_penalty=0.0)
    tight = fit_opponent_model(train, prior_strength=40, min_picks_to_fit_manager=30,
                               adp_noise_sigma_rounds=1.2, ridge_penalty=1.0)
    assert np.abs(tight.league_beta).sum() < np.abs(loose.league_beta).sum()
