"""M15 promotion tests — an unproven model must reach nothing, a promoted one substitutes."""

import polars as pl

from ffdraft.models.evaluation import GateResult
from ffdraft.models.promotion import (
    CALIBRATED_ECR,
    MODEL,
    ProjectionSource,
    apply_promotion,
    decide_source,
)


def _gate(passed, reason="because"):
    return GateResult("hard", pl.DataFrame(), pl.DataFrame(), passed, reason)


def _board():
    return pl.DataFrame(
        {"gsis_id": ["a", "b", "c"], "projected_points": [300.0, 200.0, 100.0]}
    )


def _model():
    return pl.DataFrame({"gsis_id": ["a", "b"], "projected_points": [111.0, 222.0]})


def test_a_failed_gate_leaves_the_board_untouched():
    source = decide_source(_gate(False))
    assert source.name == CALIBRATED_ECR and not source.promoted
    out = apply_promotion(_board(), _model(), source)
    assert out["projected_points"].to_list() == [300.0, 200.0, 100.0]


def test_a_passed_gate_substitutes_rather_than_blends():
    source = decide_source(_gate(True))
    assert source.name == MODEL and source.promoted
    out = apply_promotion(_board(), _model(), source)
    by_id = dict(zip(out["gsis_id"], out["projected_points"], strict=True))
    assert by_id["a"] == 111.0  # the model's number, not an average of 300 and 111
    assert by_id["b"] == 222.0


def test_a_player_the_model_does_not_cover_keeps_the_boards_projection():
    """Substituting a null would drop him off the board entirely (R4)."""
    out = apply_promotion(_board(), _model(), decide_source(_gate(True)))
    by_id = dict(zip(out["gsis_id"], out["projected_points"], strict=True))
    assert by_id["c"] == 100.0
    assert out.height == 3


def test_a_skipped_soft_gate_cannot_promote_on_its_own():
    soft = GateResult("soft", pl.DataFrame(), pl.DataFrame(), False, "SKIPPED — no CSVs")
    source = decide_source(_gate(False), soft)
    assert not source.promoted
    assert "SKIPPED" in source.describe()


def test_the_live_source_is_reported_with_its_evidence():
    source = decide_source(_gate(False, "model does not beat the incumbent board: MAE 85.5 vs 64.2"))
    described = source.describe()
    assert "calibrated ECR board" in described
    assert "85.5" in described


def test_the_promoted_description_names_the_model():
    assert "MODEL" in ProjectionSource(MODEL, True, "beats it").describe()
