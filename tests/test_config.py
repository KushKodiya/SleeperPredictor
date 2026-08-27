from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from ffdraft.config import Config, load_config

REPO = Path(__file__).resolve().parents[1]


def test_valid_config_parses():
    cfg = load_config(REPO / "config.yaml")
    assert isinstance(cfg, Config)
    assert cfg.league.teams == 12
    assert cfg.simulation.time_budget_seconds == 45
    assert cfg.calibration.method == "isotonic"


def test_missing_required_field_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("league:\n  platform: sleeper\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(bad)


def test_unknown_key_rejected():
    raw = yaml.safe_load((REPO / "config.yaml").read_text(encoding="utf-8"))
    raw["league"]["bogus_key"] = 1
    with pytest.raises(ValidationError):
        Config(**raw)


def test_out_of_range_value_rejected():
    raw = yaml.safe_load((REPO / "config.yaml").read_text(encoding="utf-8"))
    raw["crosswalk"]["fuzzy_threshold"] = 250  # must be 0..100
    with pytest.raises(ValidationError):
        Config(**raw)
