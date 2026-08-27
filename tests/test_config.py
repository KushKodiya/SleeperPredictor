from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from ffdraft.config import Config, load_config

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "config.example.yaml"
LOCAL = REPO / "config.yaml"

# The example ships with the repo and is what CI validates. A real config.yaml holds
# live league IDs and is gitignored, so it is checked when present (catching a
# draft-morning typo before it reaches the CLI) and simply absent in CI.
CONFIGS = [p for p in (EXAMPLE, LOCAL) if p.exists()]


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_valid_config_parses(path):
    cfg = load_config(path)
    assert isinstance(cfg, Config)
    assert cfg.league.teams > 0
    # the offline fallback must cover every round of the draft
    assert len(cfg.league.fallback_roster_positions) == cfg.league.rounds
    assert cfg.simulation.time_budget_seconds == 45
    assert cfg.calibration.method == "isotonic"


def test_missing_required_field_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("league:\n  platform: sleeper\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(bad)


def test_unknown_key_rejected():
    raw = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    raw["league"]["bogus_key"] = 1
    with pytest.raises(ValidationError):
        Config(**raw)


def test_out_of_range_value_rejected():
    raw = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    raw["crosswalk"]["fuzzy_threshold"] = 250  # must be 0..100
    with pytest.raises(ValidationError):
        Config(**raw)
