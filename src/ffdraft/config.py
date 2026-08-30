"""Typed config loaded from `config.yaml`.

No numeric constant appears in application code (R5): every tunable is read from
here. Models forbid unknown keys so a typo in `config.yaml` fails at startup
rather than being silently ignored.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LeagueConfig(_Base):
    platform: str
    league_id: str
    draft_id: str
    my_user_id: str
    season: int
    teams: int = Field(gt=0)
    rounds: int = Field(gt=0)
    fallback_roster_positions: list[str]


class DataConfig(_Base):
    cache_dir: str
    history_seasons: list[int]
    sleeper_players_ttl_hours: int = Field(ge=0)
    adp_ttl_hours: int = Field(ge=0)


class CrosswalkConfig(_Base):
    fuzzy_threshold: int = Field(ge=0, le=100)
    unmatched_report: str


class CalibrationConfig(_Base):
    method: Literal["isotonic", "linear", "none"]
    min_training_seasons: int = Field(gt=0)
    fit_pool: dict[str, int]


class ReplacementConfig(_Base):
    method: Literal["waiver_empirical", "last_starter", "static_rank"]
    waiver_lookback_seasons: int = Field(gt=0)
    waiver_percentile: float = Field(ge=0.0, le=1.0)


class SimulationConfig(_Base):
    n_sims_live: int = Field(gt=0)
    n_sims_backtest: int = Field(gt=0)
    n_scenarios_live: int = Field(gt=0)
    n_scenarios_backtest: int = Field(gt=0)
    n_scenarios_equity: int = Field(gt=0)
    n_sims_equity: int = Field(gt=0)
    shortlist_size: int = Field(gt=0)
    force_best_at_each_position: bool
    common_random_numbers: bool
    time_budget_seconds: int = Field(gt=0)
    objective: Literal["expected_points", "championship_equity"]


class OpponentModelConfig(_Base):
    type: str
    prior: Literal["adp", "uniform"]
    fit_deviations: bool
    shrinkage: str
    shrinkage_prior_strength: int = Field(ge=0)
    min_picks_to_fit_manager: int = Field(ge=0)
    temperature_init: float = Field(gt=0)
    adp_noise_sigma_rounds: float = Field(ge=0)
    ridge_penalty: float = Field(ge=0)


class AvailabilityConfig(_Base):
    games_per_season: int = Field(gt=0)
    model: str
    lookback_seasons: int = Field(gt=0)
    age_bin_width: int = Field(gt=0)
    min_bin_count: int = Field(gt=0)
    workload_percentile_flag: float = Field(ge=0.0, le=1.0)
    rosterable_percentile: float = Field(ge=0.0, le=1.0)


class OverridesConfig(_Base):
    projection_overrides: str
    player_id_overrides: str
    reload_on_every_pick: bool


class RuntimeConfig(_Base):
    poll_interval_seconds: float = Field(gt=0)
    http_timeout_seconds: float = Field(gt=0)
    http_max_retries: int = Field(ge=0)
    anomaly_log: str


class Config(_Base):
    league: LeagueConfig
    flex_eligibility: dict[str, list[str]]
    data: DataConfig
    crosswalk: CrosswalkConfig
    calibration: CalibrationConfig
    replacement: ReplacementConfig
    simulation: SimulationConfig
    opponent_model: OpponentModelConfig
    availability: AvailabilityConfig
    overrides: OverridesConfig
    runtime: RuntimeConfig


def load_config(path: str | Path = "config.yaml") -> Config:
    """Read and validate `config.yaml`, failing fast on malformed input."""
    p = Path(path)
    if not p.exists():
        # config.yaml is gitignored (it holds live league IDs), so a fresh clone has
        # only the example. Say so rather than raising a bare FileNotFoundError.
        raise FileNotFoundError(
            f"{p} not found. Copy config.example.yaml to {p} and fill in the "
            f"league_id, draft_id and my_user_id from your Sleeper league."
        )
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    return Config(**raw)
