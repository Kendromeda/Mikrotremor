"""Preprocessing profiles.

A profile is a complete, frozen description of how raw components become a
curve. Two profiles exist because two different questions are being asked, and
answering both with one configuration would make neither answer trustworthy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from mhvsr_vs30.exceptions import ConfigError
from mhvsr_vs30.hashing import sha256_file, stable_id

__all__ = ["PreprocessingProfile", "load_profile"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class WindowingConfig(_Frozen):
    """How the record is cut into windows.

    ``fixed_count`` reproduces the reference notebook, where window length is
    duration divided by a constant window count, so different recordings get
    different windows and therefore different usable frequency floors.
    ``fixed_duration`` holds window length constant across the corpus instead.
    """

    mode: str = "fixed_duration"
    window_length_s: float | None = Field(default=None, gt=0.0)
    window_count: int | None = Field(default=None, gt=0)
    detrend: str = "linear"
    taper: str = "tukey"
    taper_width: float = Field(default=0.2, ge=0.0, le=1.0)
    significant_cycles: float = Field(default=15.0, gt=0.0)
    minimum_windows: int = Field(default=1, ge=1)

    @field_validator("mode")
    @classmethod
    def _known_mode(cls, value: str) -> str:
        if value not in {"fixed_duration", "fixed_count"}:
            raise ValueError("windowing.mode must be fixed_duration or fixed_count")
        return value


class SmoothingConfig(_Frozen):
    operator: str = "konno_and_ohmachi"
    bandwidth: float = Field(default=40.0, gt=0.0)
    internal_grid_min_hz: float = Field(default=0.05, gt=0.0)
    internal_grid_max_hz: float = Field(default=50.0, gt=0.0)
    internal_grid_points: int = Field(default=256, gt=1)


class RejectionConfig(_Frozen):
    """Automatic window rejection. Manual rejection has no place in a batch run."""

    method: str = "frequency_domain"
    n_sigma: float = Field(default=2.0, gt=0.0)
    maximum_iterations: int = Field(default=50, ge=1)
    minimum_accepted_windows: int = Field(default=1, ge=1)
    minimum_accepted_fraction: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("method")
    @classmethod
    def _known_method(cls, value: str) -> str:
        if value not in {"frequency_domain", "none"}:
            raise ValueError("rejection.method must be frequency_domain or none")
        return value


class ModelGridConfig(_Frozen):
    """The feature grid the downstream model expects."""

    min_hz: float = Field(default=0.3, gt=0.0)
    max_hz: float = Field(default=50.0, gt=0.0)
    points: int = Field(default=35, gt=1)
    allow_extrapolation: bool = False


class PreprocessingProfile(_Frozen):
    profile_id: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    hvsrpy_version: str = Field(min_length=1)
    windowing: WindowingConfig
    smoothing: SmoothingConfig = SmoothingConfig()
    rejection: RejectionConfig = RejectionConfig()
    model_grid: ModelGridConfig = ModelGridConfig()
    method_to_combine_horizontals: str = "geometric_mean"
    distribution: str = "lognormal"
    minimum_duration_s: float | None = Field(default=None, gt=0.0)
    config_hash: str = ""
    config_path: str | None = None

    @field_validator("distribution")
    @classmethod
    def _known_distribution(cls, value: str) -> str:
        if value not in {"lognormal", "normal"}:
            raise ValueError("distribution must be lognormal or normal")
        return value

    def identity(self) -> str:
        """A hash over the settings that change the numbers, not the prose.

        profile_id and purpose are deliberately excluded so that renaming a
        profile does not invalidate curves that are numerically identical.
        """
        parts = [
            self.hvsrpy_version,
            self.windowing.model_dump_json(),
            self.smoothing.model_dump_json(),
            self.rejection.model_dump_json(),
            self.model_grid.model_dump_json(),
            self.method_to_combine_horizontals,
            self.distribution,
        ]
        return stable_id(*parts)


def load_profile(path: Path | str) -> PreprocessingProfile:
    """Read and validate one preprocessing profile."""
    target = Path(path)
    if not target.is_file():
        raise ConfigError(f"preprocessing profile not found: {target}")

    try:
        payload: Any = yaml.safe_load(target.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ConfigError(f"{target} is not valid YAML: {error}") from error
    if not isinstance(payload, dict):
        raise ConfigError(f"{target} must contain a YAML mapping at the top level")

    payload["config_hash"] = sha256_file(target)
    payload["config_path"] = target.as_posix()

    try:
        profile = PreprocessingProfile(**payload)
    except ValidationError as error:
        raise ConfigError(f"{target} failed validation:\n{error}") from error

    windowing = profile.windowing
    if windowing.mode == "fixed_duration" and windowing.window_length_s is None:
        raise ConfigError(f"{target}: fixed_duration windowing needs window_length_s")
    if windowing.mode == "fixed_count" and windowing.window_count is None:
        raise ConfigError(f"{target}: fixed_count windowing needs window_count")
    if profile.smoothing.internal_grid_min_hz >= profile.smoothing.internal_grid_max_hz:
        raise ConfigError(f"{target}: internal grid bounds are not increasing")
    if profile.model_grid.min_hz >= profile.model_grid.max_hz:
        raise ConfigError(f"{target}: model grid bounds are not increasing")
    return profile
