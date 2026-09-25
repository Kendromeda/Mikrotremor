from __future__ import annotations

import numpy as np
import pytest

from mhvsr_vs30.mam.inversion import forward_rayleigh_phase, vs30_from_layers


def test_uniform_model_has_nearly_constant_dispersion() -> None:
    periods = np.array([0.04, 0.06, 0.09])
    velocity = forward_rayleigh_phase(periods, np.array([10.0]), np.array([300.0, 300.0]))
    assert len(velocity) == len(periods)
    assert np.all(np.isfinite(velocity))
    assert np.ptp(velocity) < 2.0
    assert np.all((velocity > 260.0) & (velocity < 300.0))


def test_vs30_uses_harmonic_travel_time_through_30_meters() -> None:
    assert vs30_from_layers(np.array([10.0]), np.array([200.0, 400.0])) == pytest.approx(
        300.0
    )


def test_forward_rejects_invalid_layer_structure() -> None:
    with pytest.raises(ValueError, match="one fewer"):
        forward_rayleigh_phase(np.array([0.04]), np.array([10.0, 20.0]), np.array([300.0, 400.0]))
