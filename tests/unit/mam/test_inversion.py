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


def test_groundwater_rule_follows_hayashi_and_kitsunezaki() -> None:
    from mhvsr_vs30.mam.inversion import VpRule

    thickness = np.array([4.0, 10.0])
    vs = np.array([150.0, 300.0, 600.0])
    wet = VpRule(groundwater_depth_m=0.0).vp(thickness, vs)
    np.testing.assert_allclose(wet, 1.11 * vs + 1290.0)
    partly = VpRule(groundwater_depth_m=5.0).vp(thickness, vs)
    np.testing.assert_allclose(partly, [300.0, 1.11 * 300.0 + 1290.0, 1.11 * 600.0 + 1290.0])
    dry = VpRule.from_config({"method": "groundwater", "groundwater_depth_m": None}).vp(
        thickness, vs
    )
    np.testing.assert_allclose(dry, 2.0 * vs)
    constant = VpRule.from_config(None, poisson=0.3).vp(thickness, vs)
    np.testing.assert_allclose(constant, vs * np.sqrt(2 * 0.7 / 0.4))


def test_vp_rule_round_trips_and_validates() -> None:
    from mhvsr_vs30.mam.inversion import VpRule

    rule = VpRule.from_config({"method": "groundwater", "groundwater_depth_m": None})
    assert rule.to_dict()["groundwater_depth_m"] is None
    assert VpRule.from_config(rule.to_dict()) == rule
    with pytest.raises(ValueError, match="unknown Vp method"):
        VpRule(method="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="groundwater_depth_m"):
        VpRule(groundwater_depth_m=-1.0)
    with pytest.raises(ValueError, match="sqrt"):
        VpRule(above_vp_vs_ratio=1.2)


def test_saturated_vp_raises_predicted_phase_velocity_so_inverted_vs_is_lower() -> None:
    from mhvsr_vs30.mam.inversion import VpRule

    periods = np.array([0.03, 0.05, 0.08, 0.12])
    thickness, vs = np.array([5.0, 10.0]), np.array([150.0, 250.0, 500.0])
    dry = forward_rayleigh_phase(periods, thickness, vs, vp_rule=VpRule(groundwater_depth_m=np.inf))
    wet = forward_rayleigh_phase(periods, thickness, vs, vp_rule=VpRule(groundwater_depth_m=0.0))
    default = forward_rayleigh_phase(periods, thickness, vs, poisson=0.3)
    # Higher Vp below the water table raises c for the same Vs (Hayashi et al. 2022, 7.2).
    assert np.all(wet > dry)
    assert np.all(np.isfinite(default))
