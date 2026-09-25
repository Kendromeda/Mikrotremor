from __future__ import annotations

import numpy as np
import pytest

from mhvsr_vs30.mam.inversion import VpRule, forward_rayleigh_phase
from mhvsr_vs30.mam.refinement import forward_ellipticity, log_shape_correlation, refine_candidate

THICKNESS = np.array([6.0])
VS = np.array([200.0, 600.0])
PERIODS = np.geomspace(0.05, 0.5, 12)


def test_ellipticity_depends_on_the_vp_rule() -> None:
    dry = forward_ellipticity(
        PERIODS, THICKNESS, VS, poisson=0.3, density_g_cm3=2.0,
        vp_rule=VpRule(groundwater_depth_m=np.inf),
    )
    wet = forward_ellipticity(
        PERIODS, THICKNESS, VS, poisson=0.3, density_g_cm3=2.0,
        vp_rule=VpRule(groundwater_depth_m=0.0),
    )
    assert np.all(np.isfinite(dry)) and np.all(dry > 0)
    assert not np.allclose(dry, wet)
    with pytest.raises(ValueError, match="strictly increasing"):
        forward_ellipticity(PERIODS[::-1], THICKNESS, VS, poisson=0.3, density_g_cm3=2.0)


def test_log_shape_correlation_ignores_amplitude_scale() -> None:
    curve = np.array([1.0, 2.0, 4.0, 2.0, 1.0])
    assert log_shape_correlation(curve, 3.0 * curve) == pytest.approx(1.0)
    assert np.isnan(log_shape_correlation(np.ones(5), curve))
    with pytest.raises(ValueError, match="equal-length"):
        log_shape_correlation(curve, curve[:3])


def test_refine_candidate_keeps_dispersion_fit_within_the_allowed_increase() -> None:
    rule = VpRule(groundwater_depth_m=0.0)
    observed = forward_rayleigh_phase(PERIODS, THICKNESS, VS, vp_rule=rule)
    hvsr = forward_ellipticity(
        PERIODS, THICKNESS, VS, poisson=0.3, density_g_cm3=2.0, vp_rule=rule
    )
    base = np.r_[THICKNESS * 1.1, VS * 0.95]
    best, rows = refine_candidate(
        base, [(3.0, 10.0), (150.0, 300.0), (400.0, 900.0)], PERIODS, observed, PERIODS, hvsr,
        poisson=0.3, density_g_cm3=2.0, seed=1, candidate_count=6,
        perturbation_std=0.05, max_rmse_increase_m_s=5.0, vp_rule=rule,
    )
    assert rows[0]["candidate"] == 0 and rows[0]["accepted_dispersion"]
    baseline = rows[0]["dispersion_rmse_m_s"]
    assert all(
        row["dispersion_rmse_m_s"] <= baseline + 5.0 for row in rows if row["accepted_dispersion"]
    )
    assert best.shape == base.shape
    with pytest.raises(ValueError, match="bounds"):
        refine_candidate(
            base, [(3.0, 10.0)], PERIODS, observed, PERIODS, hvsr, poisson=0.3,
            density_g_cm3=2.0, seed=1, candidate_count=1, perturbation_std=0.05,
            max_rmse_increase_m_s=5.0,
        )
