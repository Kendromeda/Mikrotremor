"""Exercise notebook boundaries with synthetic picks and a captured optimizer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_notebook_02_exposes_interactive_picker() -> None:
    notebook = json.loads((ROOT / "02_mam_spac_dispersion.ipynb").read_text(encoding="utf-8"))
    code = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    assert "create_dispersion_picker(" in code[-1]
    assert "display(picker)" in code[-1]


def cell(index: int) -> str:
    notebook = json.loads((ROOT / "03_dispersion_inversion.ipynb").read_text(encoding="utf-8"))
    return "".join(notebook["cells"][index]["source"])


def selection_context(tmp_path: Path, *, wavelength_reviewed: bool = True) -> dict:
    from mhvsr_vs30.mam.picking import review_context_hash

    frequency = np.arange(10.0, 18.0)
    common = {"frequency_hz": frequency, "period_s": 1 / frequency}
    pd.DataFrame(
        {
            **common,
            "velocity_final_m_s": 300.0,
            "accepted_for_inversion": True,
            "automatic_status": "provisional",
        }
    ).to_csv(tmp_path / "final.csv", index=False)
    pd.DataFrame({**common, "velocity_auto_m_s": 200.0, "automatic_status": "provisional"}).to_csv(
        tmp_path / "auto.csv", index=False
    )
    site_inputs = tmp_path / "site_inputs.json"
    site_inputs.write_text("{}", encoding="utf-8")
    (tmp_path / "input_inventory.csv").write_text("file,sha256\n", encoding="utf-8")
    qc = {
        "run_state": "complete",
        "method_revision": "hayashi_2022_v1",
        "geometry_verified": True,
        "physical_clock_drift_verified": True,
        "wavefield_mode_verified": True,
        "wavelength_range_reviewed": wavelength_reviewed,
        "maximum_receiver_spacing_m": 5.0,
        "processing_parameters": {"maximum_wavelength_ratio": None},
        "dispersion_auto_sha256": hashlib.sha256((tmp_path / "auto.csv").read_bytes()).hexdigest(),
        "dispersion_final_sha256": hashlib.sha256(
            (tmp_path / "final.csv").read_bytes()
        ).hexdigest(),
        "review_context_sha256": review_context_hash(
            site_inputs, tmp_path / "input_inventory.csv", tmp_path / "auto.csv"
        ),
    }
    (tmp_path / "qc.json").write_text(json.dumps(qc), encoding="utf-8")
    return {
        "np": np,
        "pd": pd,
        "json": json,
        "hashlib": hashlib,
        "PARAM": {"minimum_accepted_picks": 8},
        "FINAL_INPUT": tmp_path / "final.csv",
        "AUTO_INPUT": tmp_path / "auto.csv",
        "QC_INPUT": tmp_path / "qc.json",
        "OUT": tmp_path,
        "SITE_INPUTS": site_inputs,
        "SOURCE": tmp_path,
    }


def add_helpers(context: dict) -> dict:
    from mhvsr_vs30.mam.picking import review_context_hash
    from mhvsr_vs30.mam.wavelength import depth_guidelines, wavelength_diagnostics

    return {
        **context,
        "review_context_hash": review_context_hash,
        "depth_guidelines": depth_guidelines,
        "wavelength_diagnostics": wavelength_diagnostics,
    }


def test_stale_review_context_rejected_before_final_inversion(tmp_path: Path) -> None:
    context = add_helpers(selection_context(tmp_path))
    (tmp_path / "site_inputs.json").write_text('{"changed": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="review context changed"):
        exec(cell(5), context)


def test_numerically_rejected_pick_cannot_feed_inversion(tmp_path: Path) -> None:
    context = add_helpers(selection_context(tmp_path))
    final_path = tmp_path / "final.csv"
    final = pd.read_csv(final_path)
    final.loc[0, "automatic_status"] = "rejected_numeric"
    final.to_csv(final_path, index=False)
    qc_path = tmp_path / "qc.json"
    qc = json.loads(qc_path.read_text())
    qc["dispersion_final_sha256"] = hashlib.sha256(final_path.read_bytes()).hexdigest()
    qc_path.write_text(json.dumps(qc), encoding="utf-8")
    with pytest.raises(ValueError, match="numerical QC"):
        exec(cell(5), context)


def test_unreviewed_wavelength_band_keeps_inversion_in_preview(tmp_path: Path) -> None:
    context = add_helpers(selection_context(tmp_path, wavelength_reviewed=False))
    exec(cell(5), context)
    assert context["mode"] == "preview"


def test_final_wavelengths_use_manual_velocity_not_auto_velocity(tmp_path: Path) -> None:
    context = add_helpers(selection_context(tmp_path))
    exec(cell(5), context)
    assert context["mode"] == "final"
    diagnostic = pd.read_csv(tmp_path / "final" / "wavelength_diagnostics.csv")
    np.testing.assert_allclose(diagnostic.wavelength_m, 300.0 / diagnostic.frequency_hz)


def test_configured_wavelength_limit_cannot_be_bypassed_by_manual_pick(tmp_path: Path) -> None:
    context = add_helpers(selection_context(tmp_path))
    qc_path = tmp_path / "qc.json"
    qc = json.loads(qc_path.read_text())
    qc["processing_parameters"]["maximum_wavelength_ratio"] = 2.0
    qc_path.write_text(json.dumps(qc), encoding="utf-8")
    with pytest.raises(ValueError, match="wavelength"):
        exec(cell(5), context)


def test_incomplete_upstream_run_cannot_feed_inversion(tmp_path: Path) -> None:
    context = selection_context(tmp_path)
    qc_path = tmp_path / "qc.json"
    qc = json.loads(qc_path.read_text())
    qc["run_state"] = "running"
    qc_path.write_text(json.dumps(qc), encoding="utf-8")
    with pytest.raises(ValueError, match="complete"):
        exec(cell(5), context)


def test_changed_picks_require_fresh_upstream_qc(tmp_path: Path) -> None:
    context = add_helpers(selection_context(tmp_path))
    table = pd.read_csv(tmp_path / "final.csv")
    table["velocity_final_m_s"] = 350.0
    table.to_csv(tmp_path / "final.csv", index=False)
    with pytest.raises(ValueError, match="changed"):
        exec(cell(5), context)


def test_guided_initial_model_is_passed_to_each_optimizer_run(tmp_path: Path) -> None:
    from mhvsr_vs30.mam.inversion import VpRule
    from mhvsr_vs30.mam.wavelength import initial_model_from_wavelength, wavelength_guides

    captured = []

    def optimizer(objective, bounds, **kwargs):
        initial = kwargs["x0"]
        captured.append(initial.copy())
        misfit = objective(initial)
        return SimpleNamespace(
            x=initial,
            fun=misfit,
            population=np.array([initial]),
            population_energies=np.array([misfit]),
            success=True,
            nfev=1,
        )

    frequency = np.arange(10.0, 18.0)[::-1]
    context = {
        "np": np,
        "pd": pd,
        "PARAM": {
            "finite_layer_thickness_bounds_m": [[2.0, 8.0], [8.0, 20.0]],
            "layer_vs_bounds_m_s": [[150.0, 550.0], [200.0, 850.0], [250.0, 1200.0]],
            "poisson_ratio_assumed": 0.3,
            "vp_rule": {"method": "groundwater", "groundwater_depth_m": 0.0},
            "density_assumed_g_cm3": 2.0,
            "initial_vs_phase_velocity_factor": 1.0,
            "preview_runs": 2,
            "preview_maxiter": 2,
            "preview_popsize_multiplier": 5,
            "seeds": [1, 2],
        },
        "period": 1 / frequency,
        "observed": np.full(8, 300.0),
        "selection": pd.DataFrame(
            {"frequency_hz": frequency, "period_s": 1 / frequency, "observed_velocity_m_s": 300.0}
        ),
        "RUN_OUT": tmp_path,
        "mode": "preview",
        "DispersionError": RuntimeError,
        "initial_model_from_wavelength": initial_model_from_wavelength,
        "wavelength_guides": wavelength_guides,
        "differential_evolution": optimizer,
        "forward_rayleigh_phase": lambda period, thickness, velocity, **kw: np.full(8, velocity[0]),
        "VpRule": VpRule,
        "depth_summary": {"minimum_depth_m": 5.0, "maximum_depth_m": 15.0},
    }
    exec(cell(7), context)
    exec(cell(9), context)
    assert len(captured) == 2
    np.testing.assert_allclose(captured[0], [5.0, 14.0, 300.0, 300.0, 300.0])
    np.testing.assert_array_equal(captured[0], captured[1])
    assert (tmp_path / "wavelength_guides.csv").is_file()
    assert context["initial_method"] == "wavelength_endpoint_assumption"
    assert context["initial_df"].model.eq("wavelength_endpoint_assumption").all()
    assert context["vp_rule"].method == "groundwater"
    assert context["bounds_deeper_than_dmax"] is True
    np.testing.assert_allclose(context["initial_df"].vp_m_s, 1.11 * 300.0 + 1290.0)


def test_flat_hvsr_retains_candidate_without_claiming_improvement(tmp_path: Path) -> None:
    nb = json.loads((ROOT / "04_hvsr_refinement_final_vs.ipynb").read_text(encoding="utf-8"))
    base = np.array([5.0, 300.0, 400.0])
    context = {
        "np": np,
        "pd": pd,
        "base": base,
        "bounds": [(1, 10), (200, 400), (300, 500)],
        "period_d": np.array([0.1, 0.2]),
        "velocity_d": np.array([300.0, 300.0]),
        "period_h": np.array([0.1, 0.2]),
        "observed_h": np.ones(2),
        "poisson": 0.3,
        "density": 2.0,
        "vp_rule": None,
        "OUT": tmp_path,
        "n_finite": 1,
        "baseline_corr": float("nan"),
        "PARAM": {
            "random_seed": 1,
            "candidate_count": 1,
            "relative_perturbation_std": 0.08,
            "maximum_dispersion_rmse_increase_m_s": 3,
            "minimum_shape_correlation_gain": 0.02,
        },
        "refine_candidate": lambda *a, **kw: (
            base.copy(),
            [
                {
                    "candidate": 0,
                    "shape_correlation": float("nan"),
                    "dispersion_rmse_m_s": 0.0,
                    "accepted_dispersion": True,
                }
            ],
        ),
        "forward_ellipticity": lambda *a, **kw: np.ones(2),
        "log_shape_correlation": lambda *a: float("nan"),
        "forward_rayleigh_phase": lambda *a, **kw: np.full(2, 300.0),
    }
    exec("".join(nb["cells"][7]["source"]), context)
    assert context["improved"] is False
    np.testing.assert_array_equal(context["selected"], base)


def test_comparison_rejects_modified_final_vs30(tmp_path: Path) -> None:
    nb = json.loads((ROOT / "05_vs30_final_comparison.ipynb").read_text(encoding="utf-8"))
    final = tmp_path / "final.csv"
    pd.DataFrame([{"site_id": "test", "vs30_m_s": 350.0}]).to_csv(final, index=False)
    status = tmp_path / "status.json"
    status.write_text(json.dumps({"vs30_reported": True, "vs30_results_sha256": "stale"}))
    prediction = tmp_path / "prediction.csv"
    pd.DataFrame(
        [
            {
                "site_id": "test",
                "model_name": "comparator",
                "predicted_vs30_m_s": 300,
                "provenance": "test",
            }
        ]
    ).to_csv(prediction, index=False)
    context = {
        "json": json,
        "pd": pd,
        "np": np,
        "hashlib": hashlib,
        "SITE": "test",
        "REFINEMENT_STATUS": status,
        "PREDICTION_INPUT": prediction,
        "FINAL_VS30": final,
    }
    with pytest.raises(ValueError, match="Vs30 changed"):
        exec("".join(nb["cells"][7]["source"]), context)
