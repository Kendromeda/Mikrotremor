"""The notebook review controls must preserve picks and downstream integrity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mhvsr_vs30.mam.picking import (
    PENDING,
    create_dispersion_picker,
    prepare_review_table,
    review_context_hash,
    save_review_table,
)


def _automatic() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frequency_hz": [10.0, 15.0, 20.0],
            "period_s": [0.1, 1 / 15, 0.05],
            "velocity_auto_m_s": [200.0, 210.0, 220.0],
            "automatic_status": ["provisional"] * 3,
            "qc_reason": ["numerical_checks_passed"] * 3,
        }
    )


def test_saved_pick_restores_and_refreshes_qc_hash(tmp_path: Path) -> None:
    context_file = tmp_path / "site_inputs.json"
    context_file.write_text('{"site":"test"}', encoding="utf-8")
    context = review_context_hash(context_file)
    final_path = tmp_path / "dispersion_final.csv"
    review_path = tmp_path / "dispersion_review.json"
    qc_path = tmp_path / "dispersion_qc.json"
    qc_path.write_text(
        json.dumps(
            {
                "dispersion_final_sha256": "obsolete",
                "geometry_verified": False,
                "physical_clock_drift_verified": False,
                "wavefield_mode_verified": False,
                "wavelength_range_reviewed": False,
                "maximum_receiver_spacing_m": 5.0,
                "processing_parameters": {"maximum_wavelength_ratio": None},
            }
        ),
        encoding="utf-8",
    )
    table = prepare_review_table(_automatic(), final_path, review_path, context, [])
    table.loc[1, ["velocity_final_m_s", "accepted_for_inversion", "manual_decision"]] = (
        205.0,
        True,
        "fit antarpasangan dan cabang kurva diperiksa",
    )
    save_review_table(table, final_path, review_path, context, qc_path)

    restored = prepare_review_table(_automatic(), final_path, review_path, context, [])
    assert bool(restored.loc[1, "accepted_for_inversion"])
    assert restored.loc[1, "velocity_final_m_s"] == 205.0
    assert restored.loc[0, "manual_decision"] == PENDING
    qc = json.loads(qc_path.read_text(encoding="utf-8"))
    assert qc["n_final_accepted_picks"] == 1
    assert qc["dispersion_final_sha256"] == hashlib.sha256(final_path.read_bytes()).hexdigest()
    assert "wavelength range review" in qc["blocking_review"]


def test_saved_pick_rejects_configured_wavelength_limit(tmp_path: Path) -> None:
    final_path = tmp_path / "dispersion_final.csv"
    review_path = tmp_path / "dispersion_review.json"
    qc_path = tmp_path / "dispersion_qc.json"
    qc_path.write_text(
        json.dumps(
            {
                "maximum_receiver_spacing_m": 5.0,
                "processing_parameters": {"maximum_wavelength_ratio": 2.0},
            }
        ),
        encoding="utf-8",
    )
    table = prepare_review_table(_automatic(), final_path, review_path, "context", [])
    table.loc[0, ["velocity_final_m_s", "accepted_for_inversion", "manual_decision"]] = (
        300.0,
        True,
        "perlu diperiksa",
    )
    with pytest.raises(ValueError, match="wavelength"):
        save_review_table(table, final_path, review_path, "context", qc_path)
    assert not final_path.exists()


def test_review_context_changes_with_source_file(tmp_path: Path) -> None:
    source = tmp_path / "site_inputs.json"
    source.write_text("before", encoding="utf-8")
    old = review_context_hash(source)
    source.write_text("after", encoding="utf-8")
    assert review_context_hash(source) != old


def test_stale_picker_cannot_overwrite_new_run(tmp_path: Path) -> None:
    source = tmp_path / "site_inputs.json"
    source.write_text("before", encoding="utf-8")
    context = review_context_hash(source)
    final_path = tmp_path / "dispersion_final.csv"
    final_path.write_text("new run", encoding="utf-8")
    table = prepare_review_table(
        _automatic(), tmp_path / "missing.csv", tmp_path / "review.json", context, []
    )
    source.write_text("after", encoding="utf-8")
    with pytest.raises(ValueError, match="context"):
        save_review_table(
            table, final_path, tmp_path / "review.json", context, context_paths=(source,)
        )
    assert final_path.read_text(encoding="utf-8") == "new run"


def test_rejected_numeric_candidate_cannot_be_accepted(tmp_path: Path) -> None:
    table = prepare_review_table(
        _automatic(), tmp_path / "missing.csv", tmp_path / "review.json", "context", []
    )
    table.loc[
        0, ["automatic_status", "velocity_final_m_s", "accepted_for_inversion", "manual_decision"]
    ] = ("rejected_numeric", 200.0, True, "picked")
    final_path = tmp_path / "final.csv"
    with pytest.raises(ValueError, match="numerical QC"):
        save_review_table(table, final_path, tmp_path / "review.json", "context")
    assert not final_path.exists()


def test_widget_accept_and_save_reaches_dispersion_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import IPython.display
    from scipy.special import j0

    displayed: list = []
    monkeypatch.setattr(IPython.display, "display", displayed.append)

    automatic = _automatic().assign(fit_rmse=0.01, geometry_velocity_difference_m_s=0.0)
    frequency = automatic.frequency_hz.to_numpy()
    distances = np.array([2.0, 2.5, 3.0])
    coherency = j0(2 * np.pi * frequency[:, None] * distances[None, :] / 205.0)
    final_path = tmp_path / "dispersion_final.csv"
    review_path = tmp_path / "dispersion_review.json"
    qc_path = tmp_path / "dispersion_qc.json"
    qc_path.write_text(
        json.dumps(
            {
                "maximum_receiver_spacing_m": 3.0,
                "processing_parameters": {"maximum_wavelength_ratio": None},
                "wavelength_range_reviewed": False,
            }
        ),
        encoding="utf-8",
    )
    source = tmp_path / "site_inputs.json"
    source.write_text("source", encoding="utf-8")
    context = review_context_hash(source)
    review = prepare_review_table(automatic, final_path, review_path, context, [])
    picker = create_dispersion_picker(
        automatic,
        review,
        [("A", "B"), ("A", "C"), ("B", "C")],
        distances,
        coherency,
        np.full((3, 3), 0.01),
        np.repeat(coherency[None, :, :], 4, axis=0),
        np.linspace(100, 400, 31),
        final_path,
        review_path,
        context,
        qc_path,
        context_paths=(source,),
    )
    choose, speed = picker.children[2].children
    reason = picker.children[3]
    accept, _, _, save = picker.children[4].children
    choose.value = 1
    speed.value = 205.0
    assert displayed[-1].data[1].y[1] == pytest.approx(
        j0(2 * np.pi * 15.0 * displayed[-1].data[1].x[1] / 205.0)
    )
    reason.value = "cabang kurva dan pasangan konsisten"
    accept.click()
    save.click()

    final = pd.read_csv(final_path)
    assert bool(final.loc[1, "accepted_for_inversion"])
    assert final.loc[1, "velocity_final_m_s"] == 205.0
    assert (
        json.loads(qc_path.read_text())["dispersion_final_sha256"]
        == hashlib.sha256(final_path.read_bytes()).hexdigest()
    )
    source.write_text("new source", encoding="utf-8")
    final_path.write_text("new run", encoding="utf-8")
    save.click()
    assert "Gagal menyimpan" in picker.children[5].value
    assert final_path.read_text(encoding="utf-8") == "new run"
