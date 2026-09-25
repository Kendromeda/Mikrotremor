"""Notebook controls for reviewing SPAC dispersion candidates.

The plot helps select and document picks. It does not verify array geometry,
clock drift, wavefield isotropy, or the physical surface-wave mode.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import j0  # type: ignore[import-untyped]

from mhvsr_vs30.mam.wavelength import wavelength_diagnostics

PENDING = "pending_geometry_clock_and_mode_review"


def review_context_hash(*paths: Path) -> str:
    """Tie saved decisions to the data, geometry, and automatic candidates."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def prepare_review_table(
    automatic: pd.DataFrame,
    final_path: Path,
    review_path: Path,
    context_hash: str,
    manual_picks: list[dict[str, Any]],
) -> pd.DataFrame:
    """Restore decisions only when the underlying data have not changed."""
    columns = ["frequency_hz", "period_s", "velocity_auto_m_s", "automatic_status", "qc_reason"]
    table = automatic[columns].copy()
    table["velocity_final_m_s"] = np.nan
    table["accepted_for_inversion"] = False
    table["manual_decision"] = PENDING

    if final_path.exists():
        previous = pd.read_csv(final_path)
        decisions = previous.manual_decision.fillna(PENDING).ne(PENDING)
        decisions |= previous.accepted_for_inversion.astype(str).str.lower().eq("true")
        if decisions.any():
            if not review_path.exists():
                raise RuntimeError(
                    f"{final_path.name} has earlier manual edits without provenance; preserve it "
                    "before starting the notebook picker"
                )
            review = json.loads(review_path.read_text(encoding="utf-8"))
            if review.get("context_sha256") != context_hash:
                raise RuntimeError(
                    "Waveforms, coordinates, or automatic picks changed. Preserve the previous "
                    "dispersion review before starting a new one."
                )
            if len(previous) != len(table) or not np.allclose(
                previous.frequency_hz, table.frequency_hz, rtol=1e-9, atol=1e-9
            ):
                raise RuntimeError("Saved pick frequencies differ from the current frequency grid")
            table["velocity_final_m_s"] = previous.velocity_final_m_s.to_numpy()
            table["accepted_for_inversion"] = (
                previous.accepted_for_inversion.astype(str).str.lower().eq("true").to_numpy()
            )
            table["manual_decision"] = previous.manual_decision.fillna(PENDING).to_numpy()

    used: set[int] = set()
    for pick in manual_picks:
        frequency = float(pick["frequency_hz"])
        velocity = float(pick["velocity_m_s"])
        reason = str(pick["reason"]).strip()
        matches = np.flatnonzero(
            np.isclose(table.frequency_hz.to_numpy(), frequency, rtol=1e-6, atol=1e-6)
        )
        if (
            len(matches) != 1
            or int(matches[0]) in used
            or not np.isfinite(velocity)
            or velocity <= 0
            or not reason
        ):
            raise ValueError(f"Invalid or duplicate manual pick: {pick}")
        index = int(matches[0])
        used.add(index)
        table.loc[index, ["velocity_final_m_s", "accepted_for_inversion", "manual_decision"]] = (
            velocity,
            True,
            reason,
        )
    return table


def save_review_table(
    table: pd.DataFrame,
    final_path: Path,
    review_path: Path,
    context_hash: str,
    qc_path: Path | None = None,
    *,
    context_paths: Sequence[Path] | None = None,
) -> None:
    """Save decisions and refresh downstream QC metadata."""
    if context_paths is not None and review_context_hash(*context_paths) != context_hash:
        raise ValueError("Review context changed; rerun notebook 02 before saving picks")
    accepted = table.accepted_for_inversion.astype(bool)
    if (accepted & table.automatic_status.ne("provisional")).any():
        raise ValueError("Cannot accept a candidate that failed numerical QC")
    velocities = pd.to_numeric(table.velocity_final_m_s, errors="coerce")
    if (accepted & (~np.isfinite(velocities) | velocities.le(0))).any():
        raise ValueError("Every accepted pick needs a positive finite phase velocity")
    if (accepted & table.manual_decision.fillna("").isin(["", PENDING])).any():
        raise ValueError("Every accepted pick needs a review reason")
    qc = (
        json.loads(qc_path.read_text(encoding="utf-8"))
        if qc_path is not None and qc_path.exists()
        else None
    )
    if qc is not None and accepted.any():
        selected = table.loc[accepted]
        wavelength = wavelength_diagnostics(
            selected.frequency_hz.to_numpy(dtype=float),
            velocities.loc[accepted].to_numpy(dtype=float),
            float(qc["maximum_receiver_spacing_m"]),
            maximum_wavelength_ratio=qc["processing_parameters"].get("maximum_wavelength_ratio"),
        )
        if not wavelength["within_wavelength_limit"].all():
            raise ValueError("Accepted pick exceeds the project wavelength limit")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(final_path, index=False)
    review = {
        "context_sha256": context_hash,
        "saved_at_utc": datetime.now(UTC).isoformat(),
        "accepted_picks": int(accepted.sum()),
        "rejected_by_reviewer": int(
            table.manual_decision.fillna("").str.startswith("rejected:").sum()
        ),
        "source": "notebook_02_review_controls",
    }
    review_path.write_text(json.dumps(review, indent=2), encoding="utf-8")
    if qc is not None and qc_path is not None:
        qc["n_final_accepted_picks"] = int(accepted.sum())
        qc["dispersion_final_sha256"] = hashlib.sha256(final_path.read_bytes()).hexdigest()
        qc["review_context_sha256"] = context_hash
        qc["review_saved_at_utc"] = review["saved_at_utc"]
        qc["blocking_review"] = [
            label
            for key, label in (
                ("geometry_verified", "true sensor positions"),
                ("physical_clock_drift_verified", "instrument clock drift"),
                ("wavefield_mode_verified", "wavefield isotropy and mode"),
                ("wavelength_range_reviewed", "wavelength range review"),
            )
            if not qc.get(key, False)
        ] + (["manual dispersion pick review"] if not accepted.any() else [])
        qc_path.write_text(json.dumps(qc, indent=2), encoding="utf-8")


def create_dispersion_picker(
    automatic: pd.DataFrame,
    review: pd.DataFrame,
    pair_names: list[tuple[str, str]],
    distances_m: np.ndarray,
    coherency: np.ndarray,
    block_sem: np.ndarray,
    block_curves: np.ndarray,
    velocities_m_s: np.ndarray,
    final_path: Path,
    review_path: Path,
    context_hash: str,
    qc_path: Path,
    *,
    context_paths: Sequence[Path] | None = None,
) -> Any:
    """Return a Jupyter widget with clickable picks and per-frequency diagnostics."""
    import ipywidgets as widgets  # type: ignore[import-untyped]
    import plotly.graph_objects as go  # type: ignore[import-untyped]
    from IPython.display import display
    from plotly.subplots import make_subplots  # type: ignore[import-untyped]

    frequency = automatic.frequency_hz.to_numpy(dtype=float)
    velocity_auto = automatic.velocity_auto_m_s.to_numpy(dtype=float)
    provisional = automatic.automatic_status.eq("provisional").to_numpy()
    if coherency.shape != (len(frequency), len(pair_names)):
        raise ValueError("SPAC array does not match the automatic pick table")

    figure = go.FigureWidget()
    for indices, label, color, symbol in (
        (np.flatnonzero(~provisional), "QC numerik gagal", "#9ca3af", "x"),
        (np.flatnonzero(provisional), "Kandidat otomatis", "#2563eb", "circle"),
    ):
        figure.add_scatter(
            x=frequency[indices],
            y=velocity_auto[indices],
            mode="markers",
            name=label,
            customdata=indices,
            marker={"color": color, "symbol": symbol, "size": 8},
            hovertemplate="%{x:.2f} Hz · %{y:.1f} m/s<extra>" + label + "</extra>",
        )
    figure.add_scatter(
        x=[],
        y=[],
        mode="lines+markers",
        name="Pick diterima",
        marker={"color": "#dc2626", "size": 10},
        line={"color": "#dc2626"},
    )
    figure.add_scatter(
        x=[],
        y=[],
        mode="markers",
        name="Frekuensi dipilih",
        marker={"color": "#f59e0b", "size": 17, "symbol": "circle-open", "line": {"width": 3}},
    )
    figure.update_layout(
        title="Klik kandidat untuk memeriksa dan memilih pick dispersi",
        xaxis={"title": "Frekuensi (Hz)", "type": "log"},
        yaxis={"title": "Kecepatan fase (m/s)"},
        hovermode="closest",
        height=430,
        margin={"t": 55, "b": 55},
    )

    options = [
        (f"{hz:.2f} Hz · {automatic.automatic_status.iloc[i]}", i) for i, hz in enumerate(frequency)
    ]
    first = int(np.flatnonzero(provisional)[0]) if provisional.any() else 0
    choose = widgets.Dropdown(
        options=options, value=first, description="Frekuensi:", layout=widgets.Layout(width="420px")
    )
    speed = widgets.FloatText(description="c (m/s):", layout=widgets.Layout(width="230px"))
    reason = widgets.Textarea(
        description="Alasan:",
        rows=2,
        placeholder="Contoh: cabang kontinu; fit keenam pasangan konsisten",
        layout=widgets.Layout(width="600px"),
    )
    accept_button = widgets.Button(description="Terima / perbarui", button_style="success")
    reject_button = widgets.Button(description="Tolak", button_style="warning")
    clear_button = widgets.Button(description="Bersihkan")
    save_button = widgets.Button(description="Simpan untuk notebook 03", button_style="primary")
    info = widgets.HTML()
    message = widgets.HTML(value="Keputusan baru tersimpan setelah tombol Simpan ditekan.")
    diagnostic = widgets.Output()

    def refresh_pick_line() -> None:
        selected = review.accepted_for_inversion.astype(bool).to_numpy()
        with figure.batch_update():
            figure.data[2].x = frequency[selected]
            figure.data[2].y = review.loc[selected, "velocity_final_m_s"].to_numpy(dtype=float)

    def draw_diagnostic(index: int) -> None:
        if not np.isfinite(speed.value) or speed.value <= 0:
            return
        observed = coherency[index]
        predicted = j0(
            2 * np.pi * frequency[index] * distances_m[:, None] / velocities_m_s[None, :]
        )
        misfit = np.sqrt(np.mean((predicted - observed[:, None]) ** 2, axis=0))
        graph = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=("SPAC vs jarak", "Galat J₀ vs kecepatan", "Variasi antarblok"),
        )
        graph.add_scatter(
            x=distances_m,
            y=observed,
            mode="markers",
            name="Terukur",
            error_y={"type": "data", "array": block_sem[index], "visible": True},
            marker={"size": 9},
            row=1,
            col=1,
        )
        radius = np.linspace(0, distances_m.max() * 1.1, 250)
        graph.add_scatter(
            x=radius,
            y=j0(2 * np.pi * frequency[index] * radius / speed.value),
            name="J₀ pada c terpilih",
            row=1,
            col=1,
        )
        graph.add_scatter(x=velocities_m_s, y=misfit, name="RMSE", row=1, col=2)
        graph.add_vline(x=velocity_auto[index], line_dash="dot", line_color="#2563eb", row=1, col=2)
        graph.add_vline(x=speed.value, line_dash="dash", line_color="#dc2626", row=1, col=2)
        for pair_index, pair in enumerate(pair_names):
            graph.add_scatter(
                x=np.arange(1, len(block_curves) + 1),
                y=block_curves[:, index, pair_index],
                mode="lines+markers",
                name="-".join(pair),
                showlegend=False,
                row=1,
                col=3,
            )
        graph.update_xaxes(title_text="Jarak (m)", row=1, col=1)
        graph.update_xaxes(title_text="Kecepatan fase (m/s)", type="log", row=1, col=2)
        graph.update_xaxes(title_text="Blok 4 window", row=1, col=3)
        graph.update_yaxes(title_text="Koherensi", row=1, col=1)
        graph.update_yaxes(title_text="RMSE", row=1, col=2)
        graph.update_yaxes(title_text="Koherensi", row=1, col=3)
        graph.update_layout(height=400, width=1250, margin={"t": 55, "b": 45})
        with diagnostic:
            diagnostic.clear_output(wait=True)
            display(graph)  # type: ignore[no-untyped-call]

    changing_frequency = False

    def show_frequency(index: int) -> None:
        nonlocal changing_frequency
        row = automatic.iloc[index]
        chosen = review.iloc[index]
        changing_frequency = True
        try:
            speed.value = (
                float(chosen.velocity_final_m_s)
                if bool(chosen.accepted_for_inversion)
                else float(row.velocity_auto_m_s)
            )
            decision = str(chosen.manual_decision)
            reason.value = "" if decision == PENDING else decision.removeprefix("rejected: ")
        finally:
            changing_frequency = False
        info.value = (
            f"<b>{frequency[index]:.2f} Hz</b> · status otomatis: {row.automatic_status} · "
            f"QC: {row.qc_reason} · RMSE J₀: {row.fit_rmse:.3f} · "
            f"selisih geometri: {row.geometry_velocity_difference_m_s:.1f} m/s"
        )
        with figure.batch_update():
            figure.data[3].x = [frequency[index]]
            figure.data[3].y = [velocity_auto[index]]
        draw_diagnostic(index)

    def on_speed(change: dict[str, Any]) -> None:
        if change.get("name") == "value" and not changing_frequency:
            draw_diagnostic(int(choose.value))

    def on_choose(change: dict[str, Any]) -> None:
        if change.get("name") == "value" and change.get("new") is not None:
            show_frequency(int(change["new"]))

    def on_plot_click(trace: Any, points: Any, state: Any) -> None:
        if points.point_inds:
            choose.value = int(trace.customdata[points.point_inds[0]])

    def on_accept(button: Any) -> None:
        index = int(choose.value)
        if not provisional[index]:
            message.value = "<b>Kandidat ini gagal QC numerik dan tidak dapat diterima.</b>"
            return
        note = reason.value.strip()
        if not np.isfinite(speed.value) or speed.value <= 0 or not note:
            message.value = "<b>Isi kecepatan positif dan alasan sebelum menerima pick.</b>"
            return
        review.loc[index, ["velocity_final_m_s", "accepted_for_inversion", "manual_decision"]] = (
            float(speed.value),
            True,
            note,
        )
        refresh_pick_line()
        message.value = f"Pick {frequency[index]:.2f} Hz diterima di memori; tekan Simpan."

    def on_reject(button: Any) -> None:
        index = int(choose.value)
        note = reason.value.strip()
        if not note:
            message.value = "<b>Isi alasan sebelum menolak frekuensi.</b>"
            return
        review.loc[index, ["velocity_final_m_s", "accepted_for_inversion", "manual_decision"]] = (
            np.nan,
            False,
            f"rejected: {note}",
        )
        refresh_pick_line()
        message.value = f"Frekuensi {frequency[index]:.2f} Hz ditolak di memori; tekan Simpan."

    def on_clear(button: Any) -> None:
        index = int(choose.value)
        review.loc[index, ["velocity_final_m_s", "accepted_for_inversion", "manual_decision"]] = (
            np.nan,
            False,
            PENDING,
        )
        reason.value = ""
        refresh_pick_line()
        message.value = f"Keputusan {frequency[index]:.2f} Hz dibersihkan di memori; tekan Simpan."

    def on_save(button: Any) -> None:
        try:
            save_review_table(
                review,
                final_path,
                review_path,
                context_hash,
                qc_path,
                context_paths=context_paths,
            )
        except (ValueError, OSError) as error:
            message.value = f"<b>Gagal menyimpan:</b> {error}"
            return
        count = int(review.accepted_for_inversion.sum())
        message.value = (
            f"<b>Tersimpan:</b> {count} pick diterima di {final_path.name}. "
            "Notebook 03 tetap preview sampai seluruh verifikasi ilmiah lulus."
        )

    for trace in figure.data[:2]:
        trace.on_click(on_plot_click)
    choose.observe(on_choose, names="value")
    speed.observe(on_speed, names="value")
    accept_button.on_click(on_accept)
    reject_button.on_click(on_reject)
    clear_button.on_click(on_clear)
    save_button.on_click(on_save)
    refresh_pick_line()
    show_frequency(first)
    return widgets.VBox(
        [
            info,
            figure,
            widgets.HBox([choose, speed]),
            reason,
            widgets.HBox([accept_button, reject_button, clear_button, save_button]),
            message,
            diagnostic,
        ]
    )
