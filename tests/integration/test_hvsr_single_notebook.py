"""Notebook-level checks for the single-site HVSR inversion workflow."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

from mhvsr_vs30.hvsr_inversion import forward_body_wave_hvsr

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "06_hvsr_single_site_pso_inversion.ipynb"


def _write_synthetic_hv(path: Path) -> None:
    frequency = np.geomspace(0.5, 10.0, 80)
    amplitude = forward_body_wave_hvsr(
        frequency,
        thickness_m=np.array([20.0]),
        vs_m_s=np.array([250.0, 800.0]),
        qp=30.0,
        qs=10.0,
    )
    f0_index = int(np.argmax(amplitude))
    f0 = float(frequency[f0_index])
    lines = [
        "# GEOPSY output version 1.1",
        "# Number of windows = 20",
        f"# f0 from average {f0}",
        "# Number of windows for f0 = 20",
        f"# f0 from windows {f0} {0.9 * f0} {1.1 * f0}",
        f"# f0 amplitude {amplitude[f0_index]}",
        "# Frequency Average Min Max",
    ]
    lines.extend(
        f"{value:.8f} {mean:.8f} {0.8 * mean:.8f} {1.2 * mean:.8f}"
        for value, mean in zip(frequency, amplitude, strict=True)
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_notebook_declares_body_wave_forward_without_ellipticity() -> None:
    text = NOTEBOOK.read_text(encoding="utf-8")

    assert "forward_body_wave_hvsr" in text
    assert "Herak-style" in text
    assert "disba.Ellipticity" not in text
    assert "run_pso_hvsr_inversion" in text


def test_notebook_executes_end_to_end_with_a_small_geopsy_curve(tmp_path: Path) -> None:
    matplotlib.use("Agg")
    hv_path = tmp_path / "synthetic.hv"
    output_dir = tmp_path / "outputs"
    _write_synthetic_hv(hv_path)
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    namespace: dict[str, object] = {"__name__": "__main__"}

    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"]).replace("%matplotlib inline", "")
        source = source.replace(
            'HV_FILE = Path(r"D:\\Kulter_2026\\hasil_HV\\T68.hv")',
            f'HV_FILE = Path(r"{hv_path}")',
        )
        source = source.replace("PSO_PARTICLES = 50", "PSO_PARTICLES = 6")
        source = source.replace("PSO_ITERATIONS = 80", "PSO_ITERATIONS = 2")
        source = source.replace(
            "PSO_SEEDS = [2024, 2025, 2026, 2027]", "PSO_SEEDS = [2024, 2025]"
        )
        source = source.replace(
            "REQUIRE_NONDECREASING_VS = True", "REQUIRE_NONDECREASING_VS = False"
        )
        source = source.replace(
            'OUTPUT_DIR = PROJECT_ROOT / "outputs" / "hvsr_single" / HV_FILE.stem',
            f'OUTPUT_DIR = Path(r"{output_dir}")',
        )
        exec(compile(source, NOTEBOOK.name, "exec"), namespace)

    summary = json.loads((output_dir / "inversion_summary.json").read_text(encoding="utf-8"))
    assert summary["run_state"] == "complete"
    assert summary["method"]["forward"].startswith("1D vertically incident S/P body-wave")
    assert (output_dir / "all_models.csv").is_file()
    assert (output_dir / "best_models.csv").is_file()
    assert (output_dir / "inversion_qc.png").is_file()
