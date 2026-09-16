"""Shared fixtures.

The synthetic source tree mirrors the shape of a real USGS ARRA site archive
(nested site folder, HVSR subfolder, survey files, photos, reports) at a size
that keeps unit tests fast.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

REAL_USGS_CONFIG = Path("configs/datasets/usgs_arra.yaml")
REAL_USGS_ROOT = Path("datasets_global/source_05_usgs_arra_2013/raw/extracted")


def _write_ascii_3c(path: Path, samples: int = 40) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{i}\t{-i}\t{i * 2}" for i in range(samples)]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A miniature two-site source tree plus a config that describes it."""
    root = tmp_path / "datasets_global" / "demo" / "raw" / "extracted"

    for site, vs30 in (("XX.AAA", 200.0), ("XX.BBB", 800.0)):
        short = site.split(".")[1]
        base = root / site / site / f"{site.replace('.', '_')}_Raw_Data"
        for stamp in ("20110101000000", "20110101010000"):
            _write_ascii_3c(base / f"{short}_HVSR" / f"{stamp}.Q288.txt")
        (base / f"{short}_MASW").mkdir(parents=True, exist_ok=True)
        (base / f"{short}_MASW" / "1001.dat").write_bytes(b"seg2-ish")
        (base / f"{short}_SASW").mkdir(parents=True, exist_ok=True)
        (base / f"{short}_SASW" / "A1.DAT").write_bytes(b"hp-sdf-ish")
        (base / f"{short}_Photos").mkdir(parents=True, exist_ok=True)
        (base / f"{short}_Photos" / "DSC00001.JPG").write_bytes(b"jpeg-ish")
        (base / f"{short}_Photos" / "Thumbs.db").write_bytes(b"windows-junk")
        (base / f"CI-{short} Locations.xls").write_bytes(b"xls-ish")
        (base / f"{site}_Forms.pdf").write_bytes(b"pdf-ish")
        (root / site / site / f"{site}.pdf").write_bytes(b"report-ish")
        del vs30

    config = tmp_path / "configs" / "datasets" / "demo.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        textwrap.dedent(
            """
            source_id: demo_source
            title: Demo source
            license: CC0
            country: Nowhere
            root: datasets_global/demo/raw/extracted

            recording:
              format: usgs_ascii_3c
              component_order: [vertical, north, east]
              sampling_rate_hz: 200
              units: counts
              include:
                - "*_HVSR/*.txt"

            asset_rules:
              - pattern: "*_HVSR/*.txt"
                media_type: ambient_noise_waveform
              - pattern: "*_MASW/*.dat"
                media_type: surface_wave_survey
              - pattern: "*_SASW/*.DAT"
                media_type: surface_wave_survey

            vs30_range_mps: [50, 3000]

            expectations:
              site_count: 2
              recording_count: 4
              recordings_per_site: 2

            sites:
              - site_code: XX.AAA
                label:
                  vs30_mps: 200
                  label_method: surface_wave_joint_inversion
                  label_independence: independent_of_hvsr
                  reference: Demo report
                  reference_locator: Table 1
              - site_code: XX.BBB
                label:
                  vs30_mps: 800
                  label_method: surface_wave_joint_inversion
                  label_independence: independent_of_hvsr
                  reference: Demo report
                  reference_locator: Table 1
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def demo_config(workspace: Path) -> Path:
    return workspace / "configs" / "datasets" / "demo.yaml"


@pytest.fixture
def real_usgs_config() -> Path:
    """Path to the real USGS config, skipping the test when the data is absent."""
    if not REAL_USGS_CONFIG.exists() or not REAL_USGS_ROOT.exists():
        pytest.skip("real USGS ARRA dataset is not present in this workspace")
    return REAL_USGS_CONFIG
