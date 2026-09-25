"""Integration contract for the locally archived Meulaboh waveform release."""
from __future__ import annotations

from pathlib import Path

import pytest

from mhvsr_vs30.manifest.builder import build_manifest
from mhvsr_vs30.manifest.schemas import load_source_config

pytestmark = pytest.mark.requires_real_data


def test_meulaboh_manifest_keeps_qcpt_and_hvsr_labels_separate() -> None:
    config = load_source_config(Path("configs/datasets/meulaboh_mendeley_v3.yaml"))
    if not Path(config.root).is_dir():
        pytest.skip("Meulaboh release is not present in this workspace")
    manifest = build_manifest(config, ".")
    assert len(manifest.sites) == 20
    assert len(manifest.recordings) == 40
    assert not manifest.labels
