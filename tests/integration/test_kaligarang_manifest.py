"""Integration contract for the locally archived Kaligarang raw recordings."""
from __future__ import annotations

from pathlib import Path

import pytest

from mhvsr_vs30.manifest.builder import build_manifest
from mhvsr_vs30.manifest.schemas import load_source_config

pytestmark = pytest.mark.requires_real_data


def test_kaligarang_manifest_tracks_inversion_profiles_without_labels() -> None:
    config = load_source_config(Path("configs/datasets/kaligarang_semarang_mendeley_v1.yaml"))
    if not Path(config.root).is_dir():
        pytest.skip("Kaligarang release is not present in this workspace")
    manifest = build_manifest(config, ".")
    assert len(manifest.sites) == 41
    assert len(manifest.recordings) == 41
    assert not manifest.labels
