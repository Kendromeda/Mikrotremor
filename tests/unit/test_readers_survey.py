"""Phase 3 tests for multichannel survey inspection."""

from __future__ import annotations

from pathlib import Path

import pytest

from mhvsr_vs30.io.base import SurveyInspection, ThreeComponentRecord
from mhvsr_vs30.io.registry import get_reader
from tests.reader_helpers import make_request

SEG2_SAMPLES = {
    "shot_gather": Path(
        "datasets_global/source_02_usgs_california_sscm/raw/extracted/"
        "2024.SSCM.data/1041_5i_actF.sg2"
    ),
    "three_channel": Path(
        "datasets/source_09_tutorial_hvsr_geopsy_yogyakarta/raw/data/uT12/"
        "20241022_145601000.WIT.3c.cont.0.seg2"
    ),
}


def inspect(path: Path, **overrides: object) -> object:
    request = make_request(path, "seg2", **overrides)
    return get_reader("seg2").read(request)


def _require(name: str) -> Path:
    path = SEG2_SAMPLES[name]
    if not path.is_file():
        pytest.skip(f"SEG-2 sample {name} is not present in this workspace")
    return path


@pytest.mark.requires_real_data
def test_seg2_is_not_assumed_to_be_three_component() -> None:
    """A six-channel shot gather stays a survey, however many channels it has."""
    result = inspect(_require("shot_gather"))

    assert isinstance(result, SurveyInspection)
    assert not isinstance(result, ThreeComponentRecord)
    assert result.trace_count == 6
    assert result.is_three_component_candidate is False


@pytest.mark.requires_real_data
def test_seg2_three_channel_file_is_only_a_candidate() -> None:
    """Three unnamed traces are a candidate, not a recording: orientation is unknown."""
    result = inspect(_require("three_channel"))

    assert isinstance(result, SurveyInspection)
    assert result.trace_count == 3
    assert result.is_three_component_candidate is True
    assert "component_traces" in result.reason


@pytest.mark.requires_real_data
def test_seg2_becomes_a_record_only_when_orientation_is_documented() -> None:
    result = inspect(
        _require("three_channel"),
        component_traces={"Z": 0, "N": 1, "E": 2},
        declared_sampling_rate_hz=2000.0,
        units="counts",
    )

    assert isinstance(result, ThreeComponentRecord)
    assert result.sampling_rate_hz == 2000.0
    assert result.source_channels == {"Z": "trace_0", "N": "trace_1", "E": "trace_2"}
