"""Phase 3 tests for the ObsPy-backed readers, using synthetic streams."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mhvsr_vs30.exceptions import (
    AmbiguousChannelError,
    MetadataConflictError,
    MissingComponentError,
    NoCommonTimeWindowError,
    SamplingRateMismatchError,
)
from mhvsr_vs30.io.base import ThreeComponentRecord
from mhvsr_vs30.io.registry import get_reader
from tests.reader_helpers import make_request

obspy = pytest.importorskip("obspy")

RATE = 100.0


def trace(channel: str, values: list[float], start_offset: float = 0.0, rate: float = RATE):
    header = {
        "network": "XX",
        "station": "AAA",
        "channel": channel,
        "sampling_rate": rate,
        "starttime": obspy.UTCDateTime(2011, 2, 2, 19, 26, 37) + start_offset,
    }
    return obspy.Trace(data=np.asarray(values, dtype=np.float64), header=header)


def write_mseed(path: Path, traces: list) -> Path:
    obspy.Stream(traces).write(str(path), format="MSEED")
    return path


def read_mseed(path: Path, **overrides: object) -> ThreeComponentRecord:
    options: dict[str, object] = {"declared_sampling_rate_hz": RATE, "units": "counts"}
    options.update(overrides)
    result = get_reader("miniseed").read(make_request(path, "miniseed", **options))
    assert isinstance(result, ThreeComponentRecord)
    return result


def test_miniseed_maps_orientation_codes(tmp_path: Path) -> None:
    path = write_mseed(
        tmp_path / "three.mseed",
        [
            trace("BHZ", [1.0, 2.0, 3.0, 4.0]),
            trace("BHN", [5.0, 6.0, 7.0, 8.0]),
            trace("BHE", [9.0, 10.0, 11.0, 12.0]),
        ],
    )
    record = read_mseed(path)

    assert np.allclose(record.z, [1.0, 2.0, 3.0, 4.0])
    assert np.allclose(record.north, [5.0, 6.0, 7.0, 8.0])
    assert record.source_channels == {"Z": "BHZ", "N": "BHN", "E": "BHE"}


def test_miniseed_trims_to_common_interval(tmp_path: Path) -> None:
    """Components that start at different times are trimmed, never padded."""
    path = write_mseed(
        tmp_path / "ragged.mseed",
        [
            trace("BHZ", [float(i) for i in range(10)]),
            trace("BHN", [float(i) for i in range(10)], start_offset=0.02),
            trace("BHE", [float(i) for i in range(8)], start_offset=0.02),
        ],
    )
    record = read_mseed(path)

    assert record.n_samples == 8
    assert np.allclose(record.z, [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    assert np.allclose(record.north, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])


def test_miniseed_rejects_missing_vertical(tmp_path: Path) -> None:
    path = write_mseed(
        tmp_path / "horizontal_only.mseed",
        [trace("BHN", [1.0, 2.0]), trace("BHE", [3.0, 4.0])],
    )
    with pytest.raises(MissingComponentError):
        read_mseed(path)


def test_miniseed_rejects_ambiguous_channels(tmp_path: Path) -> None:
    """Codes like BH1/BH2 carry no documented orientation, so they are refused."""
    path = write_mseed(
        tmp_path / "numbered.mseed",
        [trace("BHZ", [1.0, 2.0]), trace("BH1", [3.0, 4.0]), trace("BH2", [5.0, 6.0])],
    )
    with pytest.raises(AmbiguousChannelError):
        read_mseed(path)


def test_miniseed_accepts_explicitly_configured_channel_codes(tmp_path: Path) -> None:
    path = write_mseed(
        tmp_path / "numbered.mseed",
        [trace("BHZ", [1.0, 2.0]), trace("BH1", [3.0, 4.0]), trace("BH2", [5.0, 6.0])],
    )
    record = read_mseed(path, channel_codes={"Z": "BHZ", "N": "BH1", "E": "BH2"})

    assert np.allclose(record.north, [3.0, 4.0])
    assert record.source_channels["E"] == "BH2"


def test_miniseed_rejects_duplicate_component(tmp_path: Path) -> None:
    path = write_mseed(
        tmp_path / "dupe.mseed",
        [
            trace("BHZ", [1.0, 2.0]),
            trace("HHZ", [1.0, 2.0]),
            trace("BHN", [3.0, 4.0]),
            trace("BHE", [5.0, 6.0]),
        ],
    )
    with pytest.raises(AmbiguousChannelError):
        read_mseed(path)


def test_miniseed_rejects_mixed_sampling_rates(tmp_path: Path) -> None:
    path = write_mseed(
        tmp_path / "mixed.mseed",
        [
            trace("BHZ", [1.0, 2.0]),
            trace("BHN", [3.0, 4.0], rate=50.0),
            trace("BHE", [5.0, 6.0]),
        ],
    )
    with pytest.raises(SamplingRateMismatchError):
        read_mseed(path)


def test_miniseed_rejects_rate_conflicting_with_source_metadata(tmp_path: Path) -> None:
    path = write_mseed(
        tmp_path / "three.mseed",
        [trace("BHZ", [1.0, 2.0]), trace("BHN", [3.0, 4.0]), trace("BHE", [5.0, 6.0])],
    )
    with pytest.raises(MetadataConflictError):
        read_mseed(path, declared_sampling_rate_hz=200.0)


def test_miniseed_rejects_disjoint_components(tmp_path: Path) -> None:
    path = write_mseed(
        tmp_path / "disjoint.mseed",
        [
            trace("BHZ", [1.0, 2.0]),
            trace("BHN", [3.0, 4.0], start_offset=600.0),
            trace("BHE", [5.0, 6.0], start_offset=600.0),
        ],
    )
    with pytest.raises(NoCommonTimeWindowError):
        read_mseed(path)
