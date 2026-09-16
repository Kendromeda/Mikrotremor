"""Phase 3 tests for readers whose components live in separate files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mhvsr_vs30.exceptions import MissingComponentError
from mhvsr_vs30.io.base import ThreeComponentRecord
from mhvsr_vs30.io.registry import get_reader
from tests.reader_helpers import make_request

obspy = pytest.importorskip("obspy")

RATE = 200.0


def sac_trace(channel: str, values: list[float]):
    header = {
        "network": "XX",
        "station": "AAA",
        "channel": channel,
        "sampling_rate": RATE,
        "starttime": obspy.UTCDateTime(2021, 11, 4, 10, 38, 30),
    }
    return obspy.Trace(data=np.asarray(values, dtype=np.float32), header=header)


def write_sac_bundle(directory: Path, suffix: str) -> list[Path]:
    """Write one component per file, naming files so the name hides the channel."""
    paths = []
    for index, (channel, values) in enumerate(
        (("Z", [1.0, 2.0, 3.0]), ("N", [4.0, 5.0, 6.0]), ("E", [7.0, 8.0, 9.0])), start=1
    ):
        path = directory / f"2021.308.10.38.30.AAA.{index}{suffix}"
        obspy.Stream([sac_trace(channel, values)]).write(str(path), format="SAC")
        paths.append(path)
    return paths


def read_bundle(paths: list[Path]) -> ThreeComponentRecord:
    request = make_request(
        paths[0],
        "sac_bundle",
        siblings=tuple(paths),
        declared_sampling_rate_hz=RATE,
        units="counts",
    )
    result = get_reader("sac_bundle").read(request)
    assert isinstance(result, ThreeComponentRecord)
    return result


def test_sac_bundle_groups_same_recording(tmp_path: Path) -> None:
    record = read_bundle(write_sac_bundle(tmp_path, ".sac"))

    assert np.allclose(record.z, [1.0, 2.0, 3.0])
    assert np.allclose(record.north, [4.0, 5.0, 6.0])
    assert np.allclose(record.east, [7.0, 8.0, 9.0])
    assert record.n_samples == 3


def test_sac_hd_is_read_by_content(tmp_path: Path) -> None:
    """The .sac-hd extension is not in ObsPy's table, so detection must be by content."""
    record = read_bundle(write_sac_bundle(tmp_path, ".sac-hd"))

    assert record.source_channels == {"Z": "Z", "N": "N", "E": "E"}
    assert record.sampling_rate_hz == RATE


def test_sac_bundle_rejects_an_incomplete_group(tmp_path: Path) -> None:
    paths = write_sac_bundle(tmp_path, ".sac")
    with pytest.raises(MissingComponentError):
        read_bundle(paths[:2])


def test_sac_bundle_does_not_write_intermediate_files(tmp_path: Path) -> None:
    paths = write_sac_bundle(tmp_path, ".sac")
    before = {p.name for p in tmp_path.rglob("*")}

    read_bundle(paths)

    assert {p.name for p in tmp_path.rglob("*")} == before
