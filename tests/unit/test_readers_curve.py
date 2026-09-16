"""Phase 3 tests for processed H/V curve reading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mhvsr_vs30.exceptions import MalformedAsciiError, MetadataConflictError
from mhvsr_vs30.io.base import HvsrCurve
from mhvsr_vs30.io.registry import get_reader
from tests.reader_helpers import make_request

ASC_COLUMNS = ("frequency", "mean", "std")
GEOPSY_COLUMNS = ("frequency", "mean", "min", "max")


def read_curve(path: Path, columns: tuple[str, ...] = ASC_COLUMNS) -> HvsrCurve:
    request = make_request(path, "processed_hvsr", curve_columns=columns)
    result = get_reader("processed_hvsr").read(request)
    assert isinstance(result, HvsrCurve)
    return result


def test_reads_a_three_column_curve_with_std(tmp_path: Path) -> None:
    path = tmp_path / "M001.asc"
    path.write_text(
        "Frequency[Hz] H/V STD\n0.125 0.5667 0.1370\n0.15625 0.6082 0.1248\n0.1875 0.7119 0.1396\n",
        encoding="ascii",
    )
    curve = read_curve(path)

    assert np.allclose(curve.frequency_hz, [0.125, 0.15625, 0.1875])
    assert np.allclose(curve.mean_amplitude, [0.5667, 0.6082, 0.7119])
    assert curve.std_amplitude is not None
    assert curve.input_level == "processed_curve"


def test_reads_a_geopsy_curve_without_std(tmp_path: Path) -> None:
    """Geopsy .hv files carry min/max, not a standard deviation."""
    path = tmp_path / "1.hv"
    path.write_text(
        "# GEOPSY output version 1.1\n"
        "# Number of windows = 13\n"
        "# Frequency\tAverage\tMin\tMax\n"
        "0.5\t0.696\t0.4185\t1.1574\n"
        "0.5153\t0.7584\t0.4630\t1.2422\n",
        encoding="ascii",
    )
    curve = read_curve(path, GEOPSY_COLUMNS)

    assert np.allclose(curve.frequency_hz, [0.5, 0.5153])
    assert curve.std_amplitude is None


def test_processed_curve_requires_monotonic_frequency(tmp_path: Path) -> None:
    path = tmp_path / "unsorted.asc"
    path.write_text("0.5 1.0 0.1\n0.4 1.1 0.1\n0.6 1.2 0.1\n", encoding="ascii")

    with pytest.raises(MetadataConflictError):
        read_curve(path)


def test_processed_curve_rejects_non_positive_frequency(tmp_path: Path) -> None:
    path = tmp_path / "zero.asc"
    path.write_text("0.0 1.0 0.1\n0.5 1.1 0.1\n", encoding="ascii")

    with pytest.raises(MetadataConflictError):
        read_curve(path)


def test_processed_curve_rejects_wrong_column_count(tmp_path: Path) -> None:
    path = tmp_path / "wide.asc"
    path.write_text("0.5 1.0 0.1 0.2\n0.6 1.1 0.1 0.2\n", encoding="ascii")

    with pytest.raises(MalformedAsciiError):
        read_curve(path)


def test_curve_columns_must_be_declared(tmp_path: Path) -> None:
    path = tmp_path / "curve.asc"
    path.write_text("0.5 1.0 0.1\n0.6 1.1 0.1\n", encoding="ascii")

    with pytest.raises(MetadataConflictError):
        read_curve(path, ())


def test_curve_samples_are_read_only(tmp_path: Path) -> None:
    path = tmp_path / "curve.asc"
    path.write_text("0.5 1.0 0.1\n0.6 1.1 0.1\n", encoding="ascii")
    curve = read_curve(path)

    with pytest.raises(ValueError):
        curve.mean_amplitude[0] = 5.0
