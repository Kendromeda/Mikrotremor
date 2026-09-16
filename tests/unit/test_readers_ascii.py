"""Phase 3 tests for the ASCII readers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mhvsr_vs30.exceptions import (
    MalformedAsciiError,
    MetadataConflictError,
    NonFiniteSampleError,
    UnsupportedFormatError,
)
from mhvsr_vs30.io.base import ThreeComponentRecord
from mhvsr_vs30.io.registry import available_formats, get_reader
from tests.reader_helpers import make_request


def write_lines(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path


def usgs_file(tmp_path: Path, rows: list[tuple[float, float, float]]) -> Path:
    return write_lines(
        tmp_path / "20110202192637.Q292.txt",
        [f"{v}\t{n}\t{e}" for v, n, e in rows],
    )


def read_usgs(path: Path, **overrides: object) -> ThreeComponentRecord:
    options: dict[str, object] = {"declared_sampling_rate_hz": 200.0, "units": "counts"}
    options.update(overrides)
    request = make_request(path, "usgs_ascii_3c", **options)
    result = get_reader("usgs_ascii_3c").read(request)
    assert isinstance(result, ThreeComponentRecord)
    return result


def test_usgs_ascii_maps_vertical_north_east(tmp_path: Path) -> None:
    """Column order comes from the documented order, not from column statistics."""
    path = usgs_file(tmp_path, [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)])
    record = read_usgs(path)

    assert np.array_equal(record.z, [1.0, 4.0])
    assert np.array_equal(record.north, [2.0, 5.0])
    assert np.array_equal(record.east, [3.0, 6.0])
    assert record.source_channels == {"Z": "column_1", "N": "column_2", "E": "column_3"}


def test_usgs_ascii_honours_a_different_documented_order(tmp_path: Path) -> None:
    path = usgs_file(tmp_path, [(1.0, 2.0, 3.0)])
    record = read_usgs(path, component_order=("north", "east", "vertical"))

    assert record.north[0] == 1.0
    assert record.east[0] == 2.0
    assert record.z[0] == 3.0


def test_usgs_ascii_rejects_two_columns(tmp_path: Path) -> None:
    path = write_lines(tmp_path / "two.txt", ["1\t2", "3\t4"])
    with pytest.raises(MalformedAsciiError):
        read_usgs(path)


def test_usgs_ascii_rejects_four_columns(tmp_path: Path) -> None:
    path = write_lines(tmp_path / "four.txt", ["1\t2\t3\t4"])
    with pytest.raises(MalformedAsciiError):
        read_usgs(path)


def test_usgs_ascii_rejects_nan(tmp_path: Path) -> None:
    path = write_lines(tmp_path / "nan.txt", ["1\t2\t3", "nan\t5\t6"])
    with pytest.raises(NonFiniteSampleError):
        read_usgs(path)


def test_usgs_ascii_rejects_non_numeric_text(tmp_path: Path) -> None:
    path = write_lines(tmp_path / "text.txt", ["vertical\tnorth\teast", "1\t2\t3"])
    with pytest.raises(MalformedAsciiError):
        read_usgs(path)


def test_usgs_ascii_rejects_empty_file(tmp_path: Path) -> None:
    path = write_lines(tmp_path / "empty.txt", [])
    with pytest.raises(MalformedAsciiError):
        read_usgs(path)


def test_usgs_ascii_requires_a_documented_sampling_rate(tmp_path: Path) -> None:
    path = usgs_file(tmp_path, [(1.0, 2.0, 3.0)])
    with pytest.raises(MetadataConflictError):
        read_usgs(path, declared_sampling_rate_hz=None)


def test_duration_follows_from_sample_count(tmp_path: Path) -> None:
    path = usgs_file(tmp_path, [(float(i), 0.0, 0.0) for i in range(400)])
    record = read_usgs(path)

    assert record.n_samples == 400
    assert record.duration_s == pytest.approx(2.0)


def test_samples_are_read_only(tmp_path: Path) -> None:
    record = read_usgs(usgs_file(tmp_path, [(1.0, 2.0, 3.0)]))
    with pytest.raises(ValueError):
        record.z[0] = 99.0


def test_parser_does_not_write_intermediate_files(tmp_path: Path) -> None:
    path = usgs_file(tmp_path, [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)])
    before = {p.name: p.stat().st_mtime_ns for p in tmp_path.rglob("*")}

    read_usgs(path)

    assert {p.name: p.stat().st_mtime_ns for p in tmp_path.rglob("*")} == before


def test_reading_is_deterministic(tmp_path: Path) -> None:
    path = usgs_file(tmp_path, [(1.5, -2.5, 3.5), (4.0, 5.0, 6.0)])
    first, second = read_usgs(path), read_usgs(path)

    assert np.array_equal(first.z, second.z)
    assert first.recording_id == second.recording_id


def test_unknown_format_is_rejected() -> None:
    with pytest.raises(UnsupportedFormatError):
        get_reader("telepathy_3c")
    assert "usgs_ascii_3c" in available_formats()


def test_truncated_final_row_is_reported_as_ragged_not_as_nan(tmp_path: Path) -> None:
    """A file cut mid-sample must not be reported as corrupt samples."""
    path = write_lines(tmp_path / "cut.txt", ["1\t2\t3", "4\t5\t6", "706"])

    with pytest.raises(MalformedAsciiError) as excinfo:
        read_usgs(path)

    message = str(excinfo.value)
    assert "line 3 has 1 columns" in message
    assert "drop_incomplete_final_row" in message


def test_truncated_final_row_is_dropped_only_when_configured(tmp_path: Path) -> None:
    path = write_lines(tmp_path / "cut.txt", ["1\t2\t3", "4\t5\t6", "706"])
    record = read_usgs(path, drop_incomplete_final_row=True)

    assert record.n_samples == 2
    assert np.array_equal(record.z, [1.0, 4.0])


def test_the_drop_option_does_not_hide_a_ragged_row_in_the_middle(tmp_path: Path) -> None:
    path = write_lines(tmp_path / "middle.txt", ["1\t2\t3", "706", "4\t5\t6"])

    with pytest.raises(MalformedAsciiError) as excinfo:
        read_usgs(path, drop_incomplete_final_row=True)
    assert "line 2" in str(excinfo.value)


def test_the_drop_option_does_not_hide_a_genuine_nan(tmp_path: Path) -> None:
    path = write_lines(tmp_path / "nan.txt", ["1\t2\t3", "nan\t5\t6"])

    with pytest.raises(NonFiniteSampleError):
        read_usgs(path, drop_incomplete_final_row=True)
