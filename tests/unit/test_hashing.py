"""Phase 1 determinism tests: identifiers and checksums."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from mhvsr_vs30.hashing import (
    make_asset_id,
    make_label_id,
    make_recording_id,
    make_site_id,
    sha256_file,
    stable_id,
)


def test_same_input_produces_same_hash() -> None:
    assert stable_id("a", "b", "c") == stable_id("a", "b", "c")


def test_different_input_produces_different_hash() -> None:
    assert stable_id("a", "b") != stable_id("b", "a")


def test_field_boundaries_are_unambiguous() -> None:
    """Concatenation must not let a separator inside one field forge another."""
    assert stable_id("a|b", "c") != stable_id("a", "b|c")
    assert stable_id("ab", "c") != stable_id("a", "bc")


def test_stable_id_is_lowercase_hex_sha256() -> None:
    value = stable_id("x")
    assert len(value) == 64
    assert value == value.lower()
    int(value, 16)


@given(st.lists(st.text(), min_size=1, max_size=5))
def test_stable_id_is_pure(parts: list[str]) -> None:
    assert stable_id(*parts) == stable_id(*parts)


def test_deterministic_asset_id() -> None:
    first = make_asset_id("usgs_arra_2013", "CI.DRE/a.txt")
    second = make_asset_id("usgs_arra_2013", "CI.DRE/a.txt")
    assert first == second
    assert first != make_asset_id("usgs_arra_2013", "CI.CCC/a.txt")
    assert first != make_asset_id("other_source", "CI.DRE/a.txt")


def test_asset_id_is_independent_of_path_separator_style() -> None:
    assert make_asset_id("s", r"a\b.txt") == make_asset_id("s", "a/b.txt")


def test_site_id_is_stable_and_human_readable() -> None:
    assert make_site_id("usgs_arra_2013", "CI.DRE") == "usgs_arra_2013:CI.DRE"
    assert make_site_id("usgs_arra_2013", " ci.dre ") == "usgs_arra_2013:CI.DRE"


def test_site_id_rejects_empty_code() -> None:
    with pytest.raises(ValueError):
        make_site_id("usgs_arra_2013", "   ")


def test_recording_id_separates_segments_of_one_file() -> None:
    base = ("usgs_arra_2013", "usgs_arra_2013:CI.DRE", "CI.DRE/a.txt")
    assert make_recording_id(*base) == make_recording_id(*base)
    assert make_recording_id(*base) != make_recording_id(*base, segment="1")


def test_label_id_changes_with_value_and_provenance() -> None:
    base = dict(
        site_id="usgs_arra_2013:CI.DRE",
        label_method="surface_wave_joint_inversion",
        reference="USGS OFR 2013-1102",
        vs30_mps=196.0,
    )
    assert make_label_id(**base) == make_label_id(**base)
    assert make_label_id(**{**base, "vs30_mps": 197.0}) != make_label_id(**base)
    assert make_label_id(**{**base, "reference": "other"}) != make_label_id(**base)


def test_label_id_ignores_insignificant_float_spelling() -> None:
    base = dict(
        site_id="s:1",
        label_method="m",
        reference="r",
    )
    assert make_label_id(**base, vs30_mps=196) == make_label_id(**base, vs30_mps=196.0)


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    payload = b"three component counts\n" * 1000
    target = tmp_path / "sample.txt"
    target.write_bytes(payload)
    assert sha256_file(target) == hashlib.sha256(payload).hexdigest()


def test_sha256_file_handles_empty_file(tmp_path: Path) -> None:
    target = tmp_path / "empty.txt"
    target.write_bytes(b"")
    assert sha256_file(target) == hashlib.sha256(b"").hexdigest()


def test_sha256_file_is_chunk_size_independent(tmp_path: Path) -> None:
    payload = bytes(range(256)) * 500
    target = tmp_path / "blob.bin"
    target.write_bytes(payload)
    assert sha256_file(target, chunk_size=7) == sha256_file(target, chunk_size=1 << 20)
