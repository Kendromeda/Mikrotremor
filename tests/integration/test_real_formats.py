"""Phase 3 integration: every reader against real bytes from this corpus.

Each case names the file it was verified against, so a future change to a
reader is checked against the same evidence that justified it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mhvsr_vs30.io.base import HvsrCurve, SurveyInspection, ThreeComponentRecord
from mhvsr_vs30.io.registry import get_reader
from tests.reader_helpers import make_request

pytestmark = pytest.mark.requires_real_data

WELLINGTON = Path(
    "datasets_global/source_04_wellington_designsafe/raw/extracted/PRJ-2075/"
    "Anderson Park (ANPK)/Unprocessed Data/"
    "Microtremor Array Measurements (MAM)/ANPK_BigX/UT.STN11.ANPK_BigX.miniseed"
)
SISMI_SAC = Path(
    "datasets_global/source_03_italy_sismi_milan/raw/extracted/data/Array_parcoNord/sac/PN01"
)
SISMI_SAC_HD = Path(
    "datasets_global/source_03_italy_sismi_milan/raw/extracted/data/"
    "Array_Giuriati/Array_Giuriati_big/SAC/MI10"
)
PETOBO_GCF = Path("datasets/source_11_petobo_palu/raw/files/#01Petobo/Test#1")
MANFREDONIA = Path(
    "datasets_global/source_01_italy_pescara_manfredonia/raw/extracted/"
    "Manfredonia/Manfredonia_HV/M001.asc"
)
DEMAK = Path("datasets/source_03_demak/raw/extracted/Microtremor Dataset/HV Curve/1.hv")


def require(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"{path} is not present in this workspace")
    return path


def bundle(directory: Path, pattern: str) -> list[Path]:
    require(directory)
    paths = sorted(directory.glob(pattern))
    if len(paths) < 3:
        pytest.skip(f"{directory} does not hold a three-component bundle")
    return paths[:3]


def test_miniseed_reads_a_wellington_station() -> None:
    path = require(WELLINGTON)
    request = make_request(path, "miniseed", declared_sampling_rate_hz=100.0, units="counts")
    record = get_reader("miniseed").read(request)

    assert isinstance(record, ThreeComponentRecord)
    assert record.sampling_rate_hz == 100.0
    assert record.source_channels == {"Z": "BHZ", "N": "BHN", "E": "BHE"}
    assert record.duration_s == pytest.approx(1800.0, abs=1.0)


def test_sac_bundle_reads_a_sismi_array_station() -> None:
    paths = bundle(SISMI_SAC, "*.sac")
    request = make_request(
        paths[0], "sac_bundle", root=SISMI_SAC, siblings=tuple(paths), units="counts"
    )
    record = get_reader("sac_bundle").read(request)

    assert isinstance(record, ThreeComponentRecord)
    assert record.sampling_rate_hz == 200.0
    assert set(record.source_channels) == {"Z", "N", "E"}
    assert record.n_samples > 1_000_000


def test_sac_hd_is_read_by_content_not_extension() -> None:
    """ObsPy has no .sac-hd entry, so this only works via content detection."""
    paths = bundle(SISMI_SAC_HD, "*.sac-hd")
    request = make_request(
        paths[0], "sac_bundle", root=SISMI_SAC_HD, siblings=tuple(paths), units="counts"
    )
    record = get_reader("sac_bundle").read(request)

    assert isinstance(record, ThreeComponentRecord)
    assert record.sampling_rate_hz == 200.0
    assert np.isfinite(record.z).all()


def test_gcf_reads_a_petobo_station() -> None:
    require(PETOBO_GCF)
    paths = sorted(PETOBO_GCF.glob("st001_*_20181219_0300.gcf"))
    if len(paths) != 3:
        pytest.skip("Petobo station 001 does not hold exactly three GCF components")

    request = make_request(paths[0], "gcf", root=PETOBO_GCF, siblings=tuple(paths), units="counts")
    record = get_reader("gcf").read(request)

    assert isinstance(record, ThreeComponentRecord)
    assert record.sampling_rate_hz == 100.0
    assert record.source_channels == {"Z": "HHZ", "N": "HHN", "E": "HHE"}


def test_processed_asc_curve_carries_a_standard_deviation() -> None:
    path = require(MANFREDONIA)
    request = make_request(path, "processed_hvsr", curve_columns=("frequency", "mean", "std"))
    curve = get_reader("processed_hvsr").read(request)

    assert isinstance(curve, HvsrCurve)
    assert curve.std_amplitude is not None
    assert curve.frequency_hz[0] == pytest.approx(0.125)
    assert curve.input_level == "processed_curve"


def test_processed_geopsy_curve_has_no_standard_deviation() -> None:
    path = require(DEMAK)
    request = make_request(
        path, "processed_hvsr", curve_columns=("frequency", "mean", "min", "max")
    )
    curve = get_reader("processed_hvsr").read(request)

    assert isinstance(curve, HvsrCurve)
    assert curve.std_amplitude is None
    assert curve.frequency_hz[0] == pytest.approx(0.5)


def test_a_processed_curve_is_never_a_three_component_record() -> None:
    path = require(MANFREDONIA)
    request = make_request(path, "processed_hvsr", curve_columns=("frequency", "mean", "std"))
    curve = get_reader("processed_hvsr").read(request)

    assert not isinstance(curve, ThreeComponentRecord)
    assert not isinstance(curve, SurveyInspection)
