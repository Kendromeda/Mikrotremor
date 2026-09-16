"""Regenerate the golden fixtures.

Run deliberately, never automatically: a golden that regenerates itself cannot
catch a regression. Rewriting these files is a decision that belongs in a commit
message explaining why the numbers changed.

    uv run python tests/fixtures/make_goldens.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mhvsr_vs30.preprocessing.config import load_profile
from mhvsr_vs30.preprocessing.pipeline import process_recording

FIXTURES = Path(__file__).parent
NOTEBOOK_RECORDING = Path("data/UT.STN09_20130320_020000.mseed")


def synthetic_record():
    from mhvsr_vs30.io.base import ThreeComponentRecord

    rng = np.random.default_rng(20260916)
    n = 200 * 1200
    return ThreeComponentRecord(
        z=rng.normal(size=n),
        north=rng.normal(size=n),
        east=rng.normal(size=n),
        sampling_rate_hz=200.0,
        start_time=None,
        units="counts",
        source_channels={"Z": "column_1", "N": "column_2", "E": "column_3"},
        recording_id="g" * 64,
    )


def notebook_record():
    from datetime import UTC

    from obspy import read

    from mhvsr_vs30.io.base import ThreeComponentRecord

    stream = read(str(NOTEBOOK_RECORDING))
    by_channel = {trace.stats.channel[-1]: trace for trace in stream}
    start = max(t.stats.starttime for t in stream)
    end = min(t.stats.endtime for t in stream)
    trimmed = {k: v.copy().trim(start, end) for k, v in by_channel.items()}
    length = min(int(t.stats.npts) for t in trimmed.values())

    return ThreeComponentRecord(
        z=np.asarray(trimmed["Z"].data[:length], dtype=np.float64),
        north=np.asarray(trimmed["N"].data[:length], dtype=np.float64),
        east=np.asarray(trimmed["E"].data[:length], dtype=np.float64),
        sampling_rate_hz=float(trimmed["Z"].stats.sampling_rate),
        start_time=start.datetime.replace(tzinfo=UTC),
        units="counts",
        source_channels={"Z": "BHZ", "N": "BHN", "E": "BHE"},
        recording_id="n" * 64,
    )


def save(name: str, curve) -> None:
    np.savez_compressed(
        FIXTURES / f"golden_{name}.npz",
        frequency_hz=curve.frequency_hz,
        mean_curve=curve.mean_curve,
        model_amplitude=(
            curve.model_amplitude
            if curve.model_amplitude is not None
            else np.array([], dtype=np.float64)
        ),
        window_count=np.array([curve.window_count]),
        accepted_window_count=np.array([curve.accepted_window_count]),
    )
    print(f"  {name}: {curve.window_count} windows, {len(curve.frequency_hz)} frequencies")


def main() -> None:
    save(
        "training_v1",
        process_recording(
            synthetic_record(), load_profile("configs/preprocessing/training_v1.yaml")
        ),
    )
    if NOTEBOOK_RECORDING.is_file():
        save(
            "published_compat_v1",
            process_recording(
                notebook_record(),
                load_profile("configs/preprocessing/published_compat_v1.yaml"),
            ),
        )


if __name__ == "__main__":
    main()
