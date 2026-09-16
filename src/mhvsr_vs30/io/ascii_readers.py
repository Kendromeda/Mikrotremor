"""Readers for text formats: raw three-column ASCII and processed H/V curves."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mhvsr_vs30.exceptions import (
    MalformedAsciiError,
    MetadataConflictError,
    NonFiniteSampleError,
)
from mhvsr_vs30.io.base import HvsrCurve, ReadRequest, ThreeComponentRecord

__all__ = ["ProcessedHvsrReader", "UsgsAsciiReader"]

_COMPONENT_BY_NAME = {"vertical": "Z", "north": "N", "east": "E"}


class UsgsAsciiReader:
    """Three whitespace-separated numeric columns, one sample per row.

    Column meaning comes from the documented component order, never from the
    data: picking the vertical channel by variance would be a guess.
    """

    format_name = "usgs_ascii_3c"

    def read(self, request: ReadRequest) -> ThreeComponentRecord:
        rate = request.declared_sampling_rate_hz
        if rate is None:
            raise MetadataConflictError(
                f"{request.asset.relative_path}: this format carries no sampling rate, "
                "so the source configuration must declare one"
            )

        order = tuple(request.component_order)
        if sorted(order) != sorted(_COMPONENT_BY_NAME):
            raise MetadataConflictError(f"component_order must name Z, N and E, got {order}")

        try:
            frame = pd.read_csv(request.path, sep=r"\s+", header=None, dtype="float64", engine="c")
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as error:
            raise MalformedAsciiError(f"{request.asset.relative_path}: {error}") from error
        except ValueError as error:
            raise MalformedAsciiError(
                f"{request.asset.relative_path}: expected only numeric columns ({error})"
            ) from error

        if frame.shape[1] != 3:
            raise MalformedAsciiError(
                f"{request.asset.relative_path}: expected 3 columns, found {frame.shape[1]}"
            )
        if frame.empty:
            raise MalformedAsciiError(f"{request.asset.relative_path}: file holds no samples")

        values = frame.to_numpy(dtype=np.float64, copy=False)
        if not np.isfinite(values).all():
            values = self._resolve_non_finite(values, request)

        column_of = {name: position for position, name in enumerate(order)}
        return ThreeComponentRecord(
            z=values[:, column_of["vertical"]],
            north=values[:, column_of["north"]],
            east=values[:, column_of["east"]],
            sampling_rate_hz=float(rate),
            start_time=request.recording.start_time,
            units=request.units or "unknown",
            source_channels={
                _COMPONENT_BY_NAME[name]: f"column_{column_of[name] + 1}" for name in order
            },
            recording_id=request.recording.recording_id,
        )

    @staticmethod
    def _resolve_non_finite(values: np.ndarray, request: ReadRequest) -> np.ndarray:
        """Separate a truncated row from a genuinely non-finite sample.

        Pandas pads a short row with NaN, so a file cut mid-sample looks exactly
        like corrupt data. They call for different answers, and reporting one as
        the other would send a reviewer looking in the wrong place.
        """
        ragged = UsgsAsciiReader._ragged_lines(request.path, expected=values.shape[1])
        if not ragged:
            raise NonFiniteSampleError(
                f"{request.asset.relative_path}: samples contain NaN or infinity"
            )

        line_number, found = ragged[0]
        truncated_tail = len(ragged) == 1 and line_number == len(values)
        if truncated_tail and request.drop_incomplete_final_row:
            trimmed = values[:-1]
            if not np.isfinite(trimmed).all():
                raise NonFiniteSampleError(
                    f"{request.asset.relative_path}: samples contain NaN or infinity"
                )
            return trimmed

        detail = (
            "the recording was cut mid-sample; set drop_incomplete_final_row in the "
            "source config to discard it"
            if truncated_tail
            else f"{len(ragged)} rows have the wrong width"
        )
        raise MalformedAsciiError(
            f"{request.asset.relative_path}: line {line_number} has {found} columns "
            f"instead of {values.shape[1]} - {detail}"
        )

    @staticmethod
    def _ragged_lines(path: Path, expected: int, limit: int = 5) -> list[tuple[int, int]]:
        """Find lines whose field count differs from the rest of the file."""
        ragged: list[tuple[int, int]] = []
        with open(path, encoding="latin-1") as handle:
            for number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                found = len(stripped.split())
                if found != expected:
                    ragged.append((number, found))
                    if len(ragged) >= limit:
                        break
        return ragged


_CURVE_ROLES = {"frequency", "mean", "std", "min", "max", "ignore"}


class ProcessedHvsrReader:
    """A published H/V curve, kept apart from raw waveforms on purpose.

    Column roles must be declared by the source config: an .asc from Pescara
    carries frequency/mean/std, while a Geopsy .hv carries frequency/mean/min/max,
    and reading one as the other would invent a standard deviation.
    """

    format_name = "processed_hvsr"

    def read(self, request: ReadRequest) -> HvsrCurve:
        columns = tuple(request.curve_columns)
        if not columns:
            raise MetadataConflictError(
                f"{request.asset.relative_path}: curve_columns must be declared before "
                "a processed curve can be read"
            )
        unknown = sorted(set(columns) - _CURVE_ROLES)
        if unknown:
            raise MetadataConflictError(f"unknown curve column roles: {unknown}")
        for required in ("frequency", "mean"):
            if columns.count(required) != 1:
                raise MetadataConflictError(
                    f"curve_columns must name {required} exactly once, got {columns}"
                )

        rows = self._parse_rows(request, len(columns))
        table = np.asarray(rows, dtype=np.float64)
        position = {role: index for index, role in enumerate(columns)}

        std = table[:, position["std"]].copy() if "std" in position else None
        return HvsrCurve(
            frequency_hz=table[:, position["frequency"]].copy(),
            mean_amplitude=table[:, position["mean"]].copy(),
            std_amplitude=std,
            recording_id=request.recording.recording_id,
        )

    @staticmethod
    def _parse_rows(request: ReadRequest, width: int) -> list[list[float]]:
        """Skip comments and one leading text header, then require clean numbers."""
        text = request.path.read_text(encoding="latin-1")
        rows: list[list[float]] = []
        started = False

        for number, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                values = [float(field) for field in stripped.split()]
            except ValueError:
                if started:
                    raise MalformedAsciiError(
                        f"{request.asset.relative_path}: line {number} is not numeric"
                    ) from None
                continue
            if len(values) != width:
                raise MalformedAsciiError(
                    f"{request.asset.relative_path}: line {number} has {len(values)} columns, "
                    f"but curve_columns declares {width}"
                )
            started = True
            rows.append(values)

        if not rows:
            raise MalformedAsciiError(f"{request.asset.relative_path}: no numeric rows found")
        return rows
