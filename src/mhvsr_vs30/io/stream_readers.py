"""Readers backed by ObsPy: MiniSEED, SAC bundles, GCF bundles and SEG-2.

Format detection is by content, not by extension, because this corpus contains
.sac-hd files that ObsPy identifies as SAC and .dat files that are SEG-2 in one
folder and HP SDF in the next.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import UTC
from pathlib import Path
from typing import Any

import numpy as np

from mhvsr_vs30.exceptions import (
    AmbiguousChannelError,
    MetadataConflictError,
    MissingComponentError,
    NoCommonTimeWindowError,
    RecordingReadError,
    SamplingRateMismatchError,
)
from mhvsr_vs30.io.base import COMPONENT_CODES, ReadRequest, SurveyInspection, ThreeComponentRecord

__all__ = ["GcfBundleReader", "MiniSeedReader", "SacBundleReader", "Seg2Reader"]

_ORIENTATION_SUFFIX = {"Z": "Z", "N": "N", "E": "E"}


def _read_stream(path: Path) -> Any:
    from obspy import read as obspy_read

    try:
        return obspy_read(str(path))
    except Exception as error:  # ObsPy raises a wide range of format errors
        raise RecordingReadError(
            f"{path.name}: ObsPy could not read this file ({error})"
        ) from error


def _merged(stream: Any) -> Any:
    """Join gapped segments of the same channel, refusing to fill the gaps."""
    stream = stream.copy()
    stream.merge(method=0)
    return stream


def _map_by_declared_codes(traces: Sequence[Any], request: ReadRequest) -> dict[str, Any]:
    """Use the channel codes the source config states, and only those."""
    assert request.channel_codes is not None
    by_code: dict[str, list[Any]] = defaultdict(list)
    for trace in traces:
        by_code[str(trace.stats.channel)].append(trace)

    resolved: dict[str, Any] = {}
    for component in COMPONENT_CODES:
        wanted = str(request.channel_codes.get(component))
        matched = by_code.get(wanted, [])
        if len(matched) != 1:
            raise MissingComponentError(
                f"{request.asset.relative_path}: configured channel {wanted} for "
                f"{component} matched {len(matched)} traces"
            )
        resolved[component] = matched[0]
    return resolved


def _map_by_orientation_suffix(traces: Sequence[Any], request: ReadRequest) -> dict[str, Any]:
    """Read orientation from the last character of the channel code.

    Codes ending in 1/2/3 carry no orientation anybody documented here, so a
    file that uses them is refused rather than assigned an arbitrary azimuth.
    """
    by_component: dict[str, list[Any]] = defaultdict(list)
    unmapped: list[str] = []

    for trace in traces:
        code = str(trace.stats.channel)
        component = _ORIENTATION_SUFFIX.get(code[-1:].upper()) if code else None
        if component is None:
            unmapped.append(code or "<unnamed>")
            continue
        by_component[component].append(trace)

    duplicated = sorted(name for name, found in by_component.items() if len(found) > 1)
    if duplicated:
        codes = sorted(str(t.stats.channel) for name in duplicated for t in by_component[name])
        raise AmbiguousChannelError(
            f"{request.asset.relative_path}: components {duplicated} match several channels "
            f"{codes}; declare channel_codes to resolve this"
        )

    missing = [name for name in COMPONENT_CODES if name not in by_component]
    if missing and unmapped:
        raise AmbiguousChannelError(
            f"{request.asset.relative_path}: no orientation for channels {sorted(unmapped)} "
            f"and components {missing} are unresolved; declare channel_codes"
        )
    if missing:
        raise MissingComponentError(
            f"{request.asset.relative_path}: components {missing} are absent"
        )
    return {name: found[0] for name, found in by_component.items()}


def _map_components(traces: Sequence[Any], request: ReadRequest) -> dict[str, Any]:
    """Resolve traces to Z/N/E, refusing to guess an orientation."""
    if request.channel_codes:
        return _map_by_declared_codes(traces, request)
    return _map_by_orientation_suffix(traces, request)


def _common_rate(components: dict[str, Any], request: ReadRequest) -> float:
    rates = {name: float(trace.stats.sampling_rate) for name, trace in components.items()}
    if len(set(rates.values())) != 1:
        raise SamplingRateMismatchError(
            f"{request.asset.relative_path}: components disagree on sampling rate {rates}"
        )
    rate = next(iter(rates.values()))

    declared = request.declared_sampling_rate_hz
    if declared is not None and not np.isclose(rate, declared):
        raise MetadataConflictError(
            f"{request.asset.relative_path}: headers say {rate} Hz but the source "
            f"documents {declared} Hz"
        )
    return rate


def _trim_to_common_window(
    components: dict[str, Any], request: ReadRequest
) -> tuple[dict[str, Any], int]:
    """Cut every component to the interval all three share.

    Trimming happens in memory and never pads: a padded sample is invented data.
    Rounding can leave the trimmed traces one sample apart, so the caller also
    receives the length they all reach.
    """
    starts = [trace.stats.starttime for trace in components.values()]
    ends = [trace.stats.endtime for trace in components.values()]
    start, end = max(starts), min(ends)
    if start > end:
        raise NoCommonTimeWindowError(
            f"{request.asset.relative_path}: components never overlap "
            f"(latest start {start}, earliest end {end})"
        )

    trimmed = {
        name: trace.copy().trim(start, end, nearest_sample=True)
        for name, trace in components.items()
    }
    shortest = min(int(trace.stats.npts) for trace in trimmed.values())
    if shortest == 0:
        raise NoCommonTimeWindowError(
            f"{request.asset.relative_path}: the common window holds no samples"
        )
    return trimmed, shortest


def _to_record(components: dict[str, Any], request: ReadRequest) -> ThreeComponentRecord:
    rate = _common_rate(components, request)
    trimmed, length = _trim_to_common_window(components, request)
    start_time = max(trace.stats.starttime for trace in trimmed.values()).datetime

    return ThreeComponentRecord(
        z=np.asarray(trimmed["Z"].data[:length], dtype=np.float64),
        north=np.asarray(trimmed["N"].data[:length], dtype=np.float64),
        east=np.asarray(trimmed["E"].data[:length], dtype=np.float64),
        sampling_rate_hz=rate,
        start_time=start_time.replace(tzinfo=start_time.tzinfo or UTC),
        units=request.units or "unknown",
        source_channels={name: str(trace.stats.channel) for name, trace in trimmed.items()},
        recording_id=request.recording.recording_id,
    )


def _bundle_traces(paths: Iterable[Path], request: ReadRequest) -> list[Any]:
    traces: list[Any] = []
    for path in paths:
        stream = _merged(_read_stream(path))
        if len(stream) != 1:
            raise MetadataConflictError(
                f"{path.name}: expected one continuous component per file, found {len(stream)}"
            )
        traces.append(stream[0])
    return traces


class MiniSeedReader:
    """One file holding all three components."""

    format_name = "miniseed"

    def read(self, request: ReadRequest) -> ThreeComponentRecord:
        stream = _merged(_read_stream(request.path))
        return _to_record(_map_components(list(stream), request), request)


class SacBundleReader:
    """One component per file, grouped by the manifest, not by file name."""

    format_name = "sac_bundle"

    def read(self, request: ReadRequest) -> ThreeComponentRecord:
        paths = [Path(p) for p in (request.siblings or (request.path,))]
        return _to_record(_map_components(_bundle_traces(paths, request), request), request)


class GcfBundleReader(SacBundleReader):
    """Guralp GCF, which also stores one component per file and often gaps them."""

    format_name = "gcf"


class Seg2Reader:
    """Inspects SEG-2 rather than assuming it is an ambient-noise recording.

    SEG-2 in this corpus is usually a MASW or refraction shot gather. Even a
    three-trace file carries no orientation, so it becomes a record only when
    the source config states which trace is Z, N and E.
    """

    format_name = "seg2"

    def read(self, request: ReadRequest) -> ThreeComponentRecord | SurveyInspection:
        stream = _read_stream(request.path)
        traces = list(stream)
        detected = str(traces[0].stats._format) if traces else "unknown"
        rates = {float(trace.stats.sampling_rate) for trace in traces}
        codes = tuple(str(trace.stats.channel) for trace in traces)

        if request.component_traces:
            return self._to_record(traces, request)

        reason = (
            "three traces with no orientation in their headers; set component_traces "
            "in the source config once documentation states which trace is Z, N and E"
            if len(traces) == 3
            else f"{len(traces)} traces, which is a multichannel survey rather than a station"
        )
        return SurveyInspection(
            recording_id=request.recording.recording_id,
            detected_format=detected,
            trace_count=len(traces),
            sampling_rate_hz=next(iter(rates)) if len(rates) == 1 else float("nan"),
            samples_per_trace=int(traces[0].stats.npts) if traces else 0,
            channel_codes=codes,
            is_three_component_candidate=len(traces) == 3,
            reason=reason,
        )

    @staticmethod
    def _to_record(traces: list[Any], request: ReadRequest) -> ThreeComponentRecord:
        assert request.component_traces is not None
        selected: dict[str, Any] = {}
        for component in COMPONENT_CODES:
            index = request.component_traces.get(component)
            if index is None or not 0 <= index < len(traces):
                raise MissingComponentError(
                    f"{request.asset.relative_path}: component_traces has no usable "
                    f"trace index for {component}"
                )
            trace = traces[index].copy()
            trace.stats.channel = f"trace_{index}"
            selected[component] = trace
        return _to_record(selected, request)
