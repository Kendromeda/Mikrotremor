"""Reader registry.

A reader is chosen by the raw_format recorded in the manifest, which comes from
the source configuration. Extensions never select a reader.
"""

from __future__ import annotations

from mhvsr_vs30.exceptions import UnsupportedFormatError
from mhvsr_vs30.io.ascii_readers import ProcessedHvsrReader, UsgsAsciiReader
from mhvsr_vs30.io.base import RecordingReader
from mhvsr_vs30.io.stream_readers import (
    GcfBundleReader,
    MiniSeedReader,
    SacBundleReader,
    Seg2Reader,
)

__all__ = ["READERS", "available_formats", "get_reader"]

READERS: dict[str, RecordingReader] = {
    reader.format_name: reader
    for reader in (
        UsgsAsciiReader(),
        MiniSeedReader(),
        SacBundleReader(),
        GcfBundleReader(),
        ProcessedHvsrReader(),
        Seg2Reader(),
    )
}


def available_formats() -> tuple[str, ...]:
    return tuple(sorted(READERS))


def get_reader(format_name: str) -> RecordingReader:
    try:
        return READERS[format_name]
    except KeyError:
        raise UnsupportedFormatError(
            f"no reader registered for format {format_name!r}; known formats are "
            f"{', '.join(available_formats())}"
        ) from None
