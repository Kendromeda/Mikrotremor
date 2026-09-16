"""Phase 3.5: the runtime this project actually needs must import."""

from __future__ import annotations


def test_hvsrpy_imports_and_is_the_pinned_version() -> None:
    """hvsrpy imports matplotlib and IPython eagerly, so import is a real gate.

    The version is pinned because 2.1.0 changed window rejection: mixing its
    output with 2.0.0 in one snapshot would make curves incomparable.
    """
    import hvsrpy

    assert hvsrpy.__version__ == "2.0.0"


def test_obspy_imports() -> None:
    import obspy

    assert obspy.__version__
