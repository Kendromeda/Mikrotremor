"""Automatic window rejection.

The reference notebooks reject windows by hand in a plot. That cannot run
unattended and two analysts will not produce the same answer, so batch runs use
the deterministic frequency-domain method and record a reason per window.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mhvsr_vs30.preprocessing.config import PreprocessingProfile
from mhvsr_vs30.preprocessing.contracts import PreprocessingError, WindowQc

__all__ = ["reject_windows"]


def reject_windows(
    hvsr: Any, profile: PreprocessingProfile
) -> tuple[np.ndarray, tuple[WindowQc, ...]]:
    """Return the accepted-window mask and the reason recorded for each window."""
    settings = profile.rejection
    window_count = int(np.asarray(hvsr.amplitude).shape[0])

    if settings.method == "none":
        mask = np.ones(window_count, dtype=bool)
        qc = tuple(
            WindowQc(index=i, accepted=True, reason="rejection_disabled_for_profile")
            for i in range(window_count)
        )
        _enforce_minimums(mask, profile)
        return mask, qc

    import hvsrpy

    hvsrpy.frequency_domain_window_rejection(
        hvsr,
        n=settings.n_sigma,
        max_iterations=settings.maximum_iterations,
        distribution_fn=profile.distribution,
        distribution_mc=profile.distribution,
    )
    mask = np.asarray(hvsr.valid_window_boolean_mask, dtype=bool).copy()
    qc = tuple(
        WindowQc(
            index=i,
            accepted=bool(keep),
            reason="accepted" if keep else f"frequency_domain_outlier_n{settings.n_sigma:g}",
        )
        for i, keep in enumerate(mask)
    )
    _enforce_minimums(mask, profile)
    return mask, qc


def _enforce_minimums(mask: np.ndarray, profile: PreprocessingProfile) -> None:
    """Refuse a curve built from too few windows, rather than publishing it."""
    settings = profile.rejection
    accepted = int(np.count_nonzero(mask))
    total = int(mask.size)

    if accepted < settings.minimum_accepted_windows:
        raise PreprocessingError(
            "insufficient_accepted_windows",
            f"{accepted} windows survived rejection, below the minimum of "
            f"{settings.minimum_accepted_windows}",
        )
    if total and accepted / total < settings.minimum_accepted_fraction:
        raise PreprocessingError(
            "insufficient_accepted_windows",
            f"only {accepted}/{total} windows survived, below the minimum fraction "
            f"{settings.minimum_accepted_fraction}",
        )
