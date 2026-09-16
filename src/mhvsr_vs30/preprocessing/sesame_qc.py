"""SESAME 2004 reliability and clarity, recorded rather than enforced.

A curve that fails clarity is still a real measurement. Filtering on these
criteria here would quietly bias the corpus toward sites with sharp, isolated
peaks, so the results travel with the curve and the decision is left to whoever
builds a dataset from it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mhvsr_vs30.preprocessing.contracts import SesameQc

__all__ = ["evaluate_sesame"]


def evaluate_sesame(hvsr: Any, window_length_s: float, distribution: str) -> SesameQc:
    from hvsrpy import sesame

    frequency = np.asarray(hvsr.frequency, dtype=np.float64)
    mean_curve = np.asarray(hvsr.mean_curve(distribution=distribution), dtype=np.float64)
    std_curve = np.asarray(hvsr.std_curve(distribution=distribution), dtype=np.float64)
    passing = int(np.count_nonzero(np.asarray(hvsr.valid_window_boolean_mask, dtype=bool)))

    reliability: bool | None = None
    clarity: bool | None = None
    reliability_detail: dict[str, Any] = {}
    clarity_detail: dict[str, Any] = {}

    try:
        result = sesame.reliability(
            windowlength=float(window_length_s),
            passing_window_count=passing,
            frequency=frequency,
            mean_curve=mean_curve,
            std_curve=std_curve,
            verbose=0,
        )
        reliability = bool(np.all(np.asarray(result, dtype=bool)))
        reliability_detail = {"criteria": np.asarray(result, dtype=bool).tolist()}
    except Exception as error:  # SESAME needs a resolvable peak; absence is a result
        reliability_detail = {"error": f"{type(error).__name__}: {error}"}

    try:
        result = sesame.clarity(
            frequency=frequency,
            mean_curve=mean_curve,
            std_curve=std_curve,
            fn_std=float(hvsr.std_fn_frequency(distribution="normal")),
            verbose=0,
        )
        clarity = bool(np.all(np.asarray(result, dtype=bool)))
        clarity_detail = {"criteria": np.asarray(result, dtype=bool).tolist()}
    except Exception as error:
        clarity_detail = {"error": f"{type(error).__name__}: {error}"}

    return SesameQc(
        reliability_passed=reliability,
        clarity_passed=clarity,
        reliability_detail=reliability_detail,
        clarity_detail=clarity_detail,
    )
