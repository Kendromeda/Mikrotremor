"""Baseline estimators for the vertical slice.

Only two, both deliberately dull. With three independent sites, a model with
capacity to spare would fit the fold rather than the physics, and the resulting
number would say nothing. Random forests, boosting and neural networks are left
out until the corpus can support them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["MedianVs30", "RidgeLogVs30", "load_model", "save_model"]


class MedianVs30:
    """Predict the training median, ignoring the features entirely.

    This is the bar any real model has to clear. A model that cannot beat it is
    reading noise.
    """

    name = "median_vs30"

    def __init__(self) -> None:
        self.value_: float | None = None

    def fit(
        self, features: np.ndarray, target: np.ndarray, weight: np.ndarray | None = None
    ) -> MedianVs30:
        del features
        values = np.asarray(target, dtype=np.float64)
        if weight is None:
            self.value_ = float(np.median(values))
        else:
            self.value_ = float(_weighted_median(values, np.asarray(weight, dtype=np.float64)))
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.value_ is None:
            raise RuntimeError("model is not fitted")
        return np.full(len(features), self.value_, dtype=np.float64)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights)
    cutoff = cumulative[-1] / 2.0
    return float(values[np.searchsorted(cumulative, cutoff)])


class RidgeLogVs30:
    """Ridge regression on log10 amplitudes predicting ln(Vs30).

    Vs30 spans roughly an order of magnitude and is close to log-normal, so the
    target is modelled in log space and exponentiated back. The scaler lives
    inside the estimator so it can only ever be fitted on the training fold.
    """

    name = "ridge_log_vs30"

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.pipeline_: Any | None = None

    def fit(
        self, features: np.ndarray, target: np.ndarray, weight: np.ndarray | None = None
    ) -> RidgeLogVs30:
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        self.pipeline_ = Pipeline(
            [("scale", StandardScaler()), ("ridge", Ridge(alpha=self.alpha, random_state=None))]
        )
        fit_kwargs = {} if weight is None else {"ridge__sample_weight": weight}
        self.pipeline_.fit(
            _transform(features), np.log(np.asarray(target, dtype=np.float64)), **fit_kwargs
        )
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.pipeline_ is None:
            raise RuntimeError("model is not fitted")
        predicted: np.ndarray = np.exp(self.pipeline_.predict(_transform(features)))
        return predicted


def _transform(features: np.ndarray) -> np.ndarray:
    """H/V amplitudes are ratios, so they are modelled on a log axis too."""
    values = np.asarray(features, dtype=np.float64)
    if (values <= 0).any():
        raise ValueError("H/V amplitudes must be strictly positive")
    return np.log10(values)


def save_model(model: Any, path: Path | str) -> Path:
    import joblib

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, target)
    return target


def load_model(path: Path | str) -> Any:
    import joblib

    return joblib.load(Path(path))
