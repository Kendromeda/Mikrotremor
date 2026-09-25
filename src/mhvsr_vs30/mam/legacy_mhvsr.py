"""Read the published single-input mHVSR ANN for inference without training it.

Only the known Dense/BatchNormalization/ReLU/Dropout architecture is supported.
The saved .keras archive is read as data; no serialized Python object is executed.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import h5py  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d  # type: ignore[import-untyped]


def resample_hvsr_features(
    frequency_hz: NDArray[np.float64], amplitude: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float32]]:
    """Match high_dim_models.ipynb: 35 linear-amplitude samples, then ln."""
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    curve = np.asarray(amplitude, dtype=np.float64)
    if frequency.ndim != 1 or curve.shape != frequency.shape or len(frequency) < 2:
        raise ValueError(
            "HVSR frequency and amplitude arrays must be matching one-dimensional arrays"
        )
    if (
        not np.isfinite(frequency).all()
        or not np.isfinite(curve).all()
        or np.any(curve <= 0)
        or np.any(np.diff(frequency) <= 0)
    ):
        raise ValueError("HVSR frequencies must rise and amplitudes must be positive and finite")
    mask = (frequency >= 0.3) & (frequency <= 50.0)
    # The original notebook trims to this band, then linearly extrapolates at
    # its endpoints. Reject a wider gap instead of silently extrapolating it.
    if mask.sum() < 2 or frequency[mask][0] > 0.31 or frequency[mask][-1] < 49.9:
        raise ValueError("HVSR does not cover the model's 0.3-50 Hz frequency band")
    target = np.logspace(np.log10(0.3), np.log10(50.0), 35)
    interpolated = interp1d(
        frequency[mask], curve[mask], bounds_error=False, fill_value="extrapolate"
    )(target)
    return target, np.log(interpolated.astype(np.float32))


def predict_single_input_mhvsr(model_path: Path | str, log_features: NDArray[np.float32]) -> float:
    """Evaluate the saved Keras 3 network using its float32 inference equations."""
    features = np.asarray(log_features, dtype=np.float32)
    if features.shape != (35,) or not np.isfinite(features).all():
        raise ValueError("single-input mHVSR model requires 35 finite log amplitudes")
    with zipfile.ZipFile(model_path) as archive:
        config = json.loads(archive.read("config.json"))
        if config.get("class_name") != "Sequential":
            raise ValueError("unsupported model architecture")
        layers = config["config"]["layers"]
        expected = ["InputLayer"] + [
            item
            for size in (128, 256, 128)
            for item in ("Dense", "BatchNormalization", "ReLU", "Dropout")
        ] + ["Dense", "BatchNormalization", "ReLU", "Dense"]
        # The last three layers are Dense(64), BatchNorm, ReLU, then Dense(1).
        if [layer["class_name"] for layer in layers] != expected:
            raise ValueError("unsupported model layer sequence")
        with h5py.File(io.BytesIO(archive.read("model.weights.h5")), "r") as weights:
            x: NDArray[np.float32] = features.reshape(1, 35)
            dense_index = 0
            norm_index = 0
            for layer in layers[1:]:
                kind = layer["class_name"]
                if kind == "Dense":
                    group = "layers/dense" + (f"_{dense_index}" if dense_index else "") + "/vars/"
                    kernel = np.asarray(weights[group + "0"], dtype=np.float32)
                    bias = np.asarray(weights[group + "1"], dtype=np.float32)
                    if layer["config"]["activation"] != "linear":
                        raise ValueError("unsupported Dense activation")
                    x = np.asarray(x @ kernel + bias, dtype=np.float32)
                    dense_index += 1
                elif kind == "BatchNormalization":
                    group = "layers/batch_normalization" + (
                        f"_{norm_index}" if norm_index else ""
                    ) + "/vars/"
                    gamma, beta, mean, variance = (
                        np.asarray(weights[group + str(i)], dtype=np.float32) for i in range(4)
                    )
                    epsilon = np.float32(layer["config"]["epsilon"])
                    x = np.asarray(
                        (x - mean) / np.sqrt(variance + epsilon) * gamma + beta,
                        dtype=np.float32,
                    )
                    norm_index += 1
                elif kind == "ReLU":
                    x = np.asarray(np.maximum(x, 0), dtype=np.float32)
                elif kind == "Dropout":
                    continue  # Disabled during inference.
                else:
                    raise ValueError(f"unsupported layer: {kind}")
    if x.shape != (1, 1) or not np.isfinite(x).all():
        raise RuntimeError("model inference did not return one finite log Vs30")
    return float(np.exp(x[0, 0]))
