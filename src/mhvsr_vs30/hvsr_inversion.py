"""Single-site HVSR inversion with a transparent 1-D body-wave forward model.

The forward calculation follows the central physical assumptions of ModelHVSR
(Herak, 2008): vertically incident P and S body waves in horizontal viscoelastic
layers.  Synthetic H/V is the ratio of the S-wave and P-wave surface transfer
functions.  It is deliberately not Rayleigh-wave ellipticity.

Only finite-layer thickness and Vs are optimized.  Vp and density are derived
with the empirical Brocher (2005) polynomials, while Qp and Qs remain fixed.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
PathLike = str | Path


def _readonly_float(values: ArrayLike) -> FloatArray:
    array = np.asarray(values, dtype=np.float64).copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class HVSRCurve:
    """Observed HVSR curve and optional GEOPSY summary metadata."""

    frequency_hz: FloatArray
    amplitude: FloatArray
    minimum: FloatArray | None = None
    maximum: FloatArray | None = None
    source_path: Path | None = None
    number_of_windows: int | None = None
    number_of_windows_for_f0: int | None = None
    f0_from_average_hz: float | None = None
    f0_from_windows_hz: float | None = None
    f0_min_hz: float | None = None
    f0_max_hz: float | None = None
    f0_amplitude: float | None = None

    def __post_init__(self) -> None:
        frequency = _readonly_float(self.frequency_hz)
        amplitude = _readonly_float(self.amplitude)
        if frequency.ndim != 1 or amplitude.ndim != 1 or frequency.shape != amplitude.shape:
            raise ValueError("frequency and amplitude must be one-dimensional with the same shape")
        if not np.all(np.isfinite(frequency)) or np.any(frequency <= 0):
            raise ValueError("frequency values must be positive and finite")
        if not np.all(np.isfinite(amplitude)) or np.any(amplitude <= 0):
            raise ValueError("HVSR amplitude values must be positive and finite")
        if frequency.size < 2:
            raise ValueError("HVSR curve must contain at least two frequency samples")
        if np.any(np.diff(frequency) <= 0):
            raise ValueError("frequency values must be strictly increasing")

        minimum = None if self.minimum is None else _readonly_float(self.minimum)
        maximum = None if self.maximum is None else _readonly_float(self.maximum)
        if (minimum is None) != (maximum is None):
            raise ValueError("minimum and maximum HVSR envelopes must be supplied together")
        if minimum is not None and maximum is not None:
            if minimum.shape != amplitude.shape or maximum.shape != amplitude.shape:
                raise ValueError("HVSR envelope arrays must have the same shape as amplitude")
            if not np.all(np.isfinite(minimum)) or not np.all(np.isfinite(maximum)):
                raise ValueError("HVSR envelope values must be finite")
            if np.any(minimum <= 0) or np.any(minimum > amplitude) or np.any(amplitude > maximum):
                raise ValueError("HVSR envelope must be positive and satisfy Min <= Average <= Max")

        object.__setattr__(self, "frequency_hz", frequency)
        object.__setattr__(self, "amplitude", amplitude)
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)

    @property
    def log_std(self) -> FloatArray | None:
        """Return the source notebook's lognormal envelope diagnostic."""
        if self.minimum is None or self.maximum is None:
            return None
        return _readonly_float(0.5 * (np.log(self.maximum) - np.log(self.minimum)))


def _header_number(text: str, label: str, *, integer: bool = False) -> float | int | None:
    pattern = rf"^#\s*{re.escape(label)}\s*(?:=|\s)\s*([-+0-9.eE]+)\s*$"
    match = re.search(pattern, text, flags=re.MULTILINE | re.IGNORECASE)
    if match is None:
        return None
    return int(match.group(1)) if integer else float(match.group(1))


def _header_integer(text: str, label: str) -> int | None:
    value = _header_number(text, label, integer=True)
    return None if value is None else int(value)


def _f0_window_values(text: str) -> tuple[float | None, float | None, float | None]:
    match = re.search(
        r"^#\s*f0 from windows\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*$",
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    if match is None:
        return None, None, None
    values = tuple(float(value) for value in match.groups())
    return values[0], values[1], values[2]


def parse_geopsy_hv(path: PathLike) -> HVSRCurve:
    """Read a GEOPSY ``.hv`` table with either two or four numeric columns.

    Standard GEOPSY files contain Frequency/Average/Min/Max.  A two-column
    Frequency/Average variant is also accepted.  Metadata lines such as
    ``f0 amplitude`` or ``Peak amplitude`` are never interpreted as curve rows.
    """
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"GEOPSY .hv file was not found: {source}")
    text = source.read_text(encoding="utf-8-sig")
    rows: list[tuple[float, ...]] = []
    column_count: int | None = None
    pending_metadata_value = False
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lowered = stripped.lower()
        if lowered in {"f0 amplitude", "peak amplitude"}:
            pending_metadata_value = True
            continue
        parts = stripped.split()
        if pending_metadata_value and len(parts) == 1:
            try:
                float(parts[0])
            except ValueError:
                pass
            else:
                pending_metadata_value = False
                continue
        if len(parts) not in {2, 4}:
            if rows:
                raise ValueError(
                    f"GEOPSY data line {line_number} must contain two or four columns"
                )
            continue
        try:
            row = tuple(float(value) for value in parts)
        except ValueError as exc:
            if rows:
                raise ValueError(
                    f"GEOPSY data line {line_number} contains a non-numeric value"
                ) from exc
            continue
        if column_count is None:
            column_count = len(row)
        if len(row) != column_count:
            raise ValueError(f"GEOPSY data line {line_number} has an inconsistent column count")
        rows.append(row)

    if not rows:
        raise ValueError("GEOPSY .hv f0 amplitude table contains no numeric curve data")
    table = np.asarray(rows, dtype=np.float64)
    frequency = table[:, 0]
    average = table[:, 1]
    minimum = table[:, 2] if table.shape[1] == 4 else None
    maximum = table[:, 3] if table.shape[1] == 4 else None
    f0_windows, f0_min, f0_max = _f0_window_values(text)
    f0_amplitude = _header_number(text, "f0 amplitude")
    if f0_amplitude is None:
        f0_amplitude = _header_number(text, "Peak amplitude")
    return HVSRCurve(
        frequency_hz=frequency,
        amplitude=average,
        minimum=minimum,
        maximum=maximum,
        source_path=source.resolve(),
        number_of_windows=_header_integer(text, "Number of windows"),
        number_of_windows_for_f0=_header_integer(text, "Number of windows for f0"),
        f0_from_average_hz=_header_number(text, "f0 from average"),
        f0_from_windows_hz=f0_windows,
        f0_min_hz=f0_min,
        f0_max_hz=f0_max,
        f0_amplitude=None if f0_amplitude is None else float(f0_amplitude),
    )


def brocher_vp_from_vs(vs_m_s: ArrayLike) -> FloatArray:
    """Derive Vp in m/s from Vs using Brocher's polynomial (km/s internally)."""
    vs = np.asarray(vs_m_s, dtype=np.float64)
    if not np.all(np.isfinite(vs)) or np.any(vs <= 0) or np.any(vs > 4500):
        raise ValueError("Brocher Vs input must be finite and in (0, 4500] m/s")
    x = vs / 1000.0
    vp_km_s = 0.9409 + 2.0947 * x - 0.8206 * x**2 + 0.2683 * x**3 - 0.0251 * x**4
    vp = vp_km_s * 1000.0
    if np.any(vp <= vs):
        raise ValueError("Brocher relation produced Vp <= Vs outside its physical domain")
    return np.asarray(vp, dtype=np.float64)


def brocher_density_from_vp(vp_m_s: ArrayLike) -> FloatArray:
    """Derive density in kg/m3 from Vp with Brocher's polynomial."""
    vp = np.asarray(vp_m_s, dtype=np.float64)
    if not np.all(np.isfinite(vp)) or np.any(vp <= 0):
        raise ValueError("Brocher Vp input must be positive and finite")
    x = vp / 1000.0
    density_g_cm3 = (
        1.6612 * x
        - 0.4721 * x**2
        + 0.0671 * x**3
        - 0.0043 * x**4
        + 0.000106 * x**5
    )
    density = density_g_cm3 * 1000.0
    if np.any(~np.isfinite(density)) or np.any(density <= 0):
        raise ValueError("Brocher relation produced nonphysical density")
    return np.asarray(density, dtype=np.float64)


def _quality_array(value: float | ArrayLike, layer_count: int, label: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = np.full(layer_count, float(array), dtype=np.float64)
    if array.shape != (layer_count,) or not np.all(np.isfinite(array)) or np.any(array <= 0):
        raise ValueError(f"{label} must be positive and scalar or one value per layer")
    return array


def _complex_velocity(
    frequency_hz: FloatArray,
    reference_velocity_m_s: FloatArray,
    quality: FloatArray,
    reference_frequency_hz: float,
) -> NDArray[np.complex128]:
    dispersion = 1.0 + np.log(frequency_hz[:, None] / reference_frequency_hz) / (
        np.pi * quality[None, :]
    )
    if np.any(dispersion <= 0):
        raise ValueError("constant-Q dispersion became nonpositive; revise Q or frequency band")
    return reference_velocity_m_s[None, :] * dispersion * (
        1.0 + 0.5j / quality[None, :]
    )


def _vertical_transfer_function(
    frequency_hz: FloatArray,
    thickness_m: FloatArray,
    velocity_m_s: FloatArray,
    density_kg_m3: FloatArray,
    quality: FloatArray,
    reference_frequency_hz: float,
) -> FloatArray:
    complex_velocity = _complex_velocity(
        frequency_hz, velocity_m_s, quality, reference_frequency_hz
    )
    omega = 2.0 * np.pi * frequency_hz
    total = np.broadcast_to(np.eye(2, dtype=np.complex128), (frequency_hz.size, 2, 2)).copy()
    for layer_index, layer_thickness in enumerate(thickness_m):
        layer_velocity = complex_velocity[:, layer_index]
        impedance = density_kg_m3[layer_index] * layer_velocity
        phase = omega * layer_thickness / layer_velocity
        cosine = np.cos(phase)
        sine = np.sin(phase)
        matrix = np.empty_like(total)
        matrix[:, 0, 0] = cosine
        matrix[:, 0, 1] = sine / (omega * impedance)
        matrix[:, 1, 0] = -omega * impedance * sine
        matrix[:, 1, 1] = cosine
        total = np.einsum("fij,fjk->fik", matrix, total)

    halfspace_impedance = density_kg_m3[-1] * complex_velocity[:, -1]
    denominator = total[:, 0, 0] + total[:, 1, 0] / (1j * omega * halfspace_impedance)
    transfer = np.abs(2.0 / denominator)
    if not np.all(np.isfinite(transfer)) or np.any(transfer <= 0):
        raise FloatingPointError("body-wave transfer function is non-finite or nonpositive")
    return np.asarray(transfer, dtype=np.float64)


def forward_body_wave_hvsr(
    frequency_hz: ArrayLike,
    thickness_m: ArrayLike,
    vs_m_s: ArrayLike,
    *,
    qp: float | ArrayLike = 1.0e9,
    qs: float | ArrayLike = 1.0e9,
    reference_frequency_hz: float = 1.0,
) -> FloatArray:
    """Calculate Herak-style body-wave H/V for finite layers over a halfspace."""
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    thickness = np.asarray(thickness_m, dtype=np.float64)
    vs = np.asarray(vs_m_s, dtype=np.float64)
    if frequency.ndim != 1 or not np.all(np.isfinite(frequency)) or np.any(frequency <= 0):
        raise ValueError("frequency must be a positive finite one-dimensional array")
    if thickness.ndim != 1 or vs.ndim != 1 or thickness.size + 1 != vs.size:
        raise ValueError("thickness must have one fewer value than Vs, which includes a halfspace")
    if not np.all(np.isfinite(thickness)) or np.any(thickness <= 0):
        raise ValueError("finite-layer thicknesses must be positive and finite")
    if not np.isfinite(reference_frequency_hz) or reference_frequency_hz <= 0:
        raise ValueError("reference_frequency_hz must be positive and finite")
    vp = brocher_vp_from_vs(vs)
    density = brocher_density_from_vp(vp)
    qp_array = _quality_array(qp, vs.size, "Qp")
    qs_array = _quality_array(qs, vs.size, "Qs")
    s_transfer = _vertical_transfer_function(
        frequency, thickness, vs, density, qs_array, reference_frequency_hz
    )
    p_transfer = _vertical_transfer_function(
        frequency, thickness, vp, density, qp_array, reference_frequency_hz
    )
    hvsr = s_transfer / p_transfer
    if not np.all(np.isfinite(hvsr)) or np.any(hvsr <= 0):
        raise FloatingPointError("synthetic HVSR is non-finite or nonpositive")
    return np.asarray(hvsr, dtype=np.float64)


def l2_hvsr_misfit(observed: ArrayLike, synthetic: ArrayLike) -> float:
    """Return the unweighted norm-2 objective used by Zaenudin et al. (2024)."""
    observed_array = np.asarray(observed, dtype=np.float64)
    synthetic_array = np.asarray(synthetic, dtype=np.float64)
    if observed_array.shape != synthetic_array.shape:
        raise ValueError("observed and synthetic HVSR must have the same shape")
    if not np.all(np.isfinite(observed_array)) or not np.all(np.isfinite(synthetic_array)):
        raise ValueError("observed and synthetic HVSR must be finite")
    return float(np.linalg.norm(observed_array - synthetic_array, ord=2))


@dataclass(frozen=True)
class InversionBounds:
    """Search intervals for finite-layer thickness and all-layer Vs."""

    thickness_m: FloatArray
    vs_m_s: FloatArray

    @property
    def lower(self) -> FloatArray:
        return np.concatenate((np.asarray(self.thickness_m)[:, 0], np.asarray(self.vs_m_s)[:, 0]))

    @property
    def upper(self) -> FloatArray:
        return np.concatenate((np.asarray(self.thickness_m)[:, 1], np.asarray(self.vs_m_s)[:, 1]))

    @property
    def finite_layer_count(self) -> int:
        return int(np.asarray(self.thickness_m).shape[0])


def validate_inversion_bounds(bounds: InversionBounds) -> None:
    """Validate dimensions, finite intervals, and the Brocher Vs domain."""
    thickness = np.asarray(bounds.thickness_m, dtype=np.float64)
    vs = np.asarray(bounds.vs_m_s, dtype=np.float64)
    if thickness.ndim != 2 or thickness.shape[1:] != (2,):
        raise ValueError("thickness bounds must have shape (finite_layers, 2)")
    if vs.ndim != 2 or vs.shape != (thickness.shape[0] + 1, 2):
        raise ValueError("Vs bounds must include one row per finite layer plus the halfspace")
    if not np.all(np.isfinite(thickness)) or not np.all(np.isfinite(vs)):
        raise ValueError("all inversion bounds must be finite")
    if np.any(thickness <= 0) or np.any(vs <= 0):
        raise ValueError("thickness and Vs bounds must be positive")
    if np.any(thickness[:, 0] >= thickness[:, 1]) or np.any(vs[:, 0] >= vs[:, 1]):
        raise ValueError("every lower bound must be smaller than its upper bound")
    if np.any(vs > 4500):
        raise ValueError("Vs bounds exceed the 4500 m/s Brocher domain")


def split_particle(particle: ArrayLike, finite_layer_count: int) -> tuple[FloatArray, FloatArray]:
    """Split ``[h_1..h_L, Vs_1..Vs_(L+1)]`` into immutable copies."""
    values = np.asarray(particle, dtype=np.float64)
    expected = 2 * finite_layer_count + 1
    if values.shape != (expected,):
        raise ValueError(f"particle must contain {expected} values")
    return values[:finite_layer_count].copy(), values[finite_layer_count:].copy()


@dataclass(frozen=True)
class PsoInversionResult:
    best_particle: FloatArray
    best_misfit: float
    initial_best_misfit: float
    models: FloatArray
    model_misfits: FloatArray
    evaluated_models: FloatArray
    evaluated_misfits: FloatArray
    evaluated_iterations: NDArray[np.int64]
    evaluated_particle_indices: NDArray[np.int64]
    best_misfit_history: FloatArray


def _evaluate_population(
    positions: FloatArray,
    objective: Callable[[FloatArray], float],
) -> FloatArray:
    misfits = np.empty(positions.shape[0], dtype=np.float64)
    for index, position in enumerate(positions):
        try:
            value = float(objective(position))
        except (FloatingPointError, ValueError, OverflowError):
            value = np.inf
        misfits[index] = value if np.isfinite(value) and value >= 0 else np.inf
    return misfits


def _project_nondecreasing_vs(
    positions: FloatArray,
    finite_layer_count: int,
    lower: FloatArray,
    upper: FloatArray,
) -> FloatArray:
    projected = positions.copy()
    vs_start = finite_layer_count
    vs_lower = lower[vs_start:]
    vs_upper = upper[vs_start:]
    feasible_lower = np.maximum.accumulate(vs_lower)
    feasible_upper = np.minimum.accumulate(vs_upper[::-1])[::-1]
    if np.any(feasible_lower > feasible_upper):
        raise ValueError("Vs bounds do not admit a nondecreasing velocity profile")
    velocity = np.clip(projected[:, vs_start:], feasible_lower, feasible_upper)
    projected[:, vs_start:] = np.maximum.accumulate(velocity, axis=1)
    return projected


def run_pso_hvsr_inversion(
    observed: HVSRCurve,
    bounds: InversionBounds,
    *,
    n_particles: int = 80,
    n_iterations: int = 120,
    seed: int = 2024,
    inertia: float = 0.72,
    local_acceleration: float = 1.49,
    global_acceleration: float = 1.49,
    velocity_limit_fraction: float = 0.25,
    qp: float | ArrayLike = 1.0e9,
    qs: float | ArrayLike = 1.0e9,
    reference_frequency_hz: float = 1.0,
    require_nondecreasing_vs: bool = False,
    initial_particle: ArrayLike | None = None,
) -> PsoInversionResult:
    """Invert one HVSR curve with bounded, deterministic conventional PSO.

    Initial particle positions are uniform within the search space and initial
    velocities are zero, matching the supplied paper workflow.  This is the
    conventional PSO update in the supplied text, not a claim of exact RR-PSO.
    ``initial_particle`` (``[h_1..h_L, Vs_1..Vs_(L+1)]``), when given, is clipped
    to the bounds and replaces particle 0; the other particles stay random.
    """
    validate_inversion_bounds(bounds)
    if n_particles < 2 or n_iterations < 1:
        raise ValueError("PSO requires at least two particles and one iteration")
    coefficients = np.array(
        [inertia, local_acceleration, global_acceleration, velocity_limit_fraction]
    )
    if not np.all(np.isfinite(coefficients)) or inertia < 0 or np.any(coefficients[1:] <= 0):
        raise ValueError("PSO coefficients must be finite and physically valid")
    lower = bounds.lower
    upper = bounds.upper
    span = upper - lower
    rng = np.random.default_rng(seed)
    positions = rng.uniform(lower, upper, size=(n_particles, lower.size))
    if initial_particle is not None:
        start = np.asarray(initial_particle, dtype=np.float64)
        if start.shape != lower.shape or not np.all(np.isfinite(start)):
            raise ValueError(f"initial_particle must contain {lower.size} finite values")
        positions[0] = np.clip(start, lower, upper)
    if require_nondecreasing_vs:
        positions = _project_nondecreasing_vs(
            positions, bounds.finite_layer_count, lower, upper
        )
    velocities = np.zeros_like(positions)

    def objective(particle: FloatArray) -> float:
        thickness, vs = split_particle(particle, bounds.finite_layer_count)
        if require_nondecreasing_vs and np.any(np.diff(vs) < 0):
            return np.inf
        synthetic = forward_body_wave_hvsr(
            observed.frequency_hz,
            thickness,
            vs,
            qp=qp,
            qs=qs,
            reference_frequency_hz=reference_frequency_hz,
        )
        return l2_hvsr_misfit(observed.amplitude, synthetic)

    misfits = _evaluate_population(positions, objective)
    if not np.any(np.isfinite(misfits)):
        raise RuntimeError("no finite PSO model could be evaluated within the supplied bounds")
    personal_best = positions.copy()
    personal_misfit = misfits.copy()
    best_index = int(np.argmin(personal_misfit))
    global_best = personal_best[best_index].copy()
    global_misfit = float(personal_misfit[best_index])
    initial_best = global_misfit
    evaluated_models = [positions.copy()]
    evaluated_misfits = [misfits.copy()]
    evaluated_iterations = [np.zeros(n_particles, dtype=np.int64)]
    particle_indices = [np.arange(n_particles, dtype=np.int64)]
    best_history = [global_misfit]
    velocity_limit = velocity_limit_fraction * span

    for iteration in range(1, n_iterations + 1):
        random_local = rng.random(positions.shape)
        random_global = rng.random(positions.shape)
        velocities = (
            inertia * velocities
            + local_acceleration * random_local * (personal_best - positions)
            + global_acceleration * random_global * (global_best - positions)
        )
        velocities = np.clip(velocities, -velocity_limit, velocity_limit)
        positions = np.clip(positions + velocities, lower, upper)
        if require_nondecreasing_vs:
            positions = _project_nondecreasing_vs(
                positions, bounds.finite_layer_count, lower, upper
            )
        misfits = _evaluate_population(positions, objective)
        improved = misfits < personal_misfit
        personal_best = np.where(improved[:, None], positions, personal_best)
        personal_misfit = np.where(improved, misfits, personal_misfit)
        best_index = int(np.argmin(personal_misfit))
        if personal_misfit[best_index] < global_misfit:
            global_best = personal_best[best_index].copy()
            global_misfit = float(personal_misfit[best_index])
        evaluated_models.append(positions.copy())
        evaluated_misfits.append(misfits.copy())
        evaluated_iterations.append(np.full(n_particles, iteration, dtype=np.int64))
        particle_indices.append(np.arange(n_particles, dtype=np.int64))
        best_history.append(global_misfit)

    return PsoInversionResult(
        best_particle=_readonly_float(global_best),
        best_misfit=global_misfit,
        initial_best_misfit=initial_best,
        models=_readonly_float(personal_best),
        model_misfits=_readonly_float(personal_misfit),
        evaluated_models=_readonly_float(np.vstack(evaluated_models)),
        evaluated_misfits=_readonly_float(np.concatenate(evaluated_misfits)),
        evaluated_iterations=np.concatenate(evaluated_iterations),
        evaluated_particle_indices=np.concatenate(particle_indices),
        best_misfit_history=_readonly_float(best_history),
    )


@dataclass(frozen=True)
class VsDepthProfile:
    depth_m: FloatArray
    vs_mean_m_s: FloatArray
    vs_std_m_s: FloatArray
    vs_p05_m_s: FloatArray
    vs_median_m_s: FloatArray
    vs_p95_m_s: FloatArray


@dataclass(frozen=True)
class ModelEnsemble:
    models: FloatArray
    misfits: FloatArray

    @property
    def n_models(self) -> int:
        return int(self.models.shape[0])

    @property
    def finite_layer_count(self) -> int:
        return (int(self.models.shape[1]) - 1) // 2

    def depth_profile(self, depth_m: ArrayLike) -> VsDepthProfile:
        depth = np.asarray(depth_m, dtype=np.float64)
        if depth.ndim != 1 or not np.all(np.isfinite(depth)) or np.any(depth < 0):
            raise ValueError("depth grid must be one-dimensional, finite, and nonnegative")
        sampled = np.empty((self.n_models, depth.size), dtype=np.float64)
        for index, model in enumerate(self.models):
            thickness, vs = split_particle(model, self.finite_layer_count)
            layer_indices = np.searchsorted(np.cumsum(thickness), depth, side="right")
            sampled[index] = vs[layer_indices]
        return VsDepthProfile(
            depth_m=_readonly_float(depth),
            vs_mean_m_s=_readonly_float(np.mean(sampled, axis=0)),
            vs_std_m_s=_readonly_float(np.std(sampled, axis=0)),
            vs_p05_m_s=_readonly_float(np.quantile(sampled, 0.05, axis=0)),
            vs_median_m_s=_readonly_float(np.median(sampled, axis=0)),
            vs_p95_m_s=_readonly_float(np.quantile(sampled, 0.95, axis=0)),
        )


def acceptable_model_ensemble(
    models: ArrayLike,
    misfits: ArrayLike,
    *,
    top_n: int = 200,
    maximum_relative_misfit: float | None = None,
) -> ModelEnsemble:
    """Select a low-misfit search ensemble; this is not a posterior distribution."""
    model_array = np.asarray(models, dtype=np.float64)
    misfit_array = np.asarray(misfits, dtype=np.float64)
    if model_array.ndim != 2 or misfit_array.shape != (model_array.shape[0],):
        raise ValueError("models must be 2-D with one corresponding misfit each")
    if model_array.shape[1] < 3 or model_array.shape[1] % 2 != 1:
        raise ValueError("each model must encode L thicknesses and L+1 Vs values")
    if top_n < 1:
        raise ValueError("top_n must be positive")
    valid = np.isfinite(misfit_array) & np.all(np.isfinite(model_array), axis=1)
    if not np.any(valid):
        raise ValueError("no finite models are available for the ensemble")
    candidates = model_array[valid]
    candidate_misfits = misfit_array[valid]
    order = np.argsort(candidate_misfits, kind="stable")
    candidates = candidates[order]
    candidate_misfits = candidate_misfits[order]
    if maximum_relative_misfit is not None:
        if not np.isfinite(maximum_relative_misfit) or maximum_relative_misfit < 0:
            raise ValueError("maximum_relative_misfit must be finite and nonnegative")
        limit = candidate_misfits[0] * (1.0 + maximum_relative_misfit)
        within_limit = candidate_misfits <= limit
        candidates = candidates[within_limit]
        candidate_misfits = candidate_misfits[within_limit]
    count = min(top_n, candidates.shape[0])
    return ModelEnsemble(
        models=_readonly_float(candidates[:count]),
        misfits=_readonly_float(candidate_misfits[:count]),
    )


def vs30_from_layers(thickness_m: ArrayLike, vs_m_s: ArrayLike) -> float:
    """Calculate mathematical Vs30 using travel time through the upper 30 m."""
    thickness = np.asarray(thickness_m, dtype=np.float64)
    vs = np.asarray(vs_m_s, dtype=np.float64)
    if thickness.ndim != 1 or vs.ndim != 1 or thickness.size + 1 != vs.size:
        raise ValueError("thickness must have one fewer value than Vs")
    if not np.all(np.isfinite(thickness)) or np.any(thickness <= 0):
        raise ValueError("thickness values must be positive and finite")
    if not np.all(np.isfinite(vs)) or np.any(vs <= 0):
        raise ValueError("Vs values must be positive and finite")
    remaining = 30.0
    travel_time = 0.0
    for layer_thickness, layer_vs in zip(np.r_[thickness, np.inf], vs, strict=True):
        traversed = min(float(layer_thickness), remaining)
        travel_time += traversed / float(layer_vs)
        remaining -= traversed
        if remaining <= 0:
            break
    return 30.0 / travel_time


__all__ = [
    "HVSRCurve",
    "InversionBounds",
    "ModelEnsemble",
    "PsoInversionResult",
    "VsDepthProfile",
    "acceptable_model_ensemble",
    "brocher_density_from_vp",
    "brocher_vp_from_vs",
    "forward_body_wave_hvsr",
    "l2_hvsr_misfit",
    "parse_geopsy_hv",
    "run_pso_hvsr_inversion",
    "split_particle",
    "validate_inversion_bounds",
    "vs30_from_layers",
]
