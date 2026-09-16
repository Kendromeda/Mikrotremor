"""Phase 5 unit tests: grouped splitting and the two baseline estimators."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mhvsr_vs30.training.evaluate import nehrp_class
from mhvsr_vs30.training.models import MedianVs30, RidgeLogVs30, load_model, save_model
from mhvsr_vs30.training.splits import leave_one_site_out

SITES = ["s:A", "s:A", "s:A", "s:B", "s:B", "s:C"]


def test_leave_one_site_out_holds_out_whole_sites() -> None:
    folds = leave_one_site_out(SITES)
    assert [f.test_site for f in folds] == ["s:A", "s:B", "s:C"]

    labels = np.asarray(SITES)
    for fold in folds:
        assert set(labels[fold.test_index]) == {fold.test_site}
        assert fold.test_site not in set(labels[fold.train_index])


def test_every_row_is_tested_exactly_once() -> None:
    folds = leave_one_site_out(SITES)
    tested = np.concatenate([f.test_index for f in folds])
    assert sorted(tested.tolist()) == list(range(len(SITES)))


def test_no_row_is_in_both_sides_of_a_fold() -> None:
    for fold in leave_one_site_out(SITES):
        assert not set(fold.train_index.tolist()) & set(fold.test_index.tolist())


def test_split_indices_are_read_only() -> None:
    fold = leave_one_site_out(SITES)[0]
    with pytest.raises(ValueError):
        fold.train_index[0] = 99


def test_a_single_site_cannot_be_split() -> None:
    with pytest.raises(ValueError, match="at least 2 sites"):
        leave_one_site_out(["s:A", "s:A"])


def test_median_baseline_ignores_features() -> None:
    model = MedianVs30().fit(np.zeros((3, 5)), np.array([100.0, 200.0, 900.0]))
    prediction = model.predict(np.random.default_rng(0).normal(size=(4, 5)))

    assert prediction.shape == (4,)
    assert len(set(prediction.tolist())) == 1
    assert prediction[0] == 200.0


def test_median_baseline_respects_site_weights() -> None:
    """Three recordings of one soft site must not outvote two separate stiff sites."""
    target = np.array([150.0, 150.0, 150.0, 1400.0, 1500.0])
    weight = np.array([1 / 3, 1 / 3, 1 / 3, 1.0, 1.0])  # one soft site, two stiff sites
    features = np.zeros((5, 2))

    unweighted = MedianVs30().fit(features, target).value_
    weighted = MedianVs30().fit(features, target, weight).value_

    assert unweighted == 150.0  # recordings win
    assert weighted == 1400.0  # sites win


def test_weighted_median_breaks_ties_toward_the_lower_value() -> None:
    """Exactly balanced weight is ambiguous; the convention is pinned deliberately."""
    model = MedianVs30().fit(
        np.zeros((2, 2)), np.array([150.0, 1400.0]), np.array([1.0, 1.0])
    )
    assert model.value_ == 150.0


def test_ridge_recovers_a_monotone_relationship() -> None:
    rng = np.random.default_rng(7)
    amplitude = np.linspace(0.5, 8.0, 40)
    features = np.column_stack([amplitude, rng.normal(scale=0.01, size=40) + 1.0])
    target = 100.0 * amplitude**1.5

    model = RidgeLogVs30(alpha=1e-6).fit(features, target)
    predicted = model.predict(features)

    assert np.corrcoef(np.log(predicted), np.log(target))[0, 1] > 0.99


def test_ridge_rejects_non_positive_amplitudes() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        RidgeLogVs30().fit(np.array([[1.0, 0.0]]), np.array([200.0]))


def test_unfitted_models_refuse_to_predict() -> None:
    for model in (MedianVs30(), RidgeLogVs30()):
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(np.ones((2, 3)))


@pytest.mark.parametrize("model_factory", [MedianVs30, RidgeLogVs30])
def test_predictions_survive_a_save_and_reload(model_factory, tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    features = np.abs(rng.normal(loc=2.0, size=(12, 35))) + 0.1
    target = rng.uniform(150.0, 1500.0, size=12)

    model = model_factory().fit(features, target)
    before = model.predict(features)
    after = load_model(save_model(model, tmp_path / "m.joblib")).predict(features)

    assert np.array_equal(before, after)


@pytest.mark.parametrize(
    ("vs30", "expected"),
    [(120.0, "E"), (180.0, "E"), (196.0, "D"), (432.0, "C"), (1464.0, "B"), (1600.0, "A")],
)
def test_nehrp_class_boundaries(vs30: float, expected: str) -> None:
    assert nehrp_class(vs30) == expected
