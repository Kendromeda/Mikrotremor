from __future__ import annotations

import math

import pytest

from mhvsr_vs30.mam.pilot import center_on_outer_centroid, pair_distances, ring_for_distance

MEASURED = {
    "Solo1": (570738.0, 9138298.0),
    "Solo2": (570735.0, 9138295.0),
    "Solo3": (570739.0, 9138296.0),
    "Solo4": (570737.0, 9138299.0),
}


def test_center_correction_preserves_original_and_uses_outer_centroid() -> None:
    proposed = center_on_outer_centroid(MEASURED, "Solo1", ("Solo2", "Solo3", "Solo4"))

    assert MEASURED["Solo1"] == (570738.0, 9138298.0)
    assert proposed["Solo1"] == pytest.approx((570737.0, 9138296.666666667))
    assert all(proposed[name] == MEASURED[name] for name in ("Solo2", "Solo3", "Solo4"))
    assert proposed is not MEASURED


def test_proposed_geometry_has_three_pairs_in_each_existing_ring() -> None:
    proposed = center_on_outer_centroid(MEASURED, "Solo1", ("Solo2", "Solo3", "Solo4"))
    rings = ((0.914, 2.736), (3.106, 4.972))
    pairs = pair_distances(proposed)

    assert len(pairs) == 6
    counts = [
        sum(ring_for_distance(distance, rings) == index for _, _, distance in pairs)
        for index in (0, 1)
    ]
    assert counts == [3, 3]
    outer_edge = next(
        distance for left, right, distance in pairs if {left, right} == {"Solo2", "Solo3"}
    )
    assert outer_edge == pytest.approx(math.sqrt(17))


def test_geometry_rejects_unknown_station_and_overlapping_rings() -> None:
    with pytest.raises(ValueError, match="unknown"):
        center_on_outer_centroid(MEASURED, "missing", ("Solo2", "Solo3", "Solo4"))
    with pytest.raises(ValueError, match="overlap"):
        ring_for_distance(2.0, ((1.0, 3.0), (2.0, 4.0)))


def test_measured_geometry_is_not_silently_reclassified() -> None:
    rings = ((0.914, 2.736), (3.106, 4.972))
    assignments = [
        ring_for_distance(distance, rings) for _, _, distance in pair_distances(MEASURED)
    ]
    assert assignments.count(0) == 2
    assert assignments.count(1) == 4


@pytest.mark.parametrize(
    "coordinates, message",
    [
        ({"Solo1": (0.0, 0.0)}, "at least two"),
        ({"Solo1": (0.0,), "Solo2": (1.0, 1.0)}, "x and y"),
        ({"Solo1": (float("nan"), 0.0), "Solo2": (1.0, 1.0)}, "finite"),
        ({"Solo1": (0.0, 0.0), "Solo2": (0.0, 0.0)}, "distinct"),
    ],
)
def test_invalid_coordinates_are_rejected(coordinates: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        pair_distances(coordinates)


@pytest.mark.parametrize(
    "distance, rings, message",
    [
        (-1.0, ((0.0, 1.0),), "nonnegative"),
        (1.0, ((0.0,),), "lower and upper"),
        (1.0, ((2.0, 1.0),), "ordered"),
    ],
)
def test_invalid_rings_are_rejected(distance: float, rings: tuple, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ring_for_distance(distance, rings)
