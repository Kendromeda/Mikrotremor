"""Geometry calculations shared by the Solo pilot notebook and its tests.

Measured coordinates must be retained separately from a modeled correction.
This module only computes distances and a clearly named geometric hypothesis;
it cannot establish where a sensor was actually deployed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations
from math import hypot, isfinite

Coordinate = tuple[float, float]
Ring = tuple[float, float]


def _validated_coordinates(coordinates: Mapping[str, Sequence[float]]) -> dict[str, Coordinate]:
    if len(coordinates) < 2:
        raise ValueError("at least two stations are required")
    validated: dict[str, Coordinate] = {}
    for name, position in coordinates.items():
        if len(position) != 2:
            raise ValueError(f"{name}: coordinates must contain x and y")
        x, y = float(position[0]), float(position[1])
        if not (isfinite(x) and isfinite(y)):
            raise ValueError(f"{name}: coordinates must be finite")
        validated[name] = (x, y)
    if len(set(validated.values())) != len(validated):
        raise ValueError("station coordinates must be distinct")
    return validated


def center_on_outer_centroid(
    coordinates: Mapping[str, Sequence[float]], center: str, outer: Sequence[str]
) -> dict[str, Coordinate]:
    """Return a modeled triangular layout without changing the supplied positions."""
    measured = _validated_coordinates(coordinates)
    if center not in measured or any(name not in measured for name in outer):
        raise ValueError("unknown station in center or outer list")
    if len(outer) != 3 or len(set(outer)) != 3 or center in outer:
        raise ValueError("outer must name exactly three distinct non-center stations")
    centroid = (
        sum(measured[name][0] for name in outer) / 3.0,
        sum(measured[name][1] for name in outer) / 3.0,
    )
    return {**measured, center: centroid}


def pair_distances(
    coordinates: Mapping[str, Sequence[float]],
) -> list[tuple[str, str, float]]:
    """Compute every station-pair distance in the coordinates' stated unit."""
    positions = _validated_coordinates(coordinates)
    return [
        (
            left,
            right,
            hypot(
                positions[left][0] - positions[right][0], positions[left][1] - positions[right][1]
            ),
        )
        for left, right in combinations(positions, 2)
    ]


def ring_for_distance(distance: float, rings: Sequence[Sequence[float]]) -> int | None:
    """Identify one nonoverlapping inclusive distance interval, if any."""
    if not isfinite(distance) or distance < 0:
        raise ValueError("distance must be finite and nonnegative")
    previous_high = float("-inf")
    for ring in rings:
        if len(ring) != 2:
            raise ValueError("each ring needs lower and upper bounds")
        low, high = float(ring[0]), float(ring[1])
        if not (isfinite(low) and isfinite(high) and 0 <= low <= high):
            raise ValueError("ring bounds must be finite, nonnegative and ordered")
        if low <= previous_high:
            raise ValueError("ring intervals overlap or are not ordered")
        previous_high = high
    for index, (low, high) in enumerate(rings):
        if float(low) <= distance <= float(high):
            return index
    return None
