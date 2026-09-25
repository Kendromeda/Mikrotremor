"""Grouped splitting.

A physical location is the unit of independence, not a recording. Six recordings
from one site are six views of the same ground, so putting some in train and
some in test would measure how well the model recognises a site it has already
seen.
"""

from __future__ import annotations

import dataclasses

import numpy as np

__all__ = ["Fold", "external_holdout", "leave_one_group_out", "leave_one_site_out"]


@dataclasses.dataclass(frozen=True)
class Fold:
    """One held-out site and the row indices on each side of the split."""

    test_site: str
    train_index: np.ndarray
    test_index: np.ndarray

    @property
    def test_group(self) -> str:
        """The held-out group, for protocols other than site-level LOSO."""
        return self.test_site

    def __post_init__(self) -> None:
        for name in ("train_index", "test_index"):
            array = np.asarray(getattr(self, name), dtype=np.int64)
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        if set(self.train_index.tolist()) & set(self.test_index.tolist()):
            raise ValueError("train and test indices overlap")


def leave_one_site_out(groups: np.ndarray | list[str]) -> list[Fold]:
    """One fold per site, holding that whole site out.

    Raises when there are fewer than two sites: a single site cannot be both the
    training set and the held-out set, and pretending otherwise would produce a
    number that looks like a score.
    """
    return leave_one_group_out(groups, group_name="site")


def leave_one_group_out(groups: np.ndarray | list[str], *, group_name: str) -> list[Fold]:
    """Return one fold per group while retaining every recording in that group.

    ``group_name`` is part of the error contract and report provenance: callers
    must state whether the groups represent sites, studies, or geography.
    """
    labels = np.asarray(groups, dtype=str)
    if labels.ndim != 1 or not labels.size:
        raise ValueError(
            f"leave-one-{group_name}-out needs a non-empty one-dimensional group array"
        )
    if any(not value.strip() for value in labels):
        raise ValueError(f"leave-one-{group_name}-out groups must be non-empty strings")

    unique_groups = sorted(set(labels.tolist()))
    if len(unique_groups) < 2:
        raise ValueError(
            f"leave-one-{group_name}-out needs at least 2 {group_name}s, got {len(unique_groups)}"
        )

    folds: list[Fold] = []
    for group in unique_groups:
        test = np.flatnonzero(labels == group)
        train = np.flatnonzero(labels != group)
        folds.append(Fold(test_site=group, train_index=train, test_index=test))
    return folds


def external_holdout(
    site_groups: np.ndarray | list[str],
    holdout_mask: np.ndarray | list[bool],
    *,
    holdout_name: str,
) -> Fold:
    """Create one external holdout fold without splitting a physical site.

    A site represented on both sides normally means contradictory geography
    metadata or a join error. Refusing it is safer than silently assigning a
    recording-level country label and leaking a site into training.
    """
    sites = np.asarray(site_groups, dtype=str)
    mask = np.asarray(holdout_mask, dtype=bool)
    if sites.ndim != 1 or mask.ndim != 1 or len(sites) != len(mask) or not len(sites):
        raise ValueError("external holdout needs equally sized non-empty site groups and mask")
    if not holdout_name.strip():
        raise ValueError("external holdout name must be non-empty")
    if not mask.any() or mask.all():
        raise ValueError("external holdout needs at least one training and one held-out row")

    for site in sorted(set(sites.tolist())):
        site_mask = mask[sites == site]
        if site_mask.any() and not site_mask.all():
            raise ValueError(f"physical site {site!r} appears on both sides of external holdout")

    return Fold(
        test_site=holdout_name,
        train_index=np.flatnonzero(~mask),
        test_index=np.flatnonzero(mask),
    )
