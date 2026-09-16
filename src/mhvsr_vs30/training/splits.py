"""Grouped splitting.

A physical location is the unit of independence, not a recording. Six recordings
from one site are six views of the same ground, so putting some in train and
some in test would measure how well the model recognises a site it has already
seen.
"""

from __future__ import annotations

import dataclasses

import numpy as np

__all__ = ["Fold", "leave_one_site_out"]


@dataclasses.dataclass(frozen=True)
class Fold:
    """One held-out site and the row indices on each side of the split."""

    test_site: str
    train_index: np.ndarray
    test_index: np.ndarray

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
    labels = np.asarray(groups)
    sites = sorted(set(labels.tolist()))
    if len(sites) < 2:
        raise ValueError(f"leave-one-site-out needs at least 2 sites, got {len(sites)}")

    folds: list[Fold] = []
    for site in sites:
        test = np.flatnonzero(labels == site)
        train = np.flatnonzero(labels != site)
        folds.append(Fold(test_site=site, train_index=train, test_index=test))
    return folds
