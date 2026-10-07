"""One shared raw-feature boundary for all three paper-era architectures."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


ARCHITECTURES = ("TDL-B", "TDL-W", "TDL-BW")


def raw_features_for_architecture(
    dataset: Mapping[str, np.ndarray], architecture: str
) -> np.ndarray:
    """Return the common, unstandardized Ibiza feature matrix.

    This boundary deliberately performs no normalization and no architecture-
    specific feature construction.  Each frozen checkpoint must apply its own
    KLT3 ``StandardizeLayer`` after this function returns.
    """

    if architecture not in ARCHITECTURES:
        raise ValueError(
            f"architecture must be one of {ARCHITECTURES}, got {architecture!r}"
        )
    if "features" not in dataset:
        raise KeyError("dataset does not contain the raw 'features' matrix")
    features = np.asarray(dataset["features"])
    if features.dtype != np.dtype(np.float64):
        raise TypeError(f"features must be float64, got {features.dtype}")
    if features.ndim != 2 or features.shape[1] != 3:
        raise ValueError(f"features must have shape (n, 3), got {features.shape}")
    if not np.all(np.isfinite(features)):
        raise ValueError("features contain a non-finite value")
    return features
