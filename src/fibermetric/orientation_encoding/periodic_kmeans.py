#!/usr/bin/env python

"""
k-means on periodic data

Author: Bryson Gray
2024
"""

import numpy as np

from ..difference_measures.angles import periodic_distance_1d


def periodic_mean(data: np.ndarray, x: np.ndarray, period: float=None) -> float:
    if period is None:
        period = np.pi
    d2 = periodic_distance_1d(x[:, None, None], data[None, :, None], period) ** 2
    id = np.argmin(d2.sum(axis=1))

    return x[id]


def _periodic_kmeans_single(data: np.ndarray, k: int, period: float, nstarts: int) -> np.ndarray:
    metrics = []
    mus = []
    for i in range(nstarts):
        # initialize k random starting points
        x = np.arange(180) * period/180
        mu = np.random.choice(data,k,replace=False)
        while True:
            mu_old = mu.copy()
            d = periodic_distance_1d(mu[:, None, None], data[None, :, None], period)
            labels = np.argmin(d, axis=0)
            for j in range(len(mu)):
                mu[j] = periodic_mean(data[labels==j], x, period)
            if np.all(mu == mu_old):
                break
        mus.append(mu)
        metrics.append(np.sum(np.min(d, axis=0)))

    id = np.argmin(metrics)

    return mus[id]


def _apsym_kmeans_single(data, k, nstarts: int):
    metrics = []
    mus = []
    for i in range(nstarts):
        mu_id = np.random.choice(np.arange(len(data)), k, replace=False)
        mu = data[mu_id]
        # while True:
        count = 0
        for t in range(100):
            count += 1
            mu_old = mu.copy()
            d = mu.dot(data.T)
            labels = np.argmax(np.abs(d), axis=0)
            flip_ids = np.array(d[labels, range(len(labels))] < 0).nonzero()
            data[flip_ids] *= -1
            for j in range(len(mu)):
                data_j = data[labels==j]
                mu[j] = np.sum(data_j, axis=0) / len(data_j)
                mu[j] /= np.sum(mu[j]**2)**0.5
            if np.all(mu == mu_old):
                break
        mus.append(mu)
        metrics.append(np.sum(np.min(d, axis=0)))
    id = np.argmin(metrics)
    
    return mus[id]


def periodic_kmeans(data: np.ndarray, k: int, period: float, nstarts=1, shape_out=None) -> np.ndarray:
    """Cluster periodic scalar data, optionally in a grid of spatial bins.

    With ``shape_out`` the input is gathered into equal non-overlapping bins
    and clustered independently in each bin, returning ``shape_out + (k,)``.
    Bins with fewer than ``k`` samples are filled with NaN.
    """
    from ..auxiliary.utils import gather

    data = np.asarray(data, dtype=float)
    if shape_out is None:
        return _periodic_kmeans_single(data.ravel(), k, period, nstarts)

    shape_out = tuple(shape_out)
    gathered = gather(data, shape_out)
    samples = gathered.reshape((-1, gathered.shape[-1]))
    centers = np.full((len(samples), k), np.nan)
    for index, values in enumerate(samples):
        if len(values) < k:
            continue
        centers[index] = _periodic_kmeans_single(values, k, period, nstarts)

    return centers.reshape(shape_out + (k,))


def apsym_kmeans(data, k, nstarts=1, shape_out=None):
    """Cluster antipodally symmetric unit vectors, optionally in spatial bins.

    With ``shape_out`` the input is gathered into equal non-overlapping bins
    and clustered independently in each bin, returning
    ``shape_out + (k, ndim)``. Bins with fewer than ``k`` samples are filled
    with NaN.
    """
    from ..auxiliary.utils import gather

    data = np.array(data, dtype=float)
    if shape_out is None:
        return _apsym_kmeans_single(data.reshape((-1, data.shape[-1])), k, nstarts)

    shape_out = tuple(shape_out)
    gathered = gather(data, shape_out, feature_axis=-1)
    samples = gathered.reshape((-1, gathered.shape[-2], data.shape[-1]))
    centers = np.full((len(samples), k, data.shape[-1]), np.nan)
    for index, vectors in enumerate(samples):
        if len(vectors) < k:
            continue
        centers[index] = _apsym_kmeans_single(np.array(vectors), k, nstarts)

    return centers.reshape(shape_out + (k, data.shape[-1]))