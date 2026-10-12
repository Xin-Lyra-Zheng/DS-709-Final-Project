"""Hardt, Morina RTDP, and Agarwal EG on frozen scores."""

import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from sklearn.base import BaseEstimator, ClassifierMixin
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import ExponentiatedGradient, TruePositiveRateParity

GRID = np.r_[np.arange(101) / 100, np.nextafter(1., 2.)]


def cell_ids(groups, shape):
    return np.ravel_multi_index(groups[:, :2].T, shape)


def optimal_thresholds(score, y, cells, count, weights=None, exact=False):
    weights = np.ones(len(y)) if weights is None else np.asarray(weights)
    if exact:
        thresholds = np.full(count, np.nextafter(1., 2.))
        for c in range(count):
            ix = np.flatnonzero(cells == c)
            if not len(ix):
                continue
            unique, inverse = np.unique(score[ix], return_inverse=True)
            hist = np.bincount(inverse, weights=weights[ix] * (1 - 2 * y[ix]),
                               minlength=len(unique))
            costs = np.r_[np.cumsum(hist[::-1])[::-1], 0.]
            j = int(np.argmin(costs))
            thresholds[c] = np.nextafter(1., 2.) if j == len(unique) else (0. if j == 0 else unique[j])
        return thresholds
    bins = np.searchsorted(GRID, score, side="right") - 1
    hist = np.bincount(cells * len(GRID) + bins, weights=weights * (1 - 2 * y),
                       minlength=count * len(GRID)).reshape(count, -1)
    loss = np.bincount(cells, weights=weights * y, minlength=count)[:, None]
    loss = loss + np.cumsum(hist[:, ::-1], axis=1)[:, ::-1]
    return GRID[np.argmin(loss, axis=1)]


class ScoreEstimator(ClassifierMixin, BaseEstimator):
    def __init__(self):
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        p = np.asarray(X).reshape(-1)
        return np.column_stack([1 - p, p])


class ThresholdOracle(ClassifierMixin, BaseEstimator):
    def __init__(self, count=1, fallback=.5):
        self.count = count
        self.fallback = fallback

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X)
        cells = X[:, 1].astype(int)
        self.thresholds_ = optimal_thresholds(X[:, 0], np.asarray(y), cells,
                                              self.count, sample_weight)
        present = np.bincount(cells, minlength=self.count) > 0
        self.thresholds_[~present] = self.fallback
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        X = np.asarray(X)
        return (X[:, 0] >= self.thresholds_[X[:, 1].astype(int)]).astype(int)


def hardt(scores, labels, groups, shape, fallback):
    ids = {p: cell_ids(groups[p], shape) for p in scores}
    count = int(np.prod(shape))
    accepted = [c for c in range(count)
                if len(np.unique(labels["repair"][ids["repair"] == c])) == 2]
    keep = np.isin(ids["repair"], accepted)
    model = ThresholdOptimizer(
        estimator=ScoreEstimator(), prefit=True,
        constraints="true_positive_rate_parity", objective="accuracy_score",
        grid_size=1000, flip=False, predict_method="predict_proba")
    model.fit(scores["repair"][keep, None], labels["repair"][keep],
              sensitive_features=ids["repair"][keep].astype(str))
    out = {}
    for part in scores:
        ix = np.isin(ids[part], accepted)
        q = (scores[part] >= fallback).astype(float)
        q[ix] = model._pmf_predict(scores[part][ix, None],
                                   sensitive_features=ids[part][ix].astype(str))[:, 1]
        out[part] = q
    return out, {"accepted": accepted, "fallback_cells": count - len(accepted)}


def eg(scores, labels, groups, shape, fallback):
    ids = {p: cell_ids(groups[p], shape) for p in scores}
    count = int(np.prod(shape))
    X = {p: np.column_stack([scores[p], ids[p]]) for p in scores}
    model = ExponentiatedGradient(
        ThresholdOracle(count=count, fallback=fallback),
        TruePositiveRateParity(difference_bound=0),
        eps=.005, max_iter=50, nu=1e-5, run_linprog_step=True)
    model.fit(X["repair"], labels["repair"],
              sensitive_features=ids["repair"].astype(str))
    out = {p: model._pmf_predict(X[p])[:, 1] for p in scores}
    fit = {"best_gap": float(model.best_gap_), "last_iter": int(model.last_iter_),
           "oracle_calls": int(model.n_oracle_calls_),
           "weights": model.weights_.tolist(),
           "thresholds": [h.thresholds_.tolist() for h in model.predictors_]}
    return out, fit


class RTDP:
    def __init__(self, score, y, groups, shape, fallback):
        self.shape = shape
        self.cells = cell_ids(groups, shape)
        self.count = int(np.prod(shape))
        self.thresholds = optimal_thresholds(score, y, self.cells, self.count, exact=True)
        self.present = np.bincount(self.cells, minlength=self.count) > 0
        self.thresholds[~self.present] = fallback
        binary = (score >= self.thresholds[self.cells]).astype(int)
        columns = 2 * self.cells + binary
        self.B = sparse.csr_matrix((np.ones(len(y)), (np.arange(len(y)), columns)),
                                   shape=(len(y), 2 * self.count))
        self.error = np.asarray(self.B.T.dot(1 - 2 * y)).ravel() / len(y)
        self.constant = float(np.mean(y))
        self.y = y

    def row(self, mask):
        return np.asarray(self.B[mask].sum(axis=0)).ravel() / np.sum(mask)

    def solve(self):
        C = self.count
        size = 2 * C + 2
        rows = []
        for b in (0, 1):
            row = np.zeros(size)
            row[2 * np.flatnonzero(self.present) + b] = 1 / np.sum(self.present)
            row[2 * C + b] = -1
            rows.append(row)
        for c in np.flatnonzero(~self.present):
            for b in (0, 1):
                row = np.zeros(size)
                row[2 * c + b] = 1
                row[2 * C + b] = -1
                rows.append(row)
        positive = self.y == 1
        overall = self.row(positive)
        for c in range(C):
            mask = positive & (self.cells == c)
            if mask.any():
                rows.append(np.r_[self.row(mask) - overall, np.zeros(2)])
        A_eq = np.asarray(rows)
        objective = np.r_[self.error, np.zeros(2)]
        result = linprog(objective, A_eq=A_eq, b_eq=np.zeros(len(rows)),
                         bounds=(0., 1.), method="highs")
        if not result.success:
            raise RuntimeError(result.message)
        x = result.x
        residual = float(np.max(np.abs(A_eq @ x)))
        if residual > 2e-6:
            raise RuntimeError(f"constraint residual {residual}")
        return {"coefficients": x[:2 * C].reshape(C, 2),
                "pooled": x[2 * C:], "thresholds": self.thresholds,
                "shape": self.shape, "fit_objective": float(result.fun + self.constant),
                "constraint_residual": residual}


def rtdp_predict(policy, score, groups):
    cells = cell_ids(groups, policy["shape"])
    binary = (score >= policy["thresholds"][cells]).astype(int)
    return policy["coefficients"][cells, binary]
