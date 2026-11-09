"""Pure-NumPy Isolation Forest (same algorithm and scoring as scikit-learn's)."""
import numpy as np

EULER = 0.5772156649


def _c(n):
    """Average path length of an unsuccessful search in a binary tree of n points."""
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * (np.log(n - 1) + EULER) - 2.0 * (n - 1) / n


def _build(X, depth, limit, rng):
    n = len(X)
    if depth >= limit or n <= 1:
        return ("leaf", n)
    q = int(rng.integers(X.shape[1]))
    lo, hi = X[:, q].min(), X[:, q].max()
    if lo == hi:
        return ("leaf", n)
    p = rng.uniform(lo, hi)
    mask = X[:, q] < p
    return ("node", q, p, _build(X[mask], depth + 1, limit, rng), _build(X[~mask], depth + 1, limit, rng))


def _path(node, x):
    depth = 0
    while node[0] == "node":
        node = node[3] if x[node[1]] < node[2] else node[4]
        depth += 1
    return depth + _c(node[1])


class IsolationForestLite:
    def __init__(self, n_estimators=200, max_samples=256, contamination=0.02, random_state=42):
        self.n_estimators, self.max_samples = n_estimators, max_samples
        self.contamination, self.random_state = contamination, random_state

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)
        self.psi_ = min(self.max_samples, len(X))
        limit = int(np.ceil(np.log2(self.psi_)))
        self.trees_ = [_build(X[rng.choice(len(X), self.psi_, replace=False)], 0, limit, rng)
                       for _ in range(self.n_estimators)]
        sample = X[rng.choice(len(X), min(4000, len(X)), replace=False)]
        self.offset_ = np.percentile(self.score_samples(sample), 100 * self.contamination)
        return self

    def score_samples(self, X):
        X = np.asarray(X, dtype=float)
        norm = _c(self.psi_)
        out = np.empty(len(X))
        for i, x in enumerate(X):
            avg = np.mean([_path(t, x) for t in self.trees_])
            out[i] = -(2.0 ** (-avg / norm))
        return out

    def decision_function(self, X):
        return self.score_samples(X) - self.offset_

    def predict(self, X):
        return np.where(self.decision_function(X) < 0, -1, 1)
