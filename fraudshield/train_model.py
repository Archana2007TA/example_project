"""Step 1-2: build synthetic data (~2% fraud) and train the Isolation Forest."""
import numpy as np, joblib
from iforest import IsolationForestLite


def make_data(n=20000, frac=0.02, seed=42):
    rng = np.random.default_rng(seed)
    normal = np.column_stack([
        1 + rng.poisson(0.4, n),                          # velocity
        rng.normal(0, 1, n).clip(-3, 3),                  # amount z-score
        rng.exponential(15, n).clip(0, 150),              # geo speed km/h
        rng.choice([1, 2, 4, 5], n, p=[.75, .2, .04, .01]),  # merchant risk
    ])
    k = int(n * frac)
    fraud = np.column_stack([rng.integers(5, 12, k), rng.uniform(5, 25, k),
                             rng.uniform(600, 4000, k), rng.choice([4, 5], k)])
    mask = rng.random((k, 4)) < 0.5                       # each fraud has a mix of weird features
    fraud = np.where(mask, fraud, normal[:k])
    return np.vstack([normal, fraud])


def train(path="model.pkl"):
    X = make_data()
    model = IsolationForestLite(n_estimators=200, contamination=0.02, random_state=42).fit(X)
    joblib.dump(model, path)
    print(f"Saved {path} trained on {len(X)} rows")
    return model


if __name__ == "__main__":
    train()
