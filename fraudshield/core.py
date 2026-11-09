"""Shared feature engineering used by BOTH training and the live API."""
import math
from collections import defaultdict, deque
from datetime import datetime
import numpy as np

FEATURES = ["velocity", "amount_z", "geo_speed", "merchant_risk"]

CATEGORY_RISK = {"grocery": 1, "restaurant": 1, "retail": 1, "travel": 2,
                 "wire_transfer": 4, "gambling": 4, "crypto": 5}

CITIES = {"Bengaluru": (12.97, 77.59), "Mumbai": (19.07, 72.88), "Delhi": (28.61, 77.21),
          "London": (51.50, -0.12), "New York": (40.71, -74.00),
          "Singapore": (1.35, 103.82), "Dubai": (25.20, 55.27)}


def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * 6371 * math.asin(math.sqrt(a))


class FeatureEngine:
    """Keeps a sliding window of each user's recent transactions in memory."""

    def __init__(self):
        self.recent = defaultdict(lambda: deque(maxlen=50))  # (ts, amount, lat, lon)

    def compute(self, tx: dict) -> dict:
        ts = datetime.fromisoformat(tx["timestamp"].replace("Z", "+00:00")).timestamp()
        lat, lon = tx["location"]["lat"], tx["location"]["lon"]
        hist = self.recent[tx["user_id"]]

        velocity = 1 + sum(1 for h in hist if 0 <= ts - h[0] <= 300)  # last 5 minutes

        amounts = [h[1] for h in hist]
        mu, sd = (np.mean(amounts), max(np.std(amounts), 10)) if len(amounts) >= 3 else (50.0, 40.0)
        amount_z = float(np.clip((tx["amount"] - mu) / sd, -10, 30))

        geo_speed = jump_km = 0.0
        if hist:
            last = hist[-1]
            jump_km = haversine(last[2], last[3], lat, lon)
            hours = max(ts - last[0], 60) / 3600  # floor at 1 min to avoid divide-by-~0
            geo_speed = min(jump_km / hours, 5000.0)

        hist.append((ts, tx["amount"], lat, lon))
        return {"velocity": velocity, "amount_z": round(amount_z, 2),
                "geo_speed": round(geo_speed, 1), "merchant_risk": CATEGORY_RISK.get(tx["merchant_category"], 2),
                "jump_km": round(jump_km), "last_gap_s": None}
