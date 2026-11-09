"""Step 3: FastAPI engine - REST ingestion, scoring, WebSocket broadcast, built-in simulator."""
import asyncio, os, random, time, uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import joblib, numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from core import CATEGORY_RISK, CITIES, FEATURES, FeatureEngine
from train_model import train

model = joblib.load("model.pkl") if os.path.exists("model.pkl") else train()
engine = FeatureEngine()
clients: set[WebSocket] = set()
sim = {"running": True, "clock": datetime.now(timezone.utc)}


class Location(BaseModel):
    lat: float
    lon: float


class Transaction(BaseModel):
    transaction_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    amount: float
    merchant_category: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    location: Location


def score(tx: dict) -> dict:
    t0 = time.perf_counter()
    f = engine.compute(tx)
    x = np.array([[f[k] for k in FEATURES]])
    df = float(model.decision_function(x)[0])
    model_flag = int(model.predict(x)[0]) == -1

    reasons = []
    if f["velocity"] > 5: reasons.append(f"Velocity spike: {f['velocity']} txns in 5 min")
    if f["geo_speed"] > 900: reasons.append(f"Impossible travel: {f['jump_km']:,} km jump ({f['geo_speed']:,.0f} km/h)")
    if f["amount_z"] > 4: reasons.append(f"Amount {f['amount_z']:.0f}x std-dev above user norm")
    if f["merchant_risk"] >= 4 and tx["amount"] > 1000: reasons.append(f"High-risk merchant ({tx['merchant_category']})")
    if model_flag and not reasons: reasons.append("Model: unusual combination of signals")

    risk = int(np.clip((0.2 - df) / 0.3 * 100, 0, 100))  # 0 = normal, 100 = extreme outlier
    return {**tx, "features": f, "anomaly_score": round(df, 4), "risk": risk,
            "flagged": bool(model_flag or reasons), "reasons": reasons,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}


async def broadcast(msg: dict):
    for ws in list(clients):
        try:
            await ws.send_json(msg)
        except Exception:
            clients.discard(ws)


async def ingest(tx: dict) -> dict:
    result = score(tx)
    t0 = time.perf_counter()
    await broadcast({"type": "tx", "data": result})
    result["latency_ms"] = round(result["latency_ms"] + (time.perf_counter() - t0) * 1000, 2)  # scored + broadcast
    return result


# ---------- simulator (stands in for Kafka / Kinesis) ----------
USERS = {}
for i in range(40):
    city = random.choice(list(CITIES))
    USERS[f"user_{i:03d}"] = {"city": city, "avg": random.choice([15, 25, 40, 60, 90])}


def make_tx(uid, amount=None, cat=None, city=None, gap=None):
    u = USERS[uid]
    sim["clock"] += timedelta(seconds=gap if gap is not None else random.randint(15, 60))
    lat, lon = CITIES[city or u["city"]]
    cat = cat or random.choices(["grocery", "restaurant", "retail", "travel", "gambling", "wire_transfer", "crypto"],
                                [30, 25, 28, 8, 4, 3, 2])[0]
    amount = amount if amount is not None else float(np.random.lognormal(np.log(u["avg"]), 0.4))
    return {"transaction_id": str(uuid.uuid4()), "user_id": uid, "amount": round(amount, 2),
            "merchant_category": cat, "timestamp": sim["clock"].isoformat(),
            "location": {"lat": lat + random.uniform(-.05, .05), "lon": lon + random.uniform(-.05, .05)}}


async def attack(kind: str):
    uid = random.choice(list(USERS))
    for _ in range(3):  # a few normal txns first so the user has a profile
        await ingest(make_tx(uid))
    if kind == "geo":
        far = random.choice([c for c in CITIES if c != USERS[uid]["city"]])
        await ingest(make_tx(uid, amount=USERS[uid]["avg"] * 3, cat="retail", city=far, gap=120))
    elif kind == "amount":
        await ingest(make_tx(uid, amount=random.randint(4000, 9000), cat="crypto"))
    else:  # velocity burst
        for _ in range(7):
            await ingest(make_tx(uid, amount=random.randint(30, 200), cat="gambling", gap=3))
            await asyncio.sleep(0.12)


async def simulator():
    while True:
        if sim["running"]:
            if random.random() < 0.05:
                await attack(random.choice(["geo", "amount", "velocity"]))
            else:
                await ingest(make_tx(random.choice(list(USERS))))
        await asyncio.sleep(0.6)


@asynccontextmanager
async def lifespan(app):
    task = asyncio.create_task(simulator())
    yield
    task.cancel()


app = FastAPI(title="FraudShield", lifespan=lifespan)


@app.post("/api/v1/transaction")
async def post_transaction(tx: Transaction):
    return await ingest(tx.model_dump())


@app.post("/api/v1/simulate/attack/{kind}")
async def trigger_attack(kind: str):
    asyncio.create_task(attack(kind))
    return {"ok": True, "kind": kind}


@app.post("/api/v1/simulate/{state}")
async def toggle(state: str):
    sim["running"] = state == "start"
    return sim["running"]


@app.websocket("/ws/live-feed")
async def live_feed(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        clients.discard(ws)


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")
