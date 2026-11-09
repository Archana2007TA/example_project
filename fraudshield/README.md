# FraudShield - Real-Time Fraud Anomaly Detection

Event-driven ML pipeline + live analyst dashboard (FastAPI, WebSockets, Isolation Forest).

## Run it (5 minutes)
```bash
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
python train_model.py                               # creates model.pkl (optional, app auto-trains)
uvicorn app:app --reload
```
Open http://localhost:8000 - the stream starts automatically. Use the "Simulate" buttons to trigger fraud on demand.

## Files
| File | Role |
|---|---|
| `core.py` | Feature engineering (velocity, amount z-score, geo-jump speed, merchant risk). Shared by training and live API |
| `train_model.py` | Synthetic data + 2% injected fraud, trains IsolationForest, saves `model.pkl` |
| `app.py` | FastAPI: `POST /api/v1/transaction`, `WS /ws/live-feed`, built-in transaction simulator |
| `static/index.html` | Dashboard: ticker, alert console with Block/Flag actions, line chart, risk gauge, latency |

## Test the API like a real bank would
```bash
curl -X POST localhost:8000/api/v1/transaction -H "Content-Type: application/json" \
 -d '{"user_id":"user_001","amount":8500,"merchant_category":"crypto","location":{"lat":51.5,"lon":-0.12}}'
```

## 3-day plan
- **Day 1**: run `train_model.py`, tweak `make_data()`, plot score distributions for your slides.
- **Day 2**: replace the simulator with Kafka (`aiokafka` consumer calling `ingest()`), add SQLite for history.
- **Day 3**: polish dashboard, rehearse demo, record a backup video.

## Demo script (3 min)
1. Show stream flowing, all green. 2. Click "impossible travel": alert flashes with reasons. 3. Click "Block card". 4. Point at the latency counter. 5. Explain the architecture diagram and why Isolation Forest needs no labels.
