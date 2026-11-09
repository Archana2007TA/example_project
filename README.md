# example_project
# FraudShield: Real-Time Fraud Anomaly Detection

A machine learning system that scores financial transactions as they happen and pushes fraud alerts to a live analyst dashboard. It combines an unsupervised Isolation Forest model with hard business rules, and streams results over WebSockets.

Built for fintech hackathons to showcase streaming analytics, real-time model deployment and explainable alerts.

## Features

- Live transaction ticker: every scored transaction appears instantly.
- Fraud alert console: flagged transactions flash red, with plain-English reasons (for example, "Impossible travel: 8,000 km jump").
- Human-in-the-loop actions: Block card, Flag user and Mark safe buttons on every alert.
- Real-time analytics: transactions vs. flagged fraud line chart and a System Risk Index gauge.
- Latency counter: shows how long each transaction took to score and broadcast.
- Attack simulator: trigger impossible travel, crypto spike or velocity burst fraud on demand for demos.
- Public REST API: send your own transactions with a single POST request.

## Architecture

```
[ Transaction Simulator / Bank API ]
              |  JSON
              v
     [ FastAPI Ingestion ]
              |
              v
[ Feature Engine + ML Scoring ]
  - Velocity (txns in last 5 min)
  - Amount deviation (z-score vs. user history)
  - Geo-jump speed (km/h between transactions)
  - Merchant risk score
  - Isolation Forest anomaly score
  - Rule engine (hard triggers)
              |
              v
   [ WebSocket /ws/live-feed ]
              |
              v
 [ Live Analyst Dashboard (browser) ]
```

## How detection works

1. Feature engineering (core.py): for each transaction, the engine keeps a sliding window of the user's recent activity in memory and computes four features.
2. Isolation Forest