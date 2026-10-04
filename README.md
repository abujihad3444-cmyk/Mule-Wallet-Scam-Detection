# 🛡️ Mule Wallet Detection Prototype

Graph-Based Multi-Hop Mule Wallet & Scam Detection System — hackathon-ready Streamlit prototype using synthetic MFS transaction data.

## Windows + VS Code setup

Open the project folder in VS Code, then open **Terminal → New Terminal**.

Create a virtual environment:

```powershell
py -3.14 -m venv .venv
```

If PowerShell blocks activation, you do **not** need to activate it. Use the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Validate the backend:

```powershell
.\.venv\Scripts\python.exe run_check.py
```

Run the dashboard:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

## What the prototype detects

- Multi-hop temporal wallet paths (2–4 hops)
- Rapid transfers
- High-retention transfer chains
- Fan-out and fan-in behavior
- Rapid pass-through behavior
- Structuring/splitting candidates
- Burst activity
- Evidence-qualified cyclic money flows
- Isolation Forest anomalies
- Explainable 0–100 risk scores
- Wallet-level investigation summaries
- Interactive transaction graph
- Optional English/Bangla GenAI investigation assistant

## Important analytical safeguards

### Temporal path validation
A path such as `A → B → C → D` is valid only when the transaction timestamps are non-decreasing in **edge order**. The detector no longer sorts timestamps after the path is found.

### No static-shortest-path assumption
The multi-hop detector does not rely only on a static NetworkX shortest path. It enumerates time-consistent transaction paths so a random graph shortcut cannot hide the actual injected chain.

### Cycle filtering
Static dense transaction graphs can contain many ordinary graph cycles. The dashboard therefore shows only short cycles that also pass transaction-evidence filters: compact time window and meaningful edge amounts.

### Bounded search
Temporal path and cycle searches are capped so the demo remains responsive as the synthetic dataset grows.

### Explainable risk
The overall risk score is a weighted prioritization signal made from separate components such as anomaly, multi-hop exposure, velocity, fan behavior, pass-through, cycle participation, and connectivity. It is not a fraud probability or final fraud decision.

## Synthetic scenarios included

1. Multi-hop mule chain: `W001 → W005 → W012 → W019 → W027 → W035`
2. Fan-out from `W050`
3. Fan-in into `W055`
4. Cyclic flow: `W060 → W061 → W062 → W063 → W060`
5. Rapid pass-through around `W080`
6. Structuring/splitting around `W085`
7. Burst activity around `W099`

## Optional GenAI

The application works without an API key.

To enable OpenAI-generated investigation summaries, set:

```text
OPENAI_API_KEY=your_key
OPENAI_MODEL=gpt-4.1-mini
```

The prompt instructs the model to use only structured evidence supplied by the system and not invent transaction facts.

## Demo flow for judges

1. Start at **Risk Dashboard**.
2. Open **Transaction Graph**.
3. Open **Wallet Investigation** and inspect a high-risk wallet.
4. Open **Multi-Hop Analysis** and show the detected temporal chain.
5. Open **AI Assistant** and generate the evidence-based investigation summary.

## Scope

This is a synthetic-data research/hackathon prototype. It is intended to demonstrate graph-based fraud-intelligence concepts and should not be used as a real-world fraud verdict or automated enforcement system.
