# 💧 AI Agent for Water Tank Monitoring System

**An Agentic AI-Based Intelligent Water Management and Monitoring System**

A fully working, deployable demonstration of **Agentic AI**: an AI agent that
continuously monitors water-tank sensor data, analyzes trends, detects
abnormal conditions, makes autonomous decisions, and explains *why* it
reached each decision — all through an attractive, interactive Gradio
dashboard.

---

## 1. Project Description

Real IoT hardware is not required. The project ships with a realistic
**sensor simulator** that can reproduce six operating scenarios (normal,
low water, overflow risk, leakage, high consumption, sensor fault). On top
of that data stream sits a multi-stage **Agentic AI pipeline** that
mirrors how a real autonomous monitoring agent would behave:

```
Input/Sensor Data
      │
      ▼
Monitoring Agent        (ingests + validates readings, keeps history)
      │
      ▼
Analysis Agent           (filling/emptying rate, average consumption, trend)
      │
      ▼
Anomaly Detection Agent  (leakage, overflow, low water, unusual usage, sensor faults)
      │
      ▼
Decision Agent           (rule-based reasoning → status + explanation)
      │
      ▼
Notification/Action Agent (alerts, recommended actions, "why" explanation)
      │
      ▼
Monitoring History        (table, graph, downloadable report)
```

The whole pipeline runs **without any paid API** — the Decision Agent is
rule-based so the project works fully offline. It is structured so an LLM
could later be plugged into `agent.DecisionAgent` for richer, natural
language reasoning (see *Future Improvements*).

---

## 2. Features

- 💧 Real-time water level, consumption, inflow and outflow monitoring
- 📊 Interactive Plotly graph of water level over time, with overflow/low
  thresholds marked
- 🤖 Agentic AI panel that explains its detected condition, its reasoning,
  and its recommended action in plain language
- ⚠️ Color-coded alert boxes (Normal / Warning / Critical / Recommendation)
- 🧪 Six built-in demonstration scenarios, including a dedicated
  **leakage-detection** scenario
- ▶️ "Start/Stop Simulation" live auto-run mode (updates every few seconds)
- 🕑 Monitoring history table of every reading and the AI decision made
- 📄 One-click summary report, downloadable as **CSV** (raw history) and
  **TXT** (summary statistics)
- 🧹 Reset button to clear history and start a fresh demo
- 🛡️ Friendly error handling for invalid/negative/out-of-range sensor input
  — the UI never crashes on bad data, it raises a sensor-abnormality alert
- 🎨 Fully custom HTML/CSS dashboard (cards, tank visual, gradients, status
  colors) built on top of Gradio Blocks — not a default form UI

---

## 3. Architecture

```text
water-tank-ai-agent/
│
├── app.py                 # Gradio Blocks UI — wires everything together
├── agent.py                # Decision Agent + Notification/Action Agent +
│                            #   WaterTankAgent orchestrator (the full pipeline)
├── monitoring.py            # Monitoring Agent (ingestion/validation/history)
│                            #   + Analysis Agent (rates, trends, averages)
├── anomaly_detection.py     # Anomaly Detection Agent (rule-based checks)
├── simulation.py            # Simulated sensor system / TankSimulator
├── report_generator.py      # Summary statistics + CSV/TXT export helpers
├── requirements.txt
├── README.md
└── assets/
    └── style.css            # Custom dashboard styling
```

### Agent responsibilities

| Module | Class | Responsibility |
|---|---|---|
| `simulation.py` | `TankSimulator` | Produces realistic `SensorReading`s for each scenario |
| `monitoring.py` | `MonitoringAgent` | Validates and stores readings in a bounded history |
| `monitoring.py` | `AnalysisAgent` | Computes fill/drain rate, rolling average consumption, min/max level |
| `anomaly_detection.py` | `AnomalyDetectionAgent` | Flags overflow risk, low water, leakage, rapid drops, high consumption, sensor faults |
| `agent.py` | `DecisionAgent` | Turns anomalies into a status + human-readable reasoning + recommended actions |
| `agent.py` | `WaterTankAgent` | Orchestrates the full pipeline each cycle (`run_cycle()`) |
| `report_generator.py` | — | Builds the downloadable summary/CSV report |

---

## 4. Installation

```bash
git clone <this-repo-url>
cd water-tank-ai-agent
pip install -r requirements.txt
```

Requires Python 3.9+.

---

## 5. How to Run (locally)

```bash
python app.py
```

Gradio will start a local web server (by default at
`http://127.0.0.1:7860`) and print the URL in the terminal — open it in
your browser.

---

## 6. How to Deploy (Hugging Face Spaces)

1. Create a new Space on Hugging Face and choose the **Gradio** SDK.
2. Upload (or push via git) all project files: `app.py`, `agent.py`,
   `monitoring.py`, `anomaly_detection.py`, `simulation.py`,
   `report_generator.py`, `requirements.txt`, `README.md`, and the
   `assets/` folder.
3. Hugging Face Spaces automatically installs `requirements.txt` and runs
   `app.py` — no further configuration is required.
4. (Optional) If you later connect a real LLM API, add the key as a
   **Space secret** (e.g. `ANTHROPIC_API_KEY`) and read it in code with
   `os.environ.get("ANTHROPIC_API_KEY")` — never hard-code API keys.

---

## 7. Example Usage

1. Open the app. The dashboard loads with a default tank (5000 L, 55%
   level, normal conditions).
2. In the sidebar, pick the **"leakage"** scenario from the *Simulation
   Scenario* dropdown.
3. Click **🤖 Run AI Agent** a few times (or click **▶️ Start Simulation**
   to let it run automatically).
4. Watch the **AI Agent Status** panel turn 🔴 and explain:
   > *"Water level dropped by X% in the last interval... Recorded outflow
   > is much higher than recorded consumption, so the drop is not fully
   > explained by normal usage. Recommended action: inspect the tank,
   > outlet pipes and valves for leakage."*
5. Click **📊 Generate Report** to get a summary and download the full
   monitoring history as CSV.
6. Click **🧹 Reset** to clear history and try another scenario (e.g.
   `overflow_risk` or `sensor_fault`).

---

## 8. Future Improvements

- Plug a real LLM (e.g. via the Anthropic API) into `DecisionAgent.decide()`
  for richer natural-language reasoning on top of the existing rule-based
  signals, using an optional environment-variable API key.
- Connect to real IoT hardware (ultrasonic/pressure water-level sensors,
  flow meters) in place of `TankSimulator`.
- Add push notifications (email/SMS) from the Notification/Action Agent.
- Multi-tank support with a tank selector.
- Persistent storage (database) for long-term monitoring history instead
  of the in-memory session history.

---

## 9. Notes

- The application does **not** require any paid API to demonstrate its
  core functionality — the Decision Agent is rule-based by design.
- All sensor data shown is **simulated** for demonstration purposes.
