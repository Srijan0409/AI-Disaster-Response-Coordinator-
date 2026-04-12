---
title: "Disaster Env"
emoji: "🚑"
colorFrom: "blue"
colorTo: "green"
sdk: "docker"
app_port: 8000
---

# AI Disaster Response Coordinator

> Meta PyTorch x Scaler School of Technology - OpenEnv Hackathon Round 1

An OpenEnv-compatible reinforcement learning environment simulating **real-world disaster response coordination**, inspired by the 2013 Uttarakhand floods and Kedarnath disaster — one of India's most severe natural disasters.

This environment models how emergency authorities (NDRF/SDRF) must allocate limited resources across multiple disaster zones under time pressure.

---

##  Environment Description

The environment simulates real-world emergency triage decision-making:

* Multiple disaster zones with varying severity
* Limited rescue resources (ambulances, rescue teams, helicopters)
* Victim survival windows (delayed rescue leads to death)
* Spreading disasters across zones
* Dynamic victim spawning in harder scenarios

Agents must make **high-stakes prioritization decisions** to maximize survival outcomes.

---

##  OpenEnv Compliance

This environment fully implements the OpenEnv specification:

* `reset()` → Initializes environment state
* `step(action)` → Returns `(state, reward, done, info)`
* `state()` → Returns current observable state
* `openenv.yaml` → Defines API schema and environment contract

---

## Task Descriptions

| Task                | Difficulty | Zones | Victims | Steps | Spread        | Spawn         | Success Threshold |
| ------------------- | ---------- | ----- | ------- | ----- | ------------- | ------------- | ----------------- |
| task1_easy_rescue   | Easy       | 1     | 7       | 30    | No            | No            | 0.50              |
| task2_medium_rescue | Medium     | 3     | 8/zone  | 40    | Every 5 steps | No            | 0.60              |
| task3_hard_rescue   | Hard       | 5     | 8/zone  | 25    | Every 2 steps | Every 5 steps | 0.70              |

---

##  Action Space

```python
DisasterAction(
    zone_id: int,
    unit_type: str  # ambulance | rescue_team | helicopter
)
```

---

##  Observation Space

Each state includes:

* `zone_id`
* `severity_level`
* `active_victims`
* `remaining_survival_time`
* `available_resources`
* `disaster_spread_status`

---

## Reward Function

The reward is carefully designed to reflect real-world priorities and is **strictly bounded between (0, 1)**:

* +0.10 → Successful rescue
* +0.05 → Prioritizing high-severity zones
* -0.05 → Victim death due to delay
* -0.02 → Inefficient resource allocation

###  Final Score Rule:

> Final score is **strictly between 0 and 1 (not inclusive)** to satisfy evaluation constraints.

---

##  Evaluation / Grader

Each task is evaluated using an automated grader:

* Tracks number of rescued victims
* Penalizes deaths and delayed actions
* Rewards efficient and strategic coordination

The final score is normalized in the range **(0, 1)**.

⚠️ Submissions fail if:

* Score ≤ 0
* Score ≥ 1

---

##  Baseline Agent

A baseline inference script is provided:

```bash
python inference.py
```

Output format:

```
[START] task=<task_name> env=<env_name> model=<model>
[STEP] step=<n> action=<action> reward=<r>
[END] final_score=<score>
```

---

##  Setup

```bash
git clone https://github.com/YOUR_USERNAME/disaster_env
cd disaster_env
pip install -e .
```

---

##  Run Server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

---

##  Docker

```bash
docker build -t disaster_env:latest .
docker run -p 8000:8000 disaster_env:latest
```

---

##  Deployment

This environment is deployed on **Hugging Face Spaces** using Docker.

* Public API Endpoint: `https://huggingface.co/spaces/ayush4650/disaster-env-v2`
* Fully OpenEnv-compatible API

---

##  License

BSD 3-Clause
