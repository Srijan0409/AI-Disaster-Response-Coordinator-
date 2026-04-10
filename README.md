---
title: Disaster Env
emoji: 🚑
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
---

# AI Disaster Response Coordinator

> Meta PyTorch x Scaler School of Technology - OpenEnv Hackathon Round 1

An OpenEnv-compatible reinforcement learning environment simulating real-world multi-zone disaster response coordination. Based on the 2013 Uttarakhand floods and Kedarnath disaster scenario - one of India's most severe natural disasters.

An AI agent coordinates ambulances, rescue teams, and helicopters across multiple disaster zones, triaging victims under time pressure, spreading threats, and dynamic victim spawning.

---

## Environment Description

The environment models emergency triage decision-making - a task real disaster response coordinators perform under severe constraints:

* Multiple disaster zones with different severities, victim counts, and disaster types
* Limited rescue resources (ambulances, rescue teams, helicopters)
* Spreading threats - high-severity zones infect adjacent zones over time
* Victim survival decay - victims die if not rescued within their survival window
* Hard triage decisions - not all victims can be saved; prioritisation matters

---

## Task Descriptions

| Task                | Difficulty | Zones | Victims | Steps | Spread        | Spawn         | Success Threshold |
| ------------------- | ---------- | ----- | ------- | ----- | ------------- | ------------- | ----------------- |
| task1_easy_rescue   | Easy       | 1     | 7       | 30    | No            | No            | 0.50              |
| task2_medium_rescue | Medium     | 3     | 8/zone  | 40    | Every 5 steps | No            | 0.60              |
| task3_hard_rescue   | Hard       | 5     | 8/zone  | 25    | Every 2 steps | Every 5 steps | 0.70              |

---

## Action Space

DisasterAction(
zone_id: int,
unit_type: str  # ambulance | rescue_team | helicopter
)

---

## Setup

git clone https://github.com/YOUR_USERNAME/disaster_env
cd disaster_env
pip install -e .

---

## Run

uvicorn server.app:app --host 0.0.0.0 --port 8000

---

## Docker

docker build -t disaster_env:latest .
docker run -p 8000:8000 disaster_env:latest

---

## License

BSD 3-Clause
