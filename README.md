# AI Disaster Response Coordinator

> **Meta PyTorch × Scaler School of Technology — OpenEnv Hackathon Round 1**

An OpenEnv-compatible reinforcement learning environment simulating real-world multi-zone disaster response coordination. Based on the 2013 Uttarakhand floods and Kedarnath disaster scenario — one of India's most severe natural disasters.

An AI agent coordinates ambulances, rescue teams, and helicopters across multiple disaster zones, triaging victims under time pressure, spreading threats, and dynamic victim spawning.

---

## Environment Description

The environment models emergency triage decision-making — a task real disaster response coordinators perform under severe constraints:

- **Multiple disaster zones** with different severities, victim counts, and disaster types
- **Limited rescue resources** (ambulances, rescue teams, helicopters)
- **Spreading threats** — high-severity zones infect adjacent zones over time
- **Victim survival decay** — victims die if not rescued within their survival window
- **Hard triage decisions** — not all victims can be saved; prioritisation matters

The agent must decide **which zone to send which unit to** at each step, maximising the weighted rescue score before the step budget runs out.

---

## Real-World Motivation

Disaster response coordination is a genuine, high-stakes task. Every year, natural disasters in India cause mass casualties partly due to suboptimal resource allocation. An AI agent trained on this environment could:

- Learn triage prioritisation under resource constraints
- Generalise to real dispatch decisions with sensor data
- Be evaluated objectively via the grader's severity-weighted rescue score

---

## Task Descriptions

| Task | Difficulty | Zones | Victims | Steps | Spread | Spawn | Success Threshold |
|---|---|---|---|---|---|---|---|
| `task1_easy_rescue` | Easy | 1 | 7 | 30 | No | No | 0.50 |
| `task2_medium_rescue` | Medium | 3 | 8/zone | 40 | Every 5 steps | No | 0.60 |
| `task3_hard_rescue` | Hard | 5 | 8/zone | 25 | Every 2 steps | Every 5 steps | 0.70 |

### Easy — Single-Zone Flash Flood
Kedarnath Temple Area, Rudraprayag. Single zone, 7 victims. Agent has ambulances + rescue team + helicopter. No spread, no spawn. Learn basic triage.

### Medium — Multi-Zone Spreading Disaster
Kedarnath (flash flood) + Badrinath (landslide) + Rishikesh (river overflow). Disaster spreads severity to adjacent zones every 5 steps. Agent must balance cross-zone triage.

### Hard — Five-Zone Crisis
Kedarnath + Joshimath + Dehradun + Haridwar + Chamoli. Kedarnath starts at severity 1.0. Spreads every 2 steps. New victims spawn every 5 steps. **No helicopters.** Extreme time pressure.

---

## Action Space

```python
DisasterAction(
    zone_id: int,        # Target zone (0-indexed)
    unit_type: str,      # "ambulance" | "rescue_team" | "helicopter"
)
```

| Unit | Victims Rescued/Step | Available |
|---|---|---|
| `ambulance` | 2 | All difficulties |
| `rescue_team` | 3 | All difficulties |
| `helicopter` | 5 | Easy + Medium only |

---

## Observation Space

```python
DisasterObservation(
    time_step: int,              # Current step number
    max_steps: int,              # Total step budget
    difficulty: str,             # "easy" | "medium" | "hard"
    resources: dict,             # {"ambulances": n, "rescue_teams": n, "helicopters": n}
    resources_available: int,    # Sum of all resources
    zones: list[dict],           # Full zone state (see below)
    victims: list[VictimState],  # Flat victim list across all zones
    episode_done: bool,          # True when episode has ended
    reward: float,               # Reward from last step
    last_action_info: dict,      # action_valid, rescued_count, episode_reward
                                 # + grade_report on terminal step
)
```

### Zone Dict Structure
```python
{
    "zone_id": int,
    "name": str,              # e.g. "Kedarnath Temple Area"
    "district": str,          # e.g. "Rudraprayag"
    "disaster_type": str,     # e.g. "flash_flood"
    "severity": float,        # 0.0–1.0 (higher = more critical)
    "people": int,            # Total victims in zone
    "rescued": int,           # Victims rescued so far
    "time_waiting": int,      # Steps this zone has been active unrescued
    "is_active": bool,        # True if any victim alive and unrescued
    "victims": list[dict],    # Individual victim records
}
```

### Victim Dict Structure
```python
{
    "id": int,               # Globally unique across all zones
    "urgency": int,          # 1=low, 2=medium, 3=critical
    "survival_time": int,    # Steps before death if unrescued
    "distance_km": float,    # Distance from rescue base
    "alive": bool,
    "rescued": bool,
}
```

---

## Reward Function

### Per-Step Reward
Immediate feedback after each action, normalised to `(0, 1)`:

| Situation | Raw Value | Meaning |
|---|---|---|
| High severity rescue (≥0.7) | +1.0 | Best action |
| Medium severity rescue (≥0.4) | +0.5 | Good action |
| Low severity rescue | +0.2 | Acceptable |
| Wasted step (zone already clear) | -0.1 | Inefficiency |
| Invalid action (bad zone/unit) | -0.3 | Error |

### Terminal Episode Score
Computed at episode end, normalised to `[0.05, 0.95]`:

```
score = normalize(base_rescue_score - time_penalty - wait_penalty - spawn_penalty)
```

**Base rescue score** = weighted fraction of victims rescued per severity bucket:
- High severity (≥0.7): weight 0.60–0.70
- Medium severity (≥0.4): weight 0.20–0.30
- Low severity: weight 0.10

**Penalties:**
- `time_penalty` — fraction of step budget consumed
- `wait_penalty` — average wait time in still-active zones
- `spawn_penalty` — fraction of spawned victims not rescued (hard mode only)

Penalties are capped at 50% of the base score.

---

## Baseline Scores

Scores produced by the greedy LLM agent (`Qwen/Qwen2.5-72B-Instruct`) using the canonical seeds:

| Task | Seed | Score | Passed |
|---|---|---|---|
| Easy | 42 | ~0.72 | ✅ |
| Medium | 123 | ~0.61 | ✅ |
| Hard | 999 | ~0.48 | ❌ |

*Scores are deterministic given the same seed and model.*

---

## Setup & Installation

### Requirements
- Python 3.10+
- Docker (for containerised deployment)
- Hugging Face CLI (for HF Spaces deployment)

### Local Installation

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/disaster_env
cd disaster_env

# Install dependencies
pip install -e .
# or
pip install -r server/requirements.txt
```

### Run Locally

```bash
# Start the environment server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# In another terminal — run the inference baseline
export HF_TOKEN=your_token_here
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct

python inference.py
```

### Docker

```bash
# Build
docker build -t disaster_env:latest .

# Run
docker run --rm -p 8000:8000 \
  -e HF_TOKEN=$HF_TOKEN \
  disaster_env:latest
```

### Run Tests

```bash
pip install pytest
pytest tests/test_environment.py -v
```

---

## Project Structure

```
disaster_env/
├── inference.py              # Baseline inference script (root, required)
├── openenv.yaml              # OpenEnv manifest
├── client.py                 # EnvClient for connecting to the server
├── models.py                 # Pydantic Action + Observation models
├── tests/
│   └── test_environment.py  # Full test suite
└── server/
    ├── app.py                # FastAPI app
    ├── disaster_env_environment.py  # OpenEnv Environment implementation
    ├── grader.py             # Reward + scoring logic
    ├── grid.py               # GridWorld simulation
    ├── generators.py         # Zone + victim generation
    ├── tasks.py              # Task configs (easy/medium/hard)
    ├── constants.py          # Shared constants
    ├── requirements.txt      # Server dependencies
    └── Dockerfile            # Container definition
```

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `API_BASE_URL` | LLM API endpoint | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | Model identifier | `Qwen/Qwen2.5-72B-Instruct` |
| `HF_TOKEN` | Hugging Face / API key | — |
| `ENV_URL` | Override environment server URL | HF Space URL |

---

## Log Format

The inference script emits structured stdout logs:

```
[START] task=task1_easy_rescue env=disaster_env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"zone_id": 0, "unit_type": "helicopter"} reward=0.88 done=false error=null
[STEP] step=2 action={"zone_id": 0, "unit_type": "rescue_team"} reward=0.72 done=false error=null
[END] success=true steps=2 score=0.720 rewards=0.88,0.72
```

---

## OpenEnv API

```python
from disaster_env import DisasterEnv, DisasterAction

async with DisasterEnv(base_url="https://YOUR_SPACE.hf.space") as env:
    # Reset episode
    result = await env.reset(difficulty="medium", seed=123)
    obs = result.observation

    # Step
    result = await env.step(
        DisasterAction(zone_id=0, unit_type="rescue_team")
    )
    print(result.reward)
    print(result.observation.episode_done)
```

---

## License

BSD 3-Clause — see LICENSE file.
