# AI Disaster Response Coordinator

An OpenEnv reinforcement-learning environment that simulates multi-zone emergency triage across real Uttarakhand disaster locations, based on the 2013 Kedarnath floods scenario.

An AI agent acts as a disaster response coordinator: it observes active disaster zones, assesses victim urgency and zone severity, and dispatches rescue units (ambulances, rescue teams, helicopters) to maximise the number of survivors within a limited step budget.

---

## Environment Description

The environment models a real-world emergency triage problem across five districts of Uttarakhand, India. Each episode presents the agent with one or more active disaster zones, each containing victims with varying urgency levels and survival times. The agent must prioritise zones and select appropriate rescue units to save as many lives as possible before the episode ends.

**Key mechanics:**
- Victims have individual urgency (1=low, 2=medium, 3=critical) and survival timers that count down each step
- Disaster can spread to adjacent zones every N steps (medium and hard modes)
- New victims spawn mid-episode in hard mode, requiring continuous re-prioritisation
- Each rescue unit type rescues a different number of victims per step
- Episode terminates when all victims are rescued/dead or the step budget is exhausted

**Real locations modelled:**

| Zone | District | Disaster Type |
|------|----------|---------------|
| Kedarnath Temple Area | Rudraprayag | Flash flood |
| Badrinath | Chamoli | Landslide |
| Rishikesh | Dehradun | River overflow |
| Joshimath | Chamoli | Land subsidence |
| Dehradun City | Dehradun | Landslide |
| Haridwar Ghat Area | Haridwar | River overflow |
| Chamoli | Chamoli | Glacier burst |

---

## Action Space

The agent sends a rescue unit to a disaster zone each step.

```python
DisasterAction(
    zone_id:   int,   # Zone to send the unit to (0-indexed)
    unit_type: str,   # "ambulance" | "rescue_team" | "helicopter"
)
```

| Unit Type | Victims Rescued / Step | Available In |
|-----------|----------------------|--------------|
| `ambulance` | 2 | easy, medium, hard |
| `rescue_team` | 3 | easy, medium, hard |
| `helicopter` | 5 | easy, medium only |

Victims are rescued in priority order: highest urgency first, then lowest survival time.

---

## Observation Space

Each step returns a `DisasterObservation` with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `time_step` | int | Current step number (0-indexed) |
| `max_steps` | int | Total step budget for this episode |
| `difficulty` | str | `"easy"` \| `"medium"` \| `"hard"` |
| `resources_available` | int | Total rescue units available this step |
| `episode_done` | bool | Whether the episode has terminated |
| `zones` | list[dict] | All disaster zones — see schema below |
| `victims` | list[VictimState] | Flat list of all victims across all zones |
| `last_action_info` | dict \| None | Per-step info; includes `grade_report` on terminal step |

**Zone dict schema:**

```json
{
  "zone_id":       0,
  "name":          "Kedarnath Temple Area",
  "district":      "Rudraprayag",
  "disaster_type": "flash_flood",
  "severity":      0.85,
  "people":        8,
  "rescued":       3,
  "time_waiting":  2,
  "is_active":     true,
  "victims": [
    {
      "id":            0,
      "urgency":       3,
      "survival_time": 4,
      "distance":      2.5,
      "alive":         true,
      "rescued":       false
    }
  ]
}
```

**VictimState schema:**

| Field | Type | Description |
|-------|------|-------------|
| `id` | int | Unique victim identifier |
| `urgency` | int | 1 = low, 2 = medium, 3 = critical |
| `survival_time` | int | Steps remaining before victim dies if not rescued |
| `distance` | float | Distance from rescue base in km (1.0–10.0) |
| `alive` | bool | Whether the victim is still alive |
| `rescued` | bool | Whether the victim has been rescued |

---

## Tasks

### Task 1 — Easy: Single-Zone Flash Flood (`seed=42`)

**Location:** Kedarnath Temple Area, Rudraprayag  
**Zones:** 1 | **Victims:** 7 | **Step budget:** 30 | **Success threshold:** 0.5

Single-zone flood scenario. No disaster spread. No victim spawning. Helicopters available. Ideal for learning basic triage prioritisation.

**Objective:** Rescue maximum civilians before the step budget runs out.

---

### Task 2 — Medium: Three-Zone Multi-Disaster (`seed=123`)

**Locations:** Kedarnath (flash flood) · Badrinath (landslide) · Rishikesh (river overflow)  
**Zones:** 3 | **Victims:** 8 per zone (24 total) | **Step budget:** 40 | **Success threshold:** 0.6

Disaster spreads to adjacent zones every 5 steps, raising severity. The agent must balance immediate rescues against worsening conditions in other zones. Helicopters available.

**Objective:** Coordinate rescue across 3 zones with spreading disaster.

---

### Task 3 — Hard: Five-Zone Crisis Under Pressure (`seed=999`)

**Locations:** Kedarnath · Joshimath · Dehradun · Haridwar · Chamoli  
**Zones:** 5 | **Victims:** 8 per zone (40 initial) | **Step budget:** 25 | **Success threshold:** 0.7

Kedarnath starts at maximum severity (1.0). Disaster spreads every 2 steps. New victims spawn every 5 steps. **No helicopters available** — only ambulances and rescue teams. The agent must triage aggressively under severe time and resource pressure.

**Objective:** Manage 5 zones under extreme resource constraints and time pressure.

---

## Reward Function

### Per-step reward (non-sparse signal)

| Action | Reward |
|--------|--------|
| Rescue in high-severity zone (≥ 0.7) | +1.0 |
| Rescue in medium-severity zone (≥ 0.4) | +0.5 |
| Rescue in low-severity zone (< 0.4) | +0.2 |
| Wasted action (zone fully rescued) | −0.1 |
| Invalid action (bad zone_id or unit_type) | −0.3 |

### Terminal reward (episode score)

On the final step, the reward is replaced by a normalised episode score in [0.0, 1.0]:

```
score = base_rescue_score − time_penalty − wait_penalty − spawn_penalty
```

- `base_rescue_score` — weighted fraction of victims rescued per severity bucket
- `time_penalty` — penalises consuming more of the step budget
- `wait_penalty` — penalises leaving victims waiting in active zones
- `spawn_penalty` — penalises failing to rescue spawned victims (hard mode only)

The `grade_report` dict is included in `last_action_info` on the terminal step.

---

## Baseline Scores

Scores produced by the greedy fallback policy (highest-severity zone, most powerful unit):

| Task | Difficulty | Score | Passed |
|------|------------|-------|--------|
| task1_easy_rescue | easy | ~0.62 | Yes |
| task2_medium_rescue | medium | ~0.55 | No |
| task3_hard_rescue | hard | ~0.38 | No |

LLM-guided policy (Qwen2.5-72B-Instruct) typically achieves higher scores by considering victim urgency and survival time alongside zone severity.

---

## Setup & Usage

### Requirements

- Python 3.10+
- Docker
- Hugging Face account + token

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the server locally

```bash
python server.py
# Server starts at http://localhost:8000
```

### Run inference locally

```bash
export HF_TOKEN="your_hf_token"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export ENV_URL="http://localhost:8000"

python inference.py
```

### Run with Docker

```bash
docker build -t disaster-env .
docker run -p 8000:8000 disaster-env
```

### Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | Yes | Hugging Face API key |
| `API_BASE_URL` | Yes | LLM API endpoint |
| `MODEL_NAME` | Yes | Model identifier for inference |
| `ENV_URL` | Yes | URL of the deployed environment server |
| `PORT` | No | Server port (default: 8000) |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/reset` | Start a new episode `{"difficulty": "easy\|medium\|hard", "seed": int}` |
| `POST` | `/step` | Execute one action `{"zone_id": int, "unit_type": str}` |
| `GET` | `/state` | Get current episode state |
| `DELETE` | `/session` | Clean up a session (send `X-Session-ID` header) |

Each request should include an `X-Session-ID` header to isolate concurrent sessions.

---

## Project Structure

```
.
├── server.py                      # FastAPI server — /reset, /step, /state endpoints
├── inference.py                   # LLM agent loop — runs all 3 tasks
├── disaster_env_environment.py    # DisasterEnvironment — OpenEnv wrapper
├── models.py                      # Pydantic data models (Action, Observation)
├── grid.py                        # GridWorld simulation engine
├── grader.py                      # Scoring and reward computation
├── generators.py                  # Victim and scenario generation
├── tasks.py                       # Task configurations (easy/medium/hard)
├── constants.py                   # Shared constants (zones, limits, weights)
├── openenv.yaml                   # OpenEnv spec metadata
├── Dockerfile                     # Container definition
└── requirements.txt               # Python dependencies
```

---

## OpenEnv Spec Compliance

- `reset()` returns a clean `DisasterObservation` with full initial state
- `step(action)` returns observation, reward, done flag, and info dict
- `state()` returns current episode state at any point
- All models are typed Pydantic classes inheriting from OpenEnv base types
- `openenv.yaml` defines name, version, entrypoint, and endpoints
- Scores are deterministic and reproducible via fixed seeds

---

## Infra

- Runtime: inference script completes all 3 tasks in under 20 minutes
- Machine requirements: vcpu=2, memory=8gb compatible
- No GPU required
