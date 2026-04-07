---
title: AI Disaster Response Coordinator
emoji: 🚨
colorFrom: pink
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# AI Disaster Response Coordinator

A reinforcement learning environment that simulates emergency disaster triage across real Uttarakhand locations. An AI agent allocates limited rescue teams to victims with varying urgency levels, distances, and survival times — under strict time pressure.

## Quick Start

The simplest way to use the environment is through the `DisasterEnv` class:

```python
from disaster_env import DisasterAction, DisasterEnv

with DisasterEnv(base_url="http://localhost:8000") as env:
    # Reset with difficulty and seed
    result = env.reset(difficulty="medium", seed=42)
    obs = result.observation

    print(f"Victims: {len(obs.victims)}")
    print(f"Resources available: {obs.resources_available}")

    # Pick the most critical alive victim
    alive = [v for v in obs.victims if v.alive and not v.rescued]
    target = max(alive, key=lambda v: v.urgency)

    # Rescue them
    result = env.step(DisasterAction(allocate_to=target.id))
    print(f"Reward: {result.reward}")
    print(f"Done: {result.done}")
```

## Building the Docker Image

Before using the environment locally, build the Docker image:

```bash
# From project root
docker build -t disaster_env-env:latest .
```

## Deploying to Hugging Face Spaces

Deploy your environment to Hugging Face Spaces using the `openenv push` command:

```bash
# From the environment directory (where openenv.yaml is located)
openenv push

# Or specify options
openenv push --namespace my-org --private
```

The `openenv push` command will:
1. Validate that the directory is an OpenEnv environment (checks for `openenv.yaml`)
2. Prepare a custom build for Hugging Face Docker space (enables web interface)
3. Upload to Hugging Face (ensuring you're logged in)

### Prerequisites

- Authenticate with Hugging Face: The command will prompt for login if not already authenticated

### Options

- `--directory`, `-d`: Directory containing the OpenEnv environment (defaults to current directory)
- `--repo-id`, `-r`: Repository ID in format `username/repo-name` (defaults to `username/env-name` from openenv.yaml)
- `--base-image`, `-b`: Base Docker image to use (overrides Dockerfile FROM)
- `--private`: Deploy the space as private (default: public)

### Examples

```bash
# Push to your personal namespace
openenv push

# Push to a specific repository
openenv push --repo-id my-org/disaster-env

# Push as a private space
openenv push --private
```

After deployment, your space will be available at:
`https://huggingface.co/spaces/<repo-id>`

The deployed space includes:
- **Web Interface** at `/web` — Interactive UI for exploring the environment
- **API Documentation** at `/docs` — Full OpenAPI/Swagger interface
- **Health Check** at `/health` — Container health monitoring
- **WebSocket** at `/ws` — Persistent session endpoint for low-latency interactions

## Environment Details

### Action

**DisasterAction**: One field controls the agent's decision each step.
- `allocate_to` (int) — Victim ID to send a rescue team to. Use `-1` to skip this step (incurs a small penalty).

### Observation

**DisasterObservation**: Full state of the disaster scenario returned after every `reset()` and `step()`.
- `time_step` (int) — Current step number (0-indexed)
- `resources_available` (int) — Rescue teams free this step
- `max_steps` (int) — Total steps allowed this episode
- `difficulty` (str) — Episode difficulty: `easy` | `medium` | `hard`
- `victims` (list[VictimState]) — All victims and their current status
- `episode_done` (bool) — Whether the episode has terminated
- `last_action_info` (dict) — Info from the most recent step (rescued, died, invalid, reward components)

### VictimState

Each victim in the scenario has:
- `id` (int) — Unique victim identifier
- `urgency` (int) — `1`=low, `2`=medium, `3`=critical
- `distance` (float) — Distance from rescue base in km (1.0–10.0)
- `survival_time` (int) — Steps remaining before this victim dies if not rescued
- `alive` (bool) — Whether the victim is still alive
- `rescued` (bool) — Whether the victim has been rescued

### Reward

The reward function is dense — a shaped signal is returned every step:

| Component | Value |
|---|---|
| Rescue urgency=3 (critical) | +0.85 |
| Rescue urgency=2 (medium) | +0.60 |
| Rescue urgency=1 (low) | +0.30 |
| Criticality bonus (rescuing just-in-time) | up to +0.20 |
| Efficiency bonus (closer victim) | small positive |
| Skip penalty (allocate_to=-1) | −0.08 |
| Invalid action penalty | −0.12 |
| Death penalty — urgency=3 | −0.85 |
| Death penalty — urgency=2 | −0.55 |
| Death penalty — urgency=1 | −0.25 |
| Pressure penalty (urgent victims at risk) | up to −0.18 |

Final reward per step is clamped to `[0.0, 1.0]`.

## Difficulty Levels

### Easy
- 4 victims, mostly low urgency
- Survival times: 6–12 steps
- Distances: 1–5 km
- 12 max steps, time decay: 1/step

### Medium
- 5 victims, balanced urgency distribution
- Survival times: 4–9 steps
- Distances: 1–8 km
- 10 max steps, time decay: 1/step

### Hard
- 7 victims, mostly critical urgency
- Survival times: 2–6 steps
- Distances: 1–10 km
- 8 max steps, time decay: 2/step (survival drops faster)

## Advanced Usage

### Connecting to an Existing Server

```python
from disaster_env import DisasterAction, DisasterEnv

# Connect to a deployed HF Space or local server
env = DisasterEnv(base_url="https://YOUR-USERNAME-disaster-env.hf.space")

result = env.reset(difficulty="hard", seed=999)
obs = result.observation

result = env.step(DisasterAction(allocate_to=0))
print(f"Reward: {result.reward}")

env.close()
```

### Using the Context Manager

```python
from disaster_env import DisasterAction, DisasterEnv

with DisasterEnv(base_url="http://localhost:8000") as env:
    result = env.reset(difficulty="easy", seed=42)
    obs = result.observation

    while not result.done:
        # Greedy: rescue highest urgency alive victim
        alive = [v for v in obs.victims if v.alive and not v.rescued]
        if not alive:
            break
        target = max(alive, key=lambda v: (v.urgency, -v.survival_time))
        result = env.step(DisasterAction(allocate_to=target.id))
        obs = result.observation
        print(f"Step {obs.time_step} — reward: {result.reward:.2f}")
```

### Concurrent WebSocket Sessions

The server supports multiple concurrent WebSocket connections. To enable this,
modify `server/app.py` to use factory mode:

```python
# In server/app.py — use factory mode for concurrent sessions
app = create_app(
    DisasterEnvironment,   # Pass class, not instance
    DisasterAction,
    DisasterObservation,
    max_concurrent_envs=4,  # Allow 4 concurrent sessions
)
```

## Running the Inference Script

```bash
# Set environment variables
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export HF_TOKEN=hf_xxxxxxxxxxxx
export ENV_URL=https://YOUR-USERNAME-disaster-env.hf.space

# Run
python inference.py
```

Expected stdout format:
```
[START] task=task1_nearest_first env=disaster_env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=allocate_to=2 reward=0.85 done=false error=null
[STEP] step=2 action=allocate_to=0 reward=0.62 done=false error=null
...
[END] success=true steps=7 score=0.743 rewards=0.85,0.62,...
```

## Development & Testing

### Run Tests

```bash
# From the server directory
cd server
python -m pytest tests/ -v
```

Test coverage includes:
- Grid world mechanics (zone count, severity ranges, spread, spawning)
- Grader scoring (rescue score, time penalty, wait penalty)
- Task configuration (easy/medium/hard)
- Constants validation (zones, resources, step limits)
- Generator reproducibility (same seed → same scenario)

### Test the Environment Directly

```bash
# Test core environment logic without the HTTP server
python3 server/disaster_env_environment.py
```

### Run the Server Locally

```bash
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

## Project Structure

```
disaster_env/
├── __init__.py                          # Module exports
├── README.md                            # This file
├── openenv.yaml                         # OpenEnv manifest
├── pyproject.toml                       # Project metadata and dependencies
├── inference.py                         # Baseline LLM inference script
├── client.py                            # DisasterEnv client
├── models.py                            # Action and Observation models
└── server/
    ├── __init__.py                      # Server module exports
    ├── app.py                           # FastAPI application (HTTP + WebSocket)
    ├── disaster_env_environment.py      # Core RL environment logic
    ├── constants.py                     # Shared zone/difficulty configuration
    ├── generators.py                    # Scenario factory (civilians + resources)
    ├── grid.py                          # Multi-zone GridWorld simulation
    ├── grader.py                        # Episode scoring (0.0–1.0)
    ├── tasks.py                         # Task definitions (easy/medium/hard)
    ├── requirements.txt                 # Server dependencies
    └── tests/
        ├── test_constants.py
        ├── test_generators.py
        ├── test_grader.py
        ├── test_grid.py
        └── test_tasks.py
```
