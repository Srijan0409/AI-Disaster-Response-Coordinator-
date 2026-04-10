# =============================================================================
# inference.py — Agent runner for AI Disaster Response Coordinator
# =============================================================================
#
# Runs three evaluation tasks (easy / medium / hard) sequentially.
# Each task:
#   1. Resets the environment with the canonical seed.
#   2. Runs up to MAX_STEPS steps using an LLM-powered agent.
#   3. Falls back to a greedy heuristic if the LLM fails.
#   4. Logs [START], [STEP], [END] lines for the harness.
#
# Design notes
# ------------
# - validate_action() always enforces ALLOWED_UNIT_TYPES implicitly:
#     select_unit() never returns "helicopter" when resources["helicopters"]==0,
#     which is the case for hard mode (RESOURCE_CONFIG["hard"]).
# - greedy_fallback() scores zones by (max_urgency, -min_survival_time,
#     remaining, severity, time_waiting) — a stable tuple sort that matches
#     the triage priority defined in apply_action() (grid.py).
# - score extraction on terminal step reads grade_report["score"] from
#     last_action_info — set by disaster_env_environment.py when done=True.
#     Falls back to mean(rewards) only if grade_report is absent.
# =============================================================================

import asyncio
import json
import os
import textwrap
from typing import List, Optional

from openai import OpenAI
from dotenv import load_dotenv

from disaster_env import DisasterEnv, DisasterAction

load_dotenv()

# ─── Config ──────────────────────────────────────────────────────────────────

IMAGE_NAME   = os.getenv("LOCAL_IMAGE_NAME")
ENV_URL      = os.getenv("ENV_URL", "https://ayush4650-disaster-env-v2.hf.space")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

BENCHMARK               = "disaster_env"
# MAX_STEPS removed — using obs.max_steps from env (easy=30, medium=40, hard=25)
TEMPERATURE             = 0.2
MAX_TOKENS              = 120
SUCCESS_SCORE_THRESHOLD = 0.4

# Canonical seeds match tasks.py TASKS dict exactly
TASKS = [
    ("task1_easy_rescue",   "easy",   42),
    ("task2_medium_rescue", "medium", 123),
    ("task3_hard_rescue",   "hard",   999),
]

# ─── System prompt ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert disaster response coordinator managing rescue operations.

    GOAL: Maximize rescue score by saving as many victims as possible,
    prioritizing high-severity zones and critically injured victims.

    SCORING (what actually matters to your score):
    - High severity zones (severity >= 0.7) carry the highest score weight
    - Medium severity zones (severity >= 0.4) carry medium weight
    - Low severity zones carry the lowest weight
    - Rescuing more victims = higher score
    - Wasting steps on empty zones = step penalty

    TRIAGE PRIORITY (apply in this order):
    1. Choose the zone with HIGHEST severity first.
    2. Tie on severity: choose zone with MOST alive unrescued victims.
    3. Tie on victims: choose zone with LOWEST min survival_time.
    4. Never send a unit to a zone with zero alive unrescued victims.

    UNIT SELECTION (match capacity to remaining victims):
    - 1-2 victims remaining  ->  ambulance   (rescues 2 per step)
    - 3-4 victims remaining  ->  rescue_team (rescues 3 per step)
    - 5+  victims remaining  ->  helicopter  (rescues 5 per step)

    CRITICAL HARD MODE RULE:
    Always check resources["helicopters"] before choosing helicopter.
    If helicopters == 0, use rescue_team or ambulance ONLY.

    Respond ONLY with valid JSON, nothing else:
    {"zone_id": <int>, "unit_type": "<ambulance|rescue_team|helicopter>"}
""").strip()


# ─── Prompt builder ──────────────────────────────────────────────────────────

def build_prompt(obs, step: int) -> str:
    """
    Build the user-turn prompt from the current observation.

    Only active zones (is_active=True) with alive unrescued victims are
    included — inactive zones carry no useful information for the agent.

    obs.resources is a full dict {"ambulances": n, "rescue_teams": n,
    "helicopters": n} (BUG 9 FIX in environment) — passed as-is so the
    LLM knows exactly which unit types are available.
    """
    zones = []
    for z in obs.zones:
        if not z["is_active"]:
            continue

        alive = [
            v for v in z["victims"]
            if v["alive"] and not v["rescued"]
        ]
        if not alive:
            continue

        # Top 3 most critical victims only — reduces prompt size in hard mode
        # Sort: highest urgency first, then lowest survival_time
        top_victims = sorted(
            alive,
            key=lambda v: (-v["urgency"], v["survival_time"])
        )[:3]

        zones.append({
            "zone_id":      z["zone_id"],
            "severity":     z["severity"],
            "time_waiting": z["time_waiting"],
            "remaining":    len(alive),
            "top_victims": [
                {
                    "urgency":       v["urgency"],
                    "survival_time": v["survival_time"],
                }
                for v in top_victims
            ],
        })

    helicopter_warning = (
        "\nWARNING: Hard mode — helicopters=0, do NOT choose helicopter."
        if obs.resources.get("helicopters", 0) == 0 else ""
    )

    return (
        f"Difficulty: {obs.difficulty} | Step {step}/{obs.max_steps}\n"
        f"Resources: {json.dumps(obs.resources)}{helicopter_warning}\n\n"
        f"Active zones:\n{json.dumps(zones, indent=2)}\n\n"
        f"Choose the single best rescue action."
    )


# ─── Logging ─────────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ─── Unit selection ──────────────────────────────────────────────────────────

def select_unit(zone: dict, resources: dict) -> str:
    """
    Choose the best-fit unit type for a zone.

    Matches unit capacity to remaining victim count.
    Never returns "helicopter" when resources["helicopters"] == 0 —
    this implicitly enforces hard-mode helicopter restriction without
    needing to inspect task_level directly.
    """
    remaining = len([
        v for v in zone["victims"]
        if v["alive"] and not v["rescued"]
    ])

    if remaining >= 5 and resources.get("helicopters", 0) > 0:
        return "helicopter"
    elif remaining >= 3 and resources.get("rescue_teams", 0) > 0:
        return "rescue_team"
    else:
        return "ambulance"


# ─── Greedy fallback ─────────────────────────────────────────────────────────

def greedy_fallback(obs) -> DisasterAction:
    """
    Deterministic greedy policy — used when the LLM fails or returns
    an invalid action.

    Scoring tuple (higher = better):
        (severity * 100, remaining * 10, -min_survival_time, max_urgency, time_waiting)

    Matches grader weight order (severity first) AND system prompt priority —
    so LLM and fallback make consistent decisions.
    """
    active = [
        z for z in obs.zones
        if z["is_active"] and any(
            v["alive"] and not v["rescued"] for v in z["victims"]
        )
    ]

    if not active:
        # No rescuable zone — episode should be ending anyway.
        # Pick any zone that exists to avoid hardcoded zone_id=0
        # which may be inactive and cause unnecessary penalty.
        any_zone = obs.zones[0] if obs.zones else None
        fallback_id = any_zone["zone_id"] if any_zone else 0
        return DisasterAction(zone_id=fallback_id, unit_type="ambulance")

    def zone_score(z):
        victims = [v for v in z["victims"] if v["alive"] and not v["rescued"]]
        if not victims:
            return (-1,)
        # Priority matches grader weights AND system prompt:
        # severity first (grader scores high>=0.7 highest),
        # then victim count, then min survival_time, then urgency
        return (
            z["severity"] * 100,
            len(victims) * 10,
            -min(v["survival_time"] for v in victims),
            max(v["urgency"] for v in victims),
            z["time_waiting"],
        )

    best = max(active, key=zone_score)
    return DisasterAction(
        zone_id=best["zone_id"],
        unit_type=select_unit(best, obs.resources),
    )


# ─── Action validation ───────────────────────────────────────────────────────

def validate_action(action: DisasterAction, obs) -> DisasterAction:
    """
    Validates and corrects the LLM's proposed action before submission.

    Checks:
    1. Target zone exists and is active with alive unrescued victims.
    2. unit_type is the correct size for remaining victims
       AND available in the current resource pool.

    Falls back to greedy_fallback() on any invalid zone.
    Corrects only the unit_type if the zone is valid but unit is wrong.
    """
    zone = next(
        (z for z in obs.zones if z["zone_id"] == action.zone_id), None
    )

    # Invalid zone or no rescuable victims → full fallback
    if zone is None or not zone["is_active"]:
        return greedy_fallback(obs)

    has_rescuable = any(
        v["alive"] and not v["rescued"] for v in zone["victims"]
    )
    if not has_rescuable:
        return greedy_fallback(obs)

    # Correct unit type if needed — zone itself is valid
    correct_unit = select_unit(zone, obs.resources)
    if action.unit_type != correct_unit:
        return DisasterAction(zone_id=action.zone_id, unit_type=correct_unit)

    return action


# ─── LLM decision ────────────────────────────────────────────────────────────

def choose_action(client: OpenAI, obs, step: int) -> DisasterAction:
    """
    Ask the LLM to choose a rescue action, then validate the response.

    Falls back to greedy_fallback() on any exception (parse error,
    network error, invalid JSON, missing fields, etc.).
    """
    try:
        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_prompt(obs, step)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )

        raw = (res.choices[0].message.content or "").strip()
        # Strip markdown code fences if the LLM wraps the JSON
        raw = raw.replace("```json", "").replace("```", "").strip()

        parsed = json.loads(raw)

        action = DisasterAction(
            zone_id=int(parsed.get("zone_id", 0)),
            unit_type=str(parsed.get("unit_type", "rescue_team")),
        )

        return validate_action(action, obs)

    except Exception as e:
        print(f"[DEBUG] LLM failed: {e}, greedy fallback used", flush=True)
        return greedy_fallback(obs)


# ─── Environment selector ────────────────────────────────────────────────────

async def get_env():
    """
    Returns a connected DisasterEnv instance.

    Priority: ENV_URL → LOCAL_IMAGE_NAME → localhost:8000
    """
    if ENV_URL:
        print(f"[DEBUG] Using ENV_URL: {ENV_URL}", flush=True)
        return await DisasterEnv(base_url=ENV_URL).__aenter__()
    elif IMAGE_NAME:
        print(f"[DEBUG] Using Docker image: {IMAGE_NAME}", flush=True)
        return await DisasterEnv.from_docker_image(IMAGE_NAME)
    else:
        print("[DEBUG] Using localhost: http://localhost:8000", flush=True)
        return await DisasterEnv(base_url="http://localhost:8000").__aenter__()


# ─── Task runner ─────────────────────────────────────────────────────────────

async def run_task(task_name: str, difficulty: str, seed: int, client: OpenAI) -> None:
    """
    Run one complete evaluation episode.

    Score extraction on terminal step:
        Primary  : grade_report["score"] from last_action_info
                   (set by environment when done=True — normalised final score
                   from compute_reward() via _compute_score_components()).
        Fallback : mean(rewards) clamped to [0, 1] — used only when
                   grade_report is absent (should not happen in normal flow).

    Logs [START] before reset, [STEP] after each step, [END] in finally.
    """
    rewards: List[float] = []
    steps_taken = 0
    score       = 0.0
    success     = False

    log_start(task_name, BENCHMARK, MODEL_NAME)

    env = await get_env()

    try:
        result = await env.reset(difficulty=difficulty, seed=seed)
        obs    = result.observation

        max_steps = obs.max_steps  # env se lo: easy=30, medium=40, hard=25
        for step in range(1, max_steps + 1):
            # Check termination before acting
            if result.done or obs.episode_done:
                break

            action = choose_action(client, obs, step)

            try:
                result = await env.step(action)
                obs    = result.observation
                reward = result.reward or 0.0
                done   = result.done
                error  = None
            except Exception as exc:
                reward = 0.0
                done   = True
                error  = str(exc)

            rewards.append(reward)
            steps_taken = step

            action_str = json.dumps({"zone_id": action.zone_id, "unit_type": action.unit_type})
            log_step(step, action_str, reward, done, error)

            if done:
                # Extract final score from grade_report (primary path)
                info         = obs.last_action_info or {}
                grade_report = info.get("grade_report")

                if grade_report and "score" in grade_report:
                    score   = grade_report["score"]
                    success = grade_report.get("passed", score >= SUCCESS_SCORE_THRESHOLD)
                else:
                    # Fallback: should not occur in normal flow
                    score   = min(sum(rewards) / max(len(rewards), 1), 1.0)
                    success = score >= SUCCESS_SCORE_THRESHOLD

                break

    except Exception as e:
        print(f"[DEBUG] Episode error: {e}", flush=True)

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)

        log_end(success, steps_taken, score, rewards)


# ─── Entry point ─────────────────────────────────────────────────────────────

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for task_name, difficulty, seed in TASKS:
        await run_task(task_name, difficulty, seed, client)


if __name__ == "__main__":
    asyncio.run(main())
