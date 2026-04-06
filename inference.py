"""
Inference Script — AI Disaster Response Coordinator
=====================================================
MANDATORY env vars:
    API_BASE_URL     The API endpoint for the LLM.
    MODEL_NAME       The model identifier to use for inference.
    HF_TOKEN         Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local Docker image (if using from_docker_image)

STDOUT FORMAT (do not change):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import os
import json
import textwrap
from typing import List, Optional

from openai import OpenAI

from disaster_env import DisasterAction, DisasterEnv

from dotenv import load_dotenv
import os

load_dotenv()

# ─── Config ───────────────────────────────────────────────────────────────────

IMAGE_NAME   = os.getenv("LOCAL_IMAGE_NAME")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")

BENCHMARK    = "disaster_env"
MAX_STEPS    = 12          # enough for easy/medium/hard
TEMPERATURE  = 0.2         # low temp — this is a reasoning task, not creative
MAX_TOKENS   = 80          # we only need {"allocate_to": N}
SUCCESS_SCORE_THRESHOLD = 0.4   # score >= 0.4 counts as success

# ─── Tasks to run (name, difficulty) ─────────────────────────────────────────

TASKS = [
    ("task1_nearest_first",    "easy"),
    ("task2_urgency_priority", "medium"),
    ("task3_max_survivors",    "hard"),
]

# ─── Prompts ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
    You are an AI disaster response coordinator.
    Each turn you receive a JSON describing a disaster scenario.

    Victims have:
      - id          (integer)
      - urgency     (1=low, 2=medium, 3=critical)
      - distance    (km from rescue base — lower is faster to reach)
      - survival_time (steps before this victim dies if not rescued)
      - alive       (boolean)
      - rescued     (boolean)

    You have `resources_available` rescue teams this step.
    You must choose ONE victim to rescue.

    Strategy:
      1. Only consider victims where alive=true AND rescued=false.
      2. Prioritise urgency=3 (critical) first.
      3. Among equal urgency, prefer lower survival_time (most at risk).
      4. Among equal urgency and survival_time, prefer lower distance.
      5. If no alive victims remain, output -1.

    Respond with ONLY a JSON object and nothing else:
    {"allocate_to": <integer victim id or -1>}

    No explanation. No markdown. No extra text.
""").strip()


def build_user_prompt(obs_dict: dict, step: int, task_name: str) -> str:
    return textwrap.dedent(f"""
        Task: {task_name}
        Step: {step}

        Current scenario:
        {json.dumps(obs_dict, indent=2)}

        Which victim do you rescue? Reply with ONLY: {{"allocate_to": <id>}}
    """).strip()


# ─── Logging helpers (exact format required by evaluator) ────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ─── LLM call ─────────────────────────────────────────────────────────────────

def choose_action(
    client: OpenAI,
    obs_dict: dict,
    step: int,
    task_name: str,
) -> int:
    """
    Ask the LLM which victim to rescue.
    Falls back to a rule-based greedy policy on any failure.
    """
    user_prompt = build_user_prompt(obs_dict, step, task_name)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
        # Strip accidental markdown fences
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        return int(parsed["allocate_to"])

    except Exception as exc:
        print(f"[DEBUG] LLM call failed ({exc}), using greedy fallback", flush=True)
        return _greedy_fallback(obs_dict)


def _greedy_fallback(obs_dict: dict) -> int:
    """
    Rule-based fallback: highest urgency → lowest survival_time → lowest distance.
    Returns -1 if no alive unrescued victims exist.
    """
    alive = [
        v for v in obs_dict.get("victims", [])
        if v.get("alive") and not v.get("rescued")
    ]
    if not alive:
        return -1
    best = max(alive, key=lambda v: (v["urgency"], -v["survival_time"], -v["distance"]))
    return best["id"]


# ─── Single task episode ───────────────────────────────────────────────────────

async def run_task(
    client: OpenAI,
    task_name: str,
    difficulty: str,
) -> dict:
    """
    Run one full episode for the given task.
    Returns {"score": float, "steps": int, "success": bool}
    """
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    env = await DisasterEnv.from_docker_image(IMAGE_NAME)

    try:
        # Reset with the right difficulty (seed=42 for reproducibility)
        result = await env.reset(difficulty=difficulty, seed=42)
        obs = result.observation

        for step in range(1, MAX_STEPS + 1):
            if result.done or obs.episode_done:
                break

            # Build plain dict for the prompt (drop internal pydantic noise)
            obs_dict = {
                "time_step":            obs.time_step,
                "resources_available":  obs.resources_available,
                "max_steps":            obs.max_steps,
                "difficulty":           obs.difficulty,
                "victims": [
                    {
                        "id":            v.id,
                        "urgency":       v.urgency,
                        "distance":      v.distance,
                        "survival_time": v.survival_time,
                        "alive":         v.alive,
                        "rescued":       v.rescued,
                    }
                    for v in obs.victims
                ],
            }

            victim_id = choose_action(client, obs_dict, step, task_name)
            action_str = f"allocate_to={victim_id}"

            try:
                result  = await env.step(DisasterAction(allocate_to=victim_id))
                obs     = result.observation
                reward  = result.reward or 0.0
                done    = result.done
                error   = None
            except Exception as exc:
                reward  = 0.0
                done    = True
                error   = str(exc)

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        # Score: sum of positive rewards / theoretical max
        # Theoretical max = num_victims × 1.0 (all critical, no penalty)
        num_victims = len(obs.victims) if obs else 5
        max_possible = float(num_victims)
        raw_score = sum(r for r in rewards if r > 0) / max_possible if max_possible > 0 else 0.0
        score   = min(max(raw_score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Episode error: {e}", flush=True)

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"task": task_name, "score": score, "steps": steps_taken, "success": success}


# ─── Main ─────────────────────────────────────────────────────────────────────

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    results = []
    for task_name, difficulty in TASKS:
        result = await run_task(client, task_name, difficulty)
        results.append(result)

    # Summary to stderr — keeps stdout clean for the evaluator
    import sys
    print("\n=== Inference Summary ===", file=sys.stderr)
    for r in results:
        status = "SUCCESS" if r["success"] else "FAIL"
        print(
            f"  [{status}] {r['task']:<35} score={r['score']:.3f}  steps={r['steps']}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    asyncio.run(main())