import asyncio
import json
import os
import textwrap
from typing import List, Optional

from openai import OpenAI
from dotenv import load_dotenv

from disaster_env import DisasterEnv, DisasterAction

load_dotenv()

# ─── Config ─────────────────────────────────────────────────────

IMAGE_NAME   = os.getenv("LOCAL_IMAGE_NAME")
ENV_URL      = os.getenv("ENV_URL", "https://ayush4650-disaster-env-v2.hf.space")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

BENCHMARK    = "disaster_env"
MAX_STEPS    = 5
TEMPERATURE  = 0.2
MAX_TOKENS   = 120
SUCCESS_SCORE_THRESHOLD = 0.4

TASKS = [
    ("task1_easy_rescue",   "easy",   42),
    ("task2_medium_rescue", "medium", 123),
    ("task3_hard_rescue",   "hard",   999),
]

# ─── Prompt ─────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert disaster response coordinator.

Maximize total rescue score.

Rules:
1. Prioritize urgency=3 victims first.
2. Among them, choose lowest survival_time.
3. Prefer zones with MORE remaining victims.
4. NEVER waste large units on small zones.
5. Match unit size strictly:
   - 1–2 victims → ambulance
   - 3–4 victims → rescue_team
   - 5+ victims → helicopter
6. Avoid zones with no useful victims.

Respond ONLY with JSON:
{"zone_id": <int>, "unit_type": "<ambulance|rescue_team|helicopter>"}
""").strip()


def build_prompt(obs, step: int) -> str:
    zones = []

    for z in obs.zones:
        if not z["is_active"]:
            continue

        alive = [
            v for v in z["victims"]
            if v["alive"] and not v["rescued"]
        ]

        zones.append({
            "zone_id": z["zone_id"],
            "severity": z["severity"],
            "time_waiting": z["time_waiting"],
            "remaining": len(alive),
            "victims": [
                {
                    "urgency": v["urgency"],
                    "survival_time": v["survival_time"]
                }
                for v in alive
            ]
        })

    return f"""
Step {step}/{obs.max_steps}
Resources: {json.dumps(obs.resources)}

Zones:
{json.dumps(zones, indent=2)}

Choose best action.
"""


# ─── Logging ────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ─── LLM Decision ───────────────────────────────────────────────

def choose_action(client: OpenAI, obs, step: int) -> DisasterAction:
    try:
        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(obs, step)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )

        raw = (res.choices[0].message.content or "").strip()
        raw = raw.replace("```json", "").replace("```", "").strip()

        parsed = json.loads(raw)

        action = DisasterAction(
            zone_id=int(parsed.get("zone_id", 0)),
            unit_type=str(parsed.get("unit_type", "rescue_team")),
        )

        # 🚨 HARD VALIDATION (prevents penalties)
        return validate_action(action, obs)

    except Exception as e:
        print(f"[DEBUG] LLM failed: {e}, fallback used", flush=True)
        return greedy_fallback(obs)


# ─── VALIDATION LAYER (VERY IMPORTANT) ──────────────────────────

def validate_action(action: DisasterAction, obs):
    zone = next((z for z in obs.zones if z["zone_id"] == action.zone_id), None)

    if not zone or not zone["is_active"]:
        return greedy_fallback(obs)

    remaining = len([
        v for v in zone["victims"]
        if v["alive"] and not v["rescued"]
    ])

    # enforce correct unit
    correct_unit = select_unit(zone, obs.resources)

    if action.unit_type != correct_unit:
        return DisasterAction(zone_id=action.zone_id, unit_type=correct_unit)

    return action


# ─── STRONG POLICY ──────────────────────────────────────────────

def greedy_fallback(obs) -> DisasterAction:
    active = [z for z in obs.zones if z["is_active"]]

    if not active:
        return DisasterAction(zone_id=0, unit_type="ambulance")

    def score(z):
        victims = [
            v for v in z["victims"]
            if v["alive"] and not v["rescued"]
        ]

        if not victims:
            return -1

        remaining = len(victims)
        max_urgency = max(v["urgency"] for v in victims)
        min_survival = min(v["survival_time"] for v in victims)

        return (
            max_urgency * 100,
            -min_survival * 10,
            remaining * 10,
            z["severity"] * 5,
            z["time_waiting"],
        )

    best = max(active, key=score)

    return DisasterAction(
        zone_id=best["zone_id"],
        unit_type=select_unit(best, obs.resources)
    )


# ─── UNIT SELECTION (CRITICAL) ──────────────────────────────────

def select_unit(zone, resources):
    remaining = len([
        v for v in zone["victims"]
        if v["alive"] and not v["rescued"]
    ])

    if remaining >= 5 and resources.get("helicopters", 0) > 0:
        return "helicopter"
    elif 3 <= remaining <= 4 and resources.get("rescue_teams", 0) > 0:
        return "rescue_team"
    else:
        return "ambulance"


# ─── ENV SELECTOR ───────────────────────────────────────────────

async def get_env():
    if ENV_URL:
        print(f"[DEBUG] Using ENV_URL: {ENV_URL}", flush=True)
        return await DisasterEnv(base_url=ENV_URL).__aenter__()
    elif IMAGE_NAME:
        print(f"[DEBUG] Using Docker image: {IMAGE_NAME}", flush=True)
        return await DisasterEnv.from_docker_image(IMAGE_NAME)
    else:
        print("[DEBUG] Using localhost: http://localhost:8000", flush=True)
        return await DisasterEnv(base_url="http://localhost:8000").__aenter__()


# ─── RUNNER ─────────────────────────────────────────────────────

async def run_task(task_name: str, difficulty: str, seed: int, client: OpenAI):

    rewards = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task_name, BENCHMARK, MODEL_NAME)

    env = await get_env()

    try:
        result = await env.reset(difficulty=difficulty, seed=seed)
        obs = result.observation

        for step in range(1, MAX_STEPS + 1):
            if result.done or obs.episode_done:
                break

            action = choose_action(client, obs, step)

            try:
                result = await env.step(action)
                obs = result.observation
                reward = result.reward or 0.0
                done = result.done
                error = None
            except Exception as exc:
                reward = 0.0
                done = True
                error = str(exc)

            rewards.append(reward)
            steps_taken = step

            action_str = f"zone_id={action.zone_id},unit_type={action.unit_type}"
            log_step(step, action_str, reward, done, error)

            if done:
                info = obs.last_action_info or {}
                grade_report = info.get("grade_report")

                if grade_report and "score" in grade_report:
                    score = grade_report["score"]
                    success = grade_report.get("passed", score >= SUCCESS_SCORE_THRESHOLD)
                else:
                    score = min(sum(rewards) / max(len(rewards), 1), 1.0)
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


# ─── MAIN ───────────────────────────────────────────────────────

async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for task_name, difficulty, seed in TASKS:
        await run_task(task_name, difficulty, seed, client)


if __name__ == "__main__":
    asyncio.run(main())
