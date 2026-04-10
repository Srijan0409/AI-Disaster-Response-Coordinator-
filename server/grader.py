# =============================================================================
# grader.py - Scoring for AI Disaster Response Coordinator
# =============================================================================
#
# Computes a normalised episode score strictly in (0.05, 0.95).
#
# Score = normalize(base_rescue_score - capped_penalties)
#
# Per-step rewards use raw values (-0.3 to +1.0) so they accumulate
# meaningfully into episode_reward. Only the final episode score passes
# through _normalize() to reach the open (0, 1) interval.
#
# Public API:
#   calculate_step_reward()  - called inside grid.step() after every action
#   compute_reward()         - called inside grid.step() when done=True
#   grade_episode()          - called inside env.step() when done=True,
#                              returns full breakdown dict for inference.py
#
# Both compute_reward() and grade_episode() call _compute_score_components()
# - the single shared calculation core - so their scores are always identical.
# =============================================================================

from disaster_env.server.tasks import get_task

SCORE_MIN = 0.05
SCORE_MAX = 0.95

HIGH_SEVERITY_THRESHOLD = 0.7
MED_SEVERITY_THRESHOLD  = 0.4

REWARD_WEIGHTS = {
    "easy": {
        "high_severity_weight": 0.70,
        "med_severity_weight":  0.20,
        "low_severity_weight":  0.10,
        "time_penalty_weight":  0.05,
        "wait_penalty_weight":  0.01,
        "spawn_penalty_weight": 0.00,
    },
    "medium": {
        "high_severity_weight": 0.65,
        "med_severity_weight":  0.25,
        "low_severity_weight":  0.10,
        "time_penalty_weight":  0.10,
        "wait_penalty_weight":  0.02,
        "spawn_penalty_weight": 0.00,
    },
    "hard": {
        "high_severity_weight": 0.60,
        "med_severity_weight":  0.30,
        "low_severity_weight":  0.10,
        "time_penalty_weight":  0.15,
        "wait_penalty_weight":  0.03,
        "spawn_penalty_weight": 0.05,
    },
}


def _bucket(severity):
    """Returns 'high', 'med', or 'low' for a given severity float."""
    if severity >= HIGH_SEVERITY_THRESHOLD:
        return "high"
    elif severity >= MED_SEVERITY_THRESHOLD:
        return "med"
    return "low"


def _normalize(raw_score):
    """
    Maps a raw score from [0.0, 1.0] to [SCORE_MIN, SCORE_MAX].
    Guarantees output is always strictly inside (0, 1).
    Only applied to the final episode score - not per-step rewards.
    """
    clamped = max(0.0, min(1.0, raw_score))
    return SCORE_MIN + clamped * (SCORE_MAX - SCORE_MIN)


#  Per-step reward 

def calculate_step_reward(zone_id, zones, action_valid):
    """
    Immediate reward signal for a single action.

    Returns a raw float - NOT normalized - so values accumulate correctly
    into episode_reward inside grid.step(). Normalization is only applied
    to the final episode score via compute_reward().

    Raw values:
        High severity rescue   -> +1.0
        Medium severity rescue -> +0.5
        Low severity rescue    -> +0.2
        Wasted step            -> -0.1  (zone already fully rescued)
        Invalid action         -> -0.3  (bad zone_id or unit_type)

    Caller must pass zones_before (snapshot taken before apply_action()) so
    is_active correctly reflects the pre-rescue state. Passing zones_after
    makes the last-victim rescue appear as a wasted step.
    """
    if not action_valid:
        return -0.3

    target = next((z for z in zones if z["zone_id"] == zone_id), None)
    if target is None:
        return -0.3

    # Safe key access - missing "is_active" treated as active (conservative)
    if not target.get("is_active", True):
        return -0.1

    if target["severity"] >= HIGH_SEVERITY_THRESHOLD:
        return 1.0
    elif target["severity"] >= MED_SEVERITY_THRESHOLD:
        return 0.5
    else:
        return 0.2


#  Component calculators 

def _compute_rescue_score(zones_initial, zones_final, weights):
    """
    Base rescue score - fraction of each severity bucket rescued, weighted
    by importance. Uses initial severity to determine bucket so that
    spread_threat() during the episode does not retroactively inflate scores.

    Per-victim tracking (v["rescued"]) is used when a victims list exists.
    Falls back to zone-level rescued/people fields otherwise.

    Zones in zones_final that were not present at episode start are skipped.

    Returns float in [0.0, 1.0].
    """
    init_map = {z["zone_id"]: z for z in zones_initial}
    totals   = {"high": 0, "med": 0, "low": 0}
    rescued  = {"high": 0, "med": 0, "low": 0}

    for z in zones_initial:
        bucket  = _bucket(z["severity"])
        victims = z.get("victims")
        if victims is not None and len(victims) > 0:
            totals[bucket] += len(victims)
        else:
            totals[bucket] += z.get("people", 0)

    for z in zones_final:
        zid = z["zone_id"]
        if zid not in init_map:
            continue

        bucket          = _bucket(init_map[zid]["severity"])
        initial_victims = init_map[zid].get("victims")
        final_victims   = z.get("victims", [])

        if initial_victims is not None and len(initial_victims) > 0:
            initial_ids     = {v["id"] for v in initial_victims}
            rescued[bucket] += sum(
                1 for v in final_victims
                if v["rescued"] and v["id"] in initial_ids
            )
        else:
            rescued[bucket] += min(
                z.get("rescued", 0),
                init_map[zid].get("people", 0)
            )

    def bucket_score(k):
        if totals[k] == 0:
            return 0.0
        return weights[f"{k}_severity_weight"] * (rescued[k] / totals[k])

    return bucket_score("high") + bucket_score("med") + bucket_score("low")


def _compute_time_penalty(steps_taken, max_steps, weight):
    """
    Penalises consuming more of the step budget.
    Formula: weight * (steps_taken / max_steps)
    Returns float in [0.0, weight].
    """
    if max_steps <= 0:
        return 0.0
    if steps_taken <= 1:
        return 0.0
    return weight * (steps_taken / max_steps)


def _compute_wait_penalty(zones_final, weight):
    """
    Penalises leaving victims waiting in still-active zones.
    Uses average time_waiting across active zones.
    Returns float in [0.0, weight].

    Missing "is_active" key treated as active (conservative default).
    """
    active = [z for z in zones_final if z.get("is_active", True)]
    if not active:
        return 0.0
    avg_wait = sum(z.get("time_waiting", 0) for z in active) / len(active)
    return weight * min(avg_wait / 10.0, 1.0)


def _compute_spawn_penalty(spawned_victims, rescued_spawned, weight):
    """
    Penalises failing to rescue mid-episode spawned victims.
    Only meaningful in hard mode (weight > 0).
    Returns float in [0.0, weight].

    missed_fraction clamped to [0.0, 1.0] to prevent negative penalty
    if rescued_spawned somehow exceeds spawned_victims.
    """
    if spawned_victims == 0:
        return 0.0
    missed_fraction = max(0.0, min(1.0, (spawned_victims - rescued_spawned) / spawned_victims))
    return weight * missed_fraction


#  Shared calculation core 

def _compute_score_components(task_level, zones_initial, zones_final,
                               steps_taken, spawned_victims, rescued_spawned):
    """
    Single shared calculation core used by both compute_reward() and
    grade_episode() to guarantee identical scores on the terminal step.

    Penalties are capped at 50% of the base score so a very slow but
    successful rescue still receives meaningful credit.

    Returns dict with: base, t_pen, w_pen, s_pen, total_penalty,
    raw_score, final_score, task.
    """
    task    = get_task(task_level)
    weights = REWARD_WEIGHTS[task_level]

    base  = _compute_rescue_score(zones_initial, zones_final, weights)
    t_pen = _compute_time_penalty(steps_taken, task["max_steps"], weights["time_penalty_weight"])
    w_pen = _compute_wait_penalty(zones_final, weights["wait_penalty_weight"])
    s_pen = _compute_spawn_penalty(spawned_victims, rescued_spawned, weights["spawn_penalty_weight"])

    total_penalty = min(t_pen + w_pen + s_pen, base * 0.5)
    raw_score     = max(0.0, base - total_penalty)
    final_score   = round(_normalize(raw_score), 4)

    return {
        "base":          base,
        "t_pen":         t_pen,
        "w_pen":         w_pen,
        "s_pen":         s_pen,
        "total_penalty": total_penalty,
        "raw_score":     raw_score,
        "final_score":   final_score,
        "task":          task,
    }


#  Public API 

def compute_reward(
    task_level,
    zones_initial,
    zones_final,
    steps_taken,
    spawned_victims=0,
    rescued_spawned=0,
):
    """
    Final normalised episode score. Called inside grid.step() when done=True.
    Returns float strictly in (0.05, 0.95).
    """
    components = _compute_score_components(
        task_level, zones_initial, zones_final,
        steps_taken, spawned_victims, rescued_spawned,
    )
    return components["final_score"]


def grade_episode(
    task_level,
    zones_initial,
    zones_final,
    steps_taken,
    spawned_victims=0,
    rescued_spawned=0,
):
    """
    Full grading report for one completed episode.
    Called inside env.step() when done=True. Logged by inference.py.

    total_rescued counts only victims present at episode start so that
    total_rescued <= total_people always holds. Spawned victim stats are
    reported separately.

    Returns dict with: task_level, score, passed, success_threshold,
    breakdown, stats.
    """
    components = _compute_score_components(
        task_level, zones_initial, zones_final,
        steps_taken, spawned_victims, rescued_spawned,
    )

    final_score   = components["final_score"]
    task          = components["task"]

    # Count rescued victims - only those present at episode start
    initial_victims_exist = any(
        z.get("victims") is not None and len(z.get("victims", [])) > 0
        for z in zones_initial
    )

    if initial_victims_exist:
        initial_id_map = {
            z["zone_id"]: {v["id"] for v in z.get("victims", [])}
            for z in zones_initial
        }
        total_people  = sum(len(ids) for ids in initial_id_map.values())
        final_zone_map = {z["zone_id"]: z for z in zones_final}
        total_rescued  = sum(
            1
            for zid, initial_ids in initial_id_map.items()
            for v in final_zone_map.get(zid, {}).get("victims", [])
            if v["rescued"] and v["id"] in initial_ids
        )
    else:
        total_people  = sum(z.get("people",  0) for z in zones_initial)
        total_rescued = sum(z.get("rescued", 0) for z in zones_final)

    return {
        "task_level":        task_level,
        "score":             final_score,
        "passed":            final_score >= task["success_threshold"],
        "success_threshold": task["success_threshold"],
        "breakdown": {
            "base_rescue_score": round(components["base"],          4),
            "time_penalty":      round(components["t_pen"],         4),
            "wait_penalty":      round(components["w_pen"],         4),
            "spawn_penalty":     round(components["s_pen"],         4),
            "total_penalty":     round(components["total_penalty"], 4),
            "raw_score":         round(components["raw_score"],     4),
        },
        "stats": {
            "total_people":    total_people,
            "total_rescued":   total_rescued,
            "steps_taken":     steps_taken,
            "max_steps":       task["max_steps"],
            "spawned_victims": spawned_victims,
            "rescued_spawned": rescued_spawned,
        },
    }