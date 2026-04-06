from tasks import get_task

# =============================================================================
# GRADER — AI Disaster Response Coordinator
# =============================================================================
# Computes a normalised episode score in [0.0, 1.0].
#
# Score = base_rescue_score - time_penalty - wait_penalty - spawn_penalty
#
# calculate_step_reward / calculate_waiting_penalty
#     Called inside env.step() after every action — non-sparse per-step signal.
#
# Dashboard rules satisfied:
#   ✅ Scores between 0.0 and 1.0
#   ✅ Deterministic and reproducible
#   ✅ Meaningful varying signal — not just 0 or 1
#   ✅ Partial progress rewarded
#   ✅ Clear success/failure criteria per task
#   ✅ Hard task genuinely harder to score well on
# =============================================================================

# ---------------------------------------------------------------------------
# Reward weights per difficulty level
# Positive weights (high + med + low) always sum to 1.0
# ---------------------------------------------------------------------------
REWARD_WEIGHTS = {
    "easy": {
        "high_severity_weight": 0.7,
        "med_severity_weight":  0.2,
        "low_severity_weight":  0.1,
        "time_penalty_weight":  0.05,
        "wait_penalty_weight":  0.01,
        "spawn_penalty_weight": 0.0,
    },
    "medium": {
        "high_severity_weight": 0.65,
        "med_severity_weight":  0.25,
        "low_severity_weight":  0.10,
        "time_penalty_weight":  0.10,
        "wait_penalty_weight":  0.02,
        "spawn_penalty_weight": 0.0,
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

# ---------------------------------------------------------------------------
# Severity thresholds
# ---------------------------------------------------------------------------
HIGH_SEVERITY_THRESHOLD  = 0.7
MED_SEVERITY_THRESHOLD   = 0.4

# ---------------------------------------------------------------------------
# Per-step reward constants
# ---------------------------------------------------------------------------
HIGH_SEVERITY_REWARD     =  1.0
MEDIUM_SEVERITY_REWARD   =  0.5
LOW_SEVERITY_REWARD      =  0.2
WASTED_STEP_PENALTY      = -0.1
INVALID_ACTION_PENALTY   = -0.3
WAITING_PENALTY_PER_STEP =  0.02


def _bucket(severity):
    """Returns 'high', 'med', or 'low' for a given severity value."""
    if severity >= HIGH_SEVERITY_THRESHOLD:
        return "high"
    elif severity >= MED_SEVERITY_THRESHOLD:
        return "med"
    return "low"


# ---------------------------------------------------------------------------
# Per-step functions — Person A calls these inside env.step()
# ---------------------------------------------------------------------------

def calculate_step_reward(zone_id, zones, action_valid):
    """
    Calculates the immediate reward for a single action at each step.
    Called inside env.step() after every action — non-sparse per-step signal.

    High severity rescue   → +1.0
    Medium severity rescue → +0.5
    Low severity rescue    → +0.2
    Wasted step            → -0.1  (zone already fully rescued)
    Invalid action         → -0.3  (zone does not exist)

    Args:
        zone_id      : the zone the agent sent a resource to
        zones        : current list of zone dicts
        action_valid : whether the action was a valid choice

    Returns:
        Float reward value
    """
    if not action_valid:
        return INVALID_ACTION_PENALTY

    target = next((z for z in zones if z["zone_id"] == zone_id), None)
    if target is None:
        return INVALID_ACTION_PENALTY

    if not target["is_active"]:
        return WASTED_STEP_PENALTY

    if target["severity"] >= HIGH_SEVERITY_THRESHOLD:
        return HIGH_SEVERITY_REWARD
    elif target["severity"] >= MED_SEVERITY_THRESHOLD:
        return MEDIUM_SEVERITY_REWARD
    else:
        return LOW_SEVERITY_REWARD


def calculate_waiting_penalty(zones):
    """
    Calculates the total waiting penalty for all unrescued high-severity zones.
    Called inside env.step() after every tick.

    Penalises the agent for letting critical victims wait too long.
    Encourages the agent to act quickly on high-severity zones.

    Args:
        zones: current list of zone dicts

    Returns:
        Float penalty value (always negative or zero)
    """
    penalty = 0.0
    for zone in zones:
        if zone["is_active"] and zone["severity"] >= HIGH_SEVERITY_THRESHOLD:
            penalty -= zone["time_waiting"] * WAITING_PENALTY_PER_STEP
    return penalty


# ---------------------------------------------------------------------------
# Episode-end functions — Person A calls these when done=True
# ---------------------------------------------------------------------------

def compute_rescue_score(zones_initial, zones_final, weights):
    """
    Base rescue score — fraction of each severity bucket rescued,
    weighted by importance.

    Uses zones_initial severity so the agent is scored on what it faced
    at the start. Rescued count is capped at initial people count so
    spawned victims do not inflate this score.

    Args:
        zones_initial : zone dicts at episode start, from reset()
        zones_final   : zone dicts at episode end, from last step()
        weights       : REWARD_WEIGHTS entry for this difficulty

    Returns:
        Float in [0.0, 1.0]
    """
    init_map = {z["zone_id"]: z for z in zones_initial}
    totals   = {"high": 0, "med": 0, "low": 0}
    rescued  = {"high": 0, "med": 0, "low": 0}

    for z in zones_initial:
        totals[_bucket(z["severity"])] += z["people"]

    for z in zones_final:
        zid    = z["zone_id"]
        bucket = _bucket(init_map[zid]["severity"])
        rescued[bucket] += min(z["rescued"], init_map[zid]["people"])

    def bucket_score(key, weight):
        if totals[key] == 0:
            return 0.0
        return weight * (rescued[key] / totals[key])

    raw = (
        bucket_score("high", weights["high_severity_weight"]) +
        bucket_score("med",  weights["med_severity_weight"])  +
        bucket_score("low",  weights["low_severity_weight"])
    )

    max_possible = sum(
        weights[f"{k}_severity_weight"]
        for k in ["high", "med", "low"]
        if totals[k] > 0
    )

    if max_possible == 0:
        return 0.0

    return raw / max_possible


def compute_time_penalty(steps_taken, max_steps, weight):
    """
    Penalises the agent for consuming more of the step budget.
    Formula: weight * (steps_taken / max_steps)

    Args:
        steps_taken : steps the agent used this episode
        max_steps   : total step budget (from tasks.py)
        weight      : time_penalty_weight for this difficulty

    Returns:
        Float >= 0.0
    """
    if steps_taken <= 1:
        return 0.0
    return weight * (steps_taken / max_steps)


def compute_wait_penalty(zones_final, weight):
    """
    Penalises leaving victims waiting in still-active zones.
    Uses average time_waiting across zones with unrescued people.

    Args:
        zones_final : zone dicts at episode end
        weight      : wait_penalty_weight for this difficulty

    Returns:
        Float >= 0.0
    """
    active = [z for z in zones_final if z["is_active"]]
    if not active:
        return 0.0
    avg_wait = sum(z["time_waiting"] for z in active) / len(active)
    return weight * min(avg_wait / 10.0, 1.0)


def compute_spawn_penalty(spawned_victims, rescued_spawned, weight):
    """
    Penalises failing to rescue victims that spawned mid-episode.
    Only relevant in hard mode. Returns 0.0 if no victims spawned.

    Args:
        spawned_victims : total new victims that appeared mid-episode
        rescued_spawned : how many of those spawned victims were rescued
        weight          : spawn_penalty_weight for this difficulty

    Returns:
        Float >= 0.0
    """
    if spawned_victims == 0:
        return 0.0
    return weight * ((spawned_victims - rescued_spawned) / spawned_victims)


def compute_reward(
    task_level,
    zones_initial,
    zones_final,
    steps_taken,
    spawned_victims=0,
    rescued_spawned=0,
):
    """
    Computes the final normalised score for one completed episode.
    Called inside env.step() when done=True. Returns float in [0.0, 1.0].

    Score = base_rescue_score - time_penalty - wait_penalty - spawn_penalty

    Args:
        task_level      : 'easy', 'medium', or 'hard'
        zones_initial   : zone dicts from reset()
        zones_final     : zone dicts from last step()
        steps_taken     : how many steps the agent used
        spawned_victims : total new victims mid-episode (hard only, default 0)
        rescued_spawned : how many spawned victims rescued (hard only, default 0)

    Returns:
        Float in [0.0, 1.0]
    """
    task    = get_task(task_level)
    weights = REWARD_WEIGHTS[task_level]

    base  = compute_rescue_score(zones_initial, zones_final, weights)
    t_pen = compute_time_penalty(steps_taken, task["max_steps"], weights["time_penalty_weight"])
    w_pen = compute_wait_penalty(zones_final, weights["wait_penalty_weight"])
    s_pen = compute_spawn_penalty(spawned_victims, rescued_spawned, weights["spawn_penalty_weight"])

    return round(max(0.0, min(1.0, base - t_pen - w_pen - s_pen)), 4)


def grade_episode(
    task_level,
    zones_initial,
    zones_final,
    steps_taken,
    spawned_victims=0,
    rescued_spawned=0,
):
    """
    Returns a full grading report for one episode.
    inference.py logs this dict in the [END] block.

    Args:
        task_level      : 'easy', 'medium', or 'hard'
        zones_initial   : zone dicts from reset()
        zones_final     : zone dicts from last step()
        steps_taken     : steps used this episode
        spawned_victims : total new victims mid-episode (hard only)
        rescued_spawned : spawned victims rescued (hard only)

    Returns:
        Dict with keys: task_level, score, passed, success_threshold,
                        breakdown, stats
    """
    task    = get_task(task_level)
    weights = REWARD_WEIGHTS[task_level]

    base  = compute_rescue_score(zones_initial, zones_final, weights)
    t_pen = compute_time_penalty(steps_taken, task["max_steps"], weights["time_penalty_weight"])
    w_pen = compute_wait_penalty(zones_final, weights["wait_penalty_weight"])
    s_pen = compute_spawn_penalty(spawned_victims, rescued_spawned, weights["spawn_penalty_weight"])

    final_score   = round(max(0.0, min(1.0, base - t_pen - w_pen - s_pen)), 4)
    total_people  = sum(z["people"]  for z in zones_initial)
    total_rescued = sum(z["rescued"] for z in zones_final)

    return {
        "task_level":        task_level,
        "score":             final_score,
        "passed":            final_score >= task["success_threshold"],
        "success_threshold": task["success_threshold"],
        "breakdown": {
            "base_rescue_score": round(base,  4),
            "time_penalty":      round(t_pen, 4),
            "wait_penalty":      round(w_pen, 4),
            "spawn_penalty":     round(s_pen, 4),
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