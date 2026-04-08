import copy
from disaster_env.server.tasks import get_task

# =============================================================================
# GRADER — AI Disaster Response Coordinator
# =============================================================================
# Computes a normalised episode score in [0.0, 1.0].
#
# Score = base_rescue_score - time_penalty - wait_penalty - spawn_penalty
#
# calculate_step_reward
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
HIGH_SEVERITY_THRESHOLD = 0.7
MED_SEVERITY_THRESHOLD  = 0.4


def _bucket(severity):
    """Returns 'high', 'med', or 'low' for a given severity value."""
    if severity >= HIGH_SEVERITY_THRESHOLD:
        return "high"
    elif severity >= MED_SEVERITY_THRESHOLD:
        return "med"
    return "low"


# ---------------------------------------------------------------------------
# Per-step function — called inside env.step() after every action
# ---------------------------------------------------------------------------

def calculate_step_reward(zone_id, zones, action_valid):
    """
    Calculates the immediate reward for a single action at each step.

    High severity rescue   → +1.0
    Medium severity rescue → +0.5
    Low severity rescue    → +0.2
    Wasted step            → -0.1  (zone already fully rescued)
    Invalid action         → -0.3  (zone does not exist or bad unit_type)
    """
    if not action_valid:
        return -0.3

    target = next((z for z in zones if z["zone_id"] == zone_id), None)
    if target is None:
        return -0.3

    if not target["is_active"]:
        return -0.1

    if target["severity"] >= HIGH_SEVERITY_THRESHOLD:
        return 1.0
    elif target["severity"] >= MED_SEVERITY_THRESHOLD:
        return 0.5
    else:
        return 0.2


# ---------------------------------------------------------------------------
# Episode-end functions — called when done=True
# ---------------------------------------------------------------------------

def compute_rescue_score(zones_initial, zones_final, weights):
    """
    Base rescue score — fraction of each severity bucket rescued,
    weighted by importance.

    Uses victim counts from zones_initial so spawned victims don't
    inflate the score. Victim-aware: counts rescued victims per zone.

    FIX: Counts v["rescued"] == True directly instead of using alive=False
         as a proxy.

    DESIGN FIX: victims list presence check uses `is not None` instead of
         truthiness — an empty list `[]` is falsy but is a valid (empty)
         victims list and should trigger the fallback branch correctly.
         Changed to explicit `is not None and len(...) > 0` guard.

    Args:
        zones_initial : zone dicts at episode start, from reset()
        zones_final   : zone dicts at episode end, from last step()
        weights       : REWARD_WEIGHTS entry for this difficulty

    Returns:
        Float in [0.0, 1.0]
    """
    init_map = {z["zone_id"]: z for z in zones_initial}
    totals  = {"high": 0, "med": 0, "low": 0}
    rescued = {"high": 0, "med": 0, "low": 0}

    for z in zones_initial:
        bucket = _bucket(z["severity"])
        victims = z.get("victims")
        # DESIGN FIX: explicit check — empty list [] is falsy but valid
        if victims is not None and len(victims) > 0:
            totals[bucket] += len(victims)
        else:
            totals[bucket] += z.get("people", 0)

    for z in zones_final:
        zid    = z["zone_id"]
        bucket = _bucket(init_map[zid]["severity"])
        initial_victims = init_map[zid].get("victims")
        final_victims   = z.get("victims", [])

        # DESIGN FIX: same explicit check for initial_victims
        if initial_victims is not None and len(initial_victims) > 0:
            initial_ids      = {v["id"] for v in initial_victims}
            # FIX: check v["rescued"] directly — do not rely on alive=False
            rescued[bucket] += sum(
                1 for v in final_victims
                if v["rescued"] and v["id"] in initial_ids
            )
        else:
            # fallback: use zone-level rescued count
            rescued[bucket] += min(
                z.get("rescued", 0),
                init_map[zid].get("people", 0)
            )

    def bucket_score(k):
        if totals[k] == 0:
            return 0.0
        return weights[f"{k}_severity_weight"] * (rescued[k] / totals[k])

    return bucket_score("high") + bucket_score("med") + bucket_score("low")


def compute_time_penalty(steps_taken, max_steps, weight):
    """
    Penalises the agent for consuming more of the step budget.
    Formula: weight * (steps_taken / max_steps)
    """
    if steps_taken <= 1:
        return 0.0
    return weight * (steps_taken / max_steps)


def compute_wait_penalty(zones_final, weight):
    """
    Penalises leaving victims waiting in still-active zones.
    Uses average time_waiting across zones with unrescued people.
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
    Returns a full grading report for one completed episode.
    inference.py logs this dict in the [END] block.

    BUG 15 FIX: stats["total_rescued"] previously counted ALL rescued victims
    including spawned ones, while stats["total_people"] only counted initial
    victims from zones_initial. This caused total_rescued > total_people in
    hard mode (misleading and confusing for dashboards/callers).

    Fix: total_rescued now only counts rescued victims whose IDs appear in
    zones_initial (same logic as compute_rescue_score) — spawned victims are
    excluded from both totals to keep them consistent.
    The spawned rescue stats are separately reported as spawned_victims /
    rescued_spawned in the stats dict for full visibility.

    BUG 20 FIX: initial_victims_exist previously only checked zones_initial[0].
    If zone 0 had no victims (e.g. empty zone in a custom scenario) but other
    zones did, the check would return False and fall back to the zone-level
    rescued/people fields for ALL zones — silently under-counting rescued victims
    in zones 1..N that do have individual victim dicts.
    Fix: check whether ANY zone in zones_initial has a non-empty victims list.
    This is consistent with how compute_rescue_score() handles the same decision
    per-zone (already correct there), but grade_episode() needs a single branch
    decision for the stats block — we use the most permissive check: if any zone
    has victims, use the victim-ID path globally.
    """
    task    = get_task(task_level)
    weights = REWARD_WEIGHTS[task_level]

    base  = compute_rescue_score(zones_initial, zones_final, weights)
    t_pen = compute_time_penalty(steps_taken, task["max_steps"], weights["time_penalty_weight"])
    w_pen = compute_wait_penalty(zones_final, weights["wait_penalty_weight"])
    s_pen = compute_spawn_penalty(spawned_victims, rescued_spawned, weights["spawn_penalty_weight"])

    final_score = round(max(0.0, min(1.0, base - t_pen - w_pen - s_pen)), 4)

    # BUG 20 FIX: check whether ANY zone has a non-empty victims list, not just
    # zones_initial[0]. A zone-0-empty scenario would incorrectly fall through to
    # the zone-level fallback for all zones, missing per-victim data in zones 1..N.
    # Old code:
    #   initial_victims_exist = (
    #       zones_initial
    #       and zones_initial[0].get("victims") is not None
    #       and len(zones_initial[0].get("victims", [])) > 0
    #   )
    initial_victims_exist = any(
        z.get("victims") is not None and len(z.get("victims", [])) > 0
        for z in zones_initial
    )

    if initial_victims_exist:
        # Build set of all initial victim IDs across all zones
        initial_id_map = {
            z["zone_id"]: {v["id"] for v in z.get("victims", [])}
            for z in zones_initial
        }
        total_people = sum(
            len(ids) for ids in initial_id_map.values()
        )
        # BUG 15 FIX: only count rescued victims whose IDs existed at episode start
        final_zone_map = {z["zone_id"]: z for z in zones_final}
        total_rescued = sum(
            1
            for zid, initial_ids in initial_id_map.items()
            for v in final_zone_map.get(zid, {}).get("victims", [])
            if v["rescued"] and v["id"] in initial_ids
        )
    else:
        total_people  = sum(z.get("people", 0)  for z in zones_initial)
        total_rescued = sum(z.get("rescued", 0) for z in zones_final)

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