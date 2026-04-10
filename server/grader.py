import copy
from disaster_env.server.tasks import get_task

# =============================================================================
# GRADER — AI Disaster Response Coordinator
# =============================================================================
# Computes a normalised episode score strictly in (0.0, 1.0) open interval.
#
# Score = normalize(base_rescue_score - time_penalty - wait_penalty - spawn_penalty)
#
# All final scores are mapped to [0.05, 0.95] so the open-interval (0, 1)
# constraint is satisfied naturally — no epsilon hacks needed.
#
# Fixes applied (original revision):
#   FIX A: _compute_spawn_penalty clamps missed_fraction to [0.0, 1.0].
#           Previously, if rescued_spawned > spawned_victims (e.g. due to
#           double-counting), missed_fraction went negative → negative penalty
#           → acted as a score bonus. Now clamped: max(0.0, missed_fraction).
#
#   FIX B: _compute_rescue_score skips zones_final entries whose zone_id is
#           not present in init_map (zones_initial). Previously a KeyError was
#           raised if the validator or hard-mode spawning added a zone that
#           didn't exist at episode start.
#
#   FIX C: _compute_wait_penalty uses z.get("is_active", True) instead of
#           z["is_active"] direct key access. Prevents KeyError when a zone
#           dict is missing the "is_active" field (e.g. minimal test fixtures
#           or partially-constructed state dicts).
#
#   FIX D: calculate_step_reward uses target.get("is_active", True) instead
#           of target["is_active"] direct key access — same defensive reason
#           as FIX C. A missing key raised KeyError instead of treating the
#           zone as active (the safe default).
#
# BUG I FIX (grade_episode / compute_reward inconsistency):
#   grade_episode() previously duplicated the entire score calculation:
#   base, t_pen, w_pen, s_pen, total_penalty, raw_score, _normalize() —
#   all computed independently. Any floating-point difference between the two
#   identical call chains produced grade_episode()["score"] != the reward
#   returned by env.step() on the terminal step.
#
#   Fix: grade_episode() now calls _compute_score_components() — the single
#   shared calculation core — guaranteeing score consistency.
#
# BUG G NOTE (severity bucket from initial state):
#   _compute_rescue_score() deliberately uses the zone's INITIAL severity to
#   determine the reward bucket (high/med/low). Using the final severity would
#   mean spread_threat() passively inflates scores: rescuing victims in a zone
#   whose severity crept up during the episode would grant a higher weight than
#   it deserved at rescue time. The initial severity is the correct anchor.
# =============================================================================

# Score range bounds — keeps every output strictly inside (0, 1)
SCORE_MIN = 0.05
SCORE_MAX = 0.95

# ---------------------------------------------------------------------------
# Severity thresholds
# ---------------------------------------------------------------------------
HIGH_SEVERITY_THRESHOLD = 0.7
MED_SEVERITY_THRESHOLD  = 0.4


def _bucket(severity):
    """Returns 'high', 'med', or 'low' for a given severity float."""
    if severity >= HIGH_SEVERITY_THRESHOLD:
        return "high"
    elif severity >= MED_SEVERITY_THRESHOLD:
        return "med"
    return "low"


# ---------------------------------------------------------------------------
# Reward weights per difficulty level
# Positive weights (high + med + low) always sum to 1.0
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Internal helper — maps any raw score to (0, 1) open interval
# ---------------------------------------------------------------------------

def _normalize(raw_score):
    """
    Maps raw_score from [0.0, 1.0] → [SCORE_MIN, SCORE_MAX].
    This guarantees the output is always strictly inside (0, 1).

    raw_score = 0.0  →  SCORE_MIN (0.05)
    raw_score = 1.0  →  SCORE_MAX (0.95)
    """
    clamped = max(0.0, min(1.0, raw_score))
    return SCORE_MIN + clamped * (SCORE_MAX - SCORE_MIN)


# ---------------------------------------------------------------------------
# Per-step reward — called inside env.step() after every action
# ---------------------------------------------------------------------------

def calculate_step_reward(zone_id, zones, action_valid):
    """
    Immediate reward signal for a single action.

    Raw reward values:
        High severity rescue   → +1.0
        Medium severity rescue → +0.5
        Low severity rescue    → +0.2
        Wasted step            → -0.1  (zone already fully rescued)
        Invalid action         → -0.3  (bad zone_id or unit_type)

    Final value is normalized to (0, 1) open interval using _normalize()
    so even per-step rewards satisfy the validator constraint.

    NOTE: caller (step()) must pass zones_before (snapshot taken BEFORE
    apply_action()) so is_active correctly reflects the pre-rescue state.
    Passing zones_after caused the last-victim rescue to appear as a wasted
    step (is_active=False after rescue) — see BUG 2 FIX in grid.py.

    FIX D: uses target.get("is_active", True) instead of target["is_active"]
           to avoid KeyError when zone dict is missing the "is_active" field.
           Default True means an unknown zone status is treated as active
           (conservative: prefer rescuing over skipping).
    """
    if not action_valid:
        raw = -0.3
    else:
        target = next((z for z in zones if z["zone_id"] == zone_id), None)
        if target is None:
            raw = -0.3
        # FIX D: use .get() with safe default instead of direct key access
        elif not target.get("is_active", True):
            raw = -0.1
        elif target["severity"] >= HIGH_SEVERITY_THRESHOLD:
            raw = 1.0
        elif target["severity"] >= MED_SEVERITY_THRESHOLD:
            raw = 0.5
        else:
            raw = 0.2

    # Shift raw from [-0.3, 1.0] to [0.0, 1.0] before normalizing
    shifted = (raw + 0.3) / 1.3
    return round(_normalize(shifted), 4)


# ---------------------------------------------------------------------------
# Component score functions
# ---------------------------------------------------------------------------

def _compute_rescue_score(zones_initial, zones_final, weights):
    """
    Base rescue score — fraction of each severity bucket rescued,
    weighted by importance.

    Uses victim counts from zones_initial so mid-episode spawned victims
    do not inflate the denominator.

    Per-victim tracking (v["rescued"]) is used when victims list exists.
    Falls back to zone-level rescued/people fields otherwise.

    BUG G NOTE: severity bucket is intentionally derived from the INITIAL
    zone state (init_map[zid]["severity"]), not the final state.

    FIX B: zones_final entries whose zone_id is not in init_map are skipped
           instead of raising KeyError.

    Returns float in [0.0, 1.0].
    """
    init_map = {z["zone_id"]: z for z in zones_initial}
    totals   = {"high": 0, "med": 0, "low": 0}
    rescued  = {"high": 0, "med": 0, "low": 0}

    # Count total people per bucket from initial state
    for z in zones_initial:
        bucket  = _bucket(z["severity"])
        victims = z.get("victims")
        if victims is not None and len(victims) > 0:
            totals[bucket] += len(victims)
        else:
            totals[bucket] += z.get("people", 0)

    # Count rescued people per bucket from final state
    for z in zones_final:
        zid = z["zone_id"]
        # FIX B: skip zones that were not present at episode start
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

    FIX C: uses z.get("is_active", True) instead of z["is_active"] to avoid
           KeyError when zone dict is missing the "is_active" field.
           Default True means zones with unknown status are treated as active
           (conservative — includes them in the wait penalty calculation).
    """
    # FIX C: safe key access with default
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

    FIX A: clamps missed_fraction to [0.0, 1.0] using max(0.0, ...).
           Previously, if rescued_spawned > spawned_victims (possible due to
           double-counting in hard mode or test fixtures), missed_fraction
           became negative, making the penalty negative — i.e. a score bonus.
    """
    if spawned_victims == 0:
        return 0.0
    # FIX A: clamp to [0.0, 1.0]
    missed_fraction = max(0.0, min(1.0, (spawned_victims - rescued_spawned) / spawned_victims))
    return weight * missed_fraction


# ---------------------------------------------------------------------------
# Internal helper — shared calculation core used by compute_reward and
# grade_episode to guarantee identical results (BUG I FIX)
# ---------------------------------------------------------------------------

def _compute_score_components(task_level, zones_initial, zones_final,
                               steps_taken, spawned_victims, rescued_spawned):
    """
    Computes all score components and returns them as a dict.
    Used by both compute_reward() and grade_episode() to eliminate
    the duplicate calculation that caused floating-point inconsistencies
    between the two functions (BUG I FIX).

    Returns dict with keys: base, t_pen, w_pen, s_pen, total_penalty,
    raw_score, final_score.
    """
    task    = get_task(task_level)
    weights = REWARD_WEIGHTS[task_level]

    base  = _compute_rescue_score(zones_initial, zones_final, weights)
    t_pen = _compute_time_penalty(steps_taken, task["max_steps"], weights["time_penalty_weight"])
    w_pen = _compute_wait_penalty(zones_final, weights["wait_penalty_weight"])
    s_pen = _compute_spawn_penalty(spawned_victims, rescued_spawned, weights["spawn_penalty_weight"])

    # Cap total penalties so they never exceed the base score
    total_penalty = min(t_pen + w_pen + s_pen, base * 0.5)

    raw_score   = max(0.0, base - total_penalty)
    final_score = round(_normalize(raw_score), 4)

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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_reward(
    task_level,
    zones_initial,
    zones_final,
    steps_taken,
    spawned_victims=0,
    rescued_spawned=0,
):
    """
    Final normalised score for one completed episode.
    Called inside env.step() when done=True.

    Returns float strictly in (0.0, 1.0) — open interval, validator-safe.

    Score = normalize(base_rescue_score - time_penalty - wait_penalty - spawn_penalty)
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
    inference.py logs this dict in the [END] block.

    total_rescued counts only victims present at episode start
    (spawned victims excluded to keep total_rescued <= total_people).
    Spawned rescue stats reported separately for visibility.

    BUG I FIX: previously duplicated the entire score calculation independently
    from compute_reward(). Any floating-point difference made
    grade_episode()["score"] != env.step() reward on the terminal step.
    Now calls _compute_score_components() — the single shared core.

    Returns dict with keys: task_level, score, passed, success_threshold,
    breakdown, stats.
    """
    # BUG I FIX: use shared calculation core — no more duplicate computation.
    components = _compute_score_components(
        task_level, zones_initial, zones_final,
        steps_taken, spawned_victims, rescued_spawned,
    )

    final_score   = components["final_score"]
    base          = components["base"]
    t_pen         = components["t_pen"]
    w_pen         = components["w_pen"]
    s_pen         = components["s_pen"]
    total_penalty = components["total_penalty"]
    raw_score     = components["raw_score"]
    task          = components["task"]

    # -----------------------------------------------------------------------
    # Stats — total_rescued counts only initial victims to avoid inflation
    # -----------------------------------------------------------------------
    initial_victims_exist = any(
        z.get("victims") is not None and len(z.get("victims", [])) > 0
        for z in zones_initial
    )

    if initial_victims_exist:
        initial_id_map = {
            z["zone_id"]: {v["id"] for v in z.get("victims", [])}
            for z in zones_initial
        }
        total_people = sum(len(ids) for ids in initial_id_map.values())

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
            "base_rescue_score": round(base,          4),
            "time_penalty":      round(t_pen,         4),
            "wait_penalty":      round(w_pen,         4),
            "spawn_penalty":     round(s_pen,         4),
            "total_penalty":     round(total_penalty, 4),
            "raw_score":         round(raw_score,     4),
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