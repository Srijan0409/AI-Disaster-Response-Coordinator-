from disaster_env.server.tasks import get_task

# =============================================================================

# FINAL GRADER — VALIDATOR SAFE (STRICT 0 < score < 1)

# =============================================================================

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

HIGH_SEVERITY_THRESHOLD = 0.7
MED_SEVERITY_THRESHOLD  = 0.4

def _bucket(severity):
if severity >= HIGH_SEVERITY_THRESHOLD:
return "high"
elif severity >= MED_SEVERITY_THRESHOLD:
return "med"
return "low"

# ───────────────────────────────────────────────────────────────

# STEP REWARD (SAFE)

# ───────────────────────────────────────────────────────────────

def calculate_step_reward(zone_id, zones, action_valid):
if not action_valid:
raw = -0.3
else:
target = next((z for z in zones if z["zone_id"] == zone_id), None)
if target is None:
raw = -0.3
elif not target.get("is_active", True):
raw = -0.1
elif target["severity"] >= HIGH_SEVERITY_THRESHOLD:
raw = 1.0
elif target["severity"] >= MED_SEVERITY_THRESHOLD:
raw = 0.5
else:
raw = 0.2

```
# shift to [0,1]
score = (raw + 0.3) / 1.3

# STRICT SAFE RANGE
score = max(0.05, min(0.95, score))

return round(score, 4)
```

# ───────────────────────────────────────────────────────────────

# CORE FUNCTIONS

# ───────────────────────────────────────────────────────────────

def compute_rescue_score(zones_initial, zones_final, weights):
init_map = {z["zone_id"]: z for z in zones_initial}
totals = {"high": 0, "med": 0, "low": 0}
rescued = {"high": 0, "med": 0, "low": 0}

```
for z in zones_initial:
    bucket = _bucket(z["severity"])
    victims = z.get("victims")
    if victims:
        totals[bucket] += len(victims)
    else:
        totals[bucket] += z.get("people", 0)

for z in zones_final:
    zid = z["zone_id"]
    if zid not in init_map:
        continue

    bucket = _bucket(init_map[zid]["severity"])
    initial_victims = init_map[zid].get("victims")
    final_victims = z.get("victims", [])

    if initial_victims:
        ids = {v["id"] for v in initial_victims}
        rescued[bucket] += sum(
            1 for v in final_victims if v["rescued"] and v["id"] in ids
        )
    else:
        rescued[bucket] += min(
            z.get("rescued", 0),
            init_map[zid].get("people", 0)
        )

def score(k):
    if totals[k] == 0:
        return 0.0
    return weights[f"{k}_severity_weight"] * (rescued[k] / totals[k])

return score("high") + score("med") + score("low")
```

def compute_time_penalty(steps_taken, max_steps, weight):
if steps_taken <= 1:
return 0.0
return weight * (steps_taken / max_steps)

def compute_wait_penalty(zones_final, weight):
active = [z for z in zones_final if z.get("is_active", True)]
if not active:
return 0.0
avg = sum(z.get("time_waiting", 0) for z in active) / len(active)
return weight * min(avg / 10.0, 1.0)

def compute_spawn_penalty(spawned_victims, rescued_spawned, weight):
if spawned_victims == 0:
return 0.0
missed = max(0.0, (spawned_victims - rescued_spawned) / spawned_victims)
return weight * min(missed, 1.0)

# ───────────────────────────────────────────────────────────────

# FINAL REWARD

# ───────────────────────────────────────────────────────────────

def compute_reward(
task_level,
zones_initial,
zones_final,
steps_taken,
spawned_victims=0,
rescued_spawned=0,
):
task = get_task(task_level)
weights = REWARD_WEIGHTS[task_level]

```
base = compute_rescue_score(zones_initial, zones_final, weights)
t_pen = compute_time_penalty(steps_taken, task["max_steps"], weights["time_penalty_weight"])
w_pen = compute_wait_penalty(zones_final, weights["wait_penalty_weight"])
s_pen = compute_spawn_penalty(spawned_victims, rescued_spawned, weights["spawn_penalty_weight"])

score = base - t_pen - w_pen - s_pen

# FINAL SAFE RANGE
score = max(0.05, min(0.95, score))

return round(score, 4)
```

def grade_episode(
task_level,
zones_initial,
zones_final,
steps_taken,
spawned_victims=0,
rescued_spawned=0,
):
task = get_task(task_level)
weights = REWARD_WEIGHTS[task_level]

```
base = compute_rescue_score(zones_initial, zones_final, weights)
t_pen = compute_time_penalty(steps_taken, task["max_steps"], weights["time_penalty_weight"])
w_pen = compute_wait_penalty(zones_final, weights["wait_penalty_weight"])
s_pen = compute_spawn_penalty(spawned_victims, rescued_spawned, weights["spawn_penalty_weight"])

score = base - t_pen - w_pen - s_pen

# FINAL SAFE RANGE
score = max(0.05, min(0.95, score))
final_score = round(score, 4)

total_people = sum(z.get("people", 0) for z in zones_initial)
total_rescued = sum(z.get("rescued", 0) for z in zones_final)

return {
    "task_level": task_level,
    "score": final_score,
    "passed": final_score >= task["success_threshold"],
    "success_threshold": task["success_threshold"],
    "stats": {
        "total_people": total_people,
        "total_rescued": total_rescued,
        "steps_taken": steps_taken,
        "max_steps": task["max_steps"],
    },
}
```
