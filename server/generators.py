import random
from constants import (
    UTTARAKHAND_ZONES,
    SEVERITY_RANGES,
    PEOPLE_RANGES,
    RESOURCE_CONFIG,
    STEP_LIMITS,
)


def generate_resources(task_level):
    """
    Returns available rescue resources for a given difficulty level.

    Easy  : 2 ambulances, 1 rescue team, 1 helicopter
    Medium: 3 ambulances, 2 rescue teams, 1 helicopter
    Hard  : 2 ambulances, 1 rescue team, 0 helicopters

    Returns a copy so the original RESOURCE_CONFIG stays unchanged.
    """
    if task_level not in RESOURCE_CONFIG:
        raise ValueError(
            f"Invalid task level: '{task_level}'. "
            f"Must be one of: {list(RESOURCE_CONFIG.keys())}"
        )
    return dict(RESOURCE_CONFIG[task_level])


def generate_civilians(task_level, seed):
    """
    Generates a list of disaster-affected civilian groups for each zone.

    Uses a fixed seed to ensure the same scenario is produced every time.
    Severity range comes from constants.py — matches grid.py exactly.

    Hard mode: Kedarnath always starts at severity 1.0 (immediate crisis).

    Args:
        task_level: "easy", "medium", or "hard"
        seed      : fixed integer for reproducibility

    Returns:
        List of zone dicts with severity, people, rescued, time_waiting
    """
    random.seed(seed)

    zone_templates   = UTTARAKHAND_ZONES[task_level]
    sev_min, sev_max = SEVERITY_RANGES[task_level]
    ppl_min, ppl_max = PEOPLE_RANGES[task_level]

    civilians = []
    for i, template in enumerate(zone_templates):

        # Hard mode: Kedarnath always starts at maximum severity (1.0)
        # Matches grid.py GridWorld.reset() exactly
        if task_level == "hard" and i == 0:
            severity = 1.0
        else:
            severity = round(random.uniform(sev_min, sev_max), 2)

        civilians.append({
            "zone_id":       i,
            "name":          template["name"],
            "district":      template["district"],
            "disaster_type": template["disaster_type"],
            "severity":      severity,
            "people":        random.randint(ppl_min, ppl_max),
            "rescued":       0,
            "time_waiting":  0,
            "is_active":     True,
        })

    return civilians


def generate_scenario(task_level, seed):
    """
    Generates a complete episode package — civilians, resources, max steps.
    Person A calls this directly inside env.reset().

    Args:
        task_level: "easy", "medium", or "hard"
        seed      : fixed integer for reproducibility

    Returns:
        Dict with task_level, seed, max_steps, resources, zones
    """
    return {
        "task_level": task_level,
        "seed":       seed,
        "max_steps":  STEP_LIMITS[task_level],
        "resources":  generate_resources(task_level),
        "zones":      generate_civilians(task_level, seed),
    }


def sync_with_grid(civilians, grid):
    """
    Syncs the civilians list with the current grid state.

    After spread_threat() or tick(), severity and time_waiting may have
    changed in grid.py. This keeps the civilians list in sync.

    Person A calls this inside step() after every tick().

    Args:
        civilians: list from generate_civilians()
        grid     : GridWorld instance from grid.py

    Returns:
        Updated civilians list
    """
    for zone_dict in grid.get_state()["zones"]:
        zid = zone_dict["zone_id"]
        civilians[zid]["severity"]     = round(zone_dict["severity"], 2)
        civilians[zid]["time_waiting"] = zone_dict["time_waiting"]
        civilians[zid]["is_active"]    = zone_dict["is_active"]
    return civilians