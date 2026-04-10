import random
from disaster_env.server.constants import (
    UTTARAKHAND_ZONES,
    SEVERITY_RANGES,
    PEOPLE_RANGES,
    RESOURCE_CONFIG,
    STEP_LIMITS,
    VICTIM_URGENCY_WEIGHTS,
    VICTIM_SURVIVAL_TIME,
    VICTIM_DISTANCE_KM,
    VICTIMS_PER_ZONE,
)


def _generate_victims(task_level, zone_index, seed_offset):
    """
    Generates individual victims for a single zone.

    Each victim has:
        id           : globally unique int across ALL zones (not zone-local).
                       Computed as zone_index * VICTIMS_PER_ZONE[task_level] + local_vid.
                       This eliminates cross-zone ID collisions without requiring a
                       mutable global counter here.
        urgency      : 1 (low), 2 (medium), 3 (critical)
        survival_time: steps before victim dies if not rescued
        distance_km  : distance from rescue base in km
        alive        : True initially
        rescued      : False initially

    Uses an isolated random.Random instance seeded per zone so that
    victim generation is fully reproducible and independent of any
    global random state.

    FIX (global IDs): Previously IDs were zone-local (0, 1, 2 ...).
        Callers like grader.py that look up victims by ID across zones could
        silently match the wrong victim if two zones both had id=0.
        Fix: ID = zone_index * VICTIMS_PER_ZONE[task_level] + local_vid,
        guaranteeing uniqueness across all zones for initial victims.
        spawn_new_victims() in grid.py continues to use _next_victim_id
        (a monotonically increasing counter) for spawned victims, which
        are always > max initial ID, so no collision is possible.

    Args:
        task_level  : "easy", "medium", or "hard"
        zone_index  : used to offset the seed AND compute global IDs
        seed_offset : base seed value

    Returns:
        List of victim dicts
    """
    rng = random.Random(seed_offset + zone_index * 97)
    count = VICTIMS_PER_ZONE[task_level]
    st_min, st_max = VICTIM_SURVIVAL_TIME[task_level]
    d_min, d_max   = VICTIM_DISTANCE_KM[task_level]
    weights        = VICTIM_URGENCY_WEIGHTS[task_level]

    # Global ID base: ensures IDs are unique across all zones.
    # zone 0 -> IDs 0..count-1, zone 1 -> IDs count..2*count-1, etc.
    id_base = zone_index * count

    victims = []
    for local_vid in range(count):
        victims.append({
            "id":            id_base + local_vid,   # FIX: globally unique ID
            "urgency":       rng.choices([1, 2, 3], weights=weights)[0],
            "survival_time": rng.randint(st_min, st_max),
            "distance_km":   round(rng.uniform(d_min, d_max), 2),
            "alive":         True,
            "rescued":       False,
        })
    return victims


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
    Severity range comes from constants.py - matches grid.py exactly.

    Hard mode: Kedarnath always starts at severity 1.0 (immediate crisis).

    FIX: 'people' is derived from len(victims) - single source of truth.
         Matches grid.py where zone.people = len(zone.victims).

    WARNING FIX (global random isolation): Uses an isolated random.Random
         instance for severity generation instead of global random.seed() +
         random.uniform(). This prevents generate_civilians() from polluting
         global random state when called concurrently with reset() or other
         functions that also set random.seed(). The scenario output is
         identical - just no global side-effect.

    Args:
        task_level: "easy", "medium", or "hard"
        seed      : fixed integer for reproducibility

    Returns:
        List of zone dicts with severity, people, rescued, time_waiting
    """
    # WARNING FIX: isolated RNG - no global random state mutation
    sev_rng = random.Random(seed)

    zone_templates   = UTTARAKHAND_ZONES[task_level]
    sev_min, sev_max = SEVERITY_RANGES[task_level]

    civilians = []
    for i, template in enumerate(zone_templates):

        # Hard mode: Kedarnath always starts at maximum severity (1.0)
        # Matches grid.py GridWorld.reset() exactly
        if task_level == "hard" and i == 0:
            severity = 1.0
        else:
            # WARNING FIX: use isolated sev_rng instead of global random
            severity = round(sev_rng.uniform(sev_min, sev_max), 2)

        # FIX: generate victims first, then derive people count from them.
        # _generate_victims now produces globally unique IDs (FIX above).
        victims = _generate_victims(task_level, i, seed)

        civilians.append({
            "zone_id":       i,
            "name":          template["name"],
            "district":      template["district"],
            "disaster_type": template["disaster_type"],
            "severity":      severity,
            "people":        len(victims),   # derived from victims, not random
            "rescued":       0,
            "time_waiting":  0,
            "is_active":     True,
            "victims":       victims,
        })

    return civilians


def generate_scenario(task_level, seed):
    """
    Generates a complete episode package - civilians, resources, max steps.

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

    After spread_threat() or tick(), severity, time_waiting, is_active,
    people (due to spawning), and rescued counts may have changed in
    grid.py. This keeps the civilians list fully in sync.

    # BUG 7 FIX: previously only updated severity, time_waiting, is_active.
    In hard mode, spawn_new_victims() increases zone.people and adds new
    victim dicts to zone.victims. Callers using civilians["people"] or
    civilians["rescued"] after a hard episode would read stale values -
    e.g., civilians[0]["people"] = 8 while grid.zones[0].people = 11.
    Fix: sync people, rescued, and victims fields as well.

    Args:
        civilians: list from generate_civilians()
        grid     : GridWorld instance from grid.py

    Returns:
        Updated civilians list (same object, mutated in-place)
    """
    for zone_dict in grid.get_state()["zones"]:
        zid = zone_dict["zone_id"]
        civilians[zid]["severity"]     = round(zone_dict["severity"], 2)
        civilians[zid]["time_waiting"] = zone_dict["time_waiting"]
        civilians[zid]["is_active"]    = zone_dict["is_active"]
        # BUG 7 FIX: sync people, rescued, and victims so hard-mode spawning
        # and rescue actions are reflected in the civilians list.
        civilians[zid]["people"]       = zone_dict["people"]
        civilians[zid]["rescued"]      = zone_dict["rescued"]
        civilians[zid]["victims"]      = zone_dict["victims"]
    return civilians