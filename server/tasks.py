import copy
from disaster_env.server.constants import UTTARAKHAND_ZONES, STEP_LIMITS, RESOURCE_CONFIG
from disaster_env.server.generators import generate_scenario

TASKS = {
    "easy": {
        "seed": 42, "task_level": "easy",
        "max_steps": STEP_LIMITS["easy"], "resources": RESOURCE_CONFIG["easy"],
        "spread": False, "spread_interval": None, "spawn_victims": False,
        "success_threshold": 0.5,
        "description": ("Single-zone flash flood at Kedarnath Temple Area, Rudraprayag. "
            "The agent manages 2 ambulances and 1 rescue team to evacuate "
            "trapped civilians before the step budget runs out. "
            "Disaster does not spread. Ideal for learning basic prioritisation."),
        "objective": "Rescue maximum civilians in a single disaster zone.",
        "observation_space": {"zones": "List of active disaster zones with severity, people, rescued, time_waiting",
            "resources": "Available ambulances, rescue_teams, helicopters",
            "step": "Current step number", "max_steps": "Total step budget for the episode"},
        "action_space": {"type": "discrete",
            "description": "Send a rescue unit to a zone: {zone_id, unit_type}",
            "unit_types": ["ambulance", "rescue_team", "helicopter"]},
    },
    "medium": {
        "seed": 123, "task_level": "medium",
        "max_steps": STEP_LIMITS["medium"], "resources": RESOURCE_CONFIG["medium"],
        "spread": True, "spread_interval": 5, "spawn_victims": False,
        "success_threshold": 0.6,
        "description": ("Three-zone disaster across Kedarnath (flash flood), "
            "Badrinath (landslide), and Rishikesh (river overflow). "
            "The agent has more resources but disaster spreads every 5 steps, "
            "raising severity in adjacent zones. Prioritisation and timing both matter."),
        "objective": "Coordinate rescue across 3 zones with spreading disaster.",
        "observation_space": {"zones": "List of active disaster zones with severity, people, rescued, time_waiting",
            "resources": "Available ambulances, rescue_teams, helicopters",
            "step": "Current step number", "max_steps": "Total step budget for the episode"},
        "action_space": {"type": "discrete",
            "description": "Send a rescue unit to a zone: {zone_id, unit_type}",
            "unit_types": ["ambulance", "rescue_team", "helicopter"]},
    },
    "hard": {
        "seed": 999, "task_level": "hard",
        "max_steps": STEP_LIMITS["hard"], "resources": RESOURCE_CONFIG["hard"],
        "spread": True, "spread_interval": 2, "spawn_victims": True,
        "success_threshold": 0.7,
        "description": ("Five-zone multi-disaster scenario across Uttarakhand: "
            "Kedarnath (flash flood), Joshimath (land subsidence), "
            "Dehradun (landslide), Haridwar (river overflow), Chamoli (glacier burst). "
            "Kedarnath starts at maximum severity. Disaster spreads every 2 steps. "
            "New victims spawn every 5 steps. No helicopters available. "
            "The agent must triage aggressively under severe time pressure."),
        "objective": "Manage 5 zones under extreme resource constraints and time pressure.",
        "observation_space": {"zones": "List of active disaster zones with severity, people, rescued, time_waiting",
            "resources": "Available ambulances, rescue_teams, helicopters",
            "step": "Current step number", "max_steps": "Total step budget for the episode"},
        "action_space": {"type": "discrete",
            "description": "Send a rescue unit to a zone: {zone_id, unit_type}",
            "unit_types": ["ambulance", "rescue_team"]},
    },
}


def get_task(task_level):
    """
    Returns the task configuration for the given difficulty level.

    BUG 14 FIX: previously returned TASKS[task_level] directly - a live
    reference to the shared mutable dict. Any caller mutating the returned
    dict (e.g., task["resources"]["ambulances"] = 0) would permanently
    corrupt the TASKS registry for all subsequent calls in the same process.
    This is a classic mutable-default / shared-state bug.

    Fix: return a deep copy so every caller gets an independent snapshot.
    deepcopy is used (not shallow copy) because nested dicts like
    "resources", "observation_space", and "action_space" would still be
    shared references with a shallow copy.
    """
    if task_level not in TASKS:
        raise ValueError(
            f"Unknown task level: {task_level!r}. "
            f"Choose from: {list(TASKS.keys())}"
        )
    return copy.deepcopy(TASKS[task_level])


def get_task_scenario(task_level):
    config = get_task(task_level)
    return generate_scenario(task_level, config["seed"])


def list_tasks():
    return list(TASKS.keys())