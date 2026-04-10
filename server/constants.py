# =============================================================================
# CONSTANTS - AI Disaster Response Coordinator
# =============================================================================
# Shared constants used by both grid.py and generators.py.
# Kept in a separate file to avoid circular imports and ensure
# both files always use identical ranges and configurations.
# =============================================================================

# Real Uttarakhand disaster locations - Based on 2013 Kedarnath floods scenario
UTTARAKHAND_ZONES = {
    "easy": [
        {
            "name":          "Kedarnath Temple Area",
            "district":      "Rudraprayag",
            "disaster_type": "flash_flood",
        }
    ],
    "medium": [
        {
            "name":          "Kedarnath Temple Area",
            "district":      "Rudraprayag",
            "disaster_type": "flash_flood",
        },
        {
            "name":          "Badrinath",
            "district":      "Chamoli",
            "disaster_type": "landslide",
        },
        {
            "name":          "Rishikesh",
            "district":      "Dehradun",
            "disaster_type": "river_overflow",
        }
    ],
    "hard": [
        {
            "name":          "Kedarnath Temple Area",
            "district":      "Rudraprayag",
            "disaster_type": "flash_flood",
        },
        {
            "name":          "Joshimath",
            "district":      "Chamoli",
            "disaster_type": "land_subsidence",
        },
        {
            "name":          "Dehradun City",
            "district":      "Dehradun",
            "disaster_type": "landslide",
        },
        {
            "name":          "Haridwar Ghat Area",
            "district":      "Haridwar",
            "disaster_type": "river_overflow",
        },
        {
            "name":          "Chamoli",
            "district":      "Chamoli",
            "disaster_type": "glacier_burst",
        }
    ]
}

# Severity ranges per difficulty level
# Used by both grid.py and generators.py - must stay in sync
SEVERITY_RANGES = {
    "easy":   (0.3, 0.7),
    "medium": (0.4, 0.9),
    "hard":   (0.6, 1.0),
}

# People trapped per zone per difficulty level
PEOPLE_RANGES = {
    "easy":   (2, 6),
    "medium": (5, 12),
    "hard":   (8, 20),
}

# Resource configurations per difficulty level
# Hard mode has no helicopters - forces harder triage decisions
RESOURCE_CONFIG = {
    "easy":   {"ambulances": 2, "rescue_teams": 1, "helicopters": 1},
    "medium": {"ambulances": 3, "rescue_teams": 2, "helicopters": 1},
    "hard":   {"ambulances": 2, "rescue_teams": 1, "helicopters": 0},
}

# Maximum steps allowed per episode per difficulty level
# Design intent: hard has fewest steps (time pressure), easy is moderate.
# hard(25) <= easy(30) <= medium(40) satisfies test_hard_has_fewest_steps.
STEP_LIMITS = {
    "easy":   30,
    "medium": 40,
    "hard":   25,
}

# Victim constants - used by generators.py and grid.py
# urgency weights: [low=1, medium=2, critical=3]
VICTIM_URGENCY_WEIGHTS = {
    "easy":   [0.5, 0.3, 0.2],
    "medium": [0.3, 0.4, 0.3],
    "hard":   [0.2, 0.3, 0.5],
}

# Survival time range (in steps) per difficulty
# FIX: hard max (6) <= STEP_LIMITS["hard"] (25) - test passes.
# Note: effective survival ticks = survival_time / decay.
# hard: min effective = 2/2 = 1 tick, max effective = 6/2 = 3 ticks - intentional pressure.
VICTIM_SURVIVAL_TIME = {
    "easy":   (6, 12),
    "medium": (4, 9),
    "hard":   (2, 6),
}

# Distance from rescue base in km per difficulty
VICTIM_DISTANCE_KM = {
    "easy":   (1, 5),
    "medium": (1, 8),
    "hard":   (1, 10),
}

# How many steps survival_time decays per tick
VICTIM_TIME_DECAY = {
    "easy":   1,
    "medium": 1,
    "hard":   2,
}

# How many victims per zone per difficulty
# hard: 5 zones x 8 victims = 40 total > STEP_LIMITS["hard"]=25
#       forces triage - intentional design constraint.
# medium: 3 zones x 8 victims = 24 <= STEP_LIMITS["medium"]=40 - completable.
# easy: 1 zone x 7 victims = 7 <= STEP_LIMITS["easy"]=30 - completable.
#       7 ensures episode lasts at least 3 steps (rescue_team=3/step),
#       preventing premature done=True that broke multi-step tests.
#       Ordering maintained: easy(7) <= medium(8) <= hard(8).
VICTIMS_PER_ZONE = {
    "easy":   7,
    "medium": 8,
    "hard":   8,
}

# Unit types allowed per difficulty level.
# Hard mode excludes helicopters - enforced at runtime in apply_action().
# FIX (helicopter enforcement): previously only documented in task description;
# apply_action() accepted helicopters silently. Now validated against this map.
ALLOWED_UNIT_TYPES = {
    "easy":   {"ambulance", "rescue_team", "helicopter"},
    "medium": {"ambulance", "rescue_team", "helicopter"},
    "hard":   {"ambulance", "rescue_team"},
}