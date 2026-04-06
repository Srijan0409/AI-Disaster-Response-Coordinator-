# =============================================================================
# CONSTANTS — AI Disaster Response Coordinator
# =============================================================================
# Shared constants used by both grid.py and generators.py.
# Kept in a separate file to avoid circular imports and ensure
# both files always use identical ranges and configurations.
# =============================================================================

# Real Uttarakhand disaster locations — Based on 2013 Kedarnath floods scenario
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
# Used by both grid.py and generators.py — must stay in sync
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
# Hard mode has no helicopters — forces harder triage decisions
RESOURCE_CONFIG = {
    "easy":   {"ambulances": 2, "rescue_teams": 1, "helicopters": 1},
    "medium": {"ambulances": 3, "rescue_teams": 2, "helicopters": 1},
    "hard":   {"ambulances": 2, "rescue_teams": 1, "helicopters": 0},
}

# Maximum steps allowed per episode per difficulty level
# Hard has only 35 steps but 5 zones — intentional time pressure
STEP_LIMITS = {
    "easy":   30,
    "medium": 40,
    "hard":   35,
}