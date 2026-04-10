# =============================================================================
# models.py — Data models for AI Disaster Response Coordinator
# =============================================================================
# Defines the Action and Observation types used by:
#   - disaster_env_environment.py  (environment layer)
#   - inference.py                 (agent layer)
#   - grid.py                      (simulation layer, indirectly via env)
#
# Field names must match exactly what grid.py / generators.py produce:
#   zone dict  : zone_id, name, district, disaster_type, severity, people,
#                rescued, time_waiting, is_active, victims
#   victim dict: id, urgency, survival_time, distance_km, alive, rescued
#
# VictimState.distance maps grid's "distance_km" field (BUG 8 FIX carried
# forward — field renamed at the boundary here, not in grid internals).
# =============================================================================

from typing import Any, Optional
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class VictimState(Observation):
    """
    Represents a single victim inside a disaster zone.

    Mirrors the victim dict produced by generators._generate_victims()
    and grid.Zone.to_dict(), with one rename:
        grid "distance_km"  →  model "distance"   (BUG 8 FIX)
    """

    id:            int   = Field(..., description="Globally unique victim ID across all zones")
    urgency:       int   = Field(..., description="Urgency level: 1=low, 2=medium, 3=critical")
    distance:      float = Field(..., description="Distance from rescue base in km")
    survival_time: int   = Field(..., description="Steps remaining before victim dies if unrescued")
    alive:         bool  = Field(default=True,  description="True if victim is still alive")
    rescued:       bool  = Field(default=False, description="True if victim has been rescued")


class DisasterAction(Action):
    """
    Action sent by the agent each step.

    zone_id   — 0-indexed zone to send the rescue unit to.
    unit_type — one of:
                  "ambulance"   → rescues 2 victims/step  (all difficulties)
                  "rescue_team" → rescues 3 victims/step  (all difficulties)
                  "helicopter"  → rescues 5 victims/step  (easy + medium ONLY)

    Hard mode rejects helicopters — apply_action() returns (False, 0) and
    the agent receives a -0.3 step penalty. ALLOWED_UNIT_TYPES in
    constants.py is the authoritative source for per-difficulty rules.
    """

    zone_id:   int = Field(
        ...,
        description="Zone ID to send the rescue unit to (0-indexed).",
    )
    unit_type: str = Field(
        default="rescue_team",
        description="Rescue unit: 'ambulance', 'rescue_team', or 'helicopter'.",
    )


class DisasterObservation(Observation):
    """
    Observation returned after every reset() and step().

    zones
        Full list of zone dicts straight from grid.Zone.to_dict().
        Each zone dict contains:
            zone_id, name, district, disaster_type,
            severity (float 0–1), people (int), rescued (int),
            time_waiting (int), is_active (bool),
            victims (list of victim dicts with the grid field names)

    victims
        Flat list of VictimState objects across ALL zones.
        Kept for backwards-compatibility with inference.py helpers that
        iterate victims without drilling into zones.

    resources
        Full resource dict from RESOURCE_CONFIG[difficulty]:
            {"ambulances": int, "rescue_teams": int, "helicopters": int}
        Never collapsed to a single integer (BUG 9 FIX).

    resources_available
        Sum of all resource counts — convenience field for quick checks.

    last_action_info
        None on reset(). On step():
            action_valid   : bool
            rescued_count  : int
            episode_reward : float  (accumulated so far)
            grade_report   : dict   (only on terminal step, contains score,
                                     passed, breakdown, stats)

    done / episode_done
        Both set to True on the terminal step. episode_done is the
        OpenEnv-spec field; done is a convenience alias.
    """

    time_step:           int              = Field(default=0,        description="Current step number")
    resources_available: int              = Field(default=0,        description="Total rescue units available")
    resources:           dict             = Field(default_factory=dict, description="Full resource dict")
    max_steps:           int              = Field(default=30,       description="Step budget for this episode")
    difficulty:          str              = Field(default="medium", description="easy | medium | hard")
    zones:               list[Any]        = Field(default_factory=list, description="All disaster zones")
    victims:             list[VictimState]= Field(default_factory=list, description="Flat victim list")
    episode_done:        bool             = Field(default=False,    description="True when episode has ended")
    done:                bool             = Field(default=False,    description="Alias for episode_done")
    reward:              float            = Field(default=0.0,      description="Reward from last step")
    last_action_info:    Optional[dict]   = Field(default=None,     description="Info dict from last step")
    metadata:            Optional[dict]   = Field(default=None,     description="Episode metadata")
