# =============================================================================
# models.py — Data models for AI Disaster Response Coordinator
# =============================================================================
#
# Defines the Action and Observation types used by the environment,
# inference script, and client.
#
# Field names match what grid.py / generators.py produce, with one rename:
#   victim dict: "distance_km"  →  VictimState: "distance"
#   (rename happens at the boundary in disaster_env_environment.py)
# =============================================================================

from typing import Any, Optional
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class VictimState(Observation):
    """Single victim inside a disaster zone."""

    id:            int   = Field(..., description="Globally unique victim ID across all zones")
    urgency:       int   = Field(..., description="1=low, 2=medium, 3=critical")
    distance:      float = Field(..., description="Distance from rescue base in km")
    survival_time: int   = Field(..., description="Steps remaining before victim dies if unrescued")
    alive:         bool  = Field(default=True,  description="True if victim is still alive")
    rescued:       bool  = Field(default=False, description="True if victim has been rescued")


class DisasterAction(Action):
    """
    Action sent by the agent each step.

    zone_id   — 0-indexed zone to send the rescue unit to.
    unit_type — "ambulance"   → rescues 2 victims/step  (all difficulties)
              | "rescue_team" → rescues 3 victims/step  (all difficulties)
              | "helicopter"  → rescues 5 victims/step  (easy + medium only)

    Hard mode has no helicopters — RESOURCE_CONFIG["hard"]["helicopters"] == 0.
    Sending a helicopter in hard mode returns (False, 0) from apply_action()
    and the agent receives a -0.3 step penalty.
    """

    zone_id:   int = Field(..., description="Zone ID to send the rescue unit to (0-indexed)")
    unit_type: str = Field(default="rescue_team",
                           description="'ambulance', 'rescue_team', or 'helicopter'")


class DisasterObservation(Observation):
    """
    Observation returned after every reset() and step().

    zones
        Full list of zone dicts from grid.Zone.to_dict(). Each contains:
        zone_id, name, district, disaster_type, severity (0–1), people,
        rescued, time_waiting, is_active, victims (list of victim dicts).

    victims
        Flat list of VictimState objects across all zones — kept for
        convenience so inference helpers can iterate without drilling into zones.

    resources
        Full dict from RESOURCE_CONFIG[difficulty]:
        {"ambulances": int, "rescue_teams": int, "helicopters": int}

    last_action_info
        None after reset(). After step() contains:
            action_valid, rescued_count, episode_reward.
        On the terminal step also contains grade_report with the full
        score breakdown.
    """

    time_step:           int               = Field(default=0)
    resources_available: int               = Field(default=0,
                                                   description="Sum of all resource counts")
    resources:           dict              = Field(default_factory=dict)
    max_steps:           int               = Field(default=30)
    difficulty:          str               = Field(default="medium")
    zones:               list[Any]         = Field(default_factory=list)
    victims:             list[VictimState] = Field(default_factory=list)
    episode_done:        bool              = Field(default=False)
    done:                bool              = Field(default=False,
                                                   description="Alias for episode_done")
    reward:              float             = Field(default=0.0)
    last_action_info:    Optional[dict]    = Field(default=None)
    metadata:            Optional[dict]    = Field(default=None)