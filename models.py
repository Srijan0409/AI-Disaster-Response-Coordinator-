# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the AI Disaster Response Coordinator environment.

Models a real-world multi-zone emergency triage problem:
  - Agent observes disaster zones (each with victims, severity, resources)
  - Agent sends a rescue unit (ambulance / rescue_team / helicopter) to a zone
  - Environment updates zone + victim state and scores decisions using grader.py
"""

from typing import Any, Optional
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class VictimState(Observation):
    """Represents a single victim inside a disaster zone."""

    id: int = Field(..., description="Unique victim identifier")
    urgency: int = Field(..., description="Urgency level: 1=low, 2=medium, 3=critical")
    distance: float = Field(..., description="Distance from rescue base in km (1.0-10.0)")
    survival_time: int = Field(..., description="Steps remaining before victim dies if not rescued")
    alive: bool = Field(default=True, description="Whether the victim is still alive")
    rescued: bool = Field(default=False, description="Whether the victim has been rescued")


class DisasterAction(Action):
    """
    Action for the Disaster Response environment.

    The agent selects a zone and a rescue unit type to deploy.

    zone_id   — which zone to send the unit to (0-indexed)
    unit_type — "ambulance"   (rescues 2 victims per step)
              | "rescue_team" (rescues 3 victims per step)
              | "helicopter"  (rescues 5 victims per step — NOT available in hard mode)
    """

    zone_id: int = Field(
        ...,
        description="Zone ID to send the rescue unit to (0-indexed).",
    )
    unit_type: str = Field(
        default="rescue_team",
        description="Rescue unit type: 'ambulance', 'rescue_team', or 'helicopter'.",
    )


class DisasterObservation(Observation):
    """
    Observation returned after each step / reset.

    Contains the full current state of all disaster zones, the flat
    victim list across all zones, resource availability, step budget,
    and termination flag.

    zones   — list of zone dicts, each with:
                zone_id, name, district, disaster_type,
                severity (0.0-1.0), people, rescued,
                time_waiting, is_active, victims (list of victim dicts)
    victims — flat list of ALL victims across all zones (VictimState objects),
              kept for backwards-compatibility with the inference script.

    BUG 12 FIX: zones was typed as list[Any] — changed to list[dict] so
    callers know they get dicts with string keys. Using list[Any] silently
    allowed the greedy_fallback and LLM prompt builder to make wrong
    assumptions about the access pattern (attribute vs dict access).
    """

    time_step: int = Field(default=0, description="Current step number (0-indexed)")
    resources_available: int = Field(default=2, description="Total rescue units available this step")
    resources: dict = Field(
        default_factory=dict,
        description="Full resource dict: {ambulances, rescue_teams, helicopters}",
    )
    max_steps: int = Field(default=30, description="Total steps allowed this episode")
    difficulty: str = Field(default="medium", description="Episode difficulty: easy|medium|hard")
    zones: list[dict] = Field(
        default_factory=list,
        description=(
            "All disaster zones with severity, people, rescued, time_waiting, "
            "is_active, and individual victim dicts."
        ),
    )
    victims: list[VictimState] = Field(
        default_factory=list,
        description="Flat list of all victims across all zones.",
    )
    episode_done: bool = Field(default=False, description="Whether the episode has terminated")
    last_action_info: Optional[dict] = Field(
        default=None,
        description=(
            "Info dict from the most recent step: action_valid, rescued_count, "
            "episode_reward, and (on terminal step) grade_report."
        ),
    )