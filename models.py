# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the AI Disaster Response Coordinator environment.

Models a real-world emergency triage problem:
  - Agent observes victims with urgency, distance, and survival time
  - Agent allocates limited rescue teams each step
  - Environment updates victim state and scores decisions
"""

from typing import Optional
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class VictimState(Observation):
    """Represents a single victim in the disaster scenario."""

    id: int = Field(..., description="Unique victim identifier")
    urgency: int = Field(..., description="Urgency level: 1=low, 2=medium, 3=critical")
    distance: float = Field(..., description="Distance from rescue base in km (1.0–10.0)")
    survival_time: int = Field(..., description="Steps remaining before victim dies")
    alive: bool = Field(default=True, description="Whether the victim is still alive")
    rescued: bool = Field(default=False, description="Whether the victim has been rescued")


class DisasterAction(Action):
    """
    Action for the Disaster Response environment.

    The agent chooses one victim to allocate a rescue team to.
    Use allocate_to=-1 to skip this step (incurs a small penalty).
    """

    allocate_to: int = Field(
        ...,
        description="Victim ID to rescue this step. Use -1 to skip.",
    )


class DisasterObservation(Observation):
    """
    Observation returned after each step / reset.

    Contains the full current state of the disaster scenario:
    active victims, available resources, time pressure, and
    whether the episode has ended.
    """

    time_step: int = Field(default=0, description="Current step number (0-indexed)")
    resources_available: int = Field(default=2, description="Rescue teams free this step")
    max_steps: int = Field(default=10, description="Total steps allowed this episode")
    difficulty: str = Field(default="medium", description="Episode difficulty: easy|medium|hard")
    victims: list[VictimState] = Field(default_factory=list, description="All victims and their status")
    episode_done: bool = Field(default=False, description="Whether the episode has terminated")
    last_action_info: Optional[dict] = Field(
        default=None,
        description="Info dict from the most recent step (rescued, died, invalid, etc.)"
    )