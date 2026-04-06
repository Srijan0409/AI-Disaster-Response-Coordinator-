# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""AI Disaster Response Coordinator — Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import DisasterAction, DisasterObservation, VictimState


class DisasterEnv(
    EnvClient[DisasterAction, DisasterObservation, State]
):
    """
    Client for the AI Disaster Response Coordinator environment.

    Maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session.

    The agent must rescue victims with limited teams under time pressure.
    Each step, allocate one rescue team to a victim by their ID.

    Example:
        >>> with DisasterEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     obs = result.observation
        ...     print(obs.victims)              # list of VictimState
        ...     print(obs.resources_available)  # 2
        ...
        ...     # Rescue victim 0
        ...     result = client.step(DisasterAction(allocate_to=0))
        ...     print(result.reward)
        ...     print(result.observation.episode_done)

    Example with Docker:
        >>> client = DisasterEnv.from_docker_image("disaster_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     # Pick the most critical victim
        ...     alive = [v for v in result.observation.victims if v.alive]
        ...     target = max(alive, key=lambda v: v.urgency)
        ...     result = client.step(DisasterAction(allocate_to=target.id))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: DisasterAction) -> Dict:
        """
        Convert DisasterAction to JSON payload for the step message.

        Args:
            action: DisasterAction with allocate_to victim id (or -1 to skip)

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "allocate_to": action.allocate_to,
        }

    def _parse_result(self, payload: Dict) -> StepResult[DisasterObservation]:
        """
        Parse the server response into StepResult[DisasterObservation].

        Args:
            payload: JSON response data from the server

        Returns:
            StepResult containing a DisasterObservation
        """
        obs_data = payload.get("observation", {})

        # Re-inflate the victims list from raw dicts
        raw_victims = obs_data.get("victims", [])
        victims = [
            VictimState(
                id=v.get("id", 0),
                urgency=v.get("urgency", 1),
                distance=v.get("distance", 1.0),
                survival_time=v.get("survival_time", 0),
                alive=v.get("alive", False),
                rescued=v.get("rescued", False),
            )
            for v in raw_victims
        ]

        observation = DisasterObservation(
            time_step=obs_data.get("time_step", 0),
            resources_available=obs_data.get("resources_available", 0),
            max_steps=obs_data.get("max_steps", 10),
            difficulty=obs_data.get("difficulty", "medium"),
            victims=victims,
            episode_done=obs_data.get("episode_done", False),
            last_action_info=obs_data.get("last_action_info"),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse the server response into a State object.

        Args:
            payload: JSON response from the /state request

        Returns:
            State with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )