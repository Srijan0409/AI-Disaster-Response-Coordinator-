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

from disaster_env.models import DisasterAction, DisasterObservation, VictimState


class DisasterEnv(
    EnvClient[DisasterAction, DisasterObservation, State]
):
    """
    Client for the AI Disaster Response Coordinator environment.

    Maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session.

    Example:
        >>> with DisasterEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     obs = result.observation
        ...     print(obs.zones[0]["name"])       # "Kedarnath Temple Area"
        ...     print(obs.resources)              # {"ambulances": 2, ...}
        ...
        ...     result = client.step(
        ...         DisasterAction(zone_id=0, unit_type="helicopter")
        ...     )
        ...     print(result.reward)
        ...     print(result.observation.episode_done)

    Example with Docker:
        >>> client = DisasterEnv.from_docker_image("disaster_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     obs = result.observation
        ...     active = [z for z in obs.zones if z["is_active"]]
        ...     best   = max(active, key=lambda z: z["severity"])
        ...     result = client.step(
        ...         DisasterAction(zone_id=best["zone_id"], unit_type="rescue_team")
        ...     )
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: DisasterAction) -> Dict:
        """
        Convert DisasterAction to JSON payload for the step message.

        BUG 7 FIX: previously used action.model_dump() which serialises ALL
        pydantic fields including internal ones (e.g., done, reward inherited
        from Action base class). The server's action parser only expects the
        declared action fields — extra keys can cause validation errors or
        silent ignored fields depending on server config.
        Fix: explicitly send only the two action fields the server expects.

        Args:
            action: DisasterAction with zone_id and unit_type

        Returns:
            Dict with exactly the two keys the server expects
        """
        return {
            "zone_id":   action.zone_id,
            "unit_type": action.unit_type,
        }

    def _parse_result(self, payload: Dict) -> StepResult[DisasterObservation]:
        """
        Parse the server response into StepResult[DisasterObservation].

        BUG 8 FIX: grid.py stores victim distance as "distance_km" internally.
        VictimState has a field named "distance". The previous parse used
        v.get("distance", 1.0) which would always return the default 1.0 because
        the key is "distance_km" — silently zeroing out all victim distances.
        Fix: read "distance_km" from the raw dict, assign to VictimState.distance.

        Args:
            payload: JSON response data from the server

        Returns:
            StepResult containing a DisasterObservation
        """
        obs_data = payload.get("observation", {})

        # Rebuild VictimState list — victims are flattened across all zones
        raw_victims = obs_data.get("victims", [])
        victims = [
            VictimState(
                id            = v.get("id", 0),
                urgency       = v.get("urgency", 1),
                # BUG 8 FIX: grid key is "distance_km", not "distance"
                distance      = v.get("distance_km", v.get("distance", 1.0)),
                survival_time = v.get("survival_time", 0),
                alive         = v.get("alive", True),
                rescued       = v.get("rescued", False),
            )
            for v in raw_victims
        ]

        observation = DisasterObservation(
            time_step            = obs_data.get("time_step", 0),
            resources_available  = obs_data.get("resources_available", 0),
            resources            = obs_data.get("resources", {}),
            max_steps            = obs_data.get("max_steps", 30),
            difficulty           = obs_data.get("difficulty", "medium"),
            zones                = obs_data.get("zones", []),
            victims              = victims,
            episode_done         = obs_data.get("episode_done", False),
            last_action_info     = obs_data.get("last_action_info"),
            done                 = payload.get("done", False),
            reward               = payload.get("reward", 0.0),
            metadata             = obs_data.get("metadata", {}),
        )

        return StepResult(
            observation = observation,
            reward      = payload.get("reward", 0.0),
            done        = payload.get("done", False),
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
            episode_id = payload.get("episode_id"),
            step_count = payload.get("step_count", 0),
        )