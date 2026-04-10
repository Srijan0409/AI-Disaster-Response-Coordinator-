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

    Connects via WebSocket to the environment server (local or HF Space),
    providing an async step()/reset()/state() interface.

    Difficulty levels
    -----------------
    easy   — 1 zone,  7 victims, 30 steps, no spread/spawn
    medium — 3 zones, 8 victims/zone, 40 steps, spreads every 5 steps
    hard   — 5 zones, 8 victims/zone, 25 steps, spreads every 2 steps,
              victims spawn every 5 steps, NO helicopters

    Unit types
    ----------
    ambulance   — rescues 2 victims/step  (all difficulties)
    rescue_team — rescues 3 victims/step  (all difficulties)
    helicopter  — rescues 5 victims/step  (easy + medium ONLY)

    Example (async):
        async with DisasterEnv(base_url="http://localhost:8000") as client:
            result = await client.reset(difficulty="easy", seed=42)
            obs = result.observation
            print(obs.zones[0]["name"])      # "Kedarnath Temple Area"
            print(obs.resources)            # {"ambulances": 2, ...}

            result = await client.step(
                DisasterAction(zone_id=0, unit_type="rescue_team")
            )
            print(result.reward)
            print(result.observation.episode_done)

    Example (sync wrapper):
        with DisasterEnv(base_url="http://localhost:8000").sync() as client:
            result = client.reset(difficulty="medium", seed=123)
            obs = result.observation
            active = [z for z in obs.zones if z["is_active"]]
            best = max(active, key=lambda z: z["severity"])
            result = client.step(
                DisasterAction(zone_id=best["zone_id"], unit_type="rescue_team")
            )
    """

    def _step_payload(self, action: DisasterAction) -> Dict:
        """
        Convert DisasterAction to JSON payload for the step message.

        Sends only the two fields the server expects — avoids serialising
        inherited Pydantic base-class fields (done, reward, etc.) which
        can cause server-side validation errors.

        Args:
            action: DisasterAction with zone_id and unit_type

        Returns:
            Dict with exactly the keys the server expects.
        """
        return {
            "zone_id":   action.zone_id,
            "unit_type": action.unit_type,
        }

    def _parse_result(self, payload: Dict) -> StepResult[DisasterObservation]:
        """
        Parse the server response into StepResult[DisasterObservation].

        Victim distance field mapping:
            grid/server stores "distance_km"
            VictimState model uses "distance"
            Falls back to "distance" key for forward-compatibility.

        Zones are passed through as-is — they are already deepcopied
        by Zone.to_dict() server-side (FIX 17 in grid.py).

        Args:
            payload: JSON response data from the server

        Returns:
            StepResult containing a DisasterObservation.
        """
        obs_data = payload.get("observation", {})

        # Rebuild flat VictimState list from raw victim dicts.
        # Victims in obs_data come from _zone_victims_to_victim_states()
        # in disaster_env_environment.py which already maps distance_km→distance.
        # The fallback chain handles both old and new server versions.
        raw_victims = obs_data.get("victims", [])
        victims = [
            VictimState(
                id            = v.get("id", 0),
                urgency       = v.get("urgency", 1),
                distance      = v.get("distance", v.get("distance_km", 1.0)),
                survival_time = v.get("survival_time", 0),
                alive         = v.get("alive", True),
                rescued       = v.get("rescued", False),
            )
            for v in raw_victims
        ]

        observation = DisasterObservation(
            time_step           = obs_data.get("time_step", 0),
            resources_available = obs_data.get("resources_available", 0),
            resources           = obs_data.get("resources", {}),
            max_steps           = obs_data.get("max_steps", 30),
            difficulty          = obs_data.get("difficulty", "medium"),
            zones               = obs_data.get("zones", []),
            victims             = victims,
            episode_done        = obs_data.get("episode_done", False),
            done                = payload.get("done", False),
            reward              = payload.get("reward", 0.0),
            last_action_info    = obs_data.get("last_action_info"),
            metadata            = obs_data.get("metadata", {}),
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
            payload: JSON response from the /state endpoint.

        Returns:
            State with episode_id and step_count.
        """
        return State(
            episode_id = payload.get("episode_id"),
            step_count = payload.get("step_count", 0),
        )
