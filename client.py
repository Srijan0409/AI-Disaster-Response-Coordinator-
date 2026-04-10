# =============================================================================
# client.py - WebSocket client for AI Disaster Response Coordinator
# =============================================================================

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from disaster_env.models import DisasterAction, DisasterObservation, VictimState


class DisasterEnv(EnvClient[DisasterAction, DisasterObservation, State]):
    """
    Client for the AI Disaster Response Coordinator environment.

    Maintains a persistent WebSocket connection to the environment server.
    Each instance has its own dedicated episode session.

    Example:
        async with DisasterEnv(base_url="http://localhost:8000") as client:
            result = await client.reset(difficulty="easy", seed=42)
            obs = result.observation
            print(obs.zones[0]["name"])   # "Kedarnath Temple Area"
            print(obs.resources)          # {"ambulances": 2, ...}

            result = await client.step(
                DisasterAction(zone_id=0, unit_type="rescue_team")
            )
            print(result.reward)
            print(result.observation.episode_done)
    """

    def _step_payload(self, action: DisasterAction) -> Dict:
        """
        Convert DisasterAction to JSON payload for the step message.
        Sends only the two fields the server expects - avoids serialising
        inherited Pydantic base-class fields which can cause validation errors.
        """
        return {
            "zone_id":   action.zone_id,
            "unit_type": action.unit_type,
        }

    def _parse_result(self, payload: Dict) -> StepResult[DisasterObservation]:
        """
        Parse the server response into StepResult[DisasterObservation].

        Victim distance mapping: server stores "distance_km", VictimState
        uses "distance". Falls back to "distance" key for compatibility.
        """
        obs_data = payload.get("observation", {})

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
        return State(
            episode_id = payload.get("episode_id"),
            step_count = payload.get("step_count", 0),
        )