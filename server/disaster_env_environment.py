# =============================================================================
# disaster_env_environment.py - OpenEnv Environment for AI Disaster Response
# =============================================================================
#
# Wraps GridWorld (grid.py) inside the OpenEnv Environment interface so it
# can be served over HTTP and evaluated by the hackathon harness.
#
# Design invariants:
#   1. Single source of truth for zones_initial: grid._zones_initial is used
#      by both compute_reward() (inside grid.step()) and grade_episode()
#      called here - no separate env-level snapshot.
#
#   2. No duplicate score computation: grid.step() already runs compute_reward()
#      when done=True and returns the final normalised score as result["reward"].
#      grade_episode() is called once here only to produce the full breakdown
#      dict for last_action_info. Both use _compute_score_components() so their
#      scores are always identical.
#
#   3. Full resource dict exposed: _build_observation() passes the complete
#      RESOURCE_CONFIG dict so the agent can make correct unit-type decisions.
#
#   4. Victim field rename: grid stores "distance_km"; VictimState uses
#      "distance". The rename happens once in _zone_victims_to_victim_states().
# =============================================================================

from typing import Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from disaster_env.models import DisasterAction, DisasterObservation, VictimState
from disaster_env.server.grid import GridWorld
from disaster_env.server.tasks import get_task
from disaster_env.server.grader import grade_episode
from disaster_env.server.constants import RESOURCE_CONFIG, STEP_LIMITS


def _zone_victims_to_victim_states(victims_dicts: list[dict]) -> list[VictimState]:
    """Convert victim dicts from GridWorld into VictimState objects."""
    return [
        VictimState(
            id=v["id"],
            urgency=v["urgency"],
            distance=v["distance_km"],   # grid field -> model field rename
            survival_time=v["survival_time"],
            alive=v["alive"],
            rescued=v["rescued"],
        )
        for v in victims_dicts
    ]


class DisasterEnvironment(Environment):
    """
    AI Disaster Response Coordinator - OpenEnv-compatible environment.

    Difficulty levels
    -----------------
    easy   - 1 zone  (Kedarnath), 7 victims, 30 steps, no spread/spawn
    medium - 3 zones, 8 victims/zone, 40 steps, spreads every 5 steps
    hard   - 5 zones, 8 victims/zone, 25 steps, spreads every 2 steps,
             victims spawn every 5 steps, NO helicopters

    Action: DisasterAction(zone_id=<int>, unit_type=<str>)
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        self._state          = State(episode_id=str(uuid4()), step_count=0)
        self._difficulty: str        = "medium"
        self._seed: int              = 123
        self._grid: GridWorld | None = None
        self._task: dict             = {}
        self._done: bool             = False
        self._episode_reward: float  = 0.0

    #  Reset 

    def reset(
        self,
        difficulty: str = "medium",
        seed: int | None = None,
    ) -> DisasterObservation:
        """
        Start a new episode.

        Args:
            difficulty: "easy" | "medium" | "hard"
            seed:       Optional override. When None, the task's canonical
                        seed is used (42 / 123 / 999) for reproducibility.
        """
        valid = ("easy", "medium", "hard")
        if difficulty not in valid:
            raise ValueError(f"difficulty must be one of {list(valid)}, got '{difficulty}'")

        self._difficulty     = difficulty
        self._task           = get_task(difficulty)
        self._seed           = seed if seed is not None else self._task["seed"]

        self._grid           = GridWorld(difficulty, self._seed)
        self._grid.reset()

        self._state          = State(episode_id=str(uuid4()), step_count=0)
        self._done           = False
        self._episode_reward = 0.0

        return self._build_observation(reward=0.0, info=None)

    #  Step 

    def step(self, action: DisasterAction) -> DisasterObservation:  # type: ignore[override]
        """
        Execute one rescue action.

        grid.step() handles: apply_action -> calculate_step_reward -> tick ->
        compute_reward (when done). When done=True, result["reward"] is the
        final normalised score. grade_episode() is called once here for the
        full breakdown dict.
        """
        if self._done or self._grid is None:
            raise RuntimeError("Episode has ended. Call reset() to start a new episode.")

        self._state.step_count += 1

        result = self._grid.step({"zone_id": action.zone_id, "unit_type": action.unit_type})

        reward: float = result["reward"]
        done: bool    = result["done"]

        info: dict[str, Any] = {
            "action_valid":   result["info"]["action_valid"],
            "rescued_count":  result["info"]["rescued_count"],
            "episode_reward": result["info"]["episode_reward"],
        }

        self._episode_reward = result["info"]["episode_reward"]
        self._done           = done

        if done:
            grade_report = grade_episode(
                task_level      = self._difficulty,
                zones_initial   = self._grid._zones_initial,
                zones_final     = self._grid.get_state()["zones"],
                steps_taken     = self._grid.step_num,
                spawned_victims = self._grid._spawned_victims,
                rescued_spawned = self._grid._rescued_spawned,
            )
            info["grade_report"] = grade_report

        return self._build_observation(reward=round(reward, 4), info=info)

    #  State 

    @property
    def state(self) -> State:
        """OpenEnv State - episode_id and step_count."""
        return self._state

    #  Internal helpers 

    def _build_observation(
        self,
        reward: float,
        info: dict | None,
    ) -> DisasterObservation:
        """Construct a DisasterObservation from the current GridWorld state."""
        if self._grid is None:
            return DisasterObservation(
                time_step=0, resources_available=0, resources={},
                max_steps=0, difficulty=self._difficulty,
                zones=[], victims=[], episode_done=False,
                done=False, reward=0.0, last_action_info=info,
            )

        grid_state = self._grid.get_state()
        zones_raw  = grid_state["zones"]

        all_victims: list[VictimState] = []
        for z in zones_raw:
            all_victims.extend(_zone_victims_to_victim_states(z["victims"]))

        resource_config     = RESOURCE_CONFIG[self._difficulty]
        resources_available = sum(resource_config.values())

        return DisasterObservation(
            time_step           = grid_state["step_num"],
            resources_available = resources_available,
            resources           = dict(resource_config),
            max_steps           = STEP_LIMITS[self._difficulty],
            difficulty          = self._difficulty,
            zones               = zones_raw,
            victims             = all_victims,
            episode_done        = self._done,
            done                = self._done,
            reward              = reward,
            last_action_info    = info,
            metadata            = {
                "episode_reward": self._episode_reward,
                "step":          grid_state["step_num"],
                "difficulty":    self._difficulty,
            },
        )