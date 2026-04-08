# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
AI Disaster Response Coordinator — Environment Implementation.

Wraps the GridWorld simulation (grid.py) inside the OpenEnv Environment
interface so it can be served over HTTP/WebSocket and evaluated by the
hackathon harness.

BUG 9 FIX: _build_observation previously collapsed the full resource dict
    into a single integer (rescue_teams + ambulances), discarding helicopter
    availability. Fixed to expose the full RESOURCE_CONFIG dict so the agent
    and inference script can make unit-type decisions correctly.

BUG 10 / 11 FIX: The environment was calling grade_episode() again after
    grid.step() had already computed and returned the final score via
    compute_reward(). This caused two problems:
      a) Duplicate computation with a potentially different zones_final
         (the grid had already advanced past tick()).
      b) The environment's grade_episode() call used self._grid._zones_initial
         which is the grid's own snapshot — consistent, but the env also
         maintained _zones_initial independently, creating two separate sources.
    Fix: trust grid.step()'s returned reward when done=True (it already IS
    the normalised final score from compute_reward()). Read grade metadata
    from result["info"]["episode_reward"]. Call grade_episode() only once —
    here in the environment — using the grid's own _zones_initial so the
    snapshot is guaranteed consistent with what grid.step() used.
"""

import copy
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
    """Convert victim dicts (from GridWorld) to VictimState objects."""
    return [
        VictimState(
            id=v["id"],
            urgency=v["urgency"],
            # BUG 8 FIX: grid stores "distance_km", VictimState field is "distance"
            distance=v["distance_km"],
            survival_time=v["survival_time"],
            alive=v["alive"],
            rescued=v["rescued"],
        )
        for v in victims_dicts
    ]


class DisasterEnvironment(Environment):
    """
    AI Disaster Response Coordinator — OpenEnv-compatible environment.

    Wraps the full GridWorld simulation (grid.py) inside the OpenEnv
    Environment interface.

    Difficulty levels
    -----------------
    easy   — 1 zone  (Kedarnath flash flood)
               7 victims, ambulances + rescue_team + helicopter,
               30 step budget, no spread, no spawn
    medium — 3 zones (Kedarnath, Badrinath, Rishikesh)
               8 victims/zone, disaster spreads every 5 steps, 40 steps
    hard   — 5 zones (Kedarnath, Joshimath, Dehradun, Haridwar, Chamoli)
               8 victims/zone, spreads every 2 steps, victims spawn
               every 5 steps, NO helicopters, 25 step budget

    Action
    ------
    DisasterAction(zone_id=<int>, unit_type=<"ambulance"|"rescue_team"|"helicopter">)

    Observation
    -----------
    DisasterObservation — full zone + victim state, full resources dict, step info.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._difficulty: str = "medium"
        self._seed: int = 42
        self._grid: GridWorld | None = None
        self._task: dict = {}
        self._done: bool = False
        self._episode_reward: float = 0.0

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(
        self,
        difficulty: str = "medium",
        seed: int | None = None,
    ) -> DisasterObservation:
        """
        Start a new episode.

        Args:
            difficulty: "easy" | "medium" | "hard"  (default: "medium")
            seed:       Optional int — uses the task's canonical seed when
                        None, ensuring reproducible evaluation scores.

        Returns:
            DisasterObservation with the initial scenario state.
        """
        valid = ("easy", "medium", "hard")
        if difficulty not in valid:
            raise ValueError(
                f"difficulty must be one of {list(valid)}, got '{difficulty}'"
            )

        self._difficulty = difficulty
        self._task = get_task(difficulty)
        self._seed = seed if seed is not None else self._task["seed"]

        self._grid = GridWorld(difficulty, self._seed)
        self._grid.reset()

        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._done = False
        self._episode_reward = 0.0

        return self._build_observation(reward=0.0, info=None)

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, action: DisasterAction) -> DisasterObservation:  # type: ignore[override]
        """
        Execute one rescue action.

        Args:
            action: DisasterAction(zone_id=<int>, unit_type=<str>)

        Returns:
            DisasterObservation — updated state, per-step reward, done flag.
            When done=True, reward IS the final normalised score [0.0, 1.0]
            and last_action_info contains the full grade_report.

        Raises:
            RuntimeError: if called after the episode has already ended.
        """
        if self._done or self._grid is None:
            raise RuntimeError(
                "Episode has ended. Call reset() to start a new episode."
            )

        self._state.step_count += 1

        grid_action = {
            "zone_id":   action.zone_id,
            "unit_type": action.unit_type,
        }

        # grid.step() handles: apply_action → calculate_step_reward → tick →
        # compute_reward (when done). When done=True, result["reward"] is
        # already the final normalised score — do NOT call grade_episode again.
        result = self._grid.step(grid_action)

        reward: float = result["reward"]
        done: bool    = result["done"]

        info: dict[str, Any] = {
            "action_valid":  result["info"]["action_valid"],
            "rescued_count": result["info"]["rescued_count"],
            "episode_reward": result["info"]["episode_reward"],
        }

        self._episode_reward = result["info"]["episode_reward"]
        self._done = done

        # BUG 10/11 FIX: grid.step() already ran compute_reward() when done=True
        # and returned the final normalised score as result["reward"].
        # We call grade_episode() once here for the full breakdown dict (score,
        # passed, breakdown, stats) — using the grid's own _zones_initial so
        # the snapshot is guaranteed identical to what compute_reward() used.
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

    # ── State ─────────────────────────────────────────────────────────────────

    @property
    def state(self) -> State:
        """OpenEnv State — episode_id and step_count."""
        return self._state

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_observation(
        self,
        reward: float,
        info: dict | None,
    ) -> DisasterObservation:
        """Construct a DisasterObservation from the current GridWorld state."""
        if self._grid is None:
            return DisasterObservation(
                time_step=0,
                resources_available=0,
                resources={},
                max_steps=0,
                difficulty=self._difficulty,
                zones=[],
                victims=[],
                episode_done=False,
                last_action_info=info,
            )

        grid_state = self._grid.get_state()
        zones_raw  = grid_state["zones"]

        # Flat victim list across all zones (VictimState objects)
        all_victims: list[VictimState] = []
        for z in zones_raw:
            all_victims.extend(_zone_victims_to_victim_states(z["victims"]))

        # BUG 9 FIX: expose the full resource dict, not a collapsed integer.
        # Old code: resources_available = resources.get("rescue_teams") + resources.get("ambulances")
        # This discarded helicopter count and gave the agent no way to know
        # what units were actually available.
        resource_config = RESOURCE_CONFIG[self._difficulty]
        resources_available = sum(resource_config.values())

        return DisasterObservation(
            time_step            = grid_state["step_num"],
            resources_available  = resources_available,
            resources            = dict(resource_config),
            max_steps            = STEP_LIMITS[self._difficulty],
            difficulty           = self._difficulty,
            zones                = zones_raw,    # already deepcopied by Zone.to_dict()
            victims              = all_victims,
            episode_done         = self._done,
            last_action_info     = info,
            done                 = self._done,
            reward               = reward,
            metadata             = {
                "episode_reward": self._episode_reward,
                "step":          grid_state["step_num"],
                "difficulty":    self._difficulty,
            },
        )