# =============================================================================
# disaster_env_environment.py — OpenEnv Environment for AI Disaster Response
# =============================================================================
#
# Wraps GridWorld (grid.py) inside the OpenEnv Environment interface so it
# can be served over HTTP and evaluated by the hackathon harness.
#
# Key design invariants
# ---------------------
# 1. Single source of truth for zones_initial:
#    grid._zones_initial is the ONLY snapshot used — both by compute_reward()
#    inside grid.step() and by grade_episode() called here. No separate env-
#    level snapshot exists (BUG 10/11 FIX).
#
# 2. No duplicate score computation:
#    When done=True, grid.step() already ran compute_reward() and returned the
#    final normalised score as result["reward"]. grade_episode() is called once
#    here solely to produce the full breakdown dict for last_action_info.
#    The score in grade_report will always equal result["reward"] because both
#    call _compute_score_components() — the shared core (BUG I FIX in grader).
#
# 3. Full resource dict exposed (BUG 9 FIX):
#    _build_observation() passes the complete RESOURCE_CONFIG dict, not a
#    collapsed integer, so the agent can make correct unit-type decisions.
#
# 4. Victim field mapping (BUG 8 FIX):
#    Grid stores "distance_km"; VictimState model uses "distance". The rename
#    happens exactly once — in _zone_victims_to_victim_states() — nowhere else.
#
# 5. Seed enforcement:
#    GridWorld.__init__() validates that the passed seed matches the task's
#    canonical seed (BUG 1 & 3 FIX in grid). reset() always uses the
#    task-canonical seed unless an explicit override is passed (e.g. for
#    testing). Passing the wrong seed raises ValueError immediately.
# =============================================================================

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


# ---------------------------------------------------------------------------
# Helper — victim dict → VictimState
# ---------------------------------------------------------------------------

def _zone_victims_to_victim_states(victims_dicts: list[dict]) -> list[VictimState]:
    """
    Convert victim dicts from GridWorld into VictimState objects.

    BUG 8 FIX: grid stores "distance_km"; VictimState uses "distance".
    This is the single rename point — grid internals are never touched.
    """
    return [
        VictimState(
            id=v["id"],
            urgency=v["urgency"],
            distance=v["distance_km"],        # BUG 8 FIX: field rename boundary
            survival_time=v["survival_time"],
            alive=v["alive"],
            rescued=v["rescued"],
        )
        for v in victims_dicts
    ]


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class DisasterEnvironment(Environment):
    """
    AI Disaster Response Coordinator — OpenEnv-compatible environment.

    Difficulty levels
    -----------------
    easy   — 1 zone  (Kedarnath flash flood)
               7 victims, ambulances + rescue_team + helicopter,
               30 step budget, no spread, no spawn.
    medium — 3 zones (Kedarnath, Badrinath, Rishikesh)
               8 victims/zone, spreads every 5 steps, 40 steps.
    hard   — 5 zones (Kedarnath, Joshimath, Dehradun, Haridwar, Chamoli)
               8 victims/zone, spreads every 2 steps, victims spawn every
               5 steps, NO helicopters, 25 step budget.

    Action
    ------
    DisasterAction(zone_id=<int>, unit_type=<"ambulance"|"rescue_team"|"helicopter">)

    Observation
    -----------
    DisasterObservation — full zone + victim state, full resource dict,
    step info, and (on terminal step) complete grade_report.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        self._state          = State(episode_id=str(uuid4()), step_count=0)
        self._difficulty: str          = "medium"
        self._seed: int                = 123       # medium canonical seed
        self._grid: GridWorld | None   = None
        self._task: dict               = {}
        self._done: bool               = False
        self._episode_reward: float    = 0.0

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(
        self,
        difficulty: str = "medium",
        seed: int | None = None,
    ) -> DisasterObservation:
        """
        Start a new episode.

        Args:
            difficulty : "easy" | "medium" | "hard"  (default: "medium")
            seed       : Optional override. When None the task's canonical
                         seed is used (42 / 123 / 999) — guarantees
                         reproducible evaluation scores.

        Returns:
            DisasterObservation with the initial scenario state.

        Raises:
            ValueError: if difficulty is not one of the three valid levels,
                        or if the passed seed doesn't match the canonical one
                        (raised inside GridWorld.__init__).
        """
        valid = ("easy", "medium", "hard")
        if difficulty not in valid:
            raise ValueError(
                f"difficulty must be one of {list(valid)}, got '{difficulty}'"
            )

        self._difficulty     = difficulty
        self._task           = get_task(difficulty)
        self._seed           = seed if seed is not None else self._task["seed"]

        self._grid           = GridWorld(difficulty, self._seed)
        self._grid.reset()

        self._state          = State(episode_id=str(uuid4()), step_count=0)
        self._done           = False
        self._episode_reward = 0.0

        return self._build_observation(reward=0.0, info=None)

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, action: DisasterAction) -> DisasterObservation:  # type: ignore[override]
        """
        Execute one rescue action.

        Internally calls grid.step() which:
            1. Captures zones_before snapshot (BUG 2 FIX)
            2. Runs apply_action()  — validates unit type (helicopter enforcement)
            3. Calls calculate_step_reward() on zones_before
            4. Calls tick()  — survival decay, spread, spawn
            5. When done=True: calls compute_reward() → final normalised score

        When done=True:
            - result["reward"] IS the final episode score from compute_reward().
            - grade_episode() is called once here for the full breakdown dict.
            - grade_report["score"] == result["reward"] guaranteed because
              both use _compute_score_components() (BUG I FIX in grader).

        Args:
            action: DisasterAction(zone_id=<int>, unit_type=<str>)

        Returns:
            DisasterObservation with updated state.
            last_action_info always contains: action_valid, rescued_count,
            episode_reward. On terminal step also contains grade_report.

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

        result = self._grid.step(grid_action)

        reward: float = result["reward"]
        done: bool    = result["done"]

        info: dict[str, Any] = {
            "action_valid":   result["info"]["action_valid"],
            "rescued_count":  result["info"]["rescued_count"],
            "episode_reward": result["info"]["episode_reward"],
        }

        self._episode_reward = result["info"]["episode_reward"]
        self._done           = done

        # BUG 10/11 FIX:
        # grid.step() already ran compute_reward() when done=True.
        # grade_episode() called once here — uses grid._zones_initial
        # (same snapshot compute_reward used) so score is guaranteed identical.
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

    def state(self) -> State:
        """OpenEnv State — episode_id and step_count."""
        return self._state

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_observation(
        self,
        reward: float,
        info: dict | None,
    ) -> DisasterObservation:
        """
        Construct a DisasterObservation from the current GridWorld state.

        BUG 9 FIX: exposes the full RESOURCE_CONFIG dict (ambulances,
        rescue_teams, helicopters) instead of collapsing to a single integer.
        resources_available is still provided as the sum for quick checks.

        Zone dicts are already deepcopied by Zone.to_dict() (FIX 17 in grid)
        so no additional copy is needed here.
        """
        # Grid not yet initialised (called before first reset)
        if self._grid is None:
            return DisasterObservation(
                time_step           = 0,
                resources_available = 0,
                resources           = {},
                max_steps           = 0,
                difficulty          = self._difficulty,
                zones               = [],
                victims             = [],
                episode_done        = False,
                done                = False,
                reward              = 0.0,
                last_action_info    = info,
            )

        grid_state = self._grid.get_state()
        zones_raw  = grid_state["zones"]

        # Flat VictimState list across all zones
        all_victims: list[VictimState] = []
        for z in zones_raw:
            all_victims.extend(_zone_victims_to_victim_states(z["victims"]))

        # BUG 9 FIX: full resource dict, never collapsed
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
