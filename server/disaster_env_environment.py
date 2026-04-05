# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
AI Disaster Response Coordinator — Environment Implementation.

A reinforcement learning environment that simulates emergency triage:
  - Victims have urgency (1–3), distance (km), and survival_time (steps)
  - The agent allocates limited rescue teams each step
  - Victims die if not rescued in time, triggering death penalties
  - Three difficulty levels (easy / medium / hard) control scenario parameters

This models a real Markov Decision Process:
  State:      victims + time_step + resources_available
  Action:     allocate_to (victim id, or -1 to skip)
  Transition: rescue victim, decay survival_time of others, kill if expired
  Reward:     urgency bonus − distance penalty − death penalties
"""

import random
from typing import Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import DisasterAction, DisasterObservation, VictimState
except ImportError:
    from models import DisasterAction, DisasterObservation, VictimState


# ─── Difficulty configuration ────────────────────────────────────────────────

DIFFICULTY_CONFIG: dict[str, dict] = {
    "easy": {
        "num_victims": 4,
        "num_resources": 2,
        "max_steps": 12,
        "survival_time_range": (6, 12),
        "urgency_weights": [0.5, 0.3, 0.2],   # mostly low urgency — forgiving
        "distance_range": (1.0, 5.0),
        "time_decay": 1,
    },
    "medium": {
        "num_victims": 5,
        "num_resources": 2,
        "max_steps": 10,
        "survival_time_range": (4, 9),
        "urgency_weights": [0.3, 0.4, 0.3],   # balanced
        "distance_range": (1.0, 8.0),
        "time_decay": 1,
    },
    "hard": {
        "num_victims": 7,
        "num_resources": 2,
        "max_steps": 8,
        "survival_time_range": (2, 6),
        "urgency_weights": [0.2, 0.3, 0.5],   # mostly critical — time pressure
        "distance_range": (1.0, 10.0),
        "time_decay": 2,                        # survival time drops 2/step
    },
}

# Reward values
URGENCY_REWARD  = {1: 0.3, 2: 0.6, 3: 1.0}
URGENCY_DEATH_PENALTY = {1: 0.4, 2: 0.7, 3: 1.0}


class DisasterEnvironment(Environment):
    """
    AI Disaster Response Coordinator environment.

    The agent must decide which victim to rescue each step, given:
      - Victims with different urgency levels (1=low, 2=medium, 3=critical)
      - Distances from the rescue base (affects rescue cost)
      - Survival times that count down every step
      - A fixed number of rescue teams per step

    Supports three difficulty levels:
      easy   — 4 victims, forgiving survival times, short distances
      medium — 5 victims, balanced scenario
      hard   — 7 victims, critical urgency dominates, fast time decay

    Episode ends when:
      - All victims are rescued or dead, OR
      - max_steps is reached

    Example:
        >>> env = DisasterEnvironment()
        >>> obs = env.reset()
        >>> print(obs.victims)           # list of VictimState
        >>> print(obs.resources_available)  # 2

        >>> obs2 = env.step(DisasterAction(allocate_to=0))
        >>> print(obs2.reward)           # reward for rescuing victim 0
        >>> print(obs2.episode_done)     # False (usually)
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialise with a default medium-difficulty scenario."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._difficulty: str = "medium"
        self._config: dict = DIFFICULTY_CONFIG["medium"]
        self._rng: random.Random = random.Random()

        # Live episode state
        self._victims: list[VictimState] = []
        self._time_step: int = 0
        self._resources_available: int = 2
        self._done: bool = False
        self._episode_reward: float = 0.0
        self._rescued_ids: set[int] = set()
        self._dead_ids: set[int] = set()

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(
        self,
        difficulty: str = "medium",
        seed: int | None = None,
    ) -> DisasterObservation:
        """
        Generate a new random disaster scenario.

        Args:
            difficulty: "easy" | "medium" | "hard"
            seed:       Optional int for reproducible scenarios

        Returns:
            DisasterObservation with the initial scenario state
        """
        if difficulty not in DIFFICULTY_CONFIG:
            raise ValueError(
                f"difficulty must be one of {list(DIFFICULTY_CONFIG.keys())}, got '{difficulty}'"
            )

        self._difficulty = difficulty
        self._config = DIFFICULTY_CONFIG[difficulty]
        self._rng = random.Random(seed)

        # Reset episode counters
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._time_step = 0
        self._resources_available = self._config["num_resources"]
        self._done = False
        self._episode_reward = 0.0
        self._rescued_ids = set()
        self._dead_ids = set()

        # Generate victims
        cfg = self._config
        st_min, st_max = cfg["survival_time_range"]
        d_min, d_max = cfg["distance_range"]

        self._victims = []
        for i in range(cfg["num_victims"]):
            urgency = self._rng.choices([1, 2, 3], weights=cfg["urgency_weights"])[0]
            distance = round(self._rng.uniform(d_min, d_max), 2)
            survival_time = self._rng.randint(st_min, st_max)
            self._victims.append(
                VictimState(
                    id=i,
                    urgency=urgency,
                    distance=distance,
                    survival_time=survival_time,
                )
            )

        return self._make_observation(reward=0.0, info=None)

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, action: DisasterAction) -> DisasterObservation:  # type: ignore[override]
        """
        Execute one rescue allocation step.

        Args:
            action: DisasterAction with allocate_to = victim id (or -1 to skip)

        Returns:
            DisasterObservation with updated state, reward, done flag

        Raises:
            RuntimeError: if called after episode has ended
        """
        if self._done:
            raise RuntimeError("Episode has ended. Call reset() to start a new one.")

        self._state.step_count += 1
        reward = 0.0
        info: dict[str, Any] = {
            "rescued": False,
            "died": [],
            "invalid": False,
            "skipped": False,
        }

        victim_id = action.allocate_to

        # ── Validate and execute allocation ───────────────────────────────────
        if victim_id == -1:
            reward -= 0.05
            info["skipped"] = True

        elif self._resources_available <= 0:
            reward -= 0.1
            info["invalid"] = True
            info["reason"] = "no_resources"

        else:
            target = self._find_victim(victim_id)

            if target is None:
                reward -= 0.15
                info["invalid"] = True
                info["reason"] = "victim_not_found"

            elif not target.alive or target.rescued:
                reward -= 0.15
                info["invalid"] = True
                info["reason"] = "victim_unavailable"

            else:
                # ── Successful rescue ─────────────────────────────────────
                target.rescued = True
                target.alive = False
                self._resources_available -= 1
                self._rescued_ids.add(victim_id)

                # Reward: urgency bonus − distance penalty + time bonus
                urgency_reward = URGENCY_REWARD[target.urgency]
                distance_penalty = min(0.3, target.distance / 30.0)
                time_bonus = 0.1 if target.survival_time > 3 else 0.0

                reward += urgency_reward - distance_penalty + time_bonus
                info["rescued"] = True
                info["rescued_id"] = victim_id
                info["urgency"] = target.urgency

        # ── Advance time: decay survival_time of all living victims ───────────
        self._time_step += 1
        self._resources_available = self._config["num_resources"]  # reset each step
        decay = self._config["time_decay"]

        died_this_step: list[int] = []
        for v in self._victims:
            if not v.alive or v.rescued:
                continue
            v.survival_time -= decay
            if v.survival_time <= 0:
                v.alive = False
                self._dead_ids.add(v.id)
                died_this_step.append(v.id)
                # Death penalty proportional to urgency
                reward -= 0.5 * URGENCY_DEATH_PENALTY[v.urgency]

        info["died"] = died_this_step

        # ── Check episode termination ──────────────────────────────────────────
        living = [v for v in self._victims if v.alive and not v.rescued]
        if self._time_step >= self._config["max_steps"] or not living:
            self._done = True

        self._episode_reward += reward
        return self._make_observation(reward=round(reward, 4), info=info)

    # ── State ─────────────────────────────────────────────────────────────────

    @property
    def state(self) -> State:
        """Current OpenEnv State with episode_id and step_count."""
        return self._state

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _make_observation(
        self,
        reward: float,
        info: dict | None,
    ) -> DisasterObservation:
        return DisasterObservation(
            time_step=self._time_step,
            resources_available=self._resources_available,
            max_steps=self._config["max_steps"],
            difficulty=self._difficulty,
            victims=list(self._victims),
            episode_done=self._done,
            last_action_info=info,
            done=self._done,
            reward=reward,
            metadata=self._episode_summary(),
        )

    def _find_victim(self, victim_id: int) -> VictimState | None:
        for v in self._victims:
            if v.id == victim_id:
                return v
        return None

    def _episode_summary(self) -> dict:
        total = len(self._victims)
        return {
            "total_victims": total,
            "rescued": len(self._rescued_ids),
            "dead": len(self._dead_ids),
            "still_alive": total - len(self._rescued_ids) - len(self._dead_ids),
            "episode_reward": round(self._episode_reward, 4),
            "steps_taken": self._time_step,
            "difficulty": self._difficulty,
        }