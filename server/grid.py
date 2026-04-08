import random
import copy
from disaster_env.server.constants import (
    UTTARAKHAND_ZONES, SEVERITY_RANGES, PEOPLE_RANGES,
    VICTIM_TIME_DECAY, VICTIM_URGENCY_WEIGHTS,
    VICTIM_SURVIVAL_TIME, VICTIM_DISTANCE_KM,
)
from disaster_env.server.generators import _generate_victims
from disaster_env.server.tasks import get_task
from disaster_env.server.grader import calculate_step_reward, compute_reward

class Zone:
    def __init__(self, zone_id, name, district, disaster_type, severity):
        self.zone_id       = zone_id
        self.name          = name
        self.district      = district
        self.disaster_type = disaster_type
        self.severity      = severity
        self.victims       = []
        self.people        = 0   # derived from victims — single source of truth
        self.rescued       = 0
        self.time_waiting  = 0

    def to_dict(self):
        return {
            "zone_id":       self.zone_id,
            "name":          self.name,
            "district":      self.district,
            "disaster_type": self.disaster_type,
            "severity":      round(self.severity, 2),
            "people":        self.people,
            "rescued":       self.rescued,
            "time_waiting":  self.time_waiting,
            "is_active":     self.people > self.rescued,
            "victims":       copy.deepcopy(self.victims),
        }


class GridWorld:
    """
    Simulates a multi-zone disaster scenario across Uttarakhand.
    Supports three difficulty levels: easy, medium, and hard.
    Uses a fixed random seed to ensure reproducible scenarios.

    Fixes applied (original):
        1. spawn_new_victims() uses VICTIM_URGENCY_WEIGHTS for urgency.
        2. spawn_new_victims() uses VICTIM_SURVIVAL_TIME and VICTIM_DISTANCE_KM.
        3. step() raises RuntimeError if called after episode is done.
        4. spawn_new_victims() uses a global _next_victim_id counter.
        5. apply_action() does NOT set alive=False on rescued victims.
        6. step() imports moved to module level (lazy but cached by sys.modules).
        7. spawn_new_victims() active_zone check uses alive victim count.
        8. reset() uses a module-level Random instance for severity generation.

    Additional fixes (previous revision):
        17. Zone.to_dict() uses copy.deepcopy(self.victims) — prevents external
            mutation of the grid's internal victim state through returned dicts.
        16. step() updates self._episode_reward with the final compute_reward()
            result when done=True, so info["episode_reward"] always reflects
            the true episode score on the terminal step.

    Additional fixes (this revision):
        18. step() no longer clamps the per-step reward to [0.0, 1.0].
            calculate_step_reward() returns negative values (-0.3, -0.1) for
            invalid or wasted actions. Clamping these to 0.0 made bad actions
            indistinguishable from low-value actions, silently hiding penalties.
            Only the final episode reward (compute_reward) is clamped to [0, 1].
        19. spawn_new_victims() derives zone.people from len(zone.victims) after
            appending new victims instead of manually incrementing zone.people.
            Manual increment broke the "victims = single source of truth" contract
            established in reset() (zone.people = len(zone.victims)). A mismatch
            between zone.people and len(zone.victims) would cause incorrect
            is_active, rescued-ratio, and scoring calculations downstream.
    """

    def __init__(self, task_level, seed):
        self.task_level = task_level
        self.seed       = seed
        self.zones      = []
        self.step_num   = 0
        # FIX 8: isolated RNG for severity — does not touch global random state
        self._severity_rng    = random.Random(seed)
        self.rng              = random.Random(seed + 1)

        self._done            = False
        self._episode_reward  = 0.0
        self._zones_initial   = []
        self._spawned_victims = 0
        self._rescued_spawned = 0
        self._spawned_ids     = set()
        # FIX 4: global counter — IDs unique across all zones
        self._next_victim_id  = 0

    def reset(self):
        """
        Resets the environment to its initial state.
        Uses the fixed seed so the same scenario is generated every time.
        Generates individual victims for each zone.
        people = len(victims) — victims are the single source of truth.
        Returns the initial state as a dictionary.
        """
        # FIX 8: re-seed isolated RNG instances instead of global random
        self._severity_rng    = random.Random(self.seed)
        self.rng              = random.Random(self.seed + 1)
        self.zones    = []
        self.step_num = 0

        self._done            = False
        self._episode_reward  = 0.0
        self._spawned_victims = 0
        self._rescued_spawned = 0
        self._spawned_ids     = set()

        zone_templates   = UTTARAKHAND_ZONES[self.task_level]
        sev_min, sev_max = SEVERITY_RANGES[self.task_level]

        for i, template in enumerate(zone_templates):

            if self.task_level == "hard" and i == 0:
                severity = 1.0
            else:
                # FIX 8: use isolated _severity_rng, not global random
                severity = round(self._severity_rng.uniform(sev_min, sev_max), 2)

            zone = Zone(
                zone_id=i,
                name=template["name"],
                district=template["district"],
                disaster_type=template["disaster_type"],
                severity=severity,
            )

            zone.victims = _generate_victims(self.task_level, i, self.seed)
            zone.people  = len(zone.victims)   # victims = source of truth

            self.zones.append(zone)

        # FIX 4: initialise global ID counter after all zones are created
        self._next_victim_id = (
            max((v["id"] for z in self.zones for v in z.victims), default=-1) + 1
        )

        # NOTE: get_state() now returns deepcopied victim dicts (FIX 17),
        # so _zones_initial is a fully independent snapshot — mutations
        # during tick() (survival_time decay, alive=False) do NOT bleed into it.
        self._zones_initial = copy.deepcopy(self.get_state()["zones"])
        return self.get_state()

    def spawn_new_victims(self):
        """
        Spawns additional trapped people in random active zones.
        Only triggered in hard mode every 5 steps.

        FIX 1: urgency uses VICTIM_URGENCY_WEIGHTS[task_level].
        FIX 2: survival_time and distance_km use VICTIM_SURVIVAL_TIME and
                VICTIM_DISTANCE_KM constants per task_level.
        FIX 4: uses self._next_victim_id (global counter) so spawned IDs
                are unique across all zones.
        FIX 7: active_zones filtered by alive unrescued victim count —
                prevents spawning into zones where all victims are dead
                (unrescued) which would waste spawn budget and distort
                spawn_penalty calculation.
        FIX 19: zone.people derived from len(zone.victims) after appending
                new victims — maintains single source of truth contract.
                Previously zone.people was manually incremented by new_count
                which could diverge from the actual victim list length if any
                exception or early return occurred mid-loop.
        """
        # FIX 7: only spawn into zones that still have alive unrescued victims
        active_zones = [
            z for z in self.zones
            if any(v["alive"] and not v["rescued"] for v in z.victims)
        ]
        if not active_zones:
            return

        target_zone = self.rng.choice(active_zones)
        new_count   = self.rng.randint(2, 5)

        urgency_weights = VICTIM_URGENCY_WEIGHTS[self.task_level]
        st_min, st_max  = VICTIM_SURVIVAL_TIME[self.task_level]
        d_min, d_max    = VICTIM_DISTANCE_KM[self.task_level]

        for _ in range(new_count):
            # FIX 4: global unique ID — no collision across zones
            vid = self._next_victim_id
            self._next_victim_id += 1

            new_victim = {
                "id":            vid,
                "urgency":       self.rng.choices([1, 2, 3], weights=urgency_weights)[0],
                "survival_time": self.rng.randint(st_min, st_max),
                "distance_km":   round(self.rng.uniform(d_min, d_max), 2),
                "alive":         True,
                "rescued":       False,
            }
            target_zone.victims.append(new_victim)
            self._spawned_ids.add(vid)

        self._spawned_victims += new_count
        # FIX 19: derive people from len(victims) — single source of truth.
        # Old code: target_zone.people += new_count
        # Risk: manual increment diverges from victim list if loop exits early.
        target_zone.people = len(target_zone.victims)

    def tick(self, spread=False, spread_interval=5, spawn_victims=False):
        """
        Advances the simulation by one time step.
        - Increments the step counter.
        - Increases waiting time for all unrescued zones.
        - Decays survival_time for all living, unrescued victims.
        - Marks victims as dead (alive=False) if survival_time <= 0.
        - Triggers threat spread if enabled and interval is reached.
        - Spawns new victims in hard mode every 5 steps.
        """
        self.step_num += 1
        decay = VICTIM_TIME_DECAY[self.task_level]

        for zone in self.zones:
            if zone.people > zone.rescued:
                zone.time_waiting += 1

            for v in zone.victims:
                if v["alive"] and not v["rescued"]:
                    v["survival_time"] -= decay
                    if v["survival_time"] <= 0:
                        v["alive"] = False

        if spread and self.step_num % spread_interval == 0:
            self.spread_threat()

        if spawn_victims and self.step_num % 5 == 0:
            self.spawn_new_victims()

    def spread_threat(self):
        """
        Simulates disaster spreading to adjacent zones.
        If a zone has severity above 0.6, the next zone's severity
        increases by 0.1 (capped at 1.0).
        """
        for i in range(len(self.zones) - 1):
            if self.zones[i].severity > 0.6:
                self.zones[i + 1].severity = min(
                    1.0,
                    self.zones[i + 1].severity + 0.1
                )
                self.zones[i + 1].time_waiting += 1

    def apply_action(self, zone_id, unit_type):
        """
        Applies a rescue action to a zone — rescues individual victims.

        Rescue amounts per unit type:
            ambulance    → rescues 2 victims
            rescue_team  → rescues 3 victims
            helicopter   → rescues 5 victims

        Victims rescued in priority order: highest urgency first,
        then lowest survival_time (most critical first).

        FIX 5: rescued victims keep alive=True — previously both rescued=True
                and alive=False were set, making rescued victims
                indistinguishable from dead ones when iterating alive=False
                victims downstream.
        """
        RESCUE_AMOUNTS = {
            "ambulance":   2,
            "rescue_team": 3,
            "helicopter":  5,
        }

        zone = next((z for z in self.zones if z.zone_id == zone_id), None)

        if zone is None or unit_type not in RESCUE_AMOUNTS:
            return False, 0

        available = [v for v in zone.victims if v["alive"] and not v["rescued"]]
        if not available:
            return False, 0

        available.sort(key=lambda v: (-v["urgency"], v["survival_time"]))

        slots         = RESCUE_AMOUNTS[unit_type]
        rescued_count = 0

        for v in available[:slots]:
            v["rescued"] = True
            # FIX 5: do NOT set alive=False — rescued victims stay alive=True
            # so they are distinguishable from dead (unrescued) victims.
            rescued_count += 1

            if v["id"] in self._spawned_ids:
                self._rescued_spawned += 1

        zone.rescued = min(zone.people, zone.rescued + rescued_count)
        return True, rescued_count

    def step(self, action: dict) -> dict:
        """
        OpenEnv-compatible step().
        Accepts action dict: {"zone_id": int, "unit_type": str}

        Raises RuntimeError if called after episode is already done.

        Returns observation dict with current state, reward, done flag.

        FIX 6: tasks and grader are imported at module level (top of file)
                to avoid repeated import calls per step and stale-module risk.
        FIX 16: self._episode_reward is updated with the final compute_reward()
                result when done=True, so info["episode_reward"] on the terminal
                step always reflects the true final episode score — not just the
                sum of intermediate step rewards which uses a different scale.
        FIX 18: Per-step reward is NOT clamped to [0.0, 1.0].
                calculate_step_reward() returns -0.3 for invalid actions and
                -0.1 for wasted steps. The previous clamp (max(0.0, ...)) zeroed
                these out, making bad actions appear the same as low-value actions
                and removing the learning signal for the agent. Only the final
                episode reward from compute_reward() is bounded to [0, 1].
        """
        if self._done:
            raise RuntimeError(
                "step() called after episode is already done. "
                "Call reset() to start a new episode."
            )

        # FIX 6: use module-level imports (moved to top of step body —
        # kept here as local names to preserve the lazy-import pattern
        # that breaks the circular dependency, but imported only once
        # per call stack frame — Python caches sys.modules so no overhead).
        

        task = get_task(self.task_level)

        zone_id   = action.get("zone_id")
        unit_type = action.get("unit_type", "rescue_team")

        action_valid, rescued_count = self.apply_action(zone_id, unit_type)
        zones_now = [z.to_dict() for z in self.zones]

        step_reward = calculate_step_reward(zone_id, zones_now, action_valid)
        # FIX 18: do NOT clamp per-step reward — negative values (-0.3 invalid,
        # -0.1 wasted) are meaningful signals that must reach the caller.
        # Old code: reward = round(max(0.0, min(1.0, step_reward)), 4)
        reward = round(step_reward, 4)

        self.tick(
            spread=task.get("spread", False),
            spread_interval=task.get("spread_interval", 5) or 5,
            spawn_victims=task.get("spawn_victims", False),
        )

        self._episode_reward += reward

        zones_after = [z.to_dict() for z in self.zones]

        # FIX 5 consequence: all_done correctly checks rescued flag,
        # not alive — a rescued victim has alive=True but rescued=True.
        all_done = all(
            v["rescued"] or not v["alive"]
            for z in self.zones
            for v in z.victims
        )

        self._done = self.step_num >= task["max_steps"] or all_done

        if self._done:
            reward = compute_reward(
                self.task_level,
                self._zones_initial,
                zones_after,
                self.step_num,
                self._spawned_victims,
                self._rescued_spawned,
            )
            # BUG 16 FIX: replace the accumulated step reward with the final
            # episode score so info["episode_reward"] is meaningful on the
            # terminal step. Without this, the last step's episode_reward was
            # the sum of per-step rewards (different scale/meaning), while the
            # returned reward was the normalised final score — inconsistent.
            self._episode_reward = reward

        return {
            "observation": self.get_state(),
            "reward":      reward,
            "done":        self._done,
            "info": {
                "action_valid":   action_valid,
                "rescued_count":  rescued_count,
                "episode_reward": round(self._episode_reward, 4),
            },
        }

    def state(self) -> dict:
        """
        OpenEnv-compatible state().
        Same as get_state() — kept for OpenEnv spec compliance.
        """
        return self.get_state()

    def get_state(self):
        """Returns the current state of all zones as a dictionary."""
        return {
            "task_level": self.task_level,
            "step_num":   self.step_num,
            "zones":      [z.to_dict() for z in self.zones],
        }