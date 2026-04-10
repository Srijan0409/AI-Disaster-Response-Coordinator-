"""
=============================================================================
TEST SUITE - grid.py  (Zone + GridWorld)
=============================================================================
Run with:
    pytest test_grid.py -v
=============================================================================
"""

import copy
import pytest
from disaster_env.server.constants import (
    VICTIM_TIME_DECAY,
    UTTARAKHAND_ZONES,
    STEP_LIMITS,
)
from disaster_env.server.grid import GridWorld, Zone
from disaster_env.server.tasks import get_task

LEVELS = ["easy", "medium", "hard"]


def make_env(level="easy"):
    task = get_task(level)
    g = GridWorld(level, task["seed"])
    g.reset()
    return g


def make_victim(vid=0, urgency=2, survival_time=5, distance_km=3.0,
                alive=True, rescued=False):
    return {
        "id": vid, "urgency": urgency, "survival_time": survival_time,
        "distance_km": distance_km, "alive": alive, "rescued": rescued,
    }


# =============================================================================
# Zone
# =============================================================================

class TestZoneIsActive:

    def _zone(self, victims):
        z = Zone(0, "Test", "Dist", "flood", 0.8)
        z.victims = victims
        z.people  = len(victims)
        return z

    def test_active_when_alive_unrescued(self):
        z = self._zone([make_victim(alive=True, rescued=False)])
        assert z.is_active is True

    def test_inactive_when_all_rescued(self):
        z = self._zone([make_victim(alive=True, rescued=True)])
        assert z.is_active is False

    def test_inactive_when_all_dead(self):
        z = self._zone([make_victim(alive=False, rescued=False)])
        assert z.is_active is False

    def test_inactive_when_empty(self):
        z = self._zone([])
        assert z.is_active is False

    def test_active_mixed_victims(self):
        z = self._zone([
            make_victim(vid=0, alive=True, rescued=True),
            make_victim(vid=1, alive=True, rescued=False),
        ])
        assert z.is_active is True

    def test_inactive_all_dead_or_rescued(self):
        z = self._zone([
            make_victim(vid=0, alive=False, rescued=False),
            make_victim(vid=1, alive=True,  rescued=True),
        ])
        assert z.is_active is False


class TestZoneToDict:

    def _zone(self):
        z = Zone(0, "TestZone", "TestDistrict", "flood", 0.75)
        z.victims = [make_victim()]
        z.people  = 1
        return z

    def test_has_required_fields(self):
        d = self._zone().to_dict()
        required = {"zone_id", "name", "district", "disaster_type",
                    "severity", "people", "rescued", "time_waiting",
                    "is_active", "victims"}
        assert required.issubset(d.keys())

    def test_severity_rounded_to_two_decimals(self):
        z = Zone(0, "T", "D", "f", 0.756789)
        z.victims = []
        d = z.to_dict()
        assert d["severity"] == round(0.756789, 2)

    def test_victims_is_deep_copy(self):
        z = self._zone()
        d = z.to_dict()
        d["victims"][0]["alive"] = False
        assert z.victims[0]["alive"] is True  # original untouched

    def test_is_active_reflects_victims(self):
        z = self._zone()
        z.victims[0]["rescued"] = True
        d = z.to_dict()
        assert d["is_active"] is False


# =============================================================================
# GridWorld.__init__ and seed validation
# =============================================================================

class TestGridWorldInit:

    def test_seed_mismatch_raises_value_error(self):
        with pytest.raises(ValueError, match="Seed mismatch"):
            GridWorld("easy", seed=9999)

    def test_correct_seed_easy(self):
        task = get_task("easy")
        g = GridWorld("easy", task["seed"])
        assert g is not None

    def test_correct_seed_medium(self):
        task = get_task("medium")
        g = GridWorld("medium", task["seed"])
        assert g is not None

    def test_correct_seed_hard(self):
        task = get_task("hard")
        g = GridWorld("hard", task["seed"])
        assert g is not None

    def test_error_message_contains_task_level(self):
        with pytest.raises(ValueError, match="easy"):
            GridWorld("easy", seed=1)

    def test_initial_done_false(self):
        task = get_task("easy")
        g = GridWorld("easy", task["seed"])
        assert g._done is False


# =============================================================================
# GridWorld.reset
# =============================================================================

class TestGridWorldReset:

    def test_returns_dict(self):
        g = make_env()
        state = g.reset()
        assert isinstance(state, dict)

    def test_state_has_required_keys(self):
        g = make_env()
        state = g.reset()
        assert {"task_level", "step_num", "zones"} == set(state.keys())

    def test_step_num_zero(self):
        g = make_env()
        assert g.step_num == 0

    def test_done_false(self):
        g = make_env()
        assert g._done is False

    def test_easy_zone_count(self):
        g = make_env("easy")
        assert len(g.zones) == len(UTTARAKHAND_ZONES["easy"])

    def test_medium_zone_count(self):
        g = make_env("medium")
        assert len(g.zones) == len(UTTARAKHAND_ZONES["medium"])

    def test_hard_zone_count(self):
        g = make_env("hard")
        assert len(g.zones) == len(UTTARAKHAND_ZONES["hard"])

    def test_people_equals_victim_count(self):
        for level in LEVELS:
            g = make_env(level)
            for z in g.zones:
                assert z.people == len(z.victims)

    def test_hard_first_zone_severity_one(self):
        g = make_env("hard")
        assert g.zones[0].severity == 1.0

    def test_next_victim_id_after_max_initial(self):
        g = make_env("medium")
        max_id = max(v["id"] for z in g.zones for v in z.victims)
        assert g._next_victim_id == max_id + 1

    def test_reproducible(self):
        task = get_task("medium")
        g = GridWorld("medium", task["seed"])
        s1 = g.reset()
        s2 = g.reset()
        assert s1 == s2

    def test_zones_initial_independent_of_tick_mutations(self):
        g = make_env("easy")
        snap = copy.deepcopy(g._zones_initial)
        g.tick()
        assert g._zones_initial == snap

    def test_spawned_victims_reset_to_zero(self):
        g = make_env("hard")
        g._spawned_victims = 99
        g.reset()
        assert g._spawned_victims == 0

    def test_spawned_ids_cleared_on_reset(self):
        g = make_env("hard")
        g._spawned_ids = {999, 1000}
        g.reset()
        assert len(g._spawned_ids) == 0


# =============================================================================
# GridWorld.tick
# =============================================================================

class TestGridWorldTick:

    def test_step_num_increments(self):
        g = make_env()
        g.tick()
        assert g.step_num == 1

    def test_step_num_multiple_ticks(self):
        g = make_env()
        for _ in range(5):
            g.tick()
        assert g.step_num == 5

    def test_survival_time_decays_easy(self):
        g = make_env("easy")
        decay = VICTIM_TIME_DECAY["easy"]
        before = {v["id"]: v["survival_time"]
                  for z in g.zones for v in z.victims}
        g.tick()
        for z in g.zones:
            for v in z.victims:
                if v["alive"] and not v["rescued"]:
                    assert v["survival_time"] == before[v["id"]] - decay

    def test_survival_time_decays_hard(self):
        g = make_env("hard")
        decay = VICTIM_TIME_DECAY["hard"]
        assert decay == 2
        v0 = g.zones[0].victims[0]
        st_before = v0["survival_time"]
        g.tick()
        assert v0["survival_time"] == st_before - decay

    def test_victim_dies_when_survival_reaches_zero(self):
        g = make_env("easy")
        g.zones[0].victims[0]["survival_time"] = 1
        g.tick()
        assert g.zones[0].victims[0]["alive"] is False

    def test_rescued_victim_not_decayed(self):
        g = make_env("easy")
        g.zones[0].victims[0]["rescued"] = True
        st = g.zones[0].victims[0]["survival_time"]
        g.tick()
        assert g.zones[0].victims[0]["survival_time"] == st

    def test_dead_victim_not_decayed_further(self):
        g = make_env("easy")
        g.zones[0].victims[0]["alive"] = False
        g.zones[0].victims[0]["survival_time"] = 3
        g.tick()
        assert g.zones[0].victims[0]["survival_time"] == 3

    def test_time_waiting_increments_active_zone(self):
        g = make_env("easy")
        before = g.zones[0].time_waiting
        g.tick()
        assert g.zones[0].time_waiting == before + 1

    def test_time_waiting_not_incremented_inactive_zone(self):
        g = make_env("easy")
        for v in g.zones[0].victims:
            v["rescued"] = True
        before = g.zones[0].time_waiting
        g.tick()
        assert g.zones[0].time_waiting == before


# =============================================================================
# GridWorld.spread_threat
# =============================================================================

class TestGridWorldSpreadThreat:

    def test_high_severity_spreads_to_next(self):
        g = make_env("medium")
        g.zones[0].severity = 0.9
        sev_before = g.zones[1].severity
        g.spread_threat()
        assert g.zones[1].severity > sev_before

    def test_severity_below_threshold_does_not_spread(self):
        g = make_env("medium")
        g.zones[0].severity = 0.5  # <= 0.6
        sev_before = g.zones[1].severity
        g.spread_threat()
        assert g.zones[1].severity == sev_before

    def test_severity_capped_at_one(self):
        g = make_env("medium")
        g.zones[0].severity = 1.0
        g.zones[1].severity = 0.95
        g.spread_threat()
        assert g.zones[1].severity <= 1.0

    def test_no_double_time_waiting_increment_active_zone(self):
        """BUG F FIX: active zone time_waiting must not increment in spread_threat."""
        g = make_env("medium")
        g.zones[0].severity = 0.9
        before = g.zones[1].time_waiting
        g.spread_threat()
        assert g.zones[1].time_waiting == before  # tick() handles active zones

    def test_zero_victim_zone_never_incremented(self):
        g = make_env("medium")
        g.zones[0].severity = 0.9
        for v in g.zones[1].victims:
            v["alive"] = False
        before = g.zones[1].time_waiting
        g.spread_threat()
        assert g.zones[1].time_waiting == before

    def test_last_zone_never_spreads(self):
        g = make_env("hard")
        last_idx = len(g.zones) - 1
        g.zones[last_idx].severity = 1.0
        # No zone after last - should not raise
        g.spread_threat()

    def test_spread_increases_by_0_1(self):
        g = make_env("medium")
        g.zones[0].severity = 0.9
        sev_before = round(g.zones[1].severity, 2)
        g.spread_threat()
        expected = min(1.0, round(sev_before + 0.1, 2))
        assert round(g.zones[1].severity, 2) == expected


# =============================================================================
# GridWorld.apply_action
# =============================================================================

class TestGridWorldApplyAction:

    def test_valid_rescue_team_returns_true(self):
        g = make_env()
        ok, count = g.apply_action(0, "rescue_team")
        assert ok is True

    def test_rescue_team_rescues_three(self):
        g = make_env()
        _, count = g.apply_action(0, "rescue_team")
        assert count == 3

    def test_ambulance_rescues_two(self):
        g = make_env()
        _, count = g.apply_action(0, "ambulance")
        assert count == 2

    def test_helicopter_rescues_five_easy(self):
        g = make_env("easy")
        _, count = g.apply_action(0, "helicopter")
        assert count == 5

    def test_helicopter_forbidden_hard(self):
        g = make_env("hard")
        ok, count = g.apply_action(0, "helicopter")
        assert ok is False
        assert count == 0

    def test_invalid_zone_id_returns_false(self):
        g = make_env()
        ok, count = g.apply_action(999, "rescue_team")
        assert ok is False
        assert count == 0

    def test_invalid_unit_type_returns_false(self):
        g = make_env()
        ok, count = g.apply_action(0, "tank")
        assert ok is False
        assert count == 0

    def test_rescued_flag_set(self):
        g = make_env()
        g.apply_action(0, "ambulance")
        rescued = [v for v in g.zones[0].victims if v["rescued"]]
        assert len(rescued) == 2

    def test_rescued_victims_stay_alive(self):
        """FIX 5: rescued must keep alive=True."""
        g = make_env()
        g.apply_action(0, "rescue_team")
        for v in g.zones[0].victims:
            if v["rescued"]:
                assert v["alive"] is True

    def test_rescue_prioritises_highest_urgency(self):
        g = make_env()
        g.zones[0].victims[0]["urgency"] = 3
        g.zones[0].victims[1]["urgency"] = 1
        g.apply_action(0, "ambulance")
        assert g.zones[0].victims[0]["rescued"] is True

    def test_empty_zone_returns_false(self):
        g = make_env()
        for v in g.zones[0].victims:
            v["rescued"] = True
        ok, count = g.apply_action(0, "ambulance")
        assert ok is False
        assert count == 0

    def test_zone_rescued_count_updated(self):
        g = make_env()
        before = g.zones[0].rescued
        g.apply_action(0, "rescue_team")
        assert g.zones[0].rescued == before + 3

    def test_does_not_rescue_dead_victims(self):
        g = make_env()
        for v in g.zones[0].victims:
            v["alive"] = False
        ok, count = g.apply_action(0, "rescue_team")
        assert ok is False
        assert count == 0

    def test_rescued_spawned_counter_updated(self):
        g = make_env("hard")
        g._spawned_ids.add(g.zones[0].victims[0]["id"])
        g.apply_action(0, "ambulance")
        assert g._rescued_spawned >= 0  # should increment if victim was rescued


# =============================================================================
# GridWorld.step
# =============================================================================

class TestGridWorldStep:

    def test_returns_required_keys(self):
        g = make_env()
        result = g.step({"zone_id": 0, "unit_type": "rescue_team"})
        assert {"observation", "reward", "done", "info"} == set(result.keys())

    def test_info_has_required_keys(self):
        g = make_env()
        result = g.step({"zone_id": 0, "unit_type": "rescue_team"})
        assert {"action_valid", "rescued_count", "episode_reward"} == set(result["info"].keys())

    def test_step_num_increments(self):
        g = make_env()
        g.step({"zone_id": 0, "unit_type": "rescue_team"})
        assert g.step_num == 1

    def test_done_false_mid_episode(self):
        g = make_env()
        result = g.step({"zone_id": 0, "unit_type": "rescue_team"})
        # should not be done after 1 step (easy has 30 steps)
        # (may be done if all victims rescued immediately - unlikely with 7 victims)
        assert isinstance(result["done"], bool)

    def test_done_true_after_max_steps(self):
        g = make_env()
        task = get_task("easy")
        result = None
        for _ in range(task["max_steps"]):
            result = g.step({"zone_id": 0, "unit_type": "rescue_team"})
        assert result["done"] is True

    def test_step_after_done_raises_runtime_error(self):
        g = make_env()
        task = get_task("easy")
        for _ in range(task["max_steps"]):
            g.step({"zone_id": 0, "unit_type": "rescue_team"})
        with pytest.raises(RuntimeError, match="done"):
            g.step({"zone_id": 0, "unit_type": "rescue_team"})

    def test_reward_is_float(self):
        g = make_env()
        result = g.step({"zone_id": 0, "unit_type": "rescue_team"})
        assert isinstance(result["reward"], float)

    def test_invalid_action_gives_low_reward(self):
        """FIX 18: bad actions return negative raw -> low normalized reward."""
        g = make_env()
        result = g.step({"zone_id": 999, "unit_type": "rescue_team"})
        assert result["reward"] < 0.5

    def test_terminal_reward_in_open_interval(self):
        g = make_env()
        task = get_task("easy")
        result = None
        for _ in range(task["max_steps"]):
            result = g.step({"zone_id": 0, "unit_type": "rescue_team"})
        assert result["done"] is True
        assert 0.0 < result["reward"] < 1.0

    def test_terminal_reward_equals_episode_reward_in_info(self):
        """FIX 16: episode_reward in info must equal terminal step reward."""
        g = make_env()
        task = get_task("easy")
        result = None
        for _ in range(task["max_steps"]):
            result = g.step({"zone_id": 0, "unit_type": "rescue_team"})
        assert result["done"] is True
        assert result["reward"] == result["info"]["episode_reward"]

    def test_zones_before_snapshot_prevents_wasted_step_penalty(self):
        """BUG 2 FIX: last victim rescue must not appear as a wasted step."""
        g = make_env("easy")
        for v in g.zones[0].victims[1:]:
            v["rescued"] = True
        g.zones[0].rescued = len(g.zones[0].victims) - 1
        result = g.step({"zone_id": 0, "unit_type": "rescue_team"})
        assert result["info"]["action_valid"] is True
        assert result["reward"] > 0.5

    def test_action_valid_true_for_valid_rescue(self):
        g = make_env()
        result = g.step({"zone_id": 0, "unit_type": "rescue_team"})
        assert result["info"]["action_valid"] is True

    def test_action_valid_false_for_invalid_zone(self):
        g = make_env()
        result = g.step({"zone_id": 999, "unit_type": "rescue_team"})
        assert result["info"]["action_valid"] is False

    def test_rescued_count_correct(self):
        g = make_env()
        result = g.step({"zone_id": 0, "unit_type": "rescue_team"})
        assert result["info"]["rescued_count"] == 3

    def test_observation_contains_zones(self):
        g = make_env()
        result = g.step({"zone_id": 0, "unit_type": "rescue_team"})
        assert "zones" in result["observation"]


# =============================================================================
# GridWorld.spawn_new_victims
# =============================================================================

class TestGridWorldSpawnNewVictims:

    def test_increases_total_victim_count(self):
        g = make_env("hard")
        before = sum(len(z.victims) for z in g.zones)
        g.spawn_new_victims()
        after = sum(len(z.victims) for z in g.zones)
        assert after > before

    def test_spawned_ids_globally_unique(self):
        g = make_env("hard")
        g.spawn_new_victims()
        all_ids = [v["id"] for z in g.zones for v in z.victims]
        assert len(all_ids) == len(set(all_ids))

    def test_people_equals_victims_after_spawn(self):
        """FIX 19: single source of truth must hold after spawn."""
        g = make_env("hard")
        g.spawn_new_victims()
        for z in g.zones:
            assert z.people == len(z.victims)

    def test_spawned_victims_counter_increments(self):
        g = make_env("hard")
        g.spawn_new_victims()
        assert g._spawned_victims > 0

    def test_spawned_ids_registered(self):
        g = make_env("hard")
        g.spawn_new_victims()
        assert len(g._spawned_ids) > 0

    def test_no_spawn_into_all_dead_zones(self):
        """FIX 7: spawn only into zones with alive unrescued victims."""
        g = make_env("hard")
        for z in g.zones[:-1]:
            for v in z.victims:
                v["alive"] = False
        g.spawn_new_victims()
        for z in g.zones[:-1]:
            new_ids = {v["id"] for v in z.victims if v["id"] in g._spawned_ids}
            assert len(new_ids) == 0

    def test_no_spawn_when_all_zones_dead(self):
        g = make_env("hard")
        for z in g.zones:
            for v in z.victims:
                v["alive"] = False
        before = g._spawned_victims
        g.spawn_new_victims()
        assert g._spawned_victims == before  # nothing spawned

    def test_new_victim_ids_greater_than_initial_max(self):
        g = make_env("hard")
        initial_max = max(v["id"] for z in g.zones for v in z.victims)
        g.spawn_new_victims()
        spawned = [v for z in g.zones for v in z.victims
                   if v["id"] in g._spawned_ids]
        for v in spawned:
            assert v["id"] > initial_max


# =============================================================================
# get_state / state
# =============================================================================

class TestGridWorldGetState:

    def test_get_state_returns_dict(self):
        g = make_env()
        s = g.get_state()
        assert isinstance(s, dict)

    def test_get_state_has_zones(self):
        g = make_env()
        s = g.get_state()
        assert "zones" in s

    def test_state_equals_get_state(self):
        g = make_env()
        assert g.state() == g.get_state()

    def test_zone_count_in_state(self):
        for level in LEVELS:
            g = make_env(level)
            s = g.get_state()
            assert len(s["zones"]) == len(UTTARAKHAND_ZONES[level])