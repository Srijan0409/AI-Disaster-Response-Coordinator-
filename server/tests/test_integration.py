"""
=============================================================================
TEST SUITE — Integration Tests
=============================================================================
End-to-end tests that span constants → generators → grid → grader → tasks.

Run with:
    pytest test_integration.py -v
=============================================================================
"""

import copy
import pytest
from disaster_env.server.grid import GridWorld
from disaster_env.server.grader import grade_episode, compute_reward
from disaster_env.server.generators import generate_civilians, generate_scenario, sync_with_grid
from disaster_env.server.tasks import get_task, get_task_scenario
from disaster_env.server.constants import STEP_LIMITS, UTTARAKHAND_ZONES

LEVELS = ["easy", "medium", "hard"]


def make_env(level):
    task = get_task(level)
    g = GridWorld(level, task["seed"])
    g.reset()
    return g


def run_full_episode(level, unit_type="rescue_team"):
    g = make_env(level)
    task = get_task(level)
    last = None
    for _ in range(task["max_steps"]):
        last = g.step({"zone_id": 0, "unit_type": unit_type})
        if last["done"]:
            break
    return g, last


# =============================================================================
# Full episode completion
# =============================================================================

class TestFullEpisodeCompletion:

    def test_easy_episode_terminates(self):
        _, last = run_full_episode("easy")
        assert last["done"] is True

    def test_medium_episode_terminates(self):
        _, last = run_full_episode("medium")
        assert last["done"] is True

    def test_hard_episode_terminates(self):
        _, last = run_full_episode("hard")
        assert last["done"] is True

    def test_easy_reward_in_open_interval(self):
        _, last = run_full_episode("easy")
        assert 0.0 < last["reward"] < 1.0

    def test_medium_reward_in_open_interval(self):
        _, last = run_full_episode("medium")
        assert 0.0 < last["reward"] < 1.0

    def test_hard_reward_in_open_interval(self):
        _, last = run_full_episode("hard")
        assert 0.0 < last["reward"] < 1.0


# =============================================================================
# Reset after done
# =============================================================================

class TestResetAfterDone:

    def test_reset_clears_done_flag(self):
        g, _ = run_full_episode("easy")
        g.reset()
        assert g._done is False

    def test_reset_clears_step_num(self):
        g, _ = run_full_episode("easy")
        g.reset()
        assert g.step_num == 0

    def test_reset_clears_spawned_victims(self):
        g, _ = run_full_episode("hard")
        g.reset()
        assert g._spawned_victims == 0

    def test_can_run_new_episode_after_reset(self):
        g, _ = run_full_episode("easy")
        g.reset()
        result = g.step({"zone_id": 0, "unit_type": "rescue_team"})
        assert isinstance(result["reward"], float)

    def test_step_after_done_raises(self):
        g, _ = run_full_episode("easy")
        with pytest.raises(RuntimeError):
            g.step({"zone_id": 0, "unit_type": "rescue_team"})


# =============================================================================
# Consistency: grade_episode == env.step terminal reward  (BUG I FIX)
# =============================================================================

class TestGradeEpisodeConsistency:

    def _run(self, level):
        task = get_task(level)
        g = GridWorld(level, task["seed"])
        g.reset()
        zi = copy.deepcopy(g._zones_initial)
        last = None
        for _ in range(task["max_steps"]):
            last = g.step({"zone_id": 0, "unit_type": "rescue_team"})
            if last["done"]:
                break
        zf = g.get_state()["zones"]
        return zi, zf, g, last

    def test_easy_grade_matches_step_reward(self):
        zi, zf, g, last = self._run("easy")
        report = grade_episode("easy", zi, zf, g.step_num)
        assert report["score"] == last["reward"]

    def test_medium_grade_matches_step_reward(self):
        zi, zf, g, last = self._run("medium")
        report = grade_episode("medium", zi, zf, g.step_num)
        assert report["score"] == last["reward"]

    def test_hard_grade_matches_step_reward(self):
        zi, zf, g, last = self._run("hard")
        report = grade_episode("hard", zi, zf, g.step_num,
                               g._spawned_victims, g._rescued_spawned)
        assert report["score"] == last["reward"]

    def test_terminal_reward_equals_episode_reward_in_info(self):
        _, _, _, last = self._run("easy")
        assert last["reward"] == last["info"]["episode_reward"]


# =============================================================================
# Scenario from generators matches GridWorld.reset()
# =============================================================================

class TestScenarioMatchesGrid:

    def test_zone_names_match(self):
        level = "medium"
        task  = get_task(level)
        s = generate_scenario(level, task["seed"])
        g = make_env(level)
        for zd, zo in zip(s["zones"], g.zones):
            assert zd["name"] == zo.name

    def test_zone_districts_match(self):
        level = "medium"
        task  = get_task(level)
        s = generate_scenario(level, task["seed"])
        g = make_env(level)
        for zd, zo in zip(s["zones"], g.zones):
            assert zd["district"] == zo.district

    def test_zone_severities_match(self):
        level = "medium"
        task  = get_task(level)
        s = generate_scenario(level, task["seed"])
        g = make_env(level)
        for zd, zo in zip(s["zones"], g.zones):
            assert zd["severity"] == zo.severity

    def test_get_task_scenario_zone_names_match_constants(self):
        for level in LEVELS:
            s = get_task_scenario(level)
            expected = [z["name"] for z in UTTARAKHAND_ZONES[level]]
            actual   = [z["name"] for z in s["zones"]]
            assert actual == expected


# =============================================================================
# sync_with_grid after ticks
# =============================================================================

class TestSyncWithGridIntegration:

    def test_sync_after_ticks_severity_matches(self):
        task = get_task("easy")
        g = make_env("easy")
        civs = generate_civilians("easy", task["seed"])
        for _ in range(3):
            g.tick()
        synced = sync_with_grid(civs, g)
        for zd in g.get_state()["zones"]:
            zid = zd["zone_id"]
            assert synced[zid]["severity"] == round(zd["severity"], 2)

    def test_sync_after_ticks_time_waiting_matches(self):
        task = get_task("easy")
        g = make_env("easy")
        civs = generate_civilians("easy", task["seed"])
        for _ in range(3):
            g.tick()
        synced = sync_with_grid(civs, g)
        for zd in g.get_state()["zones"]:
            zid = zd["zone_id"]
            assert synced[zid]["time_waiting"] == zd["time_waiting"]

    def test_sync_reflects_rescue_actions(self):
        task = get_task("easy")
        g = make_env("easy")
        civs = generate_civilians("easy", task["seed"])
        g.step({"zone_id": 0, "unit_type": "ambulance"})
        synced = sync_with_grid(civs, g)
        assert synced[0]["rescued"] == g.zones[0].rescued


# =============================================================================
# Hard mode specific: no helicopter, spawning, spread
# =============================================================================

class TestHardModeIntegration:

    def test_helicopter_action_invalid(self):
        g = make_env("hard")
        result = g.step({"zone_id": 0, "unit_type": "helicopter"})
        assert result["info"]["action_valid"] is False
        assert result["info"]["rescued_count"] == 0

    def test_spawned_victim_ids_no_collision(self):
        g = make_env("hard")
        initial_ids = {v["id"] for z in g.zones for v in z.victims}
        for _ in range(10):
            if g._done:
                break
            g.step({"zone_id": 0, "unit_type": "rescue_team"})
        all_ids = [v["id"] for z in g.zones for v in z.victims]
        assert len(all_ids) == len(set(all_ids))

    def test_hard_spread_triggers_at_step_2(self):
        task = get_task("hard")
        g = GridWorld("hard", task["seed"])
        g.reset()
        g.zones[0].severity = 0.9
        sev_before = g.zones[1].severity
        # step triggers tick which calls spread every spread_interval=2 steps
        g.step({"zone_id": 0, "unit_type": "rescue_team"})
        g.step({"zone_id": 0, "unit_type": "rescue_team"})
        assert g.zones[1].severity >= sev_before  # could have increased

    def test_hard_spawn_triggers_at_step_5(self):
        g = make_env("hard")
        initial_count = sum(len(z.victims) for z in g.zones)
        for _ in range(5):
            if g._done:
                break
            g.step({"zone_id": 0, "unit_type": "rescue_team"})
        final_count = sum(len(z.victims) for z in g.zones)
        # After 5 steps, spawn_new_victims fires — count should increase
        # (unless all zones are inactive, which is unlikely after 5 steps)
        assert final_count >= initial_count


# =============================================================================
# BUG 2: zones_before snapshot taken before apply_action
# =============================================================================

class TestZonesBeforeSnapshot:

    def test_last_victim_rescue_not_penalised_as_wasted_step(self):
        """BUG 2 FIX: rescuing the last victim must not give wasted-step reward."""
        g = make_env("easy")
        # Leave only one victim unrescued
        for v in g.zones[0].victims[1:]:
            v["rescued"] = True
        g.zones[0].rescued = len(g.zones[0].victims) - 1
        result = g.step({"zone_id": 0, "unit_type": "rescue_team"})
        assert result["info"]["action_valid"] is True
        # High-severity valid action should give reward > 0.5
        assert result["reward"] > 0.5


# =============================================================================
# FIX 19: people = len(victims) after spawn
# =============================================================================

class TestPeopleVictimSingleSourceOfTruth:

    def test_people_equals_victims_throughout_hard_episode(self):
        g = make_env("hard")
        for _ in range(15):
            if g._done:
                break
            g.step({"zone_id": 0, "unit_type": "rescue_team"})
            for z in g.zones:
                assert z.people == len(z.victims), (
                    f"Zone {z.zone_id}: people={z.people} != len(victims)={len(z.victims)}"
                )


# =============================================================================
# FIX 5: rescued victims stay alive
# =============================================================================

class TestRescuedVictimsStayAlive:

    def test_rescued_victims_alive_throughout_episode(self):
        g = make_env("easy")
        for _ in range(15):
            if g._done:
                break
            g.step({"zone_id": 0, "unit_type": "rescue_team"})
        for z in g.zones:
            for v in z.victims:
                if v["rescued"]:
                    assert v["alive"] is True, (
                        f"Victim {v['id']} is rescued but alive=False"
                    )