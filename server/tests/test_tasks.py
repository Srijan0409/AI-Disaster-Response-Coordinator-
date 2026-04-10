"""
=============================================================================
TEST SUITE - tasks.py
=============================================================================
Run with:
    pytest test_tasks.py -v
=============================================================================
"""

import copy
import pytest
from disaster_env.server.tasks import get_task, get_task_scenario, list_tasks, TASKS
from disaster_env.server.constants import (
    STEP_LIMITS,
    RESOURCE_CONFIG,
    UTTARAKHAND_ZONES,
)

LEVELS = ["easy", "medium", "hard"]


# =============================================================================
# list_tasks
# =============================================================================

class TestListTasks:

    def test_returns_three_levels(self):
        assert set(list_tasks()) == {"easy", "medium", "hard"}

    def test_returns_list(self):
        assert isinstance(list_tasks(), list)

    def test_no_duplicates(self):
        tasks = list_tasks()
        assert len(tasks) == len(set(tasks))


# =============================================================================
# get_task - basic
# =============================================================================

class TestGetTaskBasic:

    def test_easy_returned(self):
        t = get_task("easy")
        assert t["task_level"] == "easy"

    def test_medium_returned(self):
        t = get_task("medium")
        assert t["task_level"] == "medium"

    def test_hard_returned(self):
        t = get_task("hard")
        assert t["task_level"] == "hard"

    def test_invalid_level_raises_value_error(self):
        with pytest.raises(ValueError):
            get_task("extreme")

    def test_invalid_level_message_informative(self):
        with pytest.raises(ValueError, match="extreme"):
            get_task("extreme")

    def test_task_level_matches_key(self):
        for level in LEVELS:
            assert get_task(level)["task_level"] == level


# =============================================================================
# get_task - deep copy (BUG 14 FIX)
# =============================================================================

class TestGetTaskDeepCopy:

    def test_mutating_returned_task_does_not_affect_registry(self):
        """BUG 14 FIX: returned task must be a deep copy."""
        t = get_task("easy")
        t["resources"]["ambulances"] = 9999
        t2 = get_task("easy")
        assert t2["resources"]["ambulances"] == RESOURCE_CONFIG["easy"]["ambulances"]

    def test_mutating_description_does_not_affect_registry(self):
        t = get_task("medium")
        t["description"] = "HACKED"
        t2 = get_task("medium")
        assert t2["description"] != "HACKED"

    def test_returns_independent_copies(self):
        t1 = get_task("easy")
        t2 = get_task("easy")
        assert t1 is not t2

    def test_nested_dicts_are_copies(self):
        t1 = get_task("easy")
        t2 = get_task("easy")
        assert t1["resources"] is not t2["resources"]

    def test_mutating_action_space_does_not_affect_registry(self):
        t = get_task("hard")
        t["action_space"]["unit_types"].append("nuke")
        t2 = get_task("hard")
        assert "nuke" not in t2["action_space"]["unit_types"]


# =============================================================================
# get_task - required fields
# =============================================================================

class TestGetTaskFields:

    REQUIRED = [
        "seed", "task_level", "max_steps", "resources", "spread",
        "spawn_victims", "success_threshold", "description", "objective",
        "observation_space", "action_space",
    ]

    def test_easy_has_all_fields(self):
        t = get_task("easy")
        for key in self.REQUIRED:
            assert key in t

    def test_medium_has_all_fields(self):
        t = get_task("medium")
        for key in self.REQUIRED:
            assert key in t

    def test_hard_has_all_fields(self):
        t = get_task("hard")
        for key in self.REQUIRED:
            assert key in t


# =============================================================================
# get_task - seeds
# =============================================================================

class TestGetTaskSeeds:

    def test_seeds_unique_across_levels(self):
        seeds = [get_task(l)["seed"] for l in LEVELS]
        assert len(seeds) == len(set(seeds))

    def test_seeds_are_integers(self):
        for level in LEVELS:
            assert isinstance(get_task(level)["seed"], int)

    def test_easy_seed_is_42(self):
        assert get_task("easy")["seed"] == 42

    def test_medium_seed_is_123(self):
        assert get_task("medium")["seed"] == 123

    def test_hard_seed_is_999(self):
        assert get_task("hard")["seed"] == 999


# =============================================================================
# get_task - max_steps
# =============================================================================

class TestGetTaskMaxSteps:

    def test_max_steps_match_constants(self):
        for level in LEVELS:
            assert get_task(level)["max_steps"] == STEP_LIMITS[level]

    def test_hard_fewest_steps(self):
        assert get_task("hard")["max_steps"] <= get_task("easy")["max_steps"]

    def test_easy_leq_medium(self):
        assert get_task("easy")["max_steps"] <= get_task("medium")["max_steps"]

    def test_all_positive(self):
        for level in LEVELS:
            assert get_task(level)["max_steps"] > 0


# =============================================================================
# get_task - resources
# =============================================================================

class TestGetTaskResources:

    def test_resources_match_config(self):
        for level in LEVELS:
            assert get_task(level)["resources"] == RESOURCE_CONFIG[level]

    def test_hard_no_helicopters(self):
        assert get_task("hard")["resources"]["helicopters"] == 0

    def test_easy_has_ambulances(self):
        assert get_task("easy")["resources"]["ambulances"] >= 1


# =============================================================================
# get_task - success threshold
# =============================================================================

class TestGetTaskSuccessThreshold:

    def test_thresholds_ordered(self):
        easy   = get_task("easy")["success_threshold"]
        medium = get_task("medium")["success_threshold"]
        hard   = get_task("hard")["success_threshold"]
        assert easy < medium < hard

    def test_easy_threshold_is_0_5(self):
        assert get_task("easy")["success_threshold"] == 0.5

    def test_medium_threshold_is_0_6(self):
        assert get_task("medium")["success_threshold"] == 0.6

    def test_hard_threshold_is_0_7(self):
        assert get_task("hard")["success_threshold"] == 0.7

    def test_thresholds_between_zero_and_one(self):
        for level in LEVELS:
            t = get_task(level)["success_threshold"]
            assert 0.0 < t < 1.0


# =============================================================================
# get_task - spread / spawn settings
# =============================================================================

class TestGetTaskSpreadSpawn:

    def test_easy_spread_false(self):
        assert get_task("easy")["spread"] is False

    def test_medium_spread_true(self):
        assert get_task("medium")["spread"] is True

    def test_hard_spread_true(self):
        assert get_task("hard")["spread"] is True

    def test_easy_spawn_false(self):
        assert get_task("easy")["spawn_victims"] is False

    def test_medium_spawn_false(self):
        assert get_task("medium")["spawn_victims"] is False

    def test_hard_spawn_true(self):
        assert get_task("hard")["spawn_victims"] is True

    def test_hard_spread_interval_less_than_medium(self):
        hard_interval   = get_task("hard")["spread_interval"]
        medium_interval = get_task("medium")["spread_interval"]
        assert hard_interval < medium_interval


# =============================================================================
# get_task - action space
# =============================================================================

class TestGetTaskActionSpace:

    def test_hard_no_helicopter_in_action_space(self):
        assert "helicopter" not in get_task("hard")["action_space"]["unit_types"]

    def test_easy_has_helicopter_in_action_space(self):
        assert "helicopter" in get_task("easy")["action_space"]["unit_types"]

    def test_medium_has_helicopter_in_action_space(self):
        assert "helicopter" in get_task("medium")["action_space"]["unit_types"]

    def test_all_levels_have_ambulance_in_action_space(self):
        for level in LEVELS:
            assert "ambulance" in get_task(level)["action_space"]["unit_types"]

    def test_all_levels_have_rescue_team_in_action_space(self):
        for level in LEVELS:
            assert "rescue_team" in get_task(level)["action_space"]["unit_types"]

    def test_action_space_type_discrete(self):
        for level in LEVELS:
            assert get_task(level)["action_space"]["type"] == "discrete"


# =============================================================================
# get_task_scenario
# =============================================================================

class TestGetTaskScenario:

    def test_task_level_correct(self):
        for level in LEVELS:
            s = get_task_scenario(level)
            assert s["task_level"] == level

    def test_has_zones(self):
        for level in LEVELS:
            s = get_task_scenario(level)
            assert "zones" in s
            assert len(s["zones"]) > 0

    def test_has_resources(self):
        for level in LEVELS:
            s = get_task_scenario(level)
            assert "resources" in s

    def test_has_max_steps(self):
        for level in LEVELS:
            s = get_task_scenario(level)
            assert s["max_steps"] == STEP_LIMITS[level]

    def test_seed_matches_task_seed(self):
        for level in LEVELS:
            task = get_task(level)
            s = get_task_scenario(level)
            assert s["seed"] == task["seed"]

    def test_zone_names_match_constants(self):
        for level in LEVELS:
            s = get_task_scenario(level)
            expected_names = [z["name"] for z in UTTARAKHAND_ZONES[level]]
            actual_names   = [z["name"] for z in s["zones"]]
            assert actual_names == expected_names

    def test_reproducible(self):
        for level in LEVELS:
            s1 = get_task_scenario(level)
            s2 = get_task_scenario(level)
            assert s1 == s2