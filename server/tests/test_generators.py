"""
=============================================================================
TEST SUITE — generators.py
=============================================================================
Run with:
    pytest test_generators.py -v
=============================================================================
"""

import copy
import pytest
from disaster_env.server.constants import (
    VICTIMS_PER_ZONE,
    VICTIM_SURVIVAL_TIME,
    VICTIM_DISTANCE_KM,
    SEVERITY_RANGES,
    RESOURCE_CONFIG,
    STEP_LIMITS,
    UTTARAKHAND_ZONES,
)
from disaster_env.server.generators import (
    _generate_victims,
    generate_resources,
    generate_civilians,
    generate_scenario,
    sync_with_grid,
)
from disaster_env.server.tasks import get_task
from disaster_env.server.grid import GridWorld

LEVELS = ["easy", "medium", "hard"]


# =============================================================================
# _generate_victims
# =============================================================================

class TestGenerateVictimsCount:

    def test_easy_count_matches_constant(self):
        victims = _generate_victims("easy", 0, 42)
        assert len(victims) == VICTIMS_PER_ZONE["easy"]

    def test_medium_count_matches_constant(self):
        victims = _generate_victims("medium", 0, 42)
        assert len(victims) == VICTIMS_PER_ZONE["medium"]

    def test_hard_count_matches_constant(self):
        victims = _generate_victims("hard", 0, 42)
        assert len(victims) == VICTIMS_PER_ZONE["hard"]


class TestGenerateVictimsFields:

    def test_required_fields_present(self):
        victims = _generate_victims("easy", 0, 42)
        required = {"id", "urgency", "survival_time", "distance_km", "alive", "rescued"}
        for v in victims:
            assert required.issubset(v.keys())

    def test_alive_true_on_creation(self):
        victims = _generate_victims("medium", 0, 42)
        for v in victims:
            assert v["alive"] is True

    def test_rescued_false_on_creation(self):
        victims = _generate_victims("medium", 0, 42)
        for v in victims:
            assert v["rescued"] is False

    def test_urgency_values_valid(self):
        for level in LEVELS:
            victims = _generate_victims(level, 0, 42)
            for v in victims:
                assert v["urgency"] in {1, 2, 3}

    def test_survival_time_within_range_easy(self):
        lo, hi = VICTIM_SURVIVAL_TIME["easy"]
        for v in _generate_victims("easy", 0, 42):
            assert lo <= v["survival_time"] <= hi

    def test_survival_time_within_range_medium(self):
        lo, hi = VICTIM_SURVIVAL_TIME["medium"]
        for v in _generate_victims("medium", 0, 42):
            assert lo <= v["survival_time"] <= hi

    def test_survival_time_within_range_hard(self):
        lo, hi = VICTIM_SURVIVAL_TIME["hard"]
        for v in _generate_victims("hard", 0, 42):
            assert lo <= v["survival_time"] <= hi

    def test_distance_km_within_range(self):
        for level in LEVELS:
            d_min, d_max = VICTIM_DISTANCE_KM[level]
            for v in _generate_victims(level, 0, 42):
                assert d_min <= v["distance_km"] <= d_max


class TestGenerateVictimsIds:

    def test_ids_unique_within_zone(self):
        victims = _generate_victims("medium", 0, 42)
        ids = [v["id"] for v in victims]
        assert len(ids) == len(set(ids))

    def test_global_unique_ids_across_zones(self):
        all_ids = []
        for zone_idx in range(3):
            victims = _generate_victims("medium", zone_idx, 42)
            all_ids.extend(v["id"] for v in victims)
        assert len(all_ids) == len(set(all_ids))

    def test_zone_0_ids_start_at_zero(self):
        victims = _generate_victims("medium", 0, 42)
        assert victims[0]["id"] == 0

    def test_zone_1_ids_start_at_count(self):
        count = VICTIMS_PER_ZONE["medium"]
        victims = _generate_victims("medium", 1, 42)
        assert victims[0]["id"] == count

    def test_id_base_formula_correctness(self):
        for level in LEVELS:
            count = VICTIMS_PER_ZONE[level]
            for zone_idx in range(3):
                victims = _generate_victims(level, zone_idx, 42)
                expected_base = zone_idx * count
                assert victims[0]["id"] == expected_base


class TestGenerateVictimsReproducibility:

    def test_same_seed_same_output(self):
        a = _generate_victims("medium", 1, 123)
        b = _generate_victims("medium", 1, 123)
        assert a == b

    def test_different_seeds_different_output(self):
        a = _generate_victims("medium", 0, 1)
        b = _generate_victims("medium", 0, 2)
        assert a != b

    def test_different_zone_indices_different_output(self):
        a = _generate_victims("medium", 0, 42)
        b = _generate_victims("medium", 1, 42)
        assert [v["id"] for v in a] != [v["id"] for v in b]


# =============================================================================
# generate_resources
# =============================================================================

class TestGenerateResources:

    def test_returns_copy_not_original(self):
        r = generate_resources("easy")
        r["ambulances"] = 999
        assert RESOURCE_CONFIG["easy"]["ambulances"] != 999

    def test_correct_keys_easy(self):
        r = generate_resources("easy")
        assert {"ambulances", "rescue_teams", "helicopters"} == set(r.keys())

    def test_correct_keys_medium(self):
        r = generate_resources("medium")
        assert {"ambulances", "rescue_teams", "helicopters"} == set(r.keys())

    def test_correct_keys_hard(self):
        r = generate_resources("hard")
        assert {"ambulances", "rescue_teams", "helicopters"} == set(r.keys())

    def test_hard_helicopters_zero(self):
        assert generate_resources("hard")["helicopters"] == 0

    def test_easy_matches_config(self):
        assert generate_resources("easy") == RESOURCE_CONFIG["easy"]

    def test_medium_matches_config(self):
        assert generate_resources("medium") == RESOURCE_CONFIG["medium"]

    def test_invalid_level_raises_value_error(self):
        with pytest.raises(ValueError):
            generate_resources("extreme")

    def test_invalid_level_message_contains_level(self):
        with pytest.raises(ValueError, match="extreme"):
            generate_resources("extreme")


# =============================================================================
# generate_civilians
# =============================================================================

class TestGenerateCiviliansCount:

    def test_easy_zone_count(self):
        civs = generate_civilians("easy", 42)
        assert len(civs) == len(UTTARAKHAND_ZONES["easy"])

    def test_medium_zone_count(self):
        civs = generate_civilians("medium", 42)
        assert len(civs) == len(UTTARAKHAND_ZONES["medium"])

    def test_hard_zone_count(self):
        civs = generate_civilians("hard", 42)
        assert len(civs) == len(UTTARAKHAND_ZONES["hard"])


class TestGenerateCiviliansFields:

    def test_required_fields_present(self):
        civs = generate_civilians("easy", 42)
        required = {"zone_id", "name", "district", "disaster_type",
                    "severity", "people", "rescued", "time_waiting",
                    "is_active", "victims"}
        for z in civs:
            assert required.issubset(z.keys())

    def test_initial_rescued_zero(self):
        for level in LEVELS:
            civs = generate_civilians(level, 42)
            for z in civs:
                assert z["rescued"] == 0

    def test_initial_time_waiting_zero(self):
        for level in LEVELS:
            civs = generate_civilians(level, 42)
            for z in civs:
                assert z["time_waiting"] == 0

    def test_initial_is_active_true(self):
        for level in LEVELS:
            civs = generate_civilians(level, 42)
            for z in civs:
                assert z["is_active"] is True

    def test_people_equals_victim_count(self):
        for level in LEVELS:
            civs = generate_civilians(level, 42)
            for z in civs:
                assert z["people"] == len(z["victims"])

    def test_zone_ids_sequential(self):
        civs = generate_civilians("medium", 42)
        for i, z in enumerate(civs):
            assert z["zone_id"] == i


class TestGenerateCiviliansSeverity:

    def test_hard_first_zone_severity_is_one(self):
        civs = generate_civilians("hard", 42)
        assert civs[0]["severity"] == 1.0

    def test_easy_severity_within_range(self):
        lo, hi = SEVERITY_RANGES["easy"]
        civs = generate_civilians("easy", 42)
        assert lo <= civs[0]["severity"] <= hi

    def test_medium_all_severities_within_range(self):
        lo, hi = SEVERITY_RANGES["medium"]
        civs = generate_civilians("medium", 42)
        for z in civs:
            assert lo <= z["severity"] <= hi

    def test_hard_non_first_zones_within_range(self):
        lo, hi = SEVERITY_RANGES["hard"]
        civs = generate_civilians("hard", 42)
        for z in civs[1:]:
            assert lo <= z["severity"] <= hi


class TestGenerateCiviliansReproducibility:

    def test_same_seed_same_output(self):
        a = generate_civilians("medium", 123)
        b = generate_civilians("medium", 123)
        assert a == b

    def test_different_seeds_different_severities(self):
        a = generate_civilians("medium", 1)
        b = generate_civilians("medium", 2)
        sev_a = [z["severity"] for z in a]
        sev_b = [z["severity"] for z in b]
        assert sev_a != sev_b

    def test_does_not_mutate_global_random_state(self):
        """Isolated RNG: calling generate_civilians must not affect global random."""
        import random
        random.seed(999)
        val_before = random.random()
        random.seed(999)
        generate_civilians("hard", 42)
        val_after = random.random()
        assert val_before == val_after


# =============================================================================
# generate_scenario
# =============================================================================

class TestGenerateScenario:

    def test_returns_required_keys(self):
        s = generate_scenario("easy", 42)
        assert {"task_level", "seed", "max_steps", "resources", "zones"} == set(s.keys())

    def test_task_level_stored(self):
        for level in LEVELS:
            s = generate_scenario(level, 42)
            assert s["task_level"] == level

    def test_seed_stored(self):
        s = generate_scenario("easy", 99)
        assert s["seed"] == 99

    def test_max_steps_matches_constant(self):
        for level in LEVELS:
            s = generate_scenario(level, 42)
            from disaster_env.server.constants import STEP_LIMITS
            assert s["max_steps"] == STEP_LIMITS[level]

    def test_resources_match_config(self):
        for level in LEVELS:
            s = generate_scenario(level, 42)
            assert s["resources"] == RESOURCE_CONFIG[level]

    def test_zone_count_matches(self):
        for level in LEVELS:
            s = generate_scenario(level, 42)
            assert len(s["zones"]) == len(UTTARAKHAND_ZONES[level])

    def test_reproducible(self):
        a = generate_scenario("medium", 123)
        b = generate_scenario("medium", 123)
        assert a == b


# =============================================================================
# sync_with_grid
# =============================================================================

class TestSyncWithGrid:

    def _make_env(self, level="easy"):
        task = get_task(level)
        g = GridWorld(level, task["seed"])
        g.reset()
        return g

    def test_sync_updates_severity(self):
        g = self._make_env("easy")
        civs = generate_civilians("easy", get_task("easy")["seed"])
        g.zones[0].severity = 0.99
        synced = sync_with_grid(civs, g)
        assert synced[0]["severity"] == round(0.99, 2)

    def test_sync_updates_time_waiting(self):
        g = self._make_env("easy")
        civs = generate_civilians("easy", get_task("easy")["seed"])
        g.zones[0].time_waiting = 7
        synced = sync_with_grid(civs, g)
        assert synced[0]["time_waiting"] == 7

    def test_sync_updates_is_active(self):
        g = self._make_env("easy")
        civs = generate_civilians("easy", get_task("easy")["seed"])
        for v in g.zones[0].victims:
            v["rescued"] = True
        synced = sync_with_grid(civs, g)
        assert synced[0]["is_active"] is False

    def test_sync_updates_people(self):
        g = self._make_env("easy")
        civs = generate_civilians("easy", get_task("easy")["seed"])
        g.zones[0].people = 100
        synced = sync_with_grid(civs, g)
        assert synced[0]["people"] == 100

    def test_sync_updates_rescued(self):
        g = self._make_env("easy")
        civs = generate_civilians("easy", get_task("easy")["seed"])
        g.zones[0].rescued = 5
        synced = sync_with_grid(civs, g)
        assert synced[0]["rescued"] == 5

    def test_sync_updates_victims_list(self):
        """BUG 7 FIX: victims list must be synced."""
        g = self._make_env("easy")
        civs = generate_civilians("easy", get_task("easy")["seed"])
        g.zones[0].victims[0]["rescued"] = True
        synced = sync_with_grid(civs, g)
        assert synced[0]["victims"][0]["rescued"] is True

    def test_returns_same_list_object(self):
        g = self._make_env("easy")
        civs = generate_civilians("easy", get_task("easy")["seed"])
        result = sync_with_grid(civs, g)
        assert result is civs

    def test_all_zones_synced(self):
        g = self._make_env("medium")
        civs = generate_civilians("medium", get_task("medium")["seed"])
        for i, zone in enumerate(g.zones):
            zone.severity = 0.1 * (i + 1)
        synced = sync_with_grid(civs, g)
        for i in range(len(g.zones)):
            assert synced[i]["severity"] == round(0.1 * (i + 1), 2)