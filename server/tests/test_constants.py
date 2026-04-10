"""
=============================================================================
TEST SUITE - constants.py
=============================================================================
Run with:
    pytest test_constants.py -v
=============================================================================
"""

import pytest
from disaster_env.server.constants import (
    UTTARAKHAND_ZONES,
    SEVERITY_RANGES,
    PEOPLE_RANGES,
    RESOURCE_CONFIG,
    STEP_LIMITS,
    VICTIM_URGENCY_WEIGHTS,
    VICTIM_SURVIVAL_TIME,
    VICTIM_DISTANCE_KM,
    VICTIM_TIME_DECAY,
    VICTIMS_PER_ZONE,
    ALLOWED_UNIT_TYPES,
)

LEVELS = ["easy", "medium", "hard"]


class TestDifficultyLevelsPresent:

    def test_uttarakhand_zones_has_all_levels(self):
        assert set(UTTARAKHAND_ZONES.keys()) == {"easy", "medium", "hard"}

    def test_severity_ranges_has_all_levels(self):
        assert set(SEVERITY_RANGES.keys()) == {"easy", "medium", "hard"}

    def test_people_ranges_has_all_levels(self):
        assert set(PEOPLE_RANGES.keys()) == {"easy", "medium", "hard"}

    def test_resource_config_has_all_levels(self):
        assert set(RESOURCE_CONFIG.keys()) == {"easy", "medium", "hard"}

    def test_step_limits_has_all_levels(self):
        assert set(STEP_LIMITS.keys()) == {"easy", "medium", "hard"}

    def test_urgency_weights_has_all_levels(self):
        assert set(VICTIM_URGENCY_WEIGHTS.keys()) == {"easy", "medium", "hard"}

    def test_survival_time_has_all_levels(self):
        assert set(VICTIM_SURVIVAL_TIME.keys()) == {"easy", "medium", "hard"}

    def test_distance_km_has_all_levels(self):
        assert set(VICTIM_DISTANCE_KM.keys()) == {"easy", "medium", "hard"}

    def test_time_decay_has_all_levels(self):
        assert set(VICTIM_TIME_DECAY.keys()) == {"easy", "medium", "hard"}

    def test_victims_per_zone_has_all_levels(self):
        assert set(VICTIMS_PER_ZONE.keys()) == {"easy", "medium", "hard"}

    def test_allowed_unit_types_has_all_levels(self):
        assert set(ALLOWED_UNIT_TYPES.keys()) == {"easy", "medium", "hard"}


class TestSeverityRanges:

    def test_min_less_than_max_easy(self):
        lo, hi = SEVERITY_RANGES["easy"]
        assert lo < hi

    def test_min_less_than_max_medium(self):
        lo, hi = SEVERITY_RANGES["medium"]
        assert lo < hi

    def test_min_less_than_max_hard(self):
        lo, hi = SEVERITY_RANGES["hard"]
        assert lo < hi

    def test_values_between_zero_and_one(self):
        for level in LEVELS:
            lo, hi = SEVERITY_RANGES[level]
            assert 0.0 <= lo <= 1.0
            assert 0.0 <= hi <= 1.0


class TestPeopleRanges:

    def test_min_less_than_max(self):
        for level in LEVELS:
            lo, hi = PEOPLE_RANGES[level]
            assert lo < hi

    def test_values_positive(self):
        for level in LEVELS:
            lo, _ = PEOPLE_RANGES[level]
            assert lo >= 1


class TestResourceConfig:

    def test_has_required_keys(self):
        for level in LEVELS:
            r = RESOURCE_CONFIG[level]
            assert {"ambulances", "rescue_teams", "helicopters"} == set(r.keys())

    def test_hard_no_helicopters(self):
        assert RESOURCE_CONFIG["hard"]["helicopters"] == 0

    def test_easy_has_helicopter(self):
        assert RESOURCE_CONFIG["easy"]["helicopters"] >= 1

    def test_medium_has_helicopter(self):
        assert RESOURCE_CONFIG["medium"]["helicopters"] >= 1

    def test_all_values_non_negative(self):
        for level in LEVELS:
            for val in RESOURCE_CONFIG[level].values():
                assert val >= 0


class TestStepLimits:

    def test_hard_fewest_steps(self):
        assert STEP_LIMITS["hard"] <= STEP_LIMITS["easy"]

    def test_easy_leq_medium(self):
        assert STEP_LIMITS["easy"] <= STEP_LIMITS["medium"]

    def test_all_positive(self):
        for level in LEVELS:
            assert STEP_LIMITS[level] > 0


class TestVictimUrgencyWeights:

    def test_three_weights_per_level(self):
        for level in LEVELS:
            assert len(VICTIM_URGENCY_WEIGHTS[level]) == 3

    def test_weights_sum_to_one(self):
        for level in LEVELS:
            assert abs(sum(VICTIM_URGENCY_WEIGHTS[level]) - 1.0) < 1e-9

    def test_all_weights_positive(self):
        for level in LEVELS:
            for w in VICTIM_URGENCY_WEIGHTS[level]:
                assert w > 0


class TestVictimSurvivalTime:

    def test_min_less_than_max(self):
        for level in LEVELS:
            lo, hi = VICTIM_SURVIVAL_TIME[level]
            assert lo < hi

    def test_hard_max_within_step_limit(self):
        _, st_max = VICTIM_SURVIVAL_TIME["hard"]
        assert st_max <= STEP_LIMITS["hard"]

    def test_all_values_positive(self):
        for level in LEVELS:
            lo, _ = VICTIM_SURVIVAL_TIME[level]
            assert lo >= 1


class TestVictimDistanceKm:

    def test_min_less_than_max(self):
        for level in LEVELS:
            lo, hi = VICTIM_DISTANCE_KM[level]
            assert lo < hi

    def test_min_at_least_one_km(self):
        for level in LEVELS:
            lo, _ = VICTIM_DISTANCE_KM[level]
            assert lo >= 1


class TestVictimTimeDecay:

    def test_all_decay_at_least_one(self):
        for level in LEVELS:
            assert VICTIM_TIME_DECAY[level] >= 1

    def test_hard_decay_is_two(self):
        assert VICTIM_TIME_DECAY["hard"] == 2

    def test_easy_decay_is_one(self):
        assert VICTIM_TIME_DECAY["easy"] == 1


class TestVictimsPerZone:

    def test_easy_leq_medium(self):
        assert VICTIMS_PER_ZONE["easy"] <= VICTIMS_PER_ZONE["medium"]

    def test_easy_leq_hard(self):
        assert VICTIMS_PER_ZONE["easy"] <= VICTIMS_PER_ZONE["hard"]

    def test_all_positive(self):
        for level in LEVELS:
            assert VICTIMS_PER_ZONE[level] > 0


class TestAllowedUnitTypes:

    def test_hard_no_helicopter(self):
        assert "helicopter" not in ALLOWED_UNIT_TYPES["hard"]

    def test_easy_has_helicopter(self):
        assert "helicopter" in ALLOWED_UNIT_TYPES["easy"]

    def test_medium_has_helicopter(self):
        assert "helicopter" in ALLOWED_UNIT_TYPES["medium"]

    def test_all_levels_have_ambulance(self):
        for level in LEVELS:
            assert "ambulance" in ALLOWED_UNIT_TYPES[level]

    def test_all_levels_have_rescue_team(self):
        for level in LEVELS:
            assert "rescue_team" in ALLOWED_UNIT_TYPES[level]

    def test_values_are_sets(self):
        for level in LEVELS:
            assert isinstance(ALLOWED_UNIT_TYPES[level], set)


class TestUttarakhandZones:

    def test_required_fields_present(self):
        required = {"name", "district", "disaster_type"}
        for level in LEVELS:
            for z in UTTARAKHAND_ZONES[level]:
                assert required.issubset(z.keys())

    def test_easy_has_one_zone(self):
        assert len(UTTARAKHAND_ZONES["easy"]) == 1

    def test_medium_has_three_zones(self):
        assert len(UTTARAKHAND_ZONES["medium"]) == 3

    def test_hard_has_five_zones(self):
        assert len(UTTARAKHAND_ZONES["hard"]) == 5

    def test_kedarnath_present_in_all_levels(self):
        for level in LEVELS:
            names = [z["name"] for z in UTTARAKHAND_ZONES[level]]
            assert any("Kedarnath" in n for n in names)

    def test_no_empty_strings(self):
        for level in LEVELS:
            for z in UTTARAKHAND_ZONES[level]:
                for val in z.values():
                    assert val != ""