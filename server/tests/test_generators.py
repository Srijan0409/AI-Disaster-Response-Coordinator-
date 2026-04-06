import pytest
from generators import generate_resources, generate_civilians, generate_scenario, sync_with_grid
from constants import SEVERITY_RANGES, PEOPLE_RANGES, RESOURCE_CONFIG
from grid import GridWorld


# ---------------------------------------------------------------------------
# Group 1 — generate_resources (4 tests)
# ---------------------------------------------------------------------------

def test_resources_easy_keys():
    """Easy resources must contain all three asset types."""
    res = generate_resources("easy")
    assert "ambulances"   in res
    assert "rescue_teams" in res
    assert "helicopters"  in res


def test_resources_hard_scarcer_than_medium():
    """Hard mode must have fewer or equal total resources than medium."""
    total_med  = sum(generate_resources("medium").values())
    total_hard = sum(generate_resources("hard").values())
    assert total_hard <= total_med, \
        "Hard mode should not have more resources than medium!"


def test_resources_hard_no_helicopters():
    """Hard task must have no helicopters — forces harder decision making."""
    assert generate_resources("hard")["helicopters"] == 0


def test_resources_invalid_level():
    """Passing an unknown task level must raise ValueError."""
    with pytest.raises(ValueError):
        generate_resources("extreme")


# ---------------------------------------------------------------------------
# Group 2 — generate_civilians (15 tests)
# ---------------------------------------------------------------------------

def test_civilians_same_seed_same_output():
    """Same seed must always produce the exact same civilian layout."""
    assert generate_civilians("medium", seed=42) == generate_civilians("medium", seed=42)


def test_civilians_different_seeds_differ():
    """Different seeds must produce different civilian scenarios."""
    assert generate_civilians("medium", seed=42) != generate_civilians("medium", seed=99)


def test_civilians_easy_has_one_zone():
    """Easy task must generate exactly 1 civilian zone."""
    assert len(generate_civilians("easy", seed=42)) == 1


def test_civilians_medium_has_three_zones():
    """Medium task must generate exactly 3 civilian zones."""
    assert len(generate_civilians("medium", seed=42)) == 3


def test_civilians_hard_has_five_zones():
    """Hard task must generate exactly 5 civilian zones."""
    assert len(generate_civilians("hard", seed=999)) == 5


def test_civilians_severity_in_constants_range():
    """Every zone severity must be within the range defined in constants.py."""
    for level in ["easy", "medium", "hard"]:
        sev_min, sev_max = SEVERITY_RANGES[level]
        for z in generate_civilians(level, seed=42):
            if level == "hard" and z["zone_id"] == 0:
                assert z["severity"] == 1.0, \
                    "Kedarnath must be 1.0 in hard mode!"
            else:
                assert sev_min <= z["severity"] <= sev_max, \
                    f"{level} {z['name']} severity {z['severity']} out of constants range!"


def test_civilians_people_in_constants_range():
    """Every zone people count must be within the range defined in constants.py."""
    for level in ["easy", "medium", "hard"]:
        ppl_min, ppl_max = PEOPLE_RANGES[level]
        for z in generate_civilians(level, seed=42):
            assert ppl_min <= z["people"] <= ppl_max, \
                f"{level} {z['name']} people {z['people']} out of constants range!"


def test_civilians_people_positive():
    """Every zone must have at least 1 person trapped."""
    for z in generate_civilians("hard", seed=999):
        assert z["people"] >= 1, \
            f"{z['name']} has {z['people']} people — must be >= 1"


def test_civilians_rescued_starts_at_zero():
    """No one should be rescued at the start of an episode."""
    for z in generate_civilians("medium", seed=42):
        assert z["rescued"]      == 0, f"{z['name']} starts with rescued != 0"
        assert z["time_waiting"] == 0, f"{z['name']} starts with time_waiting != 0"


def test_civilians_all_active_at_start():
    """All zones must be active (is_active=True) at episode start."""
    for z in generate_civilians("hard", seed=42):
        assert z["is_active"] is True, f"{z['name']} is not active at start!"


def test_civilians_correct_place_names():
    """Easy zone must always be Kedarnath Temple Area, Rudraprayag."""
    z = generate_civilians("easy", seed=42)[0]
    assert z["name"]     == "Kedarnath Temple Area"
    assert z["district"] == "Rudraprayag"


def test_civilians_valid_disaster_types():
    """Every zone must carry a recognised disaster type."""
    valid = {"flash_flood", "landslide", "river_overflow",
             "land_subsidence", "glacier_burst"}
    for z in generate_civilians("hard", seed=999):
        assert z["disaster_type"] in valid, \
            f"{z['name']} has unrecognised disaster_type: {z['disaster_type']}"


def test_hard_kedarnath_always_critical():
    """In hard mode, Kedarnath must always start at maximum severity (1.0)."""
    zones = generate_civilians("hard", seed=999)
    assert zones[0]["severity"] == 1.0, \
        "Kedarnath must be 1.0 severity in hard mode!"


def test_hard_other_zones_not_fixed():
    """Only Kedarnath is fixed at 1.0 — other zones must be randomly generated."""
    for z in generate_civilians("hard", seed=999)[1:]:
        assert z["severity"] != 1.0, \
            f"{z['name']} should not be fixed at 1.0!"


def test_civilians_matches_grid():
    """generators.py and grid.py must produce identical output for same seed."""
    for level, seed in [("easy", 42), ("medium", 123), ("hard", 999)]:
        gen_zones  = generate_civilians(level, seed)
        grid_zones = GridWorld(task_level=level, seed=seed).reset()["zones"]
        for cz, gz in zip(gen_zones, grid_zones):
            assert cz["severity"] == gz["severity"], \
                f"{level} {cz['name']} severity mismatch: gen={cz['severity']} grid={gz['severity']}"
            assert cz["people"] == gz["people"], \
                f"{level} {cz['name']} people mismatch: gen={cz['people']} grid={gz['people']}"


# ---------------------------------------------------------------------------
# Group 3 — generate_scenario (3 tests)
# ---------------------------------------------------------------------------

def test_scenario_has_all_keys():
    """Scenario dict must contain all required top-level keys."""
    sc = generate_scenario("medium", seed=42)
    for key in ["task_level", "seed", "max_steps", "resources", "zones"]:
        assert key in sc, f"Scenario missing key: {key}"


def test_scenario_max_steps_positive():
    """Max steps must be a positive integer for every difficulty."""
    for level in ["easy", "medium", "hard"]:
        assert generate_scenario(level, seed=1)["max_steps"] > 0


def test_scenario_reproducible():
    """Same seed must produce the same full scenario."""
    assert generate_scenario("hard", seed=77) == generate_scenario("hard", seed=77)


# ---------------------------------------------------------------------------
# Group 4 — sync_with_grid (2 tests)
# ---------------------------------------------------------------------------

def test_sync_updates_severity():
    """After spread_threat(), sync_with_grid must update severity in civilians."""
    grid      = GridWorld(task_level="medium", seed=42)
    grid.reset()
    civilians = generate_civilians("medium", seed=42)

    grid.zones[0].severity = 0.9
    before = civilians[1]["severity"]
    grid.spread_threat()
    sync_with_grid(civilians, grid)

    assert civilians[1]["severity"] > before, \
        "sync_with_grid did not update severity after spread!"


def test_sync_updates_time_waiting():
    """After tick(), sync_with_grid must reflect increased waiting time."""
    grid      = GridWorld(task_level="easy", seed=42)
    grid.reset()
    civilians = generate_civilians("easy", seed=42)

    grid.tick(spread=False)
    sync_with_grid(civilians, grid)

    assert civilians[0]["time_waiting"] == 1, \
        "sync_with_grid did not update time_waiting after tick!"


# To run all tests:
# python -m pytest tests/test_generators.py -v