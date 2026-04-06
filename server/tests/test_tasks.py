import pytest
from tasks import get_task, get_task_scenario, list_tasks, TASKS
from constants import STEP_LIMITS, RESOURCE_CONFIG


# ---------------------------------------------------------------------------
# Group 1 — list_tasks (2 tests)
# ---------------------------------------------------------------------------

def test_list_tasks_returns_all_levels():
    """list_tasks must return all three difficulty levels."""
    levels = list_tasks()
    for level in ["easy", "medium", "hard"]:
        assert level in levels, f"list_tasks() missing level: {level}"


def test_list_tasks_returns_exactly_three():
    """list_tasks must return exactly 3 levels — no more, no less."""
    assert len(list_tasks()) == 3


# ---------------------------------------------------------------------------
# Group 2 — get_task valid levels (3 tests)
# ---------------------------------------------------------------------------

def test_get_task_easy_returns_dict():
    """get_task('easy') must return a dict."""
    assert isinstance(get_task("easy"), dict)


def test_get_task_medium_returns_dict():
    """get_task('medium') must return a dict."""
    assert isinstance(get_task("medium"), dict)


def test_get_task_hard_returns_dict():
    """get_task('hard') must return a dict."""
    assert isinstance(get_task("hard"), dict)


# ---------------------------------------------------------------------------
# Group 3 — get_task invalid level (2 tests)
# ---------------------------------------------------------------------------

def test_get_task_invalid_raises_value_error():
    """get_task with unknown level must raise ValueError."""
    with pytest.raises(ValueError):
        get_task("extreme")


def test_get_task_empty_string_raises():
    """get_task with empty string must raise ValueError."""
    with pytest.raises(ValueError):
        get_task("")


# ---------------------------------------------------------------------------
# Group 4 — Required keys in every task (1 test)
# ---------------------------------------------------------------------------

def test_all_tasks_have_required_keys():
    """Every task config must have all required top-level keys."""
    required = [
        "seed", "task_level", "max_steps", "resources",
        "spread", "spread_interval", "spawn_victims",
        "success_threshold", "description", "objective",
        "observation_space", "action_space",
    ]
    for level in ["easy", "medium", "hard"]:
        task = get_task(level)
        for key in required:
            assert key in task, f"'{level}' task missing key: '{key}'"


# ---------------------------------------------------------------------------
# Group 5 — task_level field matches the key (3 tests)
# ---------------------------------------------------------------------------

def test_easy_task_level_field_correct():
    """task_level field in easy config must be 'easy'."""
    assert get_task("easy")["task_level"] == "easy"


def test_medium_task_level_field_correct():
    """task_level field in medium config must be 'medium'."""
    assert get_task("medium")["task_level"] == "medium"


def test_hard_task_level_field_correct():
    """task_level field in hard config must be 'hard'."""
    assert get_task("hard")["task_level"] == "hard"


# ---------------------------------------------------------------------------
# Group 6 — Seeds (3 tests)
# ---------------------------------------------------------------------------

def test_easy_seed_is_42():
    """Easy task must use seed 42 for reproducibility."""
    assert get_task("easy")["seed"] == 42


def test_medium_seed_is_123():
    """Medium task must use seed 123 for reproducibility."""
    assert get_task("medium")["seed"] == 123


def test_hard_seed_is_999():
    """Hard task must use seed 999 for reproducibility."""
    assert get_task("hard")["seed"] == 999


# ---------------------------------------------------------------------------
# Group 7 — max_steps matches constants.py (3 tests)
# ---------------------------------------------------------------------------

def test_easy_max_steps_matches_constants():
    """Easy max_steps must match STEP_LIMITS['easy'] from constants.py."""
    assert get_task("easy")["max_steps"] == STEP_LIMITS["easy"]


def test_medium_max_steps_matches_constants():
    """Medium max_steps must match STEP_LIMITS['medium'] from constants.py."""
    assert get_task("medium")["max_steps"] == STEP_LIMITS["medium"]


def test_hard_max_steps_matches_constants():
    """Hard max_steps must match STEP_LIMITS['hard'] from constants.py."""
    assert get_task("hard")["max_steps"] == STEP_LIMITS["hard"]


# ---------------------------------------------------------------------------
# Group 8 — resources match constants.py (3 tests)
# ---------------------------------------------------------------------------

def test_easy_resources_match_constants():
    """Easy resources must match RESOURCE_CONFIG['easy'] from constants.py."""
    assert get_task("easy")["resources"] == RESOURCE_CONFIG["easy"]


def test_medium_resources_match_constants():
    """Medium resources must match RESOURCE_CONFIG['medium'] from constants.py."""
    assert get_task("medium")["resources"] == RESOURCE_CONFIG["medium"]


def test_hard_resources_match_constants():
    """Hard resources must match RESOURCE_CONFIG['hard'] from constants.py."""
    assert get_task("hard")["resources"] == RESOURCE_CONFIG["hard"]


# ---------------------------------------------------------------------------
# Group 9 — Spread configuration (4 tests)
# ---------------------------------------------------------------------------

def test_easy_no_spread():
    """Easy task must have spread=False — disaster does not spread."""
    assert get_task("easy")["spread"] is False


def test_medium_spread_enabled():
    """Medium task must have spread=True — disaster spreads every 5 steps."""
    assert get_task("medium")["spread"] is True


def test_medium_spread_interval_is_5():
    """Medium spread_interval must be 5 steps."""
    assert get_task("medium")["spread_interval"] == 5


def test_hard_spread_interval_is_2():
    """Hard spread_interval must be 2 steps — much more aggressive spreading."""
    assert get_task("hard")["spread_interval"] == 2


# ---------------------------------------------------------------------------
# Group 10 — Spawn victims configuration (3 tests)
# ---------------------------------------------------------------------------

def test_easy_no_spawn():
    """Easy task must not spawn new victims mid-episode."""
    assert get_task("easy")["spawn_victims"] is False


def test_medium_no_spawn():
    """Medium task must not spawn new victims mid-episode."""
    assert get_task("medium")["spawn_victims"] is False


def test_hard_spawn_enabled():
    """Hard task must spawn new victims mid-episode — worsening disaster."""
    assert get_task("hard")["spawn_victims"] is True


# ---------------------------------------------------------------------------
# Group 11 — Success thresholds (4 tests)
# ---------------------------------------------------------------------------

def test_easy_threshold_is_0_5():
    """Easy success_threshold must be 0.5."""
    assert get_task("easy")["success_threshold"] == 0.5


def test_medium_threshold_is_0_6():
    """Medium success_threshold must be 0.6."""
    assert get_task("medium")["success_threshold"] == 0.6


def test_hard_threshold_is_0_7():
    """Hard success_threshold must be 0.7."""
    assert get_task("hard")["success_threshold"] == 0.7


def test_thresholds_increase_with_difficulty():
    """Success threshold must increase easy → medium → hard."""
    easy_t   = get_task("easy")["success_threshold"]
    medium_t = get_task("medium")["success_threshold"]
    hard_t   = get_task("hard")["success_threshold"]
    assert easy_t < medium_t < hard_t, \
        "Success thresholds must strictly increase with difficulty!"


# ---------------------------------------------------------------------------
# Group 12 — Action space (3 tests)
# ---------------------------------------------------------------------------

def test_hard_action_space_no_helicopter():
    """Hard action space must not include helicopters — none available."""
    unit_types = get_task("hard")["action_space"]["unit_types"]
    assert "helicopter" not in unit_types, \
        "Hard mode has no helicopters — must not appear in action_space!"


def test_easy_action_space_has_helicopter():
    """Easy action space must include helicopter as a unit type."""
    assert "helicopter" in get_task("easy")["action_space"]["unit_types"]


def test_all_tasks_action_space_type_discrete():
    """All tasks must declare a discrete action space."""
    for level in ["easy", "medium", "hard"]:
        assert get_task(level)["action_space"]["type"] == "discrete", \
            f"{level} action_space type must be 'discrete'!"


# ---------------------------------------------------------------------------
# Group 13 — Observation space (2 tests)
# ---------------------------------------------------------------------------

def test_all_tasks_have_observation_space_keys():
    """Every task observation_space must contain zones, resources, step, max_steps."""
    required = ["zones", "resources", "step", "max_steps"]
    for level in ["easy", "medium", "hard"]:
        obs = get_task(level)["observation_space"]
        for key in required:
            assert key in obs, \
                f"'{level}' observation_space missing key: '{key}'"


def test_all_tasks_description_nonempty():
    """Every task must have a non-empty description and objective string."""
    for level in ["easy", "medium", "hard"]:
        task = get_task(level)
        assert len(task["description"]) > 10, \
            f"'{level}' task description is too short or empty!"
        assert len(task["objective"]) > 5, \
            f"'{level}' task objective is too short or empty!"


# ---------------------------------------------------------------------------
# Group 14 — get_task_scenario (5 tests)
# ---------------------------------------------------------------------------

def test_get_task_scenario_returns_all_keys():
    """get_task_scenario must return dict with all required keys."""
    sc = get_task_scenario("easy")
    for key in ["task_level", "seed", "max_steps", "resources", "zones"]:
        assert key in sc, f"Scenario missing key: {key}"


def test_get_task_scenario_task_level_correct():
    """Scenario task_level must match the requested level."""
    for level in ["easy", "medium", "hard"]:
        assert get_task_scenario(level)["task_level"] == level


def test_get_task_scenario_seed_matches_task():
    """Scenario seed must match the seed defined in the task config."""
    for level in ["easy", "medium", "hard"]:
        assert get_task_scenario(level)["seed"] == get_task(level)["seed"]


def test_get_task_scenario_reproducible():
    """get_task_scenario must produce the same result every call (fixed seed)."""
    for level in ["easy", "medium", "hard"]:
        assert get_task_scenario(level) == get_task_scenario(level), \
            f"get_task_scenario('{level}') is not reproducible!"


def test_get_task_scenario_zones_correct_count():
    """Scenario zones count must match the expected zone count per level."""
    expected = {"easy": 1, "medium": 3, "hard": 5}
    for level, count in expected.items():
        sc = get_task_scenario(level)
        assert len(sc["zones"]) == count, \
            f"'{level}' scenario has {len(sc['zones'])} zones, expected {count}!"


# ---------------------------------------------------------------------------
# Group 15 — Dashboard rules verification (4 tests)
# ---------------------------------------------------------------------------

def test_dashboard_three_tasks_exist():
    """Dashboard rule: must have minimum 3 tasks (easy, medium, hard)."""
    assert len(list_tasks()) >= 3


def test_dashboard_success_thresholds_in_range():
    """Dashboard rule: all success thresholds must be in [0.0, 1.0]."""
    for level in ["easy", "medium", "hard"]:
        t = get_task(level)["success_threshold"]
        assert 0.0 <= t <= 1.0, \
            f"'{level}' threshold {t} outside [0.0, 1.0]!"


def test_dashboard_max_steps_positive():
    """Dashboard rule: every task must have a positive step budget."""
    for level in ["easy", "medium", "hard"]:
        assert get_task(level)["max_steps"] > 0, \
            f"'{level}' max_steps must be > 0!"


def test_dashboard_hard_genuinely_harder():
    """Dashboard rule: hard must be harder than easy — higher threshold, fewer steps."""
    easy = get_task("easy")
    hard = get_task("hard")
    assert hard["success_threshold"] > easy["success_threshold"], \
        "Hard threshold must be higher than easy!"
    assert hard["max_steps"] < easy["max_steps"] or hard["spawn_victims"] is True, \
        "Hard must be genuinely harder (fewer steps or spawning victims)!"


# To run all tests:
# python -m pytest tests/test_tasks.py -v