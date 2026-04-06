from grader import (
    compute_rescue_score,
    compute_time_penalty,
    compute_wait_penalty,
    compute_spawn_penalty,
    compute_reward,
    grade_episode,
    calculate_step_reward,
    calculate_waiting_penalty,
    REWARD_WEIGHTS,
)
from tasks import get_task


def zone(zone_id, severity, people, rescued=0, time_waiting=0):
    return {
        "zone_id":      zone_id,
        "severity":     severity,
        "people":       people,
        "rescued":      rescued,
        "time_waiting": time_waiting,
        "is_active":    people > rescued,
    }


def pytest_approx(val, rel=1e-6):
    """Returns val — tests use direct == with floats that are exact here."""
    return val


# ---------------------------------------------------------------------------
# Group 1 — compute_rescue_score (5 tests)
# ---------------------------------------------------------------------------

def test_rescue_score_full_high_severity():
    """All high-severity victims rescued must give at least high_severity_weight."""
    w  = REWARD_WEIGHTS["easy"]
    zi = [zone(0, severity=0.9, people=5)]
    zf = [zone(0, severity=0.9, people=5, rescued=5)]
    assert compute_rescue_score(zi, zf, w) >= w["high_severity_weight"]


def test_rescue_score_zero_rescued():
    """Nobody rescued → score must be less than high_severity_weight."""
    w  = REWARD_WEIGHTS["easy"]
    zi = [zone(0, severity=0.9, people=5)]
    zf = [zone(0, severity=0.9, people=5, rescued=0)]
    assert compute_rescue_score(zi, zf, w) < w["high_severity_weight"]


def test_rescue_score_partial():
    """Half rescued → score must be less than full rescue score."""
    w       = REWARD_WEIGHTS["medium"]
    zi      = [zone(0, severity=0.8, people=10)]
    zf      = [zone(0, severity=0.8, people=10, rescued=5)]
    zf_full = [zone(0, severity=0.8, people=10, rescued=10)]
    score   = compute_rescue_score(zi, zf,      w)
    s_full  = compute_rescue_score(zi, zf_full, w)
    assert 0.0 < score < s_full, "Partial rescue must score less than full rescue!"


def test_rescue_score_capped_at_initial_people():
    """Rescued count must be capped at initial people — spawned victims must not inflate score."""
    w  = REWARD_WEIGHTS["hard"]
    zi = [zone(0, severity=0.9, people=5)]
    zf_spawned = [zone(0, severity=0.9, people=10, rescued=10)]
    zf_normal  = [zone(0, severity=0.9, people=5,  rescued=5)]
    assert compute_rescue_score(zi, zf_spawned, w) == compute_rescue_score(zi, zf_normal, w)


def test_rescue_score_no_victims_in_bucket_gives_full_weight():
    """If no high-severity victims exist, full high weight is awarded."""
    w  = REWARD_WEIGHTS["easy"]
    zi = [zone(0, severity=0.2, people=5)]
    zf = [zone(0, severity=0.2, people=5, rescued=5)]
    score = compute_rescue_score(zi, zf, w)
    assert score >= w["low_severity_weight"]

# ---------------------------------------------------------------------------
# Group 2 — compute_time_penalty (3 tests)
# ---------------------------------------------------------------------------

def test_time_penalty_full_steps():
    """Using all steps must give maximum penalty = weight."""
    w = REWARD_WEIGHTS["easy"]["time_penalty_weight"]
    assert compute_time_penalty(30, 30, w) == pytest_approx(w)


def test_time_penalty_zero_steps():
    """Zero steps used must give zero penalty."""
    assert compute_time_penalty(0, 30, 0.05) == 0.0


def test_time_penalty_proportional():
    """Half steps used must give half the maximum penalty."""
    w    = 0.10
    half = compute_time_penalty(20, 40, w)
    full = compute_time_penalty(40, 40, w)
    assert abs(half - full / 2) < 1e-9


# ---------------------------------------------------------------------------
# Group 3 — compute_wait_penalty (3 tests)
# ---------------------------------------------------------------------------

def test_wait_penalty_all_rescued_is_zero():
    """All zones fully rescued → no active zones → wait penalty = 0."""
    zones_f = [zone(0, 0.9, 5, rescued=5, time_waiting=10)]
    assert compute_wait_penalty(zones_f, weight=0.02) == 0.0


def test_wait_penalty_positive_when_waiting():
    """Active zones with time_waiting > 0 must give positive penalty."""
    zones_f = [zone(0, 0.9, 5, rescued=0, time_waiting=5)]
    assert compute_wait_penalty(zones_f, weight=0.02) > 0.0


def test_wait_penalty_capped():
    """Wait penalty must not exceed the weight even with very long waits."""
    zones_f = [zone(0, 0.9, 5, rescued=0, time_waiting=1000)]
    assert compute_wait_penalty(zones_f, weight=0.03) <= 0.03


# ---------------------------------------------------------------------------
# Group 4 — compute_spawn_penalty (3 tests)
# ---------------------------------------------------------------------------

def test_spawn_penalty_no_spawning():
    """Zero spawned victims must give zero penalty (easy and medium)."""
    assert compute_spawn_penalty(0, 0, weight=0.05) == 0.0


def test_spawn_penalty_all_rescued():
    """All spawned victims rescued → zero penalty."""
    assert compute_spawn_penalty(10, 10, weight=0.05) == 0.0


def test_spawn_penalty_none_rescued():
    """No spawned victims rescued → full weight penalty."""
    assert compute_spawn_penalty(10, 0, weight=0.05) == pytest_approx(0.05)


# ---------------------------------------------------------------------------
# Group 5 — compute_reward (5 tests)
# ---------------------------------------------------------------------------

def test_reward_in_range():
    """Final reward must always be in [0.0, 1.0]."""
    zi = [zone(0, severity=0.9, people=5)]
    zf = [zone(0, severity=0.9, people=5, rescued=3, time_waiting=2)]
    r  = compute_reward("easy", zi, zf, steps_taken=15)
    assert 0.0 <= r <= 1.0


def test_reward_perfect_score():
    """All rescued instantly → very high score (close to 1.0)."""
    zi = [zone(0, severity=0.9, people=5)]
    zf = [zone(0, severity=0.9, people=5, rescued=5, time_waiting=0)]
    r  = compute_reward("easy", zi, zf, steps_taken=1)
    assert r > 0.8


def test_reward_zero_rescue():
    """Nobody rescued → score must be below easy success_threshold (0.5)."""
    zi = [zone(0, severity=0.9, people=5)]
    zf = [zone(0, severity=0.9, people=5, rescued=0, time_waiting=30)]
    r  = compute_reward("easy", zi, zf, steps_taken=30)
    assert r < get_task("easy")["success_threshold"]


def test_reward_not_negative():
    """Score must never go below 0.0 even in worst case."""
    zi = [zone(0, severity=0.9, people=20)]
    zf = [zone(0, severity=0.9, people=20, rescued=0, time_waiting=100)]
    r  = compute_reward("hard", zi, zf, steps_taken=35,
                         spawned_victims=20, rescued_spawned=0)
    assert r >= 0.0


def test_reward_hard_spawn_reduces_score():
    """Unrescued spawned victims in hard mode must lower the score."""
    zi = [zone(0, severity=0.9, people=10)]
    zf = [zone(0, severity=0.9, people=10, rescued=10, time_waiting=0)]
    r_no_spawn   = compute_reward("hard", zi, zf, steps_taken=10,
                                  spawned_victims=0,  rescued_spawned=0)
    r_with_spawn = compute_reward("hard", zi, zf, steps_taken=10,
                                  spawned_victims=10, rescued_spawned=0)
    assert r_no_spawn > r_with_spawn


# ---------------------------------------------------------------------------
# Group 6 — grade_episode (5 tests)
# ---------------------------------------------------------------------------

def test_grade_has_all_keys():
    """grade_episode must return all required keys."""
    zi = [zone(0, severity=0.9, people=5)]
    zf = [zone(0, severity=0.9, people=5, rescued=3)]
    report = grade_episode("easy", zi, zf, steps_taken=10)
    for key in ["task_level", "score", "passed", "success_threshold", "breakdown", "stats"]:
        assert key in report, f"Missing key in report: {key}"


def test_grade_breakdown_keys():
    """Breakdown must contain all four component keys."""
    zi = [zone(0, severity=0.9, people=5)]
    zf = [zone(0, severity=0.9, people=5, rescued=5)]
    bd = grade_episode("medium", zi, zf, steps_taken=20)["breakdown"]
    for key in ["base_rescue_score", "time_penalty", "wait_penalty", "spawn_penalty"]:
        assert key in bd, f"Missing breakdown key: {key}"


def test_grade_passed_true_when_above_threshold():
    """passed must be True when score >= success_threshold."""
    zi = [zone(0, severity=0.9, people=5)]
    zf = [zone(0, severity=0.9, people=5, rescued=5, time_waiting=0)]
    report = grade_episode("easy", zi, zf, steps_taken=1)
    assert report["passed"] is True


def test_grade_passed_false_when_below_threshold():
    """passed must be False when score < success_threshold."""
    zi = [zone(0, severity=0.9, people=10)]
    zf = [zone(0, severity=0.9, people=10, rescued=0, time_waiting=30)]
    report = grade_episode("hard", zi, zf, steps_taken=35,
                            spawned_victims=20, rescued_spawned=0)
    assert report["passed"] is False


def test_grade_stats_correct():
    """Stats block must reflect actual episode numbers."""
    zi = [zone(0, severity=0.9, people=8)]
    zf = [zone(0, severity=0.9, people=8, rescued=5)]
    report = grade_episode("medium", zi, zf, steps_taken=20)
    assert report["stats"]["total_people"]  == 8
    assert report["stats"]["total_rescued"] == 5
    assert report["stats"]["steps_taken"]   == 20
    assert report["stats"]["max_steps"]     == get_task("medium")["max_steps"]


# ---------------------------------------------------------------------------
# Group 7 — calculate_step_reward (6 tests)
# ---------------------------------------------------------------------------

def test_step_reward_high_severity():
    """High severity zone rescue must give +1.0 reward."""
    assert calculate_step_reward(0, [zone(0, 0.9, 5)], action_valid=True) == 1.0


def test_step_reward_medium_severity():
    """Medium severity zone rescue must give +0.5 reward."""
    assert calculate_step_reward(0, [zone(0, 0.5, 5)], action_valid=True) == 0.5


def test_step_reward_low_severity():
    """Low severity zone rescue must give +0.2 reward."""
    assert calculate_step_reward(0, [zone(0, 0.2, 5)], action_valid=True) == 0.2


def test_step_reward_invalid_action():
    """Invalid action must give -0.3 penalty."""
    assert calculate_step_reward(0, [zone(0, 0.9, 5)], action_valid=False) == -0.3


def test_step_reward_wasted_step():
    """Targeting an already fully rescued zone must give -0.1 penalty."""
    assert calculate_step_reward(0, [zone(0, 0.9, 5, rescued=5)], action_valid=True) == -0.1


def test_step_reward_zone_not_exist():
    """Targeting a non-existent zone_id must give -0.3 penalty."""
    assert calculate_step_reward(99, [zone(0, 0.9, 5)], action_valid=True) == -0.3


# ---------------------------------------------------------------------------
# Group 8 — calculate_waiting_penalty (3 tests)
# ---------------------------------------------------------------------------

def test_waiting_penalty_active_high_severity():
    """Active high-severity zone with waiting time must give negative penalty."""
    zones = [zone(0, severity=0.9, people=5, rescued=0, time_waiting=5)]
    assert calculate_waiting_penalty(zones) < 0.0


def test_waiting_penalty_all_rescued_is_zero():
    """Fully rescued zone must give zero waiting penalty."""
    zones = [zone(0, severity=0.9, people=5, rescued=5, time_waiting=5)]
    assert calculate_waiting_penalty(zones) == 0.0


def test_waiting_penalty_low_severity_is_zero():
    """Low severity zones must not contribute to waiting penalty."""
    zones = [zone(0, severity=0.2, people=5, rescued=0, time_waiting=5)]
    assert calculate_waiting_penalty(zones) == 0.0


# To run all tests:
# python -m pytest tests/test_grader.py -v