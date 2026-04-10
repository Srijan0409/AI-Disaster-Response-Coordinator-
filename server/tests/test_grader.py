"""
=============================================================================
TEST SUITE - grader.py
=============================================================================
Run with:
    pytest test_grader.py -v
=============================================================================
"""

import copy
import pytest
from disaster_env.server.grader import (
    _normalize,
    _bucket,
    _compute_rescue_score,
    _compute_time_penalty,
    _compute_wait_penalty,
    _compute_spawn_penalty,
    _compute_score_components,
    calculate_step_reward,
    compute_reward,
    grade_episode,
    REWARD_WEIGHTS,
    SCORE_MIN,
    SCORE_MAX,
    HIGH_SEVERITY_THRESHOLD,
    MED_SEVERITY_THRESHOLD,
)
from disaster_env.server.grid import GridWorld
from disaster_env.server.tasks import get_task

LEVELS = ["easy", "medium", "hard"]


def make_zone(zone_id=0, severity=0.8, is_active=True, time_waiting=0,
              people=2, rescued=0, victims=None):
    return {
        "zone_id": zone_id,
        "severity": severity,
        "is_active": is_active,
        "time_waiting": time_waiting,
        "people": people,
        "rescued": rescued,
        "victims": victims or [],
    }


def make_victim(vid=0, rescued=False, alive=True):
    return {"id": vid, "rescued": rescued, "alive": alive}


# =============================================================================
# _normalize
# =============================================================================

class TestNormalize:

    def test_zero_gives_score_min(self):
        assert _normalize(0.0) == SCORE_MIN

    def test_one_gives_score_max(self):
        assert _normalize(1.0) == SCORE_MAX

    def test_half_gives_midpoint(self):
        expected = SCORE_MIN + 0.5 * (SCORE_MAX - SCORE_MIN)
        assert abs(_normalize(0.5) - expected) < 1e-9

    def test_negative_clamped_to_score_min(self):
        assert _normalize(-99.0) == SCORE_MIN

    def test_above_one_clamped_to_score_max(self):
        assert _normalize(5.0) == SCORE_MAX

    def test_output_always_in_open_interval(self):
        for x in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            v = _normalize(x)
            assert 0.0 < v < 1.0

    def test_monotone(self):
        vals = [_normalize(x / 10) for x in range(11)]
        assert vals == sorted(vals)


# =============================================================================
# _bucket
# =============================================================================

class TestBucket:

    def test_high_at_threshold(self):
        assert _bucket(HIGH_SEVERITY_THRESHOLD) == "high"

    def test_high_above_threshold(self):
        assert _bucket(1.0) == "high"

    def test_med_at_threshold(self):
        assert _bucket(MED_SEVERITY_THRESHOLD) == "med"

    def test_med_below_high(self):
        assert _bucket(HIGH_SEVERITY_THRESHOLD - 0.01) == "med"

    def test_low_below_med_threshold(self):
        assert _bucket(MED_SEVERITY_THRESHOLD - 0.01) == "low"

    def test_low_at_zero(self):
        assert _bucket(0.0) == "low"


# =============================================================================
# calculate_step_reward
# =============================================================================

class TestCalculateStepReward:

    def test_invalid_action_low_reward(self):
        zones = [make_zone(severity=0.9)]
        r = calculate_step_reward(0, zones, False)
        assert r < 0.5

    def test_high_severity_valid_high_reward(self):
        zones = [make_zone(severity=0.9)]
        r = calculate_step_reward(0, zones, True)
        assert r > 0.7

    def test_med_severity_valid_moderate_reward(self):
        zones = [make_zone(severity=0.5)]
        r = calculate_step_reward(0, zones, True)
        assert 0.4 < r < 0.8

    def test_low_severity_valid_lower_reward(self):
        zones = [make_zone(severity=0.1)]
        r = calculate_step_reward(0, zones, True)
        assert 0.3 < r < 0.7

    def test_wasted_step_inactive_zone(self):
        zones = [make_zone(is_active=False)]
        r = calculate_step_reward(0, zones, True)
        assert r < 0.5

    def test_missing_zone_id_returns_low_reward(self):
        zones = [make_zone(zone_id=0)]
        r = calculate_step_reward(99, zones, True)
        assert r < 0.5

    def test_missing_is_active_key_no_keyerror(self):
        """FIX D: missing 'is_active' must not raise KeyError."""
        zones = [{"zone_id": 0, "severity": 0.9}]
        r = calculate_step_reward(0, zones, True)
        assert r > 0.7  # treated as active

    def test_output_always_in_open_interval(self):
        for sev in [0.1, 0.5, 0.9]:
            for valid in [True, False]:
                zones = [make_zone(severity=sev)]
                r = calculate_step_reward(0, zones, valid)
                assert 0.0 < r < 1.0

    def test_high_severity_better_than_low_severity(self):
        zones_h = [make_zone(severity=0.9)]
        zones_l = [make_zone(severity=0.1)]
        r_h = calculate_step_reward(0, zones_h, True)
        r_l = calculate_step_reward(0, zones_l, True)
        assert r_h > r_l


# =============================================================================
# _compute_time_penalty
# =============================================================================

class TestComputeTimePenalty:

    def test_single_step_no_penalty(self):
        assert _compute_time_penalty(1, 30, 0.05) == 0.0

    def test_full_steps_gives_full_weight(self):
        p = _compute_time_penalty(30, 30, 0.05)
        assert abs(p - 0.05) < 1e-9

    def test_half_steps_half_weight(self):
        p = _compute_time_penalty(15, 30, 0.10)
        assert abs(p - 0.05) < 1e-9

    def test_zero_max_steps_no_error(self):
        assert _compute_time_penalty(5, 0, 0.1) == 0.0

    def test_weight_zero_always_zero(self):
        assert _compute_time_penalty(30, 30, 0.0) == 0.0

    def test_proportional(self):
        p1 = _compute_time_penalty(10, 30, 0.1)
        p2 = _compute_time_penalty(20, 30, 0.1)
        assert p2 > p1


# =============================================================================
# _compute_wait_penalty
# =============================================================================

class TestComputeWaitPenalty:

    def test_no_active_zones_no_penalty(self):
        zones = [make_zone(is_active=False, time_waiting=100)]
        assert _compute_wait_penalty(zones, 0.01) == 0.0

    def test_active_zone_with_waiting_has_penalty(self):
        zones = [make_zone(is_active=True, time_waiting=5)]
        p = _compute_wait_penalty(zones, 0.01)
        assert p > 0

    def test_missing_is_active_no_keyerror(self):
        """FIX C: missing is_active must not raise KeyError."""
        zones = [{"time_waiting": 5}]
        p = _compute_wait_penalty(zones, 0.01)
        assert p >= 0.0

    def test_weight_zero_gives_zero(self):
        zones = [make_zone(is_active=True, time_waiting=99)]
        assert _compute_wait_penalty(zones, 0.0) == 0.0

    def test_capped_at_weight(self):
        zones = [make_zone(is_active=True, time_waiting=1000)]
        p = _compute_wait_penalty(zones, 0.02)
        assert p <= 0.02

    def test_more_waiting_more_penalty(self):
        z1 = [make_zone(is_active=True, time_waiting=2)]
        z2 = [make_zone(is_active=True, time_waiting=8)]
        assert _compute_wait_penalty(z2, 0.01) > _compute_wait_penalty(z1, 0.01)


# =============================================================================
# _compute_spawn_penalty
# =============================================================================

class TestComputeSpawnPenalty:

    def test_no_spawned_no_penalty(self):
        assert _compute_spawn_penalty(0, 0, 0.05) == 0.0

    def test_all_rescued_no_penalty(self):
        assert _compute_spawn_penalty(10, 10, 0.05) == 0.0

    def test_none_rescued_full_weight(self):
        p = _compute_spawn_penalty(10, 0, 0.05)
        assert abs(p - 0.05) < 1e-9

    def test_clamped_when_rescued_exceeds_spawned(self):
        """FIX A: missed_fraction must not go negative."""
        p = _compute_spawn_penalty(5, 10, 0.05)
        assert p == 0.0

    def test_weight_zero_always_zero(self):
        assert _compute_spawn_penalty(10, 0, 0.0) == 0.0

    def test_proportional_to_missed_fraction(self):
        p_half = _compute_spawn_penalty(10, 5, 0.10)
        assert abs(p_half - 0.05) < 1e-9


# =============================================================================
# _compute_rescue_score
# =============================================================================

class TestComputeRescueScore:

    def _zone(self, zone_id, severity, victims):
        return {"zone_id": zone_id, "severity": severity, "victims": victims}

    def test_all_rescued_high_severity_gives_high_weight(self):
        v_init  = [make_victim(0, rescued=False)]
        v_final = [make_victim(0, rescued=True)]
        init    = [self._zone(0, 0.9, v_init)]
        final   = [self._zone(0, 0.9, v_final)]
        w = REWARD_WEIGHTS["easy"]
        s = _compute_rescue_score(init, final, w)
        assert abs(s - w["high_severity_weight"]) < 1e-6

    def test_none_rescued_gives_zero(self):
        v = [make_victim(0, rescued=False)]
        init  = [self._zone(0, 0.9, copy.deepcopy(v))]
        final = [self._zone(0, 0.9, copy.deepcopy(v))]
        s = _compute_rescue_score(init, final, REWARD_WEIGHTS["easy"])
        assert s == 0.0

    def test_partial_rescue_partial_score(self):
        v_init  = [make_victim(0, False), make_victim(1, False)]
        v_final = [make_victim(0, True),  make_victim(1, False)]
        init  = [self._zone(0, 0.9, v_init)]
        final = [self._zone(0, 0.9, v_final)]
        w = REWARD_WEIGHTS["easy"]
        s = _compute_rescue_score(init, final, w)
        assert abs(s - w["high_severity_weight"] * 0.5) < 1e-6

    def test_unknown_zone_in_final_skipped_no_keyerror(self):
        """FIX B: zone in final not in initial must not raise."""
        v = [make_victim(0, rescued=True)]
        init  = [self._zone(0, 0.9, [make_victim(0, False)])]
        final = [self._zone(0, 0.9, v), self._zone(99, 0.9, v)]
        s = _compute_rescue_score(init, final, REWARD_WEIGHTS["easy"])
        assert s >= 0.0

    def test_uses_initial_severity_not_final(self):
        """BUG G NOTE: bucket derived from initial severity."""
        v_init  = [make_victim(0, False)]
        v_final = [make_victim(0, True)]
        init    = [self._zone(0, 0.9, v_init)]   # high severity
        # final severity changed to low - should not affect bucket
        final_zone = self._zone(0, 0.1, v_final)
        final_zone["severity"] = 0.1
        s = _compute_rescue_score(init, [final_zone], REWARD_WEIGHTS["easy"])
        # Uses high bucket (initial) - score should equal high_severity_weight
        assert abs(s - REWARD_WEIGHTS["easy"]["high_severity_weight"]) < 1e-6

    def test_fallback_to_people_rescued_when_no_victims_list(self):
        init  = [{"zone_id": 0, "severity": 0.9, "people": 4, "victims": []}]
        final = [{"zone_id": 0, "severity": 0.9, "rescued": 2, "people": 4, "victims": []}]
        s = _compute_rescue_score(init, final, REWARD_WEIGHTS["easy"])
        assert abs(s - REWARD_WEIGHTS["easy"]["high_severity_weight"] * 0.5) < 1e-6


# =============================================================================
# _compute_score_components
# =============================================================================

class TestComputeScoreComponents:

    def _run(self, level="easy"):
        task = get_task(level)
        g = GridWorld(level, task["seed"])
        g.reset()
        zi = copy.deepcopy(g._zones_initial)
        for _ in range(task["max_steps"]):
            result = g.step({"zone_id": 0, "unit_type": "rescue_team"})
            if result["done"]:
                break
        zf = g.get_state()["zones"]
        return zi, zf, g.step_num

    def test_returns_required_keys(self):
        zi, zf, steps = self._run()
        c = _compute_score_components("easy", zi, zf, steps, 0, 0)
        for key in ["base", "t_pen", "w_pen", "s_pen", "total_penalty",
                    "raw_score", "final_score", "task"]:
            assert key in c

    def test_final_score_in_open_interval(self):
        zi, zf, steps = self._run()
        c = _compute_score_components("easy", zi, zf, steps, 0, 0)
        assert 0.0 < c["final_score"] < 1.0

    def test_raw_score_non_negative(self):
        zi, zf, steps = self._run()
        c = _compute_score_components("easy", zi, zf, steps, 0, 0)
        assert c["raw_score"] >= 0.0

    def test_total_penalty_not_exceed_half_base(self):
        zi, zf, steps = self._run()
        c = _compute_score_components("easy", zi, zf, steps, 0, 0)
        assert c["total_penalty"] <= c["base"] * 0.5 + 1e-9


# =============================================================================
# compute_reward
# =============================================================================

class TestComputeReward:

    def _run(self, level="easy"):
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
        return last["reward"], zi, zf, g

    def test_easy_reward_in_open_interval(self):
        r, _, _, _ = self._run("easy")
        assert 0.0 < r < 1.0

    def test_medium_reward_in_open_interval(self):
        r, _, _, _ = self._run("medium")
        assert 0.0 < r < 1.0

    def test_hard_reward_in_open_interval(self):
        r, _, _, _ = self._run("hard")
        assert 0.0 < r < 1.0

    def test_terminal_reward_equals_episode_reward_in_info(self):
        """FIX 16."""
        task = get_task("easy")
        g = GridWorld("easy", task["seed"])
        g.reset()
        result = None
        for _ in range(task["max_steps"]):
            result = g.step({"zone_id": 0, "unit_type": "rescue_team"})
        assert result["reward"] == result["info"]["episode_reward"]


# =============================================================================
# grade_episode
# =============================================================================

class TestGradeEpisode:

    def _run(self, level="easy"):
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
        return grade_episode(level, zi, zf, g.step_num), last

    def test_returns_required_keys(self):
        report, _ = self._run()
        required = {"task_level", "score", "passed", "success_threshold",
                    "breakdown", "stats"}
        assert required.issubset(report.keys())

    def test_score_in_open_interval(self):
        report, _ = self._run()
        assert 0.0 < report["score"] < 1.0

    def test_breakdown_keys(self):
        report, _ = self._run()
        for key in ["base_rescue_score", "time_penalty", "wait_penalty",
                    "spawn_penalty", "total_penalty", "raw_score"]:
            assert key in report["breakdown"]

    def test_stats_keys(self):
        report, _ = self._run()
        for key in ["total_people", "total_rescued", "steps_taken",
                    "max_steps", "spawned_victims", "rescued_spawned"]:
            assert key in report["stats"]

    def test_grade_score_matches_env_step_reward(self):
        """BUG I FIX: grade_episode score must equal terminal step reward."""
        task = get_task("easy")
        g = GridWorld("easy", task["seed"])
        g.reset()
        zi = copy.deepcopy(g._zones_initial)
        last = None
        for _ in range(task["max_steps"]):
            last = g.step({"zone_id": 0, "unit_type": "rescue_team"})
            if last["done"]:
                break
        zf = g.get_state()["zones"]
        report = grade_episode("easy", zi, zf, g.step_num)
        assert report["score"] == last["reward"]

    def test_passed_consistent_with_threshold(self):
        report, _ = self._run()
        task = get_task("easy")
        expected = report["score"] >= task["success_threshold"]
        assert report["passed"] == expected

    def test_total_rescued_never_exceeds_total_people(self):
        report, _ = self._run()
        assert report["stats"]["total_rescued"] <= report["stats"]["total_people"]

    def test_task_level_stored_correctly(self):
        for level in LEVELS:
            task = get_task(level)
            g = GridWorld(level, task["seed"])
            g.reset()
            zi = copy.deepcopy(g._zones_initial)
            for _ in range(task["max_steps"]):
                r = g.step({"zone_id": 0, "unit_type": "rescue_team"})
                if r["done"]:
                    break
            zf = g.get_state()["zones"]
            report = grade_episode(level, zi, zf, g.step_num)
            assert report["task_level"] == level

    def test_hard_spawn_penalty_applied(self):
        """Hard mode spawn_penalty_weight > 0 - penalty must be computed."""
        assert REWARD_WEIGHTS["hard"]["spawn_penalty_weight"] > 0
        task = get_task("hard")
        g = GridWorld("hard", task["seed"])
        g.reset()
        zi = copy.deepcopy(g._zones_initial)
        for _ in range(task["max_steps"]):
            r = g.step({"zone_id": 0, "unit_type": "rescue_team"})
            if r["done"]:
                break
        zf = g.get_state()["zones"]
        report = grade_episode("hard", zi, zf, g.step_num,
                               spawned_victims=10, rescued_spawned=0)
        assert report["breakdown"]["spawn_penalty"] > 0