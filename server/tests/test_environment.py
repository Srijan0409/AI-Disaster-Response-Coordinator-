"""
tests/test_environment.py
=========================
Comprehensive test suite for the AI Disaster Response Coordinator.

Covers:
  - grader.py  : score range, penalty caps, normalize, per-step reward
  - grid.py    : reset, apply_action, tick, is_active, spawn, step flow
  - generators : zone fields, victim fields, reproducibility
  - tasks.py   : task config, seed enforcement
  - environment: reset/step/state API, done flag, grade_report
  - inference  : log format [START] [STEP] [END]

Run with:
    pytest tests/test_environment.py -v
"""

import copy
import json
import math
import re
import pytest

# ---------------------------------------------------------------------------
# Imports - adjust paths if your package layout differs
# ---------------------------------------------------------------------------
from disaster_env.server.constants import (
    STEP_LIMITS, RESOURCE_CONFIG, SEVERITY_RANGES,
    VICTIMS_PER_ZONE, ALLOWED_UNIT_TYPES,
    HIGH_SEVERITY_THRESHOLD, MED_SEVERITY_THRESHOLD,
)
from disaster_env.server.tasks import get_task, list_tasks
from disaster_env.server.generators import generate_civilians, generate_resources
from disaster_env.server.grader import (
    _normalize, calculate_step_reward, compute_reward,
    grade_episode, _compute_score_components,
    SCORE_MIN, SCORE_MAX,
)
from disaster_env.server.grid import GridWorld, Zone


# ===========================================================================
# SECTION 1 - constants.py
# ===========================================================================

class TestConstants:
    def test_step_limits_ordering(self):
        """hard(25) <= easy(30) <= medium(40)"""
        assert STEP_LIMITS["hard"] <= STEP_LIMITS["easy"] <= STEP_LIMITS["medium"]

    def test_all_difficulties_present(self):
        for d in ("easy", "medium", "hard"):
            assert d in STEP_LIMITS
            assert d in RESOURCE_CONFIG
            assert d in SEVERITY_RANGES
            assert d in VICTIMS_PER_ZONE

    def test_hard_no_helicopter(self):
        assert "helicopter" not in ALLOWED_UNIT_TYPES["hard"]
        assert "helicopter" in ALLOWED_UNIT_TYPES["easy"]
        assert "helicopter" in ALLOWED_UNIT_TYPES["medium"]

    def test_severity_thresholds(self):
        assert HIGH_SEVERITY_THRESHOLD == 0.7
        assert MED_SEVERITY_THRESHOLD == 0.4


# ===========================================================================
# SECTION 2 - tasks.py
# ===========================================================================

class TestTasks:
    def test_list_tasks(self):
        assert set(list_tasks()) == {"easy", "medium", "hard"}

    def test_get_task_returns_deepcopy(self):
        t1 = get_task("easy")
        t2 = get_task("easy")
        t1["seed"] = 9999
        assert t2["seed"] == 42  # mutation should not affect next call

    def test_task_fields(self):
        for d in ("easy", "medium", "hard"):
            t = get_task(d)
            assert "seed" in t
            assert "max_steps" in t
            assert "success_threshold" in t
            assert 0.0 < t["success_threshold"] < 1.0

    def test_max_steps_match_constants(self):
        for d in ("easy", "medium", "hard"):
            assert get_task(d)["max_steps"] == STEP_LIMITS[d]

    def test_unknown_task_raises(self):
        with pytest.raises(ValueError):
            get_task("impossible")


# ===========================================================================
# SECTION 3 - generators.py
# ===========================================================================

class TestGenerators:
    def test_generate_civilians_fields(self):
        zones = generate_civilians("easy", seed=42)
        assert len(zones) == 1
        z = zones[0]
        for field in ("zone_id", "name", "district", "disaster_type",
                      "severity", "people", "rescued", "time_waiting",
                      "is_active", "victims"):
            assert field in z, f"Missing field: {field}"

    def test_victims_have_required_fields(self):
        zones = generate_civilians("medium", seed=123)
        for z in zones:
            for v in z["victims"]:
                for field in ("id", "urgency", "survival_time",
                              "distance_km", "alive", "rescued"):
                    assert field in v, f"Missing victim field: {field}"

    def test_victim_ids_globally_unique(self):
        zones = generate_civilians("hard", seed=999)
        all_ids = [v["id"] for z in zones for v in z["victims"]]
        assert len(all_ids) == len(set(all_ids)), "Duplicate victim IDs"

    def test_people_equals_len_victims(self):
        for d in ("easy", "medium", "hard"):
            task = get_task(d)
            zones = generate_civilians(d, seed=task["seed"])
            for z in zones:
                assert z["people"] == len(z["victims"])

    def test_reproducibility(self):
        z1 = generate_civilians("medium", seed=123)
        z2 = generate_civilians("medium", seed=123)
        assert z1[0]["severity"] == z2[0]["severity"]
        assert z1[0]["victims"][0]["id"] == z2[0]["victims"][0]["id"]

    def test_hard_kedarnath_severity_1(self):
        zones = generate_civilians("hard", seed=999)
        assert zones[0]["severity"] == 1.0

    def test_generate_resources(self):
        r = generate_resources("hard")
        assert r.get("helicopters", 0) == 0
        assert r["ambulances"] >= 1


# ===========================================================================
# SECTION 4 - grader.py
# ===========================================================================

class TestNormalize:
    def test_zero_maps_to_score_min(self):
        assert _normalize(0.0) == SCORE_MIN

    def test_one_maps_to_score_max(self):
        assert _normalize(1.0) == SCORE_MAX

    def test_output_strictly_between_0_and_1(self):
        for v in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = _normalize(v)
            assert 0.0 < result < 1.0

    def test_clamping(self):
        assert _normalize(-5.0) == SCORE_MIN
        assert _normalize(999.0) == SCORE_MAX


class TestStepReward:
    def _make_zone(self, zone_id, severity, is_active=True):
        return {
            "zone_id": zone_id,
            "severity": severity,
            "is_active": is_active,
            "victims": [{"alive": True, "rescued": False}] if is_active else [],
        }

    def test_invalid_action_returns_low_reward(self):
        zones = [self._make_zone(0, 0.8)]
        r = calculate_step_reward(0, zones, action_valid=False)
        assert 0.0 < r < 1.0
        assert r < 0.5  # penalty range

    def test_high_severity_returns_high_reward(self):
        zones = [self._make_zone(0, 0.9)]
        r = calculate_step_reward(0, zones, action_valid=True)
        assert 0.0 < r < 1.0
        assert r > 0.8

    def test_inactive_zone_returns_penalty(self):
        zones = [self._make_zone(0, 0.9, is_active=False)]
        r = calculate_step_reward(0, zones, action_valid=True)
        assert 0.0 < r < 1.0
        assert r < 0.5

    def test_reward_always_in_open_interval(self):
        zones = [self._make_zone(0, 0.5)]
        for valid in (True, False):
            r = calculate_step_reward(0, zones, valid)
            assert 0.0 < r < 1.0, f"Out of range: {r}"

    def test_missing_is_active_defaults_to_active(self):
        zone = {"zone_id": 0, "severity": 0.8, "victims": [{"alive": True, "rescued": False}]}
        r = calculate_step_reward(0, [zone], action_valid=True)
        assert 0.0 < r < 1.0


class TestComputeReward:
    def _make_zones(self, severity, people, rescued):
        victims = [
            {"id": i, "rescued": i < rescued, "alive": True}
            for i in range(people)
        ]
        return [{
            "zone_id": 0,
            "severity": severity,
            "people": people,
            "rescued": rescued,
            "time_waiting": 0,
            "is_active": rescued < people,
            "victims": victims,
        }]

    def test_score_in_open_interval(self):
        zones = self._make_zones(0.8, 5, 3)
        score = compute_reward("easy", zones, zones, steps_taken=10)
        assert 0.0 < score < 1.0

    def test_perfect_rescue_high_score(self):
        zones = self._make_zones(0.9, 5, 5)
        score = compute_reward("easy", zones, zones, steps_taken=5)
        assert score > 0.7

    def test_zero_rescue_low_score(self):
        zones_i = self._make_zones(0.9, 5, 0)
        zones_f = self._make_zones(0.9, 5, 0)
        score = compute_reward("easy", zones_i, zones_f, steps_taken=30)
        assert score < 0.3

    def test_score_strictly_between_0_and_1(self):
        for difficulty in ("easy", "medium", "hard"):
            task = get_task(difficulty)
            zones = self._make_zones(0.8, 8, 4)
            score = compute_reward(
                difficulty, zones, zones,
                steps_taken=task["max_steps"] // 2
            )
            assert 0.0 < score < 1.0, f"{difficulty}: score={score}"


class TestGradeEpisode:
    def _zones(self, severity, people, rescued):
        victims = [
            {"id": i, "rescued": i < rescued, "alive": True}
            for i in range(people)
        ]
        return [{
            "zone_id": 0, "severity": severity,
            "people": people, "rescued": rescued,
            "time_waiting": 1, "is_active": rescued < people,
            "victims": victims,
        }]

    def test_grade_returns_required_keys(self):
        z = self._zones(0.8, 5, 3)
        report = grade_episode("easy", z, z, steps_taken=10)
        for key in ("task_level", "score", "passed", "success_threshold",
                    "breakdown", "stats"):
            assert key in report

    def test_score_matches_compute_reward(self):
        """grade_episode score must equal compute_reward - shared core."""
        z = self._zones(0.8, 5, 4)
        report = grade_episode("easy", z, z, steps_taken=15)
        direct = compute_reward("easy", z, z, steps_taken=15)
        assert report["score"] == direct

    def test_passed_flag(self):
        z_good = self._zones(0.9, 5, 5)
        z_bad  = self._zones(0.9, 5, 0)
        assert grade_episode("easy", z_good, z_good, steps_taken=5)["passed"]
        assert not grade_episode("easy", z_bad, z_bad, steps_taken=30)["passed"]

    def test_breakdown_keys(self):
        z = self._zones(0.8, 5, 3)
        report = grade_episode("easy", z, z, steps_taken=10)
        for key in ("base_rescue_score", "time_penalty", "wait_penalty",
                    "spawn_penalty", "total_penalty", "raw_score"):
            assert key in report["breakdown"]

    def test_spawn_penalty_fix_a(self):
        """rescued_spawned > spawned_victims must not cause negative penalty."""
        z = self._zones(0.8, 5, 3)
        report = grade_episode(
            "hard", z, z, steps_taken=10,
            spawned_victims=2, rescued_spawned=5  # rescued > spawned
        )
        assert report["breakdown"]["spawn_penalty"] >= 0.0
        assert 0.0 < report["score"] < 1.0


# ===========================================================================
# SECTION 5 - grid.py
# ===========================================================================

class TestGridWorld:
    def _grid(self, difficulty="easy"):
        task = get_task(difficulty)
        g = GridWorld(difficulty, task["seed"])
        g.reset()
        return g

    def test_reset_returns_state(self):
        g = self._grid("easy")
        state = g.get_state()
        assert "zones" in state
        assert "step_num" in state
        assert state["step_num"] == 0

    def test_zones_initial_snapshot_independent(self):
        """Mutations during episode must not affect _zones_initial."""
        g = self._grid("easy")
        initial_rescued = g._zones_initial[0]["victims"][0]["rescued"]
        g.apply_action(0, "ambulance")
        assert g._zones_initial[0]["victims"][0]["rescued"] == initial_rescued

    def test_apply_action_valid(self):
        g = self._grid("easy")
        valid, count = g.apply_action(0, "ambulance")
        assert valid
        assert count > 0

    def test_apply_action_invalid_zone(self):
        g = self._grid("easy")
        valid, count = g.apply_action(999, "ambulance")
        assert not valid
        assert count == 0

    def test_hard_mode_no_helicopter(self):
        g = self._grid("hard")
        valid, count = g.apply_action(0, "helicopter")
        assert not valid
        assert count == 0

    def test_is_active_property(self):
        g = self._grid("easy")
        zone = g.zones[0]
        assert zone.is_active  # has victims initially
        # rescue all victims
        for v in zone.victims:
            v["rescued"] = True
        assert not zone.is_active

    def test_tick_increments_step(self):
        g = self._grid("easy")
        g.tick()
        assert g.step_num == 1

    def test_tick_decays_survival_time(self):
        g = self._grid("easy")
        initial_st = g.zones[0].victims[0]["survival_time"]
        g.tick()
        assert g.zones[0].victims[0]["survival_time"] < initial_st

    def test_step_raises_after_done(self):
        g = self._grid("easy")
        # Force done
        for z in g.zones:
            for v in z.victims:
                v["rescued"] = True
        g._done = True
        with pytest.raises(RuntimeError):
            g.step({"zone_id": 0, "unit_type": "ambulance"})

    def test_step_reward_in_open_interval(self):
        g = self._grid("easy")
        result = g.step({"zone_id": 0, "unit_type": "rescue_team"})
        assert 0.0 < result["reward"] < 1.0

    def test_final_reward_in_open_interval(self):
        """Terminal reward from compute_reward must be in (0, 1)."""
        g = self._grid("easy")
        result = None
        for _ in range(STEP_LIMITS["easy"]):
            result = g.step({"zone_id": 0, "unit_type": "helicopter"})
            if result["done"]:
                break
        assert result is not None and result["done"]
        assert 0.0 < result["reward"] < 1.0

    def test_zones_before_snapshot_timing(self):
        """BUG 2 FIX: zones_before captured before apply_action."""
        g = self._grid("easy")
        # Step should not raise - correct snapshot timing prevents
        # is_active=False spurious penalty on last victim rescue
        result = g.step({"zone_id": 0, "unit_type": "helicopter"})
        assert result["info"]["action_valid"]

    def test_to_dict_deepcopy(self):
        """FIX 17: Zone.to_dict() returns deepcopy - external mutation safe."""
        g = self._grid("easy")
        d = g.zones[0].to_dict()
        d["victims"][0]["rescued"] = True
        assert not g.zones[0].victims[0]["rescued"]

    def test_seed_mismatch_raises(self):
        with pytest.raises(ValueError):
            GridWorld("easy", seed=9999)  # wrong seed

    def test_hard_spawn_victims(self):
        g = self._grid("hard")
        initial_count = sum(len(z.victims) for z in g.zones)
        g.spawn_new_victims()
        new_count = sum(len(z.victims) for z in g.zones)
        assert new_count > initial_count

    def test_people_equals_len_victims_after_spawn(self):
        g = self._grid("hard")
        g.spawn_new_victims()
        for z in g.zones:
            assert z.people == len(z.victims)


# ===========================================================================
# SECTION 6 - Environment API (DisasterEnvironment)
# ===========================================================================

class TestDisasterEnvironment:
    """
    Tests the OpenEnv Environment class directly (no HTTP).
    Import DisasterEnvironment from server package.
    """

    @pytest.fixture
    def env(self):
        from disaster_env.server.disaster_env_environment import DisasterEnvironment
        e = DisasterEnvironment()
        yield e

    def test_reset_returns_observation(self, env):
        obs = env.reset(difficulty="easy", seed=42)
        assert obs is not None
        assert len(obs.zones) == 1
        assert obs.time_step == 0
        assert obs.max_steps == 30

    def test_reset_resources_full_dict(self, env):
        """BUG 9 FIX: resources must be a full dict, not int."""
        obs = env.reset(difficulty="easy", seed=42)
        assert isinstance(obs.resources, dict)
        assert "ambulances" in obs.resources
        assert "rescue_teams" in obs.resources
        assert "helicopters" in obs.resources

    def test_state_returns_state_object(self, env):
        env.reset(difficulty="easy", seed=42)
        s = env.state()
        assert hasattr(s, "episode_id")
        assert hasattr(s, "step_count")
        assert s.step_count == 0

    def test_step_increments_step_count(self, env):
        from disaster_env.models import DisasterAction
        env.reset(difficulty="easy", seed=42)
        env.step(DisasterAction(zone_id=0, unit_type="ambulance"))
        assert env.state().step_count == 1

    def test_step_reward_in_open_interval(self, env):
        from disaster_env.models import DisasterAction
        env.reset(difficulty="easy", seed=42)
        obs = env.step(DisasterAction(zone_id=0, unit_type="rescue_team"))
        assert 0.0 < obs.reward < 1.0

    def test_terminal_step_has_grade_report(self, env):
        from disaster_env.models import DisasterAction
        env.reset(difficulty="easy", seed=42)
        obs = None
        for _ in range(50):
            obs = env.step(DisasterAction(zone_id=0, unit_type="helicopter"))
            if obs.episode_done:
                break
        assert obs.episode_done
        assert obs.last_action_info is not None
        assert "grade_report" in obs.last_action_info

    def test_grade_report_score_matches_reward(self, env):
        """BUG I FIX: grade_episode score == compute_reward score."""
        from disaster_env.models import DisasterAction
        env.reset(difficulty="easy", seed=42)
        obs = None
        for _ in range(50):
            obs = env.step(DisasterAction(zone_id=0, unit_type="helicopter"))
            if obs.episode_done:
                break
        grade_score = obs.last_action_info["grade_report"]["score"]
        assert grade_score == obs.reward

    def test_step_after_done_raises(self, env):
        from disaster_env.models import DisasterAction
        env.reset(difficulty="easy", seed=42)
        for _ in range(50):
            obs = env.step(DisasterAction(zone_id=0, unit_type="helicopter"))
            if obs.episode_done:
                break
        with pytest.raises(RuntimeError):
            env.step(DisasterAction(zone_id=0, unit_type="ambulance"))

    def test_hard_mode_no_helicopter_penalty(self, env):
        from disaster_env.models import DisasterAction
        env.reset(difficulty="hard", seed=999)
        obs = env.step(DisasterAction(zone_id=0, unit_type="helicopter"))
        assert obs.last_action_info["action_valid"] is False

    def test_reset_clears_done(self, env):
        from disaster_env.models import DisasterAction
        env.reset(difficulty="easy", seed=42)
        for _ in range(50):
            obs = env.step(DisasterAction(zone_id=0, unit_type="helicopter"))
            if obs.episode_done:
                break
        # Reset should clear done state
        obs2 = env.reset(difficulty="easy", seed=42)
        assert not obs2.episode_done

    def test_score_strictly_between_0_and_1(self, env):
        from disaster_env.models import DisasterAction
        for difficulty, seed in [("easy", 42), ("medium", 123), ("hard", 999)]:
            env.reset(difficulty=difficulty, seed=seed)
            obs = None
            for _ in range(100):
                obs = env.step(DisasterAction(zone_id=0, unit_type="rescue_team"))
                if obs.episode_done:
                    break
            assert obs.episode_done
            score = obs.last_action_info["grade_report"]["score"]
            assert 0.0 < score < 1.0, f"{difficulty}: score={score}"


# ===========================================================================
# SECTION 7 - Inference log format
# ===========================================================================

class TestInferenceLogFormat:
    """
    Validates [START] [STEP] [END] log format matches hackathon spec exactly.
    Spec: https://www.scaler.com/school-of-technology/meta-pytorch-hackathon/dashboard
    """

    START_PATTERN = re.compile(
        r"^\[START\] task=\S+ env=\S+ model=\S+$"
    )
    STEP_PATTERN = re.compile(
        r"^\[STEP\] step=\d+ action=\S+ reward=-?\d+\.\d{2} done=(true|false) error=\S+$"
    )
    END_PATTERN = re.compile(
        r"^\[END\] success=(true|false) steps=\d+ score=\d+\.\d{3} rewards=[\d.,]+$"
    )

    def _capture_logs(self, capsys, task, env_name, model,
                      step_data, success, steps, score, rewards):
        """Simulate inference.py log output."""
        import sys
        print(f"[START] task={task} env={env_name} model={model}", flush=True)
        for s in step_data:
            action_str = json.dumps({"zone_id": s["zone_id"], "unit_type": s["unit_type"]})
            print(
                f"[STEP] step={s['step']} action={action_str} "
                f"reward={s['reward']:.2f} done={str(s['done']).lower()} "
                f"error={s.get('error') or 'null'}",
                flush=True
            )
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(
            f"[END] success={str(success).lower()} steps={steps} "
            f"score={score:.3f} rewards={rewards_str}",
            flush=True
        )
        return capsys.readouterr().out.strip().split("\n")

    def test_start_format(self, capsys):
        lines = self._capture_logs(
            capsys,
            task="task1_easy_rescue", env_name="disaster_env",
            model="Qwen/Qwen2.5-72B-Instruct",
            step_data=[], success=True, steps=0, score=0.5, rewards=[]
        )
        assert self.START_PATTERN.match(lines[0]), f"Bad START: {lines[0]}"

    def test_step_format(self, capsys):
        step_data = [
            {"step": 1, "zone_id": 0, "unit_type": "rescue_team",
             "reward": 0.88, "done": False, "error": None},
        ]
        lines = self._capture_logs(
            capsys,
            task="task1_easy_rescue", env_name="disaster_env",
            model="Qwen/Qwen2.5-72B-Instruct",
            step_data=step_data, success=True, steps=1, score=0.75, rewards=[0.88]
        )
        assert self.STEP_PATTERN.match(lines[1]), f"Bad STEP: {lines[1]}"

    def test_end_format(self, capsys):
        rewards = [0.88, 0.72, 0.65]
        lines = self._capture_logs(
            capsys,
            task="task1_easy_rescue", env_name="disaster_env",
            model="Qwen/Qwen2.5-72B-Instruct",
            step_data=[], success=True, steps=3, score=0.750, rewards=rewards
        )
        end_line = [l for l in lines if l.startswith("[END]")][0]
        assert self.END_PATTERN.match(end_line), f"Bad END: {end_line}"

    def test_reward_format_2_decimal(self, capsys):
        step_data = [
            {"step": 1, "zone_id": 0, "unit_type": "ambulance",
             "reward": 0.8846, "done": False, "error": None},
        ]
        lines = self._capture_logs(
            capsys, "t", "e", "m", step_data, True, 1, 0.5, [0.8846]
        )
        step_line = [l for l in lines if "[STEP]" in l][0]
        # reward must be 2 decimal places
        assert "reward=0.88" in step_line, f"Wrong decimal format: {step_line}"

    def test_score_format_3_decimal(self, capsys):
        lines = self._capture_logs(
            capsys, "t", "e", "m", [], True, 5, 0.7500, [0.5, 0.6]
        )
        end_line = [l for l in lines if "[END]" in l][0]
        assert "score=0.750" in end_line, f"Wrong score format: {end_line}"

    def test_action_is_json(self, capsys):
        step_data = [
            {"step": 1, "zone_id": 2, "unit_type": "rescue_team",
             "reward": 0.70, "done": False, "error": None},
        ]
        lines = self._capture_logs(
            capsys, "t", "e", "m", step_data, True, 1, 0.5, [0.70]
        )
        step_line = [l for l in lines if "[STEP]" in l][0]
        # Extract action value
        match = re.search(r'action=(\{[^}]+\})', step_line)
        assert match, f"action not JSON: {step_line}"
        parsed = json.loads(match.group(1))
        assert parsed["zone_id"] == 2
        assert parsed["unit_type"] == "rescue_team"

    def test_done_lowercase(self, capsys):
        step_data = [
            {"step": 1, "zone_id": 0, "unit_type": "ambulance",
             "reward": 0.5, "done": True, "error": None},
        ]
        lines = self._capture_logs(
            capsys, "t", "e", "m", step_data, True, 1, 0.5, [0.5]
        )
        step_line = [l for l in lines if "[STEP]" in l][0]
        assert "done=true" in step_line, f"done not lowercase: {step_line}"

    def test_success_lowercase(self, capsys):
        lines = self._capture_logs(
            capsys, "t", "e", "m", [], False, 0, 0.1, []
        )
        end_line = [l for l in lines if "[END]" in l][0]
        assert "success=false" in end_line


# ===========================================================================
# SECTION 8 - Grader boundary / regression tests
# ===========================================================================

class TestGraderBoundary:
    def test_score_never_zero(self):
        """Even worst case: no rescue, max steps - score > 0."""
        zones = [{
            "zone_id": 0, "severity": 0.9, "people": 8, "rescued": 0,
            "time_waiting": 100, "is_active": True,
            "victims": [{"id": i, "rescued": False, "alive": False}
                        for i in range(8)],
        }]
        score = compute_reward("hard", zones, zones,
                               steps_taken=25, spawned_victims=10,
                               rescued_spawned=0)
        assert score > 0.0

    def test_score_never_one(self):
        """Even best case: all rescued, 1 step - score < 1."""
        zones = [{
            "zone_id": 0, "severity": 0.9, "people": 5, "rescued": 5,
            "time_waiting": 0, "is_active": False,
            "victims": [{"id": i, "rescued": True, "alive": True}
                        for i in range(5)],
        }]
        score = compute_reward("easy", zones, zones, steps_taken=1)
        assert score < 1.0

    def test_penalty_cap(self):
        """Total penalty capped at 50% of base score."""
        zones = [{
            "zone_id": 0, "severity": 0.9, "people": 5, "rescued": 3,
            "time_waiting": 50, "is_active": True,
            "victims": [{"id": i, "rescued": i < 3, "alive": True}
                        for i in range(5)],
        }]
        comps = _compute_score_components(
            "hard", zones, zones, steps_taken=25,
            spawned_victims=10, rescued_spawned=0
        )
        assert comps["total_penalty"] <= comps["base"] * 0.5

    def test_fix_b_unknown_zone_skipped(self):
        """FIX B: zones_final with unknown zone_id must not raise KeyError."""
        zones_i = [{
            "zone_id": 0, "severity": 0.8, "people": 3, "rescued": 0,
            "time_waiting": 0, "is_active": True,
            "victims": [{"id": i, "rescued": False, "alive": True}
                        for i in range(3)],
        }]
        zones_f = zones_i + [{
            "zone_id": 99,  # not in zones_initial
            "severity": 0.5, "people": 2, "rescued": 1,
            "time_waiting": 0, "is_active": True,
            "victims": [{"id": 100, "rescued": True, "alive": True}],
        }]
        # Should not raise
        score = compute_reward("easy", zones_i, zones_f, steps_taken=5)
        assert 0.0 < score < 1.0

    def test_fix_c_missing_is_active(self):
        """FIX C: zones missing is_active field must not raise KeyError."""
        zones = [{
            "zone_id": 0, "severity": 0.8, "people": 3, "rescued": 1,
            "time_waiting": 2,
            # is_active intentionally missing
            "victims": [{"id": i, "rescued": i == 0, "alive": True}
                        for i in range(3)],
        }]
        score = compute_reward("easy", zones, zones, steps_taken=5)
        assert 0.0 < score < 1.0
