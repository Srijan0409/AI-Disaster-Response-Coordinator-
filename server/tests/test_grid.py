from grid import GridWorld
from generators import generate_civilians
from constants import SEVERITY_RANGES, PEOPLE_RANGES


# ---------------------------------------------------------------------------
# Group 1 — Zone count per difficulty (3 tests)
# ---------------------------------------------------------------------------

def test_easy_has_one_zone():
    """Easy task must contain exactly 1 disaster zone."""
    g = GridWorld(task_level="easy", seed=42)
    assert len(g.reset()["zones"]) == 1


def test_medium_has_three_zones():
    """Medium task must contain exactly 3 disaster zones."""
    g = GridWorld(task_level="medium", seed=123)
    assert len(g.reset()["zones"]) == 3


def test_hard_has_five_zones():
    """Hard task must contain exactly 5 disaster zones."""
    g = GridWorld(task_level="hard", seed=999)
    assert len(g.reset()["zones"]) == 5


# ---------------------------------------------------------------------------
# Group 2 — Reproducibility (2 tests)
# ---------------------------------------------------------------------------

def test_same_seed_same_map():
    """Same seed must always generate the exact same disaster scenario."""
    g1 = GridWorld(task_level="medium", seed=42)
    g2 = GridWorld(task_level="medium", seed=42)
    assert g1.reset() == g2.reset(), "Same seed did not produce the same map!"


def test_different_seeds_differ():
    """Different seeds must produce different scenarios."""
    g1 = GridWorld(task_level="medium", seed=42)
    g2 = GridWorld(task_level="medium", seed=99)
    assert g1.reset() != g2.reset(), "Different seeds produced identical maps!"


# ---------------------------------------------------------------------------
# Group 3 — Place names (2 tests)
# ---------------------------------------------------------------------------

def test_place_names_correct():
    """Easy task must always start at Kedarnath Temple Area, Rudraprayag."""
    g     = GridWorld(task_level="easy", seed=42)
    state = g.reset()
    assert state["zones"][0]["name"]     == "Kedarnath Temple Area"
    assert state["zones"][0]["district"] == "Rudraprayag"


def test_hard_kedarnath_always_critical():
    """Hard mode: Kedarnath must always start at severity 1.0."""
    g     = GridWorld(task_level="hard", seed=999)
    state = g.reset()
    assert state["zones"][0]["severity"] == 1.0, \
        "Kedarnath must be 1.0 severity in hard mode!"


# ---------------------------------------------------------------------------
# Group 4 — Severity ranges from constants.py (4 tests)
# ---------------------------------------------------------------------------

def test_easy_severity_in_range():
    """Easy severity must be within (0.3, 0.7) as defined in constants.py."""
    sev_min, sev_max = SEVERITY_RANGES["easy"]
    g = GridWorld(task_level="easy", seed=42)
    for zone in g.reset()["zones"]:
        assert sev_min <= zone["severity"] <= sev_max, \
            f"Easy severity {zone['severity']} out of range ({sev_min}, {sev_max})"


def test_medium_severity_in_range():
    """Medium severity must be within (0.4, 0.9) as defined in constants.py."""
    sev_min, sev_max = SEVERITY_RANGES["medium"]
    g = GridWorld(task_level="medium", seed=123)
    for zone in g.reset()["zones"]:
        assert sev_min <= zone["severity"] <= sev_max, \
            f"Medium severity {zone['severity']} out of range ({sev_min}, {sev_max})"


def test_hard_severity_in_range():
    """Hard severity must be within (0.6, 1.0) as defined in constants.py."""
    sev_min, sev_max = SEVERITY_RANGES["hard"]
    g = GridWorld(task_level="hard", seed=999)
    for zone in g.reset()["zones"]:
        assert sev_min <= zone["severity"] <= sev_max, \
            f"Hard severity {zone['severity']} out of range ({sev_min}, {sev_max})"


def test_hard_other_zones_not_fixed():
    """Hard mode: only Kedarnath is fixed at 1.0 — other zones must be random."""
    g     = GridWorld(task_level="hard", seed=999)
    state = g.reset()
    for zone in state["zones"][1:]:
        assert zone["severity"] != 1.0, \
            f"{zone['name']} should not be fixed at 1.0!"


# ---------------------------------------------------------------------------
# Group 5 — People ranges from constants.py (1 test)
# ---------------------------------------------------------------------------

def test_people_matches_constants_range():
    """People count must stay within the range defined in constants.py."""
    for level in ["easy", "medium", "hard"]:
        ppl_min, ppl_max = PEOPLE_RANGES[level]
        for zone in GridWorld(task_level=level, seed=42).reset()["zones"]:
            assert ppl_min <= zone["people"] <= ppl_max, \
                f"{level} {zone['name']} people {zone['people']} out of range!"


# ---------------------------------------------------------------------------
# Group 6 — grid.py matches generators.py (1 test)
# ---------------------------------------------------------------------------

def test_grid_and_generators_same_output():
    """
    GridWorld.reset() and generate_civilians() must produce identical
    severity and people values for the same seed — both use constants.py.
    """
    for level, seed in [("easy", 42), ("medium", 123), ("hard", 999)]:
        grid_zones = GridWorld(task_level=level, seed=seed).reset()["zones"]
        gen_zones  = generate_civilians(level, seed)
        for gz, cz in zip(grid_zones, gen_zones):
            assert gz["severity"] == cz["severity"], \
                f"{level} {gz['name']} severity mismatch: grid={gz['severity']} gen={cz['severity']}"
            assert gz["people"] == cz["people"], \
                f"{level} {gz['name']} people mismatch: grid={gz['people']} gen={cz['people']}"


# ---------------------------------------------------------------------------
# Group 7 — Disaster types (1 test)
# ---------------------------------------------------------------------------

def test_disaster_types_present():
    """Every zone must have a valid disaster type assigned."""
    valid_types = {
        "flash_flood", "landslide", "river_overflow",
        "land_subsidence", "glacier_burst"
    }
    state = GridWorld(task_level="hard", seed=999).reset()
    for zone in state["zones"]:
        assert "disaster_type" in zone
        assert zone["disaster_type"] in valid_types, \
            f"{zone['name']} has unrecognised disaster type!"


# ---------------------------------------------------------------------------
# Group 8 — Spread logic (3 tests)
# ---------------------------------------------------------------------------

def test_spread_increases_severity():
    """A high-severity zone must spread disaster to the adjacent zone."""
    g = GridWorld(task_level="medium", seed=42)
    g.reset()
    g.zones[0].severity = 0.9
    before = g.zones[1].severity
    g.spread_threat()
    assert g.zones[1].severity > before, "Threat spread is not working!"


def test_spread_capped_at_one():
    """Severity must never exceed 1.0 even after multiple spreads."""
    g = GridWorld(task_level="medium", seed=42)
    g.reset()
    g.zones[0].severity = 1.0
    g.zones[1].severity = 0.95
    g.spread_threat()
    assert g.zones[1].severity <= 1.0


def test_spread_only_from_high_severity():
    """A low-severity zone must NOT spread to the next zone."""
    g = GridWorld(task_level="medium", seed=42)
    g.reset()
    g.zones[0].severity = 0.5      # below 0.6 threshold — must not spread
    before = g.zones[1].severity
    g.spread_threat()
    assert g.zones[1].severity == before, \
        "Low severity zone should not spread to adjacent zone!"


# ---------------------------------------------------------------------------
# Group 9 — Tick logic (2 tests)
# ---------------------------------------------------------------------------

def test_waiting_time_increases():
    """Waiting time must increase by 1 each step for unrescued zones."""
    g = GridWorld(task_level="easy", seed=42)
    g.reset()
    g.tick(spread=False)
    assert g.zones[0].time_waiting == 1


def test_waiting_stops_when_rescued():
    """Waiting time must not increase for fully rescued zones."""
    g = GridWorld(task_level="easy", seed=42)
    g.reset()
    g.zones[0].rescued = g.zones[0].people   # mark as fully rescued
    g.tick(spread=False)
    assert g.zones[0].time_waiting == 0, \
        "Waiting time must not increase for fully rescued zones!"


# ---------------------------------------------------------------------------
# Group 10 — Spawn logic (2 tests)
# ---------------------------------------------------------------------------

def test_spawn_new_victims_hard_mode():
    """In hard mode, new victims must appear every 5 steps."""
    g = GridWorld(task_level="hard", seed=999)
    g.reset()
    people_before = sum(z.people for z in g.zones)
    for _ in range(5):
        g.tick(spread=True, spread_interval=2, spawn_victims=True)
    assert sum(z.people for z in g.zones) > people_before


def test_no_spawn_in_easy_mode():
    """Easy mode must never spawn new victims mid-episode."""
    g = GridWorld(task_level="easy", seed=42)
    g.reset()
    people_before = sum(z.people for z in g.zones)
    for _ in range(10):
        g.tick(spread=False, spawn_victims=False)
    assert sum(z.people for z in g.zones) == people_before


# ---------------------------------------------------------------------------
# Group 11 — Reset (1 test)
# ---------------------------------------------------------------------------

def test_reset_clears_state():
    """After reset, step number and waiting times must go back to zero."""
    g = GridWorld(task_level="medium", seed=42)
    g.reset()
    for _ in range(5):
        g.tick(spread=True, spread_interval=5)
    state = g.reset()
    assert state["step_num"] == 0
    for zone in state["zones"]:
        assert zone["time_waiting"] == 0, \
            f"{zone['name']} waiting time must be 0 after reset!"


# ---------------------------------------------------------------------------
# Group 12 — apply_action (6 tests)
# ---------------------------------------------------------------------------

def test_apply_action_valid_ambulance():
    """Valid ambulance action must return True and rescue 2 people."""
    g = GridWorld(task_level="easy", seed=42)
    g.reset()
    g.zones[0].people  = 10
    g.zones[0].rescued = 0
    valid, count = g.apply_action(zone_id=0, unit_type="ambulance")
    assert valid is True
    assert count == 2
    assert g.zones[0].rescued == 2


def test_apply_action_valid_rescue_team():
    """Valid rescue_team action must return True and rescue 3 people."""
    g = GridWorld(task_level="easy", seed=42)
    g.reset()
    g.zones[0].people  = 10
    g.zones[0].rescued = 0
    valid, count = g.apply_action(zone_id=0, unit_type="rescue_team")
    assert valid is True
    assert count == 3
    assert g.zones[0].rescued == 3


def test_apply_action_valid_helicopter():
    """Valid helicopter action must return True and rescue 5 people."""
    g = GridWorld(task_level="easy", seed=42)
    g.reset()
    g.zones[0].people  = 10
    g.zones[0].rescued = 0
    valid, count = g.apply_action(zone_id=0, unit_type="helicopter")
    assert valid is True
    assert count == 5
    assert g.zones[0].rescued == 5


def test_apply_action_zone_not_found():
    """Invalid zone_id must return (False, 0) — no rescue happens."""
    g = GridWorld(task_level="easy", seed=42)
    g.reset()
    valid, count = g.apply_action(zone_id=99, unit_type="ambulance")
    assert valid is False
    assert count == 0


def test_apply_action_already_rescued():
    """Fully rescued zone must return (False, 0) — wasted action."""
    g = GridWorld(task_level="easy", seed=42)
    g.reset()
    g.zones[0].rescued = g.zones[0].people   # mark fully rescued
    valid, count = g.apply_action(zone_id=0, unit_type="ambulance")
    assert valid is False
    assert count == 0


def test_apply_action_unknown_unit_type():
    """Unknown unit type must return (False, 0)."""
    g = GridWorld(task_level="easy", seed=42)
    g.reset()
    g.zones[0].people  = 10
    g.zones[0].rescued = 0
    valid, count = g.apply_action(zone_id=0, unit_type="tank")
    assert valid is False
    assert count == 0


def test_apply_action_cap_at_people():
    """Rescued count must never exceed total people in the zone."""
    g = GridWorld(task_level="easy", seed=42)
    g.reset()
    g.zones[0].people  = 3
    g.zones[0].rescued = 2   # only 1 left
    valid, count = g.apply_action(zone_id=0, unit_type="helicopter")  # would rescue 5
    assert valid is True
    assert count == 1                      # only 1 actually rescued
    assert g.zones[0].rescued == 3        # capped at people


# To run all tests:
# python -m pytest tests/test_grid.py -v