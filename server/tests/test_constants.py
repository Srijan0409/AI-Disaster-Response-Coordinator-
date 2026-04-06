from constants import (
    UTTARAKHAND_ZONES,
    SEVERITY_RANGES,
    PEOPLE_RANGES,
    RESOURCE_CONFIG,
    STEP_LIMITS,
)


# ---------------------------------------------------------------------------
# Group 1 — UTTARAKHAND_ZONES (5 tests)
# ---------------------------------------------------------------------------

def test_zones_all_levels_present():
    """All three difficulty levels must be defined."""
    for level in ["easy", "medium", "hard"]:
        assert level in UTTARAKHAND_ZONES, f"Missing level: {level}"


def test_zones_easy_has_one():
    """Easy must have exactly 1 zone."""
    assert len(UTTARAKHAND_ZONES["easy"]) == 1


def test_zones_medium_has_three():
    """Medium must have exactly 3 zones."""
    assert len(UTTARAKHAND_ZONES["medium"]) == 3


def test_zones_hard_has_five():
    """Hard must have exactly 5 zones."""
    assert len(UTTARAKHAND_ZONES["hard"]) == 5


def test_zones_required_keys():
    """Every zone must have name, district, disaster_type."""
    for level in ["easy", "medium", "hard"]:
        for z in UTTARAKHAND_ZONES[level]:
            for key in ["name", "district", "disaster_type"]:
                assert key in z, f"{level} zone missing key: {key}"


def test_zones_valid_disaster_types():
    """Every zone must have a recognised disaster type."""
    valid = {"flash_flood", "landslide", "river_overflow",
             "land_subsidence", "glacier_burst"}
    for level in ["easy", "medium", "hard"]:
        for z in UTTARAKHAND_ZONES[level]:
            assert z["disaster_type"] in valid, \
                f"{z['name']} has invalid disaster_type: {z['disaster_type']}"


def test_zones_first_zone_is_kedarnath():
    """First zone in every level must be Kedarnath Temple Area."""
    for level in ["easy", "medium", "hard"]:
        assert UTTARAKHAND_ZONES[level][0]["name"] == "Kedarnath Temple Area"
        assert UTTARAKHAND_ZONES[level][0]["district"] == "Rudraprayag"


# ---------------------------------------------------------------------------
# Group 2 — SEVERITY_RANGES (4 tests)
# ---------------------------------------------------------------------------

def test_severity_ranges_all_levels():
    """All three levels must have severity ranges defined."""
    for level in ["easy", "medium", "hard"]:
        assert level in SEVERITY_RANGES


def test_severity_ranges_valid():
    """Every severity range must be (min, max) with 0 <= min < max <= 1."""
    for level, (sev_min, sev_max) in SEVERITY_RANGES.items():
        assert 0.0 <= sev_min < sev_max <= 1.0, \
            f"{level} severity range invalid: ({sev_min}, {sev_max})"


def test_severity_ranges_increase_with_difficulty():
    """Hard min severity must be >= medium min, medium >= easy min."""
    assert SEVERITY_RANGES["hard"][0] >= SEVERITY_RANGES["medium"][0] >= SEVERITY_RANGES["easy"][0]


def test_severity_hard_max_is_one():
    """Hard severity max must be 1.0 — Kedarnath starts at critical."""
    assert SEVERITY_RANGES["hard"][1] == 1.0


# ---------------------------------------------------------------------------
# Group 3 — PEOPLE_RANGES (3 tests)
# ---------------------------------------------------------------------------

def test_people_ranges_all_levels():
    """All three levels must have people ranges defined."""
    for level in ["easy", "medium", "hard"]:
        assert level in PEOPLE_RANGES


def test_people_ranges_valid():
    """Every people range must be (min, max) with min >= 1."""
    for level, (ppl_min, ppl_max) in PEOPLE_RANGES.items():
        assert ppl_min >= 1, f"{level} people min must be >= 1"
        assert ppl_max > ppl_min, f"{level} people max must be > min"


def test_people_ranges_increase_with_difficulty():
    """Hard must have more people than easy — higher pressure."""
    assert PEOPLE_RANGES["hard"][1] > PEOPLE_RANGES["easy"][1]


# ---------------------------------------------------------------------------
# Group 4 — RESOURCE_CONFIG (4 tests)
# ---------------------------------------------------------------------------

def test_resource_config_all_levels():
    """All three levels must have resource configs."""
    for level in ["easy", "medium", "hard"]:
        assert level in RESOURCE_CONFIG


def test_resource_config_required_keys():
    """Every resource config must have ambulances, rescue_teams, helicopters."""
    for level in ["easy", "medium", "hard"]:
        for key in ["ambulances", "rescue_teams", "helicopters"]:
            assert key in RESOURCE_CONFIG[level], \
                f"{level} resource config missing: {key}"


def test_resource_hard_no_helicopters():
    """Hard mode must have 0 helicopters."""
    assert RESOURCE_CONFIG["hard"]["helicopters"] == 0


def test_resource_hard_scarcer_than_medium():
    """Hard total resources must be <= medium total resources."""
    hard_total   = sum(RESOURCE_CONFIG["hard"].values())
    medium_total = sum(RESOURCE_CONFIG["medium"].values())
    assert hard_total <= medium_total


# ---------------------------------------------------------------------------
# Group 5 — STEP_LIMITS (3 tests)
# ---------------------------------------------------------------------------

def test_step_limits_all_levels():
    """All three levels must have step limits defined."""
    for level in ["easy", "medium", "hard"]:
        assert level in STEP_LIMITS


def test_step_limits_positive():
    """Every step limit must be a positive integer."""
    for level, limit in STEP_LIMITS.items():
        assert isinstance(limit, int) and limit > 0, \
            f"{level} step limit must be positive int, got {limit}"


def test_step_limits_hard_less_than_medium():
    """Hard must have fewer steps than medium — more time pressure."""
    assert STEP_LIMITS["hard"] < STEP_LIMITS["medium"]