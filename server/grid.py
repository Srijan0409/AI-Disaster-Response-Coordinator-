import random
from constants import UTTARAKHAND_ZONES, SEVERITY_RANGES, PEOPLE_RANGES


class Zone:
    """
    Represents a single disaster-affected zone in Uttarakhand.
    Each zone has a real geographic location, disaster type,
    severity level, and number of people trapped.
    """

    def __init__(self, zone_id, name, district, disaster_type, severity, people):
        self.zone_id       = zone_id
        self.name          = name
        self.district      = district
        self.disaster_type = disaster_type
        self.severity      = severity
        self.people        = people
        self.rescued       = 0
        self.time_waiting  = 0

    def to_dict(self):
        """Returns zone data as a dictionary for API responses."""
        return {
            "zone_id":       self.zone_id,
            "name":          self.name,
            "district":      self.district,
            "disaster_type": self.disaster_type,
            "severity":      round(self.severity, 2),
            "people":        self.people,
            "rescued":       self.rescued,
            "time_waiting":  self.time_waiting,
            "is_active":     self.people > self.rescued
        }


class GridWorld:
    """
    Simulates a multi-zone disaster scenario across Uttarakhand.
    Supports three difficulty levels: easy, medium, and hard.
    Uses a fixed random seed to ensure reproducible scenarios.

    Severity ranges match generators.py exactly (via constants.py):
        easy   → 0.3 to 0.7
        medium → 0.4 to 0.9
        hard   → 0.6 to 1.0  (Kedarnath always starts at 1.0)

    Hard mode additionally spawns new victims every 5 steps
    to simulate a worsening disaster situation.
    """

    def __init__(self, task_level, seed):
        self.task_level = task_level   # difficulty level: easy / medium / hard
        self.seed       = seed         # fixed seed for reproducibility
        self.zones      = []
        self.step_num   = 0
        self.rng        = random.Random(seed + 1)  # separate rng for mid-episode spawns

    def reset(self):
        """
        Resets the environment to its initial state.
        Uses the fixed seed so the same scenario is generated every time.
        Severity range and people range come from constants.py —
        same values as generators.py so both produce identical output.
        Returns the initial state as a dictionary.
        """
        random.seed(self.seed)
        self.rng      = random.Random(self.seed + 1)
        self.zones    = []
        self.step_num = 0

        zone_templates   = UTTARAKHAND_ZONES[self.task_level]
        sev_min, sev_max = SEVERITY_RANGES[self.task_level]
        ppl_min, ppl_max = PEOPLE_RANGES[self.task_level]

        for i, template in enumerate(zone_templates):

            # Hard mode: Kedarnath always starts at maximum severity (1.0)
            # Matches generators.py generate_civilians() exactly
            if self.task_level == "hard" and i == 0:
                severity = 1.0
            else:
                severity = round(random.uniform(sev_min, sev_max), 2)

            zone = Zone(
                zone_id       = i,
                name          = template["name"],
                district      = template["district"],
                disaster_type = template["disaster_type"],
                severity      = severity,
                people        = random.randint(ppl_min, ppl_max)
            )
            self.zones.append(zone)

        return self.get_state()

    def spread_threat(self):
        """
        Simulates disaster spreading to adjacent zones.
        If a zone has severity above 0.6, the next zone's severity
        increases by 0.1 (capped at 1.0).
        Example: Kedarnath flood spreads toward Badrinath and Rishikesh.
        """
        for i in range(len(self.zones) - 1):
            if self.zones[i].severity > 0.6:
                self.zones[i + 1].severity = min(
                    1.0,
                    self.zones[i + 1].severity + 0.1
                )
                self.zones[i + 1].time_waiting += 1

    def spawn_new_victims(self):
        """
        Spawns additional trapped people in random active zones.
        Only triggered in hard mode every 5 steps.
        Simulates the disaster worsening mid-episode —
        the AI cannot assume the initial victim count stays fixed.
        """
        active_zones = [z for z in self.zones if z.people > z.rescued]
        if not active_zones:
            return

        target_zone         = self.rng.choice(active_zones)
        new_victims         = self.rng.randint(2, 5)
        target_zone.people += new_victims

    def tick(self, spread=False, spread_interval=5, spawn_victims=False):
        """
        Advances the simulation by one time step.
        - Increments the step counter.
        - Increases waiting time for all unrescued zones.
        - Triggers threat spread if enabled and interval is reached.
        - Spawns new victims in hard mode every 5 steps.

        Args:
            spread          : whether disaster spreads to adjacent zones
            spread_interval : how many steps between each spread event
            spawn_victims   : whether new victims appear mid-episode (hard mode only)
        """
        self.step_num += 1

        for zone in self.zones:
            if zone.people > zone.rescued:
                zone.time_waiting += 1

        if spread and self.step_num % spread_interval == 0:
            self.spread_threat()

        if spawn_victims and self.step_num % 5 == 0:
            self.spawn_new_victims()

    def apply_action(self, zone_id, unit_type):
        """
        Applies a rescue action to a zone.

        Looks up the zone by zone_id, then adds rescued people based
        on the unit type. Rescued count is capped at total people.

        Rescue amounts per unit type:
            ambulance    → 2 people
            rescue_team  → 3 people
            helicopter   → 5 people

        Args:
            zone_id   : int — which zone to send the unit to
            unit_type : str — "ambulance", "rescue_team", or "helicopter"

        Returns:
            (action_valid, rescued_count)
            action_valid  : False if zone not found or already fully rescued
            rescued_count : how many additional people were rescued (0 if invalid)

        Called by: inference.py inside step() — before tick().
        """
        RESCUE_AMOUNTS = {
            "ambulance":   2,
            "rescue_team": 3,
            "helicopter":  5,
        }

        zone = next((z for z in self.zones if z.zone_id == zone_id), None)

        # Invalid: zone does not exist
        if zone is None:
            return False, 0

        # Invalid: zone already fully rescued — wasted action
        if not zone.people > zone.rescued:
            return False, 0

        # Unknown unit type — treat as invalid
        if unit_type not in RESCUE_AMOUNTS:
            return False, 0

        amount       = RESCUE_AMOUNTS[unit_type]
        before       = zone.rescued
        zone.rescued = min(zone.people, zone.rescued + amount)
        actual       = zone.rescued - before

        return True, actual

    def get_state(self):
        """Returns the current state of all zones as a dictionary."""
        return {
            "task_level": self.task_level,
            "step_num":   self.step_num,
            "zones":      [z.to_dict() for z in self.zones]
        }