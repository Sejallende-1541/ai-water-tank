"""
simulation.py
--------------
Simulated sensor system for the AI Water Tank Monitoring project.

Since real IoT hardware is not available, this module generates realistic
sensor readings for a water tank under different operating scenarios:

    - normal            : steady filling / consumption
    - low_water         : tank level drops to a critically low value
    - overflow_risk     : tank level climbs towards 90-100%
    - leakage           : level drops faster than outflow/consumption explain
    - high_consumption  : consumption suddenly spikes
    - sensor_fault      : sensor produces inconsistent / noisy readings

The simulator is intentionally simple (rule + noise based) so the project
works without any external hardware or paid API, while still producing
data that is interesting enough for the Agentic AI pipeline to reason about.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional


SCENARIOS = [
    "normal",
    "low_water",
    "overflow_risk",
    "leakage",
    "high_consumption",
    "sensor_fault",
]


@dataclass
class SensorReading:
    """A single snapshot of tank sensor data."""

    timestamp: str
    tank_capacity_litres: float
    water_level_percent: float
    water_level_litres: float
    inflow_lph: float          # litres / hour flowing IN
    outflow_lph: float         # litres / hour flowing OUT (taps/usage)
    consumption_lph: float     # litres / hour actually consumed by users
    pump_status: str           # "ON" / "OFF"
    leakage_flag: bool         # ground-truth flag set by the simulator (for demo only)
    scenario: str

    def as_dict(self) -> dict:
        return asdict(self)


class TankSimulator:
    """
    Stateful simulator that produces one new SensorReading each time
    `step()` is called, evolving smoothly from the previous reading
    so that graphs and the AI agent see a believable time series.
    """

    def __init__(
        self,
        tank_capacity_litres: float = 5000.0,
        initial_level_percent: float = 55.0,
        scenario: str = "normal",
        seed: Optional[int] = None,
    ):
        self.rng = random.Random(seed)
        self.tank_capacity_litres = self._safe_positive(tank_capacity_litres, 5000.0)
        self.level_percent = self._clamp(initial_level_percent, 0.0, 100.0)
        self.scenario = scenario if scenario in SCENARIOS else "normal"
        self.pump_status = "OFF"
        self.inflow_lph = 200.0
        self.outflow_lph = 150.0
        self.consumption_lph = 150.0
        self.step_count = 0

    # ---------------------------------------------------------------- utils
    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    @staticmethod
    def _safe_positive(value: float, fallback: float) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return fallback
        return value if value > 0 else fallback

    def set_scenario(self, scenario: str) -> None:
        self.scenario = scenario if scenario in SCENARIOS else "normal"

    def set_capacity(self, capacity: float) -> None:
        self.tank_capacity_litres = self._safe_positive(capacity, self.tank_capacity_litres)

    def set_level_percent(self, level: float) -> None:
        self.level_percent = self._clamp(float(level), 0.0, 100.0)

    def set_pump(self, status: str) -> None:
        self.pump_status = "ON" if str(status).upper() == "ON" else "OFF"

    def set_flow(self, inflow: float, outflow: float, consumption: float) -> None:
        self.inflow_lph = max(0.0, float(inflow))
        self.outflow_lph = max(0.0, float(outflow))
        self.consumption_lph = max(0.0, float(consumption))

    # ------------------------------------------------------------- stepping
    def step(self, minutes_per_step: float = 5.0) -> SensorReading:
        """
        Advance the simulation by `minutes_per_step` minutes and return a
        new SensorReading. The physical model is simplified:

            delta_litres = (inflow - outflow) * (minutes_per_step / 60)

        Scenario-specific behaviour perturbs inflow/outflow/consumption to
        create the situations the Anomaly Detection Agent should catch.
        """
        self.step_count += 1
        leakage_flag = False
        fraction_hour = minutes_per_step / 60.0

        inflow = self.inflow_lph
        outflow = self.outflow_lph
        consumption = self.consumption_lph

        if self.scenario == "normal":
            inflow = self._jitter(200.0, 20.0)
            outflow = self._jitter(150.0, 20.0)
            consumption = outflow
            self.pump_status = "ON" if self.level_percent < 60 else self.pump_status

        elif self.scenario == "low_water":
            inflow = self._jitter(120.0, 15.0)
            outflow = self._jitter(1250.0, 60.0)
            consumption = outflow
            self.pump_status = "OFF"

        elif self.scenario == "overflow_risk":
            inflow = self._jitter(1300.0, 60.0)
            outflow = self._jitter(100.0, 15.0)
            consumption = outflow
            self.pump_status = "ON"

        elif self.scenario == "leakage":
            # Level drops noticeably faster than the recorded consumption
            # explains -> simulates water escaping through a crack/pipe.
            consumption = self._jitter(120.0, 10.0)
            outflow = consumption
            leak_rate = self._jitter(1000.0, 60.0)  # hidden extra loss
            inflow = self._jitter(60.0, 10.0)
            outflow = outflow + leak_rate
            leakage_flag = True
            self.pump_status = "OFF"

        elif self.scenario == "high_consumption":
            consumption = self._jitter(650.0, 40.0)
            outflow = consumption
            inflow = self._jitter(220.0, 20.0)
            self.pump_status = "ON"

        elif self.scenario == "sensor_fault":
            # Produce an inconsistent / noisy / occasionally invalid reading.
            inflow = self._jitter(200.0, 150.0)
            outflow = self._jitter(150.0, 150.0)
            consumption = self._jitter(150.0, 400.0)
            if self.rng.random() < 0.35:
                consumption = -abs(consumption)  # invalid negative reading
            if self.rng.random() < 0.20:
                inflow = self.inflow_lph * 50  # spike glitch

        delta_litres = (inflow - outflow) * fraction_hour
        current_litres = (self.level_percent / 100.0) * self.tank_capacity_litres
        new_litres = current_litres + delta_litres
        new_litres = self._clamp(new_litres, 0.0, self.tank_capacity_litres)
        self.level_percent = (new_litres / self.tank_capacity_litres) * 100.0 \
            if self.tank_capacity_litres > 0 else 0.0

        self.inflow_lph, self.outflow_lph, self.consumption_lph = inflow, outflow, consumption

        reading = SensorReading(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            tank_capacity_litres=round(self.tank_capacity_litres, 1),
            water_level_percent=round(self.level_percent, 2),
            water_level_litres=round(new_litres, 1),
            inflow_lph=round(inflow, 1),
            outflow_lph=round(outflow, 1),
            consumption_lph=round(consumption, 1),
            pump_status=self.pump_status,
            leakage_flag=leakage_flag,
            scenario=self.scenario,
        )
        return reading

    def _jitter(self, base: float, spread: float) -> float:
        return max(0.0, base + self.rng.uniform(-spread, spread))
