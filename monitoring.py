"""
monitoring.py
--------------
Monitoring Agent  : receives raw sensor/input data, validates it, and keeps
                     a rolling history of readings.
Analysis Agent     : analyzes the history to compute trends such as filling
                     rate, emptying rate, average consumption and rate of
                     change of the water level.

These two agents form the first stages of the Agentic AI pipeline:

    Input/Sensor Data -> Monitoring Agent -> Data Analysis -> ...
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

from simulation import SensorReading


class SensorValidationError(Exception):
    """Raised when incoming sensor data is invalid."""


@dataclass
class AnalysisResult:
    """Output of the Analysis Agent for the latest reading."""

    level_percent: float
    level_litres: float
    capacity_litres: float
    inflow_lph: float
    outflow_lph: float
    consumption_lph: float
    pump_status: str

    level_change_percent: Optional[float]   # change since previous reading
    level_change_rate_per_min: Optional[float]  # %/minute
    avg_consumption_lph: float
    max_level_percent: float
    min_level_percent: float
    readings_count: int


class MonitoringAgent:
    """
    Ingests raw readings (from the simulator or, in a future version, from
    real IoT hardware) and maintains a bounded history buffer.
    """

    def __init__(self, history_size: int = 500):
        self.history: Deque[SensorReading] = deque(maxlen=history_size)

    def validate(self, reading: SensorReading) -> List[str]:
        """Return a list of validation warnings (does not raise, so the UI
        can show a friendly 'sensor abnormality' message instead of a
        crash)."""
        warnings: List[str] = []
        if reading.tank_capacity_litres <= 0:
            warnings.append("Tank capacity must be greater than zero.")
        if reading.water_level_percent < 0 or reading.water_level_percent > 100:
            warnings.append("Water level percentage is out of the valid 0-100 range.")
        if reading.inflow_lph < 0 or reading.outflow_lph < 0:
            warnings.append("Inflow/outflow readings cannot be negative.")
        if reading.consumption_lph < 0:
            warnings.append("Consumption reading is negative - possible sensor fault.")
        if reading.inflow_lph > 5000 or reading.outflow_lph > 5000:
            warnings.append("Inflow/outflow reading is unrealistically high - possible sensor glitch.")
        return warnings

    def ingest(self, reading: SensorReading) -> List[str]:
        """Add a (possibly imperfect) reading to history and return any
        validation warnings found."""
        warnings = self.validate(reading)
        self.history.append(reading)
        return warnings

    def latest(self) -> Optional[SensorReading]:
        return self.history[-1] if self.history else None

    def as_list(self) -> List[SensorReading]:
        return list(self.history)

    def reset(self) -> None:
        self.history.clear()


class AnalysisAgent:
    """
    Looks at the Monitoring Agent's history and computes derived metrics:
    filling/emptying rate, average consumption, and short-term trend.
    """

    def __init__(self, trend_window: int = 5):
        self.trend_window = trend_window

    def analyze(self, history: List[SensorReading], minutes_per_step: float = 5.0) -> AnalysisResult:
        if not history:
            raise ValueError("Cannot analyze an empty history.")

        latest = history[-1]
        levels = [r.water_level_percent for r in history]

        level_change_percent = None
        level_change_rate = None
        if len(history) >= 2:
            previous = history[-2]
            level_change_percent = round(latest.water_level_percent - previous.water_level_percent, 3)
            level_change_rate = round(level_change_percent / minutes_per_step, 4) if minutes_per_step else None

        # Baseline average uses recent PRIOR readings only (not the latest
        # one) so a sudden spike can actually be compared against "normal".
        baseline_window = history[-(self.trend_window + 1):-1] if len(history) > 1 else []
        baseline_consumptions = [r.consumption_lph for r in baseline_window if r.consumption_lph >= 0]
        if baseline_consumptions:
            avg_consumption = round(sum(baseline_consumptions) / len(baseline_consumptions), 1)
        elif latest.consumption_lph >= 0:
            avg_consumption = round(latest.consumption_lph, 1)
        else:
            avg_consumption = 0.0

        return AnalysisResult(
            level_percent=latest.water_level_percent,
            level_litres=latest.water_level_litres,
            capacity_litres=latest.tank_capacity_litres,
            inflow_lph=latest.inflow_lph,
            outflow_lph=latest.outflow_lph,
            consumption_lph=latest.consumption_lph,
            pump_status=latest.pump_status,
            level_change_percent=level_change_percent,
            level_change_rate_per_min=level_change_rate,
            avg_consumption_lph=avg_consumption,
            max_level_percent=round(max(levels), 2),
            min_level_percent=round(min(levels), 2),
            readings_count=len(history),
        )
