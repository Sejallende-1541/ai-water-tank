"""
anomaly_detection.py
---------------------
Anomaly Detection Agent.

Takes the output of the Analysis Agent (monitoring.AnalysisResult) plus a
short window of recent SensorReadings and detects abnormal conditions:

    - Overflow risk
    - Low water level
    - Possible leakage
    - Unusual / high consumption
    - Rapid water-level drop
    - Sensor abnormalities

Each detected anomaly is returned as an `Anomaly` object carrying a
human-readable reason, so the Decision Agent (and the UI) can explain
*why* a condition was flagged - not just *that* it was flagged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from monitoring import AnalysisResult
from simulation import SensorReading

# --- Tunable thresholds -----------------------------------------------------
OVERFLOW_THRESHOLD_PERCENT = 90.0
LOW_WATER_THRESHOLD_PERCENT = 20.0
CRITICAL_LOW_WATER_PERCENT = 10.0
RAPID_DROP_PERCENT_PER_STEP = 3.0          # % drop between two consecutive readings
HIGH_CONSUMPTION_MULTIPLIER = 2.0          # vs. rolling average
HIGH_CONSUMPTION_ABSOLUTE_LPH = 500.0      # flagged regardless of baseline above this rate
LEAKAGE_UNEXPLAINED_RATIO = 1.4            # outflow markedly exceeds consumption


@dataclass
class Anomaly:
    code: str
    severity: str          # "info" | "warning" | "critical"
    title: str
    reason: str


@dataclass
class AnomalyReport:
    anomalies: List[Anomaly] = field(default_factory=list)

    @property
    def has_critical(self) -> bool:
        return any(a.severity == "critical" for a in self.anomalies)

    @property
    def has_warning(self) -> bool:
        return any(a.severity == "warning" for a in self.anomalies)

    @property
    def is_normal(self) -> bool:
        return len(self.anomalies) == 0


class AnomalyDetectionAgent:
    def detect(self, analysis: AnalysisResult, latest_reading: SensorReading,
               history_warnings: List[str]) -> AnomalyReport:
        report = AnomalyReport()

        # --- Sensor abnormalities first: these can make other numbers
        # unreliable, so flag them clearly. --------------------------------
        for warning in history_warnings:
            report.anomalies.append(Anomaly(
                code="sensor_fault",
                severity="critical",
                title="Sensor Abnormality",
                reason=warning,
            ))

        # --- Overflow risk ---------------------------------------------------
        if analysis.level_percent >= OVERFLOW_THRESHOLD_PERCENT:
            report.anomalies.append(Anomaly(
                code="overflow_risk",
                severity="critical",
                title="Overflow Risk",
                reason=(
                    f"Water level is {analysis.level_percent:.1f}%, at or above the "
                    f"{OVERFLOW_THRESHOLD_PERCENT:.0f}% overflow threshold, with an inflow "
                    f"rate of {analysis.inflow_lph:.0f} L/h."
                ),
            ))

        # --- Low water level ---------------------------------------------------
        if analysis.level_percent <= CRITICAL_LOW_WATER_PERCENT:
            report.anomalies.append(Anomaly(
                code="critical_low_water",
                severity="critical",
                title="Critical Low Water Level",
                reason=(
                    f"Water level has fallen to {analysis.level_percent:.1f}%, "
                    f"at or below the critical {CRITICAL_LOW_WATER_PERCENT:.0f}% mark."
                ),
            ))
        elif analysis.level_percent <= LOW_WATER_THRESHOLD_PERCENT:
            report.anomalies.append(Anomaly(
                code="low_water",
                severity="warning",
                title="Low Water Level",
                reason=(
                    f"Water level is {analysis.level_percent:.1f}%, below the recommended "
                    f"{LOW_WATER_THRESHOLD_PERCENT:.0f}% threshold."
                ),
            ))

        # --- Rapid drop / possible leakage -------------------------------------
        rate = analysis.level_change_rate_per_min
        if rate is not None and rate < 0:
            drop_per_step = abs(analysis.level_change_percent or 0.0)
            outflow = max(latest_reading.outflow_lph, 0.0001)
            consumption = max(latest_reading.consumption_lph, 0.0)
            unexplained_ratio = outflow / max(consumption, 0.0001)

            if drop_per_step >= RAPID_DROP_PERCENT_PER_STEP:
                if unexplained_ratio >= LEAKAGE_UNEXPLAINED_RATIO:
                    report.anomalies.append(Anomaly(
                        code="leakage",
                        severity="critical",
                        title="Possible Leakage Detected",
                        reason=(
                            f"Water level dropped by {drop_per_step:.1f}% in the last interval "
                            f"({rate:.2f} %/min). Recorded outflow ({outflow:.0f} L/h) is much "
                            f"higher than recorded consumption ({consumption:.0f} L/h), so the "
                            f"drop is not fully explained by normal usage."
                        ),
                    ))
                else:
                    report.anomalies.append(Anomaly(
                        code="rapid_drop",
                        severity="warning",
                        title="Rapid Water Level Drop",
                        reason=(
                            f"Water level dropped by {drop_per_step:.1f}% in the last interval "
                            f"({rate:.2f} %/min), faster than usual."
                        ),
                    ))

        # --- Unusual / high consumption ----------------------------------------
        avg = max(analysis.avg_consumption_lph, 0.0001)
        consumption = latest_reading.consumption_lph
        if consumption >= HIGH_CONSUMPTION_ABSOLUTE_LPH:
            report.anomalies.append(Anomaly(
                code="high_consumption",
                severity="warning",
                title="Unusual / High Consumption",
                reason=(
                    f"Current consumption is {consumption:.0f} L/h, above the "
                    f"{HIGH_CONSUMPTION_ABSOLUTE_LPH:.0f} L/h high-usage threshold."
                ),
            ))
        elif consumption >= avg * HIGH_CONSUMPTION_MULTIPLIER and avg > 0:
            report.anomalies.append(Anomaly(
                code="high_consumption",
                severity="warning",
                title="Unusual / High Consumption",
                reason=(
                    f"Current consumption is {consumption:.0f} L/h, "
                    f"more than {HIGH_CONSUMPTION_MULTIPLIER:.1f}x the rolling average of "
                    f"{avg:.0f} L/h."
                ),
            ))

        return report
