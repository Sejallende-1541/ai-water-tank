"""
report_generator.py
--------------------
Builds a monitoring summary report (as text) and can export the raw
monitoring history as CSV or the summary as TXT, ready for download
through the Gradio UI.
"""

from __future__ import annotations

import csv
import io
from datetime import datetime
from typing import List

from simulation import SensorReading

HISTORY_COLUMNS = [
    "Time", "Water Level (%)", "Water Level (L)", "Consumption (L/h)",
    "Inflow (L/h)", "Outflow (L/h)", "Pump", "Status", "AI Decision",
]


def build_history_rows(history: List[SensorReading], decisions: List[str]) -> List[list]:
    """Zips sensor history with the decision label recorded for each step."""
    rows = []
    for reading, decision_label in zip(history, decisions):
        rows.append([
            reading.timestamp,
            reading.water_level_percent,
            reading.water_level_litres,
            reading.consumption_lph,
            reading.inflow_lph,
            reading.outflow_lph,
            reading.pump_status,
            reading.scenario,
            decision_label,
        ])
    return rows


def generate_summary_text(history: List[SensorReading], alert_log: List[str],
                           minutes_per_step: float = 15.0) -> str:
    if not history:
        return "No monitoring data has been recorded yet. Run the AI agent or start a simulation first."

    levels = [r.water_level_percent for r in history]
    consumptions = [r.consumption_lph for r in history if r.consumption_lph >= 0]
    step_hours = minutes_per_step / 60.0
    total_consumption_litres = sum(c * step_hours for c in consumptions)

    warnings = sum(1 for line in alert_log if "🟡" in line)
    criticals = sum(1 for line in alert_log if "🔴" in line)
    leaks = sum(1 for line in alert_log if "LEAKAGE" in line.upper())

    lines = [
        "=" * 60,
        "  AI WATER TANK MONITORING - SUMMARY REPORT",
        "=" * 60,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Readings analyzed: {len(history)}",
        "",
        "-- Water Level --",
        f"  Average level : {sum(levels) / len(levels):.1f} %",
        f"  Maximum level : {max(levels):.1f} %",
        f"  Minimum level : {min(levels):.1f} %",
        "",
        "-- Consumption --",
        f"  Estimated total consumption : {total_consumption_litres:.0f} L",
        f"  Average consumption rate    : {(sum(consumptions) / len(consumptions)) if consumptions else 0:.0f} L/h",
        "",
        "-- AI Agent Activity --",
        f"  Warnings raised   : {warnings}",
        f"  Critical alerts   : {criticals}",
        f"  Leakage events    : {leaks}",
        "",
        "-- Recent Decisions --",
    ]
    lines.extend(alert_log[-10:] if alert_log else ["  (none recorded)"])
    lines.append("=" * 60)
    return "\n".join(lines)


def history_to_csv_string(history: List[SensorReading], decisions: List[str]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(HISTORY_COLUMNS)
    for row in build_history_rows(history, decisions):
        writer.writerow(row)
    return buffer.getvalue()
