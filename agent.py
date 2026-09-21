"""
agent.py
--------
Decision Agent + Notification/Action Agent, wired together into a single
orchestrating `WaterTankAgent` that implements the full Agentic AI pipeline:

    Input/Sensor Data -> Monitoring Agent -> Data Analysis
        -> Anomaly Detection Agent -> Decision Agent
        -> Notification/Action Agent -> Monitoring History

The Decision Agent is deliberately rule-based so the project works fully
offline / without any paid API. It is structured so that a real LLM could
later be dropped in to replace or augment `_decide()` (see README).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

from anomaly_detection import Anomaly, AnomalyDetectionAgent, AnomalyReport
from monitoring import AnalysisAgent, AnalysisResult, MonitoringAgent
from simulation import SensorReading, TankSimulator


@dataclass
class AgentDecision:
    status_code: str          # "normal" | "warning" | "critical"
    status_label: str         # e.g. "NORMAL", "LEAKAGE DETECTED"
    icon: str                 # emoji indicator
    headline: str
    reasoning: List[str]
    recommended_actions: List[str]
    anomalies: List[Anomaly] = field(default_factory=list)


@dataclass
class AgentCycleResult:
    reading: SensorReading
    analysis: AnalysisResult
    anomaly_report: AnomalyReport
    decision: AgentDecision
    sensor_warnings: List[str]


class DecisionAgent:
    """Turns an AnomalyReport + AnalysisResult into a clear decision with
    an explanation the user can trust."""

    def decide(self, analysis: AnalysisResult, anomaly_report: AnomalyReport) -> AgentDecision:
        if anomaly_report.has_critical:
            return self._decide_critical(analysis, anomaly_report)
        if anomaly_report.has_warning:
            return self._decide_warning(analysis, anomaly_report)
        return self._decide_normal(analysis)

    # ------------------------------------------------------------------ tiers
    def _decide_critical(self, analysis: AnalysisResult, report: AnomalyReport) -> AgentDecision:
        critical = [a for a in report.anomalies if a.severity == "critical"]
        primary = critical[0]
        actions = []

        if primary.code == "overflow_risk":
            actions = ["Stop the pump immediately.", "Close the inlet valve if available.",
                       "Monitor level until it drops below 80%."]
        elif primary.code == "critical_low_water":
            actions = ["Start the pump to refill the tank.", "Check the water source supply.",
                       "Restrict non-essential usage until level recovers."]
        elif primary.code == "leakage":
            actions = ["Inspect the tank, outlet pipes and valves for leakage.",
                       "Temporarily stop the pump to limit water loss.",
                       "Schedule a maintenance check."]
        elif primary.code == "sensor_fault":
            actions = ["Verify sensor wiring and connections.",
                       "Cross-check readings with a manual measurement.",
                       "Do not fully trust automated decisions until resolved."]
        else:
            actions = ["Investigate the flagged condition immediately."]

        reasoning = [a.reason for a in critical]
        return AgentDecision(
            status_code="critical",
            status_label=primary.title.upper(),
            icon="🔴",
            headline=primary.title,
            reasoning=reasoning,
            recommended_actions=actions,
            anomalies=report.anomalies,
        )

    def _decide_warning(self, analysis: AnalysisResult, report: AnomalyReport) -> AgentDecision:
        warnings = [a for a in report.anomalies if a.severity == "warning"]
        primary = warnings[0]
        actions = []

        if primary.code == "low_water":
            actions = ["Consider starting the pump soon.", "Monitor consumption over the next hour."]
        elif primary.code == "rapid_drop":
            actions = ["Keep monitoring - level is dropping faster than usual.",
                       "Check for open taps or valves left running."]
        elif primary.code == "high_consumption":
            actions = ["Reduce water usage during this period if possible.",
                       "Verify no equipment is left running unnecessarily."]
        else:
            actions = ["Keep monitoring the tank closely."]

        reasoning = [a.reason for a in warnings]
        return AgentDecision(
            status_code="warning",
            status_label=primary.title.upper(),
            icon="🟡",
            headline=primary.title,
            reasoning=reasoning,
            recommended_actions=actions,
            anomalies=report.anomalies,
        )

    def _decide_normal(self, analysis: AnalysisResult) -> AgentDecision:
        reasoning = [
            f"Water level is {analysis.level_percent:.1f}%, within the normal operating range.",
            f"Consumption ({analysis.consumption_lph:.0f} L/h) is close to the rolling average "
            f"({analysis.avg_consumption_lph:.0f} L/h).",
        ]
        return AgentDecision(
            status_code="normal",
            status_label="NORMAL",
            icon="🟢",
            headline="System Operating Normally",
            reasoning=reasoning,
            recommended_actions=["No action required. Continue routine monitoring."],
            anomalies=[],
        )


class WaterTankAgent:
    """
    High-level Agentic AI orchestrator. Owns a TankSimulator (the sensor
    input source), a MonitoringAgent (ingestion + history), an
    AnalysisAgent, an AnomalyDetectionAgent and a DecisionAgent, and runs
    the full pipeline on each `run_cycle()` call.
    """

    def __init__(self, capacity: float = 5000.0, initial_level: float = 55.0,
                 scenario: str = "normal", history_size: int = 500, seed: Optional[int] = None):
        self.simulator = TankSimulator(capacity, initial_level, scenario, seed=seed)
        self.monitor = MonitoringAgent(history_size=history_size)
        self.analyzer = AnalysisAgent()
        self.detector = AnomalyDetectionAgent()
        self.decider = DecisionAgent()
        self.status = "ACTIVE"
        self.alert_log: List[str] = []
        self.decision_labels: Deque[str] = deque(maxlen=history_size)

    # ---------------------------------------------------------------- config
    def configure(self, capacity: Optional[float] = None, level_percent: Optional[float] = None,
                  inflow: Optional[float] = None, outflow: Optional[float] = None,
                  consumption: Optional[float] = None, pump_status: Optional[str] = None,
                  scenario: Optional[str] = None) -> None:
        if capacity is not None:
            self.simulator.set_capacity(capacity)
        if level_percent is not None:
            self.simulator.set_level_percent(level_percent)
        if inflow is not None or outflow is not None or consumption is not None:
            self.simulator.set_flow(
                inflow if inflow is not None else self.simulator.inflow_lph,
                outflow if outflow is not None else self.simulator.outflow_lph,
                consumption if consumption is not None else self.simulator.consumption_lph,
            )
        if pump_status is not None:
            self.simulator.set_pump(pump_status)
        if scenario is not None:
            self.simulator.set_scenario(scenario)

    def reset(self) -> None:
        self.monitor.reset()
        self.alert_log.clear()
        self.decision_labels.clear()

    def history_with_decisions(self):
        """Returns (history_list, decision_labels_list) aligned by index."""
        return self.monitor.as_list(), list(self.decision_labels)

    # ------------------------------------------------------------------ run
    def run_cycle(self, minutes_per_step: float = 15.0) -> AgentCycleResult:
        """Executes one full pass of the Agentic AI pipeline and returns
        a structured result the UI can render."""
        reading = self.simulator.step(minutes_per_step=minutes_per_step)
        sensor_warnings = self.monitor.ingest(reading)

        history = self.monitor.as_list()
        analysis = self.analyzer.analyze(history, minutes_per_step=minutes_per_step)
        anomaly_report = self.detector.detect(analysis, reading, sensor_warnings)
        decision = self.decider.decide(analysis, anomaly_report)

        log_line = f"[{reading.timestamp}] {decision.icon} {decision.status_label} - {decision.headline}"
        self.alert_log.append(log_line)
        if len(self.alert_log) > 200:
            self.alert_log = self.alert_log[-200:]
        self.decision_labels.append(f"{decision.icon} {decision.status_label}")

        return AgentCycleResult(
            reading=reading,
            analysis=analysis,
            anomaly_report=anomaly_report,
            decision=decision,
            sensor_warnings=sensor_warnings,
        )
