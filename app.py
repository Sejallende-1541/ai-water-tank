"""
app.py
------
AI Agent for Water Tank Monitoring System
An Agentic AI-Based Intelligent Water Management and Monitoring System

Run locally with:
    python app.py

This file wires together the Agentic AI pipeline (simulation -> monitoring
-> analysis -> anomaly detection -> decision -> notification/history) into
an attractive Gradio dashboard.
"""

from __future__ import annotations

import os
import tempfile
from typing import Tuple

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

from agent import WaterTankAgent
from monitoring import AnalysisResult
from report_generator import HISTORY_COLUMNS, build_history_rows, generate_summary_text, history_to_csv_string
from simulation import SCENARIOS, TankSimulator

APP_DIR = os.path.dirname(os.path.abspath(__file__))
CSS_PATH = os.path.join(APP_DIR, "assets", "style.css")


def load_css() -> str:
    try:
        with open(CSS_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


# ============================================================================
# Rendering helpers - turn agent/analysis data into HTML snippets
# ============================================================================

def metric_card(label: str, value: str, sub: str, css_class: str) -> str:
    return (
        f'<div class="wt-metric {css_class}">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f'<div class="sub">{sub}</div>'
        f'</div>'
    )


def status_css_and_icon(status_code: str) -> Tuple[str, str]:
    if status_code == "critical":
        return "status-critical", "🔴"
    if status_code == "warning":
        return "status-warning", "🟡"
    return "status-normal", "🟢"


def render_cards(level_percent, level_litres, capacity, consumption, avg_consumption,
                  inflow, outflow, pump_status, status_code, status_label):
    level_card = metric_card(
        "💧 Water Level", f"{level_percent:.1f}%",
        f"{level_litres:,.0f} L / {capacity:,.0f} L", "level",
    )
    consume_card = metric_card(
        "📊 Consumption", f"{consumption:,.0f} L/h",
        f"avg {avg_consumption:,.0f} L/h", "consume",
    )
    inflow_card = metric_card("🚰 Inflow Rate", f"{inflow:,.0f} L/h", "into the tank", "inflow")
    outflow_card = metric_card("🔽 Outflow Rate", f"{outflow:,.0f} L/h", "out of the tank", "outflow")
    pump_card = metric_card(
        "⚙️ Pump Status", pump_status,
        "Running" if pump_status == "ON" else "Idle", "pump",
    )
    status_class, icon = status_css_and_icon(status_code)
    status_card = metric_card("⚠️ System Status", f"{icon} {status_label}", "AI-evaluated", status_class)
    return level_card, consume_card, inflow_card, outflow_card, pump_card, status_card


def render_tank(level_percent: float, status_code: str) -> str:
    level_percent = max(0.0, min(100.0, float(level_percent)))
    if status_code == "critical":
        fill_color = "linear-gradient(180deg, #f28b8b 0%, #c22b2b 100%)"
    elif status_code == "warning":
        fill_color = "linear-gradient(180deg, #f5d47a 0%, #e08a1f 100%)"
    else:
        fill_color = "linear-gradient(180deg, #4fb8f0 0%, #0f6db5 100%)"
    return f"""
    <div class="wt-tank-wrap">
        <div class="wt-tank">
            <div class="wt-tank-fill" style="height:{level_percent:.1f}%; background:{fill_color};"></div>
            <div class="wt-tank-label">{level_percent:.0f}%</div>
        </div>
        <div style="color:#dceeff; font-size:0.85rem; margin-top:8px; font-weight:600;">
            Live Tank Level
        </div>
    </div>
    """


def render_consumption_panel(current, average, level_change_rate) -> str:
    daily_estimate = average * 24
    if level_change_rate is None:
        trend = "N/A - waiting for a second reading"
        trend_icon = "⏳"
    elif level_change_rate > 0.02:
        trend = "Rising (tank filling)"
        trend_icon = "📈"
    elif level_change_rate < -0.02:
        trend = "Falling (tank draining)"
        trend_icon = "📉"
    else:
        trend = "Stable"
        trend_icon = "➖"
    return f"""
    <div class="wt-card">
        <div class="wt-section-title" style="color:#0b3d63;">📊 Consumption Analysis</div>
        <table style="width:100%; font-size:0.95rem; color:#123a5e; border-collapse:collapse;">
            <tr><td style="padding:4px 0;">Current consumption</td><td style="text-align:right; font-weight:700;">{current:,.0f} L/h</td></tr>
            <tr><td style="padding:4px 0;">Average consumption</td><td style="text-align:right; font-weight:700;">{average:,.0f} L/h</td></tr>
            <tr><td style="padding:4px 0;">Estimated daily consumption</td><td style="text-align:right; font-weight:700;">{daily_estimate:,.0f} L/day</td></tr>
            <tr><td style="padding:4px 0;">Water level trend</td><td style="text-align:right; font-weight:700;">{trend_icon} {trend}</td></tr>
        </table>
    </div>
    """


def render_agent_panel(decision, analysis: AnalysisResult, evaluated: bool) -> str:
    if not evaluated or decision is None:
        return """
        <div class="wt-agent-panel">
            <div class="wt-section-title" style="color:#eaf6ff;">🤖 AI Agent Status
                <span class="agent-active">ACTIVE</span>
            </div>
            <p style="opacity:0.85;">No decision yet - click <b>Run AI Agent</b> or <b>Start Simulation</b>
            to let the agent analyze the current tank data.</p>
        </div>
        """
    reasoning_html = "".join(f"<li>{r}</li>" for r in decision.reasoning)
    actions_html = "".join(f"<li>{a}</li>" for a in decision.recommended_actions)
    return f"""
    <div class="wt-agent-panel">
        <div class="wt-section-title" style="color:#eaf6ff;">🤖 AI Agent Status
            <span class="agent-active">ACTIVE</span>
        </div>
        <p style="font-size:1.15rem; font-weight:800; margin:6px 0 10px 0;">
            {decision.icon} {decision.status_label}
        </p>
        <p style="margin:0 0 6px 0; opacity:0.9;"><b>Detected condition:</b> {decision.headline}</p>
        <p style="margin:0 0 4px 0; opacity:0.9;"><b>Reasoning:</b></p>
        <ul style="margin:0 0 10px 18px; opacity:0.9;">{reasoning_html}</ul>
        <p style="margin:0 0 4px 0; opacity:0.9;"><b>Recommended action(s):</b></p>
        <ul style="margin:0 0 0 18px; opacity:0.9;">{actions_html}</ul>
    </div>
    """


def render_alerts(decision, sensor_warnings) -> str:
    if decision is None:
        return '<div class="wt-alert recommend"><h4>🔵 Waiting</h4>Run the AI agent to generate alerts.</div>'

    css_class = {"critical": "critical", "warning": "warning", "normal": "normal"}[decision.status_code]
    icon = {"critical": "🔴", "warning": "🟡", "normal": "🟢"}[decision.status_code]
    reasoning_html = "".join(f"<li>{r}</li>" for r in decision.reasoning) or "<li>No abnormal conditions found.</li>"
    box = f"""
    <div class="wt-alert {css_class}">
        <h4>{icon} {decision.status_label}</h4>
        <div>{decision.headline}.</div>
        <ul>{reasoning_html}</ul>
    </div>
    """
    if decision.status_code == "normal":
        box += """
        <div class="wt-alert recommend">
            <h4>🔵 RECOMMENDATION</h4>
            <div>Continue routine monitoring. No action required right now.</div>
        </div>
        """
    if sensor_warnings:
        warn_html = "".join(f"<li>{w}</li>" for w in sensor_warnings)
        box += f"""
        <div class="wt-alert warning">
            <h4>🟡 SENSOR NOTICE</h4>
            <ul>{warn_html}</ul>
        </div>
        """
    return box


def render_graph(history) -> go.Figure:
    fig = go.Figure()
    if history:
        timestamps = [r.timestamp for r in history]
        levels = [r.water_level_percent for r in history]
        fig.add_trace(go.Scatter(
            x=timestamps, y=levels, mode="lines+markers", name="Water Level (%)",
            line=dict(color="#0f6db5", width=3), marker=dict(size=5, color="#17c3d6"),
        ))
        fig.add_hline(y=90, line_dash="dash", line_color="#e64545", annotation_text="Overflow (90%)")
        fig.add_hline(y=20, line_dash="dash", line_color="#f5a623", annotation_text="Low water (20%)")
    fig.update_layout(
        title="Water Level Over Time",
        xaxis_title="Time",
        yaxis_title="Water Level (%)",
        yaxis_range=[0, 100],
        template="plotly_white",
        margin=dict(l=40, r=20, t=50, b=40),
        height=360,
    )
    return fig


def render_history_df(agent: WaterTankAgent) -> pd.DataFrame:
    history, decisions = agent.history_with_decisions()
    rows = build_history_rows(history, decisions)
    rows = list(reversed(rows))  # most recent first
    if not rows:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    return pd.DataFrame(rows, columns=HISTORY_COLUMNS)


# ============================================================================
# Core callbacks
# ============================================================================

def full_dashboard_update(agent: WaterTankAgent, decision, analysis, sensor_warnings, evaluated: bool):
    """Builds every dashboard output component from the agent's current state."""
    status_code = decision.status_code if decision else "normal"
    status_label = decision.status_label if decision else "AWAITING DATA"

    cards = render_cards(
        analysis.level_percent, analysis.level_litres, analysis.capacity_litres,
        analysis.consumption_lph, analysis.avg_consumption_lph,
        analysis.inflow_lph, analysis.outflow_lph, analysis.pump_status,
        status_code, status_label,
    )
    tank_html = render_tank(analysis.level_percent, status_code)
    consumption_html = render_consumption_panel(
        analysis.consumption_lph, analysis.avg_consumption_lph, analysis.level_change_rate_per_min,
    )
    agent_panel_html = render_agent_panel(decision, analysis, evaluated)
    alerts_html = render_alerts(decision, sensor_warnings)
    graph = render_graph(agent.monitor.as_list())
    history_df = render_history_df(agent)

    return (*cards, tank_html, consumption_html, agent_panel_html, alerts_html, graph, history_df)


def snapshot_analysis_from_simulator(sim: TankSimulator) -> AnalysisResult:
    """Builds a lightweight AnalysisResult purely from the simulator's
    current manual configuration, used before the AI agent has evaluated
    any full cycle (e.g. right after 'Update Monitoring')."""
    litres = (sim.level_percent / 100.0) * sim.tank_capacity_litres
    return AnalysisResult(
        level_percent=sim.level_percent,
        level_litres=litres,
        capacity_litres=sim.tank_capacity_litres,
        inflow_lph=sim.inflow_lph,
        outflow_lph=sim.outflow_lph,
        consumption_lph=sim.consumption_lph,
        pump_status=sim.pump_status,
        level_change_percent=None,
        level_change_rate_per_min=None,
        avg_consumption_lph=sim.consumption_lph,
        max_level_percent=sim.level_percent,
        min_level_percent=sim.level_percent,
        readings_count=0,
    )


def on_load():
    agent = WaterTankAgent()
    analysis = snapshot_analysis_from_simulator(agent.simulator)
    outputs = full_dashboard_update(agent, None, analysis, [], evaluated=False)
    return (agent, *outputs)


def on_update_monitoring(capacity, level, inflow, outflow, consumption, pump, scenario, agent: WaterTankAgent):
    try:
        agent.configure(
            capacity=capacity, level_percent=level, inflow=inflow, outflow=outflow,
            consumption=consumption, pump_status=pump, scenario=scenario,
        )
        error_note = ""
    except (TypeError, ValueError) as exc:
        error_note = f"Invalid input ignored: {exc}"

    analysis = snapshot_analysis_from_simulator(agent.simulator)
    last_decision = None
    outputs = full_dashboard_update(agent, last_decision, analysis, [], evaluated=False)
    if error_note:
        gr.Warning(error_note)
    return (agent, *outputs)


def on_run_agent(agent: WaterTankAgent):
    try:
        result = agent.run_cycle()
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI, not a crash
        gr.Warning(f"Agent cycle failed: {exc}")
        analysis = snapshot_analysis_from_simulator(agent.simulator)
        outputs = full_dashboard_update(agent, None, analysis, [], evaluated=False)
        return (agent, *outputs)

    outputs = full_dashboard_update(
        agent, result.decision, result.analysis, result.sensor_warnings, evaluated=True,
    )
    return (agent, *outputs)


def on_reset(agent: WaterTankAgent):
    agent.reset()
    analysis = snapshot_analysis_from_simulator(agent.simulator)
    outputs = full_dashboard_update(agent, None, analysis, [], evaluated=False)
    gr.Info("Monitoring history has been reset.")
    return (agent, *outputs)


def on_generate_report(agent: WaterTankAgent):
    history, decisions = agent.history_with_decisions()
    summary = generate_summary_text(history, agent.alert_log)
    csv_str = history_to_csv_string(history, decisions)

    tmp_dir = tempfile.mkdtemp(prefix="water_tank_report_")
    csv_path = os.path.join(tmp_dir, "monitoring_history.csv")
    txt_path = os.path.join(tmp_dir, "monitoring_summary.txt")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        f.write(csv_str)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary)

    return summary, csv_path, txt_path


def toggle_timer_on():
    return gr.Timer(active=True)


def toggle_timer_off():
    return gr.Timer(active=False)


# ============================================================================
# Build the Gradio Blocks UI
# ============================================================================

with gr.Blocks(title="AI Water Tank Monitoring System", css=load_css(), theme=gr.themes.Soft()) as demo:
    agent_state = gr.State()

    gr.HTML(
        """
        <div id="wt-header">
            <h1>💧 AI Water Tank Monitoring System</h1>
            <p>Agentic AI for Intelligent Water Management</p>
        </div>
        """
    )

    with gr.Row():
        # --------------------------------------------------------- Sidebar
        with gr.Column(scale=1, min_width=280):
            gr.Markdown("### ⚙️ Tank & Sensor Controls")
            capacity_in = gr.Number(label="Tank Capacity (L)", value=5000, minimum=1)
            level_in = gr.Slider(label="Water Level (%)", minimum=0, maximum=100, value=55, step=1)
            inflow_in = gr.Number(label="Water Inflow (L/h)", value=200, minimum=0)
            outflow_in = gr.Number(label="Water Outflow (L/h)", value=150, minimum=0)
            consumption_in = gr.Number(label="Consumption (L/h)", value=150, minimum=0)
            pump_in = gr.Radio(label="Pump Status", choices=["ON", "OFF"], value="OFF")
            scenario_in = gr.Dropdown(
                label="Simulation Scenario", choices=SCENARIOS, value="normal",
                info="Choose a scenario, then Run AI Agent or Start Simulation.",
            )

            update_btn = gr.Button("🔄 Update Monitoring", variant="secondary")
            run_agent_btn = gr.Button("🤖 Run AI Agent", variant="primary")
            with gr.Row():
                start_btn = gr.Button("▶️ Start Simulation")
                stop_btn = gr.Button("⏹️ Stop Simulation")
            with gr.Row():
                report_btn = gr.Button("📊 Generate Report")
                reset_btn = gr.Button("🧹 Reset")

            gr.Markdown(
                "*Tip: pick a scenario like **leakage** or **overflow_risk**, then press "
                "**Run AI Agent** a few times (or Start Simulation) to watch the agent react.*"
            )

        # ----------------------------------------------------------- Main
        with gr.Column(scale=3):
            gr.HTML('<div class="wt-section-title" style="color:#eaf6ff;">📟 Real-Time Tank Monitoring</div>')
            with gr.Row():
                level_card = gr.HTML()
                consume_card = gr.HTML()
                inflow_card = gr.HTML()
            with gr.Row():
                outflow_card = gr.HTML()
                pump_card = gr.HTML()
                status_card = gr.HTML()

            with gr.Row():
                with gr.Column(scale=1):
                    tank_html = gr.HTML()
                    consumption_html = gr.HTML()
                with gr.Column(scale=2):
                    graph = gr.Plot(label="Water Level Graph")

            agent_panel_html = gr.HTML()

            gr.HTML('<div class="wt-section-title" style="color:#eaf6ff;">🚨 Alerts</div>')
            alerts_html = gr.HTML()

            gr.HTML('<div class="wt-section-title" style="color:#eaf6ff;">🕑 Monitoring History</div>')
            history_df = gr.Dataframe(headers=HISTORY_COLUMNS, interactive=False, wrap=True)

            gr.HTML('<div class="wt-section-title" style="color:#eaf6ff;">📄 Report</div>')
            report_text = gr.Textbox(label="Monitoring Summary", lines=10, interactive=False)
            with gr.Row():
                csv_file = gr.File(label="Download History (CSV)")
                txt_file = gr.File(label="Download Summary (TXT)")

    timer = gr.Timer(2, active=False)

    all_outputs = [
        agent_state, level_card, consume_card, inflow_card, outflow_card, pump_card, status_card,
        tank_html, consumption_html, agent_panel_html, alerts_html, graph, history_df,
    ]

    demo.load(on_load, inputs=None, outputs=all_outputs)

    update_btn.click(
        on_update_monitoring,
        inputs=[capacity_in, level_in, inflow_in, outflow_in, consumption_in, pump_in, scenario_in, agent_state],
        outputs=all_outputs,
    )
    run_agent_btn.click(on_run_agent, inputs=[agent_state], outputs=all_outputs)
    reset_btn.click(on_reset, inputs=[agent_state], outputs=all_outputs)

    start_btn.click(toggle_timer_on, None, timer)
    stop_btn.click(toggle_timer_off, None, timer)
    timer.tick(on_run_agent, inputs=[agent_state], outputs=all_outputs)

    report_btn.click(on_generate_report, inputs=[agent_state], outputs=[report_text, csv_file, txt_file])


if __name__ == "__main__":
    demo.queue().launch()
