"""v0.4.6 dashboard: local Bode stability analysis for Q4."""
from __future__ import annotations

import html
import numpy as np
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Div, Select, Span
from bokeh.plotting import figure

from . import v045_final_dashboard as v045
from .stability_analysis import (
    analyze_all_loops,
    format_poles,
    phase_margin_grade,
)


def _finite(value: float) -> str:
    return f"{value:.2f}" if np.isfinite(value) else "n/a"


def _summary_table(results: dict) -> str:
    rows = []
    for result in results.values():
        grade = phase_margin_grade(
            result.phase_margin_deg, result.stable
        )
        state = "Stable" if result.stable else "Unstable"
        rows.append(
            "<tr>"
            f"<td>{html.escape(result.definition.label)}</td>"
            f"<td>{state}</td>"
            f"<td>{result.phase_margin_deg:.1f}°</td>"
            f"<td>{_finite(result.gain_crossover_rad_s)}</td>"
            f"<td>{_finite(result.closed_loop_bandwidth_rad_s)}</td>"
            f"<td>{result.closed_loop_peak_db:.2f} dB</td>"
            f"<td>{grade}</td>"
            "</tr>"
        )
    return (
        "<table style='border-collapse:collapse;width:100%;font-size:12px'>"
        "<thead><tr>"
        "<th style='text-align:left;padding:4px'>Loop</th>"
        "<th style='text-align:left;padding:4px'>Closed-loop stability</th>"
        "<th style='text-align:left;padding:4px'>PM</th>"
        "<th style='text-align:left;padding:4px'>ωgc [rad/s]</th>"
        "<th style='text-align:left;padding:4px'>BW [rad/s]</th>"
        "<th style='text-align:left;padding:4px'>Peak T</th>"
        "<th style='text-align:left;padding:4px'>Interpretation</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _metric_html(result) -> str:
    grade = phase_margin_grade(
        result.phase_margin_deg, result.stable
    )
    poles = format_poles(result.closed_loop_poles)
    gm_note = ""
    if np.isfinite(result.gain_margin_db):
        gm_note = (
            f" Classical gain-margin crossing: {result.gain_margin_db:.1f} dB."
            " For multi-integrator PID-equivalent loops this can represent a"
            " lower-gain stability boundary, so PM and closed-loop poles are"
            " used as the primary robustness indicators here."
        )
    return (
        "<div style='white-space:normal;overflow-wrap:anywhere'>"
        f"<b>{html.escape(result.definition.label)}</b> — "
        f"{html.escape(result.definition.operating_point)}<br>"
        f"<b>Small-signal result:</b> "
        f"{'stable' if result.stable else 'UNSTABLE'}; "
        f"phase margin = <b>{result.phase_margin_deg:.1f}°</b> ({grade}), "
        f"gain crossover = <b>{_finite(result.gain_crossover_rad_s)} rad/s</b>, "
        f"closed-loop bandwidth = <b>{_finite(result.closed_loop_bandwidth_rad_s)} rad/s</b>, "
        f"closed-loop resonant peak = <b>{result.closed_loop_peak_db:.2f} dB</b>."
        f"{gm_note}<br>"
        f"<b>Closed-loop poles:</b> {html.escape(poles)}<br>"
        f"<b>Model:</b> {html.escape(result.definition.derivation)}"
        "</div>"
    )


def build_dashboard(doc, simulation=None):
    """Build the current dashboard and append the Q4 Bode-analysis section."""
    sim = v045.build_dashboard(doc, simulation=simulation)
    results = analyze_all_loops()
    definitions = {
        key: result.definition for key, result in results.items()
    }

    default_key = "hover_altitude"
    default = results[default_key]
    source = ColumnDataSource(data={
        "omega": default.omega_rad_s,
        "open_mag": default.open_mag_db,
        "closed_mag": default.closed_mag_db,
        "open_phase": default.open_phase_deg,
        "closed_phase": default.closed_phase_deg,
    })

    title = Div(
        text=(
            "<h2 style='margin:8px 0 2px 0'>Q4 — Local Bode stability analysis</h2>"
            "<div style='white-space:normal;overflow-wrap:anywhere'>"
            "Bode analysis is applied only at fixed operating points where a "
            "small-signal LTI approximation is meaningful. The transition "
            "phase is gain-scheduled/time-varying and remains verified by the "
            "nonlinear mission and gust regressions above.</div>"
        ),
        sizing_mode="stretch_width",
        styles={"white-space": "normal", "overflow-wrap": "anywhere"},
    )

    selector = Select(
        title="Local loop",
        value=default_key,
        options=[(key, definition.label) for key, definition in definitions.items()],
        width=300,
    )

    magnitude = figure(
        title="Bode magnitude",
        x_axis_type="log",
        height=310,
        sizing_mode="stretch_width",
        x_axis_label="Angular frequency [rad/s]",
        y_axis_label="Magnitude [dB]",
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    magnitude.line(
        "omega", "open_mag", source=source,
        line_width=2.2, line_dash="solid", legend_label="Open loop L(s)",
    )
    magnitude.line(
        "omega", "closed_mag", source=source,
        line_width=2.0, line_dash="dashed", legend_label="Closed loop T(s)",
    )
    zero_db = Span(
        location=0.0, dimension="width", line_dash="dotted", line_width=1.4
    )
    magnitude.add_layout(zero_db)
    gain_cross = Span(
        location=float(default.gain_crossover_rad_s),
        dimension="height", line_dash="dotdash", line_width=1.4,
    )
    magnitude.add_layout(gain_cross)
    magnitude.legend.location = "bottom_left"
    magnitude.legend.click_policy = "mute"

    phase = figure(
        title="Bode phase",
        x_axis_type="log",
        height=310,
        sizing_mode="stretch_width",
        x_axis_label="Angular frequency [rad/s]",
        y_axis_label="Phase [deg]",
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    phase.line(
        "omega", "open_phase", source=source,
        line_width=2.2, line_dash="solid", legend_label="Open loop L(s)",
    )
    phase.line(
        "omega", "closed_phase", source=source,
        line_width=2.0, line_dash="dashed", legend_label="Closed loop T(s)",
    )
    minus_180 = Span(
        location=-180.0, dimension="width", line_dash="dotted", line_width=1.4
    )
    phase.add_layout(minus_180)
    phase_cross = Span(
        location=float(default.gain_crossover_rad_s),
        dimension="height", line_dash="dotdash", line_width=1.4,
    )
    phase.add_layout(phase_cross)
    phase.legend.location = "bottom_left"
    phase.legend.click_policy = "mute"

    metrics = Div(
        text=_metric_html(default),
        height=150,
        sizing_mode="stretch_width",
        styles={
            "white-space": "normal",
            "overflow-wrap": "anywhere",
            "line-height": "1.35",
        },
    )
    summary = Div(
        text=(
            "<b>Current-loop summary</b><br>"
            + _summary_table(results)
            + "<div style='margin-top:6px;white-space:normal;overflow-wrap:anywhere'>"
            "Engineering interpretation used here: PM ≥ 60° strong; 45–60° good; "
            "30–45° moderate; &lt;30° low. These are design heuristics, not "
            "certification limits. Closed-loop poles must also remain in the "
            "left-half plane.</div>"
        ),
        sizing_mode="stretch_width",
        styles={"white-space": "normal", "overflow-wrap": "anywhere"},
    )
    limitation = Div(
        text=(
            "<b>Interpretation:</b> The Bode models are reduced-order local loop "
            "models derived from the gains, inertia and actuator time constants "
            "used by the running simulator. They are not a global linearization "
            "of the full nonlinear mission. Comfort saturation, altitude-safety "
            "override, large gusts and nacelle scheduling are nonlinear and are "
            "therefore evaluated with the existing time-domain regression tests."
        ),
        height=70,
        sizing_mode="stretch_width",
        styles={"white-space": "normal", "overflow-wrap": "anywhere"},
    )

    def update_loop(attr, old, new) -> None:
        del attr, old
        result = results[str(new)]
        source.data = {
            "omega": result.omega_rad_s,
            "open_mag": result.open_mag_db,
            "closed_mag": result.closed_mag_db,
            "open_phase": result.open_phase_deg,
            "closed_phase": result.closed_phase_deg,
        }
        gain_cross.location = float(result.gain_crossover_rad_s)
        phase_cross.location = float(result.gain_crossover_rad_s)
        metrics.text = _metric_html(result)
        magnitude.title.text = f"Bode magnitude — {result.definition.label}"
        phase.title.text = f"Bode phase — {result.definition.label}"

    selector.on_change("value", update_loop)

    section = column(
        title,
        selector,
        row(magnitude, phase, sizing_mode="stretch_width"),
        metrics,
        summary,
        limitation,
        sizing_mode="stretch_width",
    )
    root = doc.roots[0]
    if hasattr(root, "children"):
        root.children.append(section)

    return sim
