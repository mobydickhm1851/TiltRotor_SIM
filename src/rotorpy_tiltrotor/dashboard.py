"""Interactive Bokeh dashboard for tiltrotor flight and urban scenarios."""
from __future__ import annotations

import numpy as np
from bokeh.layouts import column, gridplot, row
from bokeh.models import (
    Button,
    ColumnDataSource,
    Div,
    NumericInput,
    Select,
    Slider,
    Toggle,
)
from bokeh.palettes import Category10
from bokeh.plotting import figure

from .commands import AutomaticMission, FlightPhase
from .scenarios import (
    ComfortLimits,
    PassengerComfortMonitor,
    UrbanWindModel,
    WindScenarioConfig,
)
from .simulator import TiltrotorSimulation, sample_to_record


PHASE_NAMES = {int(p): p.name for p in FlightPhase}
COLORS = Category10[10]
BACK_TRANSITION_DURATION_S = 14.0
BACK_TRANSITION_COMFORT_DECEL_MPS2 = 1.25
BACK_TRANSITION_MIN_FORWARD_M = 8.0
BACK_TRANSITION_BUFFER_M = 5.0
MANUAL_PHASE_SETTLE_S = 0.75

WIND_MODE_LABELS = {
    "Steady wind": "steady",
    "Discrete gust": "discrete_gust",
    "Wind shear": "wind_shear",
    "Urban wake proxy": "urban_wake",
}


def planned_hover_target(
    state: dict,
    setpoint,
    duration_s: float = BACK_TRANSITION_DURATION_S,
    comfortable_decel_mps2: float = BACK_TRANSITION_COMFORT_DECEL_MPS2,
) -> tuple[np.ndarray, float]:
    """Return one fixed route-centreline stop point for back transition."""
    heading = float(setpoint.heading_rad)
    forward = np.array([np.cos(heading), np.sin(heading)], dtype=float)
    anchor = np.array(
        [float(setpoint.hold_x_m), float(setpoint.hold_y_m)],
        dtype=float,
    )
    position = np.asarray(state["x"], dtype=float)[:2]
    velocity = np.asarray(state["v"], dtype=float)[:2]
    along_distance = float(np.dot(position - anchor, forward))
    route_projection = anchor + along_distance * forward
    forward_speed = max(0.0, float(np.dot(velocity, forward)))

    duration_s = max(4.0, float(duration_s))
    comfortable_decel_mps2 = max(0.25, float(comfortable_decel_mps2))
    schedule_distance = 0.5 * forward_speed * duration_s
    braking_distance = (
        forward_speed * forward_speed / (2.0 * comfortable_decel_mps2)
    )
    forward_distance = (
        max(
            BACK_TRANSITION_MIN_FORWARD_M,
            schedule_distance,
            braking_distance,
        )
        + BACK_TRANSITION_BUFFER_M
    )
    return route_projection + forward_distance * forward, forward_distance


def supervise_manual_phase(
    sim: TiltrotorSimulation,
    settled_time_s: float,
    dt: float,
) -> float:
    """Advance manual phase-completion logic and return settle time."""
    setpoint = sim.commander.setpoint
    phase = setpoint.phase

    if phase == FlightPhase.VERTICAL_TAKEOFF:
        altitude_error = abs(float(sim.state["x"][2] - setpoint.altitude_m))
        settled = (
            altitude_error < 0.25
            and abs(float(sim.state["v"][2])) < 0.20
        )
        settled_time_s = settled_time_s + dt if settled else 0.0
        if settled_time_s >= MANUAL_PHASE_SETTLE_S:
            sim.commander.hover(
                sim.t,
                setpoint.altitude_m,
                setpoint.hold_x_m,
                setpoint.hold_y_m,
            )
            return 0.0
        return settled_time_s

    if phase == FlightPhase.TRANSITION_TO_HOVER:
        horizontal_speed = float(np.linalg.norm(sim.state["v"][:2]))
        vertical_speed = abs(float(sim.state["v"][2]))
        tilt = abs(float(sim.state["tilt_angle"]))
        horizontal_error = float(np.linalg.norm(
            np.array([setpoint.hold_x_m, setpoint.hold_y_m])
            - np.asarray(sim.state["x"], dtype=float)[:2]
        ))
        schedule_finished = sim.commander.phase_progress(sim.t) >= 0.995
        settled = (
            horizontal_speed < 0.75
            and vertical_speed < 0.60
            and tilt < np.deg2rad(6.0)
            and (horizontal_error < 4.0 or schedule_finished)
        )
        settled_time_s = settled_time_s + dt if settled else 0.0
        if settled_time_s >= MANUAL_PHASE_SETTLE_S:
            sim.commander.hover(
                sim.t,
                setpoint.altitude_m,
                setpoint.hold_x_m,
                setpoint.hold_y_m,
            )
            return 0.0
        return settled_time_s

    return 0.0


SOURCE_FIELDS = [
    "time_s", "phase", "x_m", "y_m", "altitude_m", "airspeed_mps",
    "target_altitude_m", "target_airspeed_mps",
    "target_x_m", "target_y_m", "route_anchor_x_m", "route_anchor_y_m",
    "x_error_m", "y_error_m", "cross_track_error_m",
    "along_track_error_m", "along_track_distance_m",
    "position_reference_type",
    "roll_deg", "pitch_deg", "yaw_deg",
    "p_deg_s", "q_deg_s", "r_deg_s", "body_rate_norm_deg_s",
    "tilt_deg", "aileron_deg", "elevator_deg", "rudder_deg",
    "rotor_1_rpm", "rotor_2_rpm", "rotor_3_rpm", "rotor_4_rpm",
    "power_kw", "soc_pct",
    "ax_mps2", "ay_mps2", "az_mps2", "jerk_mps3",
    "alpha_deg", "beta_deg", "qbar_pa", "wing_blend",
    "CL", "CD", "Cl", "Cm", "Cn",
    "aero_force_x", "aero_force_y", "aero_force_z",
    "rotor_force_x", "rotor_force_y", "rotor_force_z",
    "total_moment_x", "total_moment_y", "total_moment_z",
    "wind_x_mps", "wind_y_mps", "wind_z_mps", "wind_speed_mps",
    "gust_component_mps", "wind_scenario_code", "gust_active",
    "comfort_enabled",
    "comfort_accel_mps2", "comfort_accel_rms_mps2",
    "comfort_jerk_mps3", "comfort_jerk_rms_mps3",
    "comfort_rate_deg_s", "comfort_rate_rms_deg_s",
    "comfort_index", "comfort_rms_index", "comfort_alert",
    "comfort_exceedance_time_s",
    "comfort_peak_accel_mps2", "comfort_peak_jerk_mps3",
    "comfort_peak_rate_deg_s",
    "comfort_accel_limit_mps2", "comfort_jerk_limit_mps3",
    "comfort_rate_limit_deg_s", "comfort_index_limit",
]


def _empty_source() -> ColumnDataSource:
    return ColumnDataSource({key: [] for key in SOURCE_FIELDS})


def build_dashboard(
    doc,
    simulation: TiltrotorSimulation | None = None,
) -> TiltrotorSimulation:
    sim = TiltrotorSimulation(dt=0.01) if simulation is None else simulation
    source = _empty_source()
    path_source = ColumnDataSource({"x": [], "y": [], "z": []})
    wind_model = UrbanWindModel()
    comfort_monitor = PassengerComfortMonitor()
    original_max_accel = float(sim.controller.max_accel)

    runtime = {
        "paused": True,
        "automatic": False,
        "mission": None,
        "plot_counter": 0,
        "manual_settle_s": 0.0,
        "last_wind": wind_model.last_sample,
        "last_comfort": {},
    }

    status = Div(text="<b>Paused — choose a command.</b>", width=1500)
    scenario_status = Div(
        text=(
            "<b>Urban scenario:</b> disabled &nbsp; "
            "<b>Comfort monitor:</b> enabled &nbsp; "
            "<b>Comfort guard:</b> disabled"
        ),
        width=1500,
    )

    altitude_input = NumericInput(
        title="Target altitude [m]", value=30.0,
        low=1.0, high=200.0, width=150,
    )
    speed_input = NumericInput(
        title="Cruise speed [m/s]", value=15.0,
        low=7.0, high=45.0, width=150,
    )
    distance_input = NumericInput(
        title="Cruise distance [m]", value=180.0,
        low=20.0, high=2000.0, width=160,
    )
    update_rate = Select(
        title="Plot refresh", value="10 Hz",
        options=["5 Hz", "10 Hz", "20 Hz"], width=125,
    )

    wind_toggle = Toggle(
        label="Urban wind / gust: OFF",
        active=False,
        button_type="default",
        width=170,
    )
    wind_mode = Select(
        title="Wind scenario",
        value="Discrete gust",
        options=list(WIND_MODE_LABELS),
        width=155,
    )
    wind_x = Slider(
        title="Base wind x [m/s]", start=-12.0, end=12.0,
        value=0.0, step=0.5, width=210,
    )
    wind_y = Slider(
        title="Base wind y [m/s]", start=-12.0, end=12.0,
        value=0.0, step=0.5, width=210,
    )
    wind_z = Slider(
        title="Base wind z [m/s]", start=-5.0, end=5.0,
        value=0.0, step=0.25, width=210,
    )
    gust_amplitude = NumericInput(
        title="Disturbance amplitude [m/s]", value=5.0,
        low=0.0, high=20.0, width=185,
    )
    gust_direction = NumericInput(
        title="Direction [deg]", value=90.0,
        low=-360.0, high=360.0, width=140,
    )
    gust_start = NumericInput(
        title="Start time [s]", value=20.0,
        low=0.0, high=1000.0, width=130,
    )
    gust_duration = NumericInput(
        title="Duration [s]", value=8.0,
        low=0.1, high=300.0, width=130,
    )
    shear_gradient = NumericInput(
        title="Shear [m/s per m]", value=0.08,
        low=0.0, high=1.0, width=160,
    )
    wake_frequency = NumericInput(
        title="Wake frequency [Hz]", value=0.35,
        low=0.02, high=3.0, width=160,
    )

    comfort_toggle = Toggle(
        label="Comfort monitor: ON",
        active=True,
        button_type="success",
        width=165,
    )
    comfort_guard = Toggle(
        label="Comfort accel guard: OFF",
        active=False,
        button_type="default",
        width=190,
    )
    comfort_accel_limit = NumericInput(
        title="Acceleration limit [m/s²]", value=2.0,
        low=0.1, high=10.0, width=180,
    )
    comfort_jerk_limit = NumericInput(
        title="Jerk limit [m/s³]", value=1.5,
        low=0.1, high=20.0, width=160,
    )
    comfort_rate_limit = NumericInput(
        title="Angular-rate limit [deg/s]", value=10.0,
        low=0.5, high=100.0, width=190,
    )
    comfort_window = NumericInput(
        title="RMS window [s]", value=5.0,
        low=0.25, high=60.0, width=145,
    )

    auto_btn = Button(
        label="Run automatic mission", button_type="success", width=175,
    )
    takeoff_btn = Button(
        label="Vertical takeoff", button_type="success", width=140,
    )
    transition_btn = Button(
        label="Transition → cruise", button_type="primary", width=155,
    )
    cruise_btn = Button(label="Cruise", button_type="primary", width=95)
    back_btn = Button(
        label="Transition → hover", button_type="warning", width=155,
    )
    land_btn = Button(
        label="Vertical landing", button_type="danger", width=140,
    )
    pause_btn = Button(label="Start", button_type="default", width=85)
    reset_btn = Button(label="Reset", width=75)

    def target_altitude() -> float:
        return float(altitude_input.value or 30.0)

    def comfort_limits() -> ComfortLimits:
        return ComfortLimits(
            acceleration_mps2=float(comfort_accel_limit.value or 2.0),
            jerk_mps3=float(comfort_jerk_limit.value or 1.5),
            angular_rate_deg_s=float(comfort_rate_limit.value or 10.0),
            rms_window_s=float(comfort_window.value or 5.0),
        ).normalized()

    def update_toggle_labels() -> None:
        wind_toggle.label = (
            "Urban wind / gust: ON"
            if wind_toggle.active
            else "Urban wind / gust: OFF"
        )
        wind_toggle.button_type = "warning" if wind_toggle.active else "default"
        comfort_toggle.label = (
            "Comfort monitor: ON"
            if comfort_toggle.active
            else "Comfort monitor: OFF"
        )
        comfort_toggle.button_type = (
            "success" if comfort_toggle.active else "default"
        )
        comfort_guard.label = (
            "Comfort accel guard: ON"
            if comfort_guard.active
            else "Comfort accel guard: OFF"
        )
        comfort_guard.button_type = (
            "warning" if comfort_guard.active else "default"
        )

    def configure_scenarios() -> None:
        wind_model.configure(WindScenarioConfig(
            enabled=wind_toggle.active,
            mode=WIND_MODE_LABELS[wind_mode.value],
            base_wind_mps=np.array([
                float(wind_x.value),
                float(wind_y.value),
                float(wind_z.value),
            ]),
            disturbance_amplitude_mps=float(gust_amplitude.value or 0.0),
            disturbance_direction_deg=float(gust_direction.value or 0.0),
            start_time_s=float(gust_start.value or 0.0),
            duration_s=float(gust_duration.value or 0.1),
            shear_gradient_per_m=float(shear_gradient.value or 0.0),
            reference_altitude_m=target_altitude(),
            wake_frequency_hz=float(wake_frequency.value or 0.35),
        ))
        comfort_monitor.configure(
            enabled=comfort_toggle.active,
            limits=comfort_limits(),
        )
        sim.controller.max_accel = (
            min(
                original_max_accel,
                comfort_limits().acceleration_mps2,
            )
            if comfort_guard.active
            else original_max_accel
        )
        update_toggle_labels()

    def clear_history() -> None:
        source.data = {key: [] for key in SOURCE_FIELDS}
        path_source.data = {"x": [], "y": [], "z": []}
        runtime["plot_counter"] = 0
        runtime["manual_settle_s"] = 0.0
        wind_model.reset()
        comfort_monitor.reset()
        runtime["last_wind"] = wind_model.last_sample
        runtime["last_comfort"] = {}

    def start_running() -> None:
        runtime["paused"] = False
        pause_btn.label = "Pause"

    def enter_manual_mode() -> None:
        runtime["automatic"] = False
        runtime["mission"] = None
        runtime["manual_settle_s"] = 0.0
        start_running()

    def takeoff() -> None:
        enter_manual_mode()
        sim.commander.vertical_takeoff(
            sim.t, target_altitude(),
            sim.state["x"][0], sim.state["x"][1],
        )

    def transition() -> None:
        enter_manual_mode()
        setpoint = sim.commander.setpoint
        sim.commander.transition_to_cruise(
            sim.t,
            target_altitude(),
            float(speed_input.value or 15.0),
            heading_rad=setpoint.heading_rad,
            route_x_m=setpoint.hold_x_m,
            route_y_m=setpoint.hold_y_m,
        )

    def cruise() -> None:
        enter_manual_mode()
        setpoint = sim.commander.setpoint
        sim.commander.cruise(
            sim.t,
            target_altitude(),
            float(speed_input.value or 15.0),
            heading_rad=setpoint.heading_rad,
            route_x_m=setpoint.hold_x_m,
            route_y_m=setpoint.hold_y_m,
        )

    def back_transition() -> None:
        setpoint = sim.commander.setpoint
        if setpoint.phase == FlightPhase.TRANSITION_TO_HOVER:
            runtime["automatic"] = False
            runtime["mission"] = None
            start_running()
            status.text = (
                "<b>Transition → hover is already active.</b> "
                "The stop point and tilt schedule were preserved."
            )
            return
        if setpoint.phase not in (
            FlightPhase.TRANSITION_TO_CRUISE,
            FlightPhase.CRUISE,
        ):
            status.text = (
                "<b>Transition → hover requires forward-flight mode.</b>"
            )
            return

        enter_manual_mode()
        decel = (
            min(
                BACK_TRANSITION_COMFORT_DECEL_MPS2,
                comfort_limits().acceleration_mps2,
            )
            if comfort_guard.active
            else BACK_TRANSITION_COMFORT_DECEL_MPS2
        )
        hover_target, forward_distance = planned_hover_target(
            sim.state,
            setpoint,
            duration_s=BACK_TRANSITION_DURATION_S,
            comfortable_decel_mps2=decel,
        )
        sim.commander.transition_to_hover(
            sim.t,
            target_altitude(),
            float(hover_target[0]),
            float(hover_target[1]),
            duration_s=BACK_TRANSITION_DURATION_S,
        )
        back_btn.disabled = True
        status.text = (
            "<b>Transition → hover started.</b> "
            f"Stop target is {forward_distance:.1f} m ahead."
        )

    def land() -> None:
        enter_manual_mode()
        sim.commander.vertical_land(
            sim.t, sim.state["x"][0], sim.state["x"][1],
        )

    def automatic_mission() -> None:
        sim.reset()
        clear_history()
        runtime["mission"] = AutomaticMission(
            sim.commander,
            altitude_m=target_altitude(),
            cruise_speed_mps=float(speed_input.value or 15.0),
            cruise_distance_m=float(distance_input.value or 180.0),
        )
        runtime["automatic"] = True
        start_running()

    def toggle_pause() -> None:
        runtime["paused"] = not runtime["paused"]
        pause_btn.label = "Start" if runtime["paused"] else "Pause"

    def reset() -> None:
        sim.reset()
        clear_history()
        runtime["automatic"] = False
        runtime["mission"] = None
        runtime["paused"] = True
        sim.controller.max_accel = original_max_accel
        back_btn.disabled = False
        pause_btn.label = "Start"
        status.text = "<b>Reset complete — choose a command.</b>"

    def comfort_toggle_changed(attr, old, new) -> None:
        comfort_monitor.reset()
        update_toggle_labels()

    wind_toggle.on_change("active", lambda attr, old, new: update_toggle_labels())
    comfort_toggle.on_change("active", comfort_toggle_changed)
    comfort_guard.on_change("active", lambda attr, old, new: update_toggle_labels())
    auto_btn.on_click(automatic_mission)
    takeoff_btn.on_click(takeoff)
    transition_btn.on_click(transition)
    cruise_btn.on_click(cruise)
    back_btn.on_click(back_transition)
    land_btn.on_click(land)
    pause_btn.on_click(toggle_pause)
    reset_btn.on_click(reset)

    def time_plot(
        title: str,
        y_label: str,
        series: list[dict],
        height: int = 260,
    ):
        plot = figure(
            title=title,
            x_axis_label="Time [s]",
            y_axis_label=y_label,
            height=height,
            sizing_mode="stretch_width",
        )
        for spec in series:
            plot.line(
                "time_s",
                spec["field"],
                source=source,
                legend_label=spec["label"],
                line_width=spec.get("width", 2.2),
                line_color=spec["color"],
                line_dash=spec.get("dash", "solid"),
                line_alpha=spec.get("alpha", 0.95),
                muted_alpha=0.12,
            )
        plot.legend.click_policy = "mute"
        plot.legend.location = "top_left"
        return plot

    p_flight = time_plot("Flight state", "m / m·s⁻¹", [
        {"field": "altitude_m", "label": "Altitude", "color": COLORS[0]},
        {"field": "target_altitude_m", "label": "Altitude target",
         "color": COLORS[0], "dash": "dashed"},
        {"field": "airspeed_mps", "label": "Airspeed", "color": COLORS[1]},
        {"field": "target_airspeed_mps", "label": "Airspeed target",
         "color": COLORS[1], "dash": "dashed"},
    ])
    p_position = time_plot(
        "Position tracking (point hold / route projection)", "m", [
            {"field": "x_error_m", "label": "x tracking error",
             "color": COLORS[0]},
            {"field": "y_error_m", "label": "y tracking error",
             "color": COLORS[1]},
            {"field": "cross_track_error_m", "label": "Cross-track error",
             "color": COLORS[3], "dash": "dashed"},
        ],
    )
    p_wind = time_plot("Urban wind / gust", "m/s", [
        {"field": "wind_x_mps", "label": "Wind x", "color": COLORS[0]},
        {"field": "wind_y_mps", "label": "Wind y", "color": COLORS[1]},
        {"field": "wind_z_mps", "label": "Wind z", "color": COLORS[2]},
        {"field": "wind_speed_mps", "label": "|Wind|",
         "color": COLORS[4], "dash": "dashed"},
        {"field": "gust_component_mps", "label": "|Disturbance|",
         "color": COLORS[3], "dash": "dotdash"},
    ])
    p_comfort_index = time_plot(
        "Passenger comfort proxy", "normalized ratio", [
            {"field": "comfort_index", "label": "Instantaneous index",
             "color": COLORS[3]},
            {"field": "comfort_rms_index", "label": "Rolling RMS index",
             "color": COLORS[4]},
            {"field": "comfort_index_limit", "label": "Threshold",
             "color": COLORS[2], "dash": "dashed"},
        ],
    )
    p_comfort_accel = time_plot(
        "Passenger comfort — acceleration", "m/s²", [
            {"field": "ax_mps2", "label": "aₓ", "color": COLORS[0]},
            {"field": "ay_mps2", "label": "aᵧ", "color": COLORS[1]},
            {"field": "az_mps2", "label": "a_z", "color": COLORS[2]},
            {"field": "comfort_accel_mps2", "label": "|a|",
             "color": COLORS[3]},
            {"field": "comfort_accel_rms_mps2", "label": "RMS |a|",
             "color": COLORS[4]},
            {"field": "comfort_accel_limit_mps2", "label": "Limit",
             "color": COLORS[3], "dash": "dashed"},
        ],
    )
    p_comfort_jerk = time_plot(
        "Passenger comfort — jerk", "m/s³", [
            {"field": "comfort_jerk_mps3", "label": "|jerk|",
             "color": COLORS[3]},
            {"field": "comfort_jerk_rms_mps3", "label": "RMS |jerk|",
             "color": COLORS[4]},
            {"field": "comfort_jerk_limit_mps3", "label": "Limit",
             "color": COLORS[3], "dash": "dashed"},
        ],
    )
    p_attitude = time_plot("Attitude", "deg", [
        {"field": "roll_deg", "label": "Roll", "color": COLORS[0]},
        {"field": "pitch_deg", "label": "Pitch", "color": COLORS[1]},
        {"field": "yaw_deg", "label": "Yaw", "color": COLORS[2]},
    ])
    p_rates = time_plot("Body rates", "deg/s", [
        {"field": "p_deg_s", "label": "p", "color": COLORS[0]},
        {"field": "q_deg_s", "label": "q", "color": COLORS[1]},
        {"field": "r_deg_s", "label": "r", "color": COLORS[2]},
        {"field": "comfort_rate_deg_s", "label": "|body rate|",
         "color": COLORS[4]},
        {"field": "comfort_rate_limit_deg_s", "label": "Comfort limit",
         "color": COLORS[3], "dash": "dashed"},
    ])
    p_actuators = time_plot("Tilt and control surfaces", "deg", [
        {"field": "tilt_deg", "label": "Nacelle tilt",
         "color": COLORS[4], "width": 2.8},
        {"field": "aileron_deg", "label": "Aileron", "color": COLORS[0]},
        {"field": "elevator_deg", "label": "Elevator", "color": COLORS[1]},
        {"field": "rudder_deg", "label": "Rudder", "color": COLORS[2]},
    ])
    p_rpm = time_plot("Rotor speeds", "rpm", [
        {"field": "rotor_1_rpm", "label": "Rotor 1", "color": COLORS[0]},
        {"field": "rotor_2_rpm", "label": "Rotor 2",
         "color": COLORS[1], "dash": "dashed"},
        {"field": "rotor_3_rpm", "label": "Rotor 3",
         "color": COLORS[2], "dash": "dotdash"},
        {"field": "rotor_4_rpm", "label": "Rotor 4",
         "color": COLORS[3], "dash": "dotted"},
    ])
    p_aero = time_plot("Aerodynamic state", "deg / coefficient", [
        {"field": "alpha_deg", "label": "α", "color": COLORS[0]},
        {"field": "beta_deg", "label": "β", "color": COLORS[1]},
        {"field": "CL", "label": "CL", "color": COLORS[2], "dash": "dashed"},
        {"field": "CD", "label": "CD", "color": COLORS[3],
         "dash": "dotdash"},
        {"field": "wing_blend", "label": "Wing blend", "color": COLORS[4],
         "dash": "dotted"},
    ])
    p_power = time_plot("Power and battery", "kW / %", [
        {"field": "power_kw", "label": "Power", "color": COLORS[3]},
        {"field": "soc_pct", "label": "SOC", "color": COLORS[2],
         "dash": "dashed"},
    ])

    p_top = figure(
        title="Top view",
        x_axis_label="x [m]",
        y_axis_label="y [m]",
        height=320,
        sizing_mode="stretch_width",
        match_aspect=True,
    )
    p_top.line(
        "x", "y", source=path_source,
        line_width=2.4, line_color=COLORS[0],
        legend_label="Actual path",
    )
    p_top.scatter(
        "x", "y", source=path_source,
        size=4, fill_color=COLORS[1], line_color=None,
    )
    p_top.line(
        "target_x_m", "target_y_m", source=source,
        line_width=1.8, line_color=COLORS[3], line_dash="dashed",
        legend_label="Active point / route-projection target",
    )
    p_top.scatter(
        "route_anchor_x_m", "route_anchor_y_m", source=source,
        size=9, marker="cross", line_color=COLORS[2],
        legend_label="Fixed route anchor / hold point",
    )
    p_top.legend.click_policy = "mute"

    p_side = figure(
        title="Side view",
        x_axis_label="Along-track x [m]",
        y_axis_label="Altitude [m]",
        height=320,
        sizing_mode="stretch_width",
    )
    p_side.line(
        "x", "z", source=path_source,
        line_width=2.4, line_color=COLORS[4],
    )

    callback_period_s = 0.05
    sim_substeps = max(1, int(round(callback_period_s / sim.dt)))

    def tick() -> None:
        if runtime["paused"]:
            return

        configure_scenarios()
        latest_comfort = runtime["last_comfort"]
        for _ in range(sim_substeps):
            if runtime["automatic"] and runtime["mission"] is not None:
                runtime["mission"].update(sim.t, sim.state)
                if sim.commander.setpoint.phase == FlightPhase.COMPLETE:
                    runtime["paused"] = True
                    pause_btn.label = "Start"
                    break
            else:
                runtime["manual_settle_s"] = supervise_manual_phase(
                    sim,
                    runtime["manual_settle_s"],
                    sim.dt,
                )

            wind_sample = wind_model.sample(sim.t, sim.state)
            sim.step(wind_sample.vector_mps)
            latest_comfort = comfort_monitor.update(
                sim.t,
                sim.last_sample.diagnostics.get(
                    "accel_world", np.zeros(3)
                ),
                sim.last_sample.diagnostics.get(
                    "jerk_world", np.zeros(3)
                ),
                sim.state["w"],
            )
            runtime["last_wind"] = wind_sample
            runtime["last_comfort"] = latest_comfort

        back_btn.disabled = (
            sim.commander.setpoint.phase == FlightPhase.TRANSITION_TO_HOVER
        )
        runtime["plot_counter"] += 1
        refresh_hz = int(update_rate.value.split()[0])
        stream_every = max(
            1,
            int(round((1.0 / refresh_hz) / callback_period_s)),
        )
        if runtime["plot_counter"] % stream_every != 0:
            return

        record = sample_to_record(sim.last_sample, sim.vehicle.params)
        record.update(runtime["last_wind"].as_record())
        record.update(latest_comfort)
        record["body_rate_norm_deg_s"] = float(np.linalg.norm([
            record["p_deg_s"], record["q_deg_s"], record["r_deg_s"],
        ]))
        record["comfort_index_limit"] = 1.0
        source.stream(
            {key: [record.get(key, np.nan)] for key in SOURCE_FIELDS},
            rollover=6000,
        )
        path_source.stream(
            {
                "x": [record["x_m"]],
                "y": [record["y_m"]],
                "z": [record["altitude_m"]],
            },
            rollover=9000,
        )

        phase = PHASE_NAMES.get(record["phase"], str(record["phase"]))
        mode = (
            "automatic mission"
            if runtime["automatic"]
            else "manual commands"
        )
        reference_name = (
            "route projection"
            if record["position_reference_type"] > 0.5
            else "fixed point"
        )
        alert_text = (
            "<span style='color:#d62728'><b>LIMIT EXCEEDED</b></span>"
            if record.get("comfort_alert", 0.0) > 0.5
            else "<span style='color:#2ca02c'><b>within limits</b></span>"
        )
        status.text = (
            f"<b>Mode:</b> {mode} &nbsp; "
            f"<b>Phase:</b> {phase} &nbsp; "
            f"<b>t:</b> {record['time_s']:.1f} s &nbsp; "
            f"<b>Altitude:</b> "
            f"{record['altitude_m']:.2f}/"
            f"{record['target_altitude_m']:.1f} m &nbsp; "
            f"<b>Airspeed:</b> "
            f"{record['airspeed_mps']:.2f}/"
            f"{record['target_airspeed_mps']:.1f} m/s &nbsp; "
            f"<b>{reference_name} x/y:</b> "
            f"{record['target_x_m']:.1f}/"
            f"{record['target_y_m']:.1f} m &nbsp; "
            f"<b>XTE:</b> {record['cross_track_error_m']:.2f} m &nbsp; "
            f"<b>Wind:</b> {record['wind_speed_mps']:.2f} m/s &nbsp; "
            f"<b>Comfort:</b> {alert_text}"
        )
        scenario_status.text = (
            f"<b>Urban wind:</b> "
            f"{wind_mode.value if wind_toggle.active else 'disabled'} &nbsp; "
            f"<b>Disturbance:</b> "
            f"{record['gust_component_mps']:.2f} m/s &nbsp; "
            f"<b>Comfort index:</b> {record['comfort_index']:.2f} &nbsp; "
            f"<b>RMS index:</b> {record['comfort_rms_index']:.2f} &nbsp; "
            f"<b>Peak a:</b> "
            f"{record['comfort_peak_accel_mps2']:.2f} m/s² &nbsp; "
            f"<b>Peak jerk:</b> "
            f"{record['comfort_peak_jerk_mps3']:.2f} m/s³ &nbsp; "
            f"<b>Peak rate:</b> "
            f"{record['comfort_peak_rate_deg_s']:.2f} deg/s &nbsp; "
            f"<b>Exceedance:</b> "
            f"{record['comfort_exceedance_time_s']:.2f} s &nbsp; "
            f"<b>Acceleration guard:</b> "
            f"{'ON' if comfort_guard.active else 'OFF'}"
        )

    flight_controls = row(
        altitude_input, speed_input, distance_input, auto_btn,
        takeoff_btn, transition_btn, cruise_btn, back_btn, land_btn,
        pause_btn, reset_btn, sizing_mode="stretch_width",
    )
    wind_controls_1 = row(
        wind_toggle, wind_mode, wind_x, wind_y, wind_z,
        update_rate, sizing_mode="stretch_width",
    )
    wind_controls_2 = row(
        gust_amplitude, gust_direction, gust_start, gust_duration,
        shear_gradient, wake_frequency, sizing_mode="stretch_width",
    )
    comfort_controls = row(
        comfort_toggle, comfort_guard, comfort_accel_limit,
        comfort_jerk_limit, comfort_rate_limit, comfort_window,
        sizing_mode="stretch_width",
    )
    plots = gridplot([
        [p_flight, p_position],
        [p_wind, p_comfort_index],
        [p_comfort_accel, p_comfort_jerk],
        [p_attitude, p_rates],
        [p_actuators, p_rpm],
        [p_aero, p_power],
        [p_top, p_side],
    ], sizing_mode="stretch_width")

    doc.add_root(column(
        Div(text="<h2>RotorPy Tiltrotor Live Monitor</h2>"),
        Div(text="<h3>Flight commands</h3>"),
        flight_controls,
        Div(text="<h3>Urban wind / gust scenario</h3>"),
        wind_controls_1,
        wind_controls_2,
        Div(text="<h3>Passenger comfort monitor and guard</h3>"),
        comfort_controls,
        status,
        scenario_status,
        plots,
        sizing_mode="stretch_width",
    ))
    doc.title = "RotorPy Tiltrotor"
    doc.add_periodic_callback(tick, int(callback_period_s * 1000))
    update_toggle_labels()
    return sim
