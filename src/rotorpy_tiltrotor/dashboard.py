"""Interactive Bokeh dashboard for real-time tiltrotor monitoring and commands."""
from __future__ import annotations

import numpy as np
from bokeh.layouts import column, gridplot, row
from bokeh.models import Button, ColumnDataSource, Div, NumericInput, Select, Slider
from bokeh.palettes import Category10
from bokeh.plotting import figure

from .commands import AutomaticMission, FlightPhase
from .simulator import TiltrotorSimulation, sample_to_record

PHASE_NAMES = {int(p): p.name for p in FlightPhase}
COLORS = Category10[10]


def _empty_source() -> ColumnDataSource:
    fields = [
        "time_s","phase","x_m","y_m","altitude_m","airspeed_mps","target_altitude_m","target_airspeed_mps",
        "roll_deg","pitch_deg","yaw_deg","p_deg_s","q_deg_s","r_deg_s","tilt_deg","aileron_deg","elevator_deg","rudder_deg",
        "rotor_1_rpm","rotor_2_rpm","rotor_3_rpm","rotor_4_rpm","power_kw","soc_pct","ax_mps2","ay_mps2","az_mps2","jerk_mps3",
        "alpha_deg","beta_deg","qbar_pa","CL","CD","Cl","Cm","Cn","aero_force_x","aero_force_y","aero_force_z",
        "rotor_force_x","rotor_force_y","rotor_force_z","total_moment_x","total_moment_y","total_moment_z",
    ]
    return ColumnDataSource({k: [] for k in fields})


def build_dashboard(doc, simulation: TiltrotorSimulation | None = None) -> TiltrotorSimulation:
    sim = TiltrotorSimulation(dt=0.01) if simulation is None else simulation
    source = _empty_source()
    path_source = ColumnDataSource({"x": [], "y": [], "z": []})
    runtime = {"paused": True, "automatic": False, "mission": None, "plot_counter": 0, "manual_settle_s": 0.0}

    status = Div(text="<b>Paused — choose a command.</b>", width=1250)
    altitude_input = NumericInput(title="Target altitude [m]", value=30.0, low=1.0, high=200.0, width=160)
    speed_input = NumericInput(title="Cruise speed [m/s]", value=15.0, low=7.0, high=45.0, width=160)
    distance_input = NumericInput(title="Cruise distance [m]", value=180.0, low=20.0, high=2000.0, width=165)
    wind_x = Slider(title="Wind x [m/s]", start=-10.0, end=10.0, value=0.0, step=0.5, width=220)
    wind_y = Slider(title="Wind y [m/s]", start=-10.0, end=10.0, value=0.0, step=0.5, width=220)
    update_rate = Select(title="Plot refresh", value="10 Hz", options=["5 Hz","10 Hz","20 Hz"], width=130)

    auto_btn = Button(label="Run automatic mission", button_type="success", width=175)
    takeoff_btn = Button(label="Vertical takeoff", button_type="success", width=140)
    transition_btn = Button(label="Transition → cruise", button_type="primary", width=155)
    cruise_btn = Button(label="Cruise", button_type="primary", width=95)
    back_btn = Button(label="Transition → hover", button_type="warning", width=155)
    land_btn = Button(label="Vertical landing", button_type="danger", width=140)
    pause_btn = Button(label="Start", button_type="default", width=85)
    reset_btn = Button(label="Reset", width=75)

    def clear_history():
        source.data = {k: [] for k in source.data}
        path_source.data = {"x": [], "y": [], "z": []}
        runtime["plot_counter"] = 0
        runtime["manual_settle_s"] = 0.0

    def start_running():
        runtime["paused"] = False
        pause_btn.label = "Pause"

    def enter_manual_mode():
        runtime["automatic"] = False
        runtime["mission"] = None
        runtime["manual_settle_s"] = 0.0
        start_running()

    def target_altitude():
        return float(altitude_input.value or 30.0)

    def takeoff():
        enter_manual_mode()
        sim.commander.vertical_takeoff(sim.t, target_altitude(), sim.state["x"][0], sim.state["x"][1])

    def transition():
        enter_manual_mode()
        sim.commander.transition_to_cruise(sim.t, target_altitude(), float(speed_input.value or 15.0))

    def cruise():
        enter_manual_mode()
        sim.commander.cruise(sim.t, target_altitude(), float(speed_input.value or 15.0))

    def back_transition():
        enter_manual_mode()
        sim.commander.transition_to_hover(sim.t, target_altitude(), sim.state["x"][0] + 25.0, sim.state["x"][1])

    def land():
        enter_manual_mode()
        sim.commander.vertical_land(sim.t, sim.state["x"][0], sim.state["x"][1])

    def automatic_mission():
        sim.reset()
        clear_history()
        runtime["mission"] = AutomaticMission(sim.commander, altitude_m=target_altitude(), cruise_speed_mps=float(speed_input.value or 15.0), cruise_distance_m=float(distance_input.value or 180.0))
        runtime["automatic"] = True
        start_running()

    def toggle_pause():
        runtime["paused"] = not runtime["paused"]
        pause_btn.label = "Start" if runtime["paused"] else "Pause"

    def reset():
        sim.reset(); clear_history()
        runtime["automatic"] = False
        runtime["mission"] = None
        runtime["paused"] = True
        pause_btn.label = "Start"
        status.text = "<b>Reset complete — choose a command.</b>"

    auto_btn.on_click(automatic_mission); takeoff_btn.on_click(takeoff); transition_btn.on_click(transition); cruise_btn.on_click(cruise)
    back_btn.on_click(back_transition); land_btn.on_click(land); pause_btn.on_click(toggle_pause); reset_btn.on_click(reset)

    def time_plot(title, y_label, series, height=260):
        p = figure(title=title, x_axis_label="Time [s]", y_axis_label=y_label, height=height, sizing_mode="stretch_width")
        for spec in series:
            p.line("time_s", spec["field"], source=source, legend_label=spec["label"], line_width=spec.get("width", 2.2), line_color=spec["color"], line_dash=spec.get("dash", "solid"), line_alpha=spec.get("alpha", 0.95), muted_alpha=0.12)
        p.legend.click_policy = "mute"
        p.legend.location = "top_left"
        return p

    p_energy = time_plot("Flight state", "m / m·s⁻¹", [
        {"field":"altitude_m","label":"Altitude","color":COLORS[0]},
        {"field":"target_altitude_m","label":"Altitude target","color":COLORS[0],"dash":"dashed"},
        {"field":"airspeed_mps","label":"Airspeed","color":COLORS[1]},
        {"field":"target_airspeed_mps","label":"Airspeed target","color":COLORS[1],"dash":"dashed"},
    ])
    p_att = time_plot("Attitude", "deg", [
        {"field":"roll_deg","label":"Roll","color":COLORS[0]},
        {"field":"pitch_deg","label":"Pitch","color":COLORS[1]},
        {"field":"yaw_deg","label":"Yaw","color":COLORS[2]},
    ])
    p_rates = time_plot("Body rates", "deg/s", [
        {"field":"p_deg_s","label":"p","color":COLORS[0]},
        {"field":"q_deg_s","label":"q","color":COLORS[1]},
        {"field":"r_deg_s","label":"r","color":COLORS[2]},
    ])
    p_act = time_plot("Tilt and control surfaces", "deg", [
        {"field":"tilt_deg","label":"Nacelle tilt","color":COLORS[4],"width":2.8},
        {"field":"aileron_deg","label":"Aileron","color":COLORS[0]},
        {"field":"elevator_deg","label":"Elevator","color":COLORS[1]},
        {"field":"rudder_deg","label":"Rudder","color":COLORS[2]},
    ])
    p_rpm = time_plot("Rotor speeds", "rpm", [
        {"field":"rotor_1_rpm","label":"Rotor 1","color":COLORS[0]},
        {"field":"rotor_2_rpm","label":"Rotor 2","color":COLORS[1],"dash":"dashed"},
        {"field":"rotor_3_rpm","label":"Rotor 3","color":COLORS[2],"dash":"dotdash"},
        {"field":"rotor_4_rpm","label":"Rotor 4","color":COLORS[3],"dash":"dotted"},
    ])
    p_comfort = time_plot("Acceleration and jerk", "SI units", [
        {"field":"ax_mps2","label":"aₓ","color":COLORS[0]},
        {"field":"ay_mps2","label":"aᵧ","color":COLORS[1]},
        {"field":"az_mps2","label":"a_z","color":COLORS[2]},
        {"field":"jerk_mps3","label":"|jerk|","color":COLORS[3],"dash":"dashed"},
    ])
    p_aero = time_plot("Aerodynamic state", "deg / coefficient", [
        {"field":"alpha_deg","label":"α","color":COLORS[0]},
        {"field":"beta_deg","label":"β","color":COLORS[1]},
        {"field":"CL","label":"CL","color":COLORS[2],"dash":"dashed"},
        {"field":"CD","label":"CD","color":COLORS[3],"dash":"dotdash"},
    ])
    p_power = time_plot("Power and battery", "kW / %", [
        {"field":"power_kw","label":"Power","color":COLORS[3]},
        {"field":"soc_pct","label":"SOC","color":COLORS[2],"dash":"dashed"},
    ])

    p_top = figure(title="Top view", x_axis_label="x [m]", y_axis_label="y [m]", height=320, sizing_mode="stretch_width", match_aspect=True)
    p_top.line("x", "y", source=path_source, line_width=2.4, line_color=COLORS[0])
    p_top.scatter("x", "y", source=path_source, size=4, fill_color=COLORS[1], line_color=None)
    p_side = figure(title="Side view", x_axis_label="Along-track x [m]", y_axis_label="Altitude [m]", height=320, sizing_mode="stretch_width")
    p_side.line("x", "z", source=path_source, line_width=2.4, line_color=COLORS[4])

    callback_period_s = 0.05
    sim_substeps = max(1, int(round(callback_period_s / sim.dt)))

    def manual_phase_supervisor(dt):
        sp = sim.commander.setpoint
        if sp.phase == FlightPhase.VERTICAL_TAKEOFF:
            altitude_error = abs(float(sim.state["x"][2] - sp.altitude_m))
            settled = altitude_error < 0.25 and abs(float(sim.state["v"][2])) < 0.20
            runtime["manual_settle_s"] = runtime["manual_settle_s"] + dt if settled else 0.0
            if runtime["manual_settle_s"] >= 0.75:
                sim.commander.hover(sim.t, sp.altitude_m, sp.hold_x_m, sp.hold_y_m)
                runtime["manual_settle_s"] = 0.0

    def tick():
        if runtime["paused"]:
            return
        wind = np.array([wind_x.value, wind_y.value, 0.0])
        for _ in range(sim_substeps):
            if runtime["automatic"] and runtime["mission"] is not None:
                runtime["mission"].update(sim.t, sim.state)
                if sim.commander.setpoint.phase == FlightPhase.COMPLETE:
                    runtime["paused"] = True; pause_btn.label = "Start"; break
            else:
                manual_phase_supervisor(sim.dt)
            sim.step(wind)
        runtime["plot_counter"] += 1
        refresh_hz = int(update_rate.value.split()[0])
        stream_every = max(1, int(round((1.0 / refresh_hz) / callback_period_s)))
        if runtime["plot_counter"] % stream_every != 0:
            return
        rec = sample_to_record(sim.last_sample, sim.vehicle.params)
        source.stream({k: [rec.get(k, np.nan)] for k in source.data}, rollover=4800)
        path_source.stream({"x":[rec["x_m"]],"y":[rec["y_m"]],"z":[rec["altitude_m"]]}, rollover=8000)
        phase = PHASE_NAMES.get(rec["phase"], str(rec["phase"]))
        mode = "automatic mission" if runtime["automatic"] else "manual commands"
        status.text = f"<b>Mode:</b> {mode} &nbsp; <b>Phase:</b> {phase} &nbsp; <b>t:</b> {rec['time_s']:.1f} s &nbsp; <b>Altitude:</b> {rec['altitude_m']:.2f}/{rec['target_altitude_m']:.1f} m &nbsp; <b>Airspeed:</b> {rec['airspeed_mps']:.2f}/{rec['target_airspeed_mps']:.1f} m/s &nbsp; <b>Pitch:</b> {rec['pitch_deg']:.2f}° &nbsp; <b>Tilt:</b> {rec['tilt_deg']:.1f}° &nbsp; <b>Power:</b> {rec['power_kw']:.1f} kW &nbsp; <b>SOC:</b> {rec['soc_pct']:.1f}%"

    controls = row(altitude_input, speed_input, distance_input, auto_btn, takeoff_btn, transition_btn, cruise_btn, back_btn, land_btn, pause_btn, reset_btn, sizing_mode="stretch_width")
    environment_controls = row(wind_x, wind_y, update_rate, sizing_mode="stretch_width")
    plots = gridplot([[p_energy,p_att],[p_rates,p_act],[p_rpm,p_comfort],[p_aero,p_power],[p_top,p_side]], sizing_mode="stretch_width")
    doc.add_root(column(Div(text="<h2>RotorPy Tiltrotor Live Monitor</h2>"), controls, environment_controls, status, plots, sizing_mode="stretch_width"))
    doc.title = "RotorPy Tiltrotor"
    doc.add_periodic_callback(tick, int(callback_period_s * 1000))
    return sim
