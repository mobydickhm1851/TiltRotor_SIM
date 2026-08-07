import numpy as np

from rotorpy_tiltrotor.commands import FlightPhase
from rotorpy_tiltrotor.scenarios import UrbanWindModel, WindScenarioConfig
from rotorpy_tiltrotor.v045_dashboard import GuardAwareAutomaticMission
from rotorpy_tiltrotor.v045_final_dashboard import _new_simulation


def test_combined_guard_and_repeated_gust_preserves_safety_and_brakes_to_hover():
    sim = _new_simulation()
    sim.controller.max_accel = 0.50
    sim.controller.max_command_jerk_mps3 = 1.50
    sim.controller.command_jerk_headroom = 0.20

    mission = GuardAwareAutomaticMission(
        sim.commander,
        controller=sim.controller,
        altitude_m=30.0,
        cruise_speed_mps=15.0,
        cruise_distance_m=180.0,
    )
    wind = UrbanWindModel(WindScenarioConfig(
        enabled=True,
        mode="repeated_gusts",
        base_wind_mps=np.zeros(3),
        disturbance_amplitude_mps=5.0,
        disturbance_direction_deg=90.0,
        start_time_s=2.0,
        duration_s=3.0,
        wake_frequency_hz=0.25,
        reference_altitude_m=30.0,
        random_seed=11,
    ))

    phases = set()
    min_operational_altitude = float("inf")
    back_start_s = None
    low_ground_speed_s = None
    returned_hover = False
    max_nominal_command_accel = 0.0
    max_nominal_command_jerk = 0.0

    while sim.t < 170.0:
        mission.update(sim.t, sim.state)
        phase = sim.commander.setpoint.phase
        phases.add(phase)

        if phase == FlightPhase.TRANSITION_TO_HOVER and back_start_s is None:
            back_start_s = float(sim.t)

        sample = sim.step(wind.sample(sim.t, sim.state).vector_mps)

        if phase in {
            FlightPhase.HOVER,
            FlightPhase.TRANSITION_TO_CRUISE,
            FlightPhase.CRUISE,
            FlightPhase.TRANSITION_TO_HOVER,
        }:
            min_operational_altitude = min(
                min_operational_altitude,
                float(sim.state["x"][2]),
            )

        safety = float(getattr(
            sim.controller, "safety_override_factor", 0.0
        ))
        if safety <= 0.05:
            command_accel = float(np.linalg.norm(
                sim.controller._last_priority_accel_world
            ))
            command_jerk = float(
                sim.controller.last_command_jerk_norm_mps3
            )
            max_nominal_command_accel = max(
                max_nominal_command_accel, command_accel
            )
            max_nominal_command_jerk = max(
                max_nominal_command_jerk, command_jerk
            )

        if phase == FlightPhase.TRANSITION_TO_HOVER:
            ground_speed = float(np.linalg.norm(sim.state["v"][:2]))
            if ground_speed < 2.0 and low_ground_speed_s is None:
                low_ground_speed_s = float(sim.t)

        if (
            back_start_s is not None
            and phase == FlightPhase.HOVER
            and getattr(mission, "returning_to_land", False)
        ):
            returned_hover = True
            break

        assert np.all(np.isfinite(sim.state["x"]))
        assert np.all(np.isfinite(sim.state["v"]))
        assert np.all(np.isfinite(sim.state["q"]))

    assert FlightPhase.CRUISE in phases
    assert FlightPhase.TRANSITION_TO_HOVER in phases
    assert returned_hover
    assert min_operational_altitude > 25.0

    assert max_nominal_command_accel <= 0.5001
    assert max_nominal_command_jerk <= 0.3001

    assert back_start_s is not None
    assert low_ground_speed_s is not None
    braking_time = low_ground_speed_s - back_start_s
    # 15 m/s at a strict 0.5 m/s^2 resultant comfort target has an ideal lower
    # bound of 30 s.  Allow gust/altitude/lateral-control margin, but reject the
    # unnecessarily weak braking seen in the reported trace.
    assert braking_time < 45.0, f"braking_time={braking_time:.2f} s"
