import numpy as np

from rotorpy_tiltrotor.commands import FlightPhase
from rotorpy_tiltrotor.simulator import SimulationSample, sample_to_record


def make_record(phase, position, anchor=(0.0, 0.0), heading=0.0):
    state = {
        "x": np.asarray(position, dtype=float),
        "v": np.zeros(3),
        "q": np.array([0.0, 0.0, 0.0, 1.0]),
        "w": np.zeros(3),
        "rotor_speeds": np.zeros(4),
        "tilt_angle": np.array(0.0),
        "surfaces": np.zeros(3),
        "energy_j": np.array(100.0),
    }
    control = {
        "phase_code": np.array(float(phase)),
        "target_altitude": np.array(30.0),
        "target_airspeed": np.array(15.0),
        "route_anchor_x": np.array(anchor[0]),
        "route_anchor_y": np.array(anchor[1]),
        "route_heading": np.array(heading),
    }
    sample = SimulationSample(
        t=1.0,
        state=state,
        control=control,
        diagnostics={},
    )
    return sample_to_record(sample, {"battery_capacity_j": 100.0})


def test_cruise_target_is_route_projection_not_fixed_anchor():
    record = make_record(
        FlightPhase.CRUISE,
        position=[120.0, 5.0, 30.0],
    )

    assert np.isclose(record["target_x_m"], 120.0)
    assert np.isclose(record["target_y_m"], 0.0)
    assert np.isclose(record["x_error_m"], 0.0)
    assert np.isclose(record["y_error_m"], -5.0)
    assert np.isclose(record["cross_track_error_m"], -5.0)
    assert np.isclose(record["along_track_distance_m"], 120.0)
    assert record["position_reference_type"] == 1.0


def test_route_projection_supports_nonzero_heading():
    record = make_record(
        FlightPhase.TRANSITION_TO_CRUISE,
        position=[5.0, 120.0, 30.0],
        heading=np.deg2rad(90.0),
    )

    assert np.isclose(record["target_x_m"], 0.0, atol=1e-10)
    assert np.isclose(record["target_y_m"], 120.0)
    assert np.isclose(record["x_error_m"], -5.0)
    assert np.isclose(record["y_error_m"], 0.0, atol=1e-10)
    assert np.isclose(record["cross_track_error_m"], 5.0)


def test_hover_keeps_fixed_point_target():
    record = make_record(
        FlightPhase.HOVER,
        position=[2.0, -1.0, 30.0],
    )

    assert np.isclose(record["target_x_m"], 0.0)
    assert np.isclose(record["target_y_m"], 0.0)
    assert np.isclose(record["x_error_m"], -2.0)
    assert np.isclose(record["y_error_m"], 1.0)
    assert record["position_reference_type"] == 0.0
