import numpy as np

from rotorpy_tiltrotor import TiltrotorSimulation


def test_vertical_takeoff_climbs():
    sim = TiltrotorSimulation(dt=0.01)
    sim.commander.vertical_takeoff(0.0, altitude_m=5.0)
    df = sim.run(8.0, record_hz=10.0, stop_on_complete=False)
    assert np.isfinite(df.to_numpy(dtype=float)).all()
    assert df.altitude_m.iloc[-1] > 2.0
    assert df.altitude_m.max() < 20.0
