"""Run the automatic takeoff-transition-cruise-landing demonstration."""
from pathlib import Path

import matplotlib.pyplot as plt

from rotorpy_tiltrotor import AutomaticMission, TiltrotorSimulation


if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parents[1] / "output"
    output_dir.mkdir(exist_ok=True)
    sim = TiltrotorSimulation(dt=0.01)
    mission = AutomaticMission(sim.commander, altitude_m=30.0, cruise_speed_mps=15.0, cruise_distance_m=180.0)
    df = sim.run(duration_s=120.0, mission=mission, record_hz=20.0)
    csv_path = output_dir / "automatic_mission.csv"
    df.to_csv(csv_path, index=False)
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)
    axes[0, 0].plot(df.time_s, df.altitude_m, label="Altitude")
    axes[0, 0].plot(df.time_s, df.target_altitude_m, "--", label="Target")
    axes[0, 0].set_ylabel("Altitude [m]"); axes[0, 0].legend(); axes[0, 0].grid(True)
    axes[0, 1].plot(df.time_s, df.airspeed_mps, label="Airspeed")
    axes[0, 1].plot(df.time_s, df.target_airspeed_mps, "--", label="Target")
    axes[0, 1].set_ylabel("Airspeed [m/s]"); axes[0, 1].legend(); axes[0, 1].grid(True)
    axes[1, 0].plot(df.time_s, df.roll_deg, label="Roll")
    axes[1, 0].plot(df.time_s, df.pitch_deg, label="Pitch")
    axes[1, 0].plot(df.time_s, df.yaw_deg, label="Yaw")
    axes[1, 0].set_ylabel("Attitude [deg]"); axes[1, 0].legend(); axes[1, 0].grid(True)
    axes[1, 1].plot(df.time_s, df.tilt_deg, label="Nacelle tilt")
    axes[1, 1].plot(df.time_s, df.aileron_deg, label="Aileron")
    axes[1, 1].plot(df.time_s, df.elevator_deg, label="Elevator")
    axes[1, 1].plot(df.time_s, df.rudder_deg, label="Rudder")
    axes[1, 1].set_ylabel("Actuators [deg]"); axes[1, 1].legend(); axes[1, 1].grid(True)
    for i in range(1, 5):
        axes[2, 0].plot(df.time_s, df[f"rotor_{i}_rpm"], label=f"R{i}")
    axes[2, 0].set_ylabel("Rotor speed [rpm]"); axes[2, 0].set_xlabel("Time [s]"); axes[2, 0].legend(); axes[2, 0].grid(True)
    axes[2, 1].plot(df.x_m, df.altitude_m)
    axes[2, 1].set_xlabel("Along-track x [m]"); axes[2, 1].set_ylabel("Altitude [m]"); axes[2, 1].grid(True)
    fig_path = output_dir / "automatic_mission_summary.png"
    fig.savefig(fig_path, dpi=160)
    print(f"Saved {csv_path}")
    print(f"Saved {fig_path}")
    print(df.tail(1).to_string(index=False))
