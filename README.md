# TiltRotor_SIM

RotorPy-compatible reduced-order quad tiltrotor simulation with:

- nonlinear 3D six-degree-of-freedom rigid-body dynamics
- four tilting propulsors, motor lag, common nacelle actuator
- wing and tail aerodynamics with aileron, elevator, and rudder
- vertical takeoff/landing, hover, transition, cruise, and back-transition modes
- cascaded position/velocity PID and quaternion attitude PID/PD control
- automatic mission runner and live Bokeh dashboard
- urban wind/gust scenarios and passenger-comfort proxy monitoring

> This is a transparent reduced-order research prototype. It is not a certified or flight-validated aircraft model.

## v0.4 urban wind and passenger comfort

The dashboard now provides two experimental toggles.

### Urban wind / gust

Available scenarios:

- steady three-dimensional wind
- smooth finite-duration discrete gust
- altitude-dependent wind shear
- deterministic multi-frequency urban-wake proxy

Controls include base x/y/z wind, disturbance amplitude and direction, start time, duration, shear gradient and wake frequency. The dashboard plots the three wind components, total wind magnitude and disturbance-only magnitude.

### Passenger comfort

The comfort monitor calculates:

- instantaneous and rolling-RMS acceleration magnitude
- instantaneous and rolling-RMS jerk magnitude
- instantaneous and rolling-RMS body angular-rate magnitude
- peak values since reset
- normalized comfort index
- total time above the selected limits

The optional **Comfort accel guard** limits the controller's maximum commanded acceleration and uses the selected acceleration limit when planning a manual back-transition stopping point. Jerk and angular-rate limits are monitoring thresholds, not hard constraints.

See [`URBAN_WIND_COMFORT_REPORT.md`](URBAN_WIND_COMFORT_REPORT.md) for formulas, suggested A/B experiments and limitations.

## v0.3.2 manual transition-to-hover correction

- Repeated **Transition → hover** clicks no longer restart the nacelle schedule or move the stopping target farther forward.
- The button is disabled while a manual back transition is active.
- The stopping point is placed once on the route centreline using forward speed, transition duration, and a comfortable-deceleration estimate instead of a fixed 25 m offset.
- Manual mode automatically changes from `TRANSITION_TO_HOVER` to `HOVER` after airspeed, vertical speed, and nacelle tilt remain settled.
- The planned hover point is preserved when the phase changes.

See [`MANUAL_BACK_TRANSITION_FIX_REPORT.md`](MANUAL_BACK_TRANSITION_FIX_REPORT.md).

## v0.3 position-hold and guidance corrections

- Vertical climb no longer activates the conventional fixed-wing coefficient model from a spurious `-90°` angle of attack.
- Fuselage drag remains active in every phase, while wing/tail forces and surface authority are blended using positive forward airspeed and nacelle tilt.
- The automatic mission preserves the original takeoff x/y point when changing from vertical takeoff to hover.
- Transition and cruise use cross-track position feedback.
- Back transition preserves its planned landing point when changing to vertical landing.
- Dashboard monitoring includes target x/y, x/y error, cross-track error, and wing-aerodynamic blend.

See [`POSITION_HOLD_FIX_REPORT.md`](POSITION_HOLD_FIX_REPORT.md).

## Windows: first installation

```powershell
git clone https://github.com/mobydickhm1851/TiltRotor_SIM.git
cd TiltRotor_SIM
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\first_setup.ps1
```

The script creates `.venv` only when it does not already exist, installs the project in editable mode, and runs the tests.

## Start the dashboard

```powershell
.\scripts\run_dashboard.ps1
```

Open `http://127.0.0.1:8050/` when the browser does not open automatically.

The **Run automatic mission** button uses the same `TiltrotorSimulation`, controller, parameters, and `AutomaticMission` state machine as `examples/run_mission.py`.

## Suggested wind/comfort comparison

1. Reset and run the automatic mission with **Urban wind / gust OFF** and **Comfort accel guard OFF**.
2. Reset, enable a wind scenario, and repeat with the same mission settings.
3. Reset and repeat the identical disturbed mission with **Comfort accel guard ON**.
4. Compare cross-track error, altitude error, roll/pitch, acceleration, jerk, angular rate, comfort index and exceedance time.

## Run the automatic mission

```powershell
.\scripts\run_mission.ps1
```

Results are written into `output/`. Generated CSV and PNG files are ignored by Git.

## Update without rebuilding `.venv`

```powershell
.\scripts\update_project.ps1
```

This performs:

```text
git pull --ff-only
pip install -e ".[dev]"
pytest
```

The existing `.venv` remains in place. Because the package is installed with `-e`, ordinary source-code changes become active immediately after `git pull`; dependencies are only downloaded when `pyproject.toml` changes.

## Online execution with GitHub Codespaces

Open the repository in GitHub, select **Code → Codespaces → Create codespace**, then run:

```bash
python examples/run_dashboard.py
```

Forward port `8050` from the Codespaces Ports panel.

## Main package layout

```text
src/rotorpy_tiltrotor/
├── parameters.py     aircraft and actuator parameters
├── aerodynamics.py   reduced-order blended wing/tail wrench
├── allocator.py      rotor thrust/moment allocation
├── commands.py       flight phases and mission state machine
├── controller.py     cascaded PID, route guidance and authority blending
├── scenarios.py      wind/gust generators and comfort metrics
├── vehicle.py        nonlinear dynamics and actuator states
├── simulator.py      batch/real-time simulation and logging
└── dashboard.py      live Bokeh UI
```

## Current limitations

Not yet included: dynamic inflow, blade flapping, rotor-wing interference, post-stall lookup tables, ground effect, building-resolved CFD wakes, obstacle avoidance, sensor fusion, certified acoustic prediction, ISO 2631 processing, fault-tolerant control, or flight-test validation.
