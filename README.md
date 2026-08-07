# TiltRotor_SIM

RotorPy-compatible reduced-order quad tiltrotor simulation with:

- nonlinear 3D six-degree-of-freedom rigid-body dynamics
- four tilting propulsors, motor lag, common nacelle actuator
- wing and tail aerodynamics with aileron, elevator, and rudder
- vertical takeoff/landing, hover, transition, cruise, and back-transition modes
- cascaded position/velocity PID and quaternion attitude PID/PD control
- automatic mission runner and live Bokeh dashboard
- urban wind/gust/turbulence scenarios and passenger-comfort monitoring

> This is a transparent reduced-order research prototype. It is not a certified or flight-validated aircraft model.

## v0.4.1 gust timing, repeated disturbances and comfort metrics

### Urban wind / gust

The dashboard now distinguishes six disturbance cases:

- steady three-dimensional wind
- **single discrete gust** for one standardized finite encounter
- **repeated gusts** with recurring, non-identical events
- **continuous turbulence** using a seeded correlated stochastic proxy
- altitude-dependent wind shear
- deterministic multi-frequency urban-wake proxy

The old `Start time` control is now shown as **Delay after scenario enable [s]**. This fixes the v0.4.0 behavior where enabling a gust after its absolute simulation-time window had already passed showed only the base wind.

Repeated-gust and continuous-turbulence modes are intended for urban disturbance-rejection experiments. They are reduced-order proxies, not certified Dryden/Von-Karman turbulence or building-resolved CFD.

### Passenger comfort

The dashboard now separates two types of KPI:

1. **Transient UAM maneuver metrics**: acceleration, jerk and body-rate diagnostics, with NASA-informed default engineering targets of 0.50 m/s² and 1.50 m/s³ for acceleration and jerk. These are not FAA/EASA certification limits.
2. **Sustained vibration metric**: ISO 2631-style Wd/Wk frequency-weighted rolling RMS acceleration, compared with the 0.315 m/s² `not uncomfortable` comfort-band boundary.

Acceleration and jerk are now calculated kinematically from the actual integrated velocity after the ground constraint is applied. This removes the artificial takeoff jerk spike caused by differentiating an unconstrained force-model acceleration while the aircraft was still constrained to the ground.

The optional **Comfort accel + jerk guard** now:

- applies the selected acceleration limit in every flight phase, including vertical takeoff/landing;
- slew-limits the commanded acceleration using the selected jerk target;
- still leaves measured vehicle acceleration/jerk visible as output KPIs, because gusts, actuator dynamics and phase transitions can make actual motion differ from the command.

The ISO filter is an engineering implementation using the ISO Wd/Wk transfer-function structure and bilinear discretization. The simulation runs at 100 Hz, so it is explicitly labelled **ISO 2631-style** rather than ISO 8041 compliant.

See [`V041_GUST_COMFORT_REPORT.md`](V041_GUST_COMFORT_REPORT.md) for rationale, formulas and literature sources. The original v0.4 report remains in [`URBAN_WIND_COMFORT_REPORT.md`](URBAN_WIND_COMFORT_REPORT.md).

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

1. Reset and run the automatic mission with **Urban wind / gust OFF** and **Comfort accel + jerk guard OFF**.
2. Reset, enable **Repeated gusts** or **Continuous turbulence**, and repeat with the same mission settings.
3. Reset and repeat the identical disturbed mission with **Comfort accel + jerk guard ON**.
4. Compare cross-track error, altitude error, roll/pitch, transient acceleration/jerk, ISO-style weighted RMS acceleration, body rate, comfort indices and exceedance time.
5. Use **Single discrete gust** when a repeatable one-event disturbance-response test is desired.

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
├── parameters.py          aircraft and actuator parameters
├── aerodynamics.py        reduced-order blended wing/tail wrench
├── allocator.py           rotor thrust/moment allocation
├── commands.py            flight phases and mission state machine
├── controller.py          cascaded PID, route guidance and authority blending
├── scenarios.py           wind/gust/turbulence and comfort metrics
├── vehicle.py             nonlinear dynamics and actuator states
├── simulator.py           batch/real-time simulation and logging
├── dashboard.py           established live Bokeh UI
└── enhanced_dashboard.py  v0.4.1 disturbance/comfort wrapper
```

## Current limitations

Not yet included: dynamic inflow, blade flapping, rotor-wing interference, post-stall lookup tables, ground effect, building-resolved CFD wakes, obstacle avoidance, sensor fusion, certified acoustic prediction, ISO 8041-compliant vibration instrumentation, fault-tolerant control, or flight-test validation.
