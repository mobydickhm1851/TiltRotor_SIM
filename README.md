# TiltRotor_SIM

RotorPy-compatible reduced-order quad tiltrotor simulation with:

- nonlinear 3D six-degree-of-freedom rigid-body dynamics
- four tilting propulsors, motor lag, common nacelle actuator
- wing and tail aerodynamics with aileron, elevator, and rudder
- vertical takeoff/landing, hover, transition, cruise, and back-transition modes
- cascaded position/velocity PID and quaternion attitude PID/PD control
- automatic mission runner and live Bokeh dashboard
- distinct plot colors and line styles

> This is a transparent reduced-order research prototype. It is not a certified or flight-validated aircraft model.

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

## Manual commands

The live dashboard exposes:

- Vertical takeoff
- Transition to cruise
- Cruise
- Transition to hover
- Vertical landing
- Automatic full mission

Manual takeoff automatically changes to hover after altitude and vertical speed remain within the settling thresholds.

## Online execution with GitHub Codespaces

Open the repository in GitHub, select **Code → Codespaces → Create codespace**, then run:

```bash
python examples/run_dashboard.py
```

Forward port `8050` from the Codespaces Ports panel. The included devcontainer installs the editable project automatically.

## Main package layout

```text
src/rotorpy_tiltrotor/
├── parameters.py     aircraft and actuator parameters
├── aerodynamics.py   reduced-order wing/tail wrench
├── allocator.py      rotor thrust/moment allocation
├── commands.py       flight phases and mission state machine
├── controller.py     cascaded PID and authority blending
├── vehicle.py        nonlinear dynamics and actuator states
├── simulator.py      batch/real-time simulation and logging
└── dashboard.py      live Bokeh UI
```

## Current limitations

Not yet included: dynamic inflow, blade flapping, rotor-wing interference, post-stall lookup tables, ground effect, urban building wakes, obstacle avoidance, sensor fusion, or certified noise/comfort models.
