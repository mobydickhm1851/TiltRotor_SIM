# Urban wind and passenger-comfort scenarios (v0.4.0)

## Purpose

This update adds two repeatable experiment dimensions requested for the urban-VTOL assignment:

1. **Urban wind / gust disturbance rejection**
2. **Passenger-comfort monitoring and optional acceleration guarding**

The implementation is intentionally reduced order and transparent. It supports controller comparison and KPI generation; it is not a certified atmospheric model, building-flow CFD model, ISO 2631 implementation, or medical comfort model.

## Dashboard wind controls

The **Urban wind / gust** toggle enables one of four modes:

- **Steady wind** — constant three-dimensional wind vector.
- **Discrete gust** — smooth finite-duration gust with zero-slope entry and exit.
- **Wind shear** — disturbance changes with altitude relative to the commanded altitude.
- **Urban wake proxy** — deterministic multi-frequency longitudinal, lateral and vertical disturbance inside a finite time window.

User-settable parameters include base x/y/z wind, disturbance amplitude and direction, start time, duration, shear gradient and wake frequency.

The wind plot displays x/y/z wind, total magnitude and disturbance-only magnitude. Wind is passed into the existing vehicle model as world-frame air velocity, so it changes relative airspeed, angle of attack, sideslip, aerodynamic forces and the controller response.

## Passenger-comfort proxy

The monitor uses quantities already produced by the simulation:

- world-frame translational acceleration magnitude
- jerk magnitude
- body angular-rate magnitude

For each quantity it reports the current value, rolling RMS value and peak since reset. The dashboard also reports total time above any selected limit.

The normalized instantaneous comfort index is

`max(|a| / a_limit, |jerk| / jerk_limit, |omega| / omega_limit)`.

An index above 1 indicates that at least one user-selected threshold is exceeded. A second index uses rolling RMS values over the selected window.

These limits are experiment settings, not regulatory limits.

## Comfort acceleration guard

The **Comfort accel guard** optionally reduces the controller's existing maximum commanded-acceleration norm to the selected acceleration limit. It also uses the lower selected deceleration when planning a manual transition-to-hover stopping point. This creates a direct A/B comparison:

- same mission and disturbance with guard OFF
- same mission and disturbance with guard ON

The guard does not directly enforce jerk or angular-rate limits. Those remain monitored KPIs because actuator dynamics, aerodynamic transients and phase switching can produce values not captured by a simple acceleration-command clamp.

## Suggested experiments

### Gust-rejection experiment

1. Run the automatic mission with urban wind disabled.
2. Reset.
3. Enable **Discrete gust**, set direction to 90 degrees, amplitude to 5 m/s, start time near cruise entry and duration to 8 s.
4. Compare maximum cross-track error, altitude error, roll angle, recovery time and comfort metrics.

### Comfort-guard experiment

1. Use the same wind scenario and mission settings.
2. Run once with the acceleration guard OFF.
3. Reset and run with the guard ON.
4. Compare peak/RMS acceleration, peak/RMS jerk, angular rate, exceedance time, mission duration, tracking error and transition altitude loss.

## Limitations

- The urban-wake mode is a deterministic disturbance proxy, not a building-resolved CFD field.
- Wind is spatially uniform over the aircraft at each instant.
- Passenger comfort does not include seat transmissibility, frequency weighting, motion-sickness dose, occupant orientation or ISO 2631 processing.
- The acceleration guard is a command limiter, not a full jerk-limited trajectory generator or model-predictive controller.
- Thresholds must be justified in the final report and should be treated as study parameters.
