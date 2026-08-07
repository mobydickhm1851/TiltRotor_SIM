# v0.4.2 Continuous-Turbulence and Comfort-Jerk Correction

## Why the v0.4.1 turbulence trace looked wrong

The v0.4.1 `Continuous turbulence` option used a three-axis Ornstein-Uhlenbeck proxy. With the generic 5 m/s disturbance setting it produced large simultaneous x/y/z wind variations with a short correlation time. During low-speed transition this drove rapid changes in relative airspeed, angle of attack and sideslip, pushed the reduced-order aerodynamics/controller outside their nominal envelope, saturated actuators and could produce loss of altitude and attitude.

That process was a generic stochastic stress test, not a good aviation-regulation representation.

## Aviation-regulation references are not all the same turbulence model

### FAR / CS 25.341 — transport-airplane structural gust loads

14 CFR 25.341(b) and EASA CS 25.341 require the dynamic response to vertical and lateral continuous turbulence. AMC 25.341 describes a one-dimensional Gaussian random atmosphere with a Von Karman power spectral density and turbulence scale length L = 2500 ft = 762 m. The regulatory U_sigma values are structural limit-load design quantities. For nonlinear time-domain analysis, AMC guidance uses a long Gaussian pseudo-random stream conforming to the Von Karman spectrum with RMS amplitude 0.4 U_sigma.

Near sea level U_sigma_ref is approximately 27.43 m/s before the flight-profile factor. Those numbers are intentionally severe structural-design criteria and must not be misread as routine urban-operation RMS wind for a 25-kg eVTOL.

### EASA CS-AWO — low-altitude / approach wind and turbulence

For the project's roughly 30-m urban/vertiport operating region, CS-AWO provides a more useful operational benchmark. Its low-altitude turbulence model treats the turbulence components as Gaussian processes with Von-Karman-form spectra and uses:

- RMS turbulence intensity: sigma = 0.15 U
- scale length: L = 183 m (600 ft)

where U is the reference mean wind speed.

The v0.4.2 dashboard therefore uses the CS-AWO-style relationship for its default continuous-turbulence operational test rather than Part/CS-25 structural limit intensities.

## v0.4.2 dashboard semantics

The option is labelled:

`Continuous turbulence (CS-AWO low-altitude style)`

The user input is:

`Reference mean wind U [m/s] (sigma = 0.15 U)`

For example, U = 5 m/s gives sigma = 0.75 m/s. This is very different from the old interpretation in which the same numeric value 5 could act like a multi-axis turbulence standard deviation of several m/s.

The simulator uses a deterministic finite Fourier realization of a normalized Von-Karman-type spectrum, a short smooth fade-in, L = 183 m and a 15 m/s reference airspeed. The current default component is vertical so transition altitude/lift robustness can be studied without falsely presenting a simultaneous arbitrary 3-D noise field as the regulation itself. Repeated gusts and the urban-wake proxy remain available for non-regulatory multi-axis disturbance studies.

The FAA operational PIREP terms Light / Moderate / Severe / Extreme are not fixed turbulence velocities in m/s. They are defined by aircraft and occupant reactions, so the dashboard intentionally does not convert sigma directly into those labels.

## Why the comfort guard could still exceed 1.5 m/s^3

v0.4.1 used the passenger-facing jerk target as the commanded acceleration slew-rate limit. Aircraft output is not identical to command because the closed-loop plant contains motor lag, nonlinear rotor thrust, ground-contact release, attitude coupling, nacelle dynamics and aerodynamic response. Therefore a 1.50 m/s^3 command slew limit could still produce a measured kinematic peak around or above 1.50 m/s^3.

## v0.4.2 jerk-guard correction

The selected dashboard value remains the measured passenger-motion target:

`Measured jerk target = 1.50 m/s^3`.

The controller now uses only 60% of that target internally:

j_cmd,max = 0.60 j_target

so the default internal slew limit is 0.90 m/s^3. The remaining 40% is plant-response headroom. Actual kinematic jerk remains the KPI and is not clipped in the plot. Strong external turbulence can still exceed the passenger target; a command governor cannot physically guarantee ride comfort against arbitrarily large atmospheric disturbances.

## Sources

- 14 CFR 25.341, Gust and turbulence loads: https://www.ecfr.gov/current/title-14/chapter-I/subchapter-C/part-25/subpart-C/section-25.341
- FAA AC 25.341-1, Dynamic Gust Loads: https://www.faa.gov/regulations_policies/advisory_circulars/index.cfm/go/document.information/documentID/1024906
- EASA Easy Access Rules for Large Aeroplanes, CS 25.341 / AMC 25.341: https://www.easa.europa.eu/en/document-library/easy-access-rules/online-publications/easy-access-rules-large-aeroplanes-cs-25?page=15
- EASA Easy Access Rules for All-Weather Operations, low-altitude turbulence model: https://www.easa.europa.eu/en/document-library/easy-access-rules/online-publications/easy-access-rules-all-weather-operations-cs?page=24
- FAA AIM, PIREP turbulence reporting criteria: https://www.faa.gov/air_traffic/publications/atpubs/aim_html/chap7_section_1.html
