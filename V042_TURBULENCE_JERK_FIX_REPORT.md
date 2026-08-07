# v0.4.2 Continuous-Turbulence and Comfort-Jerk Correction

## Why the v0.4.1 turbulence trace looked wrong

The v0.4.1 `Continuous turbulence` option used a three-axis Ornstein-Uhlenbeck proxy. With the generic 5 m/s disturbance setting it produced large simultaneous x/y/z wind variations with a short correlation time. During low-speed transition this drove rapid changes in relative airspeed, angle of attack and sideslip, pushed the reduced-order aerodynamics/controller outside their nominal envelope, saturated actuators and could produce loss of altitude and attitude.

That process was a generic stochastic stress test, not a good aviation-regulation representation.

## Aviation-regulation references are not all the same turbulence model

### FAR / CS 25.341 — transport-airplane structural gust loads

14 CFR 25.341(b) and EASA CS 25.341 require dynamic response to vertical and lateral continuous turbulence. AMC 25.341 describes a one-dimensional Gaussian random atmosphere with a Von Karman power spectral density and turbulence scale length L = 2500 ft = 762 m. The regulatory U_sigma values are structural limit-load design quantities. For nonlinear time-domain analysis, AMC guidance uses a long Gaussian pseudo-random stream conforming to the Von Karman spectrum with RMS amplitude 0.4 U_sigma.

Near sea level U_sigma_ref is approximately 27.43 m/s before the flight-profile factor. These are severe structural-design criteria and must not be interpreted as routine urban-operation RMS turbulence for a 25-kg eVTOL.

### EASA CS-AWO — low-altitude / approach wind and turbulence

CS-AWO is more useful for the project's roughly 30-m vertiport/approach operating region, but its horizontal and vertical component parameters must be kept separate.

For the horizontal longitudinal/lateral turbulence components, CS-AWO wind model No. 1 permits Gaussian processes with:

- sigma_u = sigma_v = 0.15 U
- L_u = L_v = 183 m (600 ft)

For the vertical component, the same section permits the same spectral form with an alternative low-altitude parameterization:

- sigma_w = 0.09 U
- L_w = 4.6 m (15 ft) below 9.2 m altitude
- L_w = 0.5 z for 9.2 < z < 305 m

Therefore at the project's nominal z = 30 m target altitude:

- sigma_w = 0.09 U
- L_w = 15 m

The earlier intermediate v0.4.2 implementation incorrectly reused the horizontal 0.15U / 183-m values for a vertical-only benchmark. That has been corrected before merge.

## v0.4.2 dashboard semantics

The option is labelled:

`Continuous turbulence (CS-AWO vertical low-altitude style)`

The user input is:

`Reference mean wind U [m/s] (vertical sigma_w = 0.09 U)`

For example, U = 5 m/s gives sigma_w = 0.45 m/s at the nominal 30-m operating altitude, and the vertical spectrum uses L_w = 15 m. This is very different from v0.4.1, where the generic value 5 drove a short-correlation process with multi-axis standard deviations of several m/s.

The simulator uses deterministic finite Fourier synthesis of a normalized Von-Karman-form vertical spectrum and a smooth fade-in. The current synthesis is limited to the rigid-body frequency range relevant to this 6-DOF model. Repeated gusts and the urban-wake proxy remain available for deliberately multi-axis urban disturbance studies.

The FAA operational PIREP terms Light / Moderate / Severe / Extreme are not fixed turbulence velocities in m/s. They are defined from aircraft and occupant reaction, so the dashboard intentionally does not convert sigma directly into those operational labels.

## Why the comfort guard could still exceed 1.5 m/s^3

v0.4.1 used the passenger-facing jerk target as the commanded acceleration slew-rate limit. Aircraft output is not identical to command because the closed-loop plant contains motor lag, nonlinear rotor thrust, ground-contact release, attitude coupling, nacelle dynamics and aerodynamic response. Therefore a 1.50 m/s^3 command slew limit could still produce a measured kinematic peak around or above 1.50 m/s^3.

## v0.4.2 jerk-guard correction

The selected dashboard value remains the measured passenger-motion target:

`Measured jerk target = 1.50 m/s^3`.

The controller uses only 60% of that target internally:

j_cmd,max = 0.60 j_target

so the default internal slew limit is 0.90 m/s^3. The remaining 40% is plant-response headroom. Actual kinematic jerk remains the output KPI and is not clipped in the plot. Strong external turbulence can still exceed the passenger target; a command governor cannot physically guarantee ride comfort against arbitrarily large atmospheric disturbances.

## Validation logic

Regression tests separately verify:

1. the CS-AWO vertical turbulence has the expected RMS scaling and strongly correlated adjacent samples;
2. nominal vertical takeoff with comfort guard checks the *measured aircraft jerk*, not merely command jerk;
3. the transition/cruise interval remains finite and altitude-stable under the default U=5 m/s continuous-turbulence benchmark;
4. normal later landing is excluded from the transition-crash metric.

## Sources

- 14 CFR 25.341, Gust and turbulence loads: https://www.ecfr.gov/current/title-14/chapter-I/subchapter-C/part-25/subpart-C/section-25.341
- FAA AC 25.341-1, Dynamic Gust Loads: https://www.faa.gov/regulations_policies/advisory_circulars/index.cfm/go/document.information/documentID/1024906
- EASA Easy Access Rules for Large Aeroplanes, CS 25.341 / AMC 25.341: https://www.easa.europa.eu/en/document-library/easy-access-rules/online-publications/easy-access-rules-large-aeroplanes-cs-25?page=15
- EASA Easy Access Rules for All-Weather Operations, wind/turbulence models: https://www.easa.europa.eu/en/document-library/easy-access-rules/online-publications/easy-access-rules-all-weather-operations-cs?page=24
- FAA AIM, PIREP turbulence reporting criteria: https://www.faa.gov/air_traffic/publications/atpubs/aim_html/chap7_section_1.html
