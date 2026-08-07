# v0.4.2 Continuous-Turbulence and Comfort-Jerk Correction

## 1. Why the v0.4.1 turbulence trace looked wrong

The v0.4.1 `Continuous turbulence` option used a three-axis Ornstein-Uhlenbeck (OU) proxy. With the dashboard default disturbance amplitude of 5 m/s, the process created large simultaneous x/y/z wind variations with a short correlation time. During transition, the low-speed tiltrotor then experienced rapid changes in relative airspeed, angle of attack and sideslip. The reduced-order wing model and flight controller were driven far outside the nominal transition envelope, producing control-surface/rotor saturation and eventual loss of altitude and attitude.

That model was useful as a generic stochastic disturbance but it should not have been labelled as aviation-regulation-style continuous turbulence.

## 2. What FAR/CS 25.341 actually defines

14 CFR 25.341(b) and EASA CS 25.341 define continuous turbulence for transport-airplane gust-load analysis. The important modelling assumptions are:

- vertical and lateral continuous turbulence responses are considered;
- the random atmosphere is represented by a Gaussian distribution of gust velocity;
- the normalized power spectral density is the Von Karman spectrum;
- turbulence scale length is L = 2500 ft = 762 m;
- the regulatory intensity U_sigma is a structural limit-load design quantity, not a normal operational wind-speed setting;
- FAA AC 25.341-1 / EASA AMC 25.341 describe nonlinear time-domain stochastic simulation using a long Gaussian pseudo-random stream conforming to the Von Karman spectrum.

For 14 CFR 25.341(b)(5), when required for nonlinear load assessment, the turbulence field RMS velocity is 40% of U_sigma. Near sea level U_sigma_ref itself is approximately 27.43 m/s before the flight-profile factor. Those values are intentionally severe structural-design criteria and are not appropriate as the default everyday urban-turbulence setting for this 25-kg reduced-order eVTOL controller demonstration.

The FAA operational PIREP categories Light / Moderate / Severe / Extreme are also not fixed turbulence-velocity values. They are defined by the reaction of the aircraft and occupants. Therefore this project does not map a single sigma in m/s directly to an FAA `Moderate turbulence` label.

## 3. v0.4.2 continuous-turbulence benchmark

The dashboard option is renamed:

`Continuous turbulence (FAR/CS 25.341-style)`

It uses a finite Fourier synthesis of the normalized one-dimensional Von Karman spectrum

Phi(Omega) = L/pi * [1 + (8/3)(1.339 L Omega)^2] / [1 + (1.339 L Omega)^2]^(11/6)

with

- L = 762 m;
- Omega = omega / V_ref;
- V_ref = 15 m/s for the dashboard benchmark;
- a deterministic random phase set for repeatable A/B tests;
- a short smooth fade-in to avoid an artificial wind step when the toggle is enabled.

The generated unit process is normalized and multiplied by the user-entered

`Turbulence RMS sigma [m/s]`.

The default is 1.0 m/s. This is an engineering controller-test level, not a certification threshold.

The current dashboard benchmark applies the turbulence in the vertical direction. This matches one of the one-dimensional FAR/CS continuous-turbulence load cases and avoids falsely presenting a simultaneous three-axis random field as the regulation itself. Repeated gusts and the urban-wake proxy remain available for non-regulatory urban-disturbance studies.

## 4. Why the comfort guard could still exceed 1.5 m/s^3

The v0.4.1 guard limited the *commanded acceleration slew rate* to the same numerical value shown as the measured passenger jerk target. The aircraft output does not equal the command exactly because the closed-loop plant contains:

- motor first-order dynamics;
- rotor thrust nonlinearities;
- ground-contact release at lift-off;
- attitude-loop coupling;
- nacelle/actuator dynamics;
- nonlinear aerodynamic forces.

Therefore a 1.50 m/s^3 command slew limit could still produce a measured kinematic peak of roughly 1.5-1.8 m/s^3 in nominal takeoff. This is not a contradiction: command jerk and aircraft jerk are different signals.

## 5. v0.4.2 jerk-guard correction

The selected dashboard value remains the passenger-facing measured target:

`Measured jerk target = 1.50 m/s^3`.

Internally the controller uses only 60% of that value as the command acceleration slew rate:

j_cmd,max = 0.60 * j_target.

For the default target this gives

j_cmd,max = 0.90 m/s^3.

The 40% headroom is an engineering robustness margin for the reduced-order plant. The actual kinematic jerk is still calculated from integrated velocity and remains the output KPI. Strong external gusts can still make measured jerk exceed the target because a comfort command guard cannot physically cancel arbitrary atmospheric disturbances.

## 6. Interpretation for the assignment

Use two separate statements in the report:

1. **Regulatory-inspired turbulence benchmark**: continuous turbulence is modelled as a one-dimensional Gaussian Von Karman process based on FAR/CS 25.341 assumptions. The selected operational RMS sigma is intentionally much smaller than certification structural-load U_sigma values.
2. **Passenger comfort guard**: acceleration and jerk commands are shaped conservatively, while the effectiveness of the guard is evaluated using measured kinematic peak jerk, peak acceleration and ISO-style weighted RMS acceleration.

## Sources

- 14 CFR 25.341, Gust and turbulence loads: https://www.ecfr.gov/current/title-14/chapter-I/subchapter-C/part-25/subpart-C/section-25.341
- FAA AC 25.341-1, Dynamic Gust Loads: https://www.faa.gov/regulations_policies/advisory_circulars/index.cfm/go/document.information/documentID/1024906
- EASA Easy Access Rules for Large Aeroplanes, CS 25.341 / AMC 25.341: https://www.easa.europa.eu/en/document-library/easy-access-rules/easy-access-rules-large-aeroplanes-cs-25
- FAA AIM, PIREPs Relating to Turbulence / Turbulence Reporting Criteria: https://www.faa.gov/air_traffic/publications/atpubs/aim_html/chap7_section_1.html
