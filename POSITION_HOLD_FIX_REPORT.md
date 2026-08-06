# Position-hold and route-guidance correction

## Problems corrected

1. Vertical climb velocity was previously passed into the conventional wing model as approximately `-90 deg` angle of attack. This created artificial wing lift and pitching moment during rotor-borne flight.
2. The automatic mission replaced the original takeoff x/y target with the aircraft's drifted position when changing from takeoff to hover.
3. Transition and cruise damped lateral velocity but did not explicitly return the aircraft to the route centreline.

## Changes

- The fuselage drag model remains active in every flight mode.
- Fixed-wing lift, drag, side force and aerodynamic moments are blended using positive forward airspeed and nacelle tilt.
- Effective wing dynamic pressure and control-surface authority are zero during vertical climb/hover.
- `AutomaticMission` stores the initial takeoff coordinates and uses them for takeoff, hover and cruise route anchoring.
- Cruise and transition now use cross-track position error to generate lateral speed and acceleration commands.
- Back transition retains the planned landing point when switching into vertical landing.
- Dashboard logging now exposes target x/y, x/y error, cross-track error and wing-blend ratio.

## Validation

Local regression tests: `10 passed`.

Nominal automatic mission results after the correction:

- mission completed normally
- vertical takeoff/hover maximum horizontal displacement in zero wind: approximately numerical zero
- vertical-flight angle of attack display: `0 deg` rather than `-90 deg`
- vertical-flight wing blend: `0`
- back-transition minimum altitude: approximately `29.27 m` for a `30 m` command
- transition/cruise cross-track error in the zero-wind nominal mission: approximately numerical zero

## Scope

This is still a reduced-order model. The blending law avoids applying an invalid fixed-wing coefficient model in hover, but it is not a substitute for rotor-wing interference, download, propwash, dynamic inflow or wind-tunnel data.
