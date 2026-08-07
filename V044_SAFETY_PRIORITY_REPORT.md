# v0.4.4 Safety-Priority Transition / Comfort Update

## Problem reproduced from the dashboard

Two failures were addressed:

1. With repeated gusts set to 10 m/s, direction 90 deg, delay 2 s, duration 3 s, 0.25 Hz, and base vertical wind -2.25 m/s, the automatic mission could lose almost the full 30-m operating altitude during cruise/back transition.
2. With no wind but Comfort Guard enabled at 0.50 m/s^2 acceleration and 1.50 m/s^3 jerk targets, hover-to-cruise conversion could also lose almost the full operating altitude.

The second behaviour confirms that the earlier comfort governor was not merely smoothing the manoeuvre: it was removing safety-critical vertical control authority.

## Root causes

### Comfort target was used as total aircraft acceleration authority

The base dashboard set `controller.max_accel` directly to the selected comfort acceleration when the guard was ON. At 0.50 m/s^2 this reduced the controller's previous 4.0 m/s^2 authority in all axes, including altitude recovery.

At the same time, the old forward-transition tilt schedule continued to advance mainly with elapsed time. Therefore horizontal acceleration could be deliberately slowed by the comfort governor while the nacelles still moved toward 90 deg on the nominal schedule. Rotor vertical support disappeared before enough forward airspeed / wing lift had been established.

### Automatic back transition still used the old shortcut

v0.4.3 improved the manual dashboard button, but `AutomaticMission` still used a fixed point only 35 m ahead and the old default 14-s transition duration. The automatic mission therefore did not receive the safer 24-s / speed-based stopping-point logic.

### Comfort is a soft requirement; altitude is a safety requirement

A passenger comfort target must not command an unsafe reduction in flight-control authority. The v0.4.4 arbitration order is:

1. altitude / flight safety;
2. attitude stability;
3. path and speed tracking;
4. passenger comfort.

A comfort exceedance during a safety recovery is intentionally reported rather than hidden or clipped.

## v0.4.4 controller changes

### Hierarchical acceleration governor

When Comfort Guard is ON, horizontal acceleration remains limited by the user-selected comfort target. Vertical acceleration is also comfort-shaped while altitude is healthy. If altitude deficit or downward velocity grows, vertical control authority is progressively released back toward the normal 4.0 m/s^2 flight-control envelope.

The safety blend starts at approximately 0.20 m altitude deficit or 0.25 m/s downward speed and reaches full authority by approximately 1.50 m deficit or 1.20 m/s downward speed.

### Hierarchical jerk governor

Nominal acceleration-command slew continues to use the existing passenger-jerk headroom. Under altitude recovery, only the vertical jerk channel is progressively allowed to exceed the comfort command target, up to an emergency internal vertical slew rate. The measured aircraft jerk remains unmodified in the plots and can therefore show a comfort violation during a safety recovery.

### Airspeed-gated forward conversion

`TRANSITION_TO_CRUISE` no longer forces nacelles toward 90 deg simply because the nominal transition clock expired. Conversion progress is capped by forward air-relative speed. If the aircraft is accelerating slowly because Comfort Guard is active, the rotors stay more vertical until enough wing-supporting airspeed is established.

If altitude begins to decay, forward-conversion progress is reduced, returning more rotor thrust toward vertical support.

### Cruise lift-assist override

If a meaningful altitude safety factor occurs in cruise, nacelle command temporarily moves away from 90 deg toward vertical lift. Speed / propulsive efficiency is sacrificed before altitude.

### Back-transition altitude override

During `TRANSITION_TO_HOVER`, altitude loss accelerates the nacelle return toward vertical rotor lift. The v0.4.3 world-z collective floor remains active and is extended to forward transition / cruise whenever the rotor axis has useful vertical effectiveness.

## Automatic mission changes

Automatic back transition now:

- uses a 24-s baseline;
- computes the stopping point from current forward speed, transition duration, and a 0.75 m/s^2 nominal deceleration instead of fixed +35 m;
- returns to a true `HOVER` at the target altitude after conversion;
- requires a stable post-transition hover before commanding `VERTICAL_LANDING`;
- does not enter `CRUISE` while altitude is more than 1.5 m below target.

This separates an actual hover from the subsequent intentional descent to landing and makes altitude plots easier to interpret.

## UI changes

Bokeh input titles are kept short (`Delay [s]`, `Duration [s]`, `Frequency [Hz]`, etc.). Long mode descriptions and safety semantics are rendered in stretch-width `Div` elements using:

- `white-space: normal`;
- `overflow-wrap: anywhere`;
- `word-break: break-word`.

This provides reliable browser-width wrapping instead of allowing long labels to overlap adjacent fixed-width controls. A live safety-priority status block shows whether altitude override is active.

## Regression cases

The v0.4.4 tests include:

1. the reported severe repeated-gust setup: 10 m/s, 90 deg, 2-s delay, 3-s duration, 0.25 Hz, and base wind z=-2.25 m/s; the automatic mission must reach cruise and back transition without touching the ground during non-landing flight;
2. no-wind Comfort Guard at 0.50 m/s^2 / 1.50 m/s^3; hover-to-cruise must retain altitude while eventually establishing cruise;
3. elapsed-time transition with low airspeed must keep nacelle tilt low instead of forcing 90 deg;
4. an explicit altitude-loss state must trigger safety override, vertical acceleration above the comfort target, and cruise nacelle lift assist;
5. dashboard descriptions must use wrapping blocks and compact widget titles.

These remain reduced-order engineering validation cases, not certification evidence.
