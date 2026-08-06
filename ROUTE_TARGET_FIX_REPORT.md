# Route-target reporting correction (v0.3.1)

## Problem

In transition-to-cruise and cruise, `hold_x_m` and `hold_y_m` define a fixed point on the route centreline. The v0.3 dashboard incorrectly displayed that fixed route anchor as the instantaneous x/y target. As the aircraft travelled along the route, the plotted x error therefore accumulated travelled distance even though the controller was correctly using airspeed along the route and cross-track position normal to the route.

## Correction

The command geometry is now separated into:

- **Route anchor:** fixed point defining the route centreline together with heading.
- **Active route target:** closest point on that centreline to the aircraft's current position.
- **Fixed point target:** still used for takeoff, hover, back transition, and landing.

For route-following phases, the active target is

`p_target = p_anchor + dot(p - p_anchor, forward) * forward`.

The displayed x/y error is `p_target - p`. It therefore contains only route-normal displacement and no longer treats forward progress as a position error. Along-route motion remains evaluated using airspeed error and the new `along_track_distance_m` output.

## Dashboard changes

- The top view draws the moving route-projection target separately from the fixed route anchor.
- Position-tracking labels explicitly distinguish point hold from route projection.
- Status text reports along-route distance and whether the active reference is a fixed point or route projection.

## Scope

This corrects target/error semantics and visualization. It does not change the existing cruise control law: longitudinal motion is still airspeed-controlled, while lateral motion is cross-track-position-controlled.
