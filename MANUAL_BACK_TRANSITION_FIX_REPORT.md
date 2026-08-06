# Manual transition-to-hover correction (v0.3.2)

## Observed problem

Repeatedly pressing **Transition → hover** in the live dashboard did not continue one back-transition command. Each click created a new stop point at `current x + 25 m` and reset the transition start time. Because the nacelle schedule is time-based, resetting the command caused the nacelles to rotate forward again and the aircraft to accelerate. The position-error plot therefore showed repeated sawtooth resets while x continued to increase.

The manual dashboard supervisor also only completed vertical takeoff. It had no rule for changing `TRANSITION_TO_HOVER` into `HOVER`, so a successful deceleration could remain labelled and controlled as transition indefinitely.

## Corrections

1. A back-transition command is now idempotent in the dashboard. While `TRANSITION_TO_HOVER` is active, the button is disabled and any duplicate callback preserves the existing stop point and transition timer.
2. The fixed 25 m target was replaced by a route-centreline stopping point based on both the nominal transition duration and a comfortable kinematic deceleration:

   `d_target = max(d_min, 0.5 * V_forward * T_transition, V_forward^2 / (2 a_comfort)) + d_buffer`

3. The stopping target is projected onto the commanded route centreline, so lateral route errors are not frozen into the hover point.
4. The manual phase supervisor now changes to `HOVER` after horizontal speed, vertical speed and nacelle tilt remain within thresholds for 0.75 s. The fixed stop target is preserved during this phase change.
5. The new logic supports arbitrary route heading, not only motion along global +x.

## Expected dashboard behaviour

- One click starts one back transition.
- The target remains fixed throughout the manoeuvre.
- Nacelle tilt decreases monotonically toward hover orientation instead of jumping back toward 90° after repeated clicks.
- Airspeed converges toward zero.
- The phase changes automatically from `TRANSITION_TO_HOVER` to `HOVER` once settled.
- x may still increase during the physically necessary stopping distance, but it should approach a finite target rather than increase indefinitely.

## Scope

The stopping-point formula is a transparent reduced-order planning rule, not a certified trajectory optimiser. Urban obstacle clearance, gust margins and formal passenger-comfort optimisation remain outside this correction.
