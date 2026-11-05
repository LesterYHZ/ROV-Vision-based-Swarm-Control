# Underwater target tracking with CBF collision avoidance

This package extends the earlier PID / 3D ALOS closed-loop NED tracking test by inserting a body-FRD Control Barrier Function (CBF) velocity filter between the tracking controller and the controlled vehicle.

## Files

- `test_closed_loop_with_cbf.py` — full simulation, metrics, and plotting.
- `target_tracking.py` — PID and Fossen-inspired 3D ALOS tracking controllers plus target-velocity estimator.
- `cbf_collision.py` — multi-obstacle velocity-level CBF quadratic-program filter.
- `cbf_closed_loop_ned_trajectories.png` — 3D NED trajectories, obstacle bodies, and inflated CBF safety envelopes.
- `cbf_closed_loop_frd_tracking_errors.png` — forward/right/down FRD tracking errors for PID and ALOS, with and without CBF.
- `cbf_safety_diagnostics.png` — minimum obstacle clearance and CBF command correction.
- `cbf_simulation_metrics.csv` — numeric results for each case.

## Run

```bash
pip install -r requirements-c.txt
python -m test_closed_loop_with_cbf
```

For headless runs:

```bash
python test_closed_loop_with_cbf.py --no-show
```

The script runs four cases with identical target/obstacle trajectories:

1. PID without CBF (nominal baseline)
2. PID + CBF
3. 3D ALOS without CBF (nominal baseline)
4. 3D ALOS + CBF

The nominal baselines are included to make the collision-avoidance effect visible; the test is designed so the unfiltered paths intersect the obstacle/fleet-vehicle safety regions.

## Scenario and measurements

- All vehicle translational positions are propagated in NED (North-East-Down).
- The controlled vehicle attitude is fixed in this kinematic demonstration; body FRD and NED are related by a constant rotation matrix.
- The target's true velocity is used only to propagate the simulated target. The controller receives target-relative FRD position and estimates target velocity from successive measurements plus the controlled vehicle's own ground-referenced velocity.
- Obstacle and fleet-vehicle positions are converted into relative body-FRD coordinates. The CBF receives only these relative positions, their radii, persistent IDs, the controlled vehicle's actual velocity, and the sample time. Relative radial motion is estimated from position history.
- One fixed spherical obstacle and one moving fleet vehicle cross the desired vehicle locus.
- The controlled vehicle has a first-order velocity response and acceleration-magnitude limit.

## CBF assumptions and limits

The included CBF is a velocity-level (relative-degree-one) CBF. Its standard condition assumes the optimized velocity is the actual kinematic control input. This simulation retains a first-order velocity lag and acceleration cap, so it adds a 3 m response-distance buffer to the geometric safety margin. That buffer is a conservative simulation setting, not a formal higher-order CBF guarantee. For a real acceleration-limited vehicle, derive and implement a higher-order CBF or predictive safety filter using the actual acceleration dynamics, actuator delay, and sensing latency.

A newly detected obstacle has no finite-difference history. The CBF uses `unknown_closing_speed` as a conservative initial radial closing-rate bound. Stable data-association IDs are important: an ID must keep referring to the same obstacle between samples.

For ALOS, adaptive crab/attack-angle estimates are rolled back on samples where the CBF materially modifies the nominal velocity. This prevents the ALOS adaptation law from learning the safety filter's temporary avoidance maneuver as a persistent environmental drift. This is a practical supervisory/anti-windup integration rule, not part of the original Fossen ALOS stability proof.

The script's PASS/FAIL checks are scenario-specific numerical tests. They do not constitute a formal proof of closed-loop stability or safety for arbitrary trajectories, disturbances, initial conditions, or perception failures.
