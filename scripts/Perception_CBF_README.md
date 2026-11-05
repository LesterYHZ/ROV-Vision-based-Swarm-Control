# Heading-forward underwater target tracking with camera-FOV CBF

This package updates the prior camera-FOV perception-safety simulation so the
controlled vehicle performs **heading-forward (surge-dominant) motion**. The
body +x axis and camera optical axis remain aligned with the vehicle's travel
direction. The vehicle turns in yaw and pitch toward the course implied by the
filtered velocity command instead of translating sideways to follow a lateral
velocity command.

## Files

- `test_heading_forward_fov.py`: closed-loop two-vehicle NED simulation. It
  compares PID and ALOS tracking, with and without the camera-FOV CBF.
- `cbf_perception.py`: camera-FOV CBF-QP translational velocity filter.
- `target_tracking.py`: PID / 3D ALOS tracker and target-velocity estimator.
- `test_fov_cbf_unit.py`: FOV filter unit tests.
- `test_heading_forward_vehicle.py`: heading-forward kinematic model tests.
- `requirements-cbf.txt`: Python dependencies.
- `heading_forward_*.png`: generated plots.
- `heading_forward_metrics.csv`: simulation metrics.

## Run

```bash
pip install -r requirements-cbf.txt
python -m scripts.tests.test_heading_forward_fov
```

Headless mode (saves plots without opening windows):

```bash
python test_heading_forward_fov.py --no-show
```

Run the unit tests:

```bash
python -m unittest -v
```

## Heading-forward vehicle model

The guidance / FOV-CBF layers still generate a desired translational velocity
vector in body FRD. The vehicle model transforms that vector to NED and derives
its yaw and pitch references from the commanded direction:

- `yaw_ref = atan2(v_E, v_N)`
- `pitch_ref = atan2(-v_D, hypot(v_N, v_E))`
- surge speed follows the requested speed with a first-order lag and
  acceleration limit
- yaw and pitch are rate limited and have first-order rate response
- no lateral velocity command is applied; body sway decays with a damping time
  constant, with a small cross-flow disturbance included in the test
- heave is similarly damped while pitch follows the commanded vertical course

Thus the vehicle moves primarily along body +x. The camera's optical axis is
fixed to body +x, so camera orientation follows the vehicle's actual motion
rather than relying on arbitrary body-frame sway to keep up with the target.

## Simulation parameters

The default scenario uses a 90 s run at 20 Hz, unequal camera FOVs (50 deg
horizontal and 36 deg vertical), an inward guard margin (3 deg horizontal,
2 deg vertical), a yaw-rate limit of 22 deg/s, a pitch-rate limit of 10 deg/s,
and a 0.65 m/s^2 surge-acceleration limit. The nominal desired tracking point
is placed beyond the physical camera's horizontal FOV edge so that the CBF
must prioritize visibility over exact stand-off tracking.

The output plots include:

- `heading_forward_ned_trajectories.png`: leader and controlled-vehicle NED
  trajectories, with camera-frustum snapshots aligned to vehicle heading.
- `heading_forward_frd_tracking_errors.png`: forward/right/down tracking
  errors in instantaneous body FRD.
- `heading_forward_visibility_angles.png`: leader azimuth/elevation and
  effective horizontal/vertical FOV boundaries.
- `heading_forward_motion_diagnostics.png`: yaw command tracking error, body
  sway, surge, and camera-axis-to-actual-motion alignment.
- `heading_forward_fov_cbf_diagnostics.png`: FOV guard margins and CBF
  velocity corrections.

## Interpretation and limitations

This is a finite-step kinematic test, not a formal stability proof. The
controlled vehicle has yaw/pitch rate limits, a first-order rate response,
surge acceleration and speed limits, and small damped cross-flow disturbance.
It does not include full 6-DOF hydrodynamics, added mass, drag coupling,
actuator allocation, ocean-current estimation, camera latency, occlusion, or
pixel-level detector failures.

The FOV CBF solves a velocity-level QP while the vehicle's yaw/pitch and surge
actuators respond with finite dynamics. The simulation feeds actual body
velocity and angular rates back into the relative-motion estimate and uses an
inward barrier buffer, but that does not constitute a high-order CBF guarantee
for the actuator dynamics. For hardware use, incorporate yaw/pitch rate and
surge acceleration into a higher-order or predictive safety filter, include
camera measurement latency/uncertainty, and define a dedicated reacquisition
policy for target loss.

The vehicle is surge-dominant rather than strictly nonholonomic: small sway
and heave perturbations are allowed and damped. The body/camera forward axis
is aligned with the actual translational velocity to within the reported
small angular alignment metric when it is moving.

## Reference run (default parameters)

The finite 90 s run produced the following protected-case results:

| Metric | PID + FOV-CBF | ALOS + FOV-CBF |
|---|---:|---:|
| Final FRD position-error norm | 1.14 m | 1.45 m |
| Tail RMS tracking error | 1.17 m | 1.40 m |
| Physical FOV loss samples | 0 | 0 |
| Effective guard-band violation samples | 0 | 0 |
| Minimum horizontal angular margin | 1.45 deg | 1.37 deg |
| Minimum vertical angular margin | 8.46 deg | 7.82 deg |
| RMS body sway speed | 0.0032 m/s | 0.0032 m/s |
| Mean camera-axis / actual-motion misalignment | 0.29 deg | 0.29 deg |
| QP infeasible samples | 0 | 0 |

The unfiltered baselines leave the camera FOV, while both protected cases
remain inside the physical FOV and the tighter effective guard band for this
scenario. These numbers are simulation results for the configured model, not
hardware performance claims or formal guarantees.
