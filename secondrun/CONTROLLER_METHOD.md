# Controller method (secondrun/controller.py)

This repository implements a **cascade PID** controller for UAV position and yaw tracking, with an optional **disturbance observer / bias estimator** layer for wind rejection.

## Overview

The controller produces the command vector:

\[
u = (v_x, v_y, v_z, \dot{\psi})
\]

where \(v_x, v_y, v_z\) are velocity commands (m/s) and \(\dot{\psi}\) is a yaw-rate command (rad/s).

It uses a **two-loop cascade architecture** commonly found in autopilots (e.g., PX4/ArduPilot-style structure):

- **Outer loop (position loop)**: position error → **desired velocity** in the world frame
- **Inner loop (velocity loop)**: velocity error → **velocity correction** in the body frame
- **Yaw loop**: yaw angle error → **yaw-rate command**

## Outer loop: Position PID → desired velocity (world frame)

Let the measured position be \(p = [x, y, z]^T\) and the target be \(p_d\). The position error is:

\[
e_p = p_d - p
\]

The outer loop runs a PID per axis to create a desired velocity:

\[
v_d = K_{p,p}\, e_p + K_{i,p}\, \int e_p\,dt + K_{d,p}\, \frac{de_p}{dt}
\]

Key implementation details:

- **Integral anti-windup**: the integral state is clamped per axis (see `POS_INTEG_LIMIT`).
- **Velocity saturation**: \(v_d\) is clipped to `VEL_LIMIT` to avoid excessive commands.

## Frame transform: World velocity → body velocity

The simulator / Tello command interface expects body-frame velocities (relative to the drone heading). The controller rotates the world-frame velocity setpoint into the body frame using the current yaw \(\psi\):

\[
\begin{bmatrix}
v_{x,b} \\
v_{y,b} \\
v_{z,b}
\end{bmatrix}
=
\begin{bmatrix}
\cos\psi & \sin\psi & 0 \\
-\sin\psi & \cos\psi & 0 \\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
v_{x,w} \\
v_{y,w} \\
v_{z,w}
\end{bmatrix}
\]

Notes:

- Only a **yaw** rotation is applied (roll/pitch rotations are not used in this controller).

## Inner loop: Velocity PID → velocity correction (body frame)

Conceptually, the inner loop should compare desired body velocity \(v_{d,b}\) against measured body velocity \(v_b\):

\[
e_v = v_{d,b} - v_b
\]

Then apply another PID:

\[
\Delta v = K_{p,v}\, e_v + K_{i,v}\, \int e_v\,dt + K_{d,v}\, \frac{de_v}{dt}
\]

and produce the final commanded velocity:

\[
v_{cmd} = \text{clip}(v_{d,b} + \Delta v,\ \pm v_{\max})
\]

**Important implementation note:** in `secondrun/controller.py`, actual velocity feedback is *not* available in `state`, so the code currently uses a placeholder:

- `vel_estimated_body = vel_desired_body`
- hence `vel_error = 0` and the inner-loop PID output is effectively zero.

This means **the current behaviour is dominated by the outer-loop position PID**, with the inner loop present as scaffolding for when body velocity measurements (or a velocity estimator) are provided.

## Wind rejection: simplified DOBC / bias estimation

When `wind_enabled=True`, the controller applies a lightweight disturbance observer / bias estimator intended to cancel persistent wind-induced bias.

It computes a residual (here, the difference between the saturated final velocity and the desired body velocity), then low-pass filters it:

\[
\hat{d}_{k} = (1-\alpha)\,\hat{d}_{k-1} + \alpha\,r_k
\]

where:

- \(r_k = v_{cmd} - v_{d,b}\) (residual)
- \(\alpha\) is `DOBC_GAIN`
- \(\hat{d}\) is stored as `disturbance_estimate`

Then it subtracts the estimated bias from the command:

\[
v_{cmd} \leftarrow \text{clip}(v_{cmd} - \hat{d},\ \pm v_{\max})
\]

Interpretation:

- This is effectively an **exponential moving average (EMA)** of a velocity bias, used as feed-forward compensation.
- It is a simplified DOBC (it does not explicitly model drone translational dynamics).

## Yaw loop: PID on wrapped yaw error → yaw-rate

Yaw control is a single PID loop. The yaw error is wrapped to avoid \(\pm\pi\) discontinuities:

\[
e_\psi = \text{wrapToPi}(\psi_d - \psi)
\]

Then:

\[
\dot{\psi}_{cmd} = K_{p,\psi}\, e_\psi + K_{i,\psi}\, \int e_\psi\,dt + K_{d,\psi}\, \frac{de_\psi}{dt}
\]

With:

- integral clamping via `YAW_INTEG_LIMIT`
- yaw-rate saturation via `YAW_RATE_LIMIT`

## Timestep handling (sim vs real)

The controller accepts `dt` in two modes:

- **Simulation**: `dt` is already in seconds.
- **Real experiment**: `dt` is a Vicon timestamp in milliseconds; the controller converts it to seconds using the difference from the previous call.

This is done via the heuristic: if `dt > 1.0`, treat it as a timestamp (ms).

## Tuning and limits

The gains (`KP_POS`, `KI_POS`, `KD_POS`, `KP_VEL`, `KI_VEL`, `KD_VEL`, `KP_YAW`, `KI_YAW`, `KD_YAW`) are stored as constants at the top of the file.

The header documents that the parameters were tuned using **staged Bayesian optimisation** (Gaussian Process Regression + Expected Improvement) against PyBullet simulations, with a cost that penalises tracking error statistics and marking-threshold violations.

## Logging

Every controller call appends one row to `output.csv` including:

- timestamp
- state \([x, y, z, roll, pitch, yaw]\)
- command outputs \((v_x, v_y, v_z, \dot{\psi})\)
- target pose \((x_d, y_d, z_d, \psi_d)\)
- scalar position error norm and absolute yaw error

This is intended for post-run plotting and performance analysis.

