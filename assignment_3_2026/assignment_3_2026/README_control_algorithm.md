# Outer-loop control algorithm (DOB-assisted position–velocity cascade)

This document describes the logic implemented in `controller.py`: the **yaw-aligned outer loop** that maps position error to velocity setpoints `(vx_cmd, vy_cmd, vz_cmd)` and a **yaw rate** command, optional **disturbance observer (DOB)** compensation when wind is enabled, and practical **tuning** notes for the PyBullet + Tello stack used in this coursework.

---

## 1. Role in the full stack

The simulator runs a **two-layer** structure:

1. **Outer loop (`controller.py`)** — runs at **50 Hz** (see `run.py` / `run_test.py`). It outputs **body-frame velocity commands** aligned with **current yaw** (yaw-aligned body frame, not full roll/pitch body frame). It also outputs `yaw_rate_cmd` for heading.
2. **Inner loop (`src/tello_controller.py`)** — runs at **1000 Hz**. It tracks the held velocity setpoint with PID on acceleration, attitude, and rates, then mixes motor thrust/torque.

The outer loop therefore sees a **zero-order hold** on its command and the **delayed, nonlinear** response of the inner loop and airframe. Gains must be chosen for that effective plant, not for an ideal first-order integrator.

---

## 2. Coordinate frames

- **World frame**: PyBullet world axes; position and (when provided) linear velocity in `state` are in this frame.
- **Yaw-aligned body frame**: Vectors are rotated by yaw only:

  \[
  R_{\text{w}\rightarrow\text{b,yaw}} =
  \begin{bmatrix}
  \cos\psi & \sin\psi & 0 \\
  -\sin\psi & \cos\psi & 0 \\
  0 & 0 & 1
  \end{bmatrix}
  \]

  Position error \(\mathbf{e}_p = \mathbf{p}_d - \mathbf{p}\) and velocity estimate \(\hat{\mathbf{v}}\) are transformed into this frame so horizontal commands stay intuitive when the drone yaws: “forward” follows heading.

The **inner** velocity controller uses the same yaw-stripped velocity convention as the simulator when building `lin_vel` for control (see `run.py`).

---

## 3. State vector and velocity estimate

Supported formats:

| Length | Meaning |
|--------|--------|
| **6** | `[px, py, pz, roll, pitch, yaw]` — velocity is **estimated** by filtered backward difference of position at the outer-loop rate. |
| **9** | Same six plus **`[vx, vy, vz]` in world frame (m/s)** — preferred when the simulator passes `getBaseVelocity()` so the **D-term uses the same velocity** the physics and inner loop see. |

If you only have position (e.g. autograder), the fallback estimator uses:

\[
\hat{\mathbf{v}}_k = \alpha \frac{\mathbf{p}_k - \mathbf{p}_{k-1}}{\Delta t} + (1-\alpha)\hat{\mathbf{v}}_{k-1}
\]

with `vel_lpf_alpha` \(=\alpha\). Smaller \(\alpha\) smooths noise but adds **lag**, which can hurt damping if `kd_vel` is large.

---

## 4. Nominal outer loop: PD in velocity command space

In yaw-aligned body frame:

\[
\mathbf{v}_{\text{cmd,nom}} = K_p \odot \mathbf{e}_{p,\text{body}} - K_d \odot \hat{\mathbf{v}}_{\text{body}}
\]

- **`kp_pos`**: per-axis position gain (maps metres of error to m/s command).
- **`kd_vel`**: per-axis “derivative” gain on estimated body velocity (damping).

This is a **PD on position expressed as a desired velocity**, which is standard for a cascade where an inner loop tracks velocity.

Commands are then **saturated** by `max_vel` per axis.

### Horizontal dead zone

If horizontal position error and horizontal estimated speed are both below thresholds (`0.08` m in code), **`vx_cmd` and `vy_cmd` are forced to zero** to stop limit cycling or chatter at the setpoint. The dead zone is applied to the **final** command after DOB and clipping.

---

## 5. Yaw loop

A simple **P** controller on wrapped yaw error:

\[
\dot{\psi}_{\text{cmd}} = \mathrm{clip}(k_{p,\psi} \cdot \mathrm{wrap}(\psi_d - \psi), \pm \dot{\psi}_{\max})
\]

`wrap_to_pi` keeps the error in \((-\pi, \pi]\).

---

## 6. Disturbance observer (DOB) — idea and this implementation

### 6.1 Intuition

Ideal inner dynamics would make **actual body velocity** track **commanded** velocity. In reality, **unmodeled effects** (wind, drag mismatch, saturation, coupling) create a persistent **velocity tracking error**:

\[
\mathbf{e}_v = \mathbf{v}_{\text{cmd,nom}} - \hat{\mathbf{v}}_{\text{body}}
\]

A **disturbance observer** treats a filtered version of this mismatch as an estimate \(\hat{\mathbf{d}}\) of an **equivalent input disturbance** in the same frame as the velocity command. It **feeds forward** a correction so the plant behaves closer to nominal:

\[
\mathbf{v}_{\text{cmd}} = \mathbf{v}_{\text{cmd,nom}} + K_{\text{comp}} \odot \hat{\mathbf{d}}
\]

### 6.2 Discrete-time update used here

The code integrates:

\[
\dot{\hat{\mathbf{d}}} = s \cdot K_{\text{dob}} \odot \mathbf{e}_v - L \odot \hat{\mathbf{d}}
\]

- **`k_dob`**: how aggressively the observer pushes \(\hat{\mathbf{d}}\) toward explaining the velocity error.
- **`dob_leak`**: first-order **leak** (stability / forgetting); prevents \(\hat{\mathbf{d}}\) from integrating forever on bias or sensor quirks.
- **`scale` `s`**: when `wind_enabled` is true, `s = 1`; otherwise the legacy path inside `update_dob` used `s = 0.5` (only relevant if that function is called).
- **`d_hat`** is **clipped** to a box limit to avoid runaway.

This is a **simplified** DOB / disturbance estimator, not a full Luenberger observer with an explicit plant model, but it captures the same engineering goal: **reject slowly varying disturbances** that show up as velocity lag or offset.

### 6.3 When DOB is active in this project

- **`wind_enabled == False`**: DOB is **disabled**. \(\hat{\mathbf{d}}\) is **zeroed** every update and **no** compensation is added.  
  **Reason:** without wind, the dominant mismatch is often **delay, ZOH, and velocity estimation error**, not a persistent external force. Feeding that into \(\hat{\mathbf{d}}\) can create **false compensation** and **limit cycles**.

- **`wind_enabled == True`**: DOB **compensation** is applied: \(\mathbf{v}_{\text{cmd}} = \mathbf{v}_{\text{cmd,nom}} + K_{\text{comp}} \odot \hat{\mathbf{d}}\).

- **Distance gate (horizontal)**: `update_dob` is only called when \(\|e_{p,xy}\| > 0.8\) m. Inside that radius the observer state is **not** updated (but existing \(\hat{\mathbf{d}}\) still affects the command until leak/dynamics change it — be aware when tuning near the target).

---

## 7. Tuning guide

Tune in this order: **velocity source → horizontal PD → saturations/dead zone → DOB (wind on)**.

### 7.1 Use measured velocity in simulation

Prefer **`state` length 9** with world `vx,vy,vz` from the simulator. This aligns the D-term with the inner loop and reduces spurious oscillation when only 50 Hz position differencing is available.

### 7.2 `kp_pos` (horizontal vs vertical)

- **Too high**: Aggressive velocity requests → inner loop saturates → oscillation or overshoot; distance and `vx_cmd`/`vy_cmd` may show large limit cycles.
- **Too low**: Sluggish approach; large steady-state error if inner loop bias exists.

**Horizontal (x, y)** is usually **lower** than **z** because the horizontal path is longer and coupling with yaw and drag is stronger. Start from moderate values and increase only if response is clearly slow *and* telemetry stays smooth.

### 7.3 `kd_vel`

- **Too high**: Over-damping or noise amplification if \(\hat{\mathbf{v}}\) is noisy or phase-lagged (common with differenced position).
- **Too low**: Under-damped step response, ringing around the target.

If you **must** use 6-state mode, favour a **smaller** `kd_vel` or a **smaller** `vel_lpf_alpha` (smoother but more lag — trade carefully).

### 7.4 `max_vel`

Caps outer commands. Lowering horizontal limits can **soften** aggressive behaviour and sometimes removes limit cycles at the cost of slower transit.

### 7.5 Dead-zone thresholds

Larger thresholds → earlier cut-off of horizontal commands near the goal → less chatter, but **larger** terminal position band. Adjust if the drone “buzzes” at the end or stops short.

### 7.6 Yaw `kp_yaw`

Increase if heading lags during moves; decrease if yaw rate oscillates or fights the position loop.

### 7.7 DOB gains (wind-on scenarios)

- **`k_dob`**: Higher → faster disturbance rejection, but more sensitive to **modeling error** and **measurement noise** on velocity. If \(\hat{\mathbf{d}}\) chatters, reduce `k_dob` or increase `dob_leak` slightly.
- **`dob_leak`**: Higher → faster decay of \(\hat{\mathbf{d}}\) when disturbance disappears; also reduces observer “memory” that can fight the nominal loop.
- **`k_comp`**: How much of \(\hat{\mathbf{d}}\) is injected into the command. Too large can **over-correct** and destabilize; too small gives weak rejection.

Always validate **with wind on** after changing DOB gains; behaviour with wind off is intentionally **PD-only** in this implementation.

### 7.8 Outer-loop rate `dt`

The controller must receive the **same** \(\Delta t\) as the rate at which it is called (here **0.02 s** for 50 Hz). Wrong `dt` scales the DOB integrator incorrectly.

---

## 8. Debugging checklist

| Symptom | Things to check |
|--------|------------------|
| Large sinusoidal `vx_cmd`/`vy_cmd`, distance oscillates | Reduce `kp_pos` xy; reduce `kd_vel` xy if using differenced velocity; ensure 9-state velocity if possible; confirm DOB off when `wind_enabled` is false. |
| Slow approach, no wind | Increase `kp_pos` xy slightly; check `max_vel` not too low. |
| Chatter at goal | Increase dead-zone radii slightly; reduce `kp_pos` xy; check inner-loop resets on target change. |
| Wind on: drift or bias | Increase `k_comp` or `k_dob` modestly; ensure `update_dob` runs when error is large enough (distance gate). |
| Wind on: oscillation | Decrease `k_dob` or `k_comp`; increase `dob_leak`; verify velocity signal is not overly noisy. |

---

## 9. File reference

- **Outer loop**: `controller.py` (`DOBController`, `controller()`).
- **Inner loop**: `src/tello_controller.py`, `src/PID_controller.py`.
- **Simulation harness**: `run.py`, `run_test.py` (outer-loop rate, wind flag, optional extended `state`).

---

## 10. Summary

The outer loop is a **yaw-aligned PD position-to-velocity** law with optional **DOB-based feedforward** when disturbances (wind) are present. The DOB integrates **velocity tracking error** into \(\hat{\mathbf{d}}\) with leak and saturation, then adds \(K_{\text{comp}} \hat{\mathbf{d}}\) to the command. With **no wind**, DOB is **turned off** so delay and estimation mismatch are not mistaken for disturbances. **Tuning** should respect the **50 Hz / 1 kHz cascade**, prefer **measured velocity** when available, and adjust PD before aggressive DOB gains.
