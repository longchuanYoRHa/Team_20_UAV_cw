import numpy as np

# ============================================================
# Outer-loop controller for the provided coursework interface.
#
# Design idea:
# - The inner loop (velocity -> attitude -> rate -> motors) is NOT an ideal
#   first-order tracker in x/y. Step tests show delayed, underdamped,
#   second-order-like behaviour horizontally.
# - Therefore the outer loop is tuned conservatively in x/y:
#     * smaller P
#     * stronger D on filtered velocity estimate
#     * very small / gated I in x/y
# - z is allowed to be more direct because the inner z channel is much closer
#   to first-order.
# - DOB is only used as a mild wind compensation term and is intentionally
#   weak because the effective outer-loop dt is large.
#
# IMPORTANT:
# The simulator physics step is 1/240 s. In the current run setup, the outer
# controller is effectively updated every ~20 simulator steps, so the outer
# loop dt should be treated as about 20/240 = 0.08333 s.
# We therefore override dt inside controller() to 0.08333 for consistency.
# ============================================================

FIXED_OUTER_DT = 1.0 / 12.0  # 0.083333... s


def wrap_to_pi(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def rot_world_to_yaw_body(yaw: float) -> np.ndarray:
    c = np.cos(yaw)
    s = np.sin(yaw)
    return np.array([
        [c,  s, 0.0],
        [-s, c, 0.0],
        [0.0, 0.0, 1.0],
    ])


class SecondOrderAwareOuterLoop:
    def __init__(self):
        # ---------------- Position -> velocity gains ----------------
        # Horizontal axes: conservative because inner dynamics are slow and
        # underdamped. z can be more direct.
        self.kp_pos = np.array([0.34, 0.34, 1.05], dtype=float)
        self.ki_pos = np.array([0.015, 0.015, 0.30], dtype=float)
        self.kd_vel = np.array([0.42, 0.42, 0.62], dtype=float)
        self.ki_pos_sat = np.array([0.18, 0.18, 0.40], dtype=float)

        # Integrator leak helps when dt is large and the inner loop is not able
        # to track aggressive commands perfectly.
        self.int_pos_leak = np.array([1.8, 1.8, 0.9], dtype=float)

        # Yaw loop: moderate, because aggressive yaw can inject x/y coupling.
        self.kp_yaw = 1.65
        self.ki_yaw = 0.16
        self.kd_yaw = 0.05
        self.ki_yaw_sat = 0.16
        self.int_yaw_leak = 1.1

        # ---------------- Velocity estimation ----------------
        # With effective dt ~= 0.0833 s, raw backward difference is noisy and
        # phase-lagged. Use a fairly strong LPF.
        self.vel_lpf_alpha = 0.24

        # ---------------- Command shaping ----------------
        self.max_vel = np.array([1.0, 1.0, 1.0], dtype=float)
        self.max_xy_speed = 0.95
        self.max_yaw_rate = 1.25

        # Slew-rate limits (m/s^2 interpreted in command space).
        self.max_xy_slew = 2.2
        self.max_z_slew = 2.0
        self.max_yaw_slew = 2.8

        # Near-target behaviour: stronger braking and dead-zone reduce chatter.
        self.xy_brake_radius = 0.28
        self.xy_brake_gain = 1.55
        self.xy_pos_deadzone = 0.035
        self.xy_vel_deadzone = 0.05
        self.z_pos_deadzone = 0.020
        self.z_vel_deadzone = 0.04

        # Only allow horizontal motion once heading is broadly aligned.
        self.yaw_align_tol = np.deg2rad(10.0)

        # ---------------- Mild wind observer ----------------
        # Keep DOB intentionally weak because outer dt is large and fast gusts
        # cannot be estimated accurately from slow position differencing.
        self.k_dob = np.array([0.10, 0.10, 0.00], dtype=float)
        self.dob_leak = np.array([0.55, 0.55, 0.40], dtype=float)
        self.k_comp = np.array([0.14, 0.14, 0.00], dtype=float)
        self.d_hat = np.zeros(3, dtype=float)

        # ---------------- Internal states ----------------
        self.prev_pos = None
        self.prev_yaw_err = None
        self.vel_est_world = np.zeros(3, dtype=float)
        self.int_pos_body = np.zeros(3, dtype=float)
        self.int_yaw = 0.0
        self.prev_vel_cmd = np.zeros(3, dtype=float)
        self.prev_yaw_rate_cmd = 0.0
        self.last_debug = {}

    def reset(self):
        self.prev_pos = None
        self.prev_yaw_err = None
        self.vel_est_world[:] = 0.0
        self.int_pos_body[:] = 0.0
        self.int_yaw = 0.0
        self.prev_vel_cmd[:] = 0.0
        self.prev_yaw_rate_cmd = 0.0
        self.d_hat[:] = 0.0
        self.last_debug = {}

    def estimate_velocity(self, pos: np.ndarray, dt: float) -> np.ndarray:
        if self.prev_pos is None or dt <= 1e-6:
            self.prev_pos = pos.copy()
            return np.zeros(3, dtype=float)

        raw_vel = (pos - self.prev_pos) / dt
        raw_vel = np.clip(raw_vel, -3.0, 3.0)
        self.prev_pos = pos.copy()

        a = self.vel_lpf_alpha
        self.vel_est_world = a * raw_vel + (1.0 - a) * self.vel_est_world
        return self.vel_est_world.copy()

    def update_dob(self, vel_cmd_nominal: np.ndarray, vel_est_body: np.ndarray, dt: float):
        if dt <= 1e-6:
            return
        tracking_error = vel_cmd_nominal - vel_est_body
        d_dot = self.k_dob * tracking_error - self.dob_leak * self.d_hat
        self.d_hat += d_dot * dt
        self.d_hat = np.clip(self.d_hat, -0.35, 0.35)

    def apply_slew(self, vel_cmd: np.ndarray, yaw_rate_cmd: float, dt: float):
        dv = vel_cmd - self.prev_vel_cmd
        max_dxy = self.max_xy_slew * dt
        max_dz = self.max_z_slew * dt
        dv[0] = np.clip(dv[0], -max_dxy, max_dxy)
        dv[1] = np.clip(dv[1], -max_dxy, max_dxy)
        dv[2] = np.clip(dv[2], -max_dz, max_dz)
        vel_cmd = self.prev_vel_cmd + dv

        dyaw = yaw_rate_cmd - self.prev_yaw_rate_cmd
        max_dyaw = self.max_yaw_slew * dt
        yaw_rate_cmd = self.prev_yaw_rate_cmd + np.clip(dyaw, -max_dyaw, max_dyaw)

        self.prev_vel_cmd = vel_cmd.copy()
        self.prev_yaw_rate_cmd = float(yaw_rate_cmd)
        return vel_cmd, float(yaw_rate_cmd)

    def compute(self, state, target_pos, dt, wind_enabled=False):
        state = np.asarray(state, dtype=float).reshape(-1)
        if state.size < 6:
            return (0.0, 0.0, 0.0, 0.0)

        x, y, z, roll, pitch, yaw = state[0:6]
        x_d, y_d, z_d, yaw_d = target_pos

        pos = np.array([x, y, z], dtype=float)
        pos_d = np.array([x_d, y_d, z_d], dtype=float)
        pos_err_world = pos_d - pos
        vel_est_world = self.estimate_velocity(pos, dt)

        R_w2b = rot_world_to_yaw_body(yaw)
        pos_err_body = R_w2b @ pos_err_world
        vel_est_body = R_w2b @ vel_est_world

        yaw_err = wrap_to_pi(yaw_d - yaw)
        yaw_aligned = abs(yaw_err) < self.yaw_align_tol

        # Dead-zone on tiny errors before integration.
        pos_for_i = pos_err_body.copy()
        if abs(pos_for_i[0]) < self.xy_pos_deadzone:
            pos_for_i[0] = 0.0
        if abs(pos_for_i[1]) < self.xy_pos_deadzone:
            pos_for_i[1] = 0.0
        if abs(pos_for_i[2]) < self.z_pos_deadzone:
            pos_for_i[2] = 0.0

        # z always integrated; x/y only when yaw roughly aligned and not moving fast.
        self.int_pos_body[2] += pos_for_i[2] * dt
        if yaw_aligned and np.linalg.norm(vel_est_body[0:2]) < 0.75:
            self.int_pos_body[0] += pos_for_i[0] * dt
            self.int_pos_body[1] += pos_for_i[1] * dt

        self.int_pos_body *= np.exp(-self.int_pos_leak * dt)
        self.int_pos_body = np.clip(self.int_pos_body, -self.ki_pos_sat, self.ki_pos_sat)

        # Core PID terms.
        p_term = self.kp_pos * pos_err_body
        i_term = self.ki_pos * self.int_pos_body
        d_term = -self.kd_vel * vel_est_body

        # If far from yaw alignment, suppress x/y integral and motion.
        if not yaw_aligned:
            i_term[0:2] = 0.0

        vel_cmd_nominal = p_term + i_term + d_term

        # Near target: add extra braking in x/y to avoid exciting inner-loop overshoot.
        xy_err_norm = np.linalg.norm(pos_err_body[0:2])
        if xy_err_norm < self.xy_brake_radius:
            brake_scale = 1.0 + self.xy_brake_gain * (1.0 - xy_err_norm / max(self.xy_brake_radius, 1e-6))
            vel_cmd_nominal[0:2] -= (brake_scale - 1.0) * vel_est_body[0:2]

        # Mild DOB for wind only.
        dob_comp = np.zeros(3, dtype=float)
        if wind_enabled and yaw_aligned and 0.10 < xy_err_norm < 1.25:
            self.update_dob(vel_cmd_nominal, vel_est_body, dt)
            dob_comp = self.k_comp * self.d_hat
        else:
            self.d_hat *= np.exp(-self.dob_leak * dt)

        vel_cmd = vel_cmd_nominal + dob_comp

        # Horizontal command gating until yaw is roughly aligned.
        if not yaw_aligned:
            vel_cmd[0] = 0.0
            vel_cmd[1] = 0.0

        # Horizontal norm limit.
        xy = vel_cmd[0:2]
        nxy = np.linalg.norm(xy)
        if nxy > self.max_xy_speed:
            vel_cmd[0:2] = xy * (self.max_xy_speed / nxy)

        # Per-axis clip.
        vel_cmd = np.clip(vel_cmd, -self.max_vel, self.max_vel)

        # Final dead-zone to stop buzzing at the goal.
        if xy_err_norm < self.xy_pos_deadzone and np.linalg.norm(vel_est_body[0:2]) < self.xy_vel_deadzone:
            vel_cmd[0] = 0.0
            vel_cmd[1] = 0.0
        if abs(pos_err_body[2]) < self.z_pos_deadzone and abs(vel_est_body[2]) < self.z_vel_deadzone:
            vel_cmd[2] = 0.0

        # Yaw PID with leak / anti-windup.
        self.int_yaw += yaw_err * dt
        self.int_yaw *= float(np.exp(-self.int_yaw_leak * dt))
        self.int_yaw = float(np.clip(self.int_yaw, -self.ki_yaw_sat, self.ki_yaw_sat))

        if self.prev_yaw_err is None or dt <= 1e-6:
            yaw_err_d = 0.0
        else:
            delta = yaw_err - self.prev_yaw_err
            if delta > np.pi:
                delta -= 2.0 * np.pi
            elif delta < -np.pi:
                delta += 2.0 * np.pi
            yaw_err_d = np.clip(delta / dt, -6.0, 6.0)
        self.prev_yaw_err = yaw_err

        yaw_rate_unsat = self.kp_yaw * yaw_err + self.ki_yaw * self.int_yaw + self.kd_yaw * yaw_err_d
        yaw_rate_cmd = float(np.clip(yaw_rate_unsat, -self.max_yaw_rate, self.max_yaw_rate))

        if abs(yaw_rate_cmd) >= self.max_yaw_rate - 1e-6:
            self.int_yaw -= yaw_err * dt

        vel_cmd, yaw_rate_cmd = self.apply_slew(vel_cmd, yaw_rate_cmd, dt)

        self.last_debug = {
            "pos": pos.copy(),
            "target": pos_d.copy(),
            "pos_err_world": pos_err_world.copy(),
            "pos_err_body": pos_err_body.copy(),
            "vel_est_world": vel_est_world.copy(),
            "vel_est_body": vel_est_body.copy(),
            "pid_p_body": p_term.copy(),
            "pid_i_body": i_term.copy(),
            "pid_d_body": d_term.copy(),
            "vel_cmd_nominal": vel_cmd_nominal.copy(),
            "dob_comp_body": dob_comp.copy(),
            "d_hat": self.d_hat.copy(),
            "vel_cmd_final": vel_cmd.copy(),
            "yaw_err": float(yaw_err),
            "yaw_rate_cmd": float(yaw_rate_cmd),
            "yaw_aligned": bool(yaw_aligned),
            "int_pos_body": self.int_pos_body.copy(),
            "int_yaw": float(self.int_yaw),
        }

        return (
            float(vel_cmd[0]),
            float(vel_cmd[1]),
            float(vel_cmd[2]),
            float(yaw_rate_cmd),
        )


_outer = SecondOrderAwareOuterLoop()


def get_integral_telemetry():
    d = _outer.last_debug
    if not d:
        return {
            "pid_i_body": (0.0, 0.0, 0.0),
            "yaw_i_term": 0.0,
            "dob_comp_body": (0.0, 0.0, 0.0),
            "vel_est_body": (0.0, 0.0, 0.0),
        }
    pi = np.asarray(d.get("pid_i_body", (0.0, 0.0, 0.0)), dtype=float).ravel()
    dob_comp = np.asarray(d.get("dob_comp_body", (0.0, 0.0, 0.0)), dtype=float).ravel()
    vel_est = np.asarray(d.get("vel_est_body", (0.0, 0.0, 0.0)), dtype=float).ravel()
    return {
        "pid_i_body": (float(pi[0]), float(pi[1]), float(pi[2])),
        "yaw_i_term": float(_outer.ki_yaw * _outer.int_yaw),
        "dob_comp_body": (float(dob_comp[0]), float(dob_comp[1]), float(dob_comp[2])),
        "vel_est_body": (float(vel_est[0]), float(vel_est[1]), float(vel_est[2])),
    }


def controller(state, target_pos, dt, wind_enabled=False):
    # Ignore supplied dt and use the effective outer-loop update period that
    # matches the current simulator / run setup.
    dt = FIXED_OUTER_DT
    return _outer.compute(state, target_pos, dt, wind_enabled)
