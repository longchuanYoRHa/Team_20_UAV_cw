import numpy as np

# ============================================================
# Outer-loop controller for AERO60492 coursework
#
# Design rationale for this version:
# 1) The provided inner loop is much slower in vx / vy than vz.
#    From identified inner-loop fits, vx/vy bandwidth is only about
#    0.25 Hz, while vz is much faster.
# 2) Therefore the outer x/y loop must be deliberately conservative,
#    strongly filtered, and protected against integral windup.
# 3) A lightweight DOB is still useful, but only as a low-bandwidth
#    bias compensator for wind; it must not fight fast transient motion.
#
# Interface must stay unchanged:
#   controller(state, target_pos, dt, wind_enabled=False)
# returns:
#   (vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd)
#
# IMPORTANT:
# The simulator code passes 1/50 s, but for this coursework stack the
# effective outer-loop timing is better matched by dt = 0.0833 s.
# This controller therefore locks dt to 0.0833 s intentionally.
# ============================================================


OUTER_DT = 0.0833


def wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def rot_world_to_yaw_body(yaw):
    c = np.cos(yaw)
    s = np.sin(yaw)
    return np.array([
        [c,  s, 0.0],
        [-s, c, 0.0],
        [0.0, 0.0, 1.0]
    ])


def norm_clip_2d(vec, limit):
    n = np.linalg.norm(vec)
    if n > limit and n > 1e-9:
        return vec * (limit / n)
    return vec


class DOBController:
    def __init__(self):
        # -----------------------------
        # Core gains matched to inner-loop response
        # -----------------------------
        # vx/vy inner loop is very slow and underdamped:
        # use modest P, very small I, moderate D from filtered velocity.
        kp_xy, kp_z = 0.26, 0.92
        ki_xy, ki_z = 0.035, 0.22
        kd_xy, kd_z = 0.17, 0.30
        ki_sat_xy, ki_sat_z = 0.28, 0.36

        self.kp_pos = np.array([kp_xy, kp_xy, kp_z], dtype=float)
        self.ki_pos = np.array([ki_xy, ki_xy, ki_z], dtype=float)
        self.kd_vel = np.array([kd_xy, kd_xy, kd_z], dtype=float)
        self.ki_pos_sat = np.array([ki_sat_xy, ki_sat_xy, ki_sat_z], dtype=float)

        # Yaw inner loop is comparatively fast, so yaw can be handled more directly.
        self.kp_yaw = 1.70
        self.ki_yaw = 0.18
        self.kd_yaw = 0.06
        self.ki_yaw_sat = 0.16

        # -----------------------------
        # Command shaping for slow xy inner loop
        # -----------------------------
        self.max_vel = np.array([1.0, 1.0, 1.0], dtype=float)
        self.max_xy_speed = 0.62
        self.max_z_speed = 0.85
        self.max_yaw_rate = 1.74533

        # First-turn strategy: do not translate aggressively until yaw is roughly aligned.
        self.yaw_align_tol = 0.22  # rad
        self.yaw_soft_tol = 0.45   # rad

        # Low-pass + slew limit on commanded velocity to protect slow vx/vy inner loop.
        self.cmd_lpf_alpha_xy = 0.28
        self.cmd_lpf_alpha_z = 0.42
        self.max_xy_slew = 1.8    # m/s^2 equivalent over outer updates
        self.max_z_slew = 2.8

        # Horizontal braking when close to target.
        self.xy_brake_radius = 0.45
        self.xy_deadzone_pos = 0.03
        self.xy_deadzone_vel = 0.05
        self.z_deadzone_pos = 0.02
        self.z_deadzone_vel = 0.04

        # -----------------------------
        # Disturbance observer (slow bias estimator)
        # -----------------------------
        # This is intentionally mild: wind model has slow direction drift + short gusts.
        # DOB should mainly remove low-frequency bias, not chase fast gust peaks.
        self.k_dob = np.array([0.18, 0.18, 0.00], dtype=float)
        self.dob_leak = np.array([0.28, 0.28, 0.40], dtype=float)
        self.k_comp = np.array([0.42, 0.42, 0.00], dtype=float)
        self.d_hat_limit = np.array([0.22, 0.22, 0.00], dtype=float)
        self.dob_active_xy_err = 1.20
        self.dob_freeze_xy_err = 1.60

        # -----------------------------
        # Internal states
        # -----------------------------
        self.prev_pos = None
        self.vel_est_world = np.zeros(3, dtype=float)
        self.vel_est_lpf_alpha = 0.22

        self.int_pos_body = np.zeros(3, dtype=float)
        self.int_pos_leak = np.array([0.55, 0.55, 0.25], dtype=float)

        self.prev_yaw_err = None
        self.int_yaw = 0.0
        self.int_yaw_leak = 0.65

        self.d_hat = np.zeros(3, dtype=float)
        self.vel_cmd_lpf = np.zeros(3, dtype=float)
        self.last_debug = {}

    def reset(self):
        self.prev_pos = None
        self.vel_est_world[:] = 0.0
        self.int_pos_body[:] = 0.0
        self.prev_yaw_err = None
        self.int_yaw = 0.0
        self.d_hat[:] = 0.0
        self.vel_cmd_lpf[:] = 0.0
        self.last_debug = {}

    def _estimate_velocity(self, pos, dt):
        if self.prev_pos is None or dt <= 1e-9:
            self.prev_pos = pos.copy()
            self.vel_est_world[:] = 0.0
            return self.vel_est_world.copy()

        raw_vel = (pos - self.prev_pos) / dt
        raw_vel = np.clip(raw_vel, -2.5, 2.5)
        a = self.vel_est_lpf_alpha
        self.vel_est_world = a * raw_vel + (1.0 - a) * self.vel_est_world
        self.prev_pos = pos.copy()
        return self.vel_est_world.copy()

    def _apply_slew_and_lpf(self, vel_cmd_raw, dt):
        target = vel_cmd_raw.copy()
        prev = self.vel_cmd_lpf.copy()

        dxy = target[0:2] - prev[0:2]
        dxy = norm_clip_2d(dxy, self.max_xy_slew * dt)
        dz = np.clip(target[2] - prev[2], -self.max_z_slew * dt, self.max_z_slew * dt)

        slewed = prev.copy()
        slewed[0:2] += dxy
        slewed[2] += dz

        out = prev.copy()
        out[0:2] = (1.0 - self.cmd_lpf_alpha_xy) * prev[0:2] + self.cmd_lpf_alpha_xy * slewed[0:2]
        out[2] = (1.0 - self.cmd_lpf_alpha_z) * prev[2] + self.cmd_lpf_alpha_z * slewed[2]

        out[0:2] = norm_clip_2d(out[0:2], self.max_xy_speed)
        out[2] = np.clip(out[2], -self.max_z_speed, self.max_z_speed)
        out = np.clip(out, -self.max_vel, self.max_vel)

        self.vel_cmd_lpf = out
        return out.copy()

    def _update_dob(self, vel_cmd_nominal_body, vel_est_body, pos_err_body, dt, wind_enabled):
        if (not wind_enabled) or dt <= 1e-9:
            self.d_hat[:] = 0.0
            return np.zeros(3, dtype=float)

        xy_err = np.linalg.norm(pos_err_body[0:2])
        # Freeze / leak far away from target, because large transient tracking error is
        # dominated by slow inner-loop lag rather than wind bias.
        if xy_err > self.dob_freeze_xy_err:
            self.d_hat *= np.exp(-self.dob_leak * dt)
            return np.zeros(3, dtype=float)

        vel_tracking_error = vel_cmd_nominal_body - vel_est_body

        # Enable observer mostly in the medium / near-target region.
        activity = 1.0
        if xy_err > self.dob_active_xy_err:
            span = max(self.dob_freeze_xy_err - self.dob_active_xy_err, 1e-6)
            activity = np.clip((self.dob_freeze_xy_err - xy_err) / span, 0.0, 1.0)

        d_dot = activity * (self.k_dob * vel_tracking_error) - self.dob_leak * self.d_hat
        self.d_hat += d_dot * dt
        self.d_hat = np.clip(self.d_hat, -self.d_hat_limit, self.d_hat_limit)
        return self.k_comp * self.d_hat

    def compute(self, state, target_pos, dt, wind_enabled=False):
        state = np.asarray(state, dtype=float).reshape(-1)
        if state.size < 6:
            return (0.0, 0.0, 0.0, 0.0)

        dt = OUTER_DT

        x, y, z, roll, pitch, yaw = state[0:6]
        x_d, y_d, z_d, yaw_d = target_pos

        pos = np.array([x, y, z], dtype=float)
        pos_d = np.array([x_d, y_d, z_d], dtype=float)
        pos_err_world = pos_d - pos
        vel_est_world = self._estimate_velocity(pos, dt)

        R_w2b = rot_world_to_yaw_body(yaw)
        pos_err_body = R_w2b @ pos_err_world
        vel_est_body = R_w2b @ vel_est_world

        # Near-origin lateral deadzone to avoid y chatter created by slow / underdamped xy dynamics.
        if abs(pos_err_body[1]) < 0.025 and abs(vel_est_body[1]) < 0.06:
            pos_err_body[1] = 0.0

        # Yaw scheduling: suppress translation when yaw error is large.
        yaw_err = wrap_to_pi(yaw_d - yaw)
        yaw_abs = abs(yaw_err)
        if yaw_abs <= self.yaw_align_tol:
            yaw_scale = 1.0
        elif yaw_abs >= self.yaw_soft_tol:
            yaw_scale = 0.0
        else:
            yaw_scale = (self.yaw_soft_tol - yaw_abs) / (self.yaw_soft_tol - self.yaw_align_tol)

        # Brake xy command when close to target so the slow xy inner loop does not overshoot repeatedly.
        xy_err_norm = np.linalg.norm(pos_err_body[0:2])
        if xy_err_norm < self.xy_brake_radius:
            xy_scale = np.clip(xy_err_norm / self.xy_brake_radius, 0.18, 1.0)
        else:
            xy_scale = 1.0

        # Integrator: only build horizontal I when yaw is roughly aligned and command is not saturated badly.
        self.int_pos_body *= np.exp(-self.int_pos_leak * dt)
        self.int_pos_body[2] += pos_err_body[2] * dt
        if yaw_scale > 0.7 and xy_err_norm < 1.2:
            self.int_pos_body[0] += pos_err_body[0] * dt
            self.int_pos_body[1] += pos_err_body[1] * dt
        self.int_pos_body = np.clip(self.int_pos_body, -self.ki_pos_sat, self.ki_pos_sat)

        p_term = self.kp_pos * pos_err_body
        i_term = self.ki_pos * self.int_pos_body
        d_term = -self.kd_vel * vel_est_body

        vel_cmd_nominal = p_term + i_term + d_term
        vel_cmd_nominal[0:2] *= yaw_scale * xy_scale
        vel_cmd_nominal[0:2] = norm_clip_2d(vel_cmd_nominal[0:2], self.max_xy_speed)
        vel_cmd_nominal[2] = np.clip(vel_cmd_nominal[2], -self.max_z_speed, self.max_z_speed)

        dob_comp = self._update_dob(vel_cmd_nominal, vel_est_body, pos_err_body, dt, wind_enabled)
        vel_cmd_raw = vel_cmd_nominal + dob_comp

        # Final deadzones near target
        if xy_err_norm < self.xy_deadzone_pos and np.linalg.norm(vel_est_body[0:2]) < self.xy_deadzone_vel:
            vel_cmd_raw[0] = 0.0
            vel_cmd_raw[1] = 0.0
            self.d_hat[0:2] *= 0.7

        if abs(pos_err_body[2]) < self.z_deadzone_pos and abs(vel_est_body[2]) < self.z_deadzone_vel:
            vel_cmd_raw[2] = 0.0

        vel_cmd = self._apply_slew_and_lpf(vel_cmd_raw, dt)

        # Yaw PID
        self.int_yaw *= float(np.exp(-self.int_yaw_leak * dt))
        self.int_yaw += yaw_err * dt
        self.int_yaw = float(np.clip(self.int_yaw, -self.ki_yaw_sat, self.ki_yaw_sat))

        if self.prev_yaw_err is None or dt <= 1e-9:
            yaw_err_deriv = 0.0
        else:
            dy = yaw_err - self.prev_yaw_err
            if dy > np.pi:
                dy -= 2.0 * np.pi
            elif dy < -np.pi:
                dy += 2.0 * np.pi
            yaw_err_deriv = np.clip(dy / dt, -8.0, 8.0)
        self.prev_yaw_err = yaw_err

        yaw_rate_unsat = self.kp_yaw * yaw_err + self.ki_yaw * self.int_yaw + self.kd_yaw * yaw_err_deriv
        yaw_rate_cmd = float(np.clip(yaw_rate_unsat, -self.max_yaw_rate, self.max_yaw_rate))
        if abs(yaw_rate_cmd) >= self.max_yaw_rate - 1e-6:
            self.int_yaw -= yaw_err * dt

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
            "yaw_scale": float(yaw_scale),
            "xy_scale": float(xy_scale),
            "yaw_rate_cmd": float(yaw_rate_cmd),
            "int_pos_body": self.int_pos_body.copy(),
            "int_yaw": float(self.int_yaw),
            "yaw_i_term": float(self.ki_yaw * self.int_yaw),
        }

        return (
            float(vel_cmd[0]),
            float(vel_cmd[1]),
            float(vel_cmd[2]),
            float(yaw_rate_cmd),
        )


_dob_controller = DOBController()


def get_integral_telemetry():
    d = _dob_controller.last_debug
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
        "yaw_i_term": float(d.get("yaw_i_term", 0.0)),
        "dob_comp_body": (float(dob_comp[0]), float(dob_comp[1]), float(dob_comp[2])),
        "vel_est_body": (float(vel_est[0]), float(vel_est[1]), float(vel_est[2])),
    }


def controller(state, target_pos, dt, wind_enabled=False):
    return _dob_controller.compute(state, target_pos, OUTER_DT, wind_enabled)
