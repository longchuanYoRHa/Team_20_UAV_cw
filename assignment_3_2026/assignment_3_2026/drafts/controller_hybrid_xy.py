import numpy as np

# ============================================================
# Hybrid outer-loop controller for AERO60492 UAV coursework
#
# Design rationale:
# - Keep Z / yaw logic close to the previously stable source controller.
# - Retune XY only, because identified inner-loop dynamics show XY is the
#   true bottleneck: low bandwidth (~0.25 Hz), noticeable delay (~0.115 s),
#   and light damping. Therefore XY outer loop must be conservative.
# - Use DOB only as a LOW-FREQUENCY bias / wind compensator for XY.
#   It is disabled without wind and disabled again near the target to avoid
#   false compensation / limit cycles caused by model mismatch and velocity
#   estimation noise.
# - Add XY command shaping (LPF + slew) and a near-target hold region.
#
# Interface must remain:
#   controller(state, target_pos, dt, wind_enabled=False)
#   -> (vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd)
# ============================================================


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


class HybridXYController:
    def __init__(self):
        # ========================================================
        # XY: newly retuned conservative loop adapted to identified
        #     slow inner-loop dynamics.
        # ========================================================
        kp_xy = 0.34
        ki_xy = 0.035
        kd_xy = 0.27
        ki_sat_xy = 0.18

        # ========================================================
        # Z / yaw: stay close to the previously stable source version.
        # ========================================================
        kp_z = 1.18
        ki_z = 0.60
        kd_z = 0.96
        ki_sat_z = 0.58

        self.kp_pos = np.array([kp_xy, kp_xy, kp_z], dtype=float)
        self.ki_pos = np.array([ki_xy, ki_xy, ki_z], dtype=float)
        self.kd_vel = np.array([kd_xy, kd_xy, kd_z], dtype=float)
        self.ki_pos_sat = np.array([ki_sat_xy, ki_sat_xy, ki_sat_z], dtype=float)

        self.kp_yaw = 2.05
        self.ki_yaw = 0.42
        self.kd_yaw = 0.16
        self.ki_yaw_sat = 0.18

        # Integral leak: faster on XY to reduce low-speed chatter.
        self.int_pos_leak = np.array([2.8, 2.8, 1.5], dtype=float)
        self.int_yaw_leak = 2.0

        # XY velocity estimator LPF
        self.vel_est_lpf_alpha = 0.52

        # Command limits
        self.max_vel = np.array([1.08, 1.08, 1.08], dtype=float)
        self.max_yaw_rate = 1.74533
        self.max_xy_speed = 0.80  # stricter than source to respect slow XY inner loop

        # First turn then translate horizontally
        self.yaw_align_tol = 0.167  # about 9.6 deg

        # XY command shaping (important for slow inner loop)
        self.cmd_xy_lpf_beta = 0.92
        self.max_xy_slew = 18.0  # m/s^2 equivalent in outer-loop command domain

        # Near-target hold region: stop XY chatter when error and speed are both small
        self.xy_hold_err = 0.050
        self.xy_hold_vel = 0.030

        # Mild anisotropic deadzone to suppress cross-axis chatter from tiny Y error
        self.xy_axis_deadzone = np.array([0.020, 0.030], dtype=float)

        # Conditional XY DOB: low-frequency wind bias estimator only.
        self.k_dob_xy = np.array([0.08, 0.08], dtype=float)
        self.dob_leak_xy = np.array([0.50, 0.50], dtype=float)
        self.k_comp_xy = np.array([0.13, 0.13], dtype=float)
        self.dob_enable_radius = 0.15
        self.dob_disable_radius = 0.08
        self.dob_cmd_min = 0.03
        self.dob_max = 0.25

        # Internal states
        self.prev_yaw_err = None
        self.prev_pos_error_world = None
        self.vel_est_world = np.zeros(3, dtype=float)
        self.int_pos_body = np.zeros(3, dtype=float)
        self.int_yaw = 0.0
        self.d_hat_xy = np.zeros(2, dtype=float)
        self.prev_xy_cmd = np.zeros(2, dtype=float)
        self.last_debug = {}

    def reset(self):
        self.prev_yaw_err = None
        self.prev_pos_error_world = None
        self.vel_est_world = np.zeros(3, dtype=float)
        self.int_pos_body = np.zeros(3, dtype=float)
        self.int_yaw = 0.0
        self.d_hat_xy = np.zeros(2, dtype=float)
        self.prev_xy_cmd = np.zeros(2, dtype=float)
        self.last_debug = {}

    def _update_vel_est_from_pos_error(self, pos_err_world, dt):
        if dt > 1e-6 and self.prev_pos_error_world is not None:
            pos_derivative = (pos_err_world - self.prev_pos_error_world) / dt
            pos_derivative = np.clip(pos_derivative, -3.0, 3.0)
            raw_vel = -pos_derivative
            a = self.vel_est_lpf_alpha
            self.vel_est_world = a * raw_vel + (1.0 - a) * self.vel_est_world
        self.prev_pos_error_world = pos_err_world.copy()

    def _shape_xy_command(self, xy_cmd, dt):
        # LPF
        xy_cmd = self.cmd_xy_lpf_beta * self.prev_xy_cmd + (1.0 - self.cmd_xy_lpf_beta) * xy_cmd

        # Slew limit
        if dt > 1e-6:
            delta = xy_cmd - self.prev_xy_cmd
            max_delta = self.max_xy_slew * dt
            delta = np.clip(delta, -max_delta, max_delta)
            xy_cmd = self.prev_xy_cmd + delta

        # XY norm limit
        nxy = np.linalg.norm(xy_cmd)
        if nxy > self.max_xy_speed and nxy > 1e-9:
            xy_cmd = xy_cmd * (self.max_xy_speed / nxy)

        self.prev_xy_cmd = xy_cmd.copy()
        return xy_cmd

    def _update_xy_dob(self, xy_cmd_nom, xy_vel_est, xy_err_norm, wind_enabled, dt):
        # No wind -> no DOB. Leak quickly to zero to prevent false compensation.
        if (not wind_enabled) or dt <= 1e-6:
            self.d_hat_xy *= np.exp(-1.8 * dt)
            return

        # Near target or very small command -> do not keep learning; leak toward zero.
        if (xy_err_norm < self.dob_enable_radius) or (np.linalg.norm(xy_cmd_nom) < self.dob_cmd_min):
            self.d_hat_xy *= np.exp(-self.dob_leak_xy * 1.5 * dt)
            if xy_err_norm < self.dob_disable_radius:
                self.d_hat_xy *= np.exp(-2.2 * dt)
            return

        vel_tracking_error = xy_cmd_nom - xy_vel_est
        d_dot = self.k_dob_xy * vel_tracking_error - self.dob_leak_xy * self.d_hat_xy
        self.d_hat_xy += d_dot * dt
        self.d_hat_xy = np.clip(self.d_hat_xy, -self.dob_max, self.dob_max)

    def compute(self, state, target_pos, dt, wind_enabled=False):
        state = np.asarray(state, dtype=float).reshape(-1)
        if state.size < 6:
            return (0.0, 0.0, 0.0, 0.0)

        x, y, z, roll, pitch, yaw = state[0:6]
        x_d, y_d, z_d, yaw_d = target_pos

        pos = np.array([x, y, z], dtype=float)
        pos_d = np.array([x_d, y_d, z_d], dtype=float)
        pos_err_world = pos_d - pos

        self._update_vel_est_from_pos_error(pos_err_world, dt)
        vel_est_world = self.vel_est_world.copy()

        R_w2b_yaw = rot_world_to_yaw_body(yaw)
        pos_err_body = R_w2b_yaw @ pos_err_world
        vel_est_body = R_w2b_yaw @ vel_est_world

        yaw_err = wrap_to_pi(yaw_d - yaw)
        yaw_aligned = abs(yaw_err) < self.yaw_align_tol

        # Small XY deadzone to avoid translating tiny transformed noise into chatter
        for i in range(2):
            if abs(pos_err_body[i]) < self.xy_axis_deadzone[i]:
                pos_err_body[i] = 0.0

        # Integrators
        self.int_pos_body[2] += pos_err_body[2] * dt
        if yaw_aligned:
            self.int_pos_body[0] += pos_err_body[0] * dt
            self.int_pos_body[1] += pos_err_body[1] * dt

        self.int_pos_body *= np.exp(-self.int_pos_leak * dt)
        self.int_pos_body = np.clip(self.int_pos_body, -self.ki_pos_sat, self.ki_pos_sat)

        # Near-target XY hold: if close enough and slow enough, freeze XY integrator and command zero.
        xy_err_norm = float(np.linalg.norm(pos_err_body[0:2]))
        xy_vel_norm = float(np.linalg.norm(vel_est_body[0:2]))
        xy_hold = yaw_aligned and (xy_err_norm < self.xy_hold_err) and (xy_vel_norm < self.xy_hold_vel)
        if xy_hold:
            self.int_pos_body[0:2] *= 0.90

        # Nominal PID terms
        p_term = self.kp_pos * pos_err_body
        i_term = self.ki_pos * self.int_pos_body
        d_term = -self.kd_vel * vel_est_body

        vel_cmd_nominal = p_term + i_term + d_term

        # XY conditional low-frequency DOB (wind only)
        self._update_xy_dob(vel_cmd_nominal[0:2], vel_est_body[0:2], xy_err_norm, wind_enabled, dt)
        dob_comp_xy = self.k_comp_xy * self.d_hat_xy if wind_enabled else np.zeros(2, dtype=float)

        # Compose final command: XY modified, Z preserved close to source behavior
        vel_cmd = vel_cmd_nominal.copy()
        vel_cmd[0:2] += dob_comp_xy

        # If yaw is not aligned, do not translate horizontally.
        if not yaw_aligned:
            vel_cmd[0:2] = 0.0

        # Near-target XY hold beats DOB/nominal control to prevent limit cycles.
        if xy_hold:
            vel_cmd[0:2] = 0.0
            self.prev_xy_cmd *= 0.8

        # Shape XY command only; Z stays mostly as previous stable implementation.
        vel_cmd[0:2] = self._shape_xy_command(vel_cmd[0:2], dt)

        # Global clipping
        vel_cmd = np.clip(vel_cmd, -self.max_vel, self.max_vel)

        # Yaw loop (kept close to source)
        self.int_yaw += yaw_err * dt
        self.int_yaw *= float(np.exp(-self.int_yaw_leak * dt))
        self.int_yaw = float(np.clip(self.int_yaw, -self.ki_yaw_sat, self.ki_yaw_sat))

        if self.prev_yaw_err is None or dt <= 1e-6:
            yaw_err_deriv = 0.0
        else:
            delta = yaw_err - self.prev_yaw_err
            if delta > np.pi:
                delta -= 2.0 * np.pi
            elif delta < -np.pi:
                delta += 2.0 * np.pi
            yaw_err_deriv = np.clip(delta / dt, -10.0, 10.0)
        self.prev_yaw_err = yaw_err

        yaw_rate_unsat = (
            self.kp_yaw * yaw_err +
            self.ki_yaw * self.int_yaw +
            self.kd_yaw * yaw_err_deriv
        )
        yaw_rate_cmd = float(np.clip(yaw_rate_unsat, -self.max_yaw_rate, self.max_yaw_rate))
        if abs(yaw_rate_cmd) >= self.max_yaw_rate - 1e-6:
            self.int_yaw -= yaw_err * dt

        # Telemetry / plotting compatibility
        dob_comp_body = np.array([dob_comp_xy[0], dob_comp_xy[1], 0.0], dtype=float)
        self.last_debug = {
            "pos": pos.copy(),
            "target": pos_d.copy(),
            "pos_err_world": pos_err_world.copy(),
            "pos_err_body": pos_err_body.copy(),
            "vel_est_world": vel_est_world.copy(),
            "vel_est_body": vel_est_body.copy(),
            "vel_cmd_nominal": vel_cmd_nominal.copy(),
            "pid_p_body": p_term.copy(),
            "pid_i_body": i_term.copy(),
            "pid_d_body": d_term.copy(),
            "d_hat": np.array([self.d_hat_xy[0], self.d_hat_xy[1], 0.0], dtype=float),
            "dob_comp_body": dob_comp_body.copy(),
            "vel_cmd_final": vel_cmd.copy(),
            "yaw_err": yaw_err,
            "yaw_aligned": yaw_aligned,
            "yaw_rate_cmd": yaw_rate_cmd,
            "int_pos_body": self.int_pos_body.copy(),
            "int_yaw": self.int_yaw,
            "yaw_i_term": float(self.ki_yaw * self.int_yaw),
            "xy_hold": xy_hold,
        }

        return (
            float(vel_cmd[0]),
            float(vel_cmd[1]),
            float(vel_cmd[2]),
            float(yaw_rate_cmd),
        )


_controller = HybridXYController()


def get_integral_telemetry():
    d = _controller.last_debug
    if not d:
        return {
            "pid_i_body": (0.0, 0.0, 0.0),
            "yaw_i_term": 0.0,
            "dob_comp_body": (0.0, 0.0, 0.0),
            "vel_est_body": (0.0, 0.0, 0.0),
        }
    pi = np.asarray(d["pid_i_body"], dtype=float).ravel()
    dob_comp = np.asarray(d.get("dob_comp_body", (0.0, 0.0, 0.0)), dtype=float).ravel()
    vel_est = np.asarray(d.get("vel_est_body", (0.0, 0.0, 0.0)), dtype=float).ravel()
    return {
        "pid_i_body": (float(pi[0]), float(pi[1]), float(pi[2])),
        "yaw_i_term": float(d["yaw_i_term"]),
        "dob_comp_body": (float(dob_comp[0]), float(dob_comp[1]), float(dob_comp[2])),
        "vel_est_body": (float(vel_est[0]), float(vel_est[1]), float(vel_est[2])),
    }


def controller(state, target_pos, dt, wind_enabled=False):
    # Lock the effective outer-loop dt to the value that best matches the
    # current run/telemetry setup discussed in this project.
    dt = 0.0833
    return _controller.compute(state, target_pos, dt, wind_enabled)
