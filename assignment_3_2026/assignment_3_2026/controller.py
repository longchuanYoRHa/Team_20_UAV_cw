import numpy as np

# ============================================================
# Outer-loop: yaw-aligned body-frame PID -> velocity setpoint + optional DOB.
# First align yaw to target (|yaw_err| less than threshold), then allow horizontal (x,y) velocity commands; z is tracked throughout.
# No command shaping (no xy LPF / slew / brake / hold); only clip to max_vel.
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


class DOBController:
    def __init__(self):
        # Horizontal axes (x/y in yaw-aligned body frame) share the same outer loop PID; z has its own.
        # Slight reduction of overshoot: slightly reduce P/I, slightly increase D (still retains final convergence ability)
        kp_xy, kp_z = 0.672, 1.18
        ki_xy, ki_z = 0.095 , 0.6
        kd_xy, kd_z = 0.464, 0.96
        ki_sat_xy, ki_sat_z = 0.60, 0.58
        self.kp_pos = np.array([kp_xy, kp_xy, kp_z])
        self.ki_pos = np.array([ki_xy, ki_xy, ki_z])
        self.kd_vel = np.array([kd_xy, kd_xy, kd_z])
        self.ki_pos_sat = np.array([ki_sat_xy, ki_sat_xy, ki_sat_z])

        self.kp_yaw = 2.05
        self.ki_yaw = 0.42
        self.kd_yaw = 0.16
        self.ki_yaw_sat = 0.18

        self.k_dob = np.array([0.10, 0.10, 0.00])
        self.dob_leak = np.array([0.55, 0.55, 0.40])
        self.k_comp = np.array([0.16, 0.16, 0.00])

        # Outer loop position integral exponential decay λ [1/s]: each step int *= exp(-λ*dt), then clip. Increasing λ → integral memory fades faster, reduces overshoot
        self.int_pos_leak = np.array([2.5, 2.5, 1.5])
        self.int_yaw_leak = 2.0

        self.max_vel = np.array([1.08, 1.08, 1.08])
        self.max_yaw_rate = 1.74533
        # First turn: only allow horizontal velocity commands in body frame after |yaw_d - yaw| is less than this threshold (rad, about 5°)
        self.yaw_align_tol = 0.167

        self.prev_yaw = None
        self.prev_yaw_err = None
        self.int_pos_body = np.zeros(3)
        self.int_yaw = 0.0
        self.vel_est_world = np.zeros(3)
        self.prev_pos_error_world = None
        self.vel_est_lpf_alpha = 0.56

        self.d_hat = np.zeros(3)
        self.last_debug = {}

        # Tuning mode: XY uses P-term only (disable XY I/D/DOB).
        # Z and yaw behavior remain unchanged.
        self.xy_p_only = False

    def reset(self):
        self.prev_yaw = None
        self.prev_yaw_err = None
        self.int_pos_body = np.zeros(3)
        self.int_yaw = 0.0
        self.vel_est_world = np.zeros(3)
        self.prev_pos_error_world = None
        self.d_hat = np.zeros(3)
        self.last_debug = {}

    def _update_vel_est_from_pos_error(self, pos_err_world, dt):
        if dt > 1e-6 and self.prev_pos_error_world is not None:
            pos_derivative = (pos_err_world - self.prev_pos_error_world) / dt
            pos_derivative = np.clip(pos_derivative, -3.0, 3.0)
            raw_vel = -pos_derivative
            a = self.vel_est_lpf_alpha
            self.vel_est_world = a * raw_vel + (1.0 - a) * self.vel_est_world
        self.prev_pos_error_world = pos_err_world.copy()

    def update_dob(self, vel_cmd_body, vel_est_body, dt, wind_enabled):
        if dt <= 1e-6:
            return
        vel_tracking_error = vel_cmd_body - vel_est_body
        scale = 1.0 if wind_enabled else 0.5
        d_dot = scale * self.k_dob * vel_tracking_error - self.dob_leak * self.d_hat
        self.d_hat += d_dot * dt
        self.d_hat = np.clip(self.d_hat, -0.55, 0.55)

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

        # z is always integrated; horizontal is only integrated after yaw alignment to avoid xy integral saturation when turning
        self.int_pos_body[2] += pos_err_body[2] * dt
        if yaw_aligned and (not self.xy_p_only):
            self.int_pos_body[0] += pos_err_body[0] * dt
            self.int_pos_body[1] += pos_err_body[1] * dt
        self.int_pos_body *= np.exp(-self.int_pos_leak * dt)
        self.int_pos_body = np.clip(
            self.int_pos_body, -self.ki_pos_sat, self.ki_pos_sat
        )

        p_term = self.kp_pos * pos_err_body
        i_term = self.ki_pos * self.int_pos_body
        d_term = -self.kd_vel * vel_est_body

        if self.xy_p_only:
            # Force XY to be strictly P-only regardless of configured gains/states.
            self.int_pos_body[0:2] = 0.0
            i_term[0:2] = 0.0
            d_term[0:2] = 0.0

        vel_cmd_nominal = p_term + i_term + d_term

        dob_comp = np.zeros(3)
        if wind_enabled and (not self.xy_p_only):
            xy_err = np.linalg.norm(pos_err_body[0:2])

            if xy_err <= 0.25:
                self.update_dob(vel_cmd_nominal, vel_est_body, dt, wind_enabled)
            else:
                self.d_hat *= np.exp(-self.dob_leak * dt)

            dob_comp = self.k_comp * self.d_hat
            vel_cmd = vel_cmd_nominal + dob_comp
            # vel_cmd = vel_cmd_nominal.copy()
        else:
            self.d_hat[:] = 0.0
            vel_cmd = vel_cmd_nominal.copy()

        vel_cmd = np.clip(vel_cmd, -self.max_vel, self.max_vel)
        if not yaw_aligned:
            vel_cmd[0] = 0.0
            vel_cmd[1] = 0.0

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
            self.kp_yaw * yaw_err + self.ki_yaw * self.int_yaw + self.kd_yaw * yaw_err_deriv
        )
        yaw_rate_cmd = float(np.clip(yaw_rate_unsat, -self.max_yaw_rate, self.max_yaw_rate))
        if abs(yaw_rate_cmd) >= self.max_yaw_rate - 1e-6:
            self.int_yaw -= yaw_err * dt

        self.prev_yaw = yaw

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
            "d_hat": self.d_hat.copy(),
            "dob_comp_body": dob_comp.copy(),
            "vel_cmd_final": vel_cmd.copy(),
            "yaw_err": yaw_err,
            "yaw_aligned": yaw_aligned,
            "yaw_rate_cmd": yaw_rate_cmd,
            "int_pos_body": self.int_pos_body.copy(),
            "int_yaw": self.int_yaw,
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
    """Integral contributions from the last controller() call (for plotting)."""
    d = _dob_controller.last_debug
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
        "dob_comp_body": (
            float(dob_comp[0]),
            float(dob_comp[1]),
            float(dob_comp[2]),
        ),
        "vel_est_body": (
            float(vel_est[0]),
            float(vel_est[1]),
            float(vel_est[2]),
        ),
    }


def controller(state, target_pos, dt, wind_enabled=False):
    dt = 0.0833
    return _dob_controller.compute(state, target_pos, dt, wind_enabled)
