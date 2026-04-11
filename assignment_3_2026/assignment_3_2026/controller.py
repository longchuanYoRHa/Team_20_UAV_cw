import numpy as np

# ============================================================
# Outer-loop: yaw-aligned body-frame PID -> velocity setpoint + optional DOB.
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
        # 水平轴（yaw 对齐机体系下 x/y）共用同一组外环 PID；z 单独一组。
        # 细微减超调：略降 P/I，略增 D（仍保留末段收敛能力）
        kp_xy, kp_z = 0.53, 1.18
        ki_xy, ki_z = 0.2, 0.85
        kd_xy, kd_z = 0.30, 0.96
        ki_sat_xy, ki_sat_z = 0.48, 0.58
        self.kp_pos = np.array([kp_xy, kp_xy, kp_z])
        self.ki_pos = np.array([ki_xy, ki_xy, ki_z])
        self.kd_vel = np.array([kd_xy, kd_xy, kd_z])
        self.ki_pos_sat = np.array([ki_sat_xy, ki_sat_xy, ki_sat_z])

        self.kp_yaw = 2.05
        self.ki_yaw = 0.42
        self.kd_yaw = 0.16
        self.ki_yaw_sat = 0.18

        self.k_dob = np.array([0.35, 0.35, 0.25])
        self.dob_leak = np.array([0.15, 0.15, 0.10])
        self.k_comp = np.array([0.60, 0.60, 0.50])

        self.max_vel = np.array([1.08, 1.08, 1.08])
        self.max_yaw_rate = 1.74533

        self.prev_yaw = None
        self.prev_yaw_err = None
        self.int_pos_body = np.zeros(3)
        self.int_yaw = 0.0
        self.vel_est_world = np.zeros(3)
        self.prev_pos_error_world = None
        self.vel_est_lpf_alpha = 0.56

        self.d_hat = np.zeros(3)
        self.last_debug = {}

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

        self.int_pos_body += pos_err_body * dt
        self.int_pos_body = np.clip(
            self.int_pos_body, -self.ki_pos_sat, self.ki_pos_sat
        )

        p_term = self.kp_pos * pos_err_body
        i_term = self.ki_pos * self.int_pos_body
        d_term = -self.kd_vel * vel_est_body
        vel_cmd_nominal = p_term + i_term + d_term

        if wind_enabled:
            if np.linalg.norm(pos_err_body[0:2]) > 0.8:
                self.update_dob(vel_cmd_nominal, vel_est_body, dt, wind_enabled)
            vel_cmd = vel_cmd_nominal + self.k_comp * self.d_hat
        else:
            self.d_hat[:] = 0.0
            vel_cmd = vel_cmd_nominal.copy()

        vel_cmd = np.clip(vel_cmd, -self.max_vel, self.max_vel)

        yaw_err = wrap_to_pi(yaw_d - yaw)
        self.int_yaw += yaw_err * dt
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
            "vel_cmd_final": vel_cmd.copy(),
            "yaw_err": yaw_err,
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
        return {"pid_i_body": (0.0, 0.0, 0.0), "yaw_i_term": 0.0}
    pi = np.asarray(d["pid_i_body"], dtype=float).ravel()
    return {
        "pid_i_body": (float(pi[0]), float(pi[1]), float(pi[2])),
        "yaw_i_term": float(d["yaw_i_term"]),
    }


def controller(state, target_pos, dt, wind_enabled=False):
    return _dob_controller.compute(state, target_pos, dt, wind_enabled)
