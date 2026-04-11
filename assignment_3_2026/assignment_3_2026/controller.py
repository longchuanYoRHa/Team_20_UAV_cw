import numpy as np

# ============================================================
# Outer-loop controller (strict 6-D state; velocity from 50 Hz position only)
# Input:
#   state      = [x, y, z, roll, pitch, yaw]
#   target_pos = (x_d, y_d, z_d, yaw_d)
#   dt
# Output:
#   (vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd)
#
# World velocity: finite difference of world-frame position error
#   e = p_d - p  =>  de/dt ≈ -v  (constant target)  =>  v_hat = -clip((e - e_prev)/dt).
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
        # horizontal: conservative P, small D (D uses noisy differenced velocity)
        self.kp_pos = np.array([0.32, 0.32, 1.00])
        self.kd_vel = np.array([0.38, 0.38, 0.72])

        self.kp_yaw = 1.4

        self.k_dob = np.array([0.35, 0.35, 0.25])
        self.dob_leak = np.array([0.15, 0.15, 0.10])
        self.k_comp = np.array([0.60, 0.60, 0.50])

        self.max_vel = np.array([1.0, 1.0, 1.0])
        self.max_yaw_rate = 1.74533

        self.prev_yaw = None
        self.vel_est_world = np.zeros(3)
        self.prev_pos_error_world = None

        self.cmd_xy_lpf_beta = 0.28
        self.vel_cmd_xy_filt = np.zeros(2)
        self.vel_cmd_xy_prev_out = np.zeros(2)
        self.max_xy_slew = 5.5

        self.d_hat = np.zeros(3)
        self.last_debug = {}

    def reset(self):
        self.prev_yaw = None
        self.vel_est_world = np.zeros(3)
        self.prev_pos_error_world = None
        self.vel_cmd_xy_filt = np.zeros(2)
        self.vel_cmd_xy_prev_out = np.zeros(2)
        self.d_hat = np.zeros(3)
        self.last_debug = {}

    def _update_vel_est_from_pos_error(self, pos_err_world, dt):
        """Estimate world velocity from successive position errors (cascade-style)."""
        if dt > 1e-6 and self.prev_pos_error_world is not None:
            pos_derivative = (pos_err_world - self.prev_pos_error_world) / dt
            pos_derivative = np.clip(pos_derivative, -3.0, 3.0)
            self.vel_est_world = -pos_derivative
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

        vel_cmd_nominal = self.kp_pos * pos_err_body - self.kd_vel * vel_est_body

        if wind_enabled:
            if np.linalg.norm(pos_err_body[0:2]) > 0.8:
                self.update_dob(vel_cmd_nominal, vel_est_body, dt, wind_enabled)
            vel_cmd = vel_cmd_nominal + self.k_comp * self.d_hat
        else:
            self.d_hat[:] = 0.0
            vel_cmd = vel_cmd_nominal.copy()

        vel_cmd = np.clip(vel_cmd, -self.max_vel, self.max_vel)

        b = self.cmd_xy_lpf_beta
        vel_cmd[0] = b * vel_cmd[0] + (1.0 - b) * self.vel_cmd_xy_filt[0]
        vel_cmd[1] = b * vel_cmd[1] + (1.0 - b) * self.vel_cmd_xy_filt[1]
        self.vel_cmd_xy_filt = vel_cmd[0:2].copy()

        step = self.max_xy_slew * dt
        vel_cmd[0] = self.vel_cmd_xy_prev_out[0] + np.clip(
            vel_cmd[0] - self.vel_cmd_xy_prev_out[0], -step, step
        )
        vel_cmd[1] = self.vel_cmd_xy_prev_out[1] + np.clip(
            vel_cmd[1] - self.vel_cmd_xy_prev_out[1], -step, step
        )
        self.vel_cmd_xy_prev_out = vel_cmd[0:2].copy()

        if (
            np.linalg.norm(pos_err_body[0:2]) < 0.10
            and np.linalg.norm(vel_est_body[0:2]) < 0.10
        ):
            vel_cmd[0] = 0.0
            vel_cmd[1] = 0.0
            self.vel_cmd_xy_filt[:] = 0.0
            self.vel_cmd_xy_prev_out[:] = 0.0

        yaw_err = wrap_to_pi(yaw_d - yaw)
        yaw_rate_cmd = np.clip(self.kp_yaw * yaw_err, -self.max_yaw_rate, self.max_yaw_rate)

        self.prev_yaw = yaw

        self.last_debug = {
            "pos": pos.copy(),
            "target": pos_d.copy(),
            "pos_err_world": pos_err_world.copy(),
            "pos_err_body": pos_err_body.copy(),
            "vel_est_world": vel_est_world.copy(),
            "vel_est_body": vel_est_body.copy(),
            "vel_cmd_nominal": vel_cmd_nominal.copy(),
            "d_hat": self.d_hat.copy(),
            "vel_cmd_final": vel_cmd.copy(),
            "yaw_err": yaw_err,
            "yaw_rate_cmd": yaw_rate_cmd,
        }

        return (
            float(vel_cmd[0]),
            float(vel_cmd[1]),
            float(vel_cmd[2]),
            float(yaw_rate_cmd),
        )


_dob_controller = DOBController()


def controller(state, target_pos, dt, wind_enabled=False):
    # state format: [position_x (m), position_y (m), position_z (m), roll (radians), pitch (radians), yaw (radians)]
    # target_pos format: (x (m), y (m), z (m), yaw (radians))
    # dt: time step (s)
    # wind_enabled: boolean flag to indicate if wind disturbance should be considered in the control algorithm
    # return velocity command format: (velocity_x_setpoint (m/s), velocity_y_setpoint (m/s), velocity_z_setpoint (m/s), yaw_rate_setpoint (radians/s))
    return _dob_controller.compute(state, target_pos, dt, wind_enabled)
