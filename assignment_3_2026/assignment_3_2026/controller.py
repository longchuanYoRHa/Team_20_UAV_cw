import numpy as np

# ============================================================
# DOB-based outer-loop controller for the provided interface
# Input:
#   state      = [x, y, z, roll, pitch, yaw]  or  + [vx, vy, vz] world (m/s) if len>=9
#   target_pos = (x_d, y_d, z_d, yaw_d)
#   dt
# Output:
#   (vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd)
# ============================================================


def wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def rot_world_to_yaw_body(yaw):
    """
    Rotate a world-frame vector into yaw-aligned body frame.
    This matches the intended high-level command frame better than full body frame.
    """
    c = np.cos(yaw)
    s = np.sin(yaw)
    return np.array([
        [c,  s, 0.0],
        [-s, c, 0.0],
        [0.0, 0.0, 1.0]
    ])


class DOBController:
    def __init__(self):
        # ---------- outer-loop gains ----------
        # position -> desired velocity (xy tuned for 50 Hz ZOH + inner velocity loop delay)
        self.kp_pos = np.array([0.45, 0.45, 1.00])
        self.kd_vel = np.array([0.78, 0.78, 0.85])

        # yaw control
        self.kp_yaw = 1.4

        # ---------- DOB gains ----------
        # disturbance observer adaptation rate
        
        self.k_dob = np.array([0.35, 0.35, 0.25])
        self.dob_leak = np.array([0.15, 0.15, 0.10])
        self.k_comp = np.array([0.60, 0.60, 0.50])

        # ---------- command limits ----------
        self.max_vel = np.array([1.0, 1.0, 1.0])
        self.max_yaw_rate = 1.74533

        # ---------- internal states ----------
        self.prev_pos = None
        self.prev_yaw = None
        self.vel_est_world = np.zeros(3)
        self.vel_lpf_alpha = 0.15

        # estimated disturbance in yaw-body frame
        self.d_hat = np.zeros(3)

        # for debugging / test
        self.last_debug = {}

    def reset(self):
        self.prev_pos = None
        self.prev_yaw = None
        self.vel_est_world = np.zeros(3)
        self.d_hat = np.zeros(3)
        self.last_debug = {}

    def estimate_velocity(self, pos, dt):
        if self.prev_pos is None or dt <= 1e-6:
            self.prev_pos = pos.copy()
            return np.zeros(3)

        raw_vel = (pos - self.prev_pos) / dt
        self.prev_pos = pos.copy()

        # low-pass filtered velocity estimate
        self.vel_est_world = (
            self.vel_lpf_alpha * raw_vel
            + (1.0 - self.vel_lpf_alpha) * self.vel_est_world
        )
        return self.vel_est_world

    def update_dob(self, vel_cmd_body, vel_est_body, dt, wind_enabled):
        """
        Simple disturbance observer / estimator:
        If commanded velocity and observed velocity differ persistently,
        treat it as equivalent disturbance and compensate it.

        d_hat_dot = k_dob * (vel_cmd - vel_est) - leak * d_hat
        """
        if dt <= 1e-6:
            return

        vel_tracking_error = vel_cmd_body - vel_est_body

        scale = 1.0 if wind_enabled else 0.5
        d_dot = scale * self.k_dob * vel_tracking_error - self.dob_leak * self.d_hat
        self.d_hat += d_dot * dt

        # limit observer output to avoid runaway
        self.d_hat = np.clip(self.d_hat, -0.55, 0.55)

    def compute(self, state, target_pos, dt, wind_enabled=False):
        state = np.asarray(state, dtype=float).reshape(-1)
        x, y, z, roll, pitch, yaw = state[0:6]
        x_d, y_d, z_d, yaw_d = target_pos

        pos = np.array([x, y, z], dtype=float)
        pos_d = np.array([x_d, y_d, z_d], dtype=float)

        # 1) world-frame velocity: prefer simulator measurement (avoids 50 Hz diff lag vs inner loop)
        if state.size >= 9:
            self.prev_pos = pos.copy()
            vel_est_world = np.array(state[6:9], dtype=float)
            self.vel_est_world = vel_est_world
        else:
            vel_est_world = self.estimate_velocity(pos, dt)

        # 2) world-frame position error
        pos_err_world = pos_d - pos

        # 3) transform both position error and velocity estimate to yaw-aligned body frame
        R_w2b_yaw = rot_world_to_yaw_body(yaw)
        pos_err_body = R_w2b_yaw @ pos_err_world
        vel_est_body = R_w2b_yaw @ vel_est_world

        # 4) nominal outer-loop velocity command (P + D)
        vel_cmd_nominal = self.kp_pos * pos_err_body - self.kd_vel * vel_est_body

        # DOB only when wind is on — without wind, mismatch is mostly delay / estimation
        # error and feeds a false disturbance, which drives limit cycles.
        if wind_enabled:
            if np.linalg.norm(pos_err_body[0:2]) > 0.8:
                self.update_dob(vel_cmd_nominal, vel_est_body, dt, wind_enabled)
            vel_cmd = vel_cmd_nominal + self.k_comp * self.d_hat
        else:
            self.d_hat[:] = 0.0
            vel_cmd = vel_cmd_nominal.copy()

        # safety clipping
        vel_cmd = np.clip(vel_cmd, -self.max_vel, self.max_vel)

        # horizontal dead zone near setpoint (must apply to vel_cmd, not nominal only)
        if (
            np.linalg.norm(pos_err_body[0:2]) < 0.08
            and np.linalg.norm(vel_est_body[0:2]) < 0.08
        ):
            vel_cmd[0] = 0.0
            vel_cmd[1] = 0.0

        # yaw control
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


# ------------------------------------------------------------
# Global singleton required by the autograder-style interface
# ------------------------------------------------------------
_dob_controller = DOBController()


def controller(state, target_pos, dt, wind_enabled=False):
    # state format: [px, py, pz, roll, pitch, yaw] optional [vx, vy, vz] world linear velocity (m/s)
    # target_pos format: (x (m), y (m), z (m), yaw (radians))
    # dt: time step (s)
    # wind_enabled: boolean flag to indicate if wind disturbance should be considered in the control algorithm
    # return velocity command format: (velocity_x_setpoint (m/s), velocity_y_setpoint (m/s), velocity_z_setpoint (m/s), yaw_rate_setpoint (radians/s))
    return _dob_controller.compute(state, target_pos, dt, wind_enabled)