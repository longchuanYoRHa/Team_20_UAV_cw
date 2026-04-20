import numpy as np
from collections import deque

# ============================================================
# Model-aided outer-loop controller for AERO60492 UAV coursework.
#
# Design intent:
# - Keep the required interface unchanged.
# - Use identified inner-loop models as nominal plants.
# - Horizontal channels (x/y): use identified 2nd-order + zero + delay model
#   to predict body-frame velocity response.
# - Vertical channel (z): use identified 1st-order + delay model.
# - Disturbance compensation is based on model mismatch filtered through a
#   second-order Q-filter (low-bandwidth DOB), instead of directly integrating
#   velocity tracking error.
#
# Practical notes for this simulator:
# - controller() fixes dt = 0.0833 s to match the effective outer-loop timing
#   used in the user's workflow.
# - The x/y inner loop is very slow and lightly damped, so the outer loop is
#   intentionally conservative and the DOB only targets low-frequency bias.
# ============================================================


FIXED_DT = 0.0833


def wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def rot_world_to_yaw_body(yaw):
    c = np.cos(yaw)
    s = np.sin(yaw)
    return np.array([
        [c, s, 0.0],
        [-s, c, 0.0],
        [0.0, 0.0, 1.0],
    ])


class SecondOrderZeroDelayModel:
    """Continuous model integrated with Euler substeps.

    G(s) = K * wn^2 * (tau_z s + 1) / (s^2 + 2 zeta wn s + wn^2) * e^{-Ls}

    Realization:
        x1_dot = x2
        x2_dot = -wn^2 x1 - 2 zeta wn x2 + wn^2 u_d
        y = K * (x1 + tau_z * x2)
    """

    def __init__(self, K, wn, zeta, tau_z, delay_s):
        self.K = float(K)
        self.wn = float(wn)
        self.zeta = float(zeta)
        self.tau_z = float(tau_z)
        self.delay_s = max(0.0, float(delay_s))
        self.x1 = 0.0
        self.x2 = 0.0
        self.history = deque([0.0] * 16, maxlen=16)

    def reset(self):
        self.x1 = 0.0
        self.x2 = 0.0
        self.history.clear()
        self.history.extend([0.0] * 16)

    def _get_delayed_u(self, current_u, dt):
        self.history.append(float(current_u))
        d = self.delay_s / max(dt, 1e-6)
        # Use linear interpolation between past samples.
        n0 = int(np.floor(d))
        frac = d - n0
        hist = list(self.history)
        idx_newer = len(hist) - 1 - n0
        idx_older = idx_newer - 1
        idx_newer = max(0, min(len(hist) - 1, idx_newer))
        idx_older = max(0, min(len(hist) - 1, idx_older))
        u_newer = hist[idx_newer]
        u_older = hist[idx_older]
        return (1.0 - frac) * u_newer + frac * u_older

    def step(self, current_u, dt):
        dt = max(dt, 1e-6)
        u_d = self._get_delayed_u(current_u, dt)
        substeps = max(1, int(np.ceil(dt / 0.01)))
        h = dt / substeps
        wn2 = self.wn * self.wn
        two_zeta_wn = 2.0 * self.zeta * self.wn
        for _ in range(substeps):
            x1_dot = self.x2
            x2_dot = -wn2 * self.x1 - two_zeta_wn * self.x2 + wn2 * u_d
            self.x1 += h * x1_dot
            self.x2 += h * x2_dot
        return self.K * (self.x1 + self.tau_z * self.x2)


class FirstOrderDelayModel:
    """Continuous model integrated with Euler substeps.

    G(s) = K / (tau s + 1) * e^{-Ls}
    """

    def __init__(self, K, tau, delay_s):
        self.K = float(K)
        self.tau = max(1e-4, float(tau))
        self.delay_s = max(0.0, float(delay_s))
        self.x = 0.0
        self.history = deque([0.0] * 16, maxlen=16)

    def reset(self):
        self.x = 0.0
        self.history.clear()
        self.history.extend([0.0] * 16)

    def _get_delayed_u(self, current_u, dt):
        self.history.append(float(current_u))
        d = self.delay_s / max(dt, 1e-6)
        n0 = int(np.floor(d))
        frac = d - n0
        hist = list(self.history)
        idx_newer = len(hist) - 1 - n0
        idx_older = idx_newer - 1
        idx_newer = max(0, min(len(hist) - 1, idx_newer))
        idx_older = max(0, min(len(hist) - 1, idx_older))
        u_newer = hist[idx_newer]
        u_older = hist[idx_older]
        return (1.0 - frac) * u_newer + frac * u_older

    def step(self, current_u, dt):
        dt = max(dt, 1e-6)
        u_d = self._get_delayed_u(current_u, dt)
        substeps = max(1, int(np.ceil(dt / 0.01)))
        h = dt / substeps
        for _ in range(substeps):
            self.x += h * ((self.K * u_d - self.x) / self.tau)
        return self.x


class SecondOrderLowpass:
    """Q-filter / low-pass mismatch filter.

    H(s) = wq^2 / (s^2 + 2 zeta_q wq s + wq^2)
    """

    def __init__(self, wq, zeta_q=0.9):
        self.wq = max(1e-3, float(wq))
        self.zeta_q = float(zeta_q)
        self.x1 = 0.0
        self.x2 = 0.0

    def reset(self):
        self.x1 = 0.0
        self.x2 = 0.0

    def step(self, u, dt):
        dt = max(dt, 1e-6)
        substeps = max(1, int(np.ceil(dt / 0.01)))
        h = dt / substeps
        w2 = self.wq * self.wq
        twozw = 2.0 * self.zeta_q * self.wq
        for _ in range(substeps):
            x1_dot = self.x2
            x2_dot = -w2 * self.x1 - twozw * self.x2 + w2 * float(u)
            self.x1 += h * x1_dot
            self.x2 += h * x2_dot
        return self.x1


class ModelAidedDOBController:
    def __init__(self):
        # ---------------- Outer-loop gains ----------------
        # Conservative xy gains to respect identified inner-loop BW ~ 0.25 Hz.
        kp_xy, kp_z = 0.28, 0.95
        ki_xy, ki_z = 0.030, 0.30
        kd_xy, kd_z = 0.18, 0.42
        ki_sat_xy, ki_sat_z = 0.40, 0.45

        self.kp_pos = np.array([kp_xy, kp_xy, kp_z], dtype=float)
        self.ki_pos = np.array([ki_xy, ki_xy, ki_z], dtype=float)
        self.kd_vel = np.array([kd_xy, kd_xy, kd_z], dtype=float)
        self.ki_pos_sat = np.array([ki_sat_xy, ki_sat_xy, ki_sat_z], dtype=float)

        # Integrator leakage; xy kept fairly leaky to avoid exciting slow inner loop.
        self.int_pos_leak = np.array([1.8, 1.8, 0.9], dtype=float)

        # ---------------- Yaw loop ----------------
        self.kp_yaw = 2.00
        self.ki_yaw = 0.36
        self.kd_yaw = 0.12
        self.ki_yaw_sat = 0.16
        self.int_yaw_leak = 1.2
        self.max_yaw_rate = 1.74533
        self.yaw_align_tol = 0.18

        # ---------------- Command shaping ----------------
        self.max_vel = np.array([1.0, 1.0, 1.0], dtype=float)
        self.max_xy_speed = 0.95
        self.xy_deadzone = 0.02
        self.xy_hold_deadzone = 0.05
        self.cmd_xy_lpf_beta = 0.80
        self.cmd_z_lpf_beta = 0.65
        self.max_xy_slew = 4.5   # m/s^2 command slew in outer loop sample time
        self.max_z_slew = 5.0
        self.xy_brake_radius = 0.20
        self.xy_brake_gain = 0.55

        # ---------------- Velocity estimation ----------------
        self.vel_est_world = np.zeros(3, dtype=float)
        self.prev_pos = None
        self.vel_est_lpf_alpha = 0.50

        # ---------------- Identified nominal inner-loop models ----------------
        self.xy_models = [
            SecondOrderZeroDelayModel(
                K=0.9791332892264056,
                wn=1.1688119784531361,
                zeta=0.42594658984568196,
                tau_z=0.2052931630745509,
                delay_s=0.1153442442895058,
            ),
            SecondOrderZeroDelayModel(
                K=0.9791332892264056,
                wn=1.1688119784531361,
                zeta=0.42594658984568196,
                tau_z=0.2052931630745509,
                delay_s=0.1153442442895058,
            ),
        ]
        self.z_model = FirstOrderDelayModel(
            K=0.9564045415827129,
            tau=0.1659312509862405,
            delay_s=0.05598021989635007,
        )

        # ---------------- Q-filter DOB ----------------
        # Low bandwidth by construction; only intended to reject low-frequency wind bias.
        self.q_filters = [
            SecondOrderLowpass(wq=0.85, zeta_q=0.95),
            SecondOrderLowpass(wq=0.85, zeta_q=0.95),
            SecondOrderLowpass(wq=1.60, zeta_q=0.95),
        ]
        self.k_comp = np.array([0.42, 0.42, 0.18], dtype=float)
        self.d_hat = np.zeros(3, dtype=float)
        self.model_vel_body = np.zeros(3, dtype=float)

        # ---------------- Internal states ----------------
        self.int_pos_body = np.zeros(3, dtype=float)
        self.int_yaw = 0.0
        self.prev_yaw_err = None
        self.prev_cmd_body = np.zeros(3, dtype=float)
        self.last_debug = {}

    def reset(self):
        self.vel_est_world[:] = 0.0
        self.prev_pos = None
        self.int_pos_body[:] = 0.0
        self.int_yaw = 0.0
        self.prev_yaw_err = None
        self.prev_cmd_body[:] = 0.0
        self.d_hat[:] = 0.0
        self.model_vel_body[:] = 0.0
        self.last_debug = {}
        for model in self.xy_models:
            model.reset()
        self.z_model.reset()
        for qf in self.q_filters:
            qf.reset()

    def _update_vel_estimate(self, pos, dt):
        if self.prev_pos is None or dt <= 1e-6:
            self.prev_pos = pos.copy()
            self.vel_est_world[:] = 0.0
            return self.vel_est_world.copy()

        raw_vel = (pos - self.prev_pos) / dt
        raw_vel = np.clip(raw_vel, -3.0, 3.0)
        a = self.vel_est_lpf_alpha
        self.vel_est_world = a * raw_vel + (1.0 - a) * self.vel_est_world
        self.prev_pos = pos.copy()
        return self.vel_est_world.copy()

    def _shape_command(self, vel_cmd_body, dt, yaw_aligned):
        cmd = np.asarray(vel_cmd_body, dtype=float).copy()

        # Near-target deadzone on y to suppress cross-coupled chatter; mild on x.
        if abs(cmd[1]) < self.xy_deadzone:
            cmd[1] = 0.0
        if abs(cmd[0]) < 0.5 * self.xy_deadzone:
            cmd[0] = 0.0

        # Yaw-first gating for horizontal channels.
        if not yaw_aligned:
            cmd[0] = 0.0
            cmd[1] = 0.0

        # Brake down as the vehicle approaches the xy target.
        xy_mag = np.linalg.norm(cmd[:2])
        if xy_mag > 1e-9:
            err_mag = np.linalg.norm(self.last_debug.get("pos_err_body", np.zeros(3))[:2])
            if err_mag < self.xy_brake_radius:
                scale = self.xy_brake_gain + (1.0 - self.xy_brake_gain) * (err_mag / self.xy_brake_radius)
                cmd[:2] *= np.clip(scale, 0.0, 1.0)

        # LPF.
        cmd[:2] = self.cmd_xy_lpf_beta * self.prev_cmd_body[:2] + (1.0 - self.cmd_xy_lpf_beta) * cmd[:2]
        cmd[2] = self.cmd_z_lpf_beta * self.prev_cmd_body[2] + (1.0 - self.cmd_z_lpf_beta) * cmd[2]

        # Slew rate.
        dxy_max = self.max_xy_slew * dt
        dz_max = self.max_z_slew * dt
        delta_xy = np.clip(cmd[:2] - self.prev_cmd_body[:2], -dxy_max, dxy_max)
        delta_z = float(np.clip(cmd[2] - self.prev_cmd_body[2], -dz_max, dz_max))
        cmd[:2] = self.prev_cmd_body[:2] + delta_xy
        cmd[2] = self.prev_cmd_body[2] + delta_z

        # Magnitude clipping.
        cmd = np.clip(cmd, -self.max_vel, self.max_vel)
        xy_speed = np.linalg.norm(cmd[:2])
        if xy_speed > self.max_xy_speed:
            cmd[:2] *= self.max_xy_speed / max(xy_speed, 1e-9)

        # Final hold region.
        if np.linalg.norm(self.last_debug.get("pos_err_body", np.zeros(3))[:2]) < self.xy_hold_deadzone:
            cmd[0] *= 0.4
            cmd[1] *= 0.4

        self.prev_cmd_body = cmd.copy()
        return cmd

    def _update_nominal_models(self, cmd_body, dt):
        self.model_vel_body[0] = self.xy_models[0].step(cmd_body[0], dt)
        self.model_vel_body[1] = self.xy_models[1].step(cmd_body[1], dt)
        self.model_vel_body[2] = self.z_model.step(cmd_body[2], dt)
        return self.model_vel_body.copy()

    def _update_model_aided_dob(self, cmd_body, vel_est_body, dt, wind_enabled):
        mismatch = vel_est_body - self.model_vel_body
        mismatch = np.clip(mismatch, -1.5, 1.5)

        # With wind off, fade the estimate instead of forcing it to zero instantly.
        enable_scale = 1.0 if wind_enabled else 0.25
        for i in range(3):
            filt = self.q_filters[i].step(enable_scale * mismatch[i], dt)
            self.d_hat[i] = np.clip(filt, -0.45, 0.45)
        return mismatch

    def compute(self, state, target_pos, dt, wind_enabled=False):
        state = np.asarray(state, dtype=float).reshape(-1)
        if state.size < 6:
            return (0.0, 0.0, 0.0, 0.0)

        x, y, z, roll, pitch, yaw = state[:6]
        x_d, y_d, z_d, yaw_d = target_pos

        pos = np.array([x, y, z], dtype=float)
        pos_d = np.array([x_d, y_d, z_d], dtype=float)
        vel_est_world = self._update_vel_estimate(pos, dt)

        R_w2b_yaw = rot_world_to_yaw_body(yaw)
        pos_err_world = pos_d - pos
        pos_err_body = R_w2b_yaw @ pos_err_world
        vel_est_body = R_w2b_yaw @ vel_est_world

        # Save early for shaping stage.
        self.last_debug["pos_err_body"] = pos_err_body.copy()

        yaw_err = wrap_to_pi(yaw_d - yaw)
        yaw_aligned = abs(yaw_err) < self.yaw_align_tol

        # Integrator update.
        self.int_pos_body[2] += pos_err_body[2] * dt
        if yaw_aligned:
            self.int_pos_body[0] += pos_err_body[0] * dt
            self.int_pos_body[1] += pos_err_body[1] * dt
        self.int_pos_body *= np.exp(-self.int_pos_leak * dt)
        self.int_pos_body = np.clip(self.int_pos_body, -self.ki_pos_sat, self.ki_pos_sat)

        # Mild lateral deadzone near zero to suppress vy dithering from coordinate noise.
        if abs(pos_err_body[1]) < 0.03:
            pos_err_body[1] = 0.0

        p_term = self.kp_pos * pos_err_body
        i_term = self.ki_pos * self.int_pos_body
        d_term = -self.kd_vel * vel_est_body
        vel_cmd_nominal = p_term + i_term + d_term

        # Shape nominal command before it enters the internal nominal model.
        vel_cmd_nominal = self._shape_command(vel_cmd_nominal, dt, yaw_aligned)

        # Propagate the nominal identified inner-loop models with the shaped command.
        model_vel_body = self._update_nominal_models(vel_cmd_nominal, dt)

        # DOB from model mismatch.
        mismatch = self._update_model_aided_dob(vel_cmd_nominal, vel_est_body, dt, wind_enabled)
        dob_comp = self.k_comp * self.d_hat

        # Near the target, allow more compensation; far away, avoid letting DOB fight transients.
        xy_err_mag = np.linalg.norm(pos_err_body[:2])
        if xy_err_mag > 0.30:
            dob_comp[:2] *= 0.25
        elif xy_err_mag > 0.18:
            dob_comp[:2] *= 0.55

        vel_cmd = vel_cmd_nominal + dob_comp
        vel_cmd = np.clip(vel_cmd, -self.max_vel, self.max_vel)
        xy_speed = np.linalg.norm(vel_cmd[:2])
        if xy_speed > self.max_xy_speed:
            vel_cmd[:2] *= self.max_xy_speed / max(xy_speed, 1e-9)

        # Yaw PID.
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
            "vel_model_body": model_vel_body.copy(),
            "vel_model_error_body": mismatch.copy(),
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


_dob_controller = ModelAidedDOBController()


def get_integral_telemetry():
    d = _dob_controller.last_debug
    if not d:
        return {
            "pid_i_body": (0.0, 0.0, 0.0),
            "yaw_i_term": 0.0,
            "dob_comp_body": (0.0, 0.0, 0.0),
            "vel_est_body": (0.0, 0.0, 0.0),
            "vel_model_body": (0.0, 0.0, 0.0),
            "vel_model_error_body": (0.0, 0.0, 0.0),
        }
    pi = np.asarray(d.get("pid_i_body", (0.0, 0.0, 0.0)), dtype=float).ravel()
    dob_comp = np.asarray(d.get("dob_comp_body", (0.0, 0.0, 0.0)), dtype=float).ravel()
    vel_est = np.asarray(d.get("vel_est_body", (0.0, 0.0, 0.0)), dtype=float).ravel()
    vel_model = np.asarray(d.get("vel_model_body", (0.0, 0.0, 0.0)), dtype=float).ravel()
    vel_model_error = np.asarray(d.get("vel_model_error_body", (0.0, 0.0, 0.0)), dtype=float).ravel()
    return {
        "pid_i_body": (float(pi[0]), float(pi[1]), float(pi[2])),
        "yaw_i_term": float(d.get("yaw_i_term", 0.0)),
        "dob_comp_body": (float(dob_comp[0]), float(dob_comp[1]), float(dob_comp[2])),
        "vel_est_body": (float(vel_est[0]), float(vel_est[1]), float(vel_est[2])),
        "vel_model_body": (float(vel_model[0]), float(vel_model[1]), float(vel_model[2])),
        "vel_model_error_body": (
            float(vel_model_error[0]),
            float(vel_model_error[1]),
            float(vel_model_error[2]),
        ),
    }


def controller(state, target_pos, dt, wind_enabled=False):
    # Intentionally fixed to the effective outer-loop dt used by this project.
    dt = FIXED_DT
    return _dob_controller.compute(state, target_pos, dt, wind_enabled)
