"""
Combined controller for the AMR CW3 Tello + Vicon lab.

This module merges two previously separate files:
    - controller.py            (outer-loop DOB + PID logic, dt in seconds)
    - controller_practical.py  (lab wrapper: ms timestamps, waypoints, CSV log)

into a single self-contained file adapted for *real* flight input, i.e. the
lab harness signature:

    controller(state, target_pos, timestamp) -> (vx, vy, vz, yaw_rate)

where
    state      = [x, y, z, roll, pitch, yaw]   positions in m, angles in rad
    target_pos = (x_d, y_d, z_d, yaw_d)        m / m / m / rad
    timestamp  = milliseconds (monotonic)

Returned values
    vx, vy, vz  m/s  (clamped to +/- 100 by the practical safety layer)
    yaw_rate    rad/s (clamped to +/- 100 by the practical safety layer)

Features
    * Yaw-aligned body-frame PID with leaky integral and optional DOB.
    * Millisecond timestamp for scheduling/logging; control core uses fixed dt.
    * Optional multi-waypoint scheduling (set_waypoints / auto-advance on arrival).
    * CSV telemetry log (disable with enable_logging(False) if not wanted).

No external project imports are required; only numpy is used at runtime.
"""

from __future__ import annotations

import os
import csv
import time
import numpy as np


# ============================================================
# Math helpers (from controller.py)
# ============================================================

def wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def rot_world_to_yaw_body(yaw):
    c = np.cos(yaw)
    s = np.sin(yaw)
    return np.array([
        [c,  s, 0.0],
        [-s, c, 0.0],
        [0.0, 0.0, 1.0],
    ])


# ============================================================
# Model-aware DOB on XY + controller.py's Z/yaw verbatim.
# Ported from the current `controller.py` into this combined lab wrapper.
# ============================================================
FIXED_DT = 0.0833  # 与 run.py 的 50 Hz 外环节拍对齐 (同 controller.py)


def smoothstep(x, edge0, edge1):
    """Hermite smoothstep; 在 [edge0, edge1] 区间从 0 平滑过渡到 1."""
    if edge1 <= edge0 + 1e-9:
        return 0.0 if x < edge0 else 1.0
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return float(t * t * (3.0 - 2.0 * t))


def soft_deadzone(x, eps):
    """连续型软死区: |x|<=eps 近似为 0，|x|>>eps 近似为 x 本身；全程 C1 光滑。"""
    if eps <= 1e-9:
        return x
    return x * (1.0 - np.exp(-(x / eps) ** 2))


# ---------- 内环辨识参数 (写死，来源：inner_loop_id_output/inner_loop_params.json) ----------
INNER = {
    "xy": {
        "K": 0.9791332892264056,
        "wn": 1.1688119784531361,
        "zeta": 0.42594658984568196,
        "tau_z": 0.2052931630745509,
        "L": 0.1153442442895058,
    },
    "z": {
        "K": 0.9564045415827129,
        "tau": 0.1659312509862405,
        "L": 0.05598021989635007,
    },
}


class _SecondOrderZeroDelay:
    """G(s) = K * wn^2 * (tau_z s + 1) / (s^2 + 2 zeta wn s + wn^2) * e^{-L s}"""

    def __init__(self, K, wn, zeta, tau_z, delay_s, hist_len=24):
        self.K = float(K)
        self.wn = float(wn)
        self.zeta = float(zeta)
        self.tau_z = float(tau_z)
        self.delay_s = max(0.0, float(delay_s))
        self.x1 = 0.0
        self.x2 = 0.0
        self._hist = [0.0] * int(max(2, hist_len))
        self._hist_len = len(self._hist)
        self._hist_write_idx = 0

    def reset(self):
        self.x1 = 0.0
        self.x2 = 0.0
        for i in range(self._hist_len):
            self._hist[i] = 0.0
        self._hist_write_idx = 0

    def _delayed(self, u, dt):
        # Ring buffer write
        self._hist[self._hist_write_idx] = float(u)
        self._hist_write_idx = (self._hist_write_idx + 1) % self._hist_len

        d = self.delay_s / max(dt, 1e-6)
        n0 = int(np.floor(d))
        frac = d - n0

        # Newest sample is at write_idx - 1
        newest_idx = (self._hist_write_idx - 1) % self._hist_len
        idx_newer = (newest_idx - n0) % self._hist_len
        idx_older = (idx_newer - 1) % self._hist_len
        return (1.0 - frac) * self._hist[idx_newer] + frac * self._hist[idx_older]

    def step(self, u, dt):
        dt = max(dt, 1e-6)
        u_d = self._delayed(u, dt)
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


class _SecondOrderLowpass:
    """H(s) = wq^2 / (s^2 + 2 zeta_q wq s + wq^2)"""

    def __init__(self, wq, zeta_q=0.95):
        self.wq = max(1e-3, float(wq))
        self.zeta_q = float(zeta_q)
        self.x1 = 0.0
        self.x2 = 0.0

    def reset(self):
        self.x1 = 0.0
        self.x2 = 0.0

    def step(self, u, dt):
        dt = max(dt, 1e-6)
        substeps = max(1, int(np.ceil(dt / 0.02)))
        h = dt / substeps
        w2 = self.wq * self.wq
        twozw = 2.0 * self.zeta_q * self.wq
        for _ in range(substeps):
            x1_dot = self.x2
            x2_dot = -w2 * self.x1 - twozw * self.x2 + w2 * float(u)
            self.x1 += h * x1_dot
            self.x2 += h * x2_dot
        return self.x1


class ModelDOBxyController:
    def __init__(self):
        kp_xy = 0.38
        ki_xy = 0.040
        kd_xy = 0.30
        ki_sat_xy = 0.40

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
        self.int_yaw_leak = 2.0
        self.max_yaw_rate = 1.74533
        self.yaw_align_tol = 0.167

        self.int_pos_leak = np.array([1.8, 1.8, 1.5], dtype=float)

        self.max_vel = np.array([0.78, 0.78, 0.78], dtype=float)
        self.max_xy_speed = 0.78

        self.vel_est_world = np.zeros(3, dtype=float)
        self.prev_pos_error_world = None
        self.vel_est_lpf_alpha = 0.56

        mx = INNER["xy"]
        self.model_x = _SecondOrderZeroDelay(mx["K"], mx["wn"], mx["zeta"], mx["tau_z"], mx["L"])
        self.model_y = _SecondOrderZeroDelay(mx["K"], mx["wn"], mx["zeta"], mx["tau_z"], mx["L"])
        self.model_vel_xy = np.zeros(2, dtype=float)

        self.q_filter_x = _SecondOrderLowpass(wq=0.80, zeta_q=0.95)
        self.q_filter_y = _SecondOrderLowpass(wq=0.80, zeta_q=0.95)
        self.d_hat_xy = np.zeros(2, dtype=float)
        self.d_hat_max = 0.40
        self.k_comp_xy = 0.80

        self.cmd_xy_lpf_beta = 0.70
        self.prev_xy_cmd = np.zeros(2, dtype=float)

        self.dob_zero_edge = 0.020
        self.dob_full_edge = 0.080
        self.d_term_zero_edge = 0.010
        self.d_term_full_edge = 0.045
        self.xy_pos_err_soft_eps = 0.004

        self.p_boost_full_edge = 0.002
        self.p_boost_zero_edge = 0.020
        self.p_boost_gain = 0.60

        self.int_pos_body = np.zeros(3, dtype=float)
        self.int_yaw = 0.0
        self.prev_yaw_err = None
        self.last_debug = {}

    def reset(self):
        self.vel_est_world[:] = 0.0
        self.prev_pos_error_world = None
        self.int_pos_body[:] = 0.0
        self.int_yaw = 0.0
        self.prev_yaw_err = None
        self.prev_xy_cmd[:] = 0.0
        self.d_hat_xy[:] = 0.0
        self.model_vel_xy[:] = 0.0
        self.model_x.reset()
        self.model_y.reset()
        self.q_filter_x.reset()
        self.q_filter_y.reset()
        self.last_debug = {}

    def _update_vel_est_from_pos_error(self, pos_err_world, dt):
        if dt > 1e-6 and self.prev_pos_error_world is not None:
            pos_derivative = (pos_err_world - self.prev_pos_error_world) / dt
            pos_derivative = np.clip(pos_derivative, -3.0, 3.0)
            raw_vel = -pos_derivative
            a = self.vel_est_lpf_alpha
            self.vel_est_world = a * raw_vel + (1.0 - a) * self.vel_est_world
        self.prev_pos_error_world = pos_err_world.copy()

    def _update_xy_dob(self, vel_cmd_nom_xy, vel_est_xy, dt, wind_enabled):
        self.model_vel_xy[0] = self.model_x.step(vel_cmd_nom_xy[0], dt)
        self.model_vel_xy[1] = self.model_y.step(vel_cmd_nom_xy[1], dt)

        mismatch = vel_est_xy - self.model_vel_xy
        mismatch = np.clip(mismatch, -1.0, 1.0)

        enable_scale = 1.0 if wind_enabled else 0.15
        u_x = enable_scale * mismatch[0]
        u_y = enable_scale * mismatch[1]

        self.d_hat_xy[0] = float(np.clip(self.q_filter_x.step(u_x, dt), -self.d_hat_max, self.d_hat_max))
        self.d_hat_xy[1] = float(np.clip(self.q_filter_y.step(u_y, dt), -self.d_hat_max, self.d_hat_max))

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

        self.int_pos_body[2] += pos_err_body[2] * dt
        yaw_i_scale = 1.0 - smoothstep(abs(yaw_err), self.yaw_align_tol, 3.0 * self.yaw_align_tol)
        self.int_pos_body[0] += yaw_i_scale * pos_err_body[0] * dt
        self.int_pos_body[1] += yaw_i_scale * pos_err_body[1] * dt
        self.int_pos_body *= np.exp(-self.int_pos_leak * dt)
        self.int_pos_body = np.clip(self.int_pos_body, -self.ki_pos_sat, self.ki_pos_sat)

        pos_err_body_for_pid = pos_err_body.copy()
        pos_err_body_for_pid[0] = soft_deadzone(pos_err_body_for_pid[0], self.xy_pos_err_soft_eps)
        pos_err_body_for_pid[1] = soft_deadzone(pos_err_body_for_pid[1], self.xy_pos_err_soft_eps)

        p_term = self.kp_pos * pos_err_body_for_pid
        i_term = self.ki_pos * self.int_pos_body
        d_term_full = -self.kd_vel * vel_est_body

        xy_err_norm = float(np.linalg.norm(pos_err_body[0:2]))

        near_factor = 1.0 - smoothstep(xy_err_norm, self.p_boost_full_edge, self.p_boost_zero_edge)
        p_boost = 1.0 + self.p_boost_gain * near_factor
        p_term[0] *= p_boost
        p_term[1] *= p_boost

        d_fade_xy = smoothstep(xy_err_norm, self.d_term_zero_edge, self.d_term_full_edge)
        d_term = d_term_full.copy()
        d_term[0] *= d_fade_xy
        d_term[1] *= d_fade_xy

        vel_cmd_nominal = p_term + i_term + d_term

        self._update_xy_dob(
            vel_cmd_nominal[0:2],
            vel_est_body[0:2],
            dt,
            wind_enabled,
        )

        dob_fade = smoothstep(xy_err_norm, self.dob_zero_edge, self.dob_full_edge)
        dob_comp_xy = self.k_comp_xy * dob_fade * self.d_hat_xy
        dob_comp = np.array([dob_comp_xy[0], dob_comp_xy[1], 0.0], dtype=float)

        vel_cmd = vel_cmd_nominal + dob_comp

        xy_cmd = vel_cmd[0:2].copy()
        xy_cmd = self.cmd_xy_lpf_beta * self.prev_xy_cmd + (1.0 - self.cmd_xy_lpf_beta) * xy_cmd
        self.prev_xy_cmd = xy_cmd.copy()
        vel_cmd[0] = xy_cmd[0]
        vel_cmd[1] = xy_cmd[1]

        vel_cmd = np.clip(vel_cmd, -self.max_vel, self.max_vel)
        xy_speed = float(np.linalg.norm(vel_cmd[0:2]))
        if xy_speed > self.max_xy_speed and xy_speed > 1e-9:
            vel_cmd[0:2] *= self.max_xy_speed / xy_speed

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
            self.kp_yaw * yaw_err
            + self.ki_yaw * self.int_yaw
            + self.kd_yaw * yaw_err_deriv
        )
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
            "vel_model_body": np.array([self.model_vel_xy[0], self.model_vel_xy[1], 0.0], dtype=float),
            "vel_cmd_nominal": vel_cmd_nominal.copy(),
            "pid_p_body": p_term.copy(),
            "pid_i_body": i_term.copy(),
            "pid_d_body": d_term.copy(),
            "d_hat": np.array([self.d_hat_xy[0], self.d_hat_xy[1], 0.0], dtype=float),
            "dob_comp_body": dob_comp.copy(),
            "dob_fade": dob_fade,
            "d_term_fade": d_fade_xy,
            "p_boost": p_boost,
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


# ============================================================
# Telemetry logger (from controller_practical.py)
# ============================================================
class TelemetryLogger:
    HEADER = [
        "timestamp_ms", "dt_s",
        "x", "y", "z", "roll", "pitch", "yaw",
        "x_d", "y_d", "z_d", "yaw_d",
        "err_x_w", "err_y_w", "err_z_w", "err_yaw",
        "vel_est_x_w", "vel_est_y_w", "vel_est_z_w",
        "vel_est_x_b", "vel_est_y_b", "vel_est_z_b",
        "cmd_vx", "cmd_vy", "cmd_vz", "cmd_yaw_rate",
        "p_x_b", "p_y_b", "p_z_b",
        "i_x_b", "i_y_b", "i_z_b",
        "d_x_b", "d_y_b", "d_z_b",
        "dob_x_b", "dob_y_b", "dob_z_b",
        "yaw_aligned", "wp_index",
    ]

    def __init__(self, path: str):
        self.path = path
        self._initialised = False

    def _ensure_file(self):
        if self._initialised:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
        need_header = (not os.path.exists(self.path)) or os.path.getsize(self.path) == 0
        if need_header:
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(self.HEADER)
        self._initialised = True

    def write(self, row: list):
        self._ensure_file()
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow(row)


# ============================================================
# Practical wrapper: millisecond timestamp, waypoint scheduling, CSV log
# (from controller_practical.py, now using the inlined ModelDOBxyController)
# ============================================================
class PracticalController:
    HARD_CLAMP = 100.0

    def __init__(self):
        self._ctrl = ModelDOBxyController()
        # Real flights: no simulated wind -> DOB path off by default.
        self._wind_enabled = False

        self._prev_ts_ms = None
        self._start_ts_ms = None

        # Each waypoint: (x, y, z, yaw, pos_tol_m, yaw_tol_rad, hold_s)
        self._waypoints: list[tuple] = []
        self._wp_idx = 0
        self._wp_hold_start_ms = None

        default_log = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "flight_logs",
            time.strftime("flight_%Y%m%d_%H%M%S.csv"),
        )
        self._logger = TelemetryLogger(default_log)
        self._logging_enabled = True

    # ---------- Configuration API ----------
    def set_log_path(self, path: str):
        self._logger = TelemetryLogger(path)

    def enable_logging(self, enabled: bool):
        self._logging_enabled = bool(enabled)

    def set_wind_enabled(self, enabled: bool):
        self._wind_enabled = bool(enabled)

    def set_waypoints(self, waypoints, pos_tol=0.15, yaw_tol=0.17, hold_s=1.0):
        """Set a multi-waypoint mission.

        Each element of `waypoints` is (x, y, z, yaw) or
        (x, y, z, yaw, tol, yaw_tol, hold_s).
        """
        normalised = []
        for wp in waypoints:
            if len(wp) == 4:
                x, y, z, yaw = wp
                normalised.append((x, y, z, yaw, pos_tol, yaw_tol, hold_s))
            elif len(wp) == 7:
                normalised.append(tuple(wp))
            else:
                raise ValueError(
                    "waypoint must be (x,y,z,yaw) or (x,y,z,yaw,tol,yaw_tol,hold)"
                )
        self._waypoints = normalised
        self._wp_idx = 0
        self._wp_hold_start_ms = None

    def reset(self):
        self._ctrl.reset()
        self._prev_ts_ms = None
        self._start_ts_ms = None
        self._wp_idx = 0
        self._wp_hold_start_ms = None

    # ---------- Internal helpers ----------
    def _pick_target(self, pos, yaw, target_pos_arg, timestamp_ms):
        if not self._waypoints:
            return tuple(target_pos_arg)

        wp = self._waypoints[self._wp_idx]
        x_d, y_d, z_d, yaw_d, pos_tol, yaw_tol, hold_s = wp

        pos_err = np.linalg.norm(np.array([x_d, y_d, z_d]) - pos)
        yaw_err = abs(wrap_to_pi(yaw_d - yaw))

        if pos_err <= pos_tol and yaw_err <= yaw_tol:
            if self._wp_hold_start_ms is None:
                self._wp_hold_start_ms = timestamp_ms
            elif (timestamp_ms - self._wp_hold_start_ms) / 1000.0 >= hold_s:
                if self._wp_idx < len(self._waypoints) - 1:
                    self._wp_idx += 1
                    self._wp_hold_start_ms = None
        else:
            self._wp_hold_start_ms = None

        return (x_d, y_d, z_d, yaw_d)

    # ---------- Main entry ----------
    def compute(self, state, target_pos, timestamp_ms):
        state_arr = np.asarray(state, dtype=float).reshape(-1)
        if state_arr.size < 6:
            return (0.0, 0.0, 0.0, 0.0)

        x, y, z, roll, pitch, yaw = state_arr[0:6]
        pos = np.array([x, y, z], dtype=float)

        ts_ms = float(timestamp_ms)
        if self._prev_ts_ms is None:
            dt = 1.0 / 12.0  # ~12 Hz assumption on first sample
            self._start_ts_ms = ts_ms
        else:
            dt = (ts_ms - self._prev_ts_ms) / 1000.0
            if dt <= 1e-4:
                dt = 1e-3
            elif dt > 0.5:
                dt = 0.5
        self._prev_ts_ms = ts_ms

        active_target = self._pick_target(pos, yaw, target_pos, ts_ms)

        # Match the current controller.py behavior: fixed effective outer-loop dt.
        vx, vy, vz, yaw_rate = self._ctrl.compute(
            state_arr, active_target, FIXED_DT, wind_enabled=self._wind_enabled
        )

        vx = float(np.clip(vx, -self.HARD_CLAMP, self.HARD_CLAMP))
        vy = float(np.clip(vy, -self.HARD_CLAMP, self.HARD_CLAMP))
        vz = float(np.clip(vz, -self.HARD_CLAMP, self.HARD_CLAMP))
        yaw_rate = float(np.clip(yaw_rate, -self.HARD_CLAMP, self.HARD_CLAMP))

        if self._logging_enabled:
            self._log_row(ts_ms, dt, state_arr, active_target,
                          (vx, vy, vz, yaw_rate))

        return (vx, vy, vz, yaw_rate)

    def _log_row(self, ts_ms, dt, state_arr, active_target, cmd):
        d = self._ctrl.last_debug or {}
        pos_err_w = d.get("pos_err_world", np.zeros(3))
        vel_w = d.get("vel_est_world", np.zeros(3))
        vel_b = d.get("vel_est_body", np.zeros(3))
        p_b = d.get("pid_p_body", np.zeros(3))
        i_b = d.get("pid_i_body", np.zeros(3))
        d_b = d.get("pid_d_body", np.zeros(3))
        dob_b = d.get("dob_comp_body", np.zeros(3))
        yaw_aligned = int(bool(d.get("yaw_aligned", False)))
        yaw_err = float(d.get("yaw_err", 0.0))

        row = [
            f"{ts_ms:.3f}", f"{dt:.6f}",
            *[f"{v:.6f}" for v in state_arr[0:6]],
            *[f"{v:.6f}" for v in active_target],
            *[f"{v:.6f}" for v in pos_err_w], f"{yaw_err:.6f}",
            *[f"{v:.6f}" for v in vel_w],
            *[f"{v:.6f}" for v in vel_b],
            *[f"{v:.6f}" for v in cmd],
            *[f"{v:.6f}" for v in p_b],
            *[f"{v:.6f}" for v in i_b],
            *[f"{v:.6f}" for v in d_b],
            *[f"{v:.6f}" for v in dob_b],
            yaw_aligned, self._wp_idx,
        ]
        self._logger.write(row)


# ============================================================
# Module-level singleton + lab-compatible entry point
# ============================================================
_practical = PracticalController()


def controller(state, target_pos, timestamp):
    """Lab harness entry point.

    Args:
        state:      [x, y, z, roll, pitch, yaw]
        target_pos: (x_d, y_d, z_d, yaw_d)
        timestamp:  milliseconds (monotonic)

    Returns:
        (vx, vy, vz, yaw_rate)
    """
    return _practical.compute(state, target_pos, timestamp)


# ---------- Convenience configuration helpers ----------
def set_waypoints(waypoints, pos_tol=0.15, yaw_tol=0.17, hold_s=1.0):
    _practical.set_waypoints(waypoints, pos_tol=pos_tol, yaw_tol=yaw_tol, hold_s=hold_s)


def set_log_path(path: str):
    _practical.set_log_path(path)


def enable_logging(enabled: bool):
    _practical.enable_logging(enabled)


def set_wind_enabled(enabled: bool):
    _practical.set_wind_enabled(enabled)


def reset_controller():
    _practical.reset()


def get_debug():
    """Return the most recent debug dict from the underlying ModelDOBxyController."""
    return dict(_practical._ctrl.last_debug)


# ---------- Integral telemetry (kept for compatibility with controller.py API) ----------
def get_integral_telemetry():
    """Integral contributions from the last controller() call (for plotting)."""
    d = _practical._ctrl.last_debug
    if not d:
        return {
            "pid_i_body": (0.0, 0.0, 0.0),
            "yaw_i_term": 0.0,
            "dob_comp_body": (0.0, 0.0, 0.0),
            "vel_est_body": (0.0, 0.0, 0.0),
            "vel_model_body": (0.0, 0.0, 0.0),
        }
    pi = np.asarray(d["pid_i_body"], dtype=float).ravel()
    dob_comp = np.asarray(d.get("dob_comp_body", (0.0, 0.0, 0.0)), dtype=float).ravel()
    vel_est = np.asarray(d.get("vel_est_body", (0.0, 0.0, 0.0)), dtype=float).ravel()
    vel_model = np.asarray(d.get("vel_model_body", (0.0, 0.0, 0.0)), dtype=float).ravel()
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
        "vel_model_body": (
            float(vel_model[0]),
            float(vel_model[1]),
            float(vel_model[2]),
        ),
    }
