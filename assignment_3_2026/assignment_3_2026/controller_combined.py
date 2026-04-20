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
    * Millisecond timestamp -> internal dt conversion (real flight friendly).
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
# Outer-loop: yaw-aligned body-frame PID -> velocity setpoint + optional DOB
# (identical tuning/logic as controller.py's DOBController)
# ============================================================
class DOBController:
    def __init__(self):
        kp_xy, kp_z = 0.672, 1.18
        ki_xy, ki_z = 0.095, 0.6
        kd_xy, kd_z = 0.504, 0.96
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

        # Position integral exponential decay lambda [1/s]: larger -> faster fade.
        self.int_pos_leak = np.array([2.5, 2.5, 1.5])
        self.int_yaw_leak = 2.0

        self.max_vel = np.array([0.68, 0.68, 0.68])
        self.max_yaw_rate = 1.74533
        # Only allow horizontal body-frame velocity after |yaw_d - yaw| < this (rad).
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

        # z always integrates; horizontal integrates only after yaw alignment.
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
            self.kp_yaw * yaw_err
            + self.ki_yaw * self.int_yaw
            + self.kd_yaw * yaw_err_deriv
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
# Practical wrapper: millisecond timestamp -> dt, waypoint scheduling, CSV log
# (from controller_practical.py, now using the inlined DOBController)
# ============================================================
class PracticalController:
    HARD_CLAMP = 100.0

    def __init__(self):
        self._dob = DOBController()
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
        self._dob.reset()
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

        vx, vy, vz, yaw_rate = self._dob.compute(
            state_arr, active_target, dt, wind_enabled=self._wind_enabled
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
        d = self._dob.last_debug or {}
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
    """Return the most recent debug dict from the underlying DOBController."""
    return dict(_practical._dob.last_debug)


# ---------- Integral telemetry (kept for compatibility with controller.py API) ----------
def get_integral_telemetry():
    """Integral contributions from the last controller() call (for plotting)."""
    d = _practical._dob.last_debug
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
