"""
Practical controller for the AMR CW3 Tello + Vicon lab.

Signature required by the lab:
    controller(state, target_pos, timestamp) -> (vx, vy, vz, yaw_rate)

where
    state      = [x, y, z, roll, pitch, yaw]   positions in m, angles in rad
    target_pos = (x_d, y_d, z_d, yaw_d)        m / m / m / rad
    timestamp  = milliseconds (monotonic)

Returned values
    vx, vy, vz  m/s  (clamped to [-100, 100])
    yaw_rate    rad/s (clamped to [-100, 100])

Features
    * Reuses the tuned DOBController from controller.py (same gains / logic).
    * Computes dt internally from the supplied millisecond timestamp.
    * Optional multi-waypoint scheduling (set_waypoints / auto-advance on arrival).
    * Writes a CSV log with state, commands, estimated velocity, errors and
      integral contributions. The log path can be configured via set_log_path().

The module-level `controller` function is what the lab harness should import.
"""

from __future__ import annotations

import os
import csv
import time
import numpy as np

from controller import DOBController, wrap_to_pi


# ---------------------------------------------------------------------------
# Logger: appends one CSV row per controller invocation.
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Practical controller wrapper.
# ---------------------------------------------------------------------------
class PracticalController:
    # Hard safety clamp required by the brief.
    HARD_CLAMP = 100.0

    def __init__(self):
        self._dob = DOBController()
        # Real flights: disturbance observer is not needed (no simulated wind),
        # we keep the PID path and leave DOB off by default.
        self._wind_enabled = False

        self._prev_ts_ms = None
        self._start_ts_ms = None

        # Waypoint scheduling.
        # Each waypoint: (x, y, z, yaw, pos_tol_m, yaw_tol_rad, hold_s)
        self._waypoints: list[tuple] = []
        self._wp_idx = 0
        self._wp_hold_start_ms = None

        # Logging.
        default_log = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "flight_logs",
            time.strftime("flight_%Y%m%d_%H%M%S.csv"),
        )
        self._logger = TelemetryLogger(default_log)
        self._logging_enabled = True

    # ------------------------------------------------------------------
    # Configuration API
    # ------------------------------------------------------------------
    def set_log_path(self, path: str):
        self._logger = TelemetryLogger(path)

    def enable_logging(self, enabled: bool):
        self._logging_enabled = bool(enabled)

    def set_wind_enabled(self, enabled: bool):
        self._wind_enabled = bool(enabled)

    def set_waypoints(self, waypoints, pos_tol=0.15, yaw_tol=0.17, hold_s=1.0):
        """Set a multi-waypoint mission.

        Each element of `waypoints` is (x, y, z, yaw) or (x, y, z, yaw, tol, yaw_tol, hold_s).
        Defaults apply when per-waypoint tolerances/hold are not provided.
        """
        normalised = []
        for wp in waypoints:
            if len(wp) == 4:
                x, y, z, yaw = wp
                normalised.append((x, y, z, yaw, pos_tol, yaw_tol, hold_s))
            elif len(wp) == 7:
                normalised.append(tuple(wp))
            else:
                raise ValueError("waypoint must be (x,y,z,yaw) or (x,y,z,yaw,tol,yaw_tol,hold)")
        self._waypoints = normalised
        self._wp_idx = 0
        self._wp_hold_start_ms = None

    def reset(self):
        self._dob.reset()
        self._prev_ts_ms = None
        self._start_ts_ms = None
        self._wp_idx = 0
        self._wp_hold_start_ms = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _pick_target(self, pos, yaw, target_pos_arg, timestamp_ms):
        """Return the current active target, auto-advancing waypoints when held."""
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

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def compute(self, state, target_pos, timestamp_ms):
        state_arr = np.asarray(state, dtype=float).reshape(-1)
        if state_arr.size < 6:
            return (0.0, 0.0, 0.0, 0.0)

        x, y, z, roll, pitch, yaw = state_arr[0:6]
        pos = np.array([x, y, z], dtype=float)

        ts_ms = float(timestamp_ms)
        if self._prev_ts_ms is None:
            dt = 1.0 / 12.0   # first sample assumption (~12 Hz like the sim)
            self._start_ts_ms = ts_ms
        else:
            dt = (ts_ms - self._prev_ts_ms) / 1000.0
            if dt <= 1e-4:
                dt = 1e-3
            elif dt > 0.5:
                dt = 0.5
        self._prev_ts_ms = ts_ms

        # Resolve active target (waypoint-aware).
        active_target = self._pick_target(pos, yaw, target_pos, ts_ms)

        vx, vy, vz, yaw_rate = self._dob.compute(
            state_arr, active_target, dt, wind_enabled=self._wind_enabled
        )

        # Safety clamp required by the brief.
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


# ---------------------------------------------------------------------------
# Module-level singleton + lab-compatible entry point.
# ---------------------------------------------------------------------------
_practical = PracticalController()


def controller(state, target_pos, timestamp):
    """Entry point required by the lab harness.

    Args:
        state:      [x, y, z, roll, pitch, yaw]
        target_pos: (x_d, y_d, z_d, yaw_d)
        timestamp:  milliseconds

    Returns:
        (vx, vy, vz, yaw_rate)
    """
    return _practical.compute(state, target_pos, timestamp)


# Convenience re-exports so the lab user can configure from outside.
def set_waypoints(waypoints, pos_tol=0.15, yaw_tol=0.17, hold_s=1.0):
    _practical.set_waypoints(waypoints, pos_tol=pos_tol, yaw_tol=yaw_tol, hold_s=hold_s)


def set_log_path(path: str):
    _practical.set_log_path(path)


def enable_logging(enabled: bool):
    _practical.enable_logging(enabled)


def reset_controller():
    _practical.reset()


def get_debug():
    """Return the most recent debug dict from the underlying DOBController."""
    return dict(_practical._dob.last_debug)
