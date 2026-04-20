"""Model-based outer-loop position controller for the Tello sim.

Design basis
------------
Identified inner-loop models (see inner_loop_id_output/INNER_LOOP_IDENTIFICATION.md)
feeding directly into the outer-loop gain selection:

    xy:  G(s) = K (tau_z s + 1) wn^2 / (s^2 + 2 zeta wn s + wn^2) * exp(-L s)
         K = 0.9791, wn = 1.169 rad/s, zeta = 0.426, tau_z = 0.2053,
         L  = 0.1153 s,  -3 dB BW = 1.58 rad/s (~0.25 Hz)

    z:   G(s) = K / (tau s + 1) * exp(-L s)
         K = 0.9564, tau = 0.1659, L = 0.0560,  BW = 6.03 rad/s (~0.96 Hz)

    yaw_rate: G(s) = K wn^2 / (s^2 + 2 zeta wn s + wn^2)
              K = 1.002, wn = 40.05, zeta = 0.180, BW = 60.8 rad/s

Outer loop is closed around position (angle for yaw), so the plant seen by
the outer PID is P(s) = G_inner(s) / s. Gains are chosen by loop-shaping
at the crossover freq wc so that |C(j wc) P(j wc)| = 1 with a
60 - 80 deg phase margin. wc is placed well below the inner bandwidth to
avoid exciting the lightly-damped xy pole pair.

Additional shaping layers on top of the pure PID:
  1. Second-order reference smoother on the xy position target (wn_ref =
     0.8 rad/s, zeta_ref = 0.9). This prevents a step target from
     directly driving the under-damped inner-loop poles.
  2. First-order smoother on z target (tau = 0.3 s) and yaw target
     (tau = 0.2 s). Generates smooth p_ref, v_ref_ff, a_ref.
  3. Velocity feed-forward v_ff = d(p_ref)/dt / K_inner - uses the reference
     model to cover the bulk of the command, leaving the PID to handle
     disturbances and initial condition mismatch only.
  4. First-order low-pass on the commanded xy velocity (tau_cmd = 1.0 s,
     i.e. cutoff ~1 rad/s). This is below wn_xy = 1.17 so the inner
     under-damped mode is never excited by rapid command changes.
  5. IMC-flavoured DOB: d_hat is updated with a Q-filter bandwidth tied to
     wc (1 rad/s for xy, 3 rad/s for z). The compensation enters the
     velocity command with K_comp = 1 / K_inner.

Structural preserved features (same as controller.py):
  * yaw-alignment gating: xy velocity is forced to 0 until |yaw_err| <
    yaw_align_tol; xy integrator is frozen during that phase.
  * leaky + saturated position integrator to avoid wind-up.
  * yaw-rate clamp with anti-windup on the yaw integrator.

Gains are derived, not hand-tuned. They are conservative by design because
xy has an under-damped inner mode and 115 ms delay; if the real plant
matches the identification closely, they can be nudged upward.
"""

from __future__ import annotations

import numpy as np


def wrap_to_pi(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def rot_world_to_yaw_body(yaw: float) -> np.ndarray:
    c = np.cos(yaw)
    s = np.sin(yaw)
    return np.array([
        [c,  s, 0.0],
        [-s, c, 0.0],
        [0.0, 0.0, 1.0],
    ])


# ----------------------------------------------------------------------
# Identified inner-loop model (frozen constants, read directly from
# inner_loop_id_output/inner_loop_params.json).
# ----------------------------------------------------------------------
INNER_XY = {"K": 0.9791, "wn": 1.169, "zeta": 0.426, "tau_z": 0.2053, "L": 0.1153}
INNER_Z  = {"K": 0.9564, "tau": 0.1659, "L": 0.0560}
INNER_YAW_RATE = {"K": 1.0019, "wn": 40.05, "zeta": 0.180, "L": 0.0}


class ReferenceShaper2:
    """Second-order critical-ish low pass: smooth step targets so that the
    under-damped xy inner loop is not directly excited. Produces p_ref,
    v_ref (for feed-forward) in world coordinates."""

    def __init__(self, wn: float, zeta: float):
        self.wn = float(wn)
        self.zeta = float(zeta)
        self.p = None  # shape (d,), lazy init on first call
        self.v = None

    def reset(self):
        self.p = None
        self.v = None

    def step(self, target: np.ndarray, dt: float):
        target = np.asarray(target, dtype=float)
        if self.p is None:
            self.p = target.copy()
            self.v = np.zeros_like(target)
            return self.p.copy(), self.v.copy()
        if dt <= 1e-6:
            return self.p.copy(), self.v.copy()
        a = self.wn * self.wn * (target - self.p) - 2.0 * self.zeta * self.wn * self.v
        self.v = self.v + a * dt
        self.p = self.p + self.v * dt
        return self.p.copy(), self.v.copy()


class ReferenceShaper1:
    """First-order low pass on a scalar target."""

    def __init__(self, tau: float):
        self.tau = float(tau)
        self.y = None
        self.y_prev = None

    def reset(self):
        self.y = None
        self.y_prev = None

    def step(self, target: float, dt: float):
        target = float(target)
        if self.y is None:
            self.y = target
            self.y_prev = target
            return self.y, 0.0
        if dt <= 1e-6 or self.tau <= 1e-6:
            self.y_prev = self.y
            self.y = target
            return self.y, 0.0
        alpha = dt / (self.tau + dt)
        self.y_prev = self.y
        self.y = self.y + alpha * (target - self.y)
        dy = (self.y - self.y_prev) / dt
        return self.y, dy


class ReferenceShaper1_Angle:
    """First-order low pass on a yaw angle (handles wrap)."""

    def __init__(self, tau: float):
        self.tau = float(tau)
        self.y = None

    def reset(self):
        self.y = None

    def step(self, target: float, dt: float):
        if self.y is None:
            self.y = float(target)
            return self.y, 0.0
        if dt <= 1e-6 or self.tau <= 1e-6:
            prev = self.y
            self.y = float(target)
            return self.y, wrap_to_pi(self.y - prev) / max(dt, 1e-6)
        delta = wrap_to_pi(float(target) - self.y)
        alpha = dt / (self.tau + dt)
        dy = alpha * delta / dt
        self.y = wrap_to_pi(self.y + alpha * delta)
        return self.y, dy


class ModelBasedController:
    """Outer-loop position controller sized against the identified inner loop."""

    def __init__(self):
        # ---- Outer-loop crossover frequencies (rad/s) ----
        # xy: keep well below wn_xy = 1.17 and below the 1/L = 8.7 rad/s
        #     delay wall; picked 0.40 rad/s (~1/4 of inner BW).
        # z : below BW_z = 6 rad/s; picked 1.5 rad/s.
        # yaw: well below the lightly-damped yaw_rate resonance at 40 rad/s;
        #     picked 5 rad/s.
        self.wc_xy = 0.40
        self.wc_z = 1.50
        self.wc_yaw = 5.00

        # ---- PID gains, derived by loop-shaping ----
        # (see module docstring for the calculation.)
        # xy: Kp chosen so |C P|=1 at wc_xy with ~20 deg D-lead.
        #     Kd/Kp = tan(20 deg)/wc = 0.91 s;  Ki/Kp = wc/10 = 0.04.
        self.kp_pos = np.array([0.36, 0.36, 1.52])
        self.kd_vel = np.array([0.33, 0.33, 0.37])
        self.ki_pos = np.array([0.02, 0.02, 0.15])

        # Integrator leak time-constant > outer-loop time scale so it only
        # prevents long-term wind-up, not the low-freq tracking action.
        self.int_pos_leak = np.array([0.50, 0.50, 1.00])  # 1/s
        self.ki_pos_sat = np.array([0.80, 0.80, 0.80])

        # ---- Yaw outer PID ----
        # Inner yaw_rate has wn=40, zeta=0.18 -> resonance peak ~ 8.9 dB.
        # Keep wc_yaw = 5 rad/s so the resonance is attenuated by >15 dB
        # inside the closed loop. Kp = wc_yaw/|G_yaw_rate(wc_yaw)| ~ 5.
        # Then scale for robustness: Kp = 2.5, with a small D term.
        self.kp_yaw = 2.50
        self.ki_yaw = 0.30
        self.kd_yaw = 0.15
        self.ki_yaw_sat = 0.20
        self.int_yaw_leak = 2.0

        # ---- Reference shapers ----
        # xy: second-order, wn_ref = 0.8 rad/s (above wc, below inner wn),
        # zeta_ref = 0.9 (near-critical). Produces smooth p_ref and v_ref.
        self.ref_xy = ReferenceShaper2(wn=0.80, zeta=0.90)
        self.ref_z = ReferenceShaper1(tau=1.0 / 3.0)
        self.ref_yaw = ReferenceShaper1_Angle(tau=0.20)

        # ---- Feed-forward gains (inverse of inner DC gain) ----
        self.kff_xy = 1.0 / INNER_XY["K"]
        self.kff_z = 1.0 / INNER_Z["K"]
        self.kff_yaw = 1.0 / INNER_YAW_RATE["K"]

        # ---- Velocity command low-pass (command shaper) ----
        # tau_cmd_xy chosen so the corner (1 rad/s) is below xy inner wn.
        # z: none needed (inner is fast first-order).
        self.tau_cmd_xy = 1.00
        self.tau_cmd_z = 0.0
        self._vcmd_lp_body = np.zeros(3)

        # ---- IMC-style DOB ----
        # Q-filter bandwidth per axis, in rad/s. Stay below wn_xy for xy.
        self.dob_wQ = np.array([1.00, 1.00, 3.00])
        self.k_comp = np.array([self.kff_xy, self.kff_xy, self.kff_z])
        self.dob_sat = np.array([0.55, 0.55, 0.40])
        self.d_hat = np.zeros(3)

        # ---- Output limits ----
        self.max_vel = np.array([1.08, 1.08, 1.08])
        self.max_yaw_rate = 1.74533
        self.yaw_align_tol = 0.167  # ~9.6 deg; enables xy once aligned

        # ---- Internal states ----
        self.prev_yaw_err = None
        self.int_pos_body = np.zeros(3)
        self.int_yaw = 0.0
        self.vel_est_world = np.zeros(3)
        self.prev_pos_world = None
        self.vel_est_lpf_alpha = 0.56
        self.last_debug = {}

    # ------------------------------------------------------------------
    def reset(self):
        self.prev_yaw_err = None
        self.int_pos_body = np.zeros(3)
        self.int_yaw = 0.0
        self.vel_est_world = np.zeros(3)
        self.prev_pos_world = None
        self._vcmd_lp_body = np.zeros(3)
        self.d_hat = np.zeros(3)
        self.ref_xy.reset()
        self.ref_z.reset()
        self.ref_yaw.reset()
        self.last_debug = {}

    # ------------------------------------------------------------------
    def _estimate_velocity_world(self, pos_world: np.ndarray, dt: float):
        """Complementary low-pass of the differentiated Vicon position."""
        if dt > 1e-6 and self.prev_pos_world is not None:
            raw_vel = (pos_world - self.prev_pos_world) / dt
            raw_vel = np.clip(raw_vel, -3.0, 3.0)
            a = self.vel_est_lpf_alpha
            self.vel_est_world = a * raw_vel + (1.0 - a) * self.vel_est_world
        self.prev_pos_world = pos_world.copy()

    # ------------------------------------------------------------------
    def _update_dob(self, vel_cmd_body: np.ndarray, vel_est_body: np.ndarray,
                    dt: float, enabled: bool):
        """Simple IMC approximation of a DOB:
            d_hat_dot = wQ * (vel_cmd - vel_est) - wQ * d_hat
        equivalent to passing the tracking error through a 1st-order LPF
        with cutoff wQ, which matches a Q*(1-G) filter for wc << wn.
        """
        if dt <= 1e-6:
            return
        if not enabled:
            # bleed any accumulated d_hat back to zero smoothly
            self.d_hat *= np.exp(-self.dob_wQ * dt)
            return
        err = vel_cmd_body - vel_est_body
        d_dot = self.dob_wQ * (err - self.d_hat)
        self.d_hat += d_dot * dt
        self.d_hat = np.clip(self.d_hat, -self.dob_sat, self.dob_sat)

    # ------------------------------------------------------------------
    def compute(self, state, target_pos, dt: float, wind_enabled: bool = False):
        state_arr = np.asarray(state, dtype=float).reshape(-1)
        if state_arr.size < 6:
            return (0.0, 0.0, 0.0, 0.0)

        # Safe dt: ignore absurd values from the host.
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 1e-4:
            dt = 1e-3
        elif dt > 0.5:
            dt = 0.5

        x, y, z, roll, pitch, yaw = state_arr[0:6]
        x_d, y_d, z_d, yaw_d = target_pos
        pos = np.array([x, y, z], dtype=float)
        pos_tgt = np.array([x_d, y_d, z_d], dtype=float)

        # ----- Estimate velocity from positions (world frame) -----
        self._estimate_velocity_world(pos, dt)
        vel_est_world = self.vel_est_world.copy()

        # ----- Reference shaping (produces smooth p_ref, v_ref in world) -----
        pref_xy, vref_xy = self.ref_xy.step(pos_tgt[0:2], dt)
        pref_z,  vref_z  = self.ref_z.step(pos_tgt[2], dt)
        yaw_ref, yaw_rate_ref = self.ref_yaw.step(yaw_d, dt)

        p_ref_world = np.array([pref_xy[0], pref_xy[1], pref_z])
        v_ref_world = np.array([vref_xy[0], vref_xy[1], vref_z])

        # ----- Map everything into the yaw-aligned body frame -----
        R = rot_world_to_yaw_body(yaw)
        pos_err_world = p_ref_world - pos
        pos_err_body  = R @ pos_err_world
        vel_est_body  = R @ vel_est_world
        v_ref_body    = R @ v_ref_world

        # ----- Yaw alignment gate for xy -----
        yaw_err = wrap_to_pi(yaw_ref - yaw)
        yaw_aligned = abs(yaw_err) < self.yaw_align_tol

        # ----- Leaky + saturated position integrator (body frame) -----
        self.int_pos_body[2] += pos_err_body[2] * dt
        if yaw_aligned:
            self.int_pos_body[0] += pos_err_body[0] * dt
            self.int_pos_body[1] += pos_err_body[1] * dt
        self.int_pos_body *= np.exp(-self.int_pos_leak * dt)
        self.int_pos_body = np.clip(self.int_pos_body,
                                    -self.ki_pos_sat, self.ki_pos_sat)

        # ----- Feedback PID + feed-forward -----
        p_term = self.kp_pos * pos_err_body
        i_term = self.ki_pos * self.int_pos_body
        d_term = self.kd_vel * (v_ref_body - vel_est_body)  # tracks v_ref
        ff_term = np.array([
            self.kff_xy * v_ref_body[0],
            self.kff_xy * v_ref_body[1],
            self.kff_z  * v_ref_body[2],
        ])
        vel_cmd_nominal = p_term + i_term + d_term + ff_term

        # ----- DOB update and compensation -----
        self._update_dob(vel_cmd_nominal, vel_est_body, dt, enabled=wind_enabled)
        dob_comp = np.zeros(3)
        if wind_enabled:
            dob_comp = self.k_comp * self.d_hat
            dob_comp[2] = 0.0  # no wind in z in this sim
        vel_cmd = vel_cmd_nominal + dob_comp

        # ----- Command low-pass (xy only): damp out any remaining
        # energy above ~1 rad/s so the inner under-damped mode at
        # wn = 1.17 is never kicked. -----
        if self.tau_cmd_xy > 1e-6:
            a = dt / (self.tau_cmd_xy + dt)
            self._vcmd_lp_body[0] += a * (vel_cmd[0] - self._vcmd_lp_body[0])
            self._vcmd_lp_body[1] += a * (vel_cmd[1] - self._vcmd_lp_body[1])
            vel_cmd[0] = self._vcmd_lp_body[0]
            vel_cmd[1] = self._vcmd_lp_body[1]
        if self.tau_cmd_z > 1e-6:
            a = dt / (self.tau_cmd_z + dt)
            self._vcmd_lp_body[2] += a * (vel_cmd[2] - self._vcmd_lp_body[2])
            vel_cmd[2] = self._vcmd_lp_body[2]
        else:
            self._vcmd_lp_body[2] = vel_cmd[2]

        # ----- Saturations -----
        vel_cmd = np.clip(vel_cmd, -self.max_vel, self.max_vel)
        if not yaw_aligned:
            vel_cmd[0] = 0.0
            vel_cmd[1] = 0.0

        # ----- Yaw-rate outer PID (tracks yaw_ref) + feed-forward -----
        self.int_yaw += yaw_err * dt
        self.int_yaw *= float(np.exp(-self.int_yaw_leak * dt))
        self.int_yaw = float(np.clip(self.int_yaw,
                                     -self.ki_yaw_sat, self.ki_yaw_sat))
        if self.prev_yaw_err is None:
            yaw_err_deriv = 0.0
        else:
            delta = wrap_to_pi(yaw_err - self.prev_yaw_err)
            yaw_err_deriv = float(np.clip(delta / dt, -10.0, 10.0))
        self.prev_yaw_err = yaw_err

        yaw_rate_unsat = (
            self.kp_yaw * yaw_err
            + self.ki_yaw * self.int_yaw
            + self.kd_yaw * yaw_err_deriv
            + self.kff_yaw * yaw_rate_ref
        )
        yaw_rate_cmd = float(np.clip(yaw_rate_unsat,
                                     -self.max_yaw_rate, self.max_yaw_rate))
        if abs(yaw_rate_cmd) >= self.max_yaw_rate - 1e-6:
            self.int_yaw -= yaw_err * dt  # anti-windup

        # ----- Debug bookkeeping (compatible with controller_practical.py) -----
        self.last_debug = {
            "pos": pos.copy(),
            "target": pos_tgt.copy(),
            "p_ref_world": p_ref_world.copy(),
            "v_ref_world": v_ref_world.copy(),
            "pos_err_world": pos_err_world.copy(),
            "pos_err_body": pos_err_body.copy(),
            "vel_est_world": vel_est_world.copy(),
            "vel_est_body": vel_est_body.copy(),
            "vel_cmd_nominal": vel_cmd_nominal.copy(),
            "pid_p_body": p_term.copy(),
            "pid_i_body": i_term.copy(),
            "pid_d_body": d_term.copy(),
            "ff_body": ff_term.copy(),
            "dob_comp_body": dob_comp.copy(),
            "d_hat": self.d_hat.copy(),
            "vel_cmd_final": vel_cmd.copy(),
            "yaw_err": yaw_err,
            "yaw_ref": yaw_ref,
            "yaw_rate_ref": yaw_rate_ref,
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


# Module-level singleton + lab-compatible entry point.
_mb_controller = ModelBasedController()


def get_integral_telemetry():
    d = _mb_controller.last_debug
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


def controller(state, target_pos, dt, wind_enabled: bool = False):
    """Lab-harness-compatible entry point.

    Unlike controller.py, this uses the caller's real dt directly so the
    loop-shaping gains and the reference shapers run at the advertised
    frequencies (50 Hz from run.py).
    """
    return _mb_controller.compute(state, target_pos, dt, wind_enabled)


def reset_controller():
    _mb_controller.reset()


def get_debug() -> dict:
    return dict(_mb_controller.last_debug)
