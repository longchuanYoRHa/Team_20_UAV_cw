"""
Inner-loop system identification for the Tello+PyBullet simulation.

目的
----
在不使用外环 controller.py 的前提下，直接向 TelloController（内环速度-姿态-
角速度级联 PID）注入速度 / 偏航角速度指令，记录本体系下的响应；然后拟合
一阶、二阶、二阶+零点三种模型，给出最适合后续抗风外环设计用的传递函数。

物理一致性
---------
PyBullet 的 `p.stepSimulation()` 默认按 `fixedTimeStep = 1/240 s` 步进。run.py
没有调用 `p.setTimeStep()`，所以真实物理步长其实是 **1/240 s**（而不是 run.py
里的 `timestep = 1/1000` —— 那个变量只被 `motor_model` / `spin_motors` / sleep
用到）。本脚本显式 `p.setTimeStep(1/240)` 并且整个采样链路（电机一阶惯性、
spin motors）都用 dt=1/240，避免已有 open_loop_inner_identification.py 里因
dt 不匹配造成的一阶拟合失败。

识别内容
---------
对 vx, vy, vz, yaw_rate 四个通道分别做阶跃测试（正负 + 多幅值），并可选做一
次 chirp 用于频响验证。每个通道同时拟合：

    1)  一阶 + 延迟         y = K/(τs+1) * exp(-Ls) * u
    2)  二阶欠阻尼 + 延迟   y = K ωn² /(s² + 2ζωn s + ωn²) * exp(-Ls) * u
    3)  二阶 + 零点 + 延迟  y = K(τz s+1) ωn² /(s² + 2ζωn s + ωn²) * exp(-Ls) * u

用 scipy.optimize.least_squares 在离散时域上做最小二乘拟合（时延用插值处理，
保证对非整步延迟也可导）。

输出
----
    inner_loop_id_output/
        <axis>_step_<amp>.csv          每条 trace 的原始数据
        <axis>_fit.png                 3 种模型对 steps 的叠加拟合图
        <axis>_bode.png                3 种模型的 Bode（含一阶对比）
        summary.csv                    所有通道 / 幅值下识别出的参数
        inner_loop_params.json         最适合外环设计的那组参数（机器可读）

运行
----
    cd assignment_3_2026/assignment_3_2026
    python inner_loop_identification.py                     # 默认配置
    python inner_loop_identification.py --quick             # 只跑一种幅值
    python inner_loop_identification.py --with-chirp        # 附加一次线性扫频验证
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Callable

import numpy as np
import pybullet as p
import pybullet_data
from scipy.optimize import least_squares

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.tello_controller import TelloController


# ---------------------------------------------------------------------------
# Physics / controller constants (keep consistent with run.py)
# ---------------------------------------------------------------------------
# PyBullet's `p.stepSimulation()` default fixedTimeStep is 1/240 s. run.py does
# NOT call p.setTimeStep(), so the real physics step is 1/240 s even though the
# variable `timestep = 1/1000` is used inside run.py for motor_model / spin_motors
# / sleep. To identify the inner loop *as experienced by the outer loop in
# run.py*, we mimic the same discrepancy by default (--match-run-py).
DT_PHYS = 1.0 / 240.0            # pybullet physics tick
DT_CTRL_DEFAULT = 1.0 / 1000.0   # dt used by TelloController/motor/spin in run.py
T_SETTLE_BEFORE_STEP = 0.5       # seconds of hover before step is applied
T_AFTER_STEP = 6.0               # seconds of data captured after step
DURATION = T_SETTLE_BEFORE_STEP + T_AFTER_STEP

M = 0.088
L_ARM = 0.06
IR = 4.95e-5
KF = 0.566e-5
KM = 0.762e-7
K_TRANS = np.array([3.365e-2, 3.365e-2, 3.365e-2])
K_ROT = np.array([4.609e-3, 4.609e-3, 4.609e-3])
TM = 0.0163


# ---------------------------------------------------------------------------
# Plant wrapper: identical physics to run.py but headless & bypasses controller.py
# ---------------------------------------------------------------------------
class InnerLoopPlant:
    """Reproduce run.py's closed-loop inner dynamics.

    Parameters
    ----------
    ctrl_dt : float
        dt passed to TelloController.compute_control / motor_model (same semantics
        as `timestep` in run.py). Leave at 1/1000 to match run.py's deployed
        behavior; set to DT_PHYS (1/240) to recover the "clean" physics.
    """

    def __init__(self, gui: bool = False, ctrl_dt: float = DT_CTRL_DEFAULT):
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(DT_PHYS)
        p.setPhysicsEngineParameter(fixedTimeStep=DT_PHYS, numSubSteps=1)

        self.plane_id = p.loadURDF("plane.urdf")
        self.default_start_pos = [0.0, 0.0, 1.0]
        self.start_pos = list(self.default_start_pos)
        self.start_quat = p.getQuaternionFromEuler([0.0, 0.0, 0.0])
        self.drone_id = p.loadURDF("resources/tello.urdf", self.start_pos, self.start_quat)

        self.tello = TelloController(9.81, M, L_ARM, 0.35, KF, KM)
        self.prev_rpm = np.zeros(4)
        self.ctrl_dt = float(ctrl_dt)

    def close(self):
        if p.isConnected(self.client):
            p.disconnect(self.client)

    def set_start_altitude(self, z: float):
        self.start_pos = [self.default_start_pos[0], self.default_start_pos[1], float(z)]

    def reset(self):
        p.resetBasePositionAndOrientation(self.drone_id, self.start_pos, self.start_quat)
        p.resetBaseVelocity(self.drone_id, [0, 0, 0], [0, 0, 0])
        self.tello.reset()
        self.prev_rpm = np.zeros(4)

    def _motor_model(self, desired_rpm, current_rpm, dt):
        return current_rpm + (desired_rpm - current_rpm) / TM * dt

    def _compute_dynamics(self, rpm, lin_vel_world, quat):
        R = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        omega = rpm * (2.0 * np.pi / 60.0)
        om2 = omega ** 2
        motor_forces = om2 * KF
        thrust = np.array([0.0, 0.0, motor_forces.sum()])
        vel_body = R.T @ lin_vel_world
        drag_body = -K_TRANS * vel_body
        force = drag_body + thrust
        ztq = om2 * KM
        z_torque = -ztq[0] - ztq[1] + ztq[2] + ztq[3]
        x_torque = (-motor_forces[0] + motor_forces[1] + motor_forces[2] - motor_forces[3]) * L_ARM
        y_torque = (-motor_forces[0] + motor_forces[1] - motor_forces[2] + motor_forces[3]) * L_ARM
        return force, np.array([x_torque, y_torque, z_torque])

    def step(self, vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd):
        pos, quat = p.getBasePositionAndOrientation(self.drone_id)
        lin_w, ang_w = p.getBaseVelocity(self.drone_id)
        roll, pitch, yaw = p.getEulerFromQuaternion(quat)
        yaw_quat = p.getQuaternionFromEuler([0, 0, yaw])
        _, inv_q = p.invertTransform([0, 0, 0], quat)
        _, inv_qyaw = p.invertTransform([0, 0, 0], yaw_quat)
        lin_body_yaw = np.array(p.rotateVector(inv_qyaw, lin_w), dtype=float)
        ang_body = np.array(p.rotateVector(inv_q, ang_w), dtype=float)

        desired_vel = np.array([vx_cmd, vy_cmd, vz_cmd], dtype=float)
        # NOTE: run.py calls these with `timestep = 1/1000` while stepSimulation
        # advances by 1/240 → we replicate the same dt mismatch when
        # ctrl_dt != DT_PHYS so the identified inner loop matches deployment.
        rpm = self.tello.compute_control(desired_vel, lin_body_yaw, quat, ang_body,
                                         float(yaw_rate_cmd), self.ctrl_dt)
        rpm = self._motor_model(rpm, self.prev_rpm, self.ctrl_dt)
        self.prev_rpm = rpm

        force, torque = self._compute_dynamics(rpm, lin_w, quat)
        p.applyExternalForce(self.drone_id, -1, force, [0, 0, 0], p.LINK_FRAME)
        p.applyExternalTorque(self.drone_id, -1, torque, p.LINK_FRAME)
        p.stepSimulation()

        pos2, quat2 = p.getBasePositionAndOrientation(self.drone_id)
        lin_w2, ang_w2 = p.getBaseVelocity(self.drone_id)
        yaw2 = p.getEulerFromQuaternion(quat2)[2]
        yaw_quat2 = p.getQuaternionFromEuler([0, 0, yaw2])
        _, inv_q2 = p.invertTransform([0, 0, 0], quat2)
        _, inv_qyaw2 = p.invertTransform([0, 0, 0], yaw_quat2)
        lin_body_yaw2 = np.array(p.rotateVector(inv_qyaw2, lin_w2), dtype=float)
        ang_body2 = np.array(p.rotateVector(inv_q2, ang_w2), dtype=float)
        return {
            "lin_body_yaw": lin_body_yaw2,
            "ang_body": ang_body2,
            "pos": np.array(pos2, dtype=float),
            "yaw": float(yaw2),
        }


# ---------------------------------------------------------------------------
# Experiment runner: step response
# ---------------------------------------------------------------------------
def run_step(plant: InnerLoopPlant, axis: str, amplitude: float):
    # Lift start altitude for vz tests so that negative commands don't slam the ground.
    if axis == "vz" and amplitude < 0:
        plant.set_start_altitude(max(3.0, 1.0 + abs(amplitude) * T_AFTER_STEP + 0.5))
    elif axis == "vz":
        plant.set_start_altitude(1.0)
    else:
        plant.set_start_altitude(1.0)
    plant.reset()
    n = int(round(DURATION / DT_PHYS))
    n_step = int(round(T_SETTLE_BEFORE_STEP / DT_PHYS))
    t = np.arange(n) * DT_PHYS

    u = np.zeros(n)
    u[n_step:] = amplitude
    y = np.zeros(n)
    cross1 = np.zeros(n)
    cross2 = np.zeros(n)

    axis_map = {"vx": 0, "vy": 1, "vz": 2}
    for k in range(n):
        if axis == "yaw_rate":
            out = plant.step(0.0, 0.0, 0.0, u[k])
            y[k] = out["ang_body"][2]
            cross1[k] = out["lin_body_yaw"][0]
            cross2[k] = out["lin_body_yaw"][1]
        else:
            cmd = [0.0, 0.0, 0.0]
            cmd[axis_map[axis]] = u[k]
            out = plant.step(cmd[0], cmd[1], cmd[2], 0.0)
            y[k] = out["lin_body_yaw"][axis_map[axis]]
            others = [i for i in range(3) if i != axis_map[axis]]
            cross1[k] = out["lin_body_yaw"][others[0]]
            cross2[k] = out["lin_body_yaw"][others[1]]
    return {"t": t, "u": u, "y": y, "cross1": cross1, "cross2": cross2,
            "axis": axis, "amp": amplitude, "t_step": T_SETTLE_BEFORE_STEP}


def run_chirp(plant: InnerLoopPlant, axis: str, amplitude: float,
              f0: float = 0.1, f1: float = 6.0, duration: float = 20.0):
    plant.reset()
    n = int(round(duration / DT_PHYS))
    t = np.arange(n) * DT_PHYS
    # linear (time) frequency sweep
    k_chirp = (f1 - f0) / duration
    phase = 2.0 * np.pi * (f0 * t + 0.5 * k_chirp * t * t)
    u = amplitude * np.sin(phase)
    y = np.zeros(n)
    axis_map = {"vx": 0, "vy": 1, "vz": 2}
    for k in range(n):
        if axis == "yaw_rate":
            out = plant.step(0.0, 0.0, 0.0, u[k])
            y[k] = out["ang_body"][2]
        else:
            cmd = [0.0, 0.0, 0.0]
            cmd[axis_map[axis]] = u[k]
            out = plant.step(cmd[0], cmd[1], cmd[2], 0.0)
            y[k] = out["lin_body_yaw"][axis_map[axis]]
    return {"t": t, "u": u, "y": y, "axis": axis, "amp": amplitude, "f0": f0, "f1": f1}


# ---------------------------------------------------------------------------
# Discrete-time simulators of the candidate models
# ---------------------------------------------------------------------------
def _apply_delay(u: np.ndarray, L: float, dt: float) -> np.ndarray:
    """Fractional-sample delay by linear interpolation. L >= 0."""
    if L <= 0:
        return u.copy()
    shift = L / dt
    int_shift = int(np.floor(shift))
    frac = shift - int_shift
    n = u.size
    out = np.zeros(n)
    for i in range(n):
        i1 = i - int_shift
        i0 = i1 - 1
        v1 = u[i1] if 0 <= i1 < n else 0.0
        v0 = u[i0] if 0 <= i0 < n else 0.0
        out[i] = (1.0 - frac) * v1 + frac * v0
    return out


def sim_first_order(u, dt, K, tau, L):
    """y_dot = (K*u_d - y) / tau"""
    u_d = _apply_delay(u, L, dt)
    n = u.size
    y = np.zeros(n)
    if tau <= 1e-6:
        return K * u_d
    a = dt / tau
    for i in range(1, n):
        y[i] = y[i - 1] + a * (K * u_d[i - 1] - y[i - 1])
    return y


def _substeps_for(wn: float, dt: float) -> int:
    """Stable Euler sub-stepping: ensure wn*dt_eff < ~0.2."""
    if wn <= 1e-6:
        return 1
    return max(1, int(np.ceil(wn * dt / 0.2)))


def sim_second_order(u, dt, K, wn, zeta, L):
    """Discrete integration of y'' + 2ζωn y' + ωn² y = K ωn² u_d with sub-stepping."""
    u_d = _apply_delay(u, L, dt)
    n = u.size
    y = np.zeros(n)
    dy = 0.0
    s = _substeps_for(wn, dt)
    h = dt / s
    for i in range(1, n):
        uh = u_d[i - 1]
        y_prev = y[i - 1]
        for _ in range(s):
            ddy = K * wn * wn * uh - 2.0 * zeta * wn * dy - wn * wn * y_prev
            dy += ddy * h
            y_prev = y_prev + dy * h
            # Hard safety clamp — avoids NaN/inf blowing up least_squares
            if not np.isfinite(y_prev) or abs(y_prev) > 1e6:
                return np.full(n, np.nan)
        y[i] = y_prev
    return y


def sim_second_order_zero(u, dt, K, wn, zeta, tau_z, L):
    """y = K (tau_z s + 1) / (s^2/ωn² + 2ζ/ωn s + 1) u_d, with sub-stepping."""
    u_d = _apply_delay(u, L, dt)
    n = u.size
    w = 0.0
    dw = 0.0
    y = np.zeros(n)
    s = _substeps_for(wn, dt)
    h = dt / s
    for i in range(1, n):
        uh = u_d[i - 1]
        for _ in range(s):
            ddw = wn * wn * uh - 2.0 * zeta * wn * dw - wn * wn * w
            dw = dw + ddw * h
            w = w + dw * h
            if not np.isfinite(w) or abs(w) > 1e6:
                return np.full(n, np.nan)
        y[i] = K * (w + tau_z * dw)
    return y


# ---------------------------------------------------------------------------
# Fitting helpers
# ---------------------------------------------------------------------------
@dataclass
class FitResult:
    model: str
    params: dict
    rmse: float
    y_fit: np.ndarray

    def to_serialisable(self) -> dict:
        return {"model": self.model, "params": self.params, "rmse": float(self.rmse)}


def _mask_after_step(t: np.ndarray, t_step: float) -> np.ndarray:
    return t >= (t_step - 0.05)  # include tiny margin to catch the step edge


def fit_first_order(trace) -> FitResult:
    t, u, y = trace["t"], trace["u"], trace["y"]
    msk = _mask_after_step(t, trace["t_step"])
    dt = float(t[1] - t[0])
    amp = trace["amp"]

    y_post = y[msk]
    yss = float(np.mean(y_post[-max(50, len(y_post)//10):]))
    K0 = (yss - y_post[0]) / amp if abs(amp) > 1e-9 else 0.0
    # first guess for τ via 63% rule
    target = y_post[0] + 0.632 * (yss - y_post[0])
    sign = np.sign(yss - y_post[0]) if yss != y_post[0] else 1.0
    idx = np.where(sign * (y_post - target) >= 0)[0]
    tau0 = max(float(t[msk][idx[0]] - trace["t_step"]), 0.02) if len(idx) else 0.3

    def residual(theta):
        K, tau, L = theta
        ys = sim_first_order(u, dt, K, max(tau, 1e-4), max(L, 0.0))
        r = (ys - y)[msk]
        if not np.all(np.isfinite(r)):
            return np.full(msk.sum(), 1e3)
        return r

    x0 = np.array([K0, tau0, 0.02])
    lb = np.array([-5.0, 1e-3, 0.0])
    ub = np.array([5.0, 20.0, 0.5])
    x0 = np.clip(x0, lb + 1e-6, ub - 1e-6)
    res = least_squares(residual, x0, bounds=(lb, ub), max_nfev=2000)
    K, tau, L = res.x
    y_fit = sim_first_order(u, dt, K, tau, L)
    rmse = float(np.sqrt(np.mean((y_fit[msk] - y[msk]) ** 2)))
    return FitResult("first_order", {"K": float(K), "tau": float(tau), "L": float(L)}, rmse, y_fit)


def fit_second_order(trace) -> FitResult:
    t, u, y = trace["t"], trace["u"], trace["y"]
    msk = _mask_after_step(t, trace["t_step"])
    dt = float(t[1] - t[0])
    amp = trace["amp"]

    y_post = y[msk]
    t_post = t[msk] - trace["t_step"]
    yss = float(np.mean(y_post[-max(50, len(y_post)//10):]))
    K0 = (yss - y_post[0]) / amp if abs(amp) > 1e-9 else 1.0
    # peak-based ζ, ωn seed
    if abs(yss - y_post[0]) > 1e-6:
        y_peak_idx = int(np.argmax(np.sign(yss - y_post[0]) * (y_post - y_post[0])))
        y_peak = y_post[y_peak_idx]
        overshoot = max((np.sign(yss - y_post[0]) * (y_peak - yss)) / (yss - y_post[0]), 1e-3)
        if overshoot >= 1.0:
            zeta0 = 0.2
        else:
            logos = np.log(overshoot)
            zeta0 = -logos / np.sqrt(np.pi**2 + logos**2)
        tp = max(t_post[y_peak_idx], 1e-3)
        wn0 = np.pi / max(tp * np.sqrt(max(1 - zeta0**2, 1e-3)), 1e-3)
    else:
        zeta0, wn0 = 0.7, 5.0

    def residual(theta):
        K, wn, zeta, L = theta
        ys = sim_second_order(u, dt, K, max(wn, 1e-2), max(zeta, 1e-3), max(L, 0.0))
        r = (ys - y)[msk]
        if not np.all(np.isfinite(r)):
            return np.full(msk.sum(), 1e3)
        return r

    x0 = np.array([K0, wn0, zeta0, 0.02])
    lb = np.array([-5.0, 0.1, 1e-2, 0.0])
    ub = np.array([5.0, 200.0, 5.0, 0.5])
    x0 = np.clip(x0, lb + 1e-6, ub - 1e-6)
    res = least_squares(residual, x0, bounds=(lb, ub), max_nfev=3000)
    K, wn, zeta, L = res.x
    y_fit = sim_second_order(u, dt, K, wn, zeta, L)
    rmse = float(np.sqrt(np.mean((y_fit[msk] - y[msk]) ** 2)))
    return FitResult(
        "second_order",
        {"K": float(K), "wn": float(wn), "zeta": float(zeta), "L": float(L)},
        rmse, y_fit,
    )


def fit_second_order_zero(trace, so_seed: FitResult) -> FitResult:
    t, u, y = trace["t"], trace["u"], trace["y"]
    msk = _mask_after_step(t, trace["t_step"])
    dt = float(t[1] - t[0])

    K0 = so_seed.params["K"]
    wn0 = so_seed.params["wn"]
    zeta0 = so_seed.params["zeta"]
    L0 = so_seed.params["L"]

    def residual(theta):
        K, wn, zeta, tau_z, L = theta
        ys = sim_second_order_zero(u, dt, K, max(wn, 1e-2), max(zeta, 1e-3), tau_z, max(L, 0.0))
        r = (ys - y)[msk]
        if not np.all(np.isfinite(r)):
            return np.full(msk.sum(), 1e3)
        return r

    x0 = np.array([K0, wn0, zeta0, 0.0, L0])
    lb = np.array([-5.0, 0.1, 1e-2, -2.0, 0.0])
    ub = np.array([5.0, 200.0, 5.0, 2.0, 0.5])
    x0 = np.clip(x0, lb + 1e-6, ub - 1e-6)
    res = least_squares(residual, x0, bounds=(lb, ub), max_nfev=4000)
    K, wn, zeta, tau_z, L = res.x
    y_fit = sim_second_order_zero(u, dt, K, wn, zeta, tau_z, L)
    rmse = float(np.sqrt(np.mean((y_fit[msk] - y[msk]) ** 2)))
    return FitResult(
        "second_order_zero",
        {"K": float(K), "wn": float(wn), "zeta": float(zeta),
         "tau_z": float(tau_z), "L": float(L)},
        rmse, y_fit,
    )


# ---------------------------------------------------------------------------
# Analysis utilities
# ---------------------------------------------------------------------------
def second_order_characteristics(wn: float, zeta: float) -> dict:
    """Derived closed-loop characteristics used for outer-loop design."""
    out = {}
    if zeta < 1.0:
        wd = wn * np.sqrt(max(1.0 - zeta * zeta, 1e-12))
        out["wd"] = float(wd)
        out["peak_time_s"] = float(np.pi / wd) if wd > 0 else float("inf")
        out["overshoot_pct"] = float(
            100.0 * np.exp(-zeta * np.pi / np.sqrt(max(1 - zeta * zeta, 1e-12)))
        )
    else:
        out["wd"] = 0.0
        out["peak_time_s"] = float("inf")
        out["overshoot_pct"] = 0.0
    out["settling_time_2pct_s"] = float(4.0 / (zeta * wn)) if zeta * wn > 1e-6 else float("inf")
    # -3 dB bandwidth of 2nd-order LP: ωn·sqrt(1 - 2ζ² + sqrt(2 - 4ζ² + 4ζ⁴))
    r = 1.0 - 2.0 * zeta * zeta
    bw = wn * np.sqrt(r + np.sqrt(r * r + 1.0))
    out["bandwidth_3dB_rad_s"] = float(bw)
    out["bandwidth_3dB_hz"] = float(bw / (2.0 * np.pi))
    return out


def bode_first_order(K, tau, L, w):
    s = 1j * w
    G = K * np.exp(-L * s) / (tau * s + 1.0)
    return G


def bode_second_order(K, wn, zeta, L, w):
    s = 1j * w
    G = K * wn ** 2 * np.exp(-L * s) / (s ** 2 + 2 * zeta * wn * s + wn ** 2)
    return G


def bode_second_order_zero(K, wn, zeta, tau_z, L, w):
    s = 1j * w
    G = K * (tau_z * s + 1.0) * wn ** 2 * np.exp(-L * s) / (s ** 2 + 2 * zeta * wn * s + wn ** 2)
    return G


def empirical_frf_from_chirp(trace):
    u, y, t = trace["u"], trace["y"], trace["t"]
    dt = float(t[1] - t[0])
    n = u.size
    U = np.fft.rfft(u)
    Y = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(n, d=dt) * 2.0 * np.pi  # rad/s
    mag = np.abs(Y) / np.maximum(np.abs(U), 1e-9)
    keep = (freqs >= 2.0 * np.pi * trace["f0"]) & (freqs <= 2.0 * np.pi * trace["f1"])
    # phase from FRF ratio
    phase = np.unwrap(np.angle(Y / np.maximum(np.abs(U) * np.exp(1j * np.angle(U)), 1e-12)))
    return freqs[keep], mag[keep], phase[keep]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_step_fits(out_dir: str, axis: str, traces: list, fits_per_trace: list):
    n = len(traces)
    fig, axs = plt.subplots(n, 1, figsize=(10, 3.0 * n), sharex=True, squeeze=False)
    for i, (trace, fits) in enumerate(zip(traces, fits_per_trace)):
        t, u, y = trace["t"], trace["u"], trace["y"]
        ax = axs[i, 0]
        ax.plot(t, u, "k:", label="cmd", alpha=0.7)
        ax.plot(t, y, "k-", label="measured", linewidth=1.8)
        styles = {"first_order": "--", "second_order": "-.", "second_order_zero": ":"}
        for f in fits:
            ax.plot(t, f.y_fit, styles.get(f.model, "--"),
                    label=f"{f.model} rmse={f.rmse:.4f}")
        ax.set_ylabel(f"{axis} (amp={trace['amp']:+.2f})")
        ax.grid(True)
        ax.legend(loc="lower right", fontsize=8)
        ax.axvline(trace["t_step"], color="grey", linestyle=":", linewidth=0.8)
    axs[-1, 0].set_xlabel("Time [s]")
    fig.suptitle(f"Inner-loop step fits: {axis}")
    fig.tight_layout()
    path = os.path.join(out_dir, f"{axis}_fit.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_bode(out_dir: str, axis: str, best_fit: FitResult,
              all_fits: list, chirp_frf=None):
    w = np.logspace(-1, np.log10(300), 400)
    fig, ax = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    plotted = []
    for f in all_fits:
        if f.model == "first_order":
            G = bode_first_order(f.params["K"], f.params["tau"], f.params["L"], w)
            lbl = f"1st: K={f.params['K']:.2f}, τ={f.params['tau']:.3f}s"
        elif f.model == "second_order":
            G = bode_second_order(f.params["K"], f.params["wn"], f.params["zeta"], f.params["L"], w)
            lbl = f"2nd: K={f.params['K']:.2f}, ωn={f.params['wn']:.2f}, ζ={f.params['zeta']:.3f}"
        else:
            G = bode_second_order_zero(f.params["K"], f.params["wn"], f.params["zeta"],
                                       f.params["tau_z"], f.params["L"], w)
            lbl = (f"2nd+zero: K={f.params['K']:.2f}, ωn={f.params['wn']:.2f}, "
                   f"ζ={f.params['zeta']:.3f}, τz={f.params['tau_z']:.3f}")
        is_best = f is best_fit
        ax[0].semilogx(w, 20 * np.log10(np.maximum(np.abs(G), 1e-9)),
                       linewidth=2.0 if is_best else 1.0,
                       label=lbl + (" (selected)" if is_best else ""))
        ax[1].semilogx(w, np.unwrap(np.angle(G)) * 180.0 / np.pi,
                       linewidth=2.0 if is_best else 1.0,
                       label=lbl + (" (selected)" if is_best else ""))
        plotted.append(lbl)

    if chirp_frf is not None:
        w_e, mag_e, ph_e = chirp_frf
        ax[0].semilogx(w_e, 20 * np.log10(np.maximum(mag_e, 1e-9)),
                       "r.", markersize=3, label="chirp FRF")
    ax[0].set_ylabel("Magnitude [dB]")
    ax[0].grid(True, which="both")
    ax[0].legend(fontsize=8, loc="lower left")
    ax[0].set_title(f"Bode – inner-loop {axis}")
    ax[1].set_ylabel("Phase [deg]")
    ax[1].set_xlabel("ω [rad/s]")
    ax[1].grid(True, which="both")
    fig.tight_layout()
    path = os.path.join(out_dir, f"{axis}_bode.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def select_best_fit(all_fits: list) -> FitResult:
    """Lowest RMSE wins; fall back to simpler model if RMSE tied to within 5%."""
    best = min(all_fits, key=lambda f: f.rmse)
    simpler_priority = {"first_order": 0, "second_order": 1, "second_order_zero": 2}
    for f in sorted(all_fits, key=lambda f: simpler_priority[f.model]):
        if f.rmse <= 1.05 * best.rmse:
            return f
    return best


def aggregate_over_amps(per_amp_fits: list[FitResult], model: str) -> dict:
    """Average parameters of a given model across amplitudes (for final export)."""
    chosen = [f for f in per_amp_fits if f.model == model]
    if not chosen:
        return {}
    keys = chosen[0].params.keys()
    avg = {k: float(np.mean([f.params[k] for f in chosen])) for k in keys}
    avg["rmse_mean"] = float(np.mean([f.rmse for f in chosen]))
    avg["rmse_max"] = float(np.max([f.rmse for f in chosen]))
    return avg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="inner_loop_id_output")
    ap.add_argument("--quick", action="store_true", help="only one amplitude per axis")
    ap.add_argument("--with-chirp", action="store_true",
                    help="also run one linear chirp per axis for frequency-domain validation")
    ap.add_argument("--axes", nargs="+", default=["vx", "vy", "vz", "yaw_rate"])
    ap.add_argument(
        "--ctrl-dt",
        type=float,
        default=DT_CTRL_DEFAULT,
        help=("dt passed to TelloController/motor_model. "
              "Default 1/1000 replicates run.py's deployment behavior "
              "(pybullet still steps at 1/240). Set 1/240 for 'clean' physics."),
    )
    ap.add_argument(
        "--clean-physics", action="store_true",
        help="Shortcut for --ctrl-dt=1/240 (dt consistent across the stack).",
    )
    args = ap.parse_args()
    ctrl_dt = DT_PHYS if args.clean_physics else args.ctrl_dt

    amps_default = {
        "vx":       [0.30, -0.30, 0.15],
        "vy":       [0.30, -0.30, 0.15],
        "vz":       [0.20, -0.20, 0.10],
        "yaw_rate": [0.60, -0.60, 0.30],
    }
    amps_quick = {k: v[:1] for k, v in amps_default.items()}
    amps = amps_quick if args.quick else amps_default

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Physics dt = {DT_PHYS:.6f} s, controller/motor dt = {ctrl_dt:.6f} s "
          f"({'match run.py (deployed)' if abs(ctrl_dt - DT_CTRL_DEFAULT) < 1e-9 else 'custom'})",
          flush=True)
    plant = InnerLoopPlant(gui=False, ctrl_dt=ctrl_dt)
    summary = []
    per_axis_best: dict[str, dict] = {}
    t_start = time.time()

    try:
        for axis in args.axes:
            print(f"--- axis: {axis} ---", flush=True)
            traces = []
            fits_per_trace = []
            per_amp_fits_flat: list[FitResult] = []

            for amp in amps[axis]:
                trace = run_step(plant, axis, amp)
                # save raw csv
                csv_path = os.path.join(out_dir, f"{axis}_step_{amp:+.2f}.csv")
                with open(csv_path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["t", "u", "y", "cross1", "cross2"])
                    for i in range(trace["t"].size):
                        w.writerow([trace["t"][i], trace["u"][i], trace["y"][i],
                                    trace["cross1"][i], trace["cross2"][i]])

                f1 = fit_first_order(trace)
                f2 = fit_second_order(trace)
                f3 = fit_second_order_zero(trace, f2)
                fits = [f1, f2, f3]
                best = select_best_fit(fits)
                per_amp_fits_flat.extend(fits)

                traces.append(trace)
                fits_per_trace.append(fits)
                summary.append({
                    "axis": axis, "amp": amp,
                    **{f"{f.model}_rmse": f.rmse for f in fits},
                    "best_model": best.model,
                    "best_params_json": json.dumps(best.params),
                })
                print(f"  amp={amp:+.2f}  "
                      f"1st rmse={f1.rmse:.4f}  2nd rmse={f2.rmse:.4f}  "
                      f"2nd+z rmse={f3.rmse:.4f}  best={best.model}", flush=True)

            plot_step_fits(out_dir, axis, traces, fits_per_trace)

            # Pick the globally preferred model by majority among per-amp bests.
            tally: dict[str, int] = {}
            for fits in fits_per_trace:
                best_here = select_best_fit(fits)
                tally[best_here.model] = tally.get(best_here.model, 0) + 1
            preferred = max(tally.items(), key=lambda kv: kv[1])[0]
            avg_params = aggregate_over_amps(per_amp_fits_flat, preferred)

            # The "best" single fit for plotting is the highest-amplitude positive step.
            pos_traces = [(tr, fts) for tr, fts in zip(traces, fits_per_trace) if tr["amp"] > 0]
            if pos_traces:
                ref_fits = max(pos_traces, key=lambda p: p[0]["amp"])[1]
            else:
                ref_fits = fits_per_trace[0]
            best_fit_for_plot = next((f for f in ref_fits if f.model == preferred), ref_fits[0])

            chirp_frf = None
            if args.with_chirp:
                cr = run_chirp(plant, axis, 0.15 if axis != "yaw_rate" else 0.4)
                chirp_frf = empirical_frf_from_chirp(cr)
            plot_bode(out_dir, axis, best_fit_for_plot, ref_fits, chirp_frf)

            derived = {}
            if preferred in ("second_order", "second_order_zero"):
                derived = second_order_characteristics(avg_params["wn"], avg_params["zeta"])
            elif preferred == "first_order":
                tau = avg_params["tau"]
                derived = {
                    "bandwidth_3dB_rad_s": float(1.0 / tau) if tau > 1e-9 else float("inf"),
                    "bandwidth_3dB_hz": float(1.0 / (2 * np.pi * tau)) if tau > 1e-9 else float("inf"),
                    "settling_time_2pct_s": float(4.0 * tau),
                }

            per_axis_best[axis] = {
                "preferred_model": preferred,
                "params_mean_over_amps": avg_params,
                "derived": derived,
                "amps_tested": amps[axis],
            }
            print(f"  -> preferred model = {preferred}, params = {avg_params}",
                  flush=True)

        # ---- write summary & json ----
        with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)

        payload = {
            "dt_phys_s": DT_PHYS,
            "ctrl_dt_s": ctrl_dt,
            "match_run_py_timing": abs(ctrl_dt - DT_CTRL_DEFAULT) < 1e-9,
            "duration_s": DURATION,
            "settle_before_step_s": T_SETTLE_BEFORE_STEP,
            "notes": (
                "Identified transfer functions from cmd -> measured body-frame "
                "velocity / yaw-rate. Use these when designing the outer (position) "
                "loop and disturbance observer. If `match_run_py_timing` is true, "
                "the inner loop reflects the dt mismatch in run.py (motor_model "
                "advances by 1/1000 per pybullet tick of 1/240 → motors ≈4x slower "
                "than spec in simulation time)."
            ),
            "axes": per_axis_best,
        }
        with open(os.path.join(out_dir, "inner_loop_params.json"), "w") as f:
            json.dump(payload, f, indent=2)

        print(f"\nDone in {time.time() - t_start:.1f} s.")
        print(f"Output written to: {out_dir}")
        for axis, info in per_axis_best.items():
            pp = info["params_mean_over_amps"]
            dd = info["derived"]
            if info["preferred_model"].startswith("second_order"):
                print(f"  {axis:9s}: {info['preferred_model']:20s} "
                      f"K={pp.get('K', float('nan')):.3f} "
                      f"ωn={pp.get('wn', float('nan')):.2f} rad/s  "
                      f"ζ={pp.get('zeta', float('nan')):.3f}  "
                      f"L={pp.get('L', float('nan')):.3f}s  "
                      f"BW≈{dd.get('bandwidth_3dB_hz', float('nan')):.2f} Hz")
            else:
                print(f"  {axis:9s}: {info['preferred_model']:20s} "
                      f"K={pp.get('K', float('nan')):.3f}  "
                      f"τ={pp.get('tau', float('nan')):.3f}s  "
                      f"L={pp.get('L', float('nan')):.3f}s  "
                      f"BW≈{dd.get('bandwidth_3dB_hz', float('nan')):.2f} Hz")
    finally:
        plant.close()


if __name__ == "__main__":
    main()
