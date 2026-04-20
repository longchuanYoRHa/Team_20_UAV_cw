#!/usr/bin/env python3
"""
在无 3D 窗口、无风条件下，按课程评价流程自动搜索外环 PID 参数。

评价流程（与说明一致）：
  - 每次 trial：仿真 10 s；目标位置为「当前机体位置」各轴 ±4 m 内均匀随机，偏航为随机 yaw
    （与评分说明：相对当前位置 ±4 m、随机 yaw、10 s 一致）
  - 仅搜索 6 个外环参数：kp_xy,kp_z, ki_xy,ki_z, kd_xy,kd_z（x/y 共用）；偏航 PID 保持 controller.py 不变
  - 在外环频率（50 Hz）采样位置误差范数 ||p_d - p|| 与 |wrap(yaw_d - yaw)|
  - 每个 trial 内计算上述序列的 mean 与 std
  - 对 NUM_TRIALS 次 trial 的 mean/std 再取平均，得到四个标量：
        位置误差 mean、位置误差 std、偏航误差 mean、偏航误差 std

达标阈值（可同时作为约束）：
  位置 mean < 0.01 m，位置 std < 0.01
  偏航 mean < 0.01 rad，偏航 std < 0.001

用法（在 assignment_3_2026/assignment_3_2026 目录下）：
  python auto_tune_pid.py --eval-only          # 只评测当前 controller.py 中的增益
  python auto_tune_pid.py                      # Twiddle 搜索（默认）
  python auto_tune_pid.py --max-iters 30

记录结果：
  --out-json run.json   # 最优 6 维增益、指标、以及每次评测点的迭代历史 iterations[]
  终端日志：tee，例如  python3 auto_tune_pid.py 2>&1 | tee tune_log.txt

环境变量 PID_TUNE_FAST=1：缩短为 5 次 trial、2 s，用于快速冒烟测试。

Twiddle 外环会显示迭代进度条（已完成 iter / --max-iters）。若已安装 tqdm 则使用其进度条；
否则使用简易 ASCII 条。加 --no-progress 可关闭。
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time

import numpy as np
import pybullet as p
import pybullet_data

import controller
from src.tello_controller import TelloController

try:
    from tqdm import tqdm

    _HAS_TQDM = True
except Exception:
    tqdm = None  # type: ignore[misc, assignment]
    _HAS_TQDM = False


class _NullIterBar:
    def set_postfix_str(self, s: str = "") -> None:
        pass

    def update(self, n: int = 1) -> None:
        pass

    def close(self) -> None:
        pass


class _AsciiIterBar:
    """无 tqdm 时的简易终端进度条（\\r 刷新）。"""

    def __init__(self, total: int, desc: str) -> None:
        self.total = max(int(total), 1)
        self.desc = desc
        self.n = 0
        self._postfix = ""
        self._width = 28

    def set_postfix_str(self, s: str = "") -> None:
        self._postfix = s

    def update(self, n: int = 1) -> None:
        self.n = min(self.n + n, self.total)
        frac = self.n / self.total
        filled = int(self._width * frac)
        bar = "=" * filled + (">" if filled < self._width else "")
        bar += "." * (self._width - len(bar))
        tail = f" {self._postfix}" if self._postfix else ""
        sys.stdout.write(f"\r{self.desc} |{bar}| {self.n}/{self.total} iter{tail}  ")
        sys.stdout.flush()

    def close(self) -> None:
        sys.stdout.write("\n")
        sys.stdout.flush()


def _make_iter_progress_bar(
    total: int, desc: str, enabled: bool
) -> _NullIterBar | _AsciiIterBar | object:
    if not enabled:
        return _NullIterBar()
    if _HAS_TQDM:
        return tqdm(  # type: ignore[operator]
            total=total,
            desc=desc,
            unit="iter",
            file=sys.stdout,
            dynamic_ncols=True,
        )
    return _AsciiIterBar(total, desc)


# --- 与 run.py 一致的物理与控制步长 ---
TIMESTEP = 1.0 / 1000.0
POS_DT = 1.0 / 50.0
STEPS_POS = int(round(POS_DT / TIMESTEP))

# 评分说明：期望位置相对无人机当前位置各轴在 [-4, 4] m；z 不低于地面附近下限
TARGET_POS_DELTA_MAX = 4.0
MIN_TARGET_Z = 0.05

# 达标阈值
TH_POS_MEAN = 0.01
TH_POS_STD = 0.01
TH_YAW_MEAN = 0.01
TH_YAW_STD = 0.001


def wrap_to_pi(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def check_action(unchecked_action):
    if isinstance(unchecked_action, (tuple, list)) and len(unchecked_action) in (4, 5):
        out = [
            float(np.clip(unchecked_action[0], -1, 1)),
            float(np.clip(unchecked_action[1], -1, 1)),
            float(np.clip(unchecked_action[2], -1, 1)),
            float(np.clip(unchecked_action[3], -1.74533, 1.74533)),
        ]
        if len(unchecked_action) == 5:
            out.append(unchecked_action[4])
        return tuple(out)
    return (0.0, 0.0, 0.0, 0.0)


def default_gain_vector_from_controller(c: controller.DOBController) -> np.ndarray:
    """6 维位置外环：[kp_xy, kp_z, ki_xy, ki_z, kd_xy, kd_z]（水平 x/y 共用）。"""
    return np.array(
        [
            float(c.kp_pos[0]),
            float(c.kp_pos[2]),
            float(c.ki_pos[0]),
            float(c.ki_pos[2]),
            float(c.kd_vel[0]),
            float(c.kd_vel[2]),
        ],
        dtype=float,
    )


def apply_gain_vector_pos_only(c: controller.DOBController, v: np.ndarray) -> None:
    """只写位置外环 6 维；kp_yaw/ki_yaw/kd_yaw 保持当前内存中的值不变。"""
    kp_xy, kp_z, ki_xy, ki_z, kd_xy, kd_z = v
    c.kp_pos = np.array([kp_xy, kp_xy, kp_z], dtype=float)
    c.ki_pos = np.array([ki_xy, ki_xy, ki_z], dtype=float)
    c.kd_vel = np.array([kd_xy, kd_xy, kd_z], dtype=float)


class HeadlessSimulator:
    """无 GUI、无风、无 matplotlib。"""

    def __init__(self):
        p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        self.plane_id = p.loadURDF("plane.urdf")
        self.start_pos = [0, 0, 1]
        self.start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.drone_id = p.loadURDF(
            "resources/tello.urdf", self.start_pos, self.start_orientation
        )

        self.M = 0.088
        self.L = 0.06
        self.KF = 0.566e-5
        self.KM = 0.762e-7
        self.K_TRANS = np.array([3.365e-2, 3.365e-2, 3.365e-2])
        self.TM = 0.0163
        self.tello_controller = TelloController(
            9.81, self.M, self.L, 0.35, self.KF, self.KM
        )

    def close(self):
        try:
            p.disconnect()
        except Exception:
            pass

    def compute_dynamics(self, rpm_values, lin_vel_world, quat):
        rotation = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        omega = rpm_values * (2 * np.pi / 60)
        omega_squared = omega**2
        motor_forces = omega_squared * self.KF
        thrust = np.array([0, 0, np.sum(motor_forces)])
        vel_body = np.dot(rotation.T, lin_vel_world)
        drag_body = -self.K_TRANS * vel_body
        force = drag_body + thrust
        z_torques = omega_squared * self.KM
        z_torque = -z_torques[0] - z_torques[1] + z_torques[2] + z_torques[3]
        x_torque = (
            -motor_forces[0] + motor_forces[1] + motor_forces[2] - motor_forces[3]
        ) * self.L
        y_torque = (
            -motor_forces[0] + motor_forces[1] - motor_forces[2] + motor_forces[3]
        ) * self.L
        torques = np.array([x_torque, y_torque, z_torque])
        return force, torques

    def spin_motors(self, rpm, timestep):
        for joint_index in range(4):
            rad_s = rpm[joint_index] * (2.0 * np.pi / 60.0)
            current_angle = p.getJointState(self.drone_id, joint_index)[0]
            new_angle = current_angle + rad_s * timestep
            p.resetJointState(self.drone_id, joint_index, new_angle)

    def motor_model(self, desired_rpm, current_rpm, dt):
        rpm_derivative = (desired_rpm - current_rpm) / self.TM
        return current_rpm + rpm_derivative * dt

    def run_trial(
        self,
        rng: np.random.Generator,
        duration_s: float,
        wind_enabled: bool = False,
    ) -> tuple[list[float], list[float]]:
        """返回该 trial 内外环采样点的位置误差范数列表与偏航误差绝对值列表。

        复位后按评分规则采样目标：位置 = 当前位置 + 各轴 U(-4,4) m，z 不低于 MIN_TARGET_Z；
        偏航为 U(-π, π)。
        """
        p.resetBasePositionAndOrientation(
            self.drone_id, self.start_pos, self.start_orientation
        )
        self.tello_controller.reset()
        ctrl = controller._dob_controller
        ctrl.reset()

        pos0, _ = p.getBasePositionAndOrientation(self.drone_id)
        pos0 = np.asarray(pos0, dtype=float)
        delta = rng.uniform(
            -TARGET_POS_DELTA_MAX, TARGET_POS_DELTA_MAX, size=3
        )
        t_xyz = pos0 + delta
        t_xyz[2] = max(float(t_xyz[2]), MIN_TARGET_Z)
        yaw_d = float(rng.uniform(-np.pi, np.pi))
        target = (float(t_xyz[0]), float(t_xyz[1]), float(t_xyz[2]), yaw_d)

        prev_rpm = np.zeros(4)
        desired_vel = np.zeros(3)
        yaw_rate_setpoint = 0.0
        loop_counter = 0  # 与 run.py 一致：累计 STEPS_POS 个子步后触发外环
        n_sub = int(round(duration_s / TIMESTEP))

        pos_norms: list[float] = []
        yaw_abs: list[float] = []

        x_d, y_d, z_d, yaw_d = target
        pos_d = np.array([x_d, y_d, z_d], dtype=float)

        for _ in range(n_sub):
            loop_counter += 1
            pos, quat = p.getBasePositionAndOrientation(self.drone_id)
            lin_vel_world, ang_vel_world = p.getBaseVelocity(self.drone_id)
            roll, pitch, yaw = p.getEulerFromQuaternion(quat)
            yaw_quat = p.getQuaternionFromEuler([0, 0, yaw])
            _, inverted_quat = p.invertTransform([0, 0, 0], quat)
            _, inverted_quat_yaw = p.invertTransform([0, 0, 0], yaw_quat)
            lin_vel = np.array(p.rotateVector(inverted_quat_yaw, lin_vel_world))
            ang_vel = np.array(p.rotateVector(inverted_quat, ang_vel_world))

            if loop_counter >= STEPS_POS:
                loop_counter = 0
                state = np.concatenate((pos, [roll, pitch, yaw]))
                controller_output = check_action(
                    controller.controller(state, target, POS_DT, wind_enabled)
                )
                desired_vel = np.array(controller_output[:3], dtype=float)
                yaw_rate_setpoint = float(controller_output[3])

                pos = np.asarray(pos, dtype=float)
                pe = pos_d - pos
                pos_norms.append(float(np.linalg.norm(pe)))
                yaw_abs.append(abs(wrap_to_pi(float(yaw_d) - float(yaw))))

            rpm = self.tello_controller.compute_control(
                desired_vel, lin_vel, quat, ang_vel, yaw_rate_setpoint, TIMESTEP
            )
            rpm = self.motor_model(rpm, prev_rpm, TIMESTEP)
            prev_rpm = rpm
            force, torque = self.compute_dynamics(rpm, lin_vel_world, quat)
            p.applyExternalForce(self.drone_id, -1, force, [0, 0, 0], p.LINK_FRAME)
            p.applyExternalTorque(self.drone_id, -1, torque, p.LINK_FRAME)
            self.spin_motors(rpm, TIMESTEP)
            p.stepSimulation()

        return pos_norms, yaw_abs


def evaluate_outer_loop(
    sim: HeadlessSimulator,
    rng: np.random.Generator,
    num_trials: int,
    duration_s: float,
    *,
    verbose_trials: bool = False,
    trial_progress_prefix: str = "评测",
) -> dict[str, float]:
    """
    对 NUM_TRIALS 次 trial 各跑 duration_s，聚合四个标量（对 trial 内统计量再平均）。

    每个 trial 内目标由 run_trial 按「相对当前位置 ±4 m + 随机 yaw」生成。

    verbose_trials: 为 True 时每完成一个 trial 打印一行（避免长时间无输出误以为卡死）。
    """
    trial_pos_means: list[float] = []
    trial_pos_stds: list[float] = []
    trial_yaw_means: list[float] = []
    trial_yaw_stds: list[float] = []

    steps_per_trial = int(round(duration_s / TIMESTEP))

    for k in range(num_trials):
        if verbose_trials:
            t0 = time.time()
            print(
                f"  {trial_progress_prefix} trial {k + 1}/{num_trials} "
                f"（本 trial 仿真 {duration_s:g}s ≈ {steps_per_trial} 物理步）...",
                flush=True,
            )
        pn, ya = sim.run_trial(rng, duration_s, wind_enabled=False)
        if verbose_trials:
            dt = time.time() - t0
            print(
                f"  {trial_progress_prefix} trial {k + 1}/{num_trials} 完成，用时 {dt:.1f}s",
                flush=True,
            )
        if len(pn) < 2:
            trial_pos_means.append(1e6)
            trial_pos_stds.append(1e6)
            trial_yaw_means.append(1e6)
            trial_yaw_stds.append(1e6)
            continue
        trial_pos_means.append(float(np.mean(pn)))
        trial_pos_stds.append(float(np.std(pn)))
        trial_yaw_means.append(float(np.mean(ya)))
        trial_yaw_stds.append(float(np.std(ya)))

    return {
        "pos_err_mean": float(np.mean(trial_pos_means)),
        "pos_err_std": float(np.mean(trial_pos_stds)),
        "yaw_err_mean": float(np.mean(trial_yaw_means)),
        "yaw_err_std": float(np.mean(trial_yaw_stds)),
    }


def passes_thresholds(m: dict[str, float]) -> bool:
    return (
        m["pos_err_mean"] < TH_POS_MEAN
        and m["pos_err_std"] < TH_POS_STD
        and m["yaw_err_mean"] < TH_YAW_MEAN
        and m["yaw_err_std"] < TH_YAW_STD
    )


def loss_from_metrics(m: dict[str, float]) -> float:
    """约束违反的平方惩罚 + 小幅跟踪误差和，供 Twiddle 使用。"""
    w = 1e6
    pen = (
        w * max(0.0, m["pos_err_mean"] - TH_POS_MEAN) ** 2
        + w * max(0.0, m["pos_err_std"] - TH_POS_STD) ** 2
        + w * max(0.0, m["yaw_err_mean"] - TH_YAW_MEAN) ** 2
        + w * max(0.0, m["yaw_err_std"] - TH_YAW_STD) ** 2
    )
    tie = (
        m["pos_err_mean"]
        + m["pos_err_std"]
        + m["yaw_err_mean"]
        + 10.0 * m["yaw_err_std"]
    )
    return pen + tie


def twiddle(
    sim: HeadlessSimulator,
    rng: np.random.Generator,
    p0: np.ndarray,
    dp0: np.ndarray,
    num_trials: int,
    duration_s: float,
    max_iters: int,
    tol: float,
    show_progress: bool = True,
    verbose_trials_on_initial_eval: bool = True,
    verbose_trials_in_search: bool = False,
) -> tuple[np.ndarray, dict[str, float], list[dict]]:
    """坐标上升式 Twiddle（与经典车辆转向角搜索同一思想）。"""
    p = np.array(p0, dtype=float, copy=True)
    dp = np.array(dp0, dtype=float, copy=True)
    apply_gain_vector_pos_only(controller._dob_controller, p)
    if show_progress:
        print(
            "Twiddle: 初始评测（第 0 轮，不计入 iter 进度）。"
            f"本轮需跑满 {num_trials} 个 trial；首个 trial 结束前终端可能安静较久。",
            flush=True,
        )
    best_m = evaluate_outer_loop(
        sim,
        rng,
        num_trials,
        duration_s,
        verbose_trials=show_progress and verbose_trials_on_initial_eval,
        trial_progress_prefix="初始评测",
    )
    best_L = loss_from_metrics(best_m)
    history: list[dict] = [{"p": p.copy(), "metrics": best_m, "loss": best_L}]
    if show_progress:
        print(
            "初始评测完成，进入 Twiddle 外环迭代（进度条仅统计外环 iter；"
            "每次尝试新参数仍会跑完整评测，期间可能长时间无新输出）。",
            flush=True,
        )

    bar = _make_iter_progress_bar(max_iters, "Twiddle", show_progress)
    it = 0
    try:
        while np.sum(dp) > tol and it < max_iters:
            it += 1
            improved = False
            for i in range(len(p)):
                if dp[i] < 1e-12:
                    continue
                # 尝试增加
                p_try = p.copy()
                p_try[i] += dp[i]
                apply_gain_vector_pos_only(controller._dob_controller, p_try)
                m = evaluate_outer_loop(
                    sim,
                    rng,
                    num_trials,
                    duration_s,
                    verbose_trials=show_progress and verbose_trials_in_search,
                    trial_progress_prefix=f"iter{it} dim{i}+",
                )
                L = loss_from_metrics(m)
                if L < best_L:
                    best_L = L
                    best_m = m
                    p = p_try
                    improved = True
                    history.append({"p": p.copy(), "metrics": m, "loss": L})
                    continue
                # 尝试减少
                p_try = p.copy()
                p_try[i] -= dp[i]
                if p_try[i] <= 0 and i < 6:
                    p_try[i] = 1e-4
                apply_gain_vector_pos_only(controller._dob_controller, p_try)
                m = evaluate_outer_loop(
                    sim,
                    rng,
                    num_trials,
                    duration_s,
                    verbose_trials=show_progress and verbose_trials_in_search,
                    trial_progress_prefix=f"iter{it} dim{i}-",
                )
                L = loss_from_metrics(m)
                if L < best_L:
                    best_L = L
                    best_m = m
                    p = p_try
                    dp[i] *= 1.1
                    improved = True
                    history.append({"p": p.copy(), "metrics": m, "loss": L})
                else:
                    dp[i] *= 0.9
            ok = passes_thresholds(best_m)
            bar.set_postfix_str(f"loss={best_L:.3g} ok={'Y' if ok else 'N'}")
            bar.update(1)
            if ok:
                break
    finally:
        bar.close()

    apply_gain_vector_pos_only(controller._dob_controller, p)
    return p, best_m, history


def main(argv: list[str] | None = None) -> int:
    fast = os.environ.get("PID_TUNE_FAST", "").lower() in ("1", "true", "yes")
    default_trials = 5 if fast else 50
    default_duration = 2.0 if fast else 10.0

    ap = argparse.ArgumentParser(description="无头、无风外环 PID 自动调参 / 评测")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-trials", type=int, default=default_trials)
    ap.add_argument("--duration", type=float, default=default_duration)
    ap.add_argument("--eval-only", action="store_true", help="只评测当前 controller 增益，不搜索")
    ap.add_argument("--max-iters", type=int, default=80)
    ap.add_argument("--tol", type=float, default=1e-3, help="sum(dp) 停止阈值")
    ap.add_argument(
        "--out-json",
        type=str,
        default="",
        help="将最优参数与指标写入 JSON 文件",
    )
    ap.add_argument(
        "--no-progress",
        action="store_true",
        help="关闭 Twiddle 外环迭代进度条与逐 trial 打印",
    )
    ap.add_argument(
        "--verbose-trials-search",
        action="store_true",
        help="Twiddle 搜索过程中每次完整评测也打印逐 trial（输出极多，默认仅初始评测打印）",
    )
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)

    sim = HeadlessSimulator()
    try:
        t0 = time.time()
        importlib.reload(controller)
        if args.eval_only:
            if not args.no_progress:
                print(
                    f"评测: {args.num_trials} trial × {args.duration:g}s 仿真/trial ...",
                    flush=True,
                )
            m = evaluate_outer_loop(
                sim,
                rng,
                args.num_trials,
                args.duration,
                verbose_trials=not args.no_progress,
                trial_progress_prefix="评测",
            )
            print("=== 评测结果（当前 controller.py）===")
        else:
            p0 = default_gain_vector_from_controller(controller._dob_controller)
            dp0 = np.maximum(0.05 * np.abs(p0), 0.02)
            p_best, m, hist = twiddle(
                sim,
                rng,
                p0,
                dp0,
                args.num_trials,
                args.duration,
                args.max_iters,
                args.tol,
                show_progress=not args.no_progress,
                verbose_trials_on_initial_eval=not args.no_progress,
                verbose_trials_in_search=args.verbose_trials_search,
            )
            print("=== Twiddle 搜索完成 ===")
            print(
                "最优 6 维位置外环 [kp_xy, kp_z, ki_xy, ki_z, kd_xy, kd_z]（偏航环未参与搜索）："
            )
            print(np.array2string(p_best, precision=4, separator=", "))
            if args.out_json:
                cf = controller._dob_controller
                iterations = [
                    {
                        "index": i,
                        "gain_vector": np.asarray(h["p"], dtype=float).tolist(),
                        "metrics": dict(h["metrics"]),
                        "loss": float(h["loss"]),
                    }
                    for i, h in enumerate(hist)
                ]
                payload = {
                    "run_config": {
                        "seed": args.seed,
                        "num_trials": args.num_trials,
                        "duration_s": args.duration,
                        "max_iters": args.max_iters,
                        "tol": args.tol,
                    },
                    "gain_vector": p_best.tolist(),
                    "gain_vector_labels": [
                        "kp_xy",
                        "kp_z",
                        "ki_xy",
                        "ki_z",
                        "kd_xy",
                        "kd_z",
                    ],
                    "yaw_pid_unchanged": {
                        "kp_yaw": cf.kp_yaw,
                        "ki_yaw": cf.ki_yaw,
                        "kd_yaw": cf.kd_yaw,
                    },
                    "final_metrics": dict(m),
                    "passes": passes_thresholds(m),
                    "history_len": len(hist),
                    "iterations": iterations,
                }
                with open(args.out_json, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
                print(f"已写入 {args.out_json}")

        dt = time.time() - t0
        print(
            f"trials={args.num_trials}, duration={args.duration}s, wall_time={dt:.1f}s"
        )
        print(f"  pos_err_mean : {m['pos_err_mean']:.6f}  (阈值 < {TH_POS_MEAN})")
        print(f"  pos_err_std  : {m['pos_err_std']:.6f}  (阈值 < {TH_POS_STD})")
        print(f"  yaw_err_mean : {m['yaw_err_mean']:.6f}  (阈值 < {TH_YAW_MEAN})")
        print(f"  yaw_err_std  : {m['yaw_err_std']:.6f}  (阈值 < {TH_YAW_STD})")
        print(f"  达标: {passes_thresholds(m)}")
        if not args.eval_only:
            print(
                "\n将下列位置外环字段拷入 controller.py 中 DOBController.__init__ "
                "（x/y 共用 kp_xy/ki_xy/kd_xy）："
            )
            cfin = controller._dob_controller
            v = default_gain_vector_from_controller(cfin)
            kp_xy, kp_z, ki_xy, ki_z, kd_xy, kd_z = v
            print(f"  kp_xy, kp_z = {kp_xy}, {kp_z}")
            print(f"  ki_xy, ki_z = {ki_xy}, {ki_z}")
            print(f"  kd_xy, kd_z = {kd_xy}, {kd_z}")
            print(f"  self.kp_pos = np.array([{kp_xy}, {kp_xy}, {kp_z}])")
            print(f"  self.ki_pos = np.array([{ki_xy}, {ki_xy}, {ki_z}])")
            print(f"  self.kd_vel = np.array([{kd_xy}, {kd_xy}, {kd_z}])")
            print(
                f"  （偏航环未调参，仍为 kp_yaw={cfin.kp_yaw}, ki_yaw={cfin.ki_yaw}, "
                f"kd_yaw={cfin.kd_yaw}）"
            )
    finally:
        sim.close()

    return 0 if passes_thresholds(m) else 1


if __name__ == "__main__":
    sys.exit(main())
