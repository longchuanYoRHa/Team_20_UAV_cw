import json
import os
import numpy as np
from collections import deque

# ============================================================
# Model-aware DOB on XY + controller.py's Z/yaw verbatim.
#
# 设计目标:
#   1. Z 通道、yaw 通道、yaw-align 门控、速度估计完全沿用 controller.py
#      的成熟配方 (前段响应丝滑、Z/yaw 响应已经调得很好)。
#   2. XY 通道使用"内环名义模型并行仿真 + Q-filter DOB"，考虑内环 0.25 Hz
#      带宽 / 115 ms 延迟 / ζ≈0.43 的真实动力学，让外环穿越频率保守在
#      ~0.4 rad/s，避免激发内环欠阻尼极点。
#   3. 消除 controller_hybrid_xy.py 末段抖动的三大根因:
#      (a) 取消 xy_hold 硬切换 (0 / 满 PID 跳变) —— 改为基于误差的平滑淡出；
#      (b) 取消 axis deadzone 清零 pos_err_body —— 改为 soft-tanh deadzone，
#          让 p_term 平滑过零而非阶跃；
#      (c) 取消 DOB 硬启用/禁用 radius —— 改为 smoothstep 淡入淡出，
#          d_hat 始终连续演化，不被反复清零。
#
# Interface (unchanged):
#   controller(state, target_pos, dt, wind_enabled=False)
#       -> (vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd)
# ============================================================


FIXED_DT = 0.0833  # 与 run.py 的 50 Hz 外环节拍对齐 (同 controller.py)


# ---------- 识别参数加载 (默认从 inner_loop_id_output/inner_loop_params.json) ----------
def _load_inner_params():
    """优先从 json 读取；读取失败则回退为 INNER_LOOP_IDENTIFICATION.md 里的值。"""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "..", "inner_loop_id_output", "inner_loop_params.json"),
        os.path.join(here, "inner_loop_id_output", "inner_loop_params.json"),
    ]
    for path in candidates:
        try:
            with open(os.path.normpath(path), "r", encoding="utf-8") as f:
                data = json.load(f)
            vx = data["axes"]["vx"]["params_mean_over_amps"]
            vz = data["axes"]["vz"]["params_mean_over_amps"]
            return {
                "xy": {
                    "K": float(vx["K"]),
                    "wn": float(vx["wn"]),
                    "zeta": float(vx["zeta"]),
                    "tau_z": float(vx["tau_z"]),
                    "L": float(vx["L"]),
                },
                "z": {
                    "K": float(vz["K"]),
                    "tau": float(vz["tau"]),
                    "L": float(vz["L"]),
                },
            }
        except (OSError, KeyError, ValueError):
            continue
    # Fallback (identical to the numbers in INNER_LOOP_IDENTIFICATION.md)
    return {
        "xy": {"K": 0.9791, "wn": 1.1688, "zeta": 0.4259, "tau_z": 0.2053, "L": 0.1153},
        "z":  {"K": 0.9564, "tau": 0.1659, "L": 0.0560},
    }


INNER = _load_inner_params()


# ---------- 辅助函数 ----------
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
    # y = x * (1 - exp(-(x/eps)^2))  —— 对称、光滑、|x|>>eps 时极快趋近 x
    return x * (1.0 - np.exp(-(x / eps) ** 2))


# ---------- 内环二阶+零点+延迟模型 (vx / vy 各一份) ----------
class _SecondOrderZeroDelay:
    """G(s) = K * wn^2 * (tau_z s + 1) / (s^2 + 2 zeta wn s + wn^2) * e^{-L s}

    状态空间 (controllable canonical):
        x1_dot = x2
        x2_dot = -wn^2 x1 - 2 zeta wn x2 + wn^2 u_delayed
        y      = K * (x1 + tau_z * x2)
    """

    def __init__(self, K, wn, zeta, tau_z, delay_s, hist_len=24):
        self.K = float(K)
        self.wn = float(wn)
        self.zeta = float(zeta)
        self.tau_z = float(tau_z)
        self.delay_s = max(0.0, float(delay_s))
        self.x1 = 0.0
        self.x2 = 0.0
        self._hist = deque([0.0] * hist_len, maxlen=hist_len)

    def reset(self):
        self.x1 = 0.0
        self.x2 = 0.0
        self._hist.clear()
        self._hist.extend([0.0] * self._hist.maxlen)

    def _delayed(self, u, dt):
        self._hist.append(float(u))
        d = self.delay_s / max(dt, 1e-6)
        n0 = int(np.floor(d))
        frac = d - n0
        hist = list(self._hist)
        n = len(hist)
        idx_newer = max(0, min(n - 1, n - 1 - n0))
        idx_older = max(0, min(n - 1, idx_newer - 1))
        return (1.0 - frac) * hist[idx_newer] + frac * hist[idx_older]

    def step(self, u, dt):
        dt = max(dt, 1e-6)
        u_d = self._delayed(u, dt)
        # Euler 子步，保证慢内环 (~0.25 Hz) 数值稳定
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


# ---------- 二阶低通 Q-filter (DOB 用) ----------
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


# ============================================================
# 主控制器
# ============================================================
class ModelDOBxyController:
    def __init__(self):
        # ---------------- XY 外环增益 ----------------
        # 内环 BW ≈ 1.58 rad/s ⇒ 外环 ωc ≤ ~0.4 rad/s 保守设计。
        # 比 controller.py (kp_xy=0.672) 低接近一半；比 hybrid_xy (kp=0.34) 略高，
        # 因为此处还有 model-based DOB 接管稳态抗风任务，P 不需要太大。
        kp_xy = 0.38
        ki_xy = 0.040
        kd_xy = 0.30
        ki_sat_xy = 0.40

        # ---------------- Z 外环增益 (直接抄 controller.py) ----------------
        kp_z = 1.18
        ki_z = 0.60
        kd_z = 0.96
        ki_sat_z = 0.58

        self.kp_pos = np.array([kp_xy, kp_xy, kp_z], dtype=float)
        self.ki_pos = np.array([ki_xy, ki_xy, ki_z], dtype=float)
        self.kd_vel = np.array([kd_xy, kd_xy, kd_z], dtype=float)
        self.ki_pos_sat = np.array([ki_sat_xy, ki_sat_xy, ki_sat_z], dtype=float)

        # ---------------- Yaw 通道 (直接抄 controller.py) ----------------
        self.kp_yaw = 2.05
        self.ki_yaw = 0.42
        self.kd_yaw = 0.16
        self.ki_yaw_sat = 0.18
        self.int_yaw_leak = 2.0
        self.max_yaw_rate = 1.74533
        self.yaw_align_tol = 0.167

        # ---------------- 积分泄漏 ----------------
        # XY 泄漏 1.8: 避免慢内环下积分累过头，但保留足够末段稳态权重
        # (λ=2.6 时末段 int≈err/2.6 几乎不起作用 → err∈[0.01,0.02]m 收敛偏慢).
        # Z 与 controller.py 一致。
        self.int_pos_leak = np.array([1.8, 1.8, 1.5], dtype=float)

        # ---------------- 命令限幅 ----------------
        self.max_vel = np.array([0.78, 0.78, 0.78], dtype=float)
        self.max_xy_speed = 0.78  # 与 controller.py 同

        # ---------------- 速度估计 (与 controller.py 完全一致) ----------------
        self.vel_est_world = np.zeros(3, dtype=float)
        self.prev_pos_error_world = None
        self.vel_est_lpf_alpha = 0.56

        # ---------------- XY 内环名义模型 (来自 inner_loop_params.json) ----------------
        mx = INNER["xy"]
        self.model_x = _SecondOrderZeroDelay(mx["K"], mx["wn"], mx["zeta"], mx["tau_z"], mx["L"])
        self.model_y = _SecondOrderZeroDelay(mx["K"], mx["wn"], mx["zeta"], mx["tau_z"], mx["L"])
        self.model_vel_xy = np.zeros(2, dtype=float)

        # ---------------- Q-filter DOB ----------------
        # wq = 0.8 rad/s 满足 ≤ BW_inner/2 = 0.8 rad/s，严格时间尺度分离；
        # zeta_q = 0.95 近临界阻尼，消除 Q 滤波器本身带来的峰值。
        self.q_filter_x = _SecondOrderLowpass(wq=0.80, zeta_q=0.95)
        self.q_filter_y = _SecondOrderLowpass(wq=0.80, zeta_q=0.95)
        self.d_hat_xy = np.zeros(2, dtype=float)
        self.d_hat_max = 0.40
        self.k_comp_xy = 0.80  # DOB 补偿基准增益；末端由 smoothstep 平滑淡出

        # ---------------- 命令平滑 (仅对 XY 做轻度 LPF) ----------------
        # 关键: β 远小于 hybrid_xy 的 0.92，避免过度相位滞后叠加 115ms 内环延迟。
        self.cmd_xy_lpf_beta = 0.70
        self.prev_xy_cmd = np.zeros(2, dtype=float)

        # ---------------- 平滑 near-target 淡出阈值 ----------------
        # DOB 补偿强度 = 1 - smoothstep(|err_xy|, dob_full_edge, dob_zero_edge)
        # 当误差 >= dob_full_edge 时 DOB 全开；<= dob_zero_edge 时完全退出。
        # 使用平滑过渡（C1 连续），杜绝硬切换引发的极限环。
        self.dob_zero_edge = 0.020
        self.dob_full_edge = 0.080
        # D 项 (-kd*vel_est_body) 在极近目标时淡出，避免与 P 项符号相反产生脉冲。
        self.d_term_zero_edge = 0.010
        self.d_term_full_edge = 0.045
        # Soft deadzone on pos_err_body: 仅在 ~0.4cm 以内做光滑衰减，不会阶跃。
        # (原 0.008 在 err=0.01m 时把 p_term 削弱到 79%, 末段收敛偏慢)
        self.xy_pos_err_soft_eps = 0.004

        # ---------------- 近目标 P 增益 boost ----------------
        # 在 err∈[p_boost_full_edge, p_boost_zero_edge] 区间平滑抬升 P 增益，
        # 用来抵消"DOB 已淡出 / I 被 leak / D 已淡出"时的末段推力不足。
        # 最大 boost = 1 + p_boost_gain (在 err <= p_boost_full_edge 处取到).
        # C1 连续 (smoothstep), 且仅在极小误差区起作用, 不会激发内环欠阻尼极点。
        self.p_boost_full_edge = 0.002
        self.p_boost_zero_edge = 0.020
        self.p_boost_gain = 0.60

        # ---------------- 内部状态 ----------------
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

    # -------- 与 controller.py 一致的速度估计 --------
    def _update_vel_est_from_pos_error(self, pos_err_world, dt):
        if dt > 1e-6 and self.prev_pos_error_world is not None:
            pos_derivative = (pos_err_world - self.prev_pos_error_world) / dt
            pos_derivative = np.clip(pos_derivative, -3.0, 3.0)
            raw_vel = -pos_derivative
            a = self.vel_est_lpf_alpha
            self.vel_est_world = a * raw_vel + (1.0 - a) * self.vel_est_world
        self.prev_pos_error_world = pos_err_world.copy()

    # -------- XY DOB: 并行模型 + 二阶 Q-filter --------
    def _update_xy_dob(self, vel_cmd_nom_xy, vel_est_xy, dt, wind_enabled):
        # 1. 用"喂给外环输出的"名义命令推进内环名义模型，得到模型速度预测
        self.model_vel_xy[0] = self.model_x.step(vel_cmd_nom_xy[0], dt)
        self.model_vel_xy[1] = self.model_y.step(vel_cmd_nom_xy[1], dt)

        # 2. 模型失配 = 实测 - 模型预测 (反映外部扰动 / 参数偏差)
        mismatch = vel_est_xy - self.model_vel_xy
        mismatch = np.clip(mismatch, -1.0, 1.0)

        # 3. 无风时缓慢衰减，不主动学习 (避免把模型参数偏差误认为是风)
        enable_scale = 1.0 if wind_enabled else 0.15
        u_x = enable_scale * mismatch[0]
        u_y = enable_scale * mismatch[1]

        # 4. 通过低带宽二阶 Q-filter 得到最终 d_hat
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

        # -------- 1) 积分 (与 controller.py 同策略) --------
        # Z 一直积；XY 不再等待 yaw 对齐（边转向边飞）。
        # 但为避免大 yaw 误差时积分 windup，XY 积分随 yaw 误差平滑衰减。
        self.int_pos_body[2] += pos_err_body[2] * dt
        yaw_i_scale = 1.0 - smoothstep(abs(yaw_err), self.yaw_align_tol, 3.0 * self.yaw_align_tol)
        self.int_pos_body[0] += yaw_i_scale * pos_err_body[0] * dt
        self.int_pos_body[1] += yaw_i_scale * pos_err_body[1] * dt
        self.int_pos_body *= np.exp(-self.int_pos_leak * dt)
        self.int_pos_body = np.clip(self.int_pos_body, -self.ki_pos_sat, self.ki_pos_sat)

        # -------- 2) XY 的 soft deadzone (不阶跃, C1 光滑) --------
        pos_err_body_for_pid = pos_err_body.copy()
        pos_err_body_for_pid[0] = soft_deadzone(pos_err_body_for_pid[0], self.xy_pos_err_soft_eps)
        pos_err_body_for_pid[1] = soft_deadzone(pos_err_body_for_pid[1], self.xy_pos_err_soft_eps)

        # -------- 3) PID 三项 --------
        p_term = self.kp_pos * pos_err_body_for_pid
        i_term = self.ki_pos * self.int_pos_body
        d_term_full = -self.kd_vel * vel_est_body

        xy_err_norm = float(np.linalg.norm(pos_err_body[0:2]))

        # 近目标 P boost: err 从 p_boost_zero_edge 降到 p_boost_full_edge 时,
        # P 项平滑增长到 (1 + p_boost_gain) 倍, 加快最后 1-2cm 的收敛速度。
        # 此时 DOB/D 均已淡出, 单独增大 P 不会激发内环欠阻尼极点 (命令幅值极小)。
        near_factor = 1.0 - smoothstep(xy_err_norm, self.p_boost_full_edge, self.p_boost_zero_edge)
        p_boost = 1.0 + self.p_boost_gain * near_factor
        p_term[0] *= p_boost
        p_term[1] *= p_boost

        # D 项近目标平滑淡出 (防止与 P 项反向产生脉冲; 与 controller.py 不同的是这里更保守)
        d_fade_xy = smoothstep(xy_err_norm, self.d_term_zero_edge, self.d_term_full_edge)
        d_term = d_term_full.copy()
        d_term[0] *= d_fade_xy
        d_term[1] *= d_fade_xy

        vel_cmd_nominal = p_term + i_term + d_term

        # -------- 4) XY Model-aided DOB --------
        # 用 "外环将要输出给内环的名义速度命令" 喂给内环名义模型，
        # 然后比较实测速度估计得到失配 -> 通过低带宽 Q-filter -> d_hat
        self._update_xy_dob(
            vel_cmd_nominal[0:2],
            vel_est_body[0:2],
            dt,
            wind_enabled,
        )

        # DOB 补偿强度基于误差大小平滑淡出 (远场全开 / 近场退出，C1 过渡)
        dob_fade = smoothstep(xy_err_norm, self.dob_zero_edge, self.dob_full_edge)
        dob_comp_xy = self.k_comp_xy * dob_fade * self.d_hat_xy
        dob_comp = np.array([dob_comp_xy[0], dob_comp_xy[1], 0.0], dtype=float)

        # -------- 5) 合成最终命令 --------
        vel_cmd = vel_cmd_nominal + dob_comp

        # yaw 未对齐时不再禁止水平移动：允许边转向边飞向目标。

        # XY 轻度 LPF: 平滑但不过度滞后；不对 Z 做 LPF (Z 内环快, 没必要)
        xy_cmd = vel_cmd[0:2].copy()
        xy_cmd = self.cmd_xy_lpf_beta * self.prev_xy_cmd + (1.0 - self.cmd_xy_lpf_beta) * xy_cmd
        self.prev_xy_cmd = xy_cmd.copy()
        vel_cmd[0] = xy_cmd[0]
        vel_cmd[1] = xy_cmd[1]

        # 幅值限幅
        vel_cmd = np.clip(vel_cmd, -self.max_vel, self.max_vel)
        xy_speed = float(np.linalg.norm(vel_cmd[0:2]))
        if xy_speed > self.max_xy_speed and xy_speed > 1e-9:
            vel_cmd[0:2] *= self.max_xy_speed / xy_speed

        # -------- 6) Yaw PID (直接抄 controller.py 原版) --------
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

        # -------- 7) telemetry --------
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


# ---------- 模块级单例 / 外部接口 (与 controller.py 保持一致) ----------
_controller = ModelDOBxyController()


def get_integral_telemetry():
    d = _controller.last_debug
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
        "dob_comp_body": (float(dob_comp[0]), float(dob_comp[1]), float(dob_comp[2])),
        "vel_est_body": (float(vel_est[0]), float(vel_est[1]), float(vel_est[2])),
        "vel_model_body": (float(vel_model[0]), float(vel_model[1]), float(vel_model[2])),
    }


def controller(state, target_pos, dt, wind_enabled=False):
    # 与 controller.py 一致：锁定有效外环步长 0.0833 s (50 Hz / run.py 的 pos_control_timestep)
    dt = FIXED_DT
    return _controller.compute(state, target_pos, dt, wind_enabled)
