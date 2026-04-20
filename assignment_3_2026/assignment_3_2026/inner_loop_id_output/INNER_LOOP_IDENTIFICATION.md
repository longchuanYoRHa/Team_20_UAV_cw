# 内环系统辨识记录

> 目的：在不改动 `controller.py` 的前提下，直接对 `TelloController` 内环（速度→姿态→角速度级联 PID）做系统辨识，给出 $vx, vy, vz,$ yaw_rate 四个通道的参数化传递函数，作为后续**抗风外环 / DOB** 设计的模型依据。
>
> 脚本：`inner_loop_identification.py`
> 输出目录：`inner_loop_id_output/`

---

## 1. 背景与关键发现

### 1.1 PyBullet 仿真步长 vs run.py 里的 `timestep`

- PyBullet 的 `p.stepSimulation()` 默认按 `fixedTimeStep = 1/240 s` 步进。
- `run.py` **没有**调用 `p.setTimeStep()`，所以**真实物理步长就是 1/240 s**。
- 但 `run.py` 里的变量 `timestep = 1/1000` 被传给了：
  - `TelloController.compute_control(..., timestep)`
  - `motor_model(desired_rpm, current_rpm, dt=timestep)`
  - `spin_motors(rpm, timestep)`
  - `time.sleep(timestep - loop_time)`（1 kHz 节拍）
- 因此**每一次 `p.stepSimulation()`（物理推进 1/240 s ≈ 4.17 ms）内，电机一阶模型只被推进 1/1000 s**，即电机时间常数被人为放大了约 **4.17×**，内环 PID 的 D/I 项也按 1/1000 s 计算。
- 结论：**run.py 实际部署的内环，比"名义物理"慢 ~4×**。要设计的外环面对的是这个"慢"的内环，而不是理论快速内环。

### 1.2 原有 `open_loop_inner_identification.py` 的问题

- 采样 dt 用的是 `1/1000`，但 pybullet 实际只推进 `1/240` —— 时间轴被压缩 4.17×。
- 直接套**一阶模型**，而实际 vx/vy 响应是**显著欠阻尼**（≈20% 超调），一阶肯定拟不出来（输出的 `K=0.000, tau=infs` 就是证据）。

### 1.3 新脚本怎么做

1. **显式设置** `p.setTimeStep(1/240)`，采样时间轴用 `DT_PHYS = 1/240 s`；
2. 提供 `--ctrl-dt` 选项（默认 `1/1000` 复刻 run.py 的"不一致"；`--clean-physics` 切换到 `1/240` 看"理论上"的内环）；
3. 同时拟合**三种模型**选最优：
   - 一阶+延迟 $\;G(s)=\dfrac{K}{\tau s+1}e^{-Ls}$
   - 二阶欠阻尼+延迟 $\;G(s)=\dfrac{K\omega_n^2}{s^2+2\zeta\omega_n s+\omega_n^2}e^{-Ls}$
   - 二阶+零点+延迟 $\;G(s)=\dfrac{K(\tau_z s+1)\omega_n^2}{s^2+2\zeta\omega_n s+\omega_n^2}e^{-Ls}$
4. 用 `scipy.optimize.least_squares` 在**离散时域**上做最小二乘（带数值保护的 Euler 子步，延迟做线性插值，对非整数步延迟也可导）；
5. 附加 chirp（0.1–6 Hz 线性扫频）输入，用 FFT 取经验 FRF，与识别模型叠加画 Bode 做**频域交叉验证**。

---

## 2. 实验配置

| 项 | 值 |
|-----|----|
| 物理 dt（pybullet） | `1/240 s ≈ 4.167 ms` |
| 控制/电机 dt（默认） | `1/1000 s`（匹配 run.py） |
| 每次阶跃总时长 | 6.5 s（0.5 s 悬停 + 6 s 阶跃响应） |
| 每通道阶跃幅值 | `vx/vy`: ±0.30, +0.15；`vz`: ±0.20, +0.10；`yaw_rate`: ±0.60, +0.30 |
| chirp 频段 | 0.1 → 6 Hz（`vx/vy/vz` 幅值 0.15，`yaw_rate` 0.4），持续 20 s |
| `vz` 负幅值测试 | 起始高度抬到 ≥3 m，避免下降到地板 |
| 最优模型判据 | RMSE 最小；若更简单模型 RMSE 在最优 5% 以内优先选简单的 |

---

## 3. 识别结果（默认 `match_run_py_timing=true`）

所有通道的 DC 增益 $K$ 都接近 1，内环稳态基本不损失指令幅度。

| 通道 | 首选模型 | $K$ | $\omega_n$ (rad/s) | $\zeta$ | $\tau_z$ / $\tau$ (s) | $L$ (s) | **-3 dB 带宽** | RMSE |
|------|----------|-----|---------------------|---------|-----------------------|---------|-------------|------|
| **vx** | 二阶+零点 | 0.9791 | 1.169 | 0.426 | $\tau_z$ = 0.2053 | 0.1153 | 1.58 rad/s ≈ **0.25 Hz** | 7.5e-4 |
| **vy** | 二阶+零点 | 0.9791 | 1.169 | 0.426 | $\tau_z$ = 0.2053 | 0.1153 | 1.58 rad/s ≈ **0.25 Hz** | 7.5e-4 |
| **vz** | 一阶 | 0.9564 | — | — | $\tau$ = 0.1659 | 0.0560 | 6.03 rad/s ≈ **0.96 Hz** | 1.1e-2 |
| **yaw_rate** | 二阶欠阻尼 | 1.0019 | 40.05 | 0.180 | — | ≈0 | 60.8 rad/s ≈ **9.68 Hz** | 3.4e-3 |

> `vx`/`vy` 参数完全一致 —— 对称机架 + 相同 PID 增益下这是预期的。

### 3.1 派生的时域特性（来自 $\omega_n, \zeta$）

| 通道 | 峰值时间 $t_p$ | 超调 $\sigma\%$ | 2% 沉降时间 $t_s$ |
|------|----------------|-----------------|-------------------|
| vx/vy | 2.97 s | 22.8 % | 8.0 s |
| yaw_rate | 0.080 s | 56.3 % | 0.56 s |

vz 一阶模型的 2% 沉降 ≈ $4\tau$ = 0.66 s。

### 3.2 Bode 交叉验证（chirp）

- **vx / yaw_rate**：识别模型 Bode 幅值曲线与 chirp FRF 红点在低频段（与外环设计相关的 < 10 rad/s）吻合良好；高频有一些未建模的谐振 / 反谐振，但超出外环带宽，不影响设计。
- **vz**：chirp 信号幅值较小 (0.15)，信噪比有限，但 1 阶模型在 < 2 rad/s 段与 FRF 一致。

---

## 4. 对"抗风外环 + DOB"设计的直接启示

### 4.1 频率分配原则（基于时间尺度分离）

内环带宽 $\omega_{\text{BW,inner}}$ 是外环可用的**上限**。保守做法是外环穿越频率 $\omega_c \le \omega_{\text{BW,inner}} / (3 \sim 5)$。

| 通道 | 内环 BW | 外环建议穿越频率 $\omega_c$ | DOB Q 滤波器带宽 |
|------|---------|------------------------------|--------------------|
| x, y | 1.58 rad/s | **≤ 0.3–0.5 rad/s** | ≤ 1 rad/s |
| z    | 6.0 rad/s  | ≤ 1.2–2 rad/s                | ≤ 3 rad/s |
| yaw  | 60.8 rad/s | ≤ 12 rad/s（已够用）         | ≤ 30 rad/s |

### 4.2 水平通道是最吃紧的

- **BW ≈ 0.25 Hz，等效延迟 115 ms，阻尼 ζ≈0.43（轻度欠阻尼）**——这三个数决定了：
  1. 外环 PID 的 P 增益不能太大，否则会激发内环欠阻尼极点产生振荡；
  2. 积分项小心积分饱和（当前 `controller.py` 已经在做 leak + saturation + yaw-aligned gating，是必要的）；
  3. DOB 的名义模型 $G_n(s)$ 就用 $G_{xy}(s)$ 本身；Q-filter 低通 cutoff 放 ~1 rad/s，确保 $G_n^{-1}Q$ 严格真（二阶+零点的逆是 $\frac{s^2+2\zeta\omega_n s+\omega_n^2}{K(\tau_z s+1)\omega_n^2}$，是 proper 的，但仍需要 Q 是二阶低通才行）。

### 4.3 抗风鲁棒性评估要用这些参数

把识别出的 $G(s)$ 作为名义模型，加上"风阻力 / 小幅参数摄动"作为不确定性，做 $\mu$-综合或环路成型 (loop shaping)，目标是：
- 灵敏度 $|S(j\omega)|$ 在 < 0.3 rad/s 低频段 ≤ −20 dB（稳态抗风能力）；
- 互补灵敏度 $|T(j\omega)|$ 在 > 3 rad/s 高频段 ≤ 0 dB 且滚降，避免放大测量噪声；
- 模块留 ≥ 6 dB 增益裕度、≥ 45° 相位裕度以对抗延迟 L ≈ 0.115 s 带来的相位损失。

---

## 5. 在外环代码里怎么用

### 5.1 直接读 JSON

```python
import json
with open("inner_loop_id_output/inner_loop_params.json") as f:
    ids = json.load(f)

vx = ids["axes"]["vx"]["params_mean_over_amps"]
K, wn, zeta, tau_z, L = vx["K"], vx["wn"], vx["zeta"], vx["tau_z"], vx["L"]
```

### 5.2 构造名义传递函数（外环设计用）

```python
import numpy as np
from scipy import signal

# vx/vy:  G(s) = K*(tau_z*s + 1)*wn^2 / (s^2 + 2*zeta*wn*s + wn^2)
num_xy = K * wn**2 * np.array([tau_z, 1.0])
den_xy = np.array([1.0, 2.0 * zeta * wn, wn**2])
G_xy = signal.TransferFunction(num_xy, den_xy)    # 延迟 L 可在频响里单独加 exp(-jωL)

# vz:     G(s) = K / (tau*s + 1)
vz = ids["axes"]["vz"]["params_mean_over_amps"]
G_z = signal.TransferFunction([vz["K"]], [vz["tau"], 1.0])
```

### 5.3 现有 `DOBController` 的意义重估

- 当前 `controller.py` 里 `k_dob = [0.10, 0.10, 0.00]`, `k_comp = [0.16, 0.16, 0.00]`, `dob_leak=[0.55,0.55,0.40]` 本质上是一个**一阶低通 + 比例补偿**的经验 DOB；
- 用识别出的 $G_{xy}(s)$ 反推出"最佳 Q 滤波器"后，可以替换掉这组经验增益，做严格意义上的 $\hat d = G_n^{-1}Q\,y - Q\,u$ 观测器。

---

## 6. 文件清单

```
inner_loop_identification.py           辨识脚本（本次新增）
inner_loop_id_output/
├─ inner_loop_params.json              机器可读识别结果（外环/DOB 代码入口）
├─ summary.csv                         每次阶跃 × 每种模型的 RMSE
├─ vx_fit.png  vx_bode.png             三种模型叠加时域/频域图
├─ vy_fit.png  vy_bode.png
├─ vz_fit.png  vz_bode.png
├─ yaw_rate_fit.png  yaw_rate_bode.png
├─ <axis>_step_<±amp>.csv              每条阶跃原始数据
└─ INNER_LOOP_IDENTIFICATION.md        本文件
```

---

## 7. 复现命令

```bash
cd assignment_3_2026/assignment_3_2026

# 默认：匹配 run.py 部署时序 + chirp 频域验证（推荐）
python inner_loop_identification.py --with-chirp

# 只跑一个通道做快速冒烟测试
python inner_loop_identification.py --quick --axes vx

# 切换到"物理一致"dt（1/240 全栈统一），看内环"理论上"多快
python inner_loop_identification.py --clean-physics --with-chirp
```

---

## 8. 遗留 / 后续工作

1. **vx/vy 的等效延迟 L=115 ms** 比较大，可能包含了 outer→inner 采样的量化效应。如果把外环调到 50 Hz 更新，这部分延迟会暴露在外环的开环增益里，建议在外环设计时显式把 `exp(-0.115 s)` 放进 Nyquist/Bode 评估。
2. **yaw_rate 在 30–40 rad/s 有一个反谐振缺口**（chirp FRF 可见），二阶模型没捕捉。对外环影响不大（工作带宽远低于此），但若后续做偏航快速跟踪（如剧烈姿态变更）需要三阶或 notch 模型。
3. **vz 的 chirp FRF 在 3–10 rad/s 有谐振/抖动**，可能是推力-姿态耦合（pitch/roll 变化时升力投影到 z 产生额外动态）。可以进一步用多正弦信号 + 多段拟合。
4. 若后续把 run.py 的 `timestep` bug 修掉（让 `motor_model`/`spin_motors` 用 `1/240`），**所有参数都要重识别**（用 `--clean-physics` 跑一遍对比）。初步对比结果：修正 dt 后，内环变快 ≈4×（vx $\omega_n$ 保持 ~1.18 但零点 $\tau_z$ 变 0.086 s，等效响应更快、延迟降到 7 ms），整体定性不变。
