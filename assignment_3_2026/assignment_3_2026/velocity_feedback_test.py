import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pybullet as p

from src.tello_controller import TelloController

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


@dataclass
class SimParams:
    g: float = 9.81
    mass: float = 0.088
    arm_length: float = 0.06
    max_angle: float = 0.35
    kf: float = 0.566e-5
    km: float = 0.762e-7
    k_trans: np.ndarray = field(default_factory=lambda: np.array([3.365e-2, 3.365e-2, 3.365e-2], dtype=float))
    motor_time_constant: float = 0.0163
    inertia: np.ndarray = field(default_factory=lambda: np.diag([0.00679, 0.00679, 0.01313]))
    max_rpm: float = 28000.0


def quat_xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=float)


def quat_wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    return np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=float)


def quat_multiply_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def normalize_quat_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat_xyzw)
    if norm < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return quat_xyzw / norm


def integrate_quaternion_xyzw(quat_xyzw: np.ndarray, omega_body: np.ndarray, dt: float) -> np.ndarray:
    quat_wxyz = quat_xyzw_to_wxyz(quat_xyzw)
    omega_quat = np.array([0.0, omega_body[0], omega_body[1], omega_body[2]], dtype=float)
    quat_dot_wxyz = 0.5 * quat_multiply_wxyz(quat_wxyz, omega_quat)
    quat_next_wxyz = quat_wxyz + quat_dot_wxyz * dt
    return normalize_quat_xyzw(quat_wxyz_to_xyzw(quat_next_wxyz))


class VelocityInnerLoopSimulator:
    def __init__(self, params: SimParams):
        self.params = params
        self.controller = TelloController(
            params.g,
            params.mass,
            params.arm_length,
            params.max_angle,
            params.kf,
            params.km,
        )
        self.reset()

    def reset(self) -> None:
        self.controller.reset()
        self.pos_world = np.array([0.0, 0.0, 1.0], dtype=float)
        self.vel_world = np.zeros(3, dtype=float)
        self.quat_xyzw = np.array(p.getQuaternionFromEuler([0.0, 0.0, 0.0]), dtype=float)
        self.omega_body = np.zeros(3, dtype=float)
        self.rpm = np.zeros(4, dtype=float)

    def current_yaw_frame_velocity(self) -> np.ndarray:
        yaw = p.getEulerFromQuaternion(self.quat_xyzw)[2]
        yaw_quat = p.getQuaternionFromEuler([0.0, 0.0, yaw])
        _, inv_yaw_quat = p.invertTransform([0.0, 0.0, 0.0], yaw_quat)
        return np.array(p.rotateVector(inv_yaw_quat, self.vel_world), dtype=float)

    def compute_dynamics(self) -> tuple[np.ndarray, np.ndarray]:
        rotation = np.array(p.getMatrixFromQuaternion(self.quat_xyzw), dtype=float).reshape(3, 3)
        omega = self.rpm * (2.0 * np.pi / 60.0)
        omega_squared = omega**2
        motor_forces = omega_squared * self.params.kf
        thrust = np.array([0.0, 0.0, np.sum(motor_forces)], dtype=float)
        vel_body = rotation.T @ self.vel_world
        drag_body = -self.params.k_trans * vel_body
        force_body = drag_body + thrust

        z_torques = omega_squared * self.params.km
        z_torque = -z_torques[0] - z_torques[1] + z_torques[2] + z_torques[3]
        x_torque = (-motor_forces[0] + motor_forces[1] + motor_forces[2] - motor_forces[3]) * self.params.arm_length
        y_torque = (-motor_forces[0] + motor_forces[1] - motor_forces[2] + motor_forces[3]) * self.params.arm_length
        torque_body = np.array([x_torque, y_torque, z_torque], dtype=float)
        return force_body, torque_body

    def step(self, desired_vel_yaw_body: np.ndarray, yaw_rate_setpoint: float, dt: float) -> dict:
        lin_vel_yaw = self.current_yaw_frame_velocity()
        desired_rpm = self.controller.compute_control(
            np.asarray(desired_vel_yaw_body, dtype=float),
            lin_vel_yaw,
            self.quat_xyzw,
            self.omega_body,
            float(yaw_rate_setpoint),
            dt,
        )

        self.rpm = self.rpm + (desired_rpm - self.rpm) / self.params.motor_time_constant * dt
        self.rpm = np.clip(self.rpm, 0.0, self.params.max_rpm)

        force_body, torque_body = self.compute_dynamics()
        rotation = np.array(p.getMatrixFromQuaternion(self.quat_xyzw), dtype=float).reshape(3, 3)
        accel_world = rotation @ force_body / self.params.mass + np.array([0.0, 0.0, -self.params.g], dtype=float)
        omega_dot_body = np.linalg.solve(
            self.params.inertia,
            torque_body - np.cross(self.omega_body, self.params.inertia @ self.omega_body),
        )

        self.pos_world = self.pos_world + self.vel_world * dt
        self.vel_world = self.vel_world + accel_world * dt
        self.quat_xyzw = integrate_quaternion_xyzw(self.quat_xyzw, self.omega_body, dt)
        self.omega_body = self.omega_body + omega_dot_body * dt

        euler = np.array(p.getEulerFromQuaternion(self.quat_xyzw), dtype=float)
        return {
            "pos_world": self.pos_world.copy(),
            "vel_world": self.vel_world.copy(),
            "vel_yaw_body": lin_vel_yaw.copy(),
            "quat_xyzw": self.quat_xyzw.copy(),
            "euler": euler,
            "omega_body": self.omega_body.copy(),
            "rpm": self.rpm.copy(),
            "force_body": force_body.copy(),
            "torque_body": torque_body.copy(),
        }


def build_command(axis: str, amplitude: float) -> np.ndarray:
    cmd = np.zeros(3, dtype=float)
    axis_to_index = {"x": 0, "y": 1, "z": 2}
    cmd[axis_to_index[axis]] = amplitude
    return cmd


def compute_step_metrics(time_s: np.ndarray, cmd: np.ndarray, response: np.ndarray, step_start_s: float) -> dict:
    mask = time_s >= step_start_s
    t = time_s[mask] - step_start_s
    u = cmd[mask]
    y = response[mask]

    final_cmd = float(np.mean(u[-max(1, min(100, len(u))):]))
    final_resp = float(np.mean(y[-max(1, min(100, len(y))):]))
    steady_state_error = final_cmd - final_resp

    if abs(final_cmd) < 1e-9:
        overshoot_pct = 0.0
        rise_time_s = np.nan
        settling_time_s = np.nan
        return {
            "steady_state_error": steady_state_error,
            "overshoot_pct": overshoot_pct,
            "rise_time_s": rise_time_s,
            "settling_time_s": settling_time_s,
        }

    peak = float(np.max(y) if final_cmd > 0 else np.min(y))
    overshoot_pct = max(0.0, (peak - final_cmd) / abs(final_cmd) * 100.0) if final_cmd > 0 else max(0.0, (final_cmd - peak) / abs(final_cmd) * 100.0)

    low = 0.1 * final_cmd
    high = 0.9 * final_cmd
    rise_time_s = np.nan
    if final_cmd > 0:
        idx_low = np.where(y >= low)[0]
        idx_high = np.where(y >= high)[0]
    else:
        idx_low = np.where(y <= low)[0]
        idx_high = np.where(y <= high)[0]
    if idx_low.size > 0 and idx_high.size > 0:
        rise_time_s = float(t[idx_high[0]] - t[idx_low[0]])

    tol = 0.02 * max(abs(final_cmd), 1e-6)
    settling_time_s = np.nan
    for i in range(len(y)):
        if np.all(np.abs(y[i:] - final_resp) <= tol):
            settling_time_s = float(t[i])
            break

    return {
        "steady_state_error": steady_state_error,
        "overshoot_pct": overshoot_pct,
        "rise_time_s": rise_time_s,
        "settling_time_s": settling_time_s,
    }


def run_axis_test(axis: str, amplitude: float, duration_s: float, step_start_s: float, dt: float) -> dict:
    sim = VelocityInnerLoopSimulator(SimParams())
    num_steps = int(round(duration_s / dt)) + 1
    time_s = np.arange(num_steps, dtype=float) * dt

    cmd_hist = np.zeros((num_steps, 3), dtype=float)
    vel_hist = np.zeros((num_steps, 3), dtype=float)
    euler_hist = np.zeros((num_steps, 3), dtype=float)
    rpm_hist = np.zeros((num_steps, 4), dtype=float)
    omega_hist = np.zeros((num_steps, 3), dtype=float)

    for k, tk in enumerate(time_s):
        cmd = np.zeros(3, dtype=float) if tk < step_start_s else build_command(axis, amplitude)
        out = sim.step(cmd, yaw_rate_setpoint=0.0, dt=dt)
        cmd_hist[k, :] = cmd
        vel_hist[k, :] = out["vel_yaw_body"]
        euler_hist[k, :] = out["euler"]
        rpm_hist[k, :] = out["rpm"]
        omega_hist[k, :] = out["omega_body"]

    axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
    metrics = compute_step_metrics(time_s, cmd_hist[:, axis_idx], vel_hist[:, axis_idx], step_start_s)

    return {
        "axis": axis,
        "time_s": time_s,
        "cmd_hist": cmd_hist,
        "vel_hist": vel_hist,
        "euler_hist": euler_hist,
        "rpm_hist": rpm_hist,
        "omega_hist": omega_hist,
        "metrics": metrics,
    }


def save_csv(result: dict, output_dir: Path) -> None:
    axis = result["axis"]
    data = np.column_stack(
        [
            result["time_s"],
            result["cmd_hist"],
            result["vel_hist"],
            result["euler_hist"],
            result["omega_hist"],
            result["rpm_hist"],
        ]
    )
    header = (
        "time_s,"
        "cmd_vx,cmd_vy,cmd_vz,"
        "vel_vx,vel_vy,vel_vz,"
        "roll,pitch,yaw,"
        "p,q,r,"
        "rpm1,rpm2,rpm3,rpm4"
    )
    np.savetxt(output_dir / f"velocity_feedback_{axis}.csv", data, delimiter=",", header=header, comments="")


def plot_results(results: list[dict], output_dir: Path | None, show_plot: bool) -> None:
    if plt is None:
        print("matplotlib not available, skip plotting and only save CSV/console metrics.")
        return

    fig, axes = plt.subplots(len(results), 3, figsize=(14, 4 * len(results)), squeeze=False)

    for row, result in enumerate(results):
        axis = result["axis"]
        idx = {"x": 0, "y": 1, "z": 2}[axis]
        t = result["time_s"]

        ax0, ax1, ax2 = axes[row]
        ax0.plot(t, result["cmd_hist"][:, idx], "--", label=f"{axis}_cmd")
        ax0.plot(t, result["vel_hist"][:, idx], label=f"{axis}_actual")
        ax0.set_title(f"{axis.upper()} axis velocity step")
        ax0.set_xlabel("time [s]")
        ax0.set_ylabel("velocity [m/s]")
        ax0.grid(True)
        ax0.legend()

        ax1.plot(t, np.rad2deg(result["euler_hist"][:, 0]), label="roll")
        ax1.plot(t, np.rad2deg(result["euler_hist"][:, 1]), label="pitch")
        ax1.plot(t, np.rad2deg(result["euler_hist"][:, 2]), label="yaw")
        ax1.set_title(f"{axis.upper()} axis attitude response")
        ax1.set_xlabel("time [s]")
        ax1.set_ylabel("angle [deg]")
        ax1.grid(True)
        ax1.legend()

        ax2.plot(t, result["rpm_hist"][:, 0], label="rpm1")
        ax2.plot(t, result["rpm_hist"][:, 1], label="rpm2")
        ax2.plot(t, result["rpm_hist"][:, 2], label="rpm3")
        ax2.plot(t, result["rpm_hist"][:, 3], label="rpm4")
        ax2.set_title(f"{axis.upper()} axis motor response")
        ax2.set_xlabel("time [s]")
        ax2.set_ylabel("rpm")
        ax2.grid(True)
        ax2.legend()

    fig.tight_layout()
    if output_dir is not None:
        fig.savefig(output_dir / "velocity_feedback_summary.png", dpi=160, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone Tello inner-loop velocity feedback test.")
    parser.add_argument("--axis", choices=["x", "y", "z", "all"], default="all", help="Axis to test.")
    parser.add_argument("--duration", type=float, default=4.0, help="Simulation duration [s].")
    parser.add_argument("--step-start", type=float, default=0.5, help="Step start time [s].")
    parser.add_argument("--dt", type=float, default=1.0 / 1000.0, help="Simulation timestep [s].")
    parser.add_argument("--amp-x", type=float, default=1.0, help="Step amplitude for x test [m/s].")
    parser.add_argument("--amp-y", type=float, default=1.0, help="Step amplitude for y test [m/s].")
    parser.add_argument("--amp-z", type=float, default=0.6, help="Step amplitude for z test [m/s].")
    parser.add_argument("--output-dir", type=Path, default=Path("velocity_feedback_outputs"), help="Directory for CSV/figure outputs.")
    parser.add_argument("--no-show", action="store_true", help="Do not open the matplotlib window.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    amplitudes = {"x": args.amp_x, "y": args.amp_y, "z": args.amp_z}
    axes = ["x", "y", "z"] if args.axis == "all" else [args.axis]

    results = []
    for axis in axes:
        result = run_axis_test(axis, amplitudes[axis], args.duration, args.step_start, args.dt)
        results.append(result)
        save_csv(result, output_dir)
        m = result["metrics"]
        print(
            f"[{axis}] rise={m['rise_time_s']:.4f}s, "
            f"settle={m['settling_time_s']:.4f}s, "
            f"overshoot={m['overshoot_pct']:.2f}%, "
            f"steady_state_error={m['steady_state_error']:.4f} m/s"
        )

    plot_results(results, output_dir, show_plot=not args.no_show)
    print(f"Saved outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
