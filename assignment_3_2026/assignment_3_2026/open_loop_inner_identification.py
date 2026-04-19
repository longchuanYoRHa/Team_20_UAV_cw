import os
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
import pybullet_data

from src.tello_controller import TelloController


class InnerLoopPlantSimulator:
    """Run the same inner-loop + rigid-body dynamics as run.py, but bypass controller.py.

    We inject desired body-frame velocity commands directly into TelloController.compute_control()
    and record the actual body-frame velocity / yaw-rate response.
    """

    def __init__(self, gui: bool = False):
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        self.plane_id = p.loadURDF("plane.urdf")
        self.start_pos = [0.0, 0.0, 1.0]
        self.start_orientation = p.getQuaternionFromEuler([0.0, 0.0, 0.0])
        self.drone_id = p.loadURDF("resources/tello.urdf", self.start_pos, self.start_orientation)

        # Same parameters as run.py
        self.M = 0.088
        self.L = 0.06
        self.IR = 4.95e-5
        self.KF = 0.566e-5
        self.KM = 0.762e-7
        self.K_TRANS = np.array([3.365e-2, 3.365e-2, 3.365e-2])
        self.K_ROT = np.array([4.609e-3, 4.609e-3, 4.609e-3])
        self.TM = 0.0163
        self.tello_controller = TelloController(9.81, self.M, self.L, 0.35, self.KF, self.KM)
        self.prev_rpm = np.zeros(4)

    def reset(self):
        p.resetBasePositionAndOrientation(self.drone_id, self.start_pos, self.start_orientation)
        p.resetBaseVelocity(self.drone_id, [0, 0, 0], [0, 0, 0])
        self.tello_controller.reset()
        self.prev_rpm = np.zeros(4)

    def close(self):
        if p.isConnected(self.client):
            p.disconnect(self.client)

    def motor_model(self, desired_rpm, current_rpm, dt):
        rpm_derivative = (desired_rpm - current_rpm) / self.TM
        real_rpm = current_rpm + rpm_derivative * dt
        return real_rpm

    def compute_dynamics(self, rpm_values, lin_vel_world, quat):
        rotation = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        omega = rpm_values * (2 * np.pi / 60)
        omega_squared = omega ** 2
        motor_forces = omega_squared * self.KF
        thrust = np.array([0.0, 0.0, np.sum(motor_forces)])
        vel_body = np.dot(rotation.T, lin_vel_world)
        drag_body = -self.K_TRANS * vel_body
        force = drag_body + thrust
        z_torques = omega_squared * self.KM
        z_torque = -z_torques[0] - z_torques[1] + z_torques[2] + z_torques[3]
        x_torque = (-motor_forces[0] + motor_forces[1] + motor_forces[2] - motor_forces[3]) * self.L
        y_torque = (-motor_forces[0] + motor_forces[1] - motor_forces[2] + motor_forces[3]) * self.L
        torques = np.array([x_torque, y_torque, z_torque])
        return force, torques

    def step(self, desired_vel_body, yaw_rate_setpoint, dt):
        pos, quat = p.getBasePositionAndOrientation(self.drone_id)
        lin_vel_world, ang_vel_world = p.getBaseVelocity(self.drone_id)

        roll, pitch, yaw = p.getEulerFromQuaternion(quat)
        yaw_quat = p.getQuaternionFromEuler([0, 0, yaw])
        _, inverted_quat = p.invertTransform([0, 0, 0], quat)
        _, inverted_quat_yaw = p.invertTransform([0, 0, 0], yaw_quat)

        lin_vel_body_yaw = np.array(p.rotateVector(inverted_quat_yaw, lin_vel_world), dtype=float)
        ang_vel_body = np.array(p.rotateVector(inverted_quat, ang_vel_world), dtype=float)

        rpm = self.tello_controller.compute_control(
            np.asarray(desired_vel_body, dtype=float),
            lin_vel_body_yaw,
            quat,
            ang_vel_body,
            float(yaw_rate_setpoint),
            dt,
        )
        rpm = self.motor_model(rpm, self.prev_rpm, dt)
        self.prev_rpm = rpm

        force, torque = self.compute_dynamics(rpm, lin_vel_world, quat)
        p.applyExternalForce(self.drone_id, -1, force, [0, 0, 0], p.LINK_FRAME)
        p.applyExternalTorque(self.drone_id, -1, torque, p.LINK_FRAME)
        p.stepSimulation()

        pos2, quat2 = p.getBasePositionAndOrientation(self.drone_id)
        lin_vel_world2, ang_vel_world2 = p.getBaseVelocity(self.drone_id)
        yaw2 = p.getEulerFromQuaternion(quat2)[2]
        yaw_quat2 = p.getQuaternionFromEuler([0, 0, yaw2])
        _, inverted_quat2 = p.invertTransform([0, 0, 0], quat2)
        _, inverted_quat_yaw2 = p.invertTransform([0, 0, 0], yaw_quat2)
        lin_vel_body_yaw2 = np.array(p.rotateVector(inverted_quat_yaw2, lin_vel_world2), dtype=float)
        ang_vel_body2 = np.array(p.rotateVector(inverted_quat2, ang_vel_world2), dtype=float)

        return {
            "pos": np.array(pos2, dtype=float),
            "quat": np.array(quat2, dtype=float),
            "yaw": float(yaw2),
            "lin_vel_world": np.array(lin_vel_world2, dtype=float),
            "lin_vel_body_yaw": lin_vel_body_yaw2,
            "ang_vel_body": ang_vel_body2,
            "rpm": rpm.copy(),
        }


def run_step_test(sim, axis="vx", amplitude=0.3, t_total=6.0, t_step=0.5, dt=1 / 1000):
    sim.reset()
    n = int(round(t_total / dt))
    time = np.arange(n) * dt

    desired = np.zeros((n, 4), dtype=float)  # vx vy vz yaw_rate
    idx = int(round(t_step / dt))

    if axis == "vx":
        desired[idx:, 0] = amplitude
        measured_index = ("lin_vel_body_yaw", 0)
    elif axis == "vy":
        desired[idx:, 1] = amplitude
        measured_index = ("lin_vel_body_yaw", 1)
    elif axis == "vz":
        desired[idx:, 2] = amplitude
        measured_index = ("lin_vel_body_yaw", 2)
    elif axis == "yaw_rate":
        desired[idx:, 3] = amplitude
        measured_index = ("ang_vel_body", 2)
    else:
        raise ValueError(f"Unsupported axis: {axis}")

    measured = np.zeros(n, dtype=float)
    cross1 = np.zeros(n, dtype=float)
    cross2 = np.zeros(n, dtype=float)
    yaw_hist = np.zeros(n, dtype=float)
    pos_hist = np.zeros((n, 3), dtype=float)
    rpm_hist = np.zeros((n, 4), dtype=float)

    for k in range(n):
        out = sim.step(desired[k, :3], desired[k, 3], dt)
        measured[k] = out[measured_index[0]][measured_index[1]]
        if axis == "vx":
            cross1[k] = out["lin_vel_body_yaw"][1]
            cross2[k] = out["lin_vel_body_yaw"][2]
        elif axis == "vy":
            cross1[k] = out["lin_vel_body_yaw"][0]
            cross2[k] = out["lin_vel_body_yaw"][2]
        elif axis == "vz":
            cross1[k] = out["lin_vel_body_yaw"][0]
            cross2[k] = out["lin_vel_body_yaw"][1]
        else:
            cross1[k] = out["lin_vel_body_yaw"][0]
            cross2[k] = out["lin_vel_body_yaw"][1]
        yaw_hist[k] = out["yaw"]
        pos_hist[k] = out["pos"]
        rpm_hist[k] = out["rpm"]

    return {
        "axis": axis,
        "time": time,
        "command": desired[:, {"vx": 0, "vy": 1, "vz": 2, "yaw_rate": 3}[axis]],
        "measured": measured,
        "cross1": cross1,
        "cross2": cross2,
        "yaw": yaw_hist,
        "pos": pos_hist,
        "rpm": rpm_hist,
        "t_step": t_step,
        "amplitude": amplitude,
    }


def estimate_first_order_from_step(time, u, y, t_step):
    mask = time >= t_step
    t = time[mask] - t_step
    u_post = u[mask]
    y_post = y[mask]

    u0 = float(np.mean(u_post[: max(10, min(100, len(u_post)//20))]))
    y0 = float(np.mean(y_post[: max(10, min(100, len(y_post)//20))]))
    yss = float(np.mean(y_post[-max(50, min(500, len(y_post)//10)):]))
    amp = float(np.max(u_post) - np.min(u_post))
    if amp < 1e-9:
        return {"K": 0.0, "tau": np.inf, "y_model": np.full_like(y, y0)}

    # For a pure step 0 -> A, gain K = delta_y / delta_u
    K = (yss - y0) / amp

    target = y0 + 0.632 * (yss - y0)
    if abs(yss - y0) < 1e-6:
        tau = np.inf
    else:
        idx = np.where((y_post - target) * np.sign(yss - y0) >= 0)[0]
        tau = t[idx[0]] if len(idx) else np.inf

    y_model = np.full_like(y, y0)
    if np.isfinite(tau) and tau > 1e-6:
        y_model[mask] = y0 + (yss - y0) * (1 - np.exp(-t / tau))
    else:
        y_model[mask] = yss

    return {"K": float(K), "tau": float(tau), "y_model": y_model}


def bode_from_first_order(K, tau, w):
    s = 1j * w
    G = K / (tau * s + 1.0)
    mag_db = 20 * np.log10(np.maximum(np.abs(G), 1e-12))
    phase_deg = np.unwrap(np.angle(G)) * 180 / np.pi
    return mag_db, phase_deg


def save_step_and_bode_plots(result, fit, out_dir):
    axis = result["axis"]
    time = result["time"]
    command = result["command"]
    measured = result["measured"]
    cross1 = result["cross1"]
    cross2 = result["cross2"]
    y_model = fit["y_model"]

    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    ax[0].plot(time, command, label="Command")
    ax[0].plot(time, measured, label="Measured")
    ax[0].plot(time, y_model, "--", label=f"1st-order fit: K={fit['K']:.3f}, tau={fit['tau']:.3f}s")
    ax[0].axvline(result["t_step"], color="k", linestyle=":", linewidth=1)
    ax[0].set_ylabel(axis)
    ax[0].set_title(f"Inner-loop step response ({axis})")
    ax[0].grid(True)
    ax[0].legend()

    ax[1].plot(time, cross1, label="Cross-axis 1")
    ax[1].plot(time, cross2, label="Cross-axis 2")
    ax[1].set_ylabel("Cross-axis response")
    ax[1].set_title("Coupling check")
    ax[1].grid(True)
    ax[1].legend()

    ax[2].plot(time, np.linalg.norm(result["rpm"], axis=1))
    ax[2].set_ylabel("||rpm||")
    ax[2].set_xlabel("Time [s]")
    ax[2].set_title("Motor activity")
    ax[2].grid(True)

    fig.tight_layout()
    step_path = os.path.join(out_dir, f"{axis}_step_response.png")
    fig.savefig(step_path, dpi=180)
    plt.close(fig)

    w = np.logspace(-1, 2, 400)
    mag_db, phase_deg = bode_from_first_order(fit["K"], max(fit["tau"], 1e-6), w)
    fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax[0].semilogx(w, mag_db)
    ax[0].set_ylabel("Magnitude [dB]")
    ax[0].set_title(f"Bode plot of identified first-order model ({axis})")
    ax[0].grid(True, which="both")

    ax[1].semilogx(w, phase_deg)
    ax[1].set_ylabel("Phase [deg]")
    ax[1].set_xlabel("Frequency [rad/s]")
    ax[1].grid(True, which="both")
    fig.tight_layout()
    bode_path = os.path.join(out_dir, f"{axis}_bode_identified_model.png")
    fig.savefig(bode_path, dpi=180)
    plt.close(fig)

    return step_path, bode_path


def save_csv(result, fit, out_dir):
    csv_path = os.path.join(out_dir, f"{result['axis']}_step_data.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "command", "measured", "model", "cross1", "cross2", "yaw"])
        for i in range(len(result["time"])):
            writer.writerow([
                float(result["time"][i]),
                float(result["command"][i]),
                float(result["measured"][i]),
                float(fit["y_model"][i]),
                float(result["cross1"][i]),
                float(result["cross2"][i]),
                float(result["yaw"][i]),
            ])
    return csv_path


def main():
    out_dir = os.path.abspath("inner_loop_analysis_output")
    os.makedirs(out_dir, exist_ok=True)

    sim = InnerLoopPlantSimulator(gui=False)
    try:
        configs = [
            ("vx", 0.30),
            ("vy", 0.30),
            ("vz", 0.20),
            ("yaw_rate", 0.60),
        ]

        summary_rows = []
        for axis, amplitude in configs:
            result = run_step_test(sim, axis=axis, amplitude=amplitude, t_total=6.0, t_step=0.5, dt=1/1000)
            fit = estimate_first_order_from_step(result["time"], result["command"], result["measured"], result["t_step"])
            step_path, bode_path = save_step_and_bode_plots(result, fit, out_dir)
            csv_path = save_csv(result, fit, out_dir)
            summary_rows.append((axis, amplitude, fit["K"], fit["tau"], step_path, bode_path, csv_path))
            print(f"[{axis}] estimated gain K={fit['K']:.4f}, tau={fit['tau']:.4f} s")

        summary_path = os.path.join(out_dir, "summary.csv")
        with open(summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["axis", "command_amplitude", "identified_gain_K", "identified_tau_s", "step_plot", "bode_plot", "csv_data"])
            writer.writerows(summary_rows)
        print(f"Saved analysis to: {out_dir}")
    finally:
        sim.close()


if __name__ == "__main__":
    main()
