import numpy as np
import pybullet as p
from src.PID_controller import PIDController


class TelloController:
    def __init__(self, g, mass, L, max_angle, KF, KM):
        self.g = g
        self.mass = mass
        self.L = L
        self.max_angle = max_angle
        self.KF = KF
        self.KM = KM

        self.vel_controller = PIDController(7, 0.6, 0.2, [10, 10, 10])
        self.attitude_controller = PIDController(0.2, 0.01, 0.015, [1, 1, 1])
        self.rate_controller = PIDController(0.05, 0.2, 0, [0.1, 0.1, 0.1])

        self.mixing_matrix = np.array(
            [
                [
                    0.25,
                    -1 / (4 * self.L),
                    -1 / (4 * self.L),
                    -KF / (4 * KM),
                ],  # Motor 1
                [
                    0.25,
                    1 / (4 * self.L),
                    1 / (4 * self.L),
                    -KF / (4 * KM),
                ],  # Motor 2
                [
                    0.25,
                    1 / (4 * self.L),
                    -1 / (4 * self.L),
                    KF / (4 * KM),
                ],  # Motor 3
                [
                    0.25,
                    -1 / (4 * self.L),
                    1 / (4 * self.L),
                    KF / (4 * KM),
                ],  # Motor 4
            ]
        )

    def reset(self):
        self.vel_controller.reset()
        self.attitude_controller.reset()
        self.rate_controller.reset()

    def quat_to_euler(self, quat):
        return np.array(p.getEulerFromQuaternion(quat))

    def velocity_control(self, velocity_setpoint, current_velocity, timestep):
        desired_acceleration = self.vel_controller.control_update(
            velocity_setpoint - current_velocity, timestep
        )
        return desired_acceleration

    def rate_control(self, rate_setpoint, current_rate, timestep):
        desired_torque = self.rate_controller.control_update(
            rate_setpoint - current_rate, timestep
        )
        return desired_torque

    def accel_to_thrust(self, acceleration_setpoint, quat):
        acceleration_setpoint[2] += self.g
        desired_thrust = acceleration_setpoint[2] * self.mass
        return acceleration_setpoint, desired_thrust

    def accel_to_angle(self, desired_accel, quat):
        euler = self.quat_to_euler(quat)
        norm_accel = np.linalg.norm(desired_accel[:2]) 

        if norm_accel < 1e-6:
            desired_angle = [0, 0, 0]
        else:
            desired_angle = [
                np.arctan2(-desired_accel[1], desired_accel[2]),
                np.arctan2(desired_accel[0], desired_accel[2]),
                0,
            ]

        desired_angle[0] = np.clip(desired_angle[0], -self.max_angle, self.max_angle)
        desired_angle[1] = np.clip(desired_angle[1], -self.max_angle, self.max_angle)
        desired_angle[2] = euler[2]

        return desired_angle

    def compute_control(
        self, desired_vel, lin_vel, quat, ang_vel, yaw_rate_setpoint, timestep
    ):
        desired_accel = self.velocity_control(desired_vel, lin_vel, timestep)

        desired_accel, thrust = self.accel_to_thrust(desired_accel, quat)

        desired_angle = self.accel_to_angle(desired_accel, quat)
        current_angle = self.quat_to_euler(quat)

        desired_rate = self.attitude_controller.control_update(
            desired_angle - current_angle, timestep
        )
        desired_rate[2] = yaw_rate_setpoint  

        desired_torque = self.rate_control(desired_rate, ang_vel, timestep)

        rpms = self.mix_controls(thrust, desired_torque)
        return rpms

    def mix_controls(self, thrust, torques):
        if not isinstance(torques, np.ndarray):
            raise ValueError("Torques must be a numpy array")
        if torques.shape != (3,):
            raise ValueError("Torques must be a 3-element numpy array")
        controls = np.append([thrust], torques)

        motor_commands = self.mixing_matrix @ controls

        motor_commands = np.clip(motor_commands, 0, 0.75202525252)

        rpms = np.sqrt(motor_commands / self.KF) * (60 / (2 * np.pi))
        rpms = np.clip(rpms, 0, 28000)
        return rpms
