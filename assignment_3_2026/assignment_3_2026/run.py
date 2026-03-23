import pybullet as p
import time
import csv
import pybullet_data
import numpy as np
from src.tello_controller import TelloController
import importlib
import controller
from src.wind import Wind
import matplotlib.pyplot as plt


class Simulator:
    def __init__(self):
        p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        self.plane_id = p.loadURDF("plane.urdf")
        self.start_pos = [0, 0, 1]
        self.start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.drone_id = p.loadURDF(
            "resources/tello.urdf", self.start_pos, self.start_orientation
        )
        self.wind_enabled = True # wind enabled by default
        self.wind_sim = Wind(max_steady_state=0.02, max_gust=0.02,k_gusts=0.1)

        self.M = 0.088
        self.L = 0.06
        self.IR = 4.95e-5
        self.KF = 0.566e-5
        self.KM = 0.762e-7
        self.K_TRANS = np.array([3.365e-2, 3.365e-2, 3.365e-2])
        self.K_ROT = np.array([4.609e-3, 4.609e-3, 4.609e-3])
        self.TM = 0.0163
        self.tello_controller = TelloController(
            9.81, self.M, self.L, 0.35, self.KF, self.KM
        )

        self.targets = self.load_targets()
        self.current_target = 0

        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 0, 0, 1]
        )
        self.marker_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=self.targets[self.current_target][0:3],
        )
        print(f"INFO: Target set to: {self.targets[self.current_target]}")

        self.init_plot()

    def init_plot(self):
        plt.ion()
        self.fig = plt.figure(figsize=(4, 4))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        self.ax.set_xlim([-1, 1])
        self.ax.set_ylim([-1, 1])
        self.ax.set_zlim([-1, 1])
        self.ax.grid(False)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title("Wind Speed and Direction")
        
        self.quiver = self.ax.quiver(0, 0, 0, 0, 0, 0, length=0, color='b')

    def update_plot(self, wind_vector):
        if self.quiver:
            self.quiver.remove()
        
        scale = 30.0 
        u, v, w = wind_vector * scale
        magnitude = np.linalg.norm([u, v, w])
        
        self.quiver = self.ax.quiver(0, 0, 0, u, v, w, length=magnitude, color='c', normalize=False)
        
        limit = max(abs(u), abs(v), abs(w)) + 0.2
        
        limit = max(limit, 0.5) 

        self.ax.set_xlim([-limit, limit])
        self.ax.set_ylim([-limit, limit])
        self.ax.set_zlim([-limit, limit])
        
        cam_data = p.getDebugVisualizerCamera()
        if cam_data:
            pb_yaw = cam_data[8]
            pb_pitch = cam_data[9]
            new_elev = -pb_pitch
            new_azim = pb_yaw - 90
            self.ax.view_init(elev=new_elev, azim=new_azim)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def load_targets(self):
        targets = []
        try:
            with open("targets.csv", "r") as file:
                csvreader = csv.reader(file)
                header = next(csvreader)
                for row in csvreader:
                    if len(row) != 4: continue
                    if float(row[2]) < 0: continue
                    targets.append(
                        (float(row[0]), float(row[1]), float(row[2]), float(row[3]))
                    )
        except FileNotFoundError:
            pass
        if not targets:
            targets.append((0.0, 0.0, 0.0, 0.0))
        return targets

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
        x_torque = (-motor_forces[0] + motor_forces[1] + motor_forces[2] - motor_forces[3]) * self.L
        y_torque = (-motor_forces[0] + motor_forces[1] - motor_forces[2] + motor_forces[3]) * self.L
        torques = np.array([x_torque, y_torque, z_torque])
        return force, torques

    def display_target(self):
        p.resetBasePositionAndOrientation(
            self.marker_id,
            self.targets[self.current_target][0:3],
            self.start_orientation,
        )
        print(f"INFO: Target set to: {self.targets[self.current_target]}")

    def check_action(self, unchecked_action):
        if isinstance(unchecked_action, (tuple, list)):
            if len(unchecked_action) not in [4, 5]:
                checked_action = (0, 0, 0, 0)
                p.disconnect()
            else:
                checked_action = [
                    np.clip(unchecked_action[0], -1, 1),
                    np.clip(unchecked_action[1], -1, 1),
                    np.clip(unchecked_action[2], -1, 1),
                    np.clip(unchecked_action[3], -1.74533, 1.74533),
                ]
                if len(unchecked_action) == 5:
                    checked_action.append(unchecked_action[4])
        else:
            checked_action = (0, 0, 0, 0)
            p.disconnect()
        return tuple(checked_action)

    def spin_motors(self, rpm, timestep):
        for joint_index in range(4):
            rad_s = rpm[joint_index] * (2.0 * np.pi / 60.0)
            current_angle = p.getJointState(self.drone_id, joint_index)[0]
            new_angle = current_angle + rad_s * timestep
            p.resetJointState(self.drone_id, joint_index, new_angle)

    def motor_model(self, desired_rpm, current_rpm, dt):
        rpm_derivative = (desired_rpm - current_rpm) / self.TM
        real_rpm = current_rpm + rpm_derivative * dt
        return real_rpm

    def reload_controller(self):
        try:
            importlib.reload(controller)
        except Exception:
            print("ERROR: Failed to reload controller module")


if __name__ == "__main__":

    sim = Simulator()
    timestep = 1.0 / 1000  # 1000 Hz
    pos_control_timestep = 1.0 / 50  # 20 Hz
    steps_between_pos_control = int(round(pos_control_timestep / timestep))
    loop_counter = 0

    prev_rpm = np.array([0, 0, 0, 0])
    desired_vel = np.array([0, 0, 0])
    yaw_rate_setpoint = 0
    
    current_wind_display = np.array([0.0, 0.0, 0.0])

    while True:
        loop_start = time.time()
        loop_counter += 1

        pos, quat = p.getBasePositionAndOrientation(sim.drone_id)
        lin_vel_world, ang_vel_world = p.getBaseVelocity(sim.drone_id)

        roll, pitch, yaw = p.getEulerFromQuaternion(quat)
        yaw_quat = p.getQuaternionFromEuler([0, 0, yaw])
        inverted_pos, inverted_quat = p.invertTransform([0, 0, 0], quat)
        inverted_pos_yaw, inverted_quat_yaw = p.invertTransform([0, 0, 0], yaw_quat)
        
        lin_vel = p.rotateVector(inverted_quat_yaw, lin_vel_world)
        ang_vel = p.rotateVector(inverted_quat, ang_vel_world)
        lin_vel = np.array(lin_vel)
        ang_vel = np.array(ang_vel)
        
        if loop_counter >= steps_between_pos_control:
            loop_counter = 0
            
            if sim.wind_enabled:
                wind_mag = np.linalg.norm(current_wind_display)

            state = np.concatenate((pos, p.getEulerFromQuaternion(quat)))
            controller_output = sim.check_action(
                controller.controller(
                    state, sim.targets[sim.current_target], pos_control_timestep, sim.wind_enabled
                )
            )
            desired_vel = np.array(controller_output[:3])
            yaw_rate_setpoint = controller_output[3]
            
            sim.update_plot(current_wind_display)

        rpm = sim.tello_controller.compute_control(
            desired_vel, lin_vel, quat, ang_vel, yaw_rate_setpoint, timestep
        )
        rpm = sim.motor_model(rpm, prev_rpm, timestep)
        prev_rpm = rpm
        force, torque = sim.compute_dynamics(rpm, lin_vel_world, quat)

        p.applyExternalForce(sim.drone_id, -1, force, [0, 0, 0], p.LINK_FRAME)
        p.applyExternalTorque(sim.drone_id, -1, torque, p.LINK_FRAME)

        current_wind_display = np.array([0.0, 0.0, 0.0])
        if sim.wind_enabled:
            current_wind_display = sim.wind_sim.get_wind(timestep)
            p.applyExternalForce(sim.drone_id, -1, current_wind_display, pos, p.WORLD_FRAME)

        sim.spin_motors(rpm, timestep)

        keys = p.getKeyboardEvents()
        if ord("k") in keys and keys[ord("k")] & p.KEY_WAS_TRIGGERED:
            sim.wind_enabled = not sim.wind_enabled
            if sim.wind_enabled:
                sim.wind_sim = Wind(max_steady_state=0.02, max_gust=0.02, k_gusts=0.1)
                print(f"INFO: Wind disturbance ENABLED.")
            else:
                print(f"INFO: Wind disturbance DISABLED.")

        if ord("r") in keys and keys[ord("r")] & p.KEY_WAS_TRIGGERED:
            p.resetBasePositionAndOrientation(sim.drone_id, sim.start_pos, sim.start_orientation)
            sim.prev_rpm = np.array([0, 0, 0, 0])
            sim.tello_controller.reset()
            sim.reload_controller()
            sim.targets = sim.load_targets()
            sim.current_target = 0
            sim.display_target()

        if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_WAS_TRIGGERED:
            sim.current_target = (sim.current_target + 1) % len(sim.targets)
            sim.tello_controller.reset()
            sim.display_target()
        
        if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_WAS_TRIGGERED:
            sim.current_target = (sim.current_target - 1) % len(sim.targets)
            sim.tello_controller.reset()
            sim.display_target()

        if ord("q") in keys and keys[ord("q")] & p.KEY_WAS_TRIGGERED:
            p.disconnect()
            plt.close()
            break

        p.stepSimulation()
        loop_time = time.time() - loop_start
        if loop_time < timestep:
            time.sleep(timestep - loop_time)