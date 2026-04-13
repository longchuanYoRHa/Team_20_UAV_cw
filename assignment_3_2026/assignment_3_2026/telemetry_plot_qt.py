import sys
import multiprocessing as mp
from collections import deque

from PySide6.QtWidgets import QApplication, QFrame, QGridLayout, QLabel, QWidget, QVBoxLayout
from PySide6.QtCore import QTimer

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class RollingBuffer:
    def __init__(self, maxlen=200):
        self.data = deque(maxlen=maxlen)

    def append(self, value):
        self.data.append(value)

    def get(self):
        return list(self.data)


class PlotWidget(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(6, 9))
        super().__init__(self.fig)

        self.ax1 = self.fig.add_subplot(4, 1, 1)
        self.ax2 = self.fig.add_subplot(4, 1, 2)
        self.ax3 = self.fig.add_subplot(4, 1, 3)
        self.ax4 = self.fig.add_subplot(4, 1, 4)
        self.ax4_twin = self.ax4.twinx()

        self.fig.tight_layout()

        # buffers
        self.err_buf = RollingBuffer()
        self.vx_buf = RollingBuffer()
        self.vy_buf = RollingBuffer()
        self.vz_buf = RollingBuffer()

        self.velx_buf = RollingBuffer()
        self.vely_buf = RollingBuffer()
        self.velz_buf = RollingBuffer()
        self.velx_est_buf = RollingBuffer()
        self.vely_est_buf = RollingBuffer()
        self.velz_est_buf = RollingBuffer()

        self.wind_wx_buf = RollingBuffer()
        self.wind_wy_buf = RollingBuffer()
        self.dob_wx_buf = RollingBuffer()
        self.dob_wy_buf = RollingBuffer()

    def update_data(self, packet):
        self.err_buf.append(packet["target_distance"])

        vx, vy, vz, _ = packet["control_cmd"]
        self.vx_buf.append(vx)
        self.vy_buf.append(vy)
        self.vz_buf.append(vz)

        ax, ay, az = packet["actual_vel"]
        self.velx_buf.append(ax)
        self.vely_buf.append(ay)
        self.velz_buf.append(az)
        ex, ey, ez = packet.get("vel_est_body", (0.0, 0.0, 0.0))
        self.velx_est_buf.append(ex)
        self.vely_est_buf.append(ey)
        self.velz_est_buf.append(ez)

        wx, wy = packet.get("wind_xy", (0.0, 0.0))
        self.wind_wx_buf.append(wx)
        self.wind_wy_buf.append(wy)
        dob_wx, dob_wy = packet.get("dob_comp_xy", (0.0, 0.0))
        self.dob_wx_buf.append(dob_wx)
        self.dob_wy_buf.append(dob_wy)

    def redraw(self):
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        self.ax4_twin.clear()

        # error
        self.ax1.plot(self.err_buf.get())
        self.ax1.set_title("Distance to Target")
        self.ax1.set_xlabel("sample @ 50 Hz")
        self.ax1.set_ylabel("distance (m)")

        # control
        self.ax2.plot(self.vx_buf.get(), label="vx_cmd")
        self.ax2.plot(self.vy_buf.get(), label="vy_cmd")
        self.ax2.plot(self.vz_buf.get(), label="vz_cmd")
        self.ax2.legend()
        self.ax2.set_title("Control")
        self.ax2.set_xlabel("sample @ 50 Hz")
        self.ax2.set_ylabel("velocity setpoint (m/s)")

        # velocity
        self.ax3.plot(self.velx_buf.get(), label="vx actual", color="C0")
        self.ax3.plot(self.vely_buf.get(), label="vy actual", color="C1")
        self.ax3.plot(self.velz_buf.get(), label="vz actual", color="C2")
        self.ax3.plot(self.velx_est_buf.get(), label="vx est", color="C0", linestyle="--")
        self.ax3.plot(self.vely_est_buf.get(), label="vy est", color="C1", linestyle="--")
        self.ax3.plot(self.velz_est_buf.get(), label="vz est", color="C2", linestyle="--")
        self.ax3.legend()
        self.ax3.set_title("Actual vs Estimated Velocity")
        self.ax3.set_xlabel("sample @ 50 Hz")
        self.ax3.set_ylabel("velocity (m/s)")

        # wind (world frame, same as PyBullet WORLD_FRAME force)
        (w1,) = self.ax4.plot(self.wind_wx_buf.get(), label="wind Wx", color="C0")
        (w2,) = self.ax4.plot(self.wind_wy_buf.get(), label="wind Wy", color="C1")
        (d1,) = self.ax4_twin.plot(self.dob_wx_buf.get(), linestyle="--", label="dob comp Wx", color="C2")
        (d2,) = self.ax4_twin.plot(self.dob_wy_buf.get(), linestyle="--", label="dob comp Wy", color="C3")
        self.ax4.set_title("Wind force vs DOB compensation (world X/Y)")
        self.ax4.set_xlabel("sample @ 50 Hz")
        self.ax4.set_ylabel("wind force (N)")
        self.ax4_twin.set_ylabel("dob compensation (m/s)")
        self.ax4.legend(handles=[w1, w2, d1, d2], loc="upper left", fontsize=8)

        self.draw()


class TelemetryWindow(QWidget):
    def __init__(self, queue):
        super().__init__()
        self.queue = queue

        self.setWindowTitle("Telemetry Plot")
        self.resize(820, 900)

        layout = QVBoxLayout()
        self.plot = PlotWidget()
        layout.addWidget(self.plot)

        self.vx_panel = QFrame()
        self.vx_panel.setFrameShape(QFrame.Box)
        self.vx_panel.setLineWidth(1)
        vx_layout = QGridLayout(self.vx_panel)
        vx_layout.addWidget(QLabel("X轴速度实时值"), 0, 0, 1, 2)
        vx_layout.addWidget(QLabel("实际速度 vx"), 1, 0)
        vx_layout.addWidget(QLabel("估计速度 vx"), 2, 0)
        self.vx_actual_label = QLabel("-")
        self.vx_est_label = QLabel("-")
        self.vx_actual_label.setTextInteractionFlags(self.vx_actual_label.textInteractionFlags())
        self.vx_est_label.setTextInteractionFlags(self.vx_est_label.textInteractionFlags())
        vx_layout.addWidget(self.vx_actual_label, 1, 1)
        vx_layout.addWidget(self.vx_est_label, 2, 1)
        layout.addWidget(self.vx_panel)

        self.setLayout(layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(50)

    def update(self):
        while not self.queue.empty():
            packet = self.queue.get()
            self.plot.update_data(packet)
            actual_vel = packet.get("actual_vel", (0.0, 0.0, 0.0))
            vel_est = packet.get("vel_est_body", (0.0, 0.0, 0.0))
            self.vx_actual_label.setText(f"{float(actual_vel[0]):+.3f} m/s")
            self.vx_est_label.setText(f"{float(vel_est[0]):+.3f} m/s")

        self.plot.redraw()


def telemetry_process(queue):
    # Fresh interpreter on Linux; avoids fork inheriting a stale QApplication from the parent
    # (PyBullet + matplotlib in the main process often trigger this libshiboken error with fork).
    app = QApplication(sys.argv)
    window = TelemetryWindow(queue)
    window.show()
    app.exec()


class TelemetryPlot:
    def __init__(self):
        # Queue and Process must use the same context; "spawn" prevents Qt singleton issues after fork.
        self._ctx = mp.get_context("spawn")
        self.queue = self._ctx.Queue()
        self.proc = self._ctx.Process(
            target=telemetry_process, args=(self.queue,), daemon=True
        )

    def start(self):
        self.proc.start()

    def update(self, packet):
        if not self.queue.full():
            self.queue.put(packet)

    def close(self):
        self.proc.terminate()