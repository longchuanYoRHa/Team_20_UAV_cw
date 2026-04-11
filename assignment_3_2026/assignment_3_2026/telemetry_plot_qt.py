import sys
import multiprocessing as mp
from collections import deque

from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout
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
        self.fig = Figure(figsize=(6, 11))
        super().__init__(self.fig)

        self.ax1 = self.fig.add_subplot(5, 1, 1)
        self.ax2 = self.fig.add_subplot(5, 1, 2)
        self.ax3 = self.fig.add_subplot(5, 1, 3)
        self.ax4 = self.fig.add_subplot(5, 1, 4)
        self.ax5 = self.fig.add_subplot(5, 1, 5)
        self.ax5_twin = self.ax5.twinx()

        self.fig.tight_layout()

        # buffers
        self.err_buf = RollingBuffer()
        self.vx_buf = RollingBuffer()
        self.vy_buf = RollingBuffer()
        self.vz_buf = RollingBuffer()

        self.velx_buf = RollingBuffer()
        self.vely_buf = RollingBuffer()
        self.velz_buf = RollingBuffer()

        self.wind_wx_buf = RollingBuffer()
        self.wind_wy_buf = RollingBuffer()

        self.pid_ix_buf = RollingBuffer()
        self.pid_iy_buf = RollingBuffer()
        self.pid_iz_buf = RollingBuffer()
        self.yaw_i_term_buf = RollingBuffer()

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

        wx, wy = packet.get("wind_xy", (0.0, 0.0))
        self.wind_wx_buf.append(wx)
        self.wind_wy_buf.append(wy)

        ix, iy, iz = packet.get("pid_i_body", (0.0, 0.0, 0.0))
        self.pid_ix_buf.append(ix)
        self.pid_iy_buf.append(iy)
        self.pid_iz_buf.append(iz)
        self.yaw_i_term_buf.append(float(packet.get("yaw_i_term", 0.0)))

    def redraw(self):
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        self.ax5.clear()
        self.ax5_twin.clear()

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
        self.ax3.plot(self.velx_buf.get(), label="vx")
        self.ax3.plot(self.vely_buf.get(), label="vy")
        self.ax3.plot(self.velz_buf.get(), label="vz")
        self.ax3.legend()
        self.ax3.set_title("Actual Velocity")
        self.ax3.set_xlabel("sample @ 50 Hz")
        self.ax3.set_ylabel("velocity (m/s)")

        # wind (world frame, same as PyBullet WORLD_FRAME force)
        self.ax4.plot(self.wind_wx_buf.get(), label="wind Wx")
        self.ax4.plot(self.wind_wy_buf.get(), label="wind Wy")
        self.ax4.legend()
        self.ax4.set_title("Wind force (world X/Y)")
        self.ax4.set_xlabel("sample @ 50 Hz")
        self.ax4.set_ylabel("force (N)")

        # outer-loop PID integral → velocity / yaw-rate command (same step as vx_cmd plot)
        (l1,) = self.ax5.plot(self.pid_ix_buf.get(), color="C0", label="I→vx (body)")
        (l2,) = self.ax5.plot(self.pid_iy_buf.get(), color="C1", label="I→vy (body)")
        (l3,) = self.ax5.plot(self.pid_iz_buf.get(), color="C2", label="I→vz (body)")
        (l4,) = self.ax5_twin.plot(
            self.yaw_i_term_buf.get(), color="C3", linestyle="--", label="I→yaw rate"
        )
        self.ax5.set_title("PID integral contribution (overshoot diagnostic)")
        self.ax5.set_xlabel("sample @ 50 Hz")
        self.ax5.set_ylabel("position loop I term (m/s)")
        self.ax5_twin.set_ylabel("yaw loop I term (rad/s)")
        self.ax5.legend(handles=[l1, l2, l3, l4], loc="upper left", fontsize=8)

        self.draw()


class TelemetryWindow(QWidget):
    def __init__(self, queue):
        super().__init__()
        self.queue = queue

        self.setWindowTitle("Telemetry Plot")
        self.resize(800, 800)

        layout = QVBoxLayout()
        self.plot = PlotWidget()
        layout.addWidget(self.plot)
        self.setLayout(layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(50)

    def update(self):
        while not self.queue.empty():
            packet = self.queue.get()
            self.plot.update_data(packet)

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