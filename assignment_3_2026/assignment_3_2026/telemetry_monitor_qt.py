from __future__ import annotations

import multiprocessing as mp
import queue
from dataclasses import dataclass
from typing import Any, Dict, Optional

# GUI dependencies are intentionally imported lazily inside the GUI process.
# This lets the main simulator keep running even if Qt is not installed yet.


@dataclass
class TelemetryPacket:
    target_error_world: tuple[float, float, float]
    target_distance: float
    control_cmd: tuple[float, float, float, float]   # vx, vy, vz, yaw_rate
    actual_vel_body: tuple[float, float, float]      # current lin_vel in yaw-aligned body frame
    actual_ang_vel_body: tuple[float, float, float]  # optional, but useful
    target: tuple[float, float, float, float]
    position: tuple[float, float, float]
    yaw: float
    wind_enabled: bool
    wind_xy: Optional[tuple[float, float]] = None
    dob_comp_xy: Optional[tuple[float, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_error_world": self.target_error_world,
            "target_distance": self.target_distance,
            "control_cmd": self.control_cmd,
            "actual_vel_body": self.actual_vel_body,
            "actual_ang_vel_body": self.actual_ang_vel_body,
            "target": self.target,
            "position": self.position,
            "yaw": self.yaw,
            "wind_enabled": self.wind_enabled,
            "wind_xy": self.wind_xy,
            "dob_comp_xy": self.dob_comp_xy,
        }


class TelemetryMonitor:
    """
    Lightweight Qt telemetry monitor running in a separate process.

    Why process instead of thread:
    - PyBullet GUI already occupies the simulator main thread.
    - Qt normally wants its own event loop in the main thread of the process.
    - This design avoids Qt / PyBullet GUI conflicts and only requires a few lines
      of integration in run.py.
    """

    def __init__(self, title: str = "Tello Telemetry Monitor") -> None:
        self.title = title
        self._ctx = mp.get_context("spawn")
        self._queue: mp.Queue = self._ctx.Queue(maxsize=8)
        self._proc: Optional[mp.Process] = None

    def start(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            return
        self._proc = self._ctx.Process(
            target=_gui_process_main,
            args=(self._queue, self.title),
            daemon=True,
        )
        self._proc.start()

    def update(self, packet: TelemetryPacket | Dict[str, Any]) -> None:
        if self._proc is None or not self._proc.is_alive():
            return
        data = packet.to_dict() if isinstance(packet, TelemetryPacket) else packet
        try:
            self._queue.put_nowait(data)
        except queue.Full:
            # Keep the most recent telemetry only.
            try:
                _ = self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(data)
            except queue.Full:
                pass

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            self._queue.put_nowait({"_close": True})
        except Exception:
            pass
        self._proc.join(timeout=1.5)
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=1.0)
        self._proc = None


# ---------------- GUI process ---------------- #

def _gui_process_main(shared_queue: mp.Queue, title: str) -> None:
    try:
        from PySide6 import QtCore, QtWidgets
    except Exception as e:
        print(
            "[TelemetryMonitor] Qt dependency missing. Install one of these on your simulator environment:\n"
            "  pip install PySide6\n"
            f"Import error: {e}"
        )
        return

    class QueueReader(QtCore.QThread):
        packet_received = QtCore.Signal(dict)
        close_requested = QtCore.Signal()

        def __init__(self, q: mp.Queue) -> None:
            super().__init__()
            self.q = q
            self._running = True

        def run(self) -> None:
            while self._running:
                try:
                    item = self.q.get(timeout=0.2)
                except queue.Empty:
                    continue
                except Exception:
                    continue
                if isinstance(item, dict) and item.get("_close"):
                    self.close_requested.emit()
                    return
                if isinstance(item, dict):
                    self.packet_received.emit(item)

        def stop(self) -> None:
            self._running = False

    class MonitorWindow(QtWidgets.QWidget):
        def __init__(self, window_title: str) -> None:
            super().__init__()
            self.setWindowTitle(window_title)
            self.resize(520, 320)

            root = QtWidgets.QVBoxLayout(self)

            header = QtWidgets.QLabel("Lightweight Flight Telemetry")
            header.setStyleSheet("font-size: 18px; font-weight: 600;")
            root.addWidget(header)

            self.status_label = QtWidgets.QLabel("Waiting for telemetry...")
            self.status_label.setStyleSheet("color: #666666;")
            root.addWidget(self.status_label)

            grid = QtWidgets.QGridLayout()
            grid.setHorizontalSpacing(14)
            grid.setVerticalSpacing(8)
            root.addLayout(grid)

            self.labels: Dict[str, QtWidgets.QLabel] = {}
            fields = [
                ("当前位置", "position"),
                ("目标点", "target"),
                ("目标差值 Δx Δy Δz", "target_error_world"),
                ("距离目标 |Δp|", "target_distance"),
                ("控制量 vx vy vz yaw_rate", "control_cmd"),
                ("实际速度 vx vy vz", "actual_vel_body"),
                ("实际角速度 wx wy wz", "actual_ang_vel_body"),
                ("风扰动 Wx Wy", "wind_xy"),
                ("DOB补偿 Wx Wy", "dob_comp_xy"),
                ("当前偏航 yaw", "yaw"),
            ]

            for row, (name, key) in enumerate(fields):
                l_name = QtWidgets.QLabel(name)
                l_name.setStyleSheet("font-weight: 500;")
                l_value = QtWidgets.QLabel("-")
                l_value.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
                grid.addWidget(l_name, row, 0)
                grid.addWidget(l_value, row, 1)
                self.labels[key] = l_value

            root.addStretch(1)

        @QtCore.Slot(dict)
        def on_packet(self, data: Dict[str, Any]) -> None:
            wind_text = "ON" if data.get("wind_enabled") else "OFF"
            self.status_label.setText(f"Telemetry live   |   Wind: {wind_text}")

            self.labels["position"].setText(_fmt_vec(data.get("position"), 3))
            self.labels["target"].setText(_fmt_vec(data.get("target"), 4))
            self.labels["target_error_world"].setText(_fmt_vec(data.get("target_error_world"), 3))
            dist = data.get("target_distance")
            self.labels["target_distance"].setText(f"{dist:.3f} m" if dist is not None else "-")
            self.labels["control_cmd"].setText(_fmt_vec(data.get("control_cmd"), 4, suffixes=("m/s", "m/s", "m/s", "rad/s")))
            self.labels["actual_vel_body"].setText(_fmt_vec(data.get("actual_vel_body"), 3, suffixes=("m/s", "m/s", "m/s")))
            self.labels["actual_ang_vel_body"].setText(_fmt_vec(data.get("actual_ang_vel_body"), 3, suffixes=("rad/s", "rad/s", "rad/s")))
            self.labels["wind_xy"].setText(_fmt_vec(data.get("wind_xy"), 2, suffixes=("N", "N")))
            self.labels["dob_comp_xy"].setText(_fmt_vec(data.get("dob_comp_xy"), 2, suffixes=("m/s", "m/s")))
            yaw = data.get("yaw")
            self.labels["yaw"].setText(f"{yaw:.3f} rad" if yaw is not None else "-")

        def closeEvent(self, event) -> None:  # type: ignore[override]
            event.accept()

    def _fmt_vec(value: Any, length: int, suffixes: Optional[tuple[str, ...]] = None) -> str:
        if value is None:
            return "-"
        try:
            items = list(value)[:length]
        except Exception:
            return str(value)
        text = []
        for i, v in enumerate(items):
            suffix = ""
            if suffixes and i < len(suffixes):
                suffix = f" {suffixes[i]}"
            text.append(f"{v:+.3f}{suffix}")
        return "   ".join(text)

    app = QtWidgets.QApplication([])
    window = MonitorWindow(title)
    reader = QueueReader(shared_queue)
    reader.packet_received.connect(window.on_packet)
    reader.close_requested.connect(app.quit)
    window.show()
    reader.start()
    app.exec()
    reader.stop()
    reader.wait(500)


__all__ = ["TelemetryPacket", "TelemetryMonitor"]
