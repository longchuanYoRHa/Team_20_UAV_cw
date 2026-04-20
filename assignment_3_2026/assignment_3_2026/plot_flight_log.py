"""Plot a CW3 flight log (controller_combined / controller_practical CSV).

Usage
-----
    python3 plot_flight_log.py \\
        --log   /path/to/flight_YYYYMMDD_HHMMSS.csv \\
        --targets /path/to/targets.csv \\
        --out   /path/to/output_dir

If --out is omitted, figures are written next to the log file in a folder
named <log_stem>_plots/.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)


class CsvTable:
    """Minimal column-oriented CSV wrapper used in place of pandas."""

    def __init__(self, columns: dict[str, np.ndarray]):
        self.columns = columns

    def __contains__(self, key: str) -> bool:
        return key in self.columns

    def __getitem__(self, key: str) -> np.ndarray:
        return self.columns[key]

    def __len__(self) -> int:
        if not self.columns:
            return 0
        return len(next(iter(self.columns.values())))

    def keys(self):
        return self.columns.keys()

    def has_all(self, names) -> bool:
        return all(n in self.columns for n in names)

    def iter_rows(self):
        keys = list(self.columns.keys())
        n = len(self)
        for i in range(n):
            yield {k: self.columns[k][i] for k in keys}


def _read_csv(path: Path) -> CsvTable:
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        rows = []
        for row in reader:
            if not row or all(not c.strip() for c in row):
                continue
            rows.append(row)
    cols: dict[str, np.ndarray] = {}
    for i, name in enumerate(header):
        vals = []
        for row in rows:
            v = row[i].strip() if i < len(row) else ""
            try:
                vals.append(float(v))
            except ValueError:
                vals.append(np.nan)
        cols[name] = np.asarray(vals, dtype=float)
    return CsvTable(cols)


def load_log(path: Path) -> CsvTable:
    tbl = _read_csv(path)
    t0 = tbl["timestamp_ms"][0]
    tbl.columns["t_s"] = (tbl["timestamp_ms"] - t0) / 1000.0
    return tbl


def load_targets(path: Path | None) -> CsvTable | None:
    if path is None or not path.exists():
        return None
    return _read_csv(path)


def _save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / name
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_xyz_tracking(df: CsvTable, out_dir: Path):
    t = df["t_s"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    for ax, actual, target, lbl in zip(
        axes,
        ["x", "y", "z"],
        ["x_d", "y_d", "z_d"],
        ["x [m]", "y [m]", "z [m]"],
    ):
        ax.plot(t, df[actual], "C0-", lw=1.4, label="actual")
        ax.plot(t, df[target], "C1--", lw=1.2, label="target")
        ax.set_ylabel(lbl)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time [s]")
    axes[0].legend(loc="upper right")
    fig.suptitle("Position tracking (actual vs. target)")
    return _save(fig, out_dir, "01_position_tracking.png")


def plot_yaw_tracking(df: CsvTable, out_dir: Path):
    t = df["t_s"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    axes[0].plot(t, np.rad2deg(df["yaw"]), "C0-", lw=1.4, label="yaw")
    axes[0].plot(t, np.rad2deg(df["yaw_d"]), "C1--", lw=1.2, label="yaw_d")
    axes[0].set_ylabel("yaw [deg]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(t, np.rad2deg(df["err_yaw"]), "C3-", lw=1.2, label="err_yaw")
    axes[1].axhline(0.0, color="k", lw=0.6, ls=":")
    # Yaw align threshold used in controller is 0.167 rad ~= 9.57 deg.
    axes[1].axhline(np.rad2deg(0.167), color="0.5", lw=0.6, ls="--", label="yaw_align_tol")
    axes[1].axhline(-np.rad2deg(0.167), color="0.5", lw=0.6, ls="--")
    axes[1].set_ylabel("err_yaw [deg]")
    axes[1].set_xlabel("time [s]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")
    fig.suptitle("Yaw tracking")
    return _save(fig, out_dir, "02_yaw_tracking.png")


def plot_position_error(df: CsvTable, out_dir: Path):
    t = df["t_s"]
    ex = df["err_x_w"]
    ey = df["err_y_w"]
    ez = df["err_z_w"]
    norm_xy = np.hypot(ex, ey)
    norm_xyz = np.sqrt(ex**2 + ey**2 + ez**2)

    fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
    axes[0].plot(t, ex, label="err_x")
    axes[0].plot(t, ey, label="err_y")
    axes[0].plot(t, ez, label="err_z")
    axes[0].axhline(0.0, color="k", lw=0.6, ls=":")
    axes[0].set_ylabel("per-axis err [m]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(t, norm_xy, label="||err_xy||")
    axes[1].plot(t, norm_xyz, label="||err_xyz||")
    axes[1].set_ylabel("error norm [m]")
    axes[1].set_xlabel("time [s]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")
    fig.suptitle("Position error (world frame)")
    return _save(fig, out_dir, "03_position_error.png")


def plot_commands(df: CsvTable, out_dir: Path):
    t = df["t_s"]
    fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(t, df["cmd_vx"], "C0", label="cmd_vx")
    axes[0].set_ylabel("vx [m/s]")
    axes[1].plot(t, df["cmd_vy"], "C2", label="cmd_vy")
    axes[1].set_ylabel("vy [m/s]")
    axes[2].plot(t, df["cmd_vz"], "C3", label="cmd_vz")
    axes[2].set_ylabel("vz [m/s]")
    axes[3].plot(t, df["cmd_yaw_rate"], "C4", label="cmd_yaw_rate")
    axes[3].set_ylabel("yaw rate [rad/s]")
    axes[3].set_xlabel("time [s]")

    # Mark the inner clamp (|v| <= 1.08) that the controller applies.
    for ax in axes[:3]:
        ax.axhline(1.08, color="0.5", lw=0.6, ls="--")
        ax.axhline(-1.08, color="0.5", lw=0.6, ls="--")
    axes[3].axhline(1.74533, color="0.5", lw=0.6, ls="--")
    axes[3].axhline(-1.74533, color="0.5", lw=0.6, ls="--")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
    fig.suptitle("Velocity / yaw-rate commands")
    return _save(fig, out_dir, "04_commands.png")


def plot_pid_breakdown(df: CsvTable, out_dir: Path):
    t = df["t_s"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axis_labels = ["x_b", "y_b", "z_b"]
    for ax, axis in zip(axes, axis_labels):
        ax.plot(t, df[f"p_{axis}"], label=f"P_{axis}")
        ax.plot(t, df[f"i_{axis}"], label=f"I_{axis}")
        ax.plot(t, df[f"d_{axis}"], label=f"D_{axis}")
        ax.plot(t, df[f"dob_{axis}"], lw=1.0, ls=":", label=f"DOB_{axis}")
        ax.set_ylabel(f"contribution ({axis})")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", ncol=4, fontsize=8)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("PID + DOB contributions in yaw-aligned body frame")
    return _save(fig, out_dir, "05_pid_breakdown.png")


def plot_velocity_estimates(df: CsvTable, out_dir: Path):
    t = df["t_s"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    for ax, axis, lbl in zip(
        axes,
        ["x", "y", "z"],
        ["vx [m/s]", "vy [m/s]", "vz [m/s]"],
    ):
        ax.plot(t, df[f"vel_est_{axis}_w"], label=f"vel_est_{axis}_w (world)")
        ax.plot(t, df[f"vel_est_{axis}_b"], label=f"vel_est_{axis}_b (body)")
        ax.plot(t, df[f"cmd_v{axis}"], label=f"cmd_v{axis}", alpha=0.6)
        ax.set_ylabel(lbl)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", ncol=3, fontsize=8)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Estimated velocity vs commanded velocity")
    return _save(fig, out_dir, "06_velocity_estimate.png")


def plot_3d_trajectory(df: CsvTable, targets: CsvTable | None, out_dir: Path):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    x = df["x"]
    y = df["y"]
    z = df["z"]
    ax.plot(x, y, z, "C0-", lw=1.4, label="actual trajectory")
    ax.scatter([x[0]], [y[0]], [z[0]], c="g", s=40, label="start")
    ax.scatter([x[-1]], [y[-1]], [z[-1]], c="r", s=40, label="end")

    # Target actually commanded to the controller over time.
    xd = df["x_d"]
    yd = df["y_d"]
    zd = df["z_d"]
    ax.plot(xd, yd, zd, "C1--", lw=1.0, alpha=0.7, label="commanded target")

    if targets is not None and targets.has_all(["target_x", "target_y", "target_z"]):
        tx = targets["target_x"]
        ty = targets["target_y"]
        tz = targets["target_z"]
        ax.scatter(tx, ty, tz, marker="*", s=120, c="m",
                   label="waypoints (targets.csv)")
        for i in range(len(targets)):
            ax.text(float(tx[i]), float(ty[i]), float(tz[i]) + 0.05,
                    f"WP{i}", fontsize=8, color="m")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("3D trajectory")
    ax.legend(loc="upper left", fontsize=8)
    return _save(fig, out_dir, "07_trajectory_3d.png")


def plot_xy_topdown(df: CsvTable, targets: CsvTable | None, out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 7))
    x = df["x"]
    y = df["y"]
    ax.plot(x, y, "C0-", lw=1.4, label="actual xy")
    ax.plot(df["x_d"], df["y_d"], "C1--", lw=1.0, label="target xy")
    ax.scatter(x[0], y[0], c="g", s=40, label="start")
    ax.scatter(x[-1], y[-1], c="r", s=40, label="end")
    if targets is not None and targets.has_all(["target_x", "target_y"]):
        tx = targets["target_x"]
        ty = targets["target_y"]
        ax.scatter(tx, ty, marker="*", s=150, c="m", label="waypoints")
        for i in range(len(targets)):
            ax.annotate(f"WP{i}", (float(tx[i]), float(ty[i])),
                        textcoords="offset points", xytext=(6, 6),
                        fontsize=8, color="m")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    ax.set_title("Top-down (xy) trajectory")
    return _save(fig, out_dir, "08_trajectory_xy.png")


def plot_yaw_aligned_flag(df: CsvTable, out_dir: Path):
    t = df["t_s"]
    fig, ax = plt.subplots(figsize=(11, 2.8))
    ax.plot(t, df["yaw_aligned"], "C2-", drawstyle="steps-post",
            label="yaw_aligned (1=True)")
    ax.plot(t, df["wp_index"], "C3-", drawstyle="steps-post",
            alpha=0.7, label="wp_index")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("flag / index")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    ax.set_title("Yaw-aligned flag & waypoint index")
    return _save(fig, out_dir, "09_flags.png")


def print_summary(df: CsvTable):
    t = df["t_s"]
    duration = t[-1] - t[0]
    dt = df["dt_s"]
    ex = df["err_x_w"]
    ey = df["err_y_w"]
    ez = df["err_z_w"]
    eyaw = df["err_yaw"]
    norm_xyz = np.sqrt(ex**2 + ey**2 + ez**2)

    print(f"  samples:            {len(df)}")
    print(f"  duration:           {duration:.2f} s")
    print(f"  mean / max dt:      {dt.mean()*1000:.1f} / {dt.max()*1000:.1f} ms")
    print(f"  max |err_x/y/z|:    {np.max(np.abs(ex)):.3f} / "
          f"{np.max(np.abs(ey)):.3f} / {np.max(np.abs(ez)):.3f} m")
    print(f"  max ||err_xyz||:    {norm_xyz.max():.3f} m")
    print(f"  final ||err_xyz||:  {norm_xyz[-1]:.3f} m")
    print(f"  max |err_yaw|:      {np.rad2deg(np.max(np.abs(eyaw))):.2f} deg")
    print(f"  yaw_aligned ratio:  {df['yaw_aligned'].mean()*100:.1f} %")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, help="flight CSV log path")
    parser.add_argument("--targets", default=None, help="waypoint CSV (optional)")
    parser.add_argument("--out", default=None, help="output directory")
    args = parser.parse_args()

    log_path = Path(args.log).expanduser().resolve()
    if not log_path.exists():
        print(f"log not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    targets_path = Path(args.targets).expanduser().resolve() if args.targets else None

    if args.out:
        out_dir = Path(args.out).expanduser().resolve()
    else:
        out_dir = log_path.with_suffix("").parent / (log_path.stem + "_plots")

    print(f"loading log:     {log_path}")
    df = load_log(log_path)
    targets = load_targets(targets_path)
    if targets is not None:
        print(f"loading targets: {targets_path}  ({len(targets)} waypoints)")

    print("summary:")
    print_summary(df)

    print(f"writing figures under: {out_dir}")
    saved = []
    saved.append(plot_xyz_tracking(df, out_dir))
    saved.append(plot_yaw_tracking(df, out_dir))
    saved.append(plot_position_error(df, out_dir))
    saved.append(plot_commands(df, out_dir))
    saved.append(plot_pid_breakdown(df, out_dir))
    saved.append(plot_velocity_estimates(df, out_dir))
    saved.append(plot_3d_trajectory(df, targets, out_dir))
    saved.append(plot_xy_topdown(df, targets, out_dir))
    saved.append(plot_yaw_aligned_flag(df, out_dir))

    for p in saved:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
