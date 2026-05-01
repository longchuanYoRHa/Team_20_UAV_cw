#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _build_time_seconds(ts: np.ndarray) -> np.ndarray:
    """
    Robustly build a monotonic time axis in seconds.

    Handles common log formats:
    - absolute epoch seconds (monotonic, large values)
    - relative dt seconds (small values, non-monotonic noise)
    - mixed: first row epoch, remaining rows dt (as seen in this repo)
    """
    ts = np.asarray(ts, dtype=float)
    if ts.size == 0:
        return ts

    # If we see an enormous backward jump (epoch -> dt), treat from index 1 onward as dt.
    if ts.size >= 2 and (ts[0] > 1e9) and (ts[1] < 1e3):
        dt = np.clip(ts[1:], 0.0, None)
        t = np.concatenate(([0.0], np.cumsum(dt)))
        return t

    # If it looks like epoch timestamps, normalize to start at 0.
    if np.nanmedian(ts) > 1e9:
        t = ts - ts[0]
        # protect against occasional negative glitches
        t = np.maximum.accumulate(np.nan_to_num(t, nan=0.0))
        return t

    # Otherwise treat as dt sequence; clip negatives and integrate.
    dt = np.clip(ts, 0.0, None)
    t = np.cumsum(dt)
    t -= t[0]
    return t


def _maybe(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns and pd.api.types.is_numeric_dtype(df[col])


def _savefig(out_dir: Path, stem: str, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}_{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_one(csv_path: Path, out_dir: Path) -> list[Path]:
    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns:
        raise ValueError(f"{csv_path}: missing 'timestamp' column")

    t = _build_time_seconds(df["timestamp"].to_numpy())
    stem = csv_path.stem
    saved: list[Path] = []

    # 1) Position tracking
    if all(_maybe(df, c) for c in ["x", "y", "z"]) and all(_maybe(df, c) for c in ["target_x", "target_y", "target_z"]):
        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for ax, c, tc, label in [
            (axs[0], "x", "target_x", "x"),
            (axs[1], "y", "target_y", "y"),
            (axs[2], "z", "target_z", "z"),
        ]:
            ax.plot(t, df[c], label=f"{label}")
            ax.plot(t, df[tc], "--", label=f"{label}_target")
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")
        axs[-1].set_xlabel("time [s]")
        fig.suptitle(f"{stem} — position tracking")
        saved.append(_savefig(out_dir, stem, "01_position_tracking"))

        # XY trajectory
        fig = plt.figure(figsize=(7, 7))
        plt.plot(df["x"], df["y"], label="traj")
        plt.plot(df["target_x"], df["target_y"], "--", label="target")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis("equal")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best")
        plt.title(f"{stem} — trajectory XY")
        saved.append(_savefig(out_dir, stem, "08_trajectory_xy"))

        # 3D trajectory (if z exists)
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(df["x"], df["y"], df["z"], label="traj")
        ax.plot(df["target_x"], df["target_y"], df["target_z"], "--", label="target")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(f"{stem} — trajectory 3D")
        ax.legend(loc="best")
        saved.append(_savefig(out_dir, stem, "07_trajectory_3d"))

    # 2) Yaw tracking
    if _maybe(df, "yaw") and _maybe(df, "target_yaw"):
        fig = plt.figure(figsize=(10, 4))
        plt.plot(t, df["yaw"], label="yaw")
        plt.plot(t, df["target_yaw"], "--", label="yaw_target")
        plt.xlabel("time [s]")
        plt.ylabel("yaw [rad]")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best")
        plt.title(f"{stem} — yaw tracking")
        saved.append(_savefig(out_dir, stem, "02_yaw_tracking"))

    # 3) Errors
    if _maybe(df, "pos_error") or _maybe(df, "yaw_error"):
        fig = plt.figure(figsize=(10, 4))
        if _maybe(df, "pos_error"):
            plt.plot(t, df["pos_error"], label="pos_error")
        if _maybe(df, "yaw_error"):
            plt.plot(t, df["yaw_error"], label="yaw_error")
        plt.xlabel("time [s]")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best")
        plt.title(f"{stem} — errors")
        saved.append(_savefig(out_dir, stem, "03_errors"))

    # 4) Commanded velocities / yaw rate
    cmd_cols = [c for c in ["vx_cmd", "vy_cmd", "vz_cmd", "yaw_rate_cmd"] if _maybe(df, c)]
    if cmd_cols:
        fig, axs = plt.subplots(len(cmd_cols), 1, figsize=(10, 2.6 * len(cmd_cols)), sharex=True)
        if len(cmd_cols) == 1:
            axs = [axs]
        for ax, c in zip(axs, cmd_cols):
            ax.plot(t, df[c], label=c)
            ax.set_ylabel(c)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")
        axs[-1].set_xlabel("time [s]")
        fig.suptitle(f"{stem} — commands")
        saved.append(_savefig(out_dir, stem, "04_commands"))

    return saved


def _expand_inputs(inputs: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for s in inputs:
        p = Path(s)
        if p.is_dir():
            paths.extend(sorted(p.glob("output_*.csv")))
        else:
            paths.append(p)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot secondrun output_*.csv logs.")
    ap.add_argument(
        "inputs",
        nargs="*",
        default=["secondrun"],
        help="CSV files or a directory containing output_*.csv (default: secondrun/)",
    )
    ap.add_argument("--out", default="secondrun/plots", help="Output directory for PNGs")
    args = ap.parse_args()

    csv_paths = _expand_inputs(args.inputs)
    if not csv_paths:
        raise SystemExit("No inputs found.")

    out_dir = Path(args.out)
    all_saved: list[Path] = []
    for csv_path in csv_paths:
        all_saved.extend(plot_one(csv_path, out_dir))

    # Print a compact summary for convenience.
    if all_saved:
        print("Saved:")
        for p in all_saved:
            print(f"- {p}")
    else:
        print("No plots generated (missing expected columns).")


if __name__ == "__main__":
    main()

