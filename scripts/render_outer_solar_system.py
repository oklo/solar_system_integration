#!/usr/bin/env python3
"""Render the outer Solar System from stored 5 kyr osculating elements."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.gridspec import GridSpec


COLORS = {
    "Mercury": "#9a958c",
    "Venus": "#c4933a",
    "Earth": "#4f8bd6",
    "Mars": "#c65f3f",
    "Jupiter": "#d9b27c",
    "Saturn": "#d0c082",
    "Uranus": "#72c7d8",
    "Neptune": "#4d72d8",
    "Pluto": "#b9aaa0",
}

INNER = ("Mercury", "Venus", "Earth", "Mars")
OUTER = ("Jupiter", "Saturn", "Uranus", "Neptune", "Pluto")


def centered_lowpass(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    if window % 2 == 0:
        window += 1
    kernel = np.hanning(window)
    kernel /= kernel.sum()
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def rotation_matrix(inc: float, Omega: float, omega: float) -> np.ndarray:
    cO, sO = math.cos(Omega), math.sin(Omega)
    co, so = math.cos(omega), math.sin(omega)
    ci, si = math.cos(inc), math.sin(inc)
    return np.array(
        [
            [cO * co - sO * so * ci, -cO * so - sO * co * ci, sO * si],
            [sO * co + cO * so * ci, -sO * so + cO * co * ci, -cO * si],
            [so * si, co * si, ci],
        ],
        dtype=float,
    )


def points_at_true_anomaly(
    a: float,
    e: float,
    inc: float,
    Omega: float,
    omega: float,
    count: int,
    sun_xyz: np.ndarray,
) -> np.ndarray:
    f = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    r = a * (1.0 - e * e) / (1.0 + e * np.cos(f))
    local = np.column_stack([r * np.cos(f), r * np.sin(f), np.zeros_like(f)])
    return local @ rotation_matrix(inc, Omega, omega).T + sun_xyz


def ellipse_line(
    a: float,
    e: float,
    inc: float,
    Omega: float,
    omega: float,
    count: int,
    sun_xyz: np.ndarray,
) -> np.ndarray:
    f = np.linspace(0.0, 2.0 * np.pi, count)
    r = a * (1.0 - e * e) / (1.0 + e * np.cos(f))
    local = np.column_stack([r * np.cos(f), r * np.sin(f), np.zeros_like(f)])
    return local @ rotation_matrix(inc, Omega, omega).T + sun_xyz


def load_basis(metadata_path: Path, body_names: list[str], states: np.ndarray, view_plane: str) -> np.ndarray:
    if view_plane == "icrf":
        return np.eye(3)
    with metadata_path.open() as handle:
        metadata = json.load(handle)
    masses = np.array(
        [
            metadata["initial_state_after_ephemeris_before_com_shift"][name]["mass_msun"]
            for name in body_names
        ],
        dtype=float,
    )
    r = states[0, :, :3]
    v = states[0, :, 3:]
    zhat = np.sum(masses[:, None] * np.cross(r, v), axis=0)
    zhat /= np.linalg.norm(zhat)
    x_seed = np.array([1.0, 0.0, 0.0])
    xhat = x_seed - np.dot(x_seed, zhat) * zhat
    if np.linalg.norm(xhat) < 1e-12:
        x_seed = np.array([0.0, 1.0, 0.0])
        xhat = x_seed - np.dot(x_seed, zhat) * zhat
    xhat /= np.linalg.norm(xhat)
    yhat = np.cross(zhat, xhat)
    return np.column_stack([xhat, yhat, zhat])


def smooth_elements(
    body_names: list[str],
    element_names: list[str],
    elements: np.ndarray,
    window: int,
) -> dict[str, np.ndarray]:
    element_lookup = {name: i for i, name in enumerate(element_names)}
    out = {}
    for body_idx, name in enumerate(body_names):
        if name == "Sun":
            continue
        a = elements[:, body_idx, element_lookup["a"]]
        e = elements[:, body_idx, element_lookup["e"]]
        inc = elements[:, body_idx, element_lookup["inc"]]
        Omega = elements[:, body_idx, element_lookup["Omega"]]
        omega = elements[:, body_idx, element_lookup["omega"]]
        varpi = Omega + omega
        k = e * np.cos(varpi)
        h = e * np.sin(varpi)
        k_smooth = centered_lowpass(k, window)
        h_smooth = centered_lowpass(h, window)
        e_smooth = np.clip(np.hypot(k_smooth, h_smooth), 0.0, 0.95)
        varpi_smooth = np.arctan2(h_smooth, k_smooth)
        out[name] = np.column_stack([a, e_smooth, inc, Omega, varpi_smooth - Omega])
    return out


def setup_axis(ax, x_half: float, y_half: float) -> None:
    ax.set_facecolor("#05070a")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-x_half, x_half)
    ax.set_ylim(-y_half, y_half)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#454b55")
        spine.set_linewidth(0.55)


def draw_odometer(ax, years: float, fontsize: float) -> None:
    ax.text(
        0.982,
        0.976,
        f"T+{int(round(years)):09,d} yr",
        transform=ax.transAxes,
        color="#d7dbe2",
        fontsize=fontsize,
        family="monospace",
        ha="right",
        va="top",
        bbox={
            "boxstyle": "round,pad=0.22,rounding_size=0.02",
            "facecolor": "#0b0e13",
            "edgecolor": "#2d333d",
            "linewidth": 0.5,
            "alpha": 0.84,
        },
    )


def draw_labels(ax, fontsize: float, marker_size: float) -> None:
    x = 0.024
    y0 = 0.965
    dy = 0.034
    for i, name in enumerate((*OUTER, *INNER)):
        y = y0 - i * dy
        ax.scatter([x], [y], s=marker_size, color=COLORS[name], transform=ax.transAxes, linewidths=0, zorder=50)
        ax.text(x + 0.017, y, name, transform=ax.transAxes, color=COLORS[name], fontsize=fontsize, ha="left", va="center")


def parse_indices(start: int, end: int, snapshot_count: int, limit_frames: int | None) -> np.ndarray:
    if start < 0 or end >= snapshot_count or end < start:
        raise ValueError(f"Snapshot range must be within 0..{snapshot_count - 1} and nondecreasing")
    indices = np.arange(start, end + 1)
    if limit_frames is not None:
        indices = indices[:limit_frames]
    return indices


def render(args: argparse.Namespace) -> None:
    snapshots = np.load(args.snapshots)
    times_yr = snapshots["times_yr"]
    body_names = [str(name) for name in snapshots["body_names"]]
    element_names = [str(name) for name in snapshots["element_names"]]
    states = snapshots["states_au_auyr"]
    elements = snapshots["elements"]
    basis = load_basis(Path(args.metadata), body_names, states, args.view_plane)
    smoothed = smooth_elements(body_names, element_names, elements, args.smooth_window_frames)
    indices = parse_indices(args.start_snapshot, args.end_snapshot, len(times_yr), args.limit_frames)

    fig_height = args.size_in * (args.box_half_z_au / args.box_half_au + 1.0)
    fig = plt.figure(figsize=(args.size_in, fig_height), dpi=args.dpi)
    fig.patch.set_facecolor("#05070a")
    gs = GridSpec(
        2,
        1,
        height_ratios=[args.box_half_au, args.box_half_z_au],
        hspace=0.012,
        left=0.0,
        right=1.0,
        top=1.0,
        bottom=0.0,
    )
    ax_main = fig.add_subplot(gs[0, 0])
    ax_side = fig.add_subplot(gs[1, 0])

    writer = FFMpegWriter(
        fps=args.fps,
        codec="libx264",
        bitrate=args.bitrate,
        extra_args=[
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-crf",
            str(args.crf),
            "-g",
            "1",
            "-keyint_min",
            "1",
            "-sc_threshold",
            "0",
        ],
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with writer.saving(fig, str(out), dpi=args.dpi):
        for frame_number, snap_idx in enumerate(indices, start=1):
            ax_main.clear()
            ax_side.clear()
            setup_axis(ax_main, args.box_half_au, args.box_half_au)
            setup_axis(ax_side, args.box_half_au, args.box_half_z_au)

            sun_xyz = states[snap_idx, 0, :3]
            sun_view = sun_xyz @ basis
            ax_main.scatter([sun_view[0]], [sun_view[1]], s=args.sun_size, color="#fff2b3", edgecolors="#fff9d6", linewidths=0.5, zorder=30)
            ax_side.axhline(0.0, color="#5a6069", linewidth=args.invariable_plane_width, alpha=0.75)
            ax_side.scatter([sun_view[0]], [sun_view[2]], s=args.sun_size * 0.42, color="#fff2b3", edgecolors="#fff9d6", linewidths=0.35, zorder=30)

            for name in INNER:
                a, e, inc, Omega, omega = smoothed[name][snap_idx]
                pts = ellipse_line(a, e, inc, Omega, omega, args.inner_line_points, sun_xyz) @ basis
                ax_main.plot(pts[:, 0], pts[:, 1], color=COLORS[name], linewidth=args.inner_line_width, alpha=0.52)
                ax_side.plot(pts[:, 0], pts[:, 2], color=COLORS[name], linewidth=args.inner_line_width, alpha=0.44)

            for name in OUTER:
                a, e, inc, Omega, omega = smoothed[name][snap_idx]
                line = ellipse_line(a, e, inc, Omega, omega, args.outer_line_points, sun_xyz) @ basis
                pts = points_at_true_anomaly(a, e, inc, Omega, omega, args.outer_markers, sun_xyz) @ basis
                color = COLORS[name]
                ax_main.plot(line[:, 0], line[:, 1], color=color, linewidth=args.outer_line_width, alpha=0.28)
                ax_side.plot(line[:, 0], line[:, 2], color=color, linewidth=args.outer_line_width, alpha=0.26)
                ax_main.scatter(pts[:, 0], pts[:, 1], s=args.outer_marker_size, color=color, alpha=0.88, linewidths=0)
                ax_side.scatter(pts[:, 0], pts[:, 2], s=args.side_marker_size, color=color, alpha=0.78, linewidths=0)
                ax_main.scatter(pts[0, 0], pts[0, 1], s=args.outer_marker_size * 2.6, color=color, alpha=0.98, linewidths=0)

            draw_labels(ax_main, args.label_size, args.label_marker_size)
            draw_odometer(ax_main, times_yr[snap_idx], args.odometer_size)
            writer.grab_frame(facecolor=fig.get_facecolor())

            if args.progress_every and (
                frame_number == 1 or frame_number == len(indices) or frame_number % args.progress_every == 0
            ):
                print(f"frame {frame_number:4d}/{len(indices)} from snapshot {snap_idx}", flush=True)

    plt.close(fig)
    print(f"Wrote {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", default="outputs/full_5myr/snapshots.npz")
    parser.add_argument("--metadata", default="outputs/full_5myr/metadata.json")
    parser.add_argument("--out", default="outputs/full_5myr/solar_system_outer_5myr_side_scrub.mp4")
    parser.add_argument("--start-snapshot", type=int, default=1)
    parser.add_argument("--end-snapshot", type=int, default=1000)
    parser.add_argument("--limit-frames", type=int)
    parser.add_argument("--fps", type=float, default=33.0)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--size-in", type=float, default=10.0)
    parser.add_argument("--bitrate", type=int, default=32_000)
    parser.add_argument("--crf", type=int, default=16)
    parser.add_argument("--box-half-au", type=float, default=55.0)
    parser.add_argument("--box-half-z-au", type=float, default=16.5)
    parser.add_argument("--outer-markers", type=int, default=100)
    parser.add_argument("--outer-line-points", type=int, default=360)
    parser.add_argument("--inner-line-points", type=int, default=180)
    parser.add_argument("--outer-marker-size", type=float, default=7.0)
    parser.add_argument("--side-marker-size", type=float, default=5.4)
    parser.add_argument("--outer-line-width", type=float, default=0.74)
    parser.add_argument("--inner-line-width", type=float, default=0.58)
    parser.add_argument("--invariable-plane-width", type=float, default=1.0)
    parser.add_argument("--sun-size", type=float, default=24.0)
    parser.add_argument("--smooth-window-frames", type=int, default=11)
    parser.add_argument("--view-plane", choices=["invariable", "icrf"], default="invariable")
    parser.add_argument("--label-size", type=float, default=12.5)
    parser.add_argument("--label-marker-size", type=float, default=14.0)
    parser.add_argument("--odometer-size", type=float, default=13.5)
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    render(parse_args())
