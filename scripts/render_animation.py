#!/usr/bin/env python3
"""Render smoothed MVEM osculating-orbit snapshots to an MP4 animation."""

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
    "Mercury": "#b7b1a7",
    "Venus": "#d9a441",
    "Earth": "#4f8bd6",
    "Mars": "#c65f3f",
}


def parse_indices(args: argparse.Namespace, snapshot_count: int) -> np.ndarray:
    if args.start_snapshot < 0 or args.end_snapshot >= snapshot_count:
        raise ValueError(f"Snapshot range must be within 0..{snapshot_count - 1}")
    if args.end_snapshot < args.start_snapshot:
        raise ValueError("--end-snapshot must be >= --start-snapshot")
    indices = np.arange(args.start_snapshot, args.end_snapshot + 1)
    if args.limit_frames is not None:
        indices = indices[: args.limit_frames]
    return indices


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


def solve_kepler_elliptic(mean_anomaly: np.ndarray, eccentricity: float) -> np.ndarray:
    mean_anomaly = np.mod(mean_anomaly, 2.0 * np.pi)
    E = mean_anomaly.copy()
    if eccentricity > 0.8:
        E[:] = np.pi
    for _ in range(20):
        delta = (E - eccentricity * np.sin(E) - mean_anomaly) / (1.0 - eccentricity * np.cos(E))
        E -= delta
        if np.max(np.abs(delta)) < 1e-13:
            break
    return E


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


def orbit_points(
    a: float,
    e: float,
    inc: float,
    Omega: float,
    omega: float,
    count: int,
    sun_xyz: np.ndarray,
) -> np.ndarray:
    mean_anomaly = np.arange(count, dtype=float) * (2.0 * np.pi / count)
    E = solve_kepler_elliptic(mean_anomaly, e)
    x_orb = a * (np.cos(E) - e)
    y_orb = a * math.sqrt(max(0.0, 1.0 - e * e)) * np.sin(E)
    local = np.column_stack([x_orb, y_orb, np.zeros_like(x_orb)])
    return local @ rotation_matrix(inc, Omega, omega).T + sun_xyz


def square_boundary_endpoint(start_xy: np.ndarray, direction_xy: np.ndarray, half_width: float) -> np.ndarray:
    direction_xy = direction_xy.astype(float)
    norm = np.linalg.norm(direction_xy)
    if norm == 0.0:
        return start_xy.copy()
    direction_xy /= norm

    candidates = []
    for axis in (0, 1):
        for boundary in (-half_width, half_width):
            if abs(direction_xy[axis]) < 1e-15:
                continue
            s = (boundary - start_xy[axis]) / direction_xy[axis]
            if s <= 0:
                continue
            point = start_xy + s * direction_xy
            if np.all(point >= -half_width - 1e-12) and np.all(point <= half_width + 1e-12):
                candidates.append((s, point))
    if not candidates:
        return start_xy.copy()
    return min(candidates, key=lambda item: item[0])[1]


def periastron_segment(
    inc: float,
    Omega: float,
    omega: float,
    sun_xyz: np.ndarray,
    half_width: float,
) -> np.ndarray:
    direction = rotation_matrix(inc, Omega, omega) @ np.array([1.0, 0.0, 0.0])
    end_xy = square_boundary_endpoint(sun_xyz[:2], direction[:2], half_width)
    return np.array([[sun_xyz[0], sun_xyz[1], sun_xyz[2]], [end_xy[0], end_xy[1], sun_xyz[2]]])


def infer_sidecar_path(input_path: str, filename: str) -> str:
    return str(Path(input_path).with_name(filename))


def load_view_basis(args: argparse.Namespace, snapshot_body_names: np.ndarray, states: np.ndarray) -> np.ndarray:
    if args.view_plane == "icrf":
        return np.eye(3)

    with Path(args.metadata).open() as handle:
        metadata = json.load(handle)

    masses = np.array(
        [
            metadata["initial_state_after_ephemeris_before_com_shift"][str(name)]["mass_msun"]
            for name in snapshot_body_names
        ],
        dtype=float,
    )
    r = states[0, :, :3]
    v = states[0, :, 3:]
    angular_momentum = np.sum(masses[:, None] * np.cross(r, v), axis=0)
    zhat = angular_momentum / np.linalg.norm(angular_momentum)

    x_seed = np.array([1.0, 0.0, 0.0])
    xhat = x_seed - np.dot(x_seed, zhat) * zhat
    if np.linalg.norm(xhat) < 1e-12:
        x_seed = np.array([0.0, 1.0, 0.0])
        xhat = x_seed - np.dot(x_seed, zhat) * zhat
    xhat /= np.linalg.norm(xhat)
    yhat = np.cross(zhat, xhat)
    return np.column_stack([xhat, yhat, zhat])


def setup_axis(ax, half: float, side_half_z: float | None = None) -> None:
    ax.set_facecolor("#05070a")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-half, half)
    if side_half_z is None:
        ax.set_ylim(-half, half)
    else:
        ax.set_ylim(-side_half_z, side_half_z)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#454b55")
        spine.set_linewidth(0.45)


def draw_odometer(ax, years: float, args: argparse.Namespace) -> None:
    if not args.odometer:
        return
    text = f"T+{int(round(years)):09,d} yr"
    ax.text(
        0.982,
        0.976,
        text,
        transform=ax.transAxes,
        color="#d7dbe2",
        fontsize=args.odometer_size,
        family="monospace",
        ha="right",
        va="top",
        bbox={
            "boxstyle": "round,pad=0.22,rounding_size=0.02",
            "facecolor": "#0b0e13",
            "edgecolor": "#2d333d",
            "linewidth": 0.45,
            "alpha": 0.84,
        },
    )


def draw_planet_labels(ax, names: list[str], args: argparse.Namespace) -> None:
    if not args.labels:
        return
    x = 0.026
    y0 = 0.965
    dy = 0.036
    for i, name in enumerate(names):
        y = y0 - i * dy
        ax.scatter([x], [y], s=args.label_marker_size, color=COLORS[name], transform=ax.transAxes, linewidths=0, zorder=50)
        ax.text(
            x + 0.018,
            y,
            name,
            transform=ax.transAxes,
            color=COLORS[name],
            fontsize=args.label_size,
            ha="left",
            va="center",
            alpha=0.92,
        )


def precompute_smoothed_elements(
    args: argparse.Namespace,
    animation_names: list[str],
    snapshot_body_names: np.ndarray,
    element_names: np.ndarray,
    elements: np.ndarray,
) -> dict[str, np.ndarray]:
    body_lookup = {str(name): i for i, name in enumerate(snapshot_body_names)}
    element_lookup = {str(name): i for i, name in enumerate(element_names)}
    out = {}

    for name in animation_names:
        body_idx = body_lookup[name]
        a = elements[:, body_idx, element_lookup["a"]]
        e = elements[:, body_idx, element_lookup["e"]]
        inc = elements[:, body_idx, element_lookup["inc"]]
        Omega = elements[:, body_idx, element_lookup["Omega"]]
        omega = elements[:, body_idx, element_lookup["omega"]]

        varpi = Omega + omega
        k = e * np.cos(varpi)
        h = e * np.sin(varpi)
        k_smooth = centered_lowpass(k, args.smooth_window_frames)
        h_smooth = centered_lowpass(h, args.smooth_window_frames)
        e_smooth = np.clip(np.hypot(k_smooth, h_smooth), 0.0, 0.95)
        varpi_smooth = np.arctan2(h_smooth, k_smooth)
        omega_smooth = varpi_smooth - Omega

        out[name] = np.column_stack([a, e_smooth, inc, Omega, omega_smooth])

    return out


def render(args: argparse.Namespace) -> None:
    data = np.load(args.input)
    snapshots = np.load(args.snapshots)
    names = [str(name) for name in data["body_names"]]
    times_yr = data["times_yr"]
    counts = data["ellipse_counts"]
    half = float(data["box_side_au"]) / 2.0
    side_half_z = half / args.main_to_side_ratio
    indices = parse_indices(args, len(times_yr))

    states = snapshots["states_au_auyr"]
    basis = load_view_basis(args, snapshots["body_names"], states)
    sun_positions = states[:, 0, :3] @ basis
    smoothed = precompute_smoothed_elements(
        args,
        names,
        snapshots["body_names"],
        snapshots["element_names"],
        snapshots["elements"],
    )

    fig = plt.figure(
        figsize=(args.size_in, args.size_in * (args.main_to_side_ratio + 1.0) / args.main_to_side_ratio),
        dpi=args.dpi,
    )
    fig.patch.set_facecolor("#05070a")
    gs = GridSpec(
        2,
        1,
        height_ratios=[args.main_to_side_ratio, 1.0],
        hspace=0.018,
        left=0.0,
        right=1.0,
        top=1.0,
        bottom=0.0,
    )
    ax_main = fig.add_subplot(gs[0, 0])
    ax_side = fig.add_subplot(gs[1, 0])
    setup_axis(ax_main, half)
    setup_axis(ax_side, half, side_half_z)

    writer = FFMpegWriter(
        fps=args.fps,
        codec="libx264",
        bitrate=args.bitrate,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart", "-crf", str(args.crf)],
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with writer.saving(fig, str(out), dpi=args.dpi):
        for frame_number, snap_idx in enumerate(indices, start=1):
            ax_main.clear()
            ax_side.clear()
            setup_axis(ax_main, half)
            setup_axis(ax_side, half, side_half_z)

            sun_xyz = sun_positions[snap_idx]
            rendered_points = []
            rendered_segments = []
            eccentricities = []

            for j, name in enumerate(names):
                a, e, inc, Omega, omega = smoothed[name][snap_idx]
                raw_points = orbit_points(a, e, inc, Omega, omega, int(counts[snap_idx, j]), states[snap_idx, 0, :3])
                raw_segment = periastron_segment(inc, Omega, omega, states[snap_idx, 0, :3], half)
                rendered_points.append(raw_points @ basis)
                rendered_segments.append(raw_segment @ basis)
                eccentricities.append(e)

                ax_main.add_patch(
                    plt.Circle(
                        sun_xyz[:2],
                        float(a),
                        fill=False,
                        color=COLORS[name],
                        linewidth=args.circle_width,
                        alpha=0.30,
                    )
                )

            for name, segment, e in zip(names, rendered_segments, eccentricities, strict=True):
                alpha = args.perihelion_alpha * min(1.0, max(0.0, e / args.perihelion_fade_e))
                ax_main.plot(
                    segment[:, 0],
                    segment[:, 1],
                    color=COLORS[name],
                    linewidth=args.perihelion_width,
                    alpha=alpha,
                    solid_capstyle="round",
                )

            for name, point_cloud in zip(names, rendered_points, strict=True):
                ax_main.scatter(
                    point_cloud[:, 0],
                    point_cloud[:, 1],
                    s=args.point_size,
                    color=COLORS[name],
                    alpha=0.86,
                    linewidths=0,
                )
                ax_main.scatter(
                    point_cloud[0, 0],
                    point_cloud[0, 1],
                    s=args.point_size * 3.2,
                    color=COLORS[name],
                    alpha=0.98,
                    linewidths=0,
                )
                ax_side.scatter(
                    point_cloud[:, 0],
                    point_cloud[:, 2],
                    s=args.side_point_size,
                    color=COLORS[name],
                    alpha=0.76,
                    linewidths=0,
                )

            ax_main.scatter(
                [sun_xyz[0]],
                [sun_xyz[1]],
                s=args.sun_size,
                color="#fff2b3",
                edgecolors="#fff9d6",
                linewidths=0.45,
                zorder=20,
            )
            ax_side.axhline(0.0, color="#454b55", linewidth=args.invariable_plane_width, alpha=0.75)
            ax_side.scatter(
                [sun_xyz[0]],
                [sun_xyz[2]],
                s=args.sun_size * 0.45,
                color="#fff2b3",
                edgecolors="#fff9d6",
                linewidths=0.30,
                zorder=20,
            )

            draw_planet_labels(ax_main, names, args)
            draw_odometer(ax_main, times_yr[snap_idx], args)

            writer.grab_frame(facecolor=fig.get_facecolor())
            if args.progress_every and (
                frame_number == 1
                or frame_number == len(indices)
                or frame_number % args.progress_every == 0
            ):
                print(f"frame {frame_number:4d}/{len(indices)} from snapshot {snap_idx}", flush=True)

    plt.close(fig)
    print(f"Wrote {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="outputs/full_5myr/animation_primitives_inner.npz")
    parser.add_argument("--snapshots", default=None)
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--out", default="outputs/full_5myr/solar_system_mvem_5myr_smoothed_side.mp4")
    parser.add_argument("--start-snapshot", type=int, default=1)
    parser.add_argument("--end-snapshot", type=int, default=1000)
    parser.add_argument("--limit-frames", type=int)
    parser.add_argument("--fps", type=float, default=33.0)
    parser.add_argument("--dpi", type=int, default=210)
    parser.add_argument("--size-in", type=float, default=8.0)
    parser.add_argument("--bitrate", type=int, default=20_000)
    parser.add_argument("--crf", type=int, default=14)
    parser.add_argument("--point-size", type=float, default=3.078)
    parser.add_argument("--side-point-size", type=float, default=2.106)
    parser.add_argument("--sun-size", type=float, default=21.6)
    parser.add_argument("--circle-width", type=float, default=0.48)
    parser.add_argument("--perihelion-width", type=float, default=0.624)
    parser.add_argument("--invariable-plane-width", type=float, default=0.84)
    parser.add_argument("--perihelion-alpha", type=float, default=0.58)
    parser.add_argument("--perihelion-fade-e", type=float, default=0.025)
    parser.add_argument("--smooth-window-frames", type=int, default=11)
    parser.add_argument("--view-plane", choices=["invariable", "icrf"], default="invariable")
    parser.add_argument("--main-to-side-ratio", type=float, default=6.0)
    parser.add_argument("--labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--label-size", type=float, default=11.7)
    parser.add_argument("--label-marker-size", type=float, default=11.52)
    parser.add_argument("--odometer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--odometer-size", type=float, default=13.5)
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()
    if args.snapshots is None:
        args.snapshots = infer_sidecar_path(args.input, "snapshots.npz")
    if args.metadata is None:
        args.metadata = infer_sidecar_path(args.input, "metadata.json")
    return args


if __name__ == "__main__":
    render(parse_args())
