#!/usr/bin/env python3
"""Render the kicked-Earth experiment as an orbit animation with a history panel."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.gridspec import GridSpec

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.render_animation import orbit_points, rotation_matrix, square_boundary_endpoint


COLORS = {
    "Mercury": "#b7b1a7",
    "Venus": "#d9a441",
    "Earth": "#5da2ff",
    "Mars": "#d96a45",
}
INNER = ("Mercury", "Venus", "Earth", "Mars")
AU_PER_YR_TO_KM_PER_S = 149_597_870.7 / (365.25 * 86_400.0)
AU_KM = 149_597_870.7
MARS_DIAMETER_KM = 6_779.0
MOON_DIAMETER_KM = 3_474.8
MOON_DISTANCE_KM = 384_400.0


def parse_indices(args: argparse.Namespace, snapshot_count: int) -> np.ndarray:
    indices = np.arange(args.start_snapshot, args.end_snapshot + 1)
    indices = indices[(indices >= 0) & (indices < snapshot_count)]
    if args.limit_frames is not None:
        indices = indices[: args.limit_frames]
    return indices


def load_view_basis(metadata_path: str, body_names: np.ndarray, states: np.ndarray) -> np.ndarray:
    with Path(metadata_path).open() as handle:
        metadata = json.load(handle)
    masses = np.array(
        [
            metadata["initial_state_after_ephemeris_before_com_shift"][str(name)]["mass_msun"]
            for name in body_names
        ],
        dtype=float,
    )
    r = states[0, :, :3]
    v = states[0, :, 3:]
    angular_momentum = np.sum(masses[:, None] * np.cross(r, v), axis=0)
    zhat = angular_momentum / np.linalg.norm(angular_momentum)
    x_seed = np.array([1.0, 0.0, 0.0])
    xhat = x_seed - np.dot(x_seed, zhat) * zhat
    xhat /= np.linalg.norm(xhat)
    yhat = np.cross(zhat, xhat)
    return np.column_stack([xhat, yhat, zhat])


def setup_main(ax, box_half: float, args: argparse.Namespace) -> None:
    ax.set_facecolor("#05070a")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-box_half, box_half)
    ax.set_ylim(-box_half, box_half)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#454b55")
        spine.set_linewidth(args.spine_width)


def setup_side(ax, box_half: float, side_half: float, args: argparse.Namespace) -> None:
    ax.set_facecolor("#05070a")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-box_half, box_half)
    ax.set_ylim(-side_half, side_half)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#454b55")
        spine.set_linewidth(args.spine_width)


def setup_history(ax, years_myr: np.ndarray, y_min: float, y_max: float, args: argparse.Namespace) -> None:
    ax.set_facecolor("#05070a")
    ax.set_xlim(years_myr[0], years_myr[-1])
    ax.set_ylim(y_min, y_max)
    ax.tick_params(colors="#8e96a3", labelsize=args.tick_size, length=3.2, width=args.tick_width)
    for spine in ax.spines.values():
        spine.set_color("#454b55")
        spine.set_linewidth(args.spine_width)
    ax.grid(color="#242a33", linewidth=args.grid_width, alpha=0.58)
    ax.set_ylabel("semi-major axis [AU]", color="#aeb6c2", fontsize=args.axis_label_size)


def setup_distance_axis(ax, years_myr: np.ndarray, distance_max: float, args: argparse.Namespace) -> None:
    ax.set_facecolor("#05070a")
    ax.set_xlim(years_myr[0], years_myr[-1])
    ax.set_ylim(distance_max, 0.0)
    ax.tick_params(colors="#8e96a3", labelsize=args.tick_size, length=3.2, width=args.tick_width)
    for spine in ax.spines.values():
        spine.set_color("#454b55")
        spine.set_linewidth(args.spine_width)
    ax.grid(color="#242a33", linewidth=args.grid_width, alpha=0.58)
    ax.set_xlabel("time [Myr]", color="#aeb6c2", fontsize=args.axis_label_size)
    ax.set_ylabel("minimum Earth-Mars distance [AU]", color="#aeb6c2", fontsize=args.axis_label_size)


def setup_angular_axis(ax, args: argparse.Namespace) -> None:
    ax.set_facecolor("#05070a")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#454b55")
        spine.set_linewidth(args.spine_width)


def sphere_texture(base_rgb: tuple[float, float, float], size: int, seed: int, crater_count: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[-1.0:1.0:complex(size), -1.0:1.0:complex(size)]
    rr = np.sqrt(xx * xx + yy * yy)
    mask = rr <= 1.0
    light = np.clip(0.92 - 0.42 * xx + 0.18 * yy, 0.18, 1.08)
    limb = np.clip(1.0 - 0.55 * rr**2, 0.0, 1.0)
    texture = light * limb
    texture += 0.055 * np.sin(16.0 * xx + 9.0 * yy) + 0.035 * np.sin(28.0 * yy - 4.0 * xx)
    for _ in range(crater_count):
        cx, cy = rng.uniform(-0.82, 0.82, 2)
        if cx * cx + cy * cy > 0.78:
            continue
        radius = rng.uniform(0.025, 0.10)
        crater_r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        depression = np.exp(-((crater_r / radius) ** 2) * 2.6)
        rim = np.exp(-(((crater_r - radius) / (0.28 * radius)) ** 2) * 2.0)
        texture -= rng.uniform(0.04, 0.12) * depression
        texture += rng.uniform(0.015, 0.055) * rim
    texture = np.clip(texture, 0.0, 1.0)
    rgb = np.dstack([texture * channel for channel in base_rgb])
    alpha = np.where(mask, 1.0, 0.0)
    edge = np.clip((1.0 - rr) / 0.025, 0.0, 1.0)
    alpha *= edge
    return np.dstack([rgb, alpha])


def add_ellipse(field: np.ndarray, xx: np.ndarray, yy: np.ndarray, ellipse: tuple[float, float, float, float, float, float]) -> None:
    cx, cy, rx, ry, angle_deg, strength = ellipse
    angle = np.deg2rad(angle_deg)
    xp = (xx - cx) * np.cos(angle) + (yy - cy) * np.sin(angle)
    yp = -(xx - cx) * np.sin(angle) + (yy - cy) * np.cos(angle)
    rr = (xp / rx) ** 2 + (yp / ry) ** 2
    field -= strength * np.exp(-2.4 * rr)


def moon_texture(size: int = 256) -> np.ndarray:
    rng = np.random.default_rng(11)
    yy, xx = np.mgrid[-1.0:1.0:complex(size), -1.0:1.0:complex(size)]
    rr = np.sqrt(xx * xx + yy * yy)
    mask = rr <= 1.0
    light = np.clip(1.00 - 0.14 * xx + 0.07 * yy, 0.42, 1.08)
    limb = np.clip(1.0 - 0.34 * rr**2, 0.0, 1.0)
    texture = 0.82 * light * limb
    texture += 0.010 * np.sin(13.0 * xx + 5.0 * yy) + 0.008 * np.sin(25.0 * yy - 6.0 * xx)

    maria = [
        (-0.48, 0.08, 0.29, 0.62, -9.0, 0.34),   # Oceanus Procellarum
        (-0.19, 0.43, 0.30, 0.22, -7.0, 0.40),   # Mare Imbrium
        (0.17, 0.43, 0.18, 0.14, 4.0, 0.34),     # Mare Serenitatis
        (0.35, 0.21, 0.26, 0.15, -13.0, 0.33),   # Mare Tranquillitatis
        (0.47, -0.07, 0.19, 0.22, -3.0, 0.27),   # Mare Fecunditatis
        (0.61, 0.28, 0.13, 0.15, 8.0, 0.34),     # Mare Crisium
        (0.25, -0.26, 0.14, 0.12, -10.0, 0.22),  # Mare Nectaris
        (-0.08, -0.39, 0.25, 0.14, 4.0, 0.24),   # Mare Nubium
        (-0.35, -0.40, 0.17, 0.14, -4.0, 0.22),  # Mare Humorum
    ]
    for ellipse in maria:
        add_ellipse(texture, xx, yy, ellipse)

    highlands = [
        (0.43, -0.50, 0.32, 0.22, -18.0, -0.04),
        (-0.08, -0.63, 0.22, 0.18, 0.0, -0.05),
    ]
    for cx, cy, rx, ry, angle, strength in highlands:
        add_ellipse(texture, xx, yy, (cx, cy, rx, ry, angle, strength))

    for cx, cy, radius, brightness in [(-0.08, -0.62, 0.028, 0.20), (-0.26, 0.02, 0.026, 0.16), (0.42, -0.32, 0.024, 0.13)]:
        crater_r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        texture += brightness * np.exp(-(crater_r / radius) ** 2)
        texture -= 0.10 * np.exp(-((crater_r - 1.8 * radius) / (0.55 * radius)) ** 2)

    tycho_x, tycho_y = -0.08, -0.62
    ray_angle = np.arctan2(yy - tycho_y, xx - tycho_x)
    ray_radius = np.sqrt((xx - tycho_x) ** 2 + (yy - tycho_y) ** 2)
    rays = np.zeros_like(texture)
    for angle in np.linspace(-2.75, 0.85, 12):
        rays += np.exp(-((np.sin(ray_angle - angle) * ray_radius) / 0.014) ** 2) * np.exp(-ray_radius / 0.55)
    texture += 0.014 * rays

    for _ in range(50):
        cx, cy = rng.uniform(-0.90, 0.90, 2)
        if cx * cx + cy * cy > 0.82:
            continue
        radius = rng.uniform(0.008, 0.034)
        crater_r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        texture -= rng.uniform(0.010, 0.030) * np.exp(-((crater_r / radius) ** 2) * 2.8)
        texture += rng.uniform(0.008, 0.026) * np.exp(-(((crater_r - radius) / (0.30 * radius)) ** 2) * 2.0)

    texture = np.clip(texture, 0.0, 1.0)
    alpha = np.where(mask, 1.0, 0.0) * np.clip((1.0 - rr) / 0.018, 0.0, 1.0)
    rgb = np.dstack([texture * 0.98, texture * 0.97, texture * 0.91])
    return np.dstack([rgb, alpha])


def mars_texture(size: int = 256) -> np.ndarray:
    yy, xx = np.mgrid[-1.0:1.0:complex(size), -1.0:1.0:complex(size)]
    rr = np.sqrt(xx * xx + yy * yy)
    mask = rr <= 1.0
    light = np.clip(1.02 - 0.34 * xx + 0.10 * yy, 0.24, 1.10)
    limb = np.clip(1.0 - 0.50 * rr**2, 0.0, 1.0)
    terrain = 0.78 * light * limb
    terrain += 0.045 * np.sin(9.0 * xx + 3.2 * yy) + 0.025 * np.sin(22.0 * yy - 6.0 * xx)
    for ellipse in [
        (-0.06, 0.12, 0.20, 0.36, 18.0, 0.25),  # Syrtis Major-like wedge
        (0.26, -0.18, 0.38, 0.045, -8.0, 0.20), # Valles Marineris belt
        (-0.42, -0.10, 0.24, 0.13, 10.0, 0.12),
        (0.45, 0.26, 0.20, 0.10, -12.0, 0.11),
    ]:
        add_ellipse(terrain, xx, yy, ellipse)
    terrain = np.clip(terrain, 0.0, 1.0)
    rgb = np.dstack([terrain * 1.00, terrain * 0.43, terrain * 0.22])
    polar_north = np.exp(-((yy - 0.78) ** 2 + (0.72 * xx) ** 2) / 0.018)
    polar_south = np.exp(-((yy + 0.82) ** 2 + (0.82 * xx) ** 2) / 0.012)
    polar = np.maximum(polar_north, polar_south)
    rgb = np.where((polar[..., None] > 0.18) & mask[..., None], np.maximum(rgb, 0.86 * polar[..., None]), rgb)
    alpha = np.where(mask, 1.0, 0.0) * np.clip((1.0 - rr) / 0.020, 0.0, 1.0)
    return np.dstack([np.clip(rgb, 0.0, 1.0), alpha])


def angular_ratio_mars_to_moon(distance_au: np.ndarray) -> np.ndarray:
    distance_km = distance_au * AU_KM
    return (MARS_DIAMETER_KM / distance_km) / (MOON_DIAMETER_KM / MOON_DISTANCE_KM)


def draw_disk_image(ax, image: np.ndarray, x: float, y: float, rx: float, ry: float, zorder: int) -> None:
    ax.imshow(
        image,
        extent=(x - rx, x + rx, y - ry, y + ry),
        transform=ax.transAxes,
        origin="lower",
        interpolation="lanczos",
        aspect="auto",
        clip_on=True,
        zorder=zorder,
    )


def draw_angular_size_panel(
    ax,
    fig,
    moon_image: np.ndarray,
    mars_image: np.ndarray,
    event_times_myr: np.ndarray,
    event_distances_au: np.ndarray,
    current_time_myr: float,
    total_time_myr: float,
    args: argparse.Namespace,
) -> None:
    setup_angular_axis(ax, args)
    bbox = ax.get_window_extent()
    moon_radius_y = 0.5 * args.moon_fill_fraction
    moon_radius_x = moon_radius_y * bbox.height / max(1.0, bbox.width)
    draw_disk_image(ax, moon_image, args.moon_x, 0.5, moon_radius_x, moon_radius_y, zorder=5)

    visible = event_times_myr <= current_time_myr
    if not np.any(visible):
        return
    times = event_times_myr[visible]
    distances = event_distances_au[visible]
    ratios = angular_ratio_mars_to_moon(distances)
    x_positions = times / total_time_myr
    for idx in range(len(x_positions)):
        mars_radius_x = moon_radius_x * ratios[idx]
        mars_radius_y = moon_radius_y * ratios[idx]
        mars_diameter_px = 2.0 * mars_radius_y * bbox.height
        if mars_diameter_px < args.min_mars_image_px:
            mars_diameter_points = mars_diameter_px * 72.0 / ax.figure.dpi
            ax.scatter(
                [x_positions[idx]],
                [0.5],
                s=(mars_diameter_points * mars_diameter_points) * args.mars_dot_area_scale,
                color=COLORS["Mars"],
                alpha=0.92,
                linewidths=0,
                clip_on=True,
                zorder=8 + idx * 0.001,
            )
            continue
        draw_disk_image(ax, mars_image, x_positions[idx], 0.5, mars_radius_x, mars_radius_y, zorder=8 + idx * 0.001)


def draw_labels(ax, args: argparse.Namespace) -> None:
    for i, name in enumerate(INNER):
        y = 0.965 - i * 0.043
        ax.scatter([0.026], [y], s=args.label_marker_size, color=COLORS[name], transform=ax.transAxes, linewidths=0)
        ax.text(
            0.047,
            y,
            name,
            transform=ax.transAxes,
            color=COLORS[name],
            fontsize=args.label_size,
            ha="left",
            va="center",
        )


def periastron_segment_view(
    inc: float,
    Omega: float,
    omega: float,
    sun_xyz_view: np.ndarray,
    basis: np.ndarray,
    box_half: float,
) -> np.ndarray:
    direction_view = (rotation_matrix(inc, Omega, omega) @ np.array([1.0, 0.0, 0.0])) @ basis
    end_xy = square_boundary_endpoint(sun_xyz_view[:2], direction_view[:2], box_half)
    return np.array([[sun_xyz_view[0], sun_xyz_view[1]], [end_xy[0], end_xy[1]]])


def render(args: argparse.Namespace) -> None:
    data = np.load(args.input)
    metadata_path = args.metadata or str(Path(args.input).with_name("metadata.json"))
    body_names = data["body_names"]
    element_names = data["element_names"]
    states = data["states_au_auyr"]
    elements = data["elements"]
    times = data["times_yr"]
    kick_times = data["kick_times_yr"]
    basis = load_view_basis(metadata_path, body_names, states)

    body_lookup = {str(name): i for i, name in enumerate(body_names)}
    elem_lookup = {str(name): i for i, name in enumerate(element_names)}
    inner_indices = [body_lookup[name] for name in INNER]
    years_myr = times / 1_000_000.0
    frame_indices = parse_indices(args, len(times))
    earth_idx = body_lookup["Earth"]
    mars_idx = body_lookup["Mars"]
    earth_a = elements[:, earth_idx, elem_lookup["a"]]
    mars_a = elements[:, mars_idx, elem_lookup["a"]]
    earth_a_plot = np.where((earth_a > 0.0) & (earth_a < args.history_ymax), earth_a, np.nan)
    mars_a_plot = np.where((mars_a > 0.0) & (mars_a < args.history_ymax), mars_a, np.nan)
    initial_a = {name: float(elements[0, body_lookup[name], elem_lookup["a"]]) for name in INNER}
    y_min = args.history_ymin
    y_max = args.history_ymax
    if "close_sample_times_yr" in data and "running_min_earth_mars_distance_au" in data:
        close_years_myr = data["close_sample_times_yr"] / 1_000_000.0
        close_running_min = data["running_min_earth_mars_distance_au"]
        close_time_diffs = np.diff(data["close_sample_times_yr"])
        if len(close_time_diffs) > 1 and np.allclose(close_time_diffs, np.median(close_time_diffs), rtol=0.01, atol=1e-9):
            close_log_label = f"sampled every {float(np.median(close_time_diffs)):g} yr"
        else:
            close_log_label = "REBOUND line-collision close log"
    else:
        earth_mars_distance = np.linalg.norm(states[:, earth_idx, :3] - states[:, mars_idx, :3], axis=1)
        close_years_myr = years_myr
        close_running_min = np.minimum.accumulate(earth_mars_distance)
        close_log_label = f"snapshot sampled every {float(np.median(np.diff(times))):g} yr"
    distance_max = args.encounter_ymax
    side_half = args.box_half / args.main_to_side_ratio
    record_mask = np.r_[False, np.diff(close_running_min) < 0.0]
    angular_event_times_myr = close_years_myr[record_mask]
    angular_event_distances = close_running_min[record_mask]
    angular_keep = angular_event_distances <= args.angular_max_distance
    angular_event_times_myr = angular_event_times_myr[angular_keep]
    angular_event_distances = angular_event_distances[angular_keep]
    moon_image = moon_texture(args.planet_image_px)
    mars_image = mars_texture(args.planet_image_px)

    with Path(metadata_path).open() as handle:
        metadata = json.load(handle)
    total_dv = metadata["kicks"]["total_impulse_m_per_s"]

    fig = plt.figure(figsize=(args.slide_width, args.slide_width / args.slide_aspect), dpi=args.dpi)
    fig.patch.set_facecolor("#05070a")
    gs = GridSpec(
        4,
        2,
        width_ratios=[args.main_column_ratio, args.side_column_ratio],
        height_ratios=[args.side_panel_ratio, args.history_ratio, args.angular_ratio, args.distance_ratio],
        hspace=0.16,
        wspace=0.085,
        left=0.035,
        right=0.985,
        top=0.965,
        bottom=0.095,
    )
    ax_main = fig.add_subplot(gs[:, 0])
    ax_side = fig.add_subplot(gs[0, 1])
    ax_hist = fig.add_subplot(gs[1, 1])
    ax_ang = fig.add_subplot(gs[2, 1])
    ax_dist = fig.add_subplot(gs[3, 1])

    writer = FFMpegWriter(
        fps=args.fps,
        codec="libx264",
        bitrate=args.bitrate,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart", "-crf", str(args.crf)],
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with writer.saving(fig, str(out), dpi=args.dpi):
        for frame_number, snap_idx in enumerate(frame_indices, start=1):
            ax_main.clear()
            ax_side.clear()
            ax_hist.clear()
            ax_ang.clear()
            ax_dist.clear()
            setup_main(ax_main, args.box_half, args)
            setup_side(ax_side, args.box_half, side_half, args)
            setup_history(ax_hist, years_myr, y_min, y_max, args)
            draw_angular_size_panel(
                ax_ang,
                fig,
                moon_image,
                mars_image,
                angular_event_times_myr,
                angular_event_distances,
                years_myr[snap_idx],
                years_myr[-1],
                args,
            )
            setup_distance_axis(ax_dist, years_myr, distance_max, args)

            sun_xyz = states[snap_idx, 0, :3] @ basis
            for name, body_idx in zip(INNER, inner_indices, strict=True):
                a = elements[snap_idx, body_idx, elem_lookup["a"]]
                e = elements[snap_idx, body_idx, elem_lookup["e"]]
                inc = elements[snap_idx, body_idx, elem_lookup["inc"]]
                Omega = elements[snap_idx, body_idx, elem_lookup["Omega"]]
                omega = elements[snap_idx, body_idx, elem_lookup["omega"]]
                period = elements[snap_idx, body_idx, elem_lookup["P"]]
                if np.isfinite(a) and np.isfinite(e) and 0.0 <= e < 0.98 and a > 0.0:
                    count = max(24, min(args.max_points, int(round(period * 365.25))))
                    points = orbit_points(a, e, inc, Omega, omega, count, states[snap_idx, 0, :3]) @ basis
                    ax_main.scatter(points[:, 0], points[:, 1], s=args.point_size, color=COLORS[name], alpha=0.82, linewidths=0)
                    ax_side.scatter(points[:, 0], points[:, 2], s=args.side_point_size, color=COLORS[name], alpha=0.74, linewidths=0)
                    ax_main.add_patch(
                        plt.Circle(
                            sun_xyz[:2],
                            initial_a[name],
                            fill=False,
                            color=COLORS[name],
                            linewidth=args.circle_width,
                            alpha=0.28,
                        )
                    )
                    segment = periastron_segment_view(inc, Omega, omega, sun_xyz, basis, args.box_half)
                    ax_main.plot(
                        segment[:, 0],
                        segment[:, 1],
                        color=COLORS[name],
                        linewidth=args.periastron_width,
                        alpha=args.periastron_alpha,
                        solid_capstyle="round",
                    )
                    peri = points[0]
                    ax_main.scatter(
                        [peri[0]],
                        [peri[1]],
                        s=args.periastron_marker_size,
                        color=COLORS[name],
                        edgecolors="#0b0e13",
                        linewidths=0.35,
                        zorder=30,
                    )
                    ax_side.scatter(
                        [peri[0]],
                        [peri[2]],
                        s=args.side_periastron_marker_size,
                        color=COLORS[name],
                        edgecolors="#0b0e13",
                        linewidths=0.25,
                        zorder=30,
                    )

            ax_main.scatter([sun_xyz[0]], [sun_xyz[1]], s=args.sun_size, color="#fff2b3", edgecolors="#fff9d6", linewidths=0.45, zorder=40)
            ax_side.axhline(0.0, color="#454b55", linewidth=args.invariable_plane_width, alpha=0.78)
            ax_side.scatter(
                [sun_xyz[0]],
                [sun_xyz[2]],
                s=args.sun_size * 0.45,
                color="#fff2b3",
                edgecolors="#fff9d6",
                linewidths=0.30,
                zorder=40,
            )
            draw_labels(ax_main, args)
            ax_main.text(
                0.982,
                0.976,
                f"T+{int(times[snap_idx]):010,d} yr",
                transform=ax_main.transAxes,
                color="#d7dbe2",
                fontsize=args.odometer_size,
                family="monospace",
                ha="right",
                va="top",
                bbox={"boxstyle": "round,pad=0.22,rounding_size=0.02", "facecolor": "#0b0e13", "edgecolor": "#2d333d", "linewidth": 0.45, "alpha": 0.84},
            )

            for name, series in (("Earth", earth_a_plot), ("Mars", mars_a_plot)):
                ax_hist.plot(years_myr, series, color=COLORS[name], linewidth=args.history_context_width, alpha=0.25)
                ax_hist.plot(
                    years_myr[: snap_idx + 1],
                    series[: snap_idx + 1],
                    color=COLORS[name],
                    linewidth=args.history_active_width,
                    alpha=0.96,
                )
                if np.isfinite(series[snap_idx]):
                    ax_hist.scatter([years_myr[snap_idx]], [series[snap_idx]], s=args.history_marker_size, color=COLORS[name], linewidths=0, zorder=10)
                ax_hist.axhline(series[0], color=COLORS[name], linewidth=args.reference_line_width, alpha=0.58, linestyle=(0, (4, 4)))
            ax_hist.axhline(1.4, color="#d7dbe2", linewidth=args.reference_line_width, alpha=0.38)
            ax_hist.axvline(years_myr[snap_idx], color="#d7dbe2", linewidth=args.cursor_line_width, alpha=0.40)
            close_idx = int(np.searchsorted(close_years_myr, years_myr[snap_idx], side="right") - 1)
            close_idx = max(0, min(close_idx, len(close_years_myr) - 1))
            current_min_distance = float(close_running_min[close_idx])
            ax_dist.plot(close_years_myr, close_running_min, color="#c4cedd", linewidth=args.history_context_width, alpha=0.25)
            ax_dist.plot(
                close_years_myr[: close_idx + 1],
                close_running_min[: close_idx + 1],
                color="#c4cedd",
                linewidth=args.distance_active_width,
                alpha=0.92,
            )
            ax_dist.scatter(
                [years_myr[snap_idx]],
                [current_min_distance],
                s=args.distance_marker_size,
                color="#c4cedd",
                linewidths=0,
                zorder=10,
            )
            kicks_so_far = int(np.searchsorted(kick_times, times[snap_idx], side="right"))
            ax_dist.text(
                0.018,
                0.86,
                f"kicks {kicks_so_far}/{len(kick_times)}   min d(E,M) {current_min_distance:.3f} AU",
                transform=ax_dist.transAxes,
                color="#aeb6c2",
                fontsize=args.panel_note_size,
                ha="left",
                va="top",
            )
            ax_dist.text(
                0.018,
                0.70,
                f"total impulse {total_dv/1000:.2f} km/s",
                transform=ax_dist.transAxes,
                color="#aeb6c2",
                fontsize=args.panel_note_size,
                ha="left",
                va="top",
            )
            writer.grab_frame(facecolor=fig.get_facecolor())
            if args.progress_every and (
                frame_number == 1 or frame_number == len(frame_indices) or frame_number % args.progress_every == 0
            ):
                print(f"frame {frame_number:4d}/{len(frame_indices)} snapshot {snap_idx}", flush=True)

    plt.close(fig)
    print(f"Wrote {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="outputs/earth_kick_10myr/earth_kick_snapshots.npz")
    parser.add_argument("--metadata")
    parser.add_argument("--out", default="outputs/earth_kick_10myr/earth_kick_10myr.mp4")
    parser.add_argument("--start-snapshot", type=int, default=0)
    parser.add_argument("--end-snapshot", type=int, default=1000)
    parser.add_argument("--limit-frames", type=int)
    parser.add_argument("--fps", type=float, default=33.0)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--slide-width", type=float, default=12.8)
    parser.add_argument("--slide-aspect", type=float, default=16.0 / 9.0)
    parser.add_argument("--bitrate", type=int, default=20_000)
    parser.add_argument("--crf", type=int, default=14)
    parser.add_argument("--box-half", type=float, default=2.25)
    parser.add_argument("--history-ymin", type=float, default=0.90)
    parser.add_argument("--history-ymax", type=float, default=1.78)
    parser.add_argument("--encounter-ymax", type=float, default=0.4)
    parser.add_argument("--main-to-side-ratio", type=float, default=6.0)
    parser.add_argument("--main-column-ratio", type=float, default=1.20)
    parser.add_argument("--side-column-ratio", type=float, default=0.80)
    parser.add_argument("--side-panel-ratio", type=float, default=0.88)
    parser.add_argument("--history-ratio", type=float, default=1.02)
    parser.add_argument("--angular-ratio", type=float, default=0.64)
    parser.add_argument("--distance-ratio", type=float, default=0.94)
    parser.add_argument("--planet-image-px", type=int, default=192)
    parser.add_argument("--moon-fill-fraction", type=float, default=0.70)
    parser.add_argument("--moon-x", type=float, default=0.065)
    parser.add_argument("--min-mars-image-px", type=float, default=8.0)
    parser.add_argument("--mars-dot-area-scale", type=float, default=1.0)
    parser.add_argument("--angular-max-distance", type=float, default=0.4)
    parser.add_argument("--max-points", type=int, default=1200)
    parser.add_argument("--point-size", type=float, default=3.2)
    parser.add_argument("--side-point-size", type=float, default=2.0)
    parser.add_argument("--periastron-marker-size", type=float, default=32.0)
    parser.add_argument("--side-periastron-marker-size", type=float, default=17.0)
    parser.add_argument("--sun-size", type=float, default=30.0)
    parser.add_argument("--circle-width", type=float, default=0.78)
    parser.add_argument("--periastron-width", type=float, default=0.42)
    parser.add_argument("--periastron-alpha", type=float, default=0.22)
    parser.add_argument("--invariable-plane-width", type=float, default=1.25)
    parser.add_argument("--history-context-width", type=float, default=1.00)
    parser.add_argument("--history-active-width", type=float, default=2.45)
    parser.add_argument("--distance-active-width", type=float, default=2.20)
    parser.add_argument("--reference-line-width", type=float, default=1.10)
    parser.add_argument("--cursor-line-width", type=float, default=1.00)
    parser.add_argument("--grid-width", type=float, default=0.70)
    parser.add_argument("--spine-width", type=float, default=0.90)
    parser.add_argument("--tick-width", type=float, default=0.75)
    parser.add_argument("--history-marker-size", type=float, default=26.0)
    parser.add_argument("--distance-marker-size", type=float, default=22.0)
    parser.add_argument("--label-size", type=float, default=11.0)
    parser.add_argument("--label-marker-size", type=float, default=12.0)
    parser.add_argument("--odometer-size", type=float, default=12.5)
    parser.add_argument("--axis-label-size", type=float, default=9.5)
    parser.add_argument("--tick-size", type=float, default=8.0)
    parser.add_argument("--panel-note-size", type=float, default=8.5)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    render(parse_args())
