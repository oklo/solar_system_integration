#!/usr/bin/env python3
"""Render one barycentric MVEM orbit frame from saved animation primitives."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "Mercury": "#b7b1a7",
    "Venus": "#d9a441",
    "Earth": "#4f8bd6",
    "Mars": "#c65f3f",
}


def render(args: argparse.Namespace) -> None:
    data = np.load(args.input)
    names = [str(name) for name in data["body_names"]]
    points = data["ellipse_points_au"][args.snapshot_index]
    counts = data["ellipse_counts"][args.snapshot_index]
    circles = data["semi_major_circle_radii_au"][args.snapshot_index]
    segments = data["periastron_segments_au"][args.snapshot_index]
    half = float(data["box_side_au"]) / 2.0
    sun_xy = segments[0, 0, :2]

    fig, ax = plt.subplots(figsize=(args.size_in, args.size_in), dpi=args.dpi)
    fig.patch.set_facecolor("#05070a")
    ax.set_facecolor("#05070a")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#454b55")
        spine.set_linewidth(0.45)

    for name, radius in zip(names, circles, strict=True):
        ax.add_patch(
            plt.Circle(
                sun_xy,
                float(radius),
                fill=False,
                color=COLORS[name],
                linewidth=0.22,
                alpha=0.33,
            )
        )

    for name, segment in zip(names, segments, strict=True):
        ax.plot(
            segment[:, 0],
            segment[:, 1],
            color=COLORS[name],
            linewidth=0.28,
            alpha=0.58,
            solid_capstyle="round",
        )

    for name, point_cloud, count in zip(names, points, counts, strict=True):
        xy = point_cloud[: int(count), :2]
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=args.point_size,
            color=COLORS[name],
            alpha=0.86,
            linewidths=0,
        )
        ax.scatter(
            xy[0, 0],
            xy[0, 1],
            s=args.point_size * 3.2,
            color=COLORS[name],
            alpha=0.98,
            linewidths=0,
        )

    ax.scatter(
        [sun_xy[0]],
        [sun_xy[1]],
        s=args.sun_size,
        color="#fff2b3",
        edgecolors="#fff9d6",
        linewidths=0.45,
        zorder=20,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Wrote {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="outputs/full_5myr/animation_primitives_inner.npz")
    parser.add_argument("--snapshot-index", type=int, default=0)
    parser.add_argument("--out", default="outputs/preview_frame.png")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--size-in", type=float, default=8.0)
    parser.add_argument("--point-size", type=float, default=1.2)
    parser.add_argument("--sun-size", type=float, default=18.0)
    return parser.parse_args()


if __name__ == "__main__":
    render(parse_args())
