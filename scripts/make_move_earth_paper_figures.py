#!/usr/bin/env python3
"""Build light-background figures for the move-the-Earth paper draft."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.render_earth_kick_animation import mars_texture

AU_KM = 149_597_870.7
EARTH_RADIUS_KM = 6_371.0
MARS_RADIUS_KM = 3_389.5
SUMMED_RADII_KM = EARTH_RADIUS_KM + MARS_RADIUS_KM

COLORS = {
    "Earth": "#1f77b4",
    "Mars": "#c65f3a",
    "Mercury": "#8f877a",
    "Venus": "#c99a2e",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    return path


def load_body_elements(path: Path, body: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    years: list[float] = []
    semimajor: list[float] = []
    eccentricity: list[float] = []
    for row in read_rows(path):
        if row["body"] != body:
            continue
        years.append(float(row["t_year"]) / 1.0e6)
        semimajor.append(float(row["a"]))
        eccentricity.append(float(row["e"]))
    return np.array(years), np.array(semimajor), np.array(eccentricity)


def earth_texture(size: int = 384) -> np.ndarray:
    yy, xx = np.mgrid[-1.0:1.0:complex(size), -1.0:1.0:complex(size)]
    rr = np.sqrt(xx * xx + yy * yy)
    mask = rr <= 1.0
    light = np.clip(1.04 - 0.24 * xx + 0.10 * yy, 0.25, 1.08)
    limb = np.clip(1.0 - 0.42 * rr**2, 0.0, 1.0)
    ocean = np.dstack([0.16 * light * limb, 0.43 * light * limb, 0.78 * light * limb])

    continents = np.zeros_like(rr)
    for cx, cy, rx, ry, angle, strength in [
        (-0.34, 0.26, 0.20, 0.35, -24.0, 0.70),
        (-0.18, -0.24, 0.16, 0.28, 8.0, 0.65),
        (0.30, 0.14, 0.33, 0.24, 10.0, 0.70),
        (0.40, -0.28, 0.22, 0.16, -8.0, 0.60),
    ]:
        theta = np.deg2rad(angle)
        xp = (xx - cx) * np.cos(theta) + (yy - cy) * np.sin(theta)
        yp = -(xx - cx) * np.sin(theta) + (yy - cy) * np.cos(theta)
        continents += strength * np.exp(-2.8 * ((xp / rx) ** 2 + (yp / ry) ** 2))
    land = np.dstack([0.45 * light * limb, 0.62 * light * limb, 0.30 * light * limb])
    clouds = 0.18 * (
        np.sin(13.0 * xx + 8.0 * yy) > 0.72
    ) * np.exp(-0.8 * rr**2)
    rgb = np.where(continents[..., None] > 0.20, land, ocean)
    rgb = np.clip(rgb + clouds[..., None], 0.0, 1.0)
    alpha = np.where(mask, 1.0, 0.0) * np.clip((1.0 - rr) / 0.018, 0.0, 1.0)
    return np.dstack([rgb, alpha])


def save_both(fig: plt.Figure, path_base: Path) -> None:
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".png"), dpi=240, bbox_inches="tight")


def figure_seed_sweep(source_root: Path, outdir: Path) -> None:
    sweep_dir = require(source_root / "earth_kick_seed_sweep")
    summary_rows = read_rows(require(sweep_dir / "summary.csv"))

    fig, axes = plt.subplots(2, 1, figsize=(7.1, 5.2), sharex=True)
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("#fbfaf7")
        ax.grid(color="#d7d2c8", linewidth=0.6, alpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    seeds = [row["seed"] for row in summary_rows]
    cmap = plt.get_cmap("viridis")
    seed_colors = {seed: cmap(0.08 + 0.84 * i / max(len(seeds) - 1, 1)) for i, seed in enumerate(seeds)}
    for row in summary_rows:
        seed = row["seed"]
        elements_path = require(sweep_dir / f"seed_{seed}" / "osculating_elements.csv")
        for body, ax in [("Earth", axes[0]), ("Mars", axes[1])]:
            t_myr, a_au, _ = load_body_elements(elements_path, body)
            ax.plot(t_myr, a_au, color=seed_colors[seed], lw=1.4, alpha=0.88)
            ax.plot(t_myr[-1], a_au[-1], "o", ms=3.0, color=seed_colors[seed], zorder=4)
    axes[0].axhline(1.000016, color=COLORS["Earth"], lw=1.0, ls="--", alpha=0.7)
    axes[0].axhline(1.4, color="#4d4d4d", lw=0.9, ls=":", alpha=0.8)
    axes[1].axhline(1.523597, color=COLORS["Mars"], lw=1.0, ls="--", alpha=0.7)
    axes[0].set_ylabel("Earth $a$ [AU]")
    axes[1].set_ylabel("Mars $a$ [AU]")
    axes[1].set_xlabel("time after 2026 July 22 [Myr]")
    axes[0].set_ylim(0.94, 1.50)
    axes[1].set_ylim(0.8, 2.9)
    axes[0].set_title("Five 10 Myr kicked-Earth integrations")

    best = min(summary_rows, key=lambda row: float(row["min_earth_mars_distance_au"]))
    text = (
        f"closest in sweep: seed {best['seed']}\n"
        f"{float(best['min_earth_mars_distance_au']) * AU_KM:,.0f} km at "
        f"{float(best['min_earth_mars_time_year']) / 1.0e6:.3f} Myr"
    )
    axes[1].text(
        0.02,
        0.95,
        text,
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.28", "fc": "white", "ec": "#b9b0a3", "lw": 0.7},
    )

    handles = [
        plt.Line2D([0], [0], color=seed_colors[seed], lw=1.7, label=seed)
        for seed in seeds
    ]
    axes[0].legend(
        handles=handles,
        title="seed",
        loc="upper left",
        ncol=5,
        fontsize=7.0,
        title_fontsize=7.5,
        frameon=False,
    )
    fig.tight_layout()
    save_both(fig, outdir / "move_earth_seed_sweep_semimajor_axes")
    plt.close(fig)


def add_disk(ax: plt.Axes, texture: np.ndarray, xy: tuple[float, float], radius_data: float) -> None:
    x, y = xy
    ax.imshow(
        texture,
        extent=(x - radius_data, x + radius_data, y - radius_data, y + radius_data),
        origin="lower",
        interpolation="lanczos",
        zorder=3,
    )
    # Transparent image edges can make the exact physical radius hard to read in print.
    circle = plt.Circle(xy, radius_data, fill=False, lw=0.85, color="#2f2f2f", alpha=0.55, zorder=4)
    ax.add_patch(circle)


def figure_close_encounter(source_root: Path, outdir: Path) -> None:
    campaign_dir = require(source_root / "earth_kick_refined_encounter_campaign_50kkm_from_20260908")
    rows = [
        row
        for row in read_rows(require(campaign_dir / "refined_campaign_summary.csv"))
        if row.get("refined") == "True" and row.get("refined_distance_km")
    ]
    earlier = source_root / "refined_encounters" / "seed_20260906" / "refined_encounter.csv"
    if earlier.exists():
        for row in read_rows(earlier):
            rows.append(
                {
                    "seed": row["seed"],
                    "refined_distance_km": row["distance_km"],
                    "earth_mars_radii_sums": row["earth_mars_radii_sums"],
                }
            )
    rows = [row for row in rows if float(row["earth_mars_radii_sums"]) < 20.0]
    rows.sort(key=lambda row: float(row["refined_distance_km"]))

    found = json.loads(require(campaign_dir / "FOUND_TARGET.json").read_text())
    d_km = float(found["refined_distance_km"])
    clearance_km = float(found["surface_clearance_km"])
    t_myr = float(found["refined_t_year"]) / 1.0e6
    d_label = f"{d_km:,.0f}".replace(",", "{,}")

    fig = plt.figure(figsize=(7.1, 3.45), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[0.9, 1.35], wspace=0.24)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    for ax in [ax0, ax1]:
        ax.set_facecolor("#fbfaf7")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    seed_suffixes = np.array([int(row["seed"]) - 20_260_900 for row in rows])
    seed_positions = np.arange(len(rows))
    radii_sums = np.array([float(row["earth_mars_radii_sums"]) for row in rows])
    distances = np.array([float(row["refined_distance_km"]) for row in rows])
    colors = np.where(seed_suffixes == int(found["seed"]) - 20_260_900, "#b2182b", "#4f7ea8")
    ax0.scatter(seed_positions, radii_sums, s=42, c=colors, edgecolor="white", linewidth=0.8, zorder=3)
    for position, row in zip(seed_positions, rows):
        if row["seed"] in {"20260917", "20260906", "20260915"}:
            ax0.annotate(
                row["seed"],
                (position, float(row["earth_mars_radii_sums"])),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7.5,
            )
    ax0.axhline(1.0, color="#222222", lw=1.0, ls="-", alpha=0.8, label="contact")
    ax0.axhline(50_000.0 / SUMMED_RADII_KM, color="#777777", lw=0.9, ls=":", label="50,000 km")
    ax0.set_yscale("log")
    ax0.set_xticks(seed_positions)
    ax0.set_xticklabels([f"{seed:02d}" for seed in seed_suffixes])
    ax0.set_xlabel("search seed suffix, sorted by distance")
    ax0.set_ylabel("minimum distance / $(R_\\oplus+R_{\\rm Mars})$")
    ax0.set_title("Refined Earth--Mars candidates", fontsize=10.0)
    ax0.grid(color="#d7d2c8", linewidth=0.6, alpha=0.8, which="both")
    ax0.legend(loc="upper right", fontsize=7.5, frameon=False)
    ax0.text(
        0.03,
        0.04,
        f"best refined: {distances.min():,.0f} km",
        transform=ax0.transAxes,
        fontsize=8.3,
        bbox={"boxstyle": "round,pad=0.26", "fc": "white", "ec": "#b9b0a3", "lw": 0.7},
    )

    earth_center = (0.0, 0.0)
    mars_center = (d_km, 0.0)
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlim(-9_000, d_km + 6_000)
    ax1.set_ylim(-8_300, 8_300)
    ax1.set_xticks([-5_000, 0, 5_000, 10_000, 15_000, 20_000])
    ax1.set_yticks([-5_000, 0, 5_000])
    ax1.grid(color="#d7d2c8", linewidth=0.6, alpha=0.75)
    ax1.set_xlabel("km")
    ax1.set_ylabel("km")
    ax1.set_title("Seed 20260917 geometry at closest approach", fontsize=10.0)

    add_disk(ax1, earth_texture(), earth_center, EARTH_RADIUS_KM)
    add_disk(ax1, mars_texture(), mars_center, MARS_RADIUS_KM)
    ax1.plot([EARTH_RADIUS_KM, d_km - MARS_RADIUS_KM], [0, 0], color="#b2182b", lw=1.6)
    ax1.annotate(
        f"{clearance_km:,.0f} km surface clearance",
        xy=((EARTH_RADIUS_KM + d_km - MARS_RADIUS_KM) / 2, 0),
        xytext=(0, -34),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=8.2,
        arrowprops={"arrowstyle": "-", "color": "#b2182b", "lw": 0.8},
    )
    ax1.text(
        0.03,
        0.96,
        f"$t={t_myr:.6f}$ Myr\n$d={d_label}\\,\\mathrm{{km}}$\n"
        f"$d/(R_\\oplus+R_M)={d_km / SUMMED_RADII_KM:.2f}$",
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
        bbox={"boxstyle": "round,pad=0.26", "fc": "white", "ec": "#b9b0a3", "lw": 0.7},
    )
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.18, top=0.88, wspace=0.28)
    save_both(fig, outdir / "move_earth_refined_encounters")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/Users/greglaughlin/Projects/solar_system/outputs"),
        help="Directory containing the original long-run output products.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=PROJECT_ROOT / "papers" / "figures",
        help="Output directory for paper figures.",
    )
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 9.0,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure_seed_sweep(args.source_root, args.outdir)
    figure_close_encounter(args.source_root, args.outdir)


if __name__ == "__main__":
    main()
