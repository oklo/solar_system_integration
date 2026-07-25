#!/usr/bin/env python3
"""Integrate the Solar System with REBOUND and prepare orbit-animation data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rebound
from rebound.horizons import HORIZONS_MASS_DATA
from skyfield.api import Loader


AU_KM = 149_597_870.7
SECONDS_PER_DAY = 86_400.0
JULIAN_YEAR_DAYS = 365.25
JULIAN_YEAR_SECONDS = JULIAN_YEAR_DAYS * SECONDS_PER_DAY
GM_SUN_KM3_S2 = 1.3271244004127942e11
G_AU3_MSUN_YR2 = GM_SUN_KM3_S2 * JULIAN_YEAR_SECONDS**2 / AU_KM**3

START_UTC = (2026, 7, 22, 14, 0, 0)
BOX_SIDE_AU = 3.2
BOX_HALF_AU = BOX_SIDE_AU / 2.0


@dataclass(frozen=True)
class BodySpec:
    name: str
    skyfield_key: str
    horizons_gm_id: int
    render_inner: bool = False


BODIES = [
    BodySpec("Sun", "sun", 10),
    BodySpec("Mercury", "mercury barycenter", 1, True),
    BodySpec("Venus", "venus barycenter", 2, True),
    BodySpec("Earth", "earth barycenter", 3, True),
    BodySpec("Mars", "mars barycenter", 4, True),
    BodySpec("Jupiter", "jupiter barycenter", 5),
    BodySpec("Saturn", "saturn barycenter", 6),
    BodySpec("Uranus", "uranus barycenter", 7),
    BodySpec("Neptune", "neptune barycenter", 8),
    BodySpec("Pluto", "pluto barycenter", 9),
]


ELEMENT_NAMES = ("a", "e", "inc", "Omega", "omega", "f", "M", "P", "q", "d")


def parse_horizons_gm_table() -> dict[int, float]:
    pattern = re.compile(r"BODY(\d+)_GM\s*=\s*\(\s*([.DE+\-0-9]+)\s*\)")
    out: dict[int, float] = {}
    for match in pattern.finditer(HORIZONS_MASS_DATA):
        out[int(match.group(1))] = float(match.group(2).replace("D", "E"))
    return out


def build_simulation(kernel_name: str, data_dir: Path, dt_days: float) -> tuple[rebound.Simulation, dict]:
    load = Loader(str(data_dir))
    ephemeris = load(kernel_name)
    ts = load.timescale()
    t = ts.utc(*START_UTC)
    gm = parse_horizons_gm_table()

    sim = rebound.Simulation()
    sim.G = G_AU3_MSUN_YR2
    sim.integrator = "whfast"
    sim.dt = dt_days / JULIAN_YEAR_DAYS
    sim.integrator.corrector = 11
    sim.integrator.safe_mode = 0

    initial_state = {}
    for spec in BODIES:
        state = ephemeris[spec.skyfield_key].at(t)
        pos = np.asarray(state.position.au, dtype=float)
        vel = np.asarray(state.velocity.au_per_d, dtype=float) * JULIAN_YEAR_DAYS
        mass = gm[spec.horizons_gm_id] / GM_SUN_KM3_S2
        sim.add(m=mass, x=pos[0], y=pos[1], z=pos[2], vx=vel[0], vy=vel[1], vz=vel[2])
        initial_state[spec.name] = {
            "skyfield_key": spec.skyfield_key,
            "mass_msun": mass,
            "position_au": pos.tolist(),
            "velocity_au_per_year": vel.tolist(),
        }

    sim.move_to_com()
    return sim, initial_state


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


def orbit_points_from_elements(orbit, sun_xyz: np.ndarray) -> tuple[np.ndarray, int]:
    period_days = max(float(orbit.P) * JULIAN_YEAR_DAYS, 1.0)
    count = max(8, int(round(period_days)))
    mean_anomaly = np.arange(count, dtype=float) * (2.0 * np.pi / count)
    E = solve_kepler_elliptic(mean_anomaly, float(orbit.e))
    cos_E = np.cos(E)
    sin_E = np.sin(E)
    x_orb = float(orbit.a) * (cos_E - float(orbit.e))
    y_orb = float(orbit.a) * math.sqrt(max(0.0, 1.0 - float(orbit.e) ** 2)) * sin_E
    local = np.column_stack([x_orb, y_orb, np.zeros_like(x_orb)])
    points = local @ rotation_matrix(float(orbit.inc), float(orbit.Omega), float(orbit.omega)).T
    return points + sun_xyz, count


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


def periastron_segment(orbit, sun_xyz: np.ndarray, half_width: float) -> np.ndarray:
    R = rotation_matrix(float(orbit.inc), float(orbit.Omega), float(orbit.omega))
    direction = R @ np.array([1.0, 0.0, 0.0])
    end_xy = square_boundary_endpoint(sun_xyz[:2], direction[:2], half_width)
    end = np.array([end_xy[0], end_xy[1], sun_xyz[2]], dtype=float)
    return np.vstack([sun_xyz, end])


def particle_state_relative_to_com(sim: rebound.Simulation) -> np.ndarray:
    com = sim.com()
    state = np.empty((len(BODIES), 6), dtype=float)
    for i, particle in enumerate(sim.particles):
        state[i] = [
            particle.x - com.x,
            particle.y - com.y,
            particle.z - com.z,
            particle.vx - com.vx,
            particle.vy - com.vy,
            particle.vz - com.vz,
        ]
    return state


def snapshot_orbits(sim: rebound.Simulation) -> tuple[np.ndarray, list]:
    elements = np.full((len(BODIES), len(ELEMENT_NAMES)), np.nan, dtype=float)
    orbits = [None] * len(BODIES)
    sun = sim.particles[0]
    for i in range(1, len(BODIES)):
        orbit = sim.particles[i].orbit(primary=sun, G=sim.G)
        orbits[i] = orbit
        values = {
            "a": orbit.a,
            "e": orbit.e,
            "inc": orbit.inc,
            "Omega": orbit.Omega,
            "omega": orbit.omega,
            "f": orbit.f,
            "M": orbit.M,
            "P": orbit.P,
            "q": orbit.a * (1.0 - orbit.e),
            "d": orbit.d,
        }
        elements[i] = [float(values[name]) for name in ELEMENT_NAMES]
    return elements, orbits


def write_elements_csv(path: Path, times_yr: np.ndarray, elements: np.ndarray) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["snapshot_index", "t_year", "body", *ELEMENT_NAMES])
        for snap_idx, t_year in enumerate(times_yr):
            for body_idx, spec in enumerate(BODIES):
                if spec.name == "Sun":
                    continue
                writer.writerow([snap_idx, f"{t_year:.12g}", spec.name, *elements[snap_idx, body_idx]])


def make_metadata(args: argparse.Namespace, initial_state: dict, snapshot_count: int) -> dict:
    return {
        "start_time": {
            "central_daylight_time": "2026-07-22T09:00:00-05:00",
            "utc": "2026-07-22T14:00:00Z",
        },
        "integration": {
            "rebound_version": rebound.__version__,
            "integrator": "whfast",
            "whfast_corrector": 11,
            "dt_days": args.dt_days,
            "years": args.years,
            "cadence_years": args.cadence_years,
            "snapshot_count_including_endpoints": snapshot_count,
            "G_AU3_Msun_yr2": G_AU3_MSUN_YR2,
        },
        "ephemeris": {
            "kernel": args.kernel,
            "source": "JPL SPK loaded through Skyfield",
        },
        "view": {
            "center": "instantaneous integrated system barycenter",
            "box_side_au": BOX_SIDE_AU,
            "inner_rendered_bodies": [spec.name for spec in BODIES if spec.render_inner],
            "ellipse_sampling": "one-day mean-anomaly cadence converted to true anomaly; first point is perihelion",
            "animation_note": "A 5 Myr run at 5000 yr cadence writes 1001 snapshots. Use snapshots 1..1000 for exactly 1000 forward-time frames.",
        },
        "initial_state_after_ephemeris_before_com_shift": initial_state,
    }


def run(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    data_dir = Path(args.data_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    cadence_ratio = args.years / args.cadence_years
    if not math.isclose(cadence_ratio, round(cadence_ratio), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("--years must be an integer multiple of --cadence-years")
    interval_count = int(round(cadence_ratio))
    times_yr = np.arange(interval_count + 1, dtype=float) * args.cadence_years

    sim, initial_state = build_simulation(args.kernel, data_dir, args.dt_days)
    body_names = np.array([spec.name for spec in BODIES])
    inner_indices = [i for i, spec in enumerate(BODIES) if spec.render_inner]
    inner_names = np.array([BODIES[i].name for i in inner_indices])

    states = np.empty((len(times_yr), len(BODIES), 6), dtype=float)
    elements = np.empty((len(times_yr), len(BODIES), len(ELEMENT_NAMES)), dtype=float)
    circle_radii = np.empty((len(times_yr), len(inner_indices)), dtype=float)
    periastron_segments = np.empty((len(times_yr), len(inner_indices), 2, 3), dtype=float)

    max_points = int(round(JULIAN_YEAR_DAYS * math.sqrt(1.75**3))) + 16
    ellipse_points = np.full((len(times_yr), len(inner_indices), max_points, 3), np.nan, dtype=np.float32)
    ellipse_counts = np.empty((len(times_yr), len(inner_indices)), dtype=np.int32)

    for snap_idx, target_t in enumerate(times_yr):
        if snap_idx:
            sim.integrate(float(target_t), exact_finish_time=1)

        states[snap_idx] = particle_state_relative_to_com(sim)
        elements[snap_idx], orbits = snapshot_orbits(sim)
        sun_xyz = states[snap_idx, 0, :3]

        for j, body_idx in enumerate(inner_indices):
            orbit = orbits[body_idx]
            points, count = orbit_points_from_elements(orbit, sun_xyz)
            if count > max_points:
                raise RuntimeError(f"{BODIES[body_idx].name} generated {count} points; increase max_points")
            ellipse_points[snap_idx, j, :count] = points.astype(np.float32)
            ellipse_counts[snap_idx, j] = count
            circle_radii[snap_idx, j] = orbit.a
            periastron_segments[snap_idx, j] = periastron_segment(orbit, sun_xyz, BOX_HALF_AU)

        if args.progress_every and (snap_idx % args.progress_every == 0 or snap_idx == len(times_yr) - 1):
            print(f"snapshot {snap_idx:5d}/{len(times_yr) - 1}: t = {target_t:.0f} yr", flush=True)

    metadata = make_metadata(args, initial_state, len(times_yr))
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_elements_csv(outdir / "osculating_elements.csv", times_yr, elements)
    np.savez_compressed(
        outdir / "snapshots.npz",
        times_yr=times_yr,
        body_names=body_names,
        element_names=np.array(ELEMENT_NAMES),
        states_au_auyr=states,
        elements=elements,
    )
    np.savez_compressed(
        outdir / "animation_primitives_inner.npz",
        times_yr=times_yr,
        body_names=inner_names,
        ellipse_points_au=ellipse_points,
        ellipse_counts=ellipse_counts,
        semi_major_circle_radii_au=circle_radii,
        periastron_segments_au=periastron_segments,
        box_side_au=np.array(BOX_SIDE_AU),
    )
    print(f"Wrote {len(times_yr)} snapshots to {outdir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=float, default=5_000_000.0)
    parser.add_argument("--cadence-years", type=float, default=5_000.0)
    parser.add_argument("--dt-days", type=float, default=8.0)
    parser.add_argument("--kernel", default="de440s.bsp")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--outdir", default="outputs/full_5myr")
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
