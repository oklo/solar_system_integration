#!/usr/bin/env python3
"""Refine the Earth-Mars minimum distance near a flagged encounter."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.integrate_solar_system import build_simulation
from scripts.run_earth_kick_experiment import (
    EARTH_INDEX,
    MARS_INDEX,
    apply_earth_kick,
    configure_whfast,
    heliocentric_orbit,
    make_kick_times,
)


AU_KM = 149_597_870.7
EARTH_RADIUS_KM = 6_371.0
MARS_RADIUS_KM = 3_389.5
PHYSICAL_DISTANCE_KM = EARTH_RADIUS_KM + MARS_RADIUS_KM


def earth_mars_distance_au(sim) -> float:
    earth = sim.particles[EARTH_INDEX]
    mars = sim.particles[MARS_INDEX]
    dx = earth.x - mars.x
    dy = earth.y - mars.y
    dz = earth.z - mars.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def replay_to(args: argparse.Namespace, t_start: float):
    sim, initial_state = build_simulation(args.kernel, Path(args.data_dir), args.dt_days)
    configure_whfast(sim, args.coordinates, args.corrector)
    initial_earth = heliocentric_orbit(sim, EARTH_INDEX)
    mu = sim.G * (sim.particles[0].m + sim.particles[EARTH_INDEX].m)
    total_delta_eps = 0.5 * mu * (1.0 / initial_earth.a - 1.0 / args.target_a)
    delta_eps_per_kick = total_delta_eps / args.kicks
    kick_times = make_kick_times(args.kick_years, args.kicks, args.seed)

    applied = 0
    cumulative_dv = 0.0
    for kick_time in kick_times:
        if kick_time >= t_start:
            break
        sim.integrate(float(kick_time), exact_finish_time=1)
        kick = apply_earth_kick(sim, delta_eps_per_kick, args.coordinates, args.corrector)
        cumulative_dv += kick["dv_m_per_s"]
        applied += 1
        if args.progress_every_kicks and applied % args.progress_every_kicks == 0:
            print(f"seed {args.seed}: replayed {applied}/{args.kicks} kicks", flush=True)

    sim.integrate(t_start, exact_finish_time=1)
    sim.synchronize()
    return sim, kick_times, applied, cumulative_dv, initial_state


def nearest_kick_gap(kick_times: np.ndarray, t0: float, t1: float) -> tuple[int, float | None]:
    inside = np.where((kick_times >= t0) & (kick_times <= t1))[0]
    if len(inside):
        return int(len(inside)), float(kick_times[inside[0]])
    nearest = float(kick_times[np.argmin(np.abs(kick_times - 0.5 * (t0 + t1)))])
    return 0, nearest


def distance_from_base(base_sim, t_year: float) -> float:
    sim = base_sim.copy()
    sim.integrator = "ias15"
    sim.integrate(t_year, exact_finish_time=1)
    sim.synchronize()
    return earth_mars_distance_au(sim)


def golden_minimize(base_sim, a: float, b: float, tol_years: float) -> tuple[float, float, int]:
    invphi = (math.sqrt(5.0) - 1.0) / 2.0
    invphi2 = (3.0 - math.sqrt(5.0)) / 2.0
    h = b - a
    if h <= tol_years:
        mid = 0.5 * (a + b)
        return mid, distance_from_base(base_sim, mid), 1

    n = int(math.ceil(math.log(tol_years / h) / math.log(invphi)))
    c = a + invphi2 * h
    d = a + invphi * h
    yc = distance_from_base(base_sim, c)
    yd = distance_from_base(base_sim, d)
    evals = 2
    for _ in range(n - 1):
        if yc < yd:
            b = d
            d = c
            yd = yc
            h = invphi * h
            c = a + invphi2 * h
            yc = distance_from_base(base_sim, c)
        else:
            a = c
            c = d
            yc = yd
            h = invphi * h
            d = a + invphi * h
            yd = distance_from_base(base_sim, d)
        evals += 1
    if yc < yd:
        return c, yc, evals
    return d, yd, evals


def write_outputs(outdir: Path, payload: dict) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "refined_encounter.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    csv_path = outdir / "refined_encounter.csv"
    with csv_path.open("w", newline="") as handle:
        fieldnames = [
            "seed",
            "t_min_year",
            "distance_au",
            "distance_km",
            "surface_clearance_km",
            "earth_mars_radii_sums",
            "window_start_year",
            "window_end_year",
            "coarse_samples",
            "golden_evaluations",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({key: payload[key] for key in fieldnames})


def run(args: argparse.Namespace) -> None:
    t_start = args.event_time_year - args.window_years
    t_end = args.event_time_year + args.window_years
    if t_start < 0.0:
        raise ValueError("Refinement window starts before t=0")

    base_sim, kick_times, applied, cumulative_dv, initial_state = replay_to(args, t_start)
    inside_count, nearest_kick = nearest_kick_gap(kick_times, t_start, t_end)
    if inside_count:
        raise RuntimeError(
            f"{inside_count} kick(s) occur inside the refinement window; "
            f"first at {nearest_kick:.12g} yr. Use a smaller window or split at kicks."
        )

    base_sim.integrator = "ias15"
    times = np.linspace(t_start, t_end, args.coarse_samples)
    distances = []
    scan_sim = base_sim.copy()
    scan_sim.integrator = "ias15"
    for t_year in times:
        scan_sim.integrate(float(t_year), exact_finish_time=1)
        scan_sim.synchronize()
        distances.append(earth_mars_distance_au(scan_sim))
    distances = np.asarray(distances)
    min_index = int(np.argmin(distances))
    left = max(0, min_index - 1)
    right = min(len(times) - 1, min_index + 1)
    if left == right:
        raise RuntimeError("Coarse minimum fell on a degenerate bracket")

    t_min, d_min_au, evals = golden_minimize(base_sim, float(times[left]), float(times[right]), args.tolerance_years)
    d_min_km = d_min_au * AU_KM
    payload = {
        "seed": args.seed,
        "event_time_year": args.event_time_year,
        "t_min_year": t_min,
        "distance_au": d_min_au,
        "distance_km": d_min_km,
        "surface_clearance_km": d_min_km - PHYSICAL_DISTANCE_KM,
        "earth_mars_radii_sums": d_min_km / PHYSICAL_DISTANCE_KM,
        "window_start_year": t_start,
        "window_end_year": t_end,
        "coarse_samples": args.coarse_samples,
        "coarse_min_time_year": float(times[min_index]),
        "coarse_min_distance_km": float(distances[min_index] * AU_KM),
        "golden_evaluations": evals,
        "kicks_applied_before_window": applied,
        "cumulative_dv_before_window_m_per_s": cumulative_dv,
        "nearest_kick_year": nearest_kick,
        "physical_contact_distance_km": PHYSICAL_DISTANCE_KM,
        "initial_state_after_ephemeris_before_com_shift": initial_state,
        "method": (
            "Replay the original kick sequence with WHFast to the start of a kick-free window, "
            "then switch to IAS15 and minimize the Earth-Mars center distance by a coarse scan "
            "followed by golden-section search."
        ),
    }
    write_outputs(Path(args.outdir), payload)
    print(
        f"seed {args.seed}: d_min={d_min_km:.6f} km "
        f"at t={t_min:.12f} yr, clearance={payload['surface_clearance_km']:.6f} km",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--event-time-year", type=float, required=True)
    parser.add_argument("--window-years", type=float, default=0.25)
    parser.add_argument("--coarse-samples", type=int, default=2001)
    parser.add_argument("--tolerance-years", type=float, default=1e-10)
    parser.add_argument("--kick-years", type=float, default=15_000_000.0)
    parser.add_argument("--kicks", type=int, default=3000)
    parser.add_argument("--target-a", type=float, default=1.75)
    parser.add_argument("--dt-days", type=float, default=8.0)
    parser.add_argument("--coordinates", default="jacobi")
    parser.add_argument("--corrector", type=int, default=11)
    parser.add_argument("--kernel", default="de440s.bsp")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--progress-every-kicks", type=int, default=500)
    parser.add_argument("--outdir", default="outputs/earth_mars_refined_encounter")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
