#!/usr/bin/env python3
"""Search kick seeds until Earth and Mars physically collide."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import rebound

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.integrate_solar_system import G_AU3_MSUN_YR2, build_simulation
from scripts.run_earth_kick_experiment import (
    EARTH_INDEX,
    MARS_INDEX,
    apply_earth_kick,
    configure_whfast,
    heliocentric_orbit,
    make_kick_times,
)


AU_KM = 149_597_870.7
EARTH_RADIUS_AU = 6_371.0 / AU_KM
MARS_RADIUS_AU = 3_389.5 / AU_KM


def earth_mars_distance(sim: rebound.Simulation) -> float:
    earth = sim.particles[EARTH_INDEX]
    mars = sim.particles[MARS_INDEX]
    dx = earth.x - mars.x
    dy = earth.y - mars.y
    dz = earth.z - mars.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def collision_distance_au(args: argparse.Namespace) -> float:
    if args.collision_distance_au is not None and args.collision_distance_km is not None:
        raise ValueError("Use only one of --collision-distance-au or --collision-distance-km")
    if args.collision_distance_km is not None:
        return args.collision_distance_km / AU_KM
    return args.collision_distance_au or (EARTH_RADIUS_AU + MARS_RADIUS_AU)


def configure_collision(sim: rebound.Simulation, args: argparse.Namespace) -> None:
    collision_distance = collision_distance_au(args)
    for particle in sim.particles:
        particle.r = 0.0
    sim.particles[EARTH_INDEX].r = 0.5 * collision_distance
    sim.particles[MARS_INDEX].r = 0.5 * collision_distance
    sim.collision = args.collision_mode
    sim.collision_resolve = "halt"


def run_seed(
    seed: int,
    args: argparse.Namespace,
    base_sim: rebound.Simulation,
    delta_eps_per_kick: float,
    initial_earth_a: float,
    mu: float,
) -> dict:
    seed_start = time.monotonic()
    sim = base_sim.copy()
    configure_whfast(sim, args.coordinates, args.corrector)
    configure_collision(sim, args)
    kick_years = args.kick_years if args.kick_years is not None else args.years
    kick_times = make_kick_times(kick_years, args.kicks, seed)
    cumulative_delta_eps = 0.0
    cumulative_dv = 0.0
    kick_index = 0
    collision = False

    kicks_applied = 0
    try:
        for kick_index, kick_time in enumerate(kick_times, start=1):
            if kick_time > args.years:
                break
            sim.integrate(float(kick_time), exact_finish_time=1)
            kick = apply_earth_kick(sim, delta_eps_per_kick, args.coordinates, args.corrector)
            configure_collision(sim, args)
            cumulative_delta_eps += delta_eps_per_kick
            cumulative_dv += kick["dv_m_per_s"]
            kicks_applied = kick_index
            if args.progress_every_kicks and (
                kicks_applied == 1 or kicks_applied % args.progress_every_kicks == 0
            ):
                elapsed = time.monotonic() - seed_start
                print(
                    f"seed {seed}: kick {kicks_applied}/{args.kicks} "
                    f"at t={sim.t:.0f} yr, elapsed={elapsed/60:.1f} min",
                    flush=True,
                )

        if args.progress_every_years and sim.t < args.years:
            next_report = (math.floor(sim.t / args.progress_every_years) + 1) * args.progress_every_years
            while next_report < args.years:
                sim.integrate(next_report, exact_finish_time=1)
                elapsed = time.monotonic() - seed_start
                print(
                    f"seed {seed}: t={sim.t:.0f}/{args.years:.0f} yr "
                    f"after {kicks_applied} kicks, elapsed={elapsed/60:.1f} min",
                    flush=True,
                )
                next_report += args.progress_every_years
        sim.integrate(args.years, exact_finish_time=1)
    except rebound.Collision:
        collision = True
        sim.synchronize()

    sim.synchronize()
    earth_orbit = heliocentric_orbit(sim, EARTH_INDEX)
    mars_orbit = heliocentric_orbit(sim, MARS_INDEX)
    naive_a = 1.0 / (1.0 / initial_earth_a - 2.0 * cumulative_delta_eps / mu)
    return {
        "seed": seed,
        "collision": collision,
        "t_year": float(sim.t),
        "kicks_applied": int(kicks_applied),
        "earth_mars_distance_au": earth_mars_distance(sim),
        "physical_collision_distance_au": EARTH_RADIUS_AU + MARS_RADIUS_AU,
        "search_collision_distance_au": collision_distance_au(args),
        "search_collision_distance_km": collision_distance_au(args) * AU_KM,
        "collision_mode": args.collision_mode,
        "naive_a_au": naive_a,
        "cumulative_dv_m_per_s": cumulative_dv,
        "earth_a_au": earth_orbit.a,
        "earth_e": earth_orbit.e,
        "mars_a_au": mars_orbit.a,
        "mars_e": mars_orbit.e,
    }


def append_csv(path: Path, row: dict) -> None:
    fieldnames = list(row.keys())
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    results_csv = outdir / "collision_search_results.csv"
    hit_json = outdir / "collision_hit.json"
    search_distance_au = collision_distance_au(args)
    print(
        f"Flagging Earth-Mars center-to-center encounters within "
        f"{search_distance_au * AU_KM:.6g} km ({search_distance_au:.12g} AU).",
        flush=True,
    )
    kick_years = args.kick_years if args.kick_years is not None else args.years
    if kick_years > args.years:
        print(
            f"Using a {kick_years:g} yr kick schedule but integrating only to {args.years:g} yr; "
            "kicks after the integration end will not be applied.",
            flush=True,
        )

    base_sim, initial_state = build_simulation(args.kernel, Path(args.data_dir), args.dt_days)
    configure_whfast(base_sim, args.coordinates, args.corrector)
    initial_earth = heliocentric_orbit(base_sim, EARTH_INDEX)
    mu = base_sim.G * (base_sim.particles[0].m + base_sim.particles[EARTH_INDEX].m)
    total_delta_eps = 0.5 * mu * (1.0 / initial_earth.a - 1.0 / args.target_a)
    delta_eps_per_kick = total_delta_eps / args.kicks

    start_time = time.monotonic()
    for offset in range(args.max_seeds):
        seed = args.start_seed + offset
        row = run_seed(seed, args, base_sim, delta_eps_per_kick, initial_earth.a, mu)
        append_csv(results_csv, row)

        if row["collision"] or offset == 0 or (offset + 1) % args.report_every == 0:
            elapsed = time.monotonic() - start_time
            print(
                f"seed {seed}: collision={row['collision']} "
                f"t={row['t_year']:.0f} yr, d={row['earth_mars_distance_au']:.8g} AU, "
                f"Earth a={row['earth_a_au']:.4f}, Mars a={row['mars_a_au']:.4f}, "
                f"elapsed={elapsed/60:.1f} min",
                flush=True,
            )

        if row["collision"]:
            payload = {
                "search": {
                    "start_seed": args.start_seed,
                    "tested_count": offset + 1,
                    "years": args.years,
                    "kick_years": kick_years,
                    "kicks": args.kicks,
                    "dt_days": args.dt_days,
                    "coordinates": args.coordinates,
                    "corrector": args.corrector,
                    "G_AU3_Msun_yr2": G_AU3_MSUN_YR2,
                    "earth_radius_au": EARTH_RADIUS_AU,
                    "mars_radius_au": MARS_RADIUS_AU,
                    "search_collision_distance_au": search_distance_au,
                    "search_collision_distance_km": search_distance_au * AU_KM,
                    "collision_mode": args.collision_mode,
                },
                "hit": row,
                "initial_state_after_ephemeris_before_com_shift": initial_state,
            }
            hit_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"Wrote collision hit to {hit_json}", flush=True)
            return

    raise RuntimeError(f"No collision found in {args.max_seeds} seeds starting at {args.start_seed}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-seed", type=int, default=20260727)
    parser.add_argument("--max-seeds", type=int, default=10_000)
    parser.add_argument("--report-every", type=int, default=10)
    parser.add_argument("--years", type=float, default=10_000_000.0)
    parser.add_argument("--kick-years", type=float)
    parser.add_argument("--kicks", type=int, default=2000)
    parser.add_argument("--target-a", type=float, default=1.4)
    parser.add_argument("--dt-days", type=float, default=8.0)
    parser.add_argument("--collision-distance-au", type=float)
    parser.add_argument("--collision-distance-km", type=float)
    parser.add_argument("--collision-mode", choices=["line", "direct"], default="line")
    parser.add_argument("--progress-every-kicks", type=int, default=0)
    parser.add_argument("--progress-every-years", type=float, default=0.0)
    parser.add_argument("--coordinates", default="jacobi")
    parser.add_argument("--corrector", type=int, default=11)
    parser.add_argument("--kernel", default="de440s.bsp")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--outdir", default="outputs/earth_kick_collision_search")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
