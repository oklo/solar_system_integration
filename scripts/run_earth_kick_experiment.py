#!/usr/bin/env python3
"""Naively migrate Earth outward with random prograde impulsive kicks."""

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

from scripts.integrate_solar_system import (
    BODIES,
    ELEMENT_NAMES,
    G_AU3_MSUN_YR2,
    JULIAN_YEAR_DAYS,
    build_simulation,
    particle_state_relative_to_com,
    snapshot_orbits,
    write_elements_csv,
)


EARTH_INDEX = 3
MARS_INDEX = 4
AU_PER_YR_TO_KM_PER_S = 149_597_870.7 / (365.25 * 86_400.0)


class CloseEncounterLogger:
    """Condense REBOUND collision callback hits into encounter minima."""

    def __init__(self, threshold_au: float, group_gap_years: float) -> None:
        self.threshold_au = threshold_au
        self.group_gap_years = group_gap_years
        self.callback_count = 0
        self.groups: list[dict] = []
        self._current: dict | None = None

    def observe(self, t_year: float, distance_au: float) -> None:
        self.callback_count += 1
        if self._current is None or t_year - self._current["end_year"] > self.group_gap_years:
            self.finalize_current()
            self._current = {
                "start_year": t_year,
                "end_year": t_year,
                "t_min_year": t_year,
                "min_distance_au": distance_au,
                "sample_count": 1,
            }
            return

        self._current["end_year"] = t_year
        self._current["sample_count"] += 1
        if distance_au < self._current["min_distance_au"]:
            self._current["min_distance_au"] = distance_au
            self._current["t_min_year"] = t_year

    def finalize_current(self) -> None:
        if self._current is not None:
            self.groups.append(self._current)
            self._current = None

    def running_min_so_far(self, initial_distance_au: float) -> float:
        best = initial_distance_au
        for group in self.groups:
            best = min(best, group["min_distance_au"])
        if self._current is not None:
            best = min(best, self._current["min_distance_au"])
        return best


def configure_whfast(sim, coordinates: str, corrector: int) -> None:
    sim.integrator = "whfast"
    sim.integrator.coordinates = coordinates
    sim.integrator.corrector = corrector
    sim.integrator.safe_mode = 0


def make_kick_times(years: float, kick_count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.sort(rng.uniform(0.0, years, kick_count))


def tangential_unit_and_speed(sim) -> tuple[np.ndarray, float]:
    sun = sim.particles[0]
    earth = sim.particles[EARTH_INDEX]
    r = np.array([earth.x - sun.x, earth.y - sun.y, earth.z - sun.z], dtype=float)
    v = np.array([earth.vx - sun.vx, earth.vy - sun.vy, earth.vz - sun.vz], dtype=float)
    rhat = r / np.linalg.norm(r)
    vt_vec = v - np.dot(v, rhat) * rhat
    vt = np.linalg.norm(vt_vec)
    if vt <= 0.0:
        raise RuntimeError("Cannot determine tangential direction for Earth kick")
    return vt_vec / vt, vt


def heliocentric_orbit(sim, body_index: int):
    return sim.particles[body_index].orbit(primary=sim.particles[0], G=sim.G)


def apply_earth_kick(sim, delta_eps: float, coordinates: str, corrector: int) -> dict:
    sim.synchronize()
    before = heliocentric_orbit(sim, EARTH_INDEX)
    unit, vt = tangential_unit_and_speed(sim)
    dv = -vt + math.sqrt(vt * vt + 2.0 * delta_eps)

    earth = sim.particles[EARTH_INDEX]
    earth.vx += dv * unit[0]
    earth.vy += dv * unit[1]
    earth.vz += dv * unit[2]
    sim.move_to_com()
    sim.synchronize()
    configure_whfast(sim, coordinates, corrector)

    after = heliocentric_orbit(sim, EARTH_INDEX)
    return {
        "dv_au_per_yr": dv,
        "dv_m_per_s": dv * AU_PER_YR_TO_KM_PER_S * 1000.0,
        "before_a": before.a,
        "before_e": before.e,
        "after_a": after.a,
        "after_e": after.e,
        "after_q": after.a * (1.0 - after.e),
        "after_Q": after.a * (1.0 + after.e),
    }


def write_kicks_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "kick_index",
        "t_year",
        "dv_au_per_yr",
        "dv_m_per_s",
        "cumulative_dv_m_per_s",
        "naive_a_au",
        "before_a",
        "before_e",
        "after_a",
        "after_e",
        "after_q",
        "after_Q",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def earth_mars_distance(sim) -> float:
    earth = sim.particles[EARTH_INDEX]
    mars = sim.particles[MARS_INDEX]
    dx = earth.x - mars.x
    dy = earth.y - mars.y
    dz = earth.z - mars.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def configure_close_encounter_log(sim, logger: CloseEncounterLogger) -> None:
    for particle in sim.particles:
        particle.r = 0.0
    sim.particles[EARTH_INDEX].r = 0.5 * logger.threshold_au
    sim.particles[MARS_INDEX].r = 0.5 * logger.threshold_au
    sim.collision = "line"

    def collision_callback(reb_sim_pointer, collision) -> int:
        if {collision.p1, collision.p2} == {EARTH_INDEX, MARS_INDEX}:
            rebound_sim = reb_sim_pointer.contents
            earth = rebound_sim.particles[EARTH_INDEX]
            mars = rebound_sim.particles[MARS_INDEX]
            dx = earth.x - mars.x
            dy = earth.y - mars.y
            dz = earth.z - mars.z
            logger.observe(float(rebound_sim.t), math.sqrt(dx * dx + dy * dy + dz * dz))
        return 0

    sim.collision_resolve = collision_callback


def write_close_approach_csv(path: Path, groups: list[dict], initial_distance: float) -> None:
    running_min = initial_distance
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "record_type",
                "t_year",
                "earth_mars_distance_au",
                "running_min_earth_mars_distance_au",
                "encounter_start_year",
                "encounter_end_year",
                "callback_samples",
            ]
        )
        writer.writerow(["initial", "0", f"{initial_distance:.12g}", f"{running_min:.12g}", "", "", ""])
        for group in groups:
            distance = group["min_distance_au"]
            running_min = min(running_min, distance)
            writer.writerow(
                [
                    "encounter",
                    f"{group['t_min_year']:.12g}",
                    f"{distance:.12g}",
                    f"{running_min:.12g}",
                    f"{group['start_year']:.12g}",
                    f"{group['end_year']:.12g}",
                    int(group["sample_count"]),
                ]
            )


def close_arrays(groups: list[dict], initial_distance: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = np.asarray([0.0, *[group["t_min_year"] for group in groups]], dtype=float)
    distances = np.asarray([initial_distance, *[group["min_distance_au"] for group in groups]], dtype=float)
    order = np.argsort(times)
    times = times[order]
    distances = distances[order]
    running_min = np.minimum.accumulate(distances)
    keep = np.r_[True, np.diff(running_min) < 0.0]
    if len(keep):
        keep[-1] = True
    return times[keep], distances[keep], running_min[keep]


def run(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    kick_years = args.kick_years if args.kick_years is not None else args.years
    if kick_years > args.years:
        raise ValueError("--kick-years cannot exceed --years")

    cadence_ratio = args.years / args.cadence_years
    if not math.isclose(cadence_ratio, round(cadence_ratio), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("--years must be an integer multiple of --cadence-years")
    times_yr = np.arange(int(round(cadence_ratio)) + 1, dtype=float) * args.cadence_years
    kick_times = make_kick_times(kick_years, args.kicks, args.seed)

    sim, initial_state = build_simulation(args.kernel, Path(args.data_dir), args.dt_days)
    configure_whfast(sim, args.coordinates, args.corrector)
    initial_earth_mars_distance = earth_mars_distance(sim)
    close_logger = CloseEncounterLogger(args.close_log_threshold, args.close_group_gap_years)
    configure_close_encounter_log(sim, close_logger)

    initial_earth = heliocentric_orbit(sim, EARTH_INDEX)
    mu = sim.G * (sim.particles[0].m + sim.particles[EARTH_INDEX].m)
    total_delta_eps = 0.5 * mu * (1.0 / initial_earth.a - 1.0 / args.target_a)
    delta_eps_per_kick = total_delta_eps / args.kicks

    states = np.empty((len(times_yr), len(BODIES), 6), dtype=float)
    elements = np.empty((len(times_yr), len(BODIES), len(ELEMENT_NAMES)), dtype=float)
    kick_rows: list[dict] = []

    snapshot_index = 0
    kick_index = 0
    cumulative_delta_eps = 0.0
    cumulative_dv = 0.0

    while snapshot_index < len(times_yr) or kick_index < len(kick_times):
        next_snapshot = times_yr[snapshot_index] if snapshot_index < len(times_yr) else math.inf
        next_kick = kick_times[kick_index] if kick_index < len(kick_times) else math.inf

        if next_kick < next_snapshot:
            sim.integrate(float(next_kick), exact_finish_time=1)
            kick = apply_earth_kick(sim, delta_eps_per_kick, args.coordinates, args.corrector)
            cumulative_delta_eps += delta_eps_per_kick
            cumulative_dv += kick["dv_m_per_s"]
            naive_a = 1.0 / (1.0 / initial_earth.a - 2.0 * cumulative_delta_eps / mu)
            kick_rows.append(
                {
                    "kick_index": kick_index + 1,
                    "t_year": next_kick,
                    "cumulative_dv_m_per_s": cumulative_dv,
                    "naive_a_au": naive_a,
                    **kick,
                }
            )
            kick_index += 1
            continue

        sim.integrate(float(next_snapshot), exact_finish_time=1)
        sim.synchronize()
        states[snapshot_index] = particle_state_relative_to_com(sim)
        elements[snapshot_index], _ = snapshot_orbits(sim)
        if args.progress_every and (
            snapshot_index == 0
            or snapshot_index == len(times_yr) - 1
            or snapshot_index % args.progress_every == 0
        ):
            earth = elements[snapshot_index, EARTH_INDEX]
            print(
                f"snapshot {snapshot_index:5d}/{len(times_yr) - 1}: "
                f"t={next_snapshot:.0f} yr, Earth a={earth[0]:.4f} AU, e={earth[1]:.4f}, "
                f"kicks={kick_index}, min d(E,M)={close_logger.running_min_so_far(initial_earth_mars_distance):.4f} AU",
                flush=True,
            )
        snapshot_index += 1

    close_logger.finalize_current()
    close_times_array, close_distances_array, running_min_array = close_arrays(
        close_logger.groups,
        initial_earth_mars_distance,
    )
    min_idx = int(np.argmin(close_distances_array))

    metadata = {
        "experiment": "Naive direct prograde Earth impulses",
        "paper_context": {
            "reference": "Korycansky, Laughlin & Adams 2001, Astronomical Engineering: A Strategy For Modifying Planetary Orbits",
            "note": "This experiment does not model the paper's asteroid/Jupiter gravity-assist loop; it injects direct prograde impulses into Earth.",
        },
        "integration": {
            "years": args.years,
            "kick_years": kick_years,
            "cadence_years": args.cadence_years,
            "dt_days": args.dt_days,
            "coordinates": args.coordinates,
            "corrector": args.corrector,
            "G_AU3_Msun_yr2": G_AU3_MSUN_YR2,
        },
        "close_approaches": {
            "method": "REBOUND collision='line' threshold log with a zero-action collision_resolve callback; callback hits are grouped into encounter minima.",
            "threshold_au": args.close_log_threshold,
            "group_gap_years": args.close_group_gap_years,
            "callback_hit_count": int(close_logger.callback_count),
            "encounter_count": int(len(close_logger.groups)),
            "animation_curve_record_count": int(len(close_times_array)),
            "minimum_earth_mars_distance_au": float(close_distances_array[min_idx]),
            "minimum_earth_mars_distance_time_year": float(close_times_array[min_idx]),
            "note": "This logs close passages at the integrator timestep scale without adding output stops to the main integration. The CSV contains all grouped encounter minima; the NPZ animation curve keeps only running-minimum change points plus the final record. It is a threshold-based close-encounter log, not an analytic continuous minimizer.",
        },
        "kicks": {
            "seed": args.seed,
            "count": args.kicks,
            "duration_years": kick_years,
            "target_a_au": args.target_a,
            "initial_earth_a_au": initial_earth.a,
            "initial_earth_e": initial_earth.e,
            "total_delta_eps_au2_per_yr2": total_delta_eps,
            "delta_eps_per_kick_au2_per_yr2": delta_eps_per_kick,
            "total_impulse_m_per_s": cumulative_dv,
        },
        "initial_state_after_ephemeris_before_com_shift": initial_state,
    }

    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_elements_csv(outdir / "osculating_elements.csv", times_yr, elements)
    write_kicks_csv(outdir / "earth_kicks.csv", kick_rows)
    write_close_approach_csv(outdir / "earth_mars_close_approaches.csv", close_logger.groups, initial_earth_mars_distance)
    np.savez_compressed(
        outdir / "earth_kick_snapshots.npz",
        times_yr=times_yr,
        body_names=np.array([spec.name for spec in BODIES]),
        element_names=np.array(ELEMENT_NAMES),
        states_au_auyr=states,
        elements=elements,
        kick_times_yr=kick_times,
        close_sample_times_yr=close_times_array,
        earth_mars_distance_au=close_distances_array,
        running_min_earth_mars_distance_au=running_min_array,
        close_log_threshold_au=np.array(args.close_log_threshold),
        close_log_callback_count=np.array(close_logger.callback_count),
    )
    print(
        f"Wrote {len(times_yr)} snapshots, {len(kick_rows)} kicks, "
        f"and {len(close_logger.groups)} close-encounter minima to {outdir}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=float, default=10_000_000.0)
    parser.add_argument("--kick-years", type=float)
    parser.add_argument("--cadence-years", type=float, default=10_000.0)
    parser.add_argument("--close-log-threshold", type=float, default=0.4)
    parser.add_argument("--close-group-gap-years", type=float, default=0.25)
    parser.add_argument("--close-sample-years", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--kicks", type=int, default=2000)
    parser.add_argument("--target-a", type=float, default=1.4)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--dt-days", type=float, default=8.0)
    parser.add_argument("--coordinates", default="jacobi")
    parser.add_argument("--corrector", type=int, default=11)
    parser.add_argument("--kernel", default="de440s.bsp")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--outdir", default="outputs/earth_kick_10myr")
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
