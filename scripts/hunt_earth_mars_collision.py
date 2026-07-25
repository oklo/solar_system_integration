#!/usr/bin/env python3
"""Resume a reduced-log Earth-Mars collision hunt and verify tight candidates."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


AU_KM = 149_597_870.7
EARTH_RADIUS_AU = 6_371.0 / AU_KM
MARS_RADIUS_AU = 3_389.5 / AU_KM
PHYSICAL_DISTANCE_AU = EARTH_RADIUS_AU + MARS_RADIUS_AU


def stream_command(command: list[str]) -> None:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def append_csv(path: Path, row: dict) -> None:
    fieldnames = [
        "seed",
        "min_distance_au",
        "min_time_year",
        "physical_distance_au",
        "candidate",
        "physical_collision_verified",
        "outdir",
    ]
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def completed_seeds(summary_csv: Path) -> set[int]:
    if not summary_csv.exists():
        return set()
    with summary_csv.open(newline="") as handle:
        return {int(row["seed"]) for row in csv.DictReader(handle)}


def physical_verifier_hit(verify_dir: Path) -> bool:
    hit_json = verify_dir / "collision_hit.json"
    if hit_json.exists():
        return True
    results_csv = verify_dir / "collision_search_results.csv"
    if not results_csv.exists():
        return False
    with results_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return bool(rows and rows[-1].get("collision") == "True")


def run(args: argparse.Namespace) -> None:
    project_root = Path(__file__).resolve().parents[1]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary_csv = outdir / "hunt_summary.csv"
    seen = completed_seeds(summary_csv)
    candidate_distance = args.candidate_radii * PHYSICAL_DISTANCE_AU

    print(
        f"Physical Earth+Mars distance = {PHYSICAL_DISTANCE_AU:.12g} AU; "
        f"candidate threshold = {candidate_distance:.12g} AU "
        f"({args.candidate_radii:g} radii sums).",
        flush=True,
    )

    for seed in range(args.start_seed, args.start_seed + args.max_seeds):
        if seed in seen:
            print(f"Skipping completed seed {seed}", flush=True)
            continue

        seed_dir = outdir / f"seed_{seed}"
        metadata_path = seed_dir / "metadata.json"
        if not metadata_path.exists():
            command = [
                sys.executable,
                str(project_root / "scripts" / "run_earth_kick_experiment.py"),
                "--seed",
                str(seed),
                "--years",
                str(args.years),
                "--kick-years",
                str(args.kick_years),
                "--cadence-years",
                str(args.cadence_years),
                "--close-log-threshold",
                str(args.close_log_threshold),
                "--outdir",
                str(seed_dir),
                "--progress-every",
                str(args.progress_every),
            ]
            print(f"\nScreening seed {seed}", flush=True)
            stream_command(command)

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        close = metadata["close_approaches"]
        min_distance = float(close["minimum_earth_mars_distance_au"])
        min_time = float(close["minimum_earth_mars_distance_time_year"])
        candidate = min_distance <= candidate_distance
        verified = False

        if candidate:
            verify_dir = outdir / f"seed_{seed}_physical_verify"
            command = [
                sys.executable,
                str(project_root / "scripts" / "search_earth_mars_collision.py"),
                "--start-seed",
                str(seed),
                "--max-seeds",
                "1",
                "--years",
                str(args.years),
                "--kick-years",
                str(args.kick_years),
                "--kicks",
                str(args.kicks),
                "--target-a",
                str(args.target_a),
                "--dt-days",
                str(args.dt_days),
                "--collision-distance-au",
                f"{PHYSICAL_DISTANCE_AU:.17g}",
                "--collision-mode",
                "line",
                "--outdir",
                str(verify_dir),
            ]
            print(
                f"Seed {seed} reduced-log minimum {min_distance:.12g} AU "
                f"at {min_time:.0f} yr; running physical verifier.",
                flush=True,
            )
            try:
                stream_command(command)
            except RuntimeError:
                raise
            except subprocess.CalledProcessError:
                # A no-hit one-seed verifier raises after writing its CSV.
                pass
            verified = physical_verifier_hit(verify_dir)

        row = {
            "seed": seed,
            "min_distance_au": min_distance,
            "min_time_year": min_time,
            "physical_distance_au": PHYSICAL_DISTANCE_AU,
            "candidate": candidate,
            "physical_collision_verified": verified,
            "outdir": seed_dir,
        }
        append_csv(summary_csv, row)
        print(
            f"Seed {seed}: min={min_distance:.12g} AU "
            f"({min_distance / PHYSICAL_DISTANCE_AU:.2f} radii sums), "
            f"candidate={candidate}, verified_collision={verified}",
            flush=True,
        )

        if verified:
            (outdir / "FOUND_SEED.txt").write_text(f"{seed}\n", encoding="utf-8")
            print(f"Found verified collision seed {seed}", flush=True)
            return

    raise RuntimeError(
        f"No verified collision found in {args.max_seeds} seeds starting at {args.start_seed}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-seed", type=int, default=20260750)
    parser.add_argument("--max-seeds", type=int, default=100)
    parser.add_argument("--years", type=float, default=30_000_000.0)
    parser.add_argument("--kick-years", type=float, default=10_000_000.0)
    parser.add_argument("--cadence-years", type=float, default=100_000.0)
    parser.add_argument("--close-log-threshold", type=float, default=0.002)
    parser.add_argument("--candidate-radii", type=float, default=10.0)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--kicks", type=int, default=2000)
    parser.add_argument("--target-a", type=float, default=1.4)
    parser.add_argument("--dt-days", type=float, default=8.0)
    parser.add_argument("--outdir", default="outputs/earth_kick_collision_hunt_30myr")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
