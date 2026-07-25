#!/usr/bin/env python3
"""Run a resumable Earth-Mars close-encounter campaign."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


AU_KM = 149_597_870.7
EARTH_RADIUS_KM = 6_371.0
MARS_RADIUS_KM = 3_389.5
PHYSICAL_DISTANCE_KM = EARTH_RADIUS_KM + MARS_RADIUS_KM


def stream_command(command: list[str], log_path: Path) -> int:
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(command) + "\n")
        log.flush()
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
            log.write(line)
            log.flush()
        return process.wait()


def latest_result(results_csv: Path) -> dict | None:
    if not results_csv.exists():
        return None
    with results_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else None


def append_summary(path: Path, row: dict) -> None:
    fieldnames = [
        "seed",
        "screen_threshold_km",
        "screen_hit",
        "screen_t_year",
        "physical_collision",
        "physical_t_year",
        "verify_to_year",
        "screen_outdir",
        "verify_outdir",
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


def run(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[1]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary_csv = outdir / "campaign_summary.csv"
    log_path = outdir / "campaign.log"
    seen = completed_seeds(summary_csv)

    print(
        f"Campaign threshold: {args.threshold_km:g} km center-to-center "
        f"({args.threshold_km / PHYSICAL_DISTANCE_KM:.2f} Earth+Mars radii sums).",
        flush=True,
    )
    print(f"Physical contact threshold: {PHYSICAL_DISTANCE_KM:.6g} km.", flush=True)

    for seed in range(args.start_seed, args.start_seed + args.max_seeds):
        if seed in seen:
            print(f"Skipping completed seed {seed}", flush=True)
            continue

        seed_dir = outdir / f"seed_{seed}_screen"
        verify_dir = outdir / f"seed_{seed}_physical_verify"
        screen_csv = seed_dir / "collision_search_results.csv"
        screen_row = latest_result(screen_csv)

        if screen_row is None:
            command = [
                sys.executable,
                str(root / "scripts" / "search_earth_mars_collision.py"),
                "--start-seed",
                str(seed),
                "--max-seeds",
                "1",
                "--report-every",
                "1",
                "--years",
                str(args.years),
                "--kick-years",
                str(args.kick_years),
                "--kicks",
                str(args.kicks),
                "--target-a",
                str(args.target_a),
                "--collision-distance-km",
                str(args.threshold_km),
                "--collision-mode",
                "line",
                "--progress-every-kicks",
                str(args.progress_every_kicks),
                "--progress-every-years",
                str(args.progress_every_years),
                "--outdir",
                str(seed_dir),
            ]
            print(f"\nScreening seed {seed}", flush=True)
            stream_command(command, log_path)
            screen_row = latest_result(screen_csv)

        if screen_row is None:
            raise RuntimeError(f"No screen result written for seed {seed}")

        screen_hit = screen_row["collision"] == "True"
        verify_row = None
        verify_to_year = ""

        if screen_hit:
            screen_t = float(screen_row["t_year"])
            verify_to_year = min(args.years, screen_t + args.verify_margin_years)
            verify_csv = verify_dir / "collision_search_results.csv"
            verify_row = latest_result(verify_csv)
            if verify_row is None:
                command = [
                    sys.executable,
                    str(root / "scripts" / "search_earth_mars_collision.py"),
                    "--start-seed",
                    str(seed),
                    "--max-seeds",
                    "1",
                    "--report-every",
                    "1",
                    "--years",
                    f"{verify_to_year:.12g}",
                    "--kick-years",
                    str(args.kick_years),
                    "--kicks",
                    str(args.kicks),
                    "--target-a",
                    str(args.target_a),
                    "--collision-mode",
                    "line",
                    "--progress-every-kicks",
                    str(args.progress_every_kicks),
                    "--progress-every-years",
                    str(args.progress_every_years),
                    "--outdir",
                    str(verify_dir),
                ]
                print(
                    f"Seed {seed} crossed {args.threshold_km:g} km at "
                    f"{screen_t:.0f} yr; verifying physical contact to {verify_to_year:.0f} yr.",
                    flush=True,
                )
                stream_command(command, log_path)
                verify_row = latest_result(verify_csv)

        physical_collision = bool(verify_row and verify_row["collision"] == "True")
        row = {
            "seed": seed,
            "screen_threshold_km": args.threshold_km,
            "screen_hit": screen_hit,
            "screen_t_year": screen_row["t_year"] if screen_hit else "",
            "physical_collision": physical_collision,
            "physical_t_year": verify_row["t_year"] if physical_collision else "",
            "verify_to_year": verify_to_year,
            "screen_outdir": seed_dir,
            "verify_outdir": verify_dir if screen_hit else "",
        }
        append_summary(summary_csv, row)

        if screen_hit and not physical_collision:
            print(
                f"Seed {seed}: SPH candidate bracket "
                f"{PHYSICAL_DISTANCE_KM:.1f} km < d < {args.threshold_km:g} km.",
                flush=True,
            )
        if physical_collision:
            hit_json = outdir / "FOUND_COLLISION.json"
            hit_json.write_text(json.dumps(row, indent=2), encoding="utf-8")
            print(f"Found physical collision seed {seed}; wrote {hit_json}", flush=True)
            return

    raise RuntimeError(f"No physical collision found in {args.max_seeds} seeds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-seed", type=int, default=20260901)
    parser.add_argument("--max-seeds", type=int, default=100)
    parser.add_argument("--threshold-km", type=float, default=50_000.0)
    parser.add_argument("--years", type=float, default=30_000_000.0)
    parser.add_argument("--kick-years", type=float, default=15_000_000.0)
    parser.add_argument("--kicks", type=int, default=3000)
    parser.add_argument("--target-a", type=float, default=1.75)
    parser.add_argument("--verify-margin-years", type=float, default=500_000.0)
    parser.add_argument("--progress-every-kicks", type=int, default=500)
    parser.add_argument("--progress-every-years", type=float, default=1_000_000.0)
    parser.add_argument("--outdir", default="outputs/earth_kick_encounter_campaign_50kkm")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
