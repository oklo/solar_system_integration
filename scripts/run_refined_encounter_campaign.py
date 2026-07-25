#!/usr/bin/env python3
"""Search seeds for refined Earth-Mars close approaches."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


PHYSICAL_DISTANCE_KM = 6_371.0 + 3_389.5


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


def latest_csv_row(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else None


def append_summary(path: Path, row: dict) -> None:
    fieldnames = [
        "seed",
        "screen_threshold_km",
        "screen_hit",
        "screen_t_year",
        "refined",
        "refined_t_year",
        "refined_distance_km",
        "surface_clearance_km",
        "earth_mars_radii_sums",
        "physical_collision",
        "stop_reason",
    ]
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def completed_seeds(path: Path) -> set[int]:
    if not path.exists():
        return set()
    with path.open(newline="") as handle:
        return {int(row["seed"]) for row in csv.DictReader(handle)}


def run(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[1]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary_path = outdir / "refined_campaign_summary.csv"
    log_path = outdir / "refined_campaign.log"
    seen = completed_seeds(summary_path)

    print(
        f"Screening at {args.threshold_km:g} km; stopping at physical collision "
        f"or refined distance <= {args.target_refined_km:g} km.",
        flush=True,
    )
    print(f"Physical contact distance is {PHYSICAL_DISTANCE_KM:.6g} km.", flush=True)

    for seed in range(args.start_seed, args.start_seed + args.max_seeds):
        if seed in seen:
            print(f"Skipping completed seed {seed}", flush=True)
            continue

        screen_dir = outdir / f"seed_{seed}_screen"
        screen_csv = screen_dir / "collision_search_results.csv"
        screen_row = latest_csv_row(screen_csv)
        if screen_row is None:
            screen_command = [
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
                str(screen_dir),
            ]
            print(f"\nScreening seed {seed}", flush=True)
            stream_command(screen_command, log_path)
            screen_row = latest_csv_row(screen_csv)
        if screen_row is None:
            raise RuntimeError(f"Screen for seed {seed} did not write a result")

        screen_hit = screen_row["collision"] == "True"
        row = {
            "seed": seed,
            "screen_threshold_km": args.threshold_km,
            "screen_hit": screen_hit,
            "screen_t_year": screen_row["t_year"] if screen_hit else "",
            "refined": False,
            "refined_t_year": "",
            "refined_distance_km": "",
            "surface_clearance_km": "",
            "earth_mars_radii_sums": "",
            "physical_collision": False,
            "stop_reason": "",
        }

        if screen_hit:
            refine_dir = outdir / f"seed_{seed}_refined"
            refine_csv = refine_dir / "refined_encounter.csv"
            refined_row = latest_csv_row(refine_csv)
            if refined_row is None:
                refine_command = [
                    sys.executable,
                    str(root / "scripts" / "refine_earth_mars_encounter.py"),
                    "--seed",
                    str(seed),
                    "--event-time-year",
                    screen_row["t_year"],
                    "--window-years",
                    str(args.refine_window_years),
                    "--coarse-samples",
                    str(args.refine_coarse_samples),
                    "--kick-years",
                    str(args.kick_years),
                    "--kicks",
                    str(args.kicks),
                    "--target-a",
                    str(args.target_a),
                    "--progress-every-kicks",
                    str(args.progress_every_kicks),
                    "--outdir",
                    str(refine_dir),
                ]
                print(f"Refining seed {seed} near t={float(screen_row['t_year']):.0f} yr.", flush=True)
                stream_command(refine_command, log_path)
                refined_row = latest_csv_row(refine_csv)
            if refined_row is None:
                raise RuntimeError(f"Refinement for seed {seed} did not write a result")

            refined_distance = float(refined_row["distance_km"])
            physical_collision = refined_distance <= PHYSICAL_DISTANCE_KM
            row.update(
                {
                    "refined": True,
                    "refined_t_year": refined_row["t_min_year"],
                    "refined_distance_km": refined_row["distance_km"],
                    "surface_clearance_km": refined_row["surface_clearance_km"],
                    "earth_mars_radii_sums": refined_row["earth_mars_radii_sums"],
                    "physical_collision": physical_collision,
                }
            )
            print(
                f"Seed {seed}: refined d={refined_distance:.3f} km "
                f"({float(refined_row['earth_mars_radii_sums']):.3f} radii sums).",
                flush=True,
            )

            if physical_collision:
                row["stop_reason"] = "physical_collision"
            elif refined_distance <= args.target_refined_km:
                row["stop_reason"] = "target_refined_distance"

        append_summary(summary_path, row)
        if row["stop_reason"]:
            hit_path = outdir / "FOUND_TARGET.json"
            hit_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
            print(f"Stopping on seed {seed}: {row['stop_reason']}. Wrote {hit_path}", flush=True)
            return

    raise RuntimeError(f"No target encounter found in {args.max_seeds} seeds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-seed", type=int, default=20260908)
    parser.add_argument("--max-seeds", type=int, default=100)
    parser.add_argument("--threshold-km", type=float, default=50_000.0)
    parser.add_argument("--target-refined-km", type=float, default=25_000.0)
    parser.add_argument("--years", type=float, default=30_000_000.0)
    parser.add_argument("--kick-years", type=float, default=15_000_000.0)
    parser.add_argument("--kicks", type=int, default=3000)
    parser.add_argument("--target-a", type=float, default=1.75)
    parser.add_argument("--progress-every-kicks", type=int, default=500)
    parser.add_argument("--progress-every-years", type=float, default=1_000_000.0)
    parser.add_argument("--refine-window-years", type=float, default=0.25)
    parser.add_argument("--refine-coarse-samples", type=int, default=2001)
    parser.add_argument("--outdir", default="outputs/earth_kick_refined_encounter_campaign_50kkm")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
