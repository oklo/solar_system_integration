# Reproducing Talk Outputs

The exact talk movies were generated under `outputs/` in the working copy and were not committed because they are large binary products.  The checked-in scripts are the source of truth for regenerating them.

## Full Inner Solar System Animation

```bash
python scripts/integrate_solar_system.py --years 5000000 --cadence-years 5000 --outdir outputs/full_5myr
python scripts/render_animation.py --out outputs/full_5myr/solar_system_mvem_5myr_smoothed_side.mp4
```

## Full Outer Solar System Animation

```bash
python scripts/render_outer_solar_system.py \
  --snapshots outputs/full_5myr/snapshots.npz \
  --metadata outputs/full_5myr/metadata.json \
  --out outputs/full_5myr/solar_system_outer_5myr_side_scrub.mp4
```

## Earth-Kick Seed Sweep

The talk used several seeds to show diversity in outcomes.  A typical run is:

```bash
python scripts/run_earth_kick_experiment.py \
  --seed 20260722 \
  --outdir outputs/earth_kick_seed_sweep/seed_20260722
```

Then render with `scripts/render_earth_kick_animation.py`.

## Keynote-Scrubbable Copies

Some talk movies were post-processed as all-intra H.264 so Keynote scrubbing would land on every frame.  That encoding choice greatly increases file size and is intentionally outside this source repository.

