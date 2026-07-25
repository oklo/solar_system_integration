# Solar System Integration

This repository contains the REBOUND workflows used for the Solar System and "move the Earth" simulations in the July 2026 distant-future talk.  The code starts the Sun, eight planets, and Pluto from a JPL DE440s ephemeris state at `2026-07-22 09:00:00` Central Daylight Time (`2026-07-22 14:00:00 UTC`), integrates forward with WHFast, writes osculating orbital elements, and renders barycentric orbit animations.

The repository intentionally excludes generated movies, long close-encounter logs, and large output directories.  The checked-in `data/de440s.bsp` kernel is included so the examples can run without a live ephemeris download.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quick Validation

```bash
make validate PYTHON=python
```

This runs a 10 kyr smoke integration at 5 kyr cadence, renders one MVEM preview frame, and byte-compiles the scripts.  Outputs land under `outputs/validation_10k/`, which is ignored by git.

## Baseline 5 Myr Solar System Run

```bash
python scripts/integrate_solar_system.py \
  --years 5000000 \
  --cadence-years 5000 \
  --data-dir data \
  --kernel de440s.bsp \
  --outdir outputs/full_5myr
```

Primary generated files:

- `outputs/full_5myr/snapshots.npz`: barycentric Cartesian states and osculating elements.
- `outputs/full_5myr/osculating_elements.csv`: CSV element table.
- `outputs/full_5myr/animation_primitives_inner.npz`: MVEM one-day-cadence ellipse point clouds.
- `outputs/full_5myr/metadata.json`: start epoch, integrator, cadence, and ephemeris metadata.

The 5 Myr run writes 1001 snapshots including both endpoints.  Use snapshots `1..1000` for exactly 1000 forward-time animation frames.

## Render Animations

MVEM top-down plus physically scaled side view:

```bash
python scripts/render_animation.py \
  --input outputs/full_5myr/animation_primitives_inner.npz \
  --out outputs/full_5myr/solar_system_mvem_5myr_smoothed_side.mp4
```

Outer Solar System view sized for Jupiter, Saturn, Uranus, Neptune, and Pluto:

```bash
python scripts/render_outer_solar_system.py \
  --snapshots outputs/full_5myr/snapshots.npz \
  --metadata outputs/full_5myr/metadata.json \
  --out outputs/full_5myr/solar_system_outer_5myr_side_scrub.mp4
```

The renderers use an initial invariable-plane basis by default, smooth the eccentricity vector for display, add fixed labels and an odometer, and preserve equal AU scale in the side panel.  The outer renderer samples the outer-planet osculating ellipses with 100 equal true-anomaly markers.

## Kicked-Earth Experiment

The kicked-Earth experiment is a deliberately naive analogue of Korycansky, Laughlin & Adams (2001): instead of modeling asteroid/Jupiter gravity assists, it applies direct prograde impulses to Earth at random times.  The default kicks are normalized so a two-body energy estimate would move Earth from about `1 AU` to `1.4 AU`.

```bash
python scripts/run_earth_kick_experiment.py \
  --outdir outputs/earth_kick_10myr

python scripts/render_earth_kick_animation.py \
  --input outputs/earth_kick_10myr/earth_kick_snapshots.npz \
  --out outputs/earth_kick_10myr/earth_kick_10myr.mp4
```

The renderer infers `metadata.json`, `osculating_elements.csv`, `earth_kicks.csv`, and `earth_mars_close_approaches.csv` from the snapshot file's directory.

The experiment also supports Earth-Mars close-encounter logging through REBOUND's line collision detection, seed sweeps, and refined encounter searches.  See:

- `scripts/search_earth_mars_collision.py`
- `scripts/hunt_earth_mars_collision.py`
- `scripts/refine_earth_mars_encounter.py`
- `scripts/run_encounter_campaign.py`
- `scripts/run_refined_encounter_campaign.py`

## Notes and Caveats

- Coordinates and units are AU, solar masses, and Julian years.
- The default integrator is WHFast with an 8 day timestep and high-order corrector.
- The baseline Solar System visualizations use heliocentric osculating elements for ellipse rendering, then display them in a barycentric/invariable-plane view.
- The kicked-Earth model is intentionally nonphysical as an engineering prescription; it is a dynamical stress test and visualization device.
- Long integrations and full animations can produce multi-GB output directories.  Keep `outputs/` untracked.

## Repository Layout

- `scripts/`: integration, rendering, Earth-kick, search, and refinement entry points.
- `data/de440s.bsp`: JPL DE440s ephemeris kernel used by Skyfield.
- `docs/`: methodology notes and talk-context descriptions.
- `examples/`: reserved for lightweight examples; generated outputs are ignored.
