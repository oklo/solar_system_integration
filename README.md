# Solar System Integration

This repository contains the REBOUND workflows used for the Solar System and "move the Earth" simulations in the July 2026 distant-future talk.  The code starts the Sun, eight planets, and Pluto from a JPL DE440s ephemeris state at `2026-07-22 09:00:00` Central Daylight Time (`2026-07-22 14:00:00 UTC`), integrates forward with WHFast, writes osculating orbital elements, and renders barycentric orbit animations.

The repository intentionally excludes generated movies, long close-encounter logs, and large output directories.  The checked-in `data/de440s.bsp` kernel is included so the examples can run without a live ephemeris download.

## Results Summary

### Five-million-year Solar System integrations

The baseline run integrates the Sun, eight planets, and Pluto for `5,000,000 yr` from the DE440s state at `2026-07-22 09:00:00` Central Daylight Time.  The production run used WHFast, an `8 day` timestep, an 11th-order corrector, and `5000 yr` output cadence, producing `1001` snapshots including the initial state.  The animation products render each epoch as osculating ellipses rather than as particle trails: the inner Solar System view focuses on Mercury, Venus, Earth, and Mars in a `3.2 AU` barycentric box, while the outer Solar System view is scaled for Jupiter, Saturn, Uranus, Neptune, and Pluto.

The stored heliocentric osculating elements remain near the expected long-term secular ranges over the run.  Examples from the `5 Myr` integration are: Earth `a = 0.999975-1.000027 AU`, `e = 0.0014-0.0624`; Mars `a = 1.523565-1.523835 AU`, `e = 0.0079-0.1239`; Jupiter `a = 5.201-5.205 AU`, `e = 0.0256-0.0614`; and Pluto `a = 39.04-39.99 AU`, `e = 0.210-0.273`.  For display, the renderers smooth the eccentricity vector components to suppress element-angle wrap artifacts, show perihelion markers and lines, add fixed labels and an odometer, and preserve the same AU scale in the top-down and side panels.

### Kicked-Earth / move-the-Earth experiments

The kicked-Earth runs are deliberately naive analogues of Korycansky, Laughlin, and Adams (2001).  Instead of modeling asteroid-mediated gravity assists, the scripts apply direct prograde impulses to Earth at random orbital phases.  In the default `10 Myr` experiment, `2000` impulses are normalized so that a two-body energy estimate would move Earth from about `1 AU` to `1.4 AU`; the cumulative impulse is `4.61 km/s`.

A five-seed sweep (`20260722` through `20260726`) moved Earth's final semimajor axis to `1.374-1.444 AU`.  Mars' response varied much more strongly, ending with semimajor axes from `1.208 AU` to `2.692 AU` and eccentricities up to `0.443`.  The closest approach in that sweep was seed `20260725`, with an Earth-Mars center-to-center distance of `0.001069 AU`, about `160,000 km` or `16.4` Earth-plus-Mars radii, at `9.227 Myr`.

Subsequent encounter searches extended the integrations and used a `50,000 km` Earth-Mars screening threshold followed by IAS15 replay/refinement of candidates.  The closest refined case currently found is seed `20260917`, with minimum center-to-center distance `16,512 km`, surface clearance `6,752 km`, or `1.69` summed Earth-plus-Mars radii at `9.210348 Myr`; no physical collision was confirmed.  Another useful high-drama non-collision case is seed `20260906`, with `36,516 km` center-to-center distance, `26,755 km` surface clearance, and `3.74` summed radii.  Threshold-trigger logs are treated as candidate finders only; refined replay is required before quoting a close-encounter distance.

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
