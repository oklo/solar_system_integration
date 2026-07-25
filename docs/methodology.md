# Methodology

## Baseline Solar System Integration

The baseline integration initializes the Sun, Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune, and Pluto from the JPL DE440s ephemeris at `2026-07-22 14:00:00 UTC`.  Positions are loaded through Skyfield, masses are taken from REBOUND's Horizons mass table, and the system is shifted to its center of mass before integration.

The integration uses REBOUND's WHFast integrator in AU, solar masses, and Julian years.  The default timestep is 8 days.  Snapshots are written every 5000 years.

For each snapshot, the scripts store:

- barycentric Cartesian states for all bodies,
- heliocentric osculating elements relative to the Sun,
- MVEM animation primitives sampled at one-day mean-anomaly cadence,
- periastron guide segments clipped to the inner Solar System view box.

## Visualization

The main visual products are not direct trajectory traces.  Each frame uses that epoch's osculating elements to render the instantaneous ellipse.  For the MVEM animation, the eccentricity vector components are low-pass smoothed to reduce visual jitter in apsidal direction while preserving the long-term secular evolution.

The side panels are rendered with the same AU scale as the top-down view, so inclination evolution is not visually exaggerated.

## Kicked-Earth Experiments

The kicked-Earth experiments apply direct prograde velocity impulses to Earth at randomly sampled times.  This is not a proposed physical implementation of orbital migration.  It is a controlled numerical experiment designed to show how a naive outward migration of Earth perturbs the rest of the Solar System, especially Mars.

Close encounters are logged using REBOUND's line collision detection by assigning artificial encounter radii to Earth and Mars.  The callback records grouped encounter minima rather than relying only on sparse output snapshots.

