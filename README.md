# tidal-pinn-data

Open, key-free geophysical & astronomical data for physics-informed learning —
one script, ~20 sources, a catalog of "how much does textbook physics explain,
and what is left for the network".

The core use case is **tides**: exact astronomical forcing (JPL ephemeris) →
NOAA harmonic prediction (the physics baseline) → observed water level →
the *non-tidal residual* driven by pressure and wind. That residual is the
target for a grey-box / physics-informed neural network (PINN). Around it sit
independent checks from Earth rotation, sea-level trends, space weather,
seismology and astrophysical catalogs, each with a zero- or one-parameter
physics identity that the data must satisfy.

Everything here is downloaded from public endpoints with **no API key**.

---

## Quick start

```bash
git clone https://github.com/BorisKriuk/temp.git
cd temp

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 60-day probe of every source (~3-6 min, ~60 MB)
python try_data.py

# recommended for tidal work: one full year (resolves K1/P1, S2/K2)
python try_data.py --days 365

# only specific sources
python try_data.py --only noaa_tides,noaa_met,ephemeris,tidal_analysis
```

`requirements.txt`

```
numpy>=1.24
pandas>=2.0
requests>=2.31
scipy>=1.10
matplotlib>=3.7
```

Output goes to `data_out/`:

```
data_out/
├── catalog.json            # per-source status, columns, stats, physics checks
├── <source>.csv            # one tidy CSV per source
└── plots/<source>.png      # quick-look plot per source
```

### Command-line options

| Flag | Default | Meaning |
|---|---|---|
| `--days N` | 60 | Length of the hourly window ending today (tides, met, ephemeris, ERA5, marine) |
| `--only a,b,c` | all | Run only these sources |
| `--skip a,b,c` | none | Skip these sources |
| `--out DIR` | `data_out` | Output directory |
| `--no-plots` | off | Skip PNG generation |

A source that fails is recorded in `catalog.json` with `"ok": false` and the
error string; the run continues.

---

## Sources

| Key | Group | What | Endpoint | Cadence / span |
|---|---|---|---|---|
| `noaa_tides` | ocean/tides | Observed water level + NOAA harmonic prediction at 5 gauges | CO-OPS Data API | 6-min → hourly, chunked ≤31 d |
| `noaa_met` | ocean/tides | Air pressure, wind at each gauge | CO-OPS Data API | hourly |
| `openmeteo_era5` | ocean/tides | ERA5 reanalysis pressure/wind (2nd met source) | Open-Meteo Historical | hourly, ~5 d latency |
| `openmeteo_marine` | ocean/tides | Waves, currents, model sea level | Open-Meteo Marine | hourly |
| `ndbc_buoy` | ocean/tides | Buoy met + spectral waves (real-time ~45 d) | NDBC realtime2 | 10-min–hourly |
| `psmsl_sea_level` | ocean/tides | Monthly mean sea level, RLR (Brest 1807-, SF 1854-) | PSMSL | monthly |
| `ephemeris` | celestial | Geocentric Moon/Sun vectors → distance, declination, phase, tidal forcing | JPL Horizons API | hourly |
| `usno_moon_phases` | celestial | Principal moon phases (fallback: Horizons elongation extrema) | USNO AA API | events |
| `iers_eop` | celestial | Polar motion, UT1-UTC, LOD | IERS finals2000A | daily, 1973- |
| `swpc_solar_wind` | solar/space | Real-time solar wind plasma + IMF | SWPC rtsw JSON | 1-min, last 24 h |
| `omni_hourly` *(opt.)* | solar/space | OMNI2 hourly solar wind + Kp/Dst/F10.7 | NASA SPDF | hourly, 1963- |
| `gfz_kp` | solar/space | Kp geomagnetic index | GFZ Potsdam | 3-hourly, 2 yr |
| `silso_sunspots` | solar/space | Daily sunspot number | SILSO | daily, 1818- |
| `usgs_earthquakes` | solid earth | All M≥4.5 events, 365 d | USGS FDSN | events |
| `exoplanets` | astro catalogs | Composite planet table | NASA Exoplanet Archive TAP | catalog |
| `gaia_dr3` | astro catalogs | 20k nearby well-measured stars | ESA Gaia TAP (VizieR fallback) | catalog |
| `gwosc` | astro catalogs | GWTC gravitational-wave events | GWOSC event API | catalog |
| `jpl_close_approaches` | astro catalogs | Asteroids within 0.05 au, ±1 yr | JPL SBDB CAD | events |
| `nasa_power` | energy | Daily irradiance, T, wind, pressure at gauge | NASA POWER | daily, 365 d |
| `cern_opendata` | particle | Dataset metadata only | CERN Open Data | catalog |
| `tidal_analysis` | derived | Harmonic fit, equilibrium tide, inverse barometer, residual spectra | — | hourly |

Tide gauges (`TIDE_STATIONS` in `try_data.py`):

| ID | Station | Regime |
|---|---|---|
| 9414290 | San Francisco, CA | mixed semidiurnal |
| 8443970 | Boston, MA | semidiurnal, ~3 m |
| 8771450 | Galveston Pier 21, TX | diurnal, small |
| 9455920 | Anchorage, AK | very large range, shallow-water nonlinearity |
| 8724580 | Key West, FL | mixed, small |

---

## What the data says (60-day probe, Jul–Sep 2026)

**Tides.** NOAA harmonics explain **99.4 %** of variance at SF / Boston /
Anchorage but only **77 % at Galveston** and **89 % at Key West**. Residual
periods there of 19–43 days are Gulf wind set-up / steric signal, not tide.
*Those two stations are the PINN targets.* Form factors recover the textbook
classification: Boston 0.20 (semidiurnal), Galveston 2.96 (diurnal), SF 0.98
(mixed). Anchorage: shallow-water ratio (M4+M6)/M2 = 0.12 and even NOAA leaves
21 cm RMS — the nonlinearity target.

| Station | Var. explained (NOAA) | Residual σ | Form factor | (M4+M6)/M2 |
|---|---|---|---|---|
| San Francisco | 0.994 | 4.2 cm | 0.98 | 0.045 |
| Boston | 0.994 | 7.4 cm | 0.20 | 0.040 |
| Galveston | **0.771** | 8.9 cm | 2.96 | 0.080 |
| Anchorage | 0.994 | 21.0 cm | 0.28 | **0.120** |
| Key West | **0.889** | 5.9 cm | 0.89 | 0.067 |

**Ephemeris.** Perigee/apogee forcing ratio (M/r³) = 1.44; syzygies land on
the right dates (2026-07-14, 07-29, 08-12, 08-28). Synodic month from
elongation extrema: 29.29–29.79 d, mean 29.50 d.

**PSMSL.** Brest **1.09 ± 0.03 mm/yr**, San Francisco **1.51 ± 0.03 mm/yr**,
with annual + semiannual peaks — consistent with the literature.

**IERS.** Chandler wobble **432.6 d**, annual term, and the fortnightly
**Mf 13.66 d** line in LOD — tidal braking detected from a CSV.

**Space weather / solid Earth / Sun.** GFZ Kp shows the **13.5 d** two-sector
recurrence; USGS Gutenberg-Richter **b = 1.18 ± 0.01**, inter-event CV 1.17
(Omori clustering); SILSO Schwabe cycle **10.87 yr**.

**Catalogs.** Exoplanets: Kepler-III ratio median 0.000 dex but kurtosis 131 →
the heavy tails are catalog inconsistencies, a clean demo of a physics loss
flagging bad rows. GWOSC chirp-mass identity 0.99 ± 0.016 (posterior medians,
so not exactly 1). JPL `focusing_check ≈ 1 ± 1e-9` is a **tautology** (JPL
derives v_inf from v_rel with the same formula) — keep as a parsing check only.

---

## Known issues and fixes

These were found in the first probe and are patched in the current script;
listed here so you know what to look for if you fork an older version.

1. **Equilibrium tide gave `-Infinity`.** Horizons reads `START_TIME` as TDB,
   so epochs were 23:59 UTC while gauges are on :00 → zero overlap in the
   merge. Fix: `TIME_TYPE='UT'` and round to the hour. `catalog.json` is now
   passed through a sanitiser (`NaN`/`±Inf` → `null`) so it stays valid JSON.
2. **Inverse-barometer slope −0.54 cm/hPa (theory −1.01), R² 0.03.** The
   regression used the local 11-constituent residual, which is dominated by
   unresolved tide (K1/P1, L2/2N2). Fix: regress on the **NOAA residual**
   (obs − 37-constituent prediction), at every station, not just SF.
3. **Gaia HTTP 500.** ESA TAP was down. Fix: sync → async → VizieR mirror
   fallback chain.
4. **60-day window can't separate K1/P1 or S2/K2** (Rayleigh needs ~183 d).
   Fix: `--days 365` uses CO-OPS `hourly_height` (verified, ≤1 yr/request)
   for the past plus 6-min `water_level` for the last 45 days, and a
   19-constituent set including P1, K2, L2, ν2, μ2, 2N2, Sa, Ssa.
5. **SWPC rtsw files hold only 24 h** (7-day files removed Apr 2026).
   `corr_speed_vs_Bt` on one day is noise. Use `omni_hourly` for history, or
   retain daily pulls.
6. **USNO API** may time out from some networks; the script falls back to
   Horizons-derived phases automatically.
7. **NASA POWER clearness index ~23 % missing** — the CERES solar product lags
   2–3 months. Expected.

---

## Physics checks encoded in `catalog.json`

| Source | Identity | Free parameters |
|---|---|---|
| tidal_analysis | Harmonic superposition; equilibrium tide M/r³ with lag/gain; IB slope −1.01 cm/hPa; wind stress ρₐC_d\|U\|U | gain, lag |
| iers_eop | Mf 13.66 d, Mm 27.55 d in LOD; Chandler 433 d | 0 |
| psmsl | Annual, semiannual, 18.61-yr nodal cycle | trend |
| exoplanets | P²M/a³ = 1 (yr, M☉, AU) | 0 |
| gwosc | M_chirp = (m₁m₂)^{3/5}/(m₁+m₂)^{1/5} | 0 |
| jpl_cad | v_rel² = v_inf² + 2GM/r (tautological in this dataset) | 0 |
| ndbc_buoy | L = gT²/2π, steepness H/L < 0.14 | 0 |
| usgs | Gutenberg-Richter b ≈ 1; Omori CV > 1 | b |
| gfz_kp / omni | 27-d rotation, 13.5-d sector recurrence | 0 |
| silso | Schwabe ~11 yr | 0 |
| nasa_power | Clear-sky irradiance baseline; clouds = residual | 0 |

---

## Where this leaves the PINN

Per station and per hour you have: exact astronomical forcing (ephemeris), a
strong physics baseline (NOAA harmonics, 99 % at three stations), an
independent model baseline (Open-Meteo `sea_level_height_msl`), two met
sources, and a residual that is 4–9 cm at open-coast stations but **23 % of
signal variance at Galveston**.

Natural first model — grey-box:

```
η(t) = η_harmonic(t) + NN(p(t−k..t), τx(t−k..t), τy(t−k..t))
```

with the inverse-barometer slope as a soft constraint and a memory of a few
hours (the residual's lag-1 autocorrelation is 0.85–0.96, so instantaneous
pressure/wind is not enough). Run the 1-year version first: the constituent
table will tell you how much of Galveston's "unexplained" 23 % is actually
long-period tide that 60 days could not resolve.

Suggested loss terms already computable from `data_out/`:

- data: MSE on `residual_m` (obs − NOAA prediction)
- physics: `(∂η/∂p + 0.0101 m/hPa)²` on the pressure input
- physics: harmonic amplitudes for resolvable constituents ≈ NOAA's published
  values (weak prior)
- consistency: model sea level (Open-Meteo) vs NN output at low frequency

---

## Project layout

```
.
├── try_data.py          # all sources, analysis, catalog + plots
├── requirements.txt
├── README.md
└── data_out/            # generated, git-ignored
```

Add a source by writing `def src_<name>(ctx: Ctx) -> dict` returning
`df`, `time_col`, optional `extra`, `url`, `auth`, `notes`, and registering it
in `SOURCES`. Use `get()` (retries + backoff) for HTTP and `dominant_periods()`
/ `harmonic_fit()` / `best_lag_fit()` from the helper section.

---

## Attribution

Please credit the providers when publishing:

- NOAA CO-OPS, NDBC, SWPC, NCEI — public domain (US Government)
- NASA/JPL Horizons & SBDB, NASA POWER, NASA Exoplanet Archive, NASA SPDF/OMNI
- Open-Meteo (CC BY 4.0) — data from ECMWF ERA5 / Copernicus, DWD, Météo-France
- PSMSL — cite Holgate et al. (2013) and the PSMSL dataset
- IERS Earth Orientation Centre
- GFZ Potsdam Kp (CC BY 4.0), SILSO/Royal Observatory of Belgium
- USGS Earthquake Hazards Program
- ESA/Gaia/DPAC; CDS VizieR
- GWOSC — LIGO/Virgo/KAGRA
- CERN Open Data Portal
- USNO Astronomical Applications Department

## License

Code: MIT. Data: per-provider terms above.