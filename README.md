# SEP Detection Pipeline

A physics-informed, rule-based detection framework for Solar Energetic Particle (SEP) events using multi-instrument satellite proton flux data. The pipeline produces reliable labeled historical detections from 1995 to 2025, validated against the GSEP catalog for 1995–2017 and extended as original detections for 2018–2025.

---

## Table of Contents

1. [Objective](#1-objective)
2. [Data Sources](#2-data-sources)
   - [GOES EPS (1995–2010)](#21-goes-eps-energetic-particle-sensor--1995--2010)
   - [GOES EPEAD (2010–2020)](#22-goes-epead-energetic-proton-electron-and-alpha-detector--2010--2020)
   - [GOES-16 SGPS (2020–2025)](#23-goes-16-sgps-solar-and-galactic-proton-sensor--2020--2025)
   - [SOHO COSTEP/EPHIN (1995–2025)](#24-soho-costepephin-1995--2025)
   - [GSEP Catalog (Ground Truth)](#25-gsep-catalog-ground-truth)
3. [Satellite-Year Mapping](#3-satellite-year-mapping)
4. [Flux Computation](#4-flux-computation)
   - [GOES EPS: Pre-computed Integral Flux](#41-goes-eps-pre-computed-integral-flux)
   - [GOES EPS: Differential-to-Integral Derivation (GOES-12 Fallback)](#42-goes-eps-differential-to-integral-derivation-goes-12-fallback)
   - [GOES EPEAD: Pre-computed Integral Flux](#43-goes-epead-pre-computed-integral-flux)
   - [GOES-16 SGPS: Differential-to-Integral Derivation](#44-goes-16-sgps-differential-to-integral-derivation)
   - [SOHO COSTEP/EPHIN: >10 MeV Proxy from Differential Channels](#45-soho-costepephin-10-mev-proxy-from-differential-channels)
5. [Detection Logic](#5-detection-logic)
6. [Multi-Instrument Fusion](#6-multi-instrument-fusion)
7. [Evaluation Methodology](#7-evaluation-methodology)
8. [Validation Results (1995–2017)](#8-validation-results-1995--2017)
9. [Known Limitations and Data Gaps](#9-known-limitations-and-data-gaps)
10. [Project Structure](#10-project-structure)
11. [Usage](#11-usage)
12. [References](#12-references)

---

## 1. Objective

The goal of this project is to build a **standardized, reproducible SEP event detection pipeline** that:

- Ingests proton flux data from multiple satellite instruments (GOES EPS, GOES EPEAD, GOES SGPS, SOHO COSTEP/EPHIN) spanning **1995–2025**.
- Applies a consistent, physics-informed detection algorithm across all instruments and all years.
- Validates detected events against the **GSEP catalog** (Papaioannou et al.) for the 1995–2017 period where ground truth exists.
- Generates **original SEP event detections** for 2018–2025, where no external catalog is available for comparison.
- Produces clean, labeled event catalogs suitable for training downstream machine learning models.

The detection logic is entirely rule-based (threshold + gradient + duration with hysteresis), intentionally avoiding ML in the core detector. This ensures the labels produced are physically interpretable and not circularly dependent on a model.

---

## 2. Data Sources

### 2.1 GOES EPS (Energetic Particle Sensor) — 1995 – 2010

**Satellites**: GOES-8 (1995–2003), GOES-12 (2003–2010)

The EPS is a single-unit instrument with one telescope and three dome detectors. It provides **7 differential proton channels** (P1–P7) covering ~0.74 to 500 MeV, and pre-computed **corrected integral proton flux** products.

| Property | Value |
|----------|-------|
| **Archive** | NOAA NCEI legacy archive |
| **Format** | Monthly CSV files with metadata header ending at `data:` marker |
| **Cadence** | 5-minute averages |
| **Key variable** | `p3_flux_ic` — corrected integral proton flux >10 MeV |
| **Units** | pfu (protons / cm² · s · sr) |
| **Missing flag** | `-99999` |
| **Dimensions** | ~8,928 records per month (1 record per 5-min interval) |

**Differential channels** (used for fallback derivation):

| Channel | Energy Range (MeV) | ΔE (MeV) | Units |
|---------|-------------------|-----------|-------|
| P3 | 8.7 – 14.5 | 5.8 | p/(cm² · s · sr · MeV) |
| P4 | 15.0 – 44.0 | 29.0 | p/(cm² · s · sr · MeV) |
| P5 | 40.0 – 80.0 | 40.0 | p/(cm² · s · sr · MeV) |
| P6 | 80.0 – 165.0 | 85.0 | p/(cm² · s · sr · MeV) |
| P7 | 165.0 – 500.0 | 335.0 | p/(cm² · s · sr · MeV) |

### 2.2 GOES EPEAD (Energetic Proton, Electron, and Alpha Detector) — 2010 – 2020

**Satellites**: GOES-13 (2010–2017), GOES-15 (2018–Mar 2020)

The EPEAD is the successor to EPS with the same 7 proton channels (P1–P7), but carries **two sensor units**: one east-facing and one west-facing. The `cpflux` (corrected proton flux) product provides pre-computed integral fluxes at multiple energy thresholds.

| Property | Value |
|----------|-------|
| **Archive** | NOAA NCEI legacy archive |
| **Format** | Monthly CSV files (`g13_epead_cpflux_5m_*`, `g15_epead_cpflux_5m_*`) |
| **Cadence** | 5-minute averages |
| **Key variable** | `ZPGT10E` — corrected integral proton flux >10 MeV (east sensor) |
| **Fallback variable** | `ZPGT10W` (west sensor, used if east is unavailable) |
| **Quality flag** | `ZPGT10E_QUAL_FLAG` — nonzero values indicate bad data |
| **Units** | pfu (protons / cm² · s · sr) |
| **Missing flag** | `-99999` |
| **Dimensions** | ~8,928 records per month |

Available integral flux thresholds in cpflux files: >1, >5, >10, >30, >50, >60, >100 MeV. This pipeline uses only the >10 MeV product.

### 2.3 GOES-16 SGPS (Solar and Galactic Proton Sensor) — 2020 – 2025

**Satellite**: GOES-16

The SGPS is a redesigned instrument on the GOES-R series. It provides **13 differential proton channels** (P1, P2A, P2B, P3, P4, P5, P6, P7, P8A, P8B, P8C, P9, P10) spanning 1–500 MeV across **2 sensor units** (east-facing and west-facing). There is **no pre-computed >10 MeV integral proton flux** in the SGPS product; the only integral channel (`AvgIntProtonFlux`) is for >500 MeV.

| Property | Value |
|----------|-------|
| **Archive** | NOAA NGDC GOES-R archive |
| **Format** | Daily NetCDF files (`sci_sgps-l2-avg5m_g16_d{date}_{version}.nc`) |
| **Cadence** | 5-minute averages |
| **Key variable** | `AvgDiffProtonFlux` — differential proton flux |
| **Energy bounds** | `DiffProtonLowerEnergy`, `DiffProtonUpperEnergy` (in keV) |
| **Units** | protons / (cm² · sr · keV · s) |
| **Fill value** | < −1×10³⁰ |
| **Dimensions** | (288 records × 2 sensors × 13 channels) per daily file |
| **Sensor used** | Index 1 (east-facing when spacecraft upright) |

**File version strings change over time** (the adapter discovers these dynamically):

| Period | Version |
|--------|---------|
| Nov 2020 – Nov 2021 | `v1-0-1` |
| Dec 2021 – May 2022 | `v2-0-0` |
| Jun 2022 – early 2023 | `v3-0-0` |
| mid 2023 | `v3-0-1` |
| 2024 – 2025 | `v3-0-2` |

**Time variable also changes**: `L2_SciData_TimeStamp` in v1-0-1 files, `time` in v2-0-0 and later. The adapter handles both automatically.

### 2.4 SOHO COSTEP/EPHIN (1995 – 2025)

**Instrument**: Comprehensive Suprathermal and Energetic Particle Analyzer (COSTEP) — Electron Proton Helium Instrument (EPHIN)

SOHO provides an independent measurement of the energetic proton environment from the L1 Lagrange point (unlike GOES, which orbits in geostationary orbit). The EPHIN Level-3 Intensities 5-minute data product provides **4 proton channels**:

| Channel | Energy Range (MeV) | ΔE (MeV) | Column Index | Used for >10 MeV Proxy |
|---------|-------------------|-----------|-------------|----------------------|
| P4 | 4.3 – 7.8 | 3.5 | 0 | Excluded (below 10 MeV) |
| P8 | 7.8 – 25.0 | 17.2 | 1 | Partially included (10.0–25.0 only) |
| P25 | 25.0 – 40.9 | 15.9 | 2 | Fully included |
| P41 | 40.9 – 53.0 | 12.1 | 3 | Fully included |

| Property | Value |
|----------|-------|
| **Archive** | NASA CDAWeb |
| **Format** | Annual CDF files (`soho_costep-ephin_l3i-5min_{year}0101_v*.cdf`) |
| **Cadence** | 5-minute averages |
| **Key variable** | `P_int` — proton differential intensities, shape (N, 4) |
| **Units** | (cm² · s · sr · MeV)⁻¹ — confirmed by CDF metadata `UNITS` attribute |
| **Fill value** | −1×10³¹ |
| **Dimensions** | ~105,120 records per year (one file per year) |

Channel boundaries are from the COSTEP-EPHIN L3 documentation (Table 3), not inferred from representative energies.

### 2.5 GSEP Catalog (Ground Truth)

The **GSEP catalog** (Papaioannou et al.) provides validated SEP event listings for solar cycles 22–24, ending around 2017. Only **Flag==1** events (significant events crossing the SWPC >10 MeV threshold at 10 pfu) are used for validation — **159 events** across 1995–2017.

The catalog includes actual event end times (`slice_end`), unlike the NOAA legacy catalog where `end_time` is actually the peak time. This is critical for accurate event-level matching.

---

## 3. Satellite-Year Mapping

The following table shows the exact satellite and instrument used for every year. Where a satellite transition occurs mid-year, the transition month is noted.

| Year(s) | Primary Satellite | Instrument | Flux Variable | Fallback Satellite |
|---------|------------------|------------|---------------|-------------------|
| 1995–2002 | **GOES-8** | EPS | `p3_flux_ic` | — |
| 2003 Jan–Mar | **GOES-8** | EPS | `p3_flux_ic` | — |
| 2003 Apr–Dec | **GOES-12** | EPS | `p3_flux_ic` (**broken**) | **GOES-11** (then GOES-10) |
| 2004–2009 | **GOES-12** | EPS | `p3_flux_ic` (**broken**) | **GOES-11** (then GOES-10) |
| 2010 Jan–Apr | **GOES-12** | EPS | `p3_flux_ic` (**broken**) | **GOES-11** (then GOES-10) |
| 2010 May–Dec | **GOES-13** | EPEAD | `ZPGT10E` | — |
| 2011–2017 | **GOES-13** | EPEAD | `ZPGT10E` | — |
| 2018–2019 | **GOES-15** | EPEAD | `ZPGT10E` | — |
| 2020 Jan–Mar | **GOES-15** | EPEAD | `ZPGT10E` | — |
| 2020 Apr–Oct | **No GOES data** | — | — | — (data gap) |
| 2020 Nov–Dec | **GOES-16** | SGPS | Derived from `AvgDiffProtonFlux` | — |
| 2021–2025 | **GOES-16** | SGPS | Derived from `AvgDiffProtonFlux` | — |

**GOES-12 fallback explained**: NOAA never computed the corrected integral flux product (`p3_flux_ic`) for GOES-12. The column exists in the CSV files but every value is `-99999`. Rather than deriving integral flux from GOES-12's differential channels, the pipeline uses the pre-computed `p3_flux_ic` from an alternative satellite that was operational during the same period:

- **GOES-11** (priority): available Jun 2003 – Dec 2009, valid `p3_flux_ic` with thousands of positive readings per month.
- **GOES-10** (secondary): available Jan 2003 – Dec 2009, used when GOES-11 is unavailable.
- **GOES-8** (Jan–Mar 2003 only): used before decommissioning.

If no alternative satellite's integral flux is available, the pipeline falls back to deriving integral flux from GOES-12's differential channels P3–P7 (see Section 4.2).

SOHO COSTEP/EPHIN runs independently for all years 1995–2025, providing a second instrument for fusion.

---

## 4. Flux Computation

The pipeline requires **>10 MeV integral proton flux** in pfu for each instrument. The path to obtain this value differs by instrument.

### 4.1 GOES EPS: Pre-computed Integral Flux

For GOES-8 (1995–2003), the `p3_flux_ic` column provides the **corrected integral proton flux >10 MeV** directly in pfu. No computation is needed — the pipeline reads the value, replaces `-99999` with `NaN`, and applies detection.

### 4.2 GOES EPS: Differential-to-Integral Derivation (GOES-12 Fallback)

When `p3_flux_ic` is unavailable (all `-99999`, as in GOES-12), the pipeline derives >10 MeV integral flux by numerically integrating the differential proton channels P3–P7:

```
J(>10 MeV) = Σ [ I(Eᵢ) × ΔEᵢ_eff ]
```

where `I(Eᵢ)` is the differential flux in p/(cm² · s · sr · MeV) and `ΔEᵢ_eff` is the effective energy width for the >10 MeV integral.

| Channel | E_min (MeV) | E_max (MeV) | Full ΔE (MeV) | Effective ΔE (MeV) | Note |
|---------|-------------|-------------|---------------|--------------------|----|
| P3 | 8.7 | 14.5 | 5.8 | **4.5** | 14.5 − 10.0; flat-spectrum partial-bin correction |
| P4 | 15.0 | 44.0 | 29.0 | 29.0 | Full channel |
| P5 | 40.0 | 80.0 | 40.0 | 40.0 | Full channel |
| P6 | 80.0 | 165.0 | 85.0 | 85.0 | Full channel |
| P7 | 165.0 | 500.0 | 335.0 | 335.0 | Full channel |

The P3 channel starts at 8.7 MeV, below the 10 MeV threshold. Under a **flat-spectrum approximation** (constant differential flux within the channel), the fraction of P3's flux above 10 MeV is `(14.5 − 10.0) / (14.5 − 8.7) ≈ 77.6%`. The effective width of **4.5 MeV** implements this correction.

The resulting formula:

```
J(>10 MeV) ≈ I_P3 × 4.5 + I_P4 × 29.0 + I_P5 × 40.0 + I_P6 × 85.0 + I_P7 × 335.0
```

Units: `p/(cm² · s · sr · MeV) × MeV = p/(cm² · s · sr) = pfu`

### 4.3 GOES EPEAD: Pre-computed Integral Flux

For GOES-13 (2010–2017) and GOES-15 (2018–2020), the `ZPGT10E` column in the cpflux product provides the **corrected integral proton flux >10 MeV** from the east-facing sensor, directly in pfu. Records with `ZPGT10E_QUAL_FLAG != 0` are set to `NaN`. If the east sensor is unavailable, the adapter falls back to the west sensor (`ZPGT10W`).

### 4.4 GOES-16 SGPS: Differential-to-Integral Derivation

GOES-16 SGPS has no pre-computed >10 MeV integral flux. The `AvgIntProtonFlux` variable is for **>500 MeV only**, which is not usable for standard SEP detection.

The pipeline derives >10 MeV integral flux by integrating the differential channels where `DiffProtonLowerEnergy >= 10,000 keV` (= 10 MeV):

```
J(>10 MeV) = Σ [ AvgDiffProtonFlux[ch] × (UpperEnergy[ch] − LowerEnergy[ch]) ]
```

for all channels with `lower_energy >= 10,000 keV`.

- **Units**: `protons/(cm² · sr · keV · s) × keV = protons/(cm² · sr · s) = pfu`
- **Sensor**: East-facing (index 1 when spacecraft upright)
- **Fill values**: `< −1×10³⁰` are replaced with `NaN`

Energy bounds are read directly from the NetCDF variables `DiffProtonLowerEnergy` and `DiffProtonUpperEnergy` for each file, ensuring robustness across SGPS versions.

### 4.5 SOHO COSTEP/EPHIN: >10 MeV Proxy from Differential Channels

SOHO EPHIN provides differential proton intensities `P_int` in units of `(cm² · s · sr · MeV)⁻¹`. The >10 MeV proxy is computed by multiplying each channel's differential intensity by its effective energy width and summing:

```
J_proxy(>10 MeV) ≈ I_P8 × 15.0 + I_P25 × 15.9 + I_P41 × 12.1
```

| Channel | E_min (MeV) | E_max (MeV) | Full ΔE (MeV) | Effective ΔE (MeV) | Note |
|---------|-------------|-------------|---------------|--------------------|----|
| P4 | 4.3 | 7.8 | 3.5 | **0.0** | Excluded: entirely below 10 MeV |
| P8 | 7.8 | 25.0 | 17.2 | **15.0** | 25.0 − 10.0; flat-spectrum partial-bin correction |
| P25 | 25.0 | 40.9 | 15.9 | 15.9 | Full channel |
| P41 | 40.9 | 53.0 | 12.1 | 12.1 | Full channel |

The P8 channel starts at 7.8 MeV, below the 10 MeV threshold. Under a **flat-spectrum approximation**, the fraction above 10 MeV is `(25.0 − 10.0) / (25.0 − 7.8) ≈ 87.2%`, yielding an effective width of **15.0 MeV**.

Units: `(cm² · s · sr · MeV)⁻¹ × MeV = (cm² · s · sr)⁻¹ = pfu`

**This proxy covers approximately 10–53 MeV**, not 10 MeV to infinity. It is labeled as a proxy, not a true NOAA-style >10 MeV integral flux. The truncation at 53 MeV means the proxy slightly underestimates the true integral flux during events where the spectrum extends beyond 53 MeV.

---

## 5. Detection Logic

All instruments use the same detection algorithm, which applies four rules sequentially:

### Rule 1 — Threshold (Event Entry)

Flux must reach or exceed the **entry threshold of 10 pfu** (protons per cm² · s · sr). This matches the NOAA/SWPC operational definition for Solar Radiation Storm (S-scale) events.

```
above_threshold[i] = (flux[i] >= 10.0)
```

### Rule 2 — Rising Gradient (Event Entry)

The flux must be actively rising, not merely sitting above threshold at a flat level. In a sliding window of 4 consecutive gradient points (point-to-point differences), **at least 3 must be positive**:

```
gradient[i] = flux[i] − flux[i−1]
positive_count = count(gradient > 0) in window of 4
rising[i] = (positive_count >= 3)
```

**An event can only START when both Rule 1 and Rule 2 are true simultaneously.**

### Rule 3 — Hysteresis Exit

Once an event starts, it continues until flux drops below the **exit threshold of 5 pfu** AND stays below it for a **quiet period of 2 hours** (24 consecutive 5-minute points). This prevents decay-phase fragmentation, where flux oscillates around the 10 pfu threshold during the gradual decline following a real event.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Entry threshold | 10 pfu | NOAA/SWPC standard |
| Exit threshold | 5 pfu | Prevents premature event termination |
| Quiet period | 2 hours (24 points) | Prevents decay-phase fragmentation |

If flux rises back above 5 pfu before the quiet period completes, the counter resets and the event continues. When the quiet period completes, the event's end time is **backdated** to when flux first dropped below 5 pfu.

### Rule 4 — Minimum Duration

Events shorter than **30 minutes** (6 data points at 5-minute cadence) are discarded as likely noise or instrumental artifacts.

---

## 6. Multi-Instrument Fusion

After each instrument produces its own list of detected events, the fusion engine merges events across instruments into a unified catalog. Two events (from the same or different instruments) are merged if:

- They **overlap** in time, OR
- They are separated by a gap of **≤ 30 minutes**

The fusion algorithm sorts all events from all instruments by start time, then walks through them sequentially, merging any event whose start falls within `(current_end + 30 minutes)`. This handles overlapping, close, and nested intervals in a single pass.

Each fused event records which instruments contributed to it (e.g., `["GOES-13/EPEAD", "SOHO/EPHIN"]`).

---

## 7. Evaluation Methodology

Evaluation is performed at the **event level**, not the pointwise level. This is because pointwise metrics (comparing each 5-minute timestamp) are misleading for event detection due to severe class imbalance and sensitivity to event boundary definitions.

### Event-Level Matching

For each **catalog event**, check if any detected event overlaps with it (any temporal intersection counts as a match). For each **detected event**, check if any catalog event overlaps with it.

### Metrics

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| **TP** (True Positive) | Catalog events matched by a detection | Events correctly found |
| **FN** (False Negative) | Catalog events with no detection | Missed events |
| **FP** (False Positive) | Detections with no catalog match | False alarms |
| **Precision** | TP / (TP + FP) | Of all detections, fraction that are real |
| **Recall** | TP / (TP + FN) | Of all real events, fraction that were found |
| **F1** | 2 × Precision × Recall / (Precision + Recall) | Harmonic mean |
| **EDR** | TP / total catalog events | Event Detection Rate |
| **FAR** | FP / total detections | False Alarm Rate |

Aggregated metrics across all years use **micro-averaging**: sum TP, FP, FN across all years first, then compute Precision, Recall, F1 from the totals. This gives equal weight to each event rather than each year.

---

## 8. Validation Results (1995–2017)

Validated against the **GSEP catalog (Flag==1)** — 159 significant SEP events across 1995–2017.

### Per-Instrument Totals (Micro-averaged)

| Metric | GOES | SOHO | Fused (GOES + SOHO) |
|--------|------|------|-----|
| **TP** | 157 | 150 | **159** |
| **FP** | 4 | 11 | 13 |
| **FN** | 2 | 9 | **0** |
| **Precision** | 0.9752 | 0.9317 | **0.9244** |
| **Recall** | 0.9874 | 0.9434 | **1.0000** |
| **F1** | 0.9812 | 0.9375 | **0.9607** |
| **EDR** | 0.9874 | 0.9434 | **1.0000** |
| **FAR** | 0.0290 | 0.0833 | **0.0922** |

Total detections: GOES 138, SOHO 132, Fused 141. Wall-clock time: ~3.9 minutes.

### Key Results

- **Fused recall is 1.0000**: every single one of the 159 GSEP catalog events is detected. Zero misses.
- **Fused F1 is 0.96** with a **FAR of ~9.2%**.
- The 13 fused false positives are predominantly real elevated-flux periods (sub-catalog events or extended decay tails) that do not match any GSEP Flag==1 entry.
- Hysteresis reduced the fused FAR from ~22% (before) to ~9% (after), while simultaneously improving recall from 0.97 to 1.00.

---

## 9. Known Limitations and Data Gaps

### 9.1 GOES-12 Missing Integral Flux (Apr 2003 – Apr 2010)

NOAA never computed `p3_flux_ic` for GOES-12. The pipeline resolves this by using `p3_flux_ic` from GOES-11 or GOES-10 (alternative satellites operational during the same period). This was validated to produce better results than deriving integral flux from GOES-12's own differential channels.

### 9.2 2020 Data Gap (April – October)

GOES-15 EPEAD data ends after March 2020 (satellite decommissioned). GOES-16 SGPS `avg5m` data does not begin until November 2020. During this 7-month gap, the pipeline has **no GOES data**. SOHO-only detection covers this period, but fusion is degraded to a single instrument.

### 9.3 Flat-Spectrum Approximation for Partial-Bin Correction

Both the SOHO P8 channel (7.8–25 MeV) and the GOES EPS P3 channel (8.7–14.5 MeV) extend below the 10 MeV threshold. The pipeline excludes the sub-10 MeV portion using a **flat-spectrum approximation** — assuming constant differential flux within the channel. In reality, SEP spectra decrease with energy (roughly as a power law E⁻ᵧ), so the sub-10 MeV portion contributes a disproportionately larger share than the flat model assumes. This means:

- The correction slightly **overestimates** the >10 MeV contribution (includes a bit more flux from near 10 MeV than it should).
- The impact on detection is minimal: the correction only matters for marginal events near the 10 pfu threshold, and the overestimate direction means slightly higher sensitivity (fewer misses) at the cost of marginally more false alarms.

A power-law correction (estimating spectral index per timestep) was considered but rejected as introducing a noisy free parameter without meaningfully improving detection accuracy.

### 9.4 SOHO Proxy Truncation at 53 MeV

The SOHO >10 MeV proxy only covers **10–53 MeV**, not 10 MeV to infinity. Events with significant flux contribution above 53 MeV will be underestimated. Since most SEP flux is concentrated at lower energies, this truncation has limited practical impact on detection.

### 9.5 SGPS Version and Format Changes

GOES-16 SGPS NetCDF files change version strings and internal variable names over time:
- **v1-0-1** (2020–2021): time variable is `L2_SciData_TimeStamp`
- **v2-0-0** and later (2022+): time variable is `time`

The pipeline discovers the version dynamically by listing the NGDC directory for each month. Results are cached to avoid repeated HTTP calls.

### 9.6 No External Catalog for 2018–2025

The GSEP catalog ends around 2017. Detections for 2018–2025 are **original pipeline detections** with no external validation. They use the same detection logic that achieved Precision=0.92, Recall=1.00, F1=0.96 on the validated 1995–2017 period.

---

## 10. Project Structure

```
sep_detection_pipeline/
├── sep_core/                          # Core detection library
│   ├── adapters/                      # Instrument-specific data loaders
│   │   ├── base.py                    # Abstract base adapter (Template Method)
│   │   ├── goes.py                    # GOES EPS/EPEAD/SGPS (1995–2025)
│   │   └── soho.py                    # SOHO COSTEP/EPHIN (1995–2025)
│   ├── evaluation/                    # Evaluation against catalogs
│   │   ├── gsep_catalog.py            # GSEP catalog loader (Flag==1 filter)
│   │   ├── matching.py                # Event-level matching (TP/FP/FN/Prec/Rec/F1)
│   │   └── metrics.py                 # Pointwise evaluation (legacy)
│   ├── detections.py                  # DetectionResult data contract
│   ├── events.py                      # Mask ↔ event interval conversion
│   ├── fusion.py                      # Multi-instrument event fusion
│   └── threshold.py                   # Detection logic (hysteresis state machine)
├── experiments/
│   ├── validation/                    # Phase 1: Validated against GSEP (1995–2017)
│   │   ├── run_full_validation.py     # Full multi-year validation with summary
│   │   ├── run_test.py               # Single-year test harness
│   │   └── diagnose_false_positives.py # Deep FP analysis and classification
│   └── detection/                     # Phase 2: Original detections (2018–2025)
│       └── run_detection_2018_2025.py # Generate new event catalog
├── data/cache/                        # Downloaded instrument data
│   ├── goes/                          # GOES monthly CSVs + daily NetCDFs
│   └── soho/                          # SOHO annual CDFs
├── output/
│   ├── full_validation/               # Validation outputs (per-year CSVs, summary)
│   └── detection_2018_2025/           # Detection outputs (per-year CSVs, summary)
├── GSEP_List.csv                      # GSEP catalog (ground truth for 1995–2017)
└── noaa_sep_catalog_1995_2025.csv     # NOAA legacy catalog (reference only)
```

---

## 11. Usage

```bash
cd sep_detection_pipeline/

# Phase 1: Run validation against GSEP catalog (1995–2017)
python -m experiments.validation.run_full_validation

# Phase 1: Test a single year
python -m experiments.validation.run_test

# Phase 1: Diagnose false positives for a specific year
python -m experiments.validation.diagnose_false_positives

# Phase 2: Generate original detections (2018–2025)
python -m experiments.detection.run_detection_2018_2025
```

### Requirements

- Python 3.8+
- `numpy`, `pandas` — core numerical computation
- `requests` — downloading data from NOAA and CDAWeb archives
- `cdflib` — reading SOHO CDF files
- `xarray`, `netcdf4` — reading GOES-16 SGPS NetCDF files

---

## 12. References

### Data Archives

- **GOES Legacy (EPS/EPEAD)**: [NOAA NCEI Space Environment Monitor Archive](https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/avg/2020/03/)
- **GOES-16 SGPS**: [NOAA NGDC GOES-R SGPS L2 avg5m](https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l2/data/sgps-l2-avg5m/2021/)
- **SOHO COSTEP/EPHIN**: [NASA CDAWeb EPHIN L3I 5-min](https://cdaweb.gsfc.nasa.gov/pub/data/soho/costep/ephin_l3i-5min/2009/)

### Catalogs

- **GSEP Catalog**: Papaioannou, A. et al. — Solar Energetic Particle Event catalog (Flag==1 significant events, ~245 total, 159 in 1995–2017)

### Instrument Documentation

- **COSTEP-EPHIN L3 Documentation**: `DOCUMENTATION-COSTEP-EPHIN-L3-20220201.pdf` — Channel energy ranges (Table 3), response factors, intensity computation
- **GOES EPS/EPEAD**: NOAA Space Weather Prediction Center — corrected integral proton flux products, quality flag definitions
- **GOES-R SGPS**: SEISS instrument documentation — differential channel definitions, sensor orientation conventions
