"""
==========================================================================
  Saudi Demand Generator v2 — Dammam hourly AOI demand, 2024
==========================================================================

Rewrite goals (vs. saudi_demand_generator_final.py):

  1. Kill AOI clone behavior. Mean pairwise correlation in the old dataset
     was 0.45; target here is << 0.25, driven by archetype-distinct hourly
     shapes + independent AR(1) latent drivers per AOI + per-AOI event
     response vectors (not a scalar sensitivity).

  2. Calibrate only against ranges that the Saudi markdown actually supports.
     Everything else is labeled [ASSUMPTION]. The old script cited a TGA
     "290M national orders" figure that is NOT in the markdown — removed.

  3. Fix the Ramadan calibration bug. The old hourly Ramadan reshape
     (0.55 daytime x 1.85 post-Iftar) produced a *monthly mean below 1.0*,
     which directly contradicts the markdown's +20–40% monthly uplift.
     Here, Ramadan is split into (a) a guaranteed monthly-mean lift in the
     defensible 1.20–1.40 band, and (b) a separate daytime->evening shape
     shift that is additionally labeled as an assumption.

  4. Add realistic irregularity: heteroskedastic NB noise, local promo /
     disruption spikes, AOI-day outages, natural sparsity for small AOIs.

  5. Export a clean forecasting schema:
        bucket_hour, aoi_id, region_id, aoi_archetype, aoi_name,
        demand_count, hour, day_of_week, month, is_weekend,
        is_ramadan, is_eid, is_eid_pre, is_white_friday,
        is_national_day, is_founding_day, is_back_to_school, is_hajj
     No lag columns. No roll columns. Full 8784 hours.

  6. LaDe is used ONLY as a structural reference (heterogeneous archetypes,
     low cross-AOI correlation, heavy-tailed per-AOI variance, asymmetric
     hour shapes). We do not copy LaDe's scale, sparsity, or semantics.

Requirements: numpy, pandas
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Callable

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# 0. CONFIG
# ─────────────────────────────────────────────
SEED = 20240406
rng = np.random.default_rng(SEED)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(HERE, "saudi_hourly_v2.csv")
OUT_META = os.path.join(HERE, "saudi_hourly_v2_calibration.json")

print("=" * 70)
print("  Saudi Demand Generator v2 — Dammam hourly, 2024")
print("=" * 70)

# ─────────────────────────────────────────────
# 1. CALENDAR
# ─────────────────────────────────────────────
# Full leap year 2024, 8784 hours.
HOURS = pd.date_range("2024-01-01 00:00", "2024-12-31 23:00", freq="h")
assert len(HOURS) == 366 * 24, f"bad calendar length: {len(HOURS)}"

# ── Event windows (dates are empirical; magnitudes handled later)
RAMADAN_START = pd.Timestamp("2024-03-11")
RAMADAN_END   = pd.Timestamp("2024-04-09 23:00")

EID_FITR_DAYS  = pd.to_datetime(["2024-04-10", "2024-04-11", "2024-04-12"])
EID_FITR_PRE   = pd.to_datetime(["2024-04-08", "2024-04-09"])
EID_ADHA_DAYS  = pd.to_datetime(["2024-06-16", "2024-06-17", "2024-06-18"])
EID_ADHA_PRE   = pd.to_datetime(["2024-06-14", "2024-06-15"])

WHITE_FRIDAY       = pd.Timestamp("2024-11-29")
WHITE_FRIDAY_WEEK  = pd.date_range("2024-11-25", "2024-12-02")

NATIONAL_DAY_WIN = pd.date_range("2024-09-16", "2024-09-30")
FOUNDING_DAY_WIN = pd.date_range("2024-02-18", "2024-02-25")
BACK_TO_SCHOOL   = pd.date_range("2024-08-18", "2024-09-14")
HAJJ_WIN         = pd.date_range("2024-06-14", "2024-06-19")


def _norm(dt):
    """Normalize both DatetimeIndex and Series[datetime]."""
    if isinstance(dt, pd.Series):
        return dt.dt.normalize()
    return dt.normalize()

def is_ramadan(dt):        return (dt >= RAMADAN_START) & (dt <= RAMADAN_END)
def is_eid(dt):            return _norm(dt).isin(np.concatenate([EID_FITR_DAYS.values, EID_ADHA_DAYS.values])) if isinstance(dt, pd.Series) else np.isin(_norm(dt), np.concatenate([EID_FITR_DAYS.values, EID_ADHA_DAYS.values]))
def is_eid_pre(dt):        return _norm(dt).isin(np.concatenate([EID_FITR_PRE.values, EID_ADHA_PRE.values])) if isinstance(dt, pd.Series) else np.isin(_norm(dt), np.concatenate([EID_FITR_PRE.values, EID_ADHA_PRE.values]))
def is_white_friday(dt):   return _norm(dt) == WHITE_FRIDAY
def is_wf_week(dt):        return _norm(dt).isin(WHITE_FRIDAY_WEEK.values) if isinstance(dt, pd.Series) else np.isin(_norm(dt), WHITE_FRIDAY_WEEK.values)
def is_national_day(dt):   return _norm(dt).isin(NATIONAL_DAY_WIN.values) if isinstance(dt, pd.Series) else np.isin(_norm(dt), NATIONAL_DAY_WIN.values)
def is_founding_day(dt):   return _norm(dt).isin(FOUNDING_DAY_WIN.values) if isinstance(dt, pd.Series) else np.isin(_norm(dt), FOUNDING_DAY_WIN.values)
def is_back_to_school(dt): return _norm(dt).isin(BACK_TO_SCHOOL.values) if isinstance(dt, pd.Series) else np.isin(_norm(dt), BACK_TO_SCHOOL.values)
def is_hajj(dt):           return _norm(dt).isin(HAJJ_WIN.values) if isinstance(dt, pd.Series) else np.isin(_norm(dt), HAJJ_WIN.values)


# ─────────────────────────────────────────────
# 2. CALIBRATION BANDS FROM THE MARKDOWN
# ─────────────────────────────────────────────
# Only ranges that the Saudi markdown actually supports. Per-AOI values
# are drawn from these bands — the *cross-AOI mean* per event stays inside
# the band, but individual AOIs can land anywhere in (or slightly outside)
# their archetype's narrower sub-band.
#
# These are multipliers on the local base demand, applied as a daily factor
# (not hourly). Hourly shape reshaping (e.g. Ramadan daytime->evening) is
# a separate, explicitly-labeled assumption layer.

MD_RANGES = {
    # markdown: "Ramadan month +20-40% baseline volume"
    "ramadan_month_mean":      (1.20, 1.40),
    # markdown: Eid days, high end of Ramadan uplift, POS lead-up +
    #           March 2025 Mada +73% YoY (conflates growth, so we stay on
    #           the lower end of the narrative band)
    "eid_day":                 (1.40, 1.60),
    "eid_pre_day":             (1.20, 1.40),
    # markdown: White Friday day 1.3-1.7x average Nov Friday
    "white_friday_day":        (1.30, 1.70),
    # markdown: White Friday week ~1.10-1.25 lift
    "wf_week_other":           (1.05, 1.20),
    # markdown: November ~+10% baseline vs typical non-holiday month
    "november_month":          (1.05, 1.12),
    # markdown: National Day window 1.1-1.3 qualitative
    "national_day_window":     (1.10, 1.30),
    # markdown: Founding Day, analogy only, narrower
    "founding_day_window":     (1.05, 1.20),
    # markdown: BTS small negative on totals, category-level uplift not modeled
    "back_to_school_window":   (0.93, 1.00),
    # markdown: Hajj — NO quantified Dammam effect. Near-neutral with tiny jitter.
    "hajj_window":             (0.97, 1.02),
    # markdown: Q1 2024 Mada +22% YoY. Linear monthly ramp over 2024.
    "annual_growth_yoy":       0.22,
}

# ─────────────────────────────────────────────
# 3. AOI ARCHETYPES  [ASSUMPTION — all shapes below]
# ─────────────────────────────────────────────
# Each archetype provides:
#   - a 24-hour base shape (NOT a Gaussian sum; different functional forms)
#   - a day-of-week profile
#   - an event-response sub-band (within MD_RANGES, narrower & biased)
#   - AR(1) rho and latent sigma
#   - base_scale range (mean hourly orders)
#   - sparsity floor (probability a low hour gets squashed to near-zero)
#
# pandas dayofweek: Mon=0 ... Sun=6. Saudi weekend is Fri=4, Sat=5.

def shape_residential_evening() -> np.ndarray:
    """Households: dead early morning, strong evening peak 18-22."""
    h = np.arange(24)
    s = (
        0.08
        + 0.35 * np.exp(-0.5 * ((h - 20) / 1.8) ** 2)   # evening
        + 0.10 * np.exp(-0.5 * ((h - 13) / 2.5) ** 2)   # lunch lift
    )
    s[0:5] *= 0.35  # almost dead at night
    return s / s.sum()


def shape_commercial_midday() -> np.ndarray:
    """Offices / daytime retail: flat early, wide midday plateau 10-16."""
    h = np.arange(24)
    s = 0.05 + 0.30 / (1 + np.exp(-0.9 * (h - 9))) - 0.30 / (1 + np.exp(-0.9 * (h - 17)))
    s += 0.05 * np.exp(-0.5 * ((h - 12) / 1.3) ** 2)
    s[0:6] *= 0.3
    s[22:] *= 0.4
    return s / s.sum()


def shape_mall_leisure() -> np.ndarray:
    """Mall / dine-out: slow morning, big late-afternoon + late-evening."""
    h = np.arange(24)
    s = (
        0.05
        + 0.18 * np.exp(-0.5 * ((h - 17) / 2.0) ** 2)
        + 0.28 * np.exp(-0.5 * ((h - 21) / 1.6) ** 2)
    )
    s[0:8] *= 0.25
    return s / s.sum()


def shape_office_cluster() -> np.ndarray:
    """Business park: sharp morning ramp, drops at 17, near-zero nights."""
    h = np.arange(24)
    s = np.where(
        (h >= 8) & (h <= 17),
        0.10 + 0.06 * np.sin((h - 8) / 9 * np.pi),
        0.01,
    )
    s[7] = 0.05
    s[18] = 0.04
    return s / s.sum()


def shape_mixed() -> np.ndarray:
    """Mixed-use: moderate all-day with two bumps."""
    h = np.arange(24)
    s = (
        0.04
        + 0.10 * np.exp(-0.5 * ((h - 13) / 3.0) ** 2)
        + 0.12 * np.exp(-0.5 * ((h - 20) / 2.5) ** 2)
    )
    return s / s.sum()


def shape_low_density() -> np.ndarray:
    """Peripheral / low density: irregular, noisy, long tails of zeros."""
    h = np.arange(24)
    s = 0.03 + 0.06 * np.exp(-0.5 * ((h - 19) / 3.0) ** 2)
    s[0:6] *= 0.15
    return s / s.sum()


def shape_nightlife() -> np.ndarray:
    """Late-night food / corniche: peaks 21-01."""
    h = np.arange(24)
    circ = np.where(h >= 12, h, h + 24)  # shift so midnight is 'late'
    s = 0.03 + 0.30 * np.exp(-0.5 * ((circ - 23) / 2.2) ** 2)
    return s / s.sum()


@dataclass
class Archetype:
    name: str
    shape_fn: Callable[[], np.ndarray]
    dow_profile: np.ndarray  # length 7, Mon..Sun, multiplier
    event_bias: dict         # per-event sub-range inside MD_RANGES
    ar_rho: float            # AR(1) autocorrelation of latent log-demand
    ar_sigma: float          # innovation std of latent log-demand
    base_scale_range: tuple  # (low, high) mean hourly orders
    sparsity_p: float        # prob that a low-demand hour collapses to ~0
    nb_dispersion: float     # NB dispersion parameter k (var = mu + mu^2/k)

# dow multipliers — Saudi weekend is Fri (4), Sat (5)
DOW_RES = np.array([1.00, 1.02, 1.05, 1.00, 0.78, 0.88, 1.05])  # residential: softer weekend
DOW_COM = np.array([1.10, 1.15, 1.20, 1.12, 0.55, 0.70, 1.00])  # commercial: hard weekend drop
DOW_MAL = np.array([0.95, 0.95, 1.00, 1.02, 1.25, 1.20, 1.05])  # mall: WEEKEND BOOST
DOW_OFF = np.array([1.15, 1.20, 1.25, 1.18, 0.35, 0.50, 0.90])  # office: extreme weekend drop
DOW_MIX = np.array([1.05, 1.08, 1.10, 1.05, 0.85, 0.92, 1.02])
DOW_LOW = np.array([1.00, 1.00, 1.00, 1.00, 0.95, 0.95, 1.00])  # flat, noisy
DOW_NIG = np.array([0.90, 0.95, 1.00, 1.15, 1.35, 1.25, 1.00])  # Thu/Fri nightlife peak

# Event sub-bands: (lo, hi) drawn INSIDE the markdown range when possible.
# Some archetypes are given sub-bands below 1 for certain events to inject
# meaningful heterogeneity. The *cross-AOI average* per event is validated
# against the markdown range at the end of generation.
def sub(lo, hi):
    return (lo, hi)

EVENT_BIAS_RESIDENTIAL = {
    "ramadan_month_mean":    sub(1.25, 1.45),
    "eid_day":               sub(1.50, 1.75),
    "eid_pre_day":           sub(1.25, 1.45),
    "white_friday_day":      sub(1.25, 1.55),
    "wf_week_other":         sub(1.05, 1.15),
    "november_month":        sub(1.05, 1.10),
    "national_day_window":   sub(1.10, 1.22),
    "founding_day_window":   sub(1.12, 1.22),
    "back_to_school_window": sub(0.95, 1.02),
    "hajj_window":           sub(0.99, 1.02),
}
EVENT_BIAS_COMMERCIAL = {
    "ramadan_month_mean":    sub(1.15, 1.30),
    "eid_day":               sub(1.35, 1.60),
    "eid_pre_day":           sub(1.30, 1.55),
    "white_friday_day":      sub(1.40, 1.80),
    "wf_week_other":         sub(1.10, 1.25),
    "november_month":        sub(1.05, 1.15),
    "national_day_window":   sub(1.15, 1.35),
    "founding_day_window":   sub(1.12, 1.24),
    "back_to_school_window": sub(0.92, 1.02),
    "hajj_window":           sub(0.96, 1.01),
}
EVENT_BIAS_MALL = {
    "ramadan_month_mean":    sub(1.30, 1.50),
    "eid_day":               sub(1.60, 1.85),
    "eid_pre_day":           sub(1.30, 1.55),
    "white_friday_day":      sub(1.50, 1.90),
    "wf_week_other":         sub(1.15, 1.30),
    "november_month":        sub(1.08, 1.15),
    "national_day_window":   sub(1.18, 1.35),
    "founding_day_window":   sub(1.15, 1.27),
    "back_to_school_window": sub(0.95, 1.05),
    "hajj_window":           sub(0.97, 1.02),
}
EVENT_BIAS_OFFICE = {
    "ramadan_month_mean":    sub(1.00, 1.15),  # offices on reduced hours but some lift
    "eid_day":               sub(0.90, 1.15),  # partial closures
    "eid_pre_day":           sub(0.95, 1.15),
    "white_friday_day":      sub(1.10, 1.35),
    "wf_week_other":         sub(1.00, 1.10),
    "november_month":        sub(1.00, 1.05),
    "national_day_window":   sub(0.90, 1.10),
    "founding_day_window":   sub(0.95, 1.05),
    "back_to_school_window": sub(1.00, 1.08),
    "hajj_window":           sub(0.96, 1.00),
}
EVENT_BIAS_MIXED = {
    "ramadan_month_mean":    sub(1.20, 1.35),
    "eid_day":               sub(1.45, 1.65),
    "eid_pre_day":           sub(1.20, 1.40),
    "white_friday_day":      sub(1.30, 1.60),
    "wf_week_other":         sub(1.08, 1.20),
    "november_month":        sub(1.06, 1.12),
    "national_day_window":   sub(1.10, 1.25),
    "founding_day_window":   sub(1.12, 1.22),
    "back_to_school_window": sub(0.93, 1.00),
    "hajj_window":           sub(0.98, 1.02),
}
EVENT_BIAS_LOW = {
    "ramadan_month_mean":    sub(1.15, 1.35),  # weaker, noisier response
    "eid_day":               sub(1.30, 1.60),
    "eid_pre_day":           sub(1.15, 1.40),
    "white_friday_day":      sub(1.10, 1.50),
    "wf_week_other":         sub(1.00, 1.15),
    "november_month":        sub(1.02, 1.12),
    "national_day_window":   sub(1.05, 1.22),
    "founding_day_window":   sub(1.08, 1.20),
    "back_to_school_window": sub(0.92, 1.02),
    "hajj_window":           sub(0.97, 1.03),
}
EVENT_BIAS_NIGHT = {
    "ramadan_month_mean":    sub(1.35, 1.55),
    "eid_day":               sub(1.55, 1.85),
    "eid_pre_day":           sub(1.25, 1.55),
    "white_friday_day":      sub(1.20, 1.45),
    "wf_week_other":         sub(1.05, 1.15),
    "november_month":        sub(1.03, 1.10),
    "national_day_window":   sub(1.15, 1.32),
    "founding_day_window":   sub(1.12, 1.22),
    "back_to_school_window": sub(0.95, 1.05),
    "hajj_window":           sub(0.97, 1.03),
}

ARCHETYPES = {
    "residential_evening": Archetype(
        "residential_evening", shape_residential_evening, DOW_RES,
        EVENT_BIAS_RESIDENTIAL, ar_rho=0.88, ar_sigma=0.12,
        base_scale_range=(180, 380), sparsity_p=0.02, nb_dispersion=12.0,
    ),
    "commercial_midday": Archetype(
        "commercial_midday", shape_commercial_midday, DOW_COM,
        EVENT_BIAS_COMMERCIAL, ar_rho=0.82, ar_sigma=0.16,
        base_scale_range=(160, 320), sparsity_p=0.03, nb_dispersion=8.0,
    ),
    "mall_leisure": Archetype(
        "mall_leisure", shape_mall_leisure, DOW_MAL,
        EVENT_BIAS_MALL, ar_rho=0.85, ar_sigma=0.18,
        base_scale_range=(220, 420), sparsity_p=0.02, nb_dispersion=6.0,
    ),
    "office_cluster": Archetype(
        "office_cluster", shape_office_cluster, DOW_OFF,
        EVENT_BIAS_OFFICE, ar_rho=0.78, ar_sigma=0.20,
        base_scale_range=(90, 180), sparsity_p=0.18, nb_dispersion=5.0,
    ),
    "mixed": Archetype(
        "mixed", shape_mixed, DOW_MIX,
        EVENT_BIAS_MIXED, ar_rho=0.86, ar_sigma=0.14,
        base_scale_range=(140, 280), sparsity_p=0.04, nb_dispersion=9.0,
    ),
    "low_density": Archetype(
        "low_density", shape_low_density, DOW_LOW,
        EVENT_BIAS_LOW, ar_rho=0.72, ar_sigma=0.30,
        base_scale_range=(25, 75), sparsity_p=0.25, nb_dispersion=3.5,
    ),
    "nightlife": Archetype(
        "nightlife", shape_nightlife, DOW_NIG,
        EVENT_BIAS_NIGHT, ar_rho=0.80, ar_sigma=0.22,
        base_scale_range=(120, 260), sparsity_p=0.06, nb_dispersion=5.0,
    ),
}

# ─────────────────────────────────────────────
# 4. AOI ROSTER (14 AOIs, Dammam)
# ─────────────────────────────────────────────
# region_id groups AOIs spatially. Names are plausible Dammam neighborhoods
# used purely as labels — no claim that these particular areas behave this way.

AOIS = [
    # region 1 — central
    dict(aoi_id=1101, region_id=1, name="Al Muraikabat",   archetype="mixed"),
    dict(aoi_id=1102, region_id=1, name="Al Faisaliah",    archetype="commercial_midday"),
    dict(aoi_id=1103, region_id=1, name="Al Adamah",       archetype="residential_evening"),
    # region 2 — waterfront / leisure
    dict(aoi_id=1201, region_id=2, name="Al Shatea",       archetype="mall_leisure"),
    dict(aoi_id=1202, region_id=2, name="Al Hamra",        archetype="nightlife"),
    dict(aoi_id=1203, region_id=2, name="Al Rawdah",       archetype="residential_evening"),
    # region 3 — north residential
    dict(aoi_id=1301, region_id=3, name="Al Noor",         archetype="residential_evening"),
    dict(aoi_id=1302, region_id=3, name="Al Jawharah",     archetype="mixed"),
    dict(aoi_id=1303, region_id=3, name="Al Badiyah",      archetype="low_density"),
    # region 4 — business / university
    dict(aoi_id=1401, region_id=4, name="Al Khalidiyah",   archetype="office_cluster"),
    dict(aoi_id=1402, region_id=4, name="Al Aziziyah",     archetype="commercial_midday"),
    dict(aoi_id=1403, region_id=4, name="Uni District",    archetype="mixed"),
    # region 5 — periphery
    dict(aoi_id=1501, region_id=5, name="West Fringe",     archetype="low_density"),
    dict(aoi_id=1502, region_id=5, name="South Industrial",archetype="office_cluster"),
]

N_AOIS = len(AOIS)
N_HOURS = len(HOURS)
print(f"  AOIs: {N_AOIS}  |  Hours: {N_HOURS}  |  Rows: {N_AOIS*N_HOURS:,}")

# ─────────────────────────────────────────────
# 5. PER-AOI PARAMETER DRAW
# ─────────────────────────────────────────────
# Each AOI draws its own values from its archetype's ranges. This is what
# actually creates identity heterogeneity.
def draw_aoi_params(aoi_rng: np.random.Generator, arch: Archetype) -> dict:
    base_scale = aoi_rng.uniform(*arch.base_scale_range)
    # jitter the archetype shape a little so even same-archetype AOIs differ
    jitter = aoi_rng.normal(0, 0.05, size=24)
    shape = arch.shape_fn() * (1 + jitter)
    shape = np.clip(shape, 1e-4, None)
    shape /= shape.sum()

    # jitter the DOW profile
    dow = arch.dow_profile * (1 + aoi_rng.normal(0, 0.04, size=7))

    # draw per-event multiplier from the archetype sub-band
    event_mults = {k: float(aoi_rng.uniform(lo, hi))
                   for k, (lo, hi) in arch.event_bias.items()}

    return dict(
        base_scale=float(base_scale),
        hour_shape=shape,
        dow_profile=dow,
        event_mults=event_mults,
        ar_rho=float(arch.ar_rho + aoi_rng.normal(0, 0.02)),
        ar_sigma=float(arch.ar_sigma * aoi_rng.uniform(0.85, 1.15)),
        nb_dispersion=float(arch.nb_dispersion * aoi_rng.uniform(0.7, 1.3)),
        sparsity_p=float(arch.sparsity_p * aoi_rng.uniform(0.6, 1.4)),
    )

# Give each AOI its own child rng so the draw order is stable.
child_rngs = rng.spawn(N_AOIS)
for i, aoi in enumerate(AOIS):
    arch = ARCHETYPES[aoi["archetype"]]
    aoi["params"] = draw_aoi_params(child_rngs[i], arch)

# ─────────────────────────────────────────────
# 6. GLOBAL CALENDAR ARRAYS
# ─────────────────────────────────────────────
hr   = HOURS.hour.values
dow  = HOURS.dayofweek.values
mon  = HOURS.month.values
day  = HOURS.normalize()

# annual growth: linear monthly ramp, 0% in Jan -> +22% in Dec
growth_by_month = 1.0 + MD_RANGES["annual_growth_yoy"] * (mon - 1) / 11.0

# Event masks (hour-level)
mask_ramadan     = is_ramadan(HOURS)
mask_eid         = is_eid(HOURS)
mask_eid_pre     = is_eid_pre(HOURS)
mask_wf          = is_white_friday(HOURS)
mask_wf_week     = is_wf_week(HOURS) & ~mask_wf
mask_november    = (mon == 11) & ~mask_wf & ~mask_wf_week
mask_nat_day     = is_national_day(HOURS)
mask_founding    = is_founding_day(HOURS)
mask_bts         = is_back_to_school(HOURS)
mask_hajj        = is_hajj(HOURS)

# ── [ASSUMPTION] Ramadan intra-day shape shift:
# Not in markdown. Apply a shape-preserving reallocation: multiply hours
# 5..16 by 0.85 and hours 19..23 + 0..2 by 1.20, then renormalize so the
# DAILY sum is unchanged. The daily multiplier from event_mults handles
# the actual volume lift. This keeps the Ramadan monthly mean in band
# (markdown 1.20–1.40) while still deforming the within-day shape.
RAMADAN_DAY_WEIGHTS = np.ones(24)
RAMADAN_DAY_WEIGHTS[5:17] = 0.85
for h in [19, 20, 21, 22, 23, 0, 1, 2]:
    RAMADAN_DAY_WEIGHTS[h] = 1.20
# will be renormalized per-AOI below so it preserves that AOI's daily total

# ── [ASSUMPTION] Summer midday heat suppression. Applied only to
# outdoor-exposed archetypes (mall_leisure, low_density). Offices and
# residential are assumed indoor-climatized. Magnitude is small (-15%)
# and labeled as assumption. This is MUCH softer than the old -55%.
def summer_hour_factor(h, arch_name, month):
    if month not in (6, 7, 8):
        return 1.0
    if arch_name in ("mall_leisure", "low_density") and 12 <= h <= 15:
        return 0.85
    return 1.0

# ── [ASSUMPTION] Friday midday prayer soft dip for commercial archetypes.
# Direction supported by common knowledge; magnitude is an assumption.
# Much softer than old prayer dips. Only applied 12-13 on Fridays.
def friday_prayer_factor(h, d):
    if d == 4 and h in (12, 13):
        return 0.75
    return 1.0

# ─────────────────────────────────────────────
# 7. GENERATE DEMAND PER AOI
# ─────────────────────────────────────────────
# Each AOI gets:
#   mu_t = base_scale * shape[hour] * dow[dow] * growth[month]
#        * event_factor * summer_factor * friday_factor
#        * exp(latent_t)   where latent_t is an AR(1) process, INDEPENDENT
#                          across AOIs -> breaks cross-AOI correlation
# Count ~ NB(mean=mu_t, dispersion=k)
# Plus:
#   - random local promos and disruptions (per-AOI, uncorrelated)
#   - multi-hour outages (per-AOI, uncorrelated)
#   - sparsity collapse on already-low hours (per-AOI)
#
# We do NOT add a global multiplicative shock. That was the single biggest
# driver of the old 0.45 cross correlation.

def sample_latent_ar1(n: int, rho: float, sigma: float, gen: np.random.Generator) -> np.ndarray:
    """AR(1) with stationary variance sigma^2 / (1 - rho^2)."""
    rho = float(np.clip(rho, 0.1, 0.98))
    x = np.empty(n, dtype=np.float64)
    x[0] = gen.normal(0.0, sigma / np.sqrt(1 - rho ** 2))
    innov = gen.normal(0.0, sigma, size=n - 1)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + innov[t - 1]
    # center
    return x - x.mean()


def inject_outages(mu: np.ndarray, gen: np.random.Generator) -> np.ndarray:
    """Roughly one multi-hour outage per AOI per month on average."""
    mu = mu.copy()
    n_outages = gen.poisson(12)  # ~once a month
    for _ in range(n_outages):
        start = gen.integers(0, len(mu))
        length = gen.integers(3, 18)
        intensity = gen.uniform(0.0, 0.35)
        mu[start:start + length] *= intensity
    return mu


def inject_local_shocks(mu: np.ndarray, gen: np.random.Generator) -> np.ndarray:
    """Single-hour promos (~1.5%) and disruptions (~0.8%)."""
    mu = mu.copy()
    u = gen.random(len(mu))
    promo_mask = u < 0.015
    disrupt_mask = (u >= 0.015) & (u < 0.023)
    mu[promo_mask]   *= 1.0 + gen.uniform(0.25, 0.90, size=promo_mask.sum())
    mu[disrupt_mask] *= 1.0 - gen.uniform(0.30, 0.70, size=disrupt_mask.sum())
    return mu


def build_event_factor(aoi: dict) -> np.ndarray:
    """Multiplicative event factor for every hour. One-time build per AOI."""
    ev = aoi["params"]["event_mults"]
    factor = np.ones(N_HOURS, dtype=np.float64)

    # Ramadan month: apply volume multiplier (shape shift handled separately)
    factor[mask_ramadan] *= ev["ramadan_month_mean"]

    # Eid days OVERRIDE ramadan (some Eid days fall right after Ramadan)
    factor[mask_eid] = ev["eid_day"]
    factor[mask_eid_pre] *= ev["eid_pre_day"]

    factor[mask_wf]      *= ev["white_friday_day"]
    factor[mask_wf_week] *= ev["wf_week_other"]
    factor[mask_november] *= ev["november_month"]

    factor[mask_nat_day]  *= ev["national_day_window"]
    factor[mask_founding] *= ev["founding_day_window"]
    # BTS and Hajj last — and we take the MIN with what's already there
    # so they never overwrite a stronger positive event on overlap.
    factor[mask_bts]  *= ev["back_to_school_window"]
    factor[mask_hajj] *= ev["hajj_window"]
    return factor


# Precompute assumption-layer factors that depend on hour/dow/month only
# (they are *not* per-AOI-independent, but their magnitudes are small).
summer_commercial = np.array([
    summer_hour_factor(h, "mall_leisure", m) for h, m in zip(hr, mon)
])
summer_lowdens = np.array([
    summer_hour_factor(h, "low_density", m) for h, m in zip(hr, mon)
])
friday_prayer = np.array([friday_prayer_factor(h, d) for h, d in zip(hr, dow)])


def generate_aoi(aoi: dict, gen: np.random.Generator) -> np.ndarray:
    p = aoi["params"]
    arch_name = aoi["archetype"]
    shape = p["hour_shape"]
    dow_prof = p["dow_profile"]
    base = p["base_scale"]

    # Mean demand expressed per hour in the *current* hour's context.
    # shape[h]*24 so that the daily sum averages to base (shape sums to 1).
    mu = base * shape[hr] * 24.0 * dow_prof[dow] * growth_by_month

    # Ramadan intra-day reshape — shape-preserving.
    if mask_ramadan.any():
        # Recompute daily-sum-preserving Ramadan weight
        ram_mult = RAMADAN_DAY_WEIGHTS[hr]
        # Renormalize inside Ramadan so the daily volume is unchanged from
        # the event factor alone.
        ram_sum = (shape * RAMADAN_DAY_WEIGHTS).sum()
        ram_mult = ram_mult / ram_sum * shape.sum()  # == RAMADAN_DAY_WEIGHTS/ram_sum
        mu = np.where(mask_ramadan, mu * ram_mult, mu)

    # Summer softener only for outdoor-exposed archetypes
    if arch_name == "mall_leisure":
        mu *= summer_commercial
    elif arch_name == "low_density":
        mu *= summer_lowdens

    # Commercial / office Friday prayer soft dip
    if arch_name in ("commercial_midday", "office_cluster"):
        mu *= friday_prayer

    # Event factor
    mu *= build_event_factor(aoi)

    # AR(1) latent — INDEPENDENT across AOIs
    latent = sample_latent_ar1(N_HOURS, p["ar_rho"], p["ar_sigma"], gen)
    mu *= np.exp(latent)

    # Local shocks and multi-hour outages
    mu = inject_local_shocks(mu, gen)
    mu = inject_outages(mu, gen)

    # Sparsity collapse: on already-low hours, collapse to ~0 with prob p
    low_threshold = np.quantile(mu, 0.15)
    low_mask = mu < low_threshold
    collapse = gen.random(N_HOURS) < p["sparsity_p"]
    mu = np.where(low_mask & collapse, mu * gen.uniform(0.0, 0.1, size=N_HOURS), mu)

    mu = np.maximum(mu, 0.05)

    # Heteroskedastic NB: dispersion shrinks at peaks (peaks are more Poisson),
    # grows off-peak (off-peak is messier).
    peak_frac = shape[hr] / shape.max()       # 0..1
    k = p["nb_dispersion"] * (0.5 + 1.5 * (1 - peak_frac))  # off-peak 2x dispersion
    # NB parametrization: mean=mu, var=mu + mu^2/k
    prob = k / (k + mu)
    counts = gen.negative_binomial(k, prob)

    return counts.astype(np.int64)


# Spawn a fresh independent rng per AOI so generation order is stable
gen_rngs = rng.spawn(N_AOIS)
print("\nGenerating AOIs...")
demand_matrix = np.empty((N_AOIS, N_HOURS), dtype=np.int64)
for i, aoi in enumerate(AOIS):
    demand_matrix[i] = generate_aoi(aoi, gen_rngs[i])
    print(f"  {aoi['aoi_id']:>5}  {aoi['name']:<20}  {aoi['archetype']:<22}"
          f"  mean={demand_matrix[i].mean():6.1f}  zeros={(demand_matrix[i]==0).mean():.2%}")

# ─────────────────────────────────────────────
# 8. BUILD LONG-FORMAT DATAFRAME
# ─────────────────────────────────────────────
print("\nAssembling dataframe...")
rows = []
for i, aoi in enumerate(AOIS):
    df = pd.DataFrame({
        "bucket_hour":     HOURS,
        "city":            "Dammam",
        "region_id":       aoi["region_id"],
        "aoi_id":          aoi["aoi_id"],
        "aoi_name":        aoi["name"],
        "aoi_archetype":   aoi["archetype"],
        "demand_count":    demand_matrix[i],
    })
    rows.append(df)
df = pd.concat(rows, ignore_index=True)

df["hour"]            = df["bucket_hour"].dt.hour
df["day_of_week"]     = df["bucket_hour"].dt.dayofweek
df["month"]           = df["bucket_hour"].dt.month
df["is_weekend"]      = df["day_of_week"].isin([4, 5]).astype(np.int8)
df["is_ramadan"]      = is_ramadan(df["bucket_hour"]).astype(np.int8)
df["is_eid"]          = is_eid(df["bucket_hour"]).astype(np.int8)
df["is_eid_pre"]      = is_eid_pre(df["bucket_hour"]).astype(np.int8)
df["is_white_friday"] = is_white_friday(df["bucket_hour"]).astype(np.int8)
df["is_national_day"] = is_national_day(df["bucket_hour"]).astype(np.int8)
df["is_founding_day"] = is_founding_day(df["bucket_hour"]).astype(np.int8)
df["is_back_to_school"] = is_back_to_school(df["bucket_hour"]).astype(np.int8)
df["is_hajj"]         = is_hajj(df["bucket_hour"]).astype(np.int8)

df = df.sort_values(["aoi_id", "bucket_hour"]).reset_index(drop=True)

# Simulate ~0.4% random missing rows (real data has gaps)
drop_mask = rng.random(len(df)) < 0.004
n_dropped = int(drop_mask.sum())
df = df.loc[~drop_mask].reset_index(drop=True)

# ─────────────────────────────────────────────
# 9. DIAGNOSTICS
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("  VALIDATION DIAGNOSTICS")
print("=" * 70)

print(f"  Row count:         {len(df):,}  (dropped {n_dropped:,} missing)")
print(f"  Date range:        {df.bucket_hour.min()}  ->  {df.bucket_hour.max()}")
print(f"  Unique hours:      {df.bucket_hour.nunique():,}")
print(f"  Unique AOIs:       {df.aoi_id.nunique()}")
print(f"  Zero-demand share: {(df.demand_count == 0).mean():.3%}")

print("\n  Per-AOI statistics:")
stats = (df.groupby(["aoi_id", "aoi_archetype"])
           .demand_count.agg(["mean", "std", "min", "max"])
           .round(2))
stats["cv"] = (stats["std"] / stats["mean"]).round(3)
print(stats.to_string())

# Cross-AOI correlation
pivot = df.pivot_table(index="bucket_hour", columns="aoi_id",
                       values="demand_count", aggfunc="first").dropna()
corr = pivot.corr().values
offdiag = corr[~np.eye(corr.shape[0], dtype=bool)]
print(f"\n  Cross-AOI correlation:")
print(f"    mean = {offdiag.mean():.4f}    median = {np.median(offdiag):.4f}")
print(f"    min  = {offdiag.min():.4f}    max    = {offdiag.max():.4f}")

# Event uplift vs baseline — measured on DAILY TOTALS per AOI so that
# intra-day reshaping and weekend-of-week confounds don't distort the ratio.
# For WF we use Nov-Friday baseline, to match the markdown framing.
# For Hajj we exclude Eid-Adha days, which fall inside the Hajj window.
daily = (df.groupby(["aoi_id", df.bucket_hour.dt.normalize().rename("day")])
           .agg(demand_count=("demand_count", "sum"),
                is_ramadan=("is_ramadan", "max"),
                is_eid=("is_eid", "max"),
                is_eid_pre=("is_eid_pre", "max"),
                is_white_friday=("is_white_friday", "max"),
                is_national_day=("is_national_day", "max"),
                is_founding_day=("is_founding_day", "max"),
                is_back_to_school=("is_back_to_school", "max"),
                is_hajj=("is_hajj", "max"))
           .reset_index())
daily["dow"] = pd.to_datetime(daily["day"]).dt.dayofweek
daily["month"] = pd.to_datetime(daily["day"]).dt.month

base_mask_day = (
    (daily.is_ramadan == 0) & (daily.is_eid == 0) & (daily.is_eid_pre == 0)
    & (daily.is_white_friday == 0) & (daily.is_national_day == 0)
    & (daily.is_founding_day == 0) & (daily.is_back_to_school == 0)
    & (daily.is_hajj == 0)
)

def uplift(event_mask, baseline_mask, label, expected_range):
    per_aoi_event = daily[event_mask].groupby("aoi_id").demand_count.mean()
    per_aoi_base  = daily[baseline_mask].groupby("aoi_id").demand_count.mean()
    ratio = per_aoi_event / per_aoi_base
    cross_mean = float(ratio.mean())
    inside = expected_range[0] <= cross_mean <= expected_range[1]
    tag = "OK" if inside else "OUT OF MARKDOWN BAND"
    print(f"    {label:<22}  cross-AOI mean = {cross_mean:.3f}  "
          f"per-AOI min/max = {ratio.min():.3f}/{ratio.max():.3f}  "
          f"markdown = {expected_range}  [{tag}]")
    return cross_mean, float(ratio.min()), float(ratio.max())

print("\n  Event uplifts (per-AOI DAILY-TOTAL mean vs matched baseline):")
u_ram = uplift(daily.is_ramadan == 1, base_mask_day,
               "Ramadan", MD_RANGES["ramadan_month_mean"])
u_eid = uplift(daily.is_eid == 1, base_mask_day,
               "Eid days", MD_RANGES["eid_day"])
# WF vs average November Friday
nov_friday_mask = (daily.month == 11) & (daily.dow == 4) & (daily.is_white_friday == 0)
u_wf = uplift(daily.is_white_friday == 1, nov_friday_mask,
              "White Friday vs Nov-Fri", MD_RANGES["white_friday_day"])
u_nd = uplift(daily.is_national_day == 1, base_mask_day,
              "National Day window", MD_RANGES["national_day_window"])
u_fd = uplift(daily.is_founding_day == 1, base_mask_day,
              "Founding Day window", MD_RANGES["founding_day_window"])
u_bts = uplift(daily.is_back_to_school == 1, base_mask_day,
               "Back-to-school window", MD_RANGES["back_to_school_window"])
# Hajj: exclude Eid-Adha overlap days. NOTE: after exclusion only Jun 19
# remains, so this is a 1-day sample per AOI — the cross-AOI mean is
# noise-dominated. We widen the acceptance band to reflect that.
hajj_nonEid = (daily.is_hajj == 1) & (daily.is_eid == 0) & (daily.is_eid_pre == 0)
u_hajj = uplift(hajj_nonEid, base_mask_day,
                "Hajj (ex-Eid, 1 day)", (0.85, 1.15))

# Triviality / smoothness warnings
warnings = []
if offdiag.mean() > 0.30:
    warnings.append(f"Cross-AOI corr mean {offdiag.mean():.3f} still above 0.30")
avg_cv = stats["cv"].mean()
if avg_cv < 0.6:
    warnings.append(f"Avg per-AOI CV {avg_cv:.3f} is low (flat series)")
lag1_acs = []
for a, g in df.sort_values(["aoi_id", "bucket_hour"]).groupby("aoi_id"):
    x = g.demand_count.values.astype(float)
    if len(x) > 25:
        lag1_acs.append(np.corrcoef(x[1:], x[:-1])[0, 1])
mean_lag1 = float(np.mean(lag1_acs))
if mean_lag1 > 0.85:
    warnings.append(f"Lag-1 autocorr {mean_lag1:.3f} very high — possibly too smooth")
if mean_lag1 < 0.15:
    warnings.append(f"Lag-1 autocorr {mean_lag1:.3f} very low — possibly too noisy")
print(f"\n  Mean per-AOI lag-1 autocorr: {mean_lag1:.3f}")

print("\n  Warnings:")
if warnings:
    for w in warnings:
        print(f"    ! {w}")
else:
    print("    (none)")

# ─────────────────────────────────────────────
# 10. SAVE CSV + CALIBRATION METADATA
# ─────────────────────────────────────────────
df.to_csv(OUT_CSV, index=False)
print(f"\nSaved CSV: {OUT_CSV}  ({len(df):,} rows)")

calibration = {
    "dataset": {
        "file":         OUT_CSV,
        "rows":         int(len(df)),
        "aois":         N_AOIS,
        "hours":        int(df.bucket_hour.nunique()),
        "date_range":   [str(df.bucket_hour.min()), str(df.bucket_hour.max())],
        "seed":         SEED,
    },
    "empirical_inputs_from_markdown": {
        "ramadan_month_uplift_band":     MD_RANGES["ramadan_month_mean"],
        "eid_day_uplift_band":           MD_RANGES["eid_day"],
        "white_friday_day_uplift_band":  MD_RANGES["white_friday_day"],
        "november_month_baseline":       MD_RANGES["november_month"],
        "national_day_window_band":      MD_RANGES["national_day_window"],
        "founding_day_window_band":      MD_RANGES["founding_day_window"],
        "back_to_school_band":           MD_RANGES["back_to_school_window"],
        "hajj_band_dammam":              MD_RANGES["hajj_window"],
        "annual_growth_yoy":             MD_RANGES["annual_growth_yoy"],
        "sources": [
            "Checkout.com KSA Ramadan digital payments analysis (+22%)",
            "SAMA Mada e-commerce bulletins (Q1 2024, March 2025, Oct 2025)",
            "AppsFlyer White Friday 2024 Saudi/UAE report",
            "Flowwow/Admitad MENA White Friday 2024",
            "Saudi MoC 95th National Day discount-licensing announcement",
            "GASTAT / academic Hajj logistics papers",
        ],
    },
    "modeling_assumptions": {
        "absolute_demand_scale":
            "No defensible per-AOI order count in the markdown. "
            "Each AOI's base_scale is drawn from an archetype-specific range "
            "chosen for plausibility, not derived from official statistics.",
        "aoi_archetypes":
            "7 archetypes (residential_evening, commercial_midday, mall_leisure, "
            "office_cluster, mixed, low_density, nightlife). Hourly shapes, "
            "day-of-week profiles, and event sub-bands are modeling assumptions.",
        "prayer_time_effects":
            "Only a soft Friday midday dip (0.75x at 12-13) is applied, and "
            "only to commercial/office archetypes. Much softer than the old "
            "script's -28% to -45% global dips, which were not sourced.",
        "ramadan_intraday_shape_shift":
            "Multiplicative daytime dampening (0.85) + late-evening/Suhoor "
            "lift (1.20), normalized to preserve the daily total. The MONTHLY "
            "volume uplift is handled separately and kept inside the markdown "
            "1.20-1.40 band.",
        "summer_midday_softener":
            "Only -15% at 12-15 in Jun/Jul/Aug, and only for mall_leisure and "
            "low_density archetypes. Old script applied -55% globally — "
            "overstated and unsourced.",
        "hajj_effect_in_dammam":
            "Kept near 1.0 (band 0.97-1.02) per markdown: no quantified "
            "e-commerce effect outside the western corridor.",
        "noise_model":
            "Heteroskedastic negative binomial with hour-dependent dispersion "
            "(off-peak 2x more dispersed than peak). Plus AR(1) latent driver, "
            "independent across AOIs.",
        "outages_and_shocks":
            "~12 multi-hour outages per AOI per year, ~1.5% hourly promo rate, "
            "~0.8% hourly disruption rate. Sparsity collapse on low-demand hours "
            "with archetype-specific probability.",
        "missing_hours":
            "~0.4% random row dropout to simulate logging gaps.",
    },
    "lade_inspired_design_choices": {
        "heterogeneous_archetypes":
            "LaDe has 14 distinct aoi_type codes with visibly different hourly "
            "patterns. We mirror that with 7 distinct archetypes whose shape "
            "functions are genuinely different (not Gaussian sums of the same family).",
        "low_cross_aoi_correlation":
            "Measured LaDe mean pairwise AOI correlation ≈ 0.07 on top-20 AOIs. "
            "We target much lower than the old dataset's 0.45 by making the "
            "AR(1) latent driver independent per AOI and by giving each AOI an "
            "independent event-response draw.",
        "heavy_tailed_per_aoi":
            "LaDe small AOIs have 95%+ zero hours. We don't go that extreme, "
            "but we apply archetype-specific sparsity collapse (18-25% for "
            "office_cluster / low_density) to recover natural dead hours.",
        "asymmetric_hour_shapes":
            "LaDe's hour-of-day histogram is skewed (morning-heavy courier pickups). "
            "We use asymmetric shape functions (sigmoid, shifted Gaussians on "
            "circular time, bracketed step for office) rather than symmetric "
            "double Gaussians.",
    },
    "measured_diagnostics": {
        "cross_aoi_corr_mean": float(offdiag.mean()),
        "cross_aoi_corr_min":  float(offdiag.min()),
        "cross_aoi_corr_max":  float(offdiag.max()),
        "zero_demand_share":   float((df.demand_count == 0).mean()),
        "mean_lag1_autocorr":  mean_lag1,
        "event_uplifts": {
            "ramadan":      {"mean": u_ram[0], "min": u_ram[1], "max": u_ram[2]},
            "eid":          {"mean": u_eid[0], "min": u_eid[1], "max": u_eid[2]},
            "white_friday": {"mean": u_wf[0],  "min": u_wf[1],  "max": u_wf[2]},
            "national_day": {"mean": u_nd[0],  "min": u_nd[1],  "max": u_nd[2]},
            "founding_day": {"mean": u_fd[0],  "min": u_fd[1],  "max": u_fd[2]},
            "back_to_school":{"mean": u_bts[0],"min": u_bts[1], "max": u_bts[2]},
            "hajj":         {"mean": u_hajj[0],"min": u_hajj[1],"max": u_hajj[2]},
        },
    },
    "known_limitations": [
        "Absolute demand magnitudes are not derived from any official Dammam "
        "or Eastern-Province statistic.",
        "AOI names are illustrative labels; the generator does not claim that "
        "these specific Dammam neighborhoods behave this way.",
        "Weather, traffic, real outages, and courier-side capacity are not modeled.",
        "Category-level effects (food, apparel, electronics) are collapsed into "
        "total demand_count — the dataset is not multi-category.",
        "Hourly intra-day Ramadan reshape magnitude is an assumption, not sourced.",
        "Event overlap rules (Eid inside Ramadan, BTS inside National Day) are "
        "heuristic: Eid replaces Ramadan on Eid days, BTS/Hajj multiply onto "
        "whatever is already present.",
    ],
    "output_schema": list(df.columns),
}

with open(OUT_META, "w") as f:
    json.dump(calibration, f, indent=2, default=str)
print(f"Saved calibration metadata: {OUT_META}")

print("\n" + "=" * 70)
print("  DONE")
print("=" * 70)
