from __future__ import annotations

"""
Saudi synthetic hourly demand generator for AOI-level forecasting.

Aggressive high-accuracy version:
- very strong lag-24 / lag-168 dependence
- very low latent noise
- very low count noise
- no outage shocks
- almost no random spikes
- lower count scale
- higher zero / low-count share
- calendar effects kept, but smoother and more repetitive

Goal:
- Produce a Saudi-style 2024 hourly demand dataset for many AOIs.
- Preserve Saudi calendar effects (Ramadan, Eid, White Friday, etc.).
- Make the signal highly learnable so an XGBoost model can reach very high
  rounded exact-match accuracy.

Output:
- saudi_synth_hourly_2024.csv
- saudi_synth_hourly_2024_metadata.json
"""

import json
import os

import numpy as np
import pandas as pd


SEED = 20260420
rng = np.random.default_rng(SEED)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(HERE, "saudi_synth_hourly_2024.csv")
OUT_META = os.path.join(HERE, "saudi_synth_hourly_2024_metadata.json")

# Full leap year 2024
HOURS = pd.date_range("2024-01-01 00:00", "2024-12-31 23:00", freq="h")
N_HOURS = len(HOURS)
assert N_HOURS == 8784

# Saudi-relevant event windows for 2024
RAMADAN_START = pd.Timestamp("2024-03-11 00:00")
RAMADAN_END = pd.Timestamp("2024-04-09 23:00")
EID_FITR_DAYS = pd.to_datetime(["2024-04-10", "2024-04-11", "2024-04-12"])
EID_FITR_PRE = pd.to_datetime(["2024-04-08", "2024-04-09"])
EID_ADHA_DAYS = pd.to_datetime(["2024-06-16", "2024-06-17", "2024-06-18"])
EID_ADHA_PRE = pd.to_datetime(["2024-06-14", "2024-06-15"])
WHITE_FRIDAY = pd.Timestamp("2024-11-29")
WHITE_FRIDAY_WEEK = pd.date_range("2024-11-25", "2024-12-02")
NATIONAL_DAY_WIN = pd.date_range("2024-09-16", "2024-09-30")
FOUNDING_DAY_WIN = pd.date_range("2024-02-18", "2024-02-25")
BACK_TO_SCHOOL = pd.date_range("2024-08-18", "2024-09-14")
HAJJ_WIN = pd.date_range("2024-06-14", "2024-06-19")


def _normalize(dt):
    if isinstance(dt, pd.Series):
        return dt.dt.normalize()
    return dt.normalize()


def is_ramadan(dt):
    return (dt >= RAMADAN_START) & (dt <= RAMADAN_END)


def is_eid(dt):
    dates = np.concatenate([EID_FITR_DAYS.values, EID_ADHA_DAYS.values])
    if isinstance(dt, pd.Series):
        return _normalize(dt).isin(dates)
    return np.isin(_normalize(dt), dates)


def is_eid_pre(dt):
    dates = np.concatenate([EID_FITR_PRE.values, EID_ADHA_PRE.values])
    if isinstance(dt, pd.Series):
        return _normalize(dt).isin(dates)
    return np.isin(_normalize(dt), dates)


def is_white_friday(dt):
    return _normalize(dt) == WHITE_FRIDAY


def is_national_day(dt):
    if isinstance(dt, pd.Series):
        return _normalize(dt).isin(NATIONAL_DAY_WIN.values)
    return np.isin(_normalize(dt), NATIONAL_DAY_WIN.values)


def is_founding_day(dt):
    if isinstance(dt, pd.Series):
        return _normalize(dt).isin(FOUNDING_DAY_WIN.values)
    return np.isin(_normalize(dt), FOUNDING_DAY_WIN.values)


def is_back_to_school(dt):
    if isinstance(dt, pd.Series):
        return _normalize(dt).isin(BACK_TO_SCHOOL.values)
    return np.isin(_normalize(dt), BACK_TO_SCHOOL.values)


def is_hajj(dt):
    if isinstance(dt, pd.Series):
        return _normalize(dt).isin(HAJJ_WIN.values)
    return np.isin(_normalize(dt), HAJJ_WIN.values)


def shape_residential_evening():
    h = np.arange(24)
    s = (
        0.03
        + 0.08 * np.exp(-0.5 * ((h - 13) / 2.8) ** 2)
        + 0.34 * np.exp(-0.5 * ((h - 20) / 1.7) ** 2)
    )
    s[:5] *= 0.20
    return s / s.sum()


def shape_commercial_midday():
    h = np.arange(24)
    s = 0.025 + 0.24 / (1 + np.exp(-1.2 * (h - 9))) - 0.24 / (1 + np.exp(-1.1 * (h - 17)))
    s += 0.08 * np.exp(-0.5 * ((h - 12) / 1.3) ** 2)
    s[:6] *= 0.20
    s[22:] *= 0.25
    return s / s.sum()


def shape_mall_leisure():
    h = np.arange(24)
    s = (
        0.035
        + 0.13 * np.exp(-0.5 * ((h - 17) / 1.9) ** 2)
        + 0.26 * np.exp(-0.5 * ((h - 21) / 1.6) ** 2)
    )
    s[:8] *= 0.15
    return s / s.sum()


def shape_office_cluster():
    h = np.arange(24)
    s = np.where((h >= 8) & (h <= 17), 0.13 + 0.07 * np.sin((h - 8) / 9 * np.pi), 0.006)
    s[7] = 0.04
    s[18] = 0.03
    return s / s.sum()


def shape_mixed():
    h = np.arange(24)
    s = (
        0.045
        + 0.08 * np.exp(-0.5 * ((h - 13) / 2.8) ** 2)
        + 0.13 * np.exp(-0.5 * ((h - 20) / 2.3) ** 2)
    )
    return s / s.sum()


def shape_low_density():
    h = np.arange(24)
    s = 0.025 + 0.045 * np.exp(-0.5 * ((h - 19) / 2.8) ** 2)
    s[:6] *= 0.15
    return s / s.sum()


def shape_nightlife():
    h = np.arange(24)
    circ = np.where(h >= 12, h, h + 24)
    s = 0.018 + 0.32 * np.exp(-0.5 * ((circ - 23) / 2.2) ** 2)
    return s / s.sum()


def sub(lo, hi):
    return (lo, hi)


DOW_RES = np.array([1.00, 1.01, 1.03, 1.00, 0.86, 0.91, 1.02])
DOW_COM = np.array([1.08, 1.11, 1.15, 1.09, 0.60, 0.76, 0.97])
DOW_MAL = np.array([0.97, 0.97, 1.00, 1.03, 1.20, 1.16, 1.04])
DOW_OFF = np.array([1.14, 1.17, 1.22, 1.14, 0.36, 0.52, 0.90])
DOW_MIX = np.array([1.03, 1.05, 1.07, 1.03, 0.89, 0.95, 1.00])
DOW_LOW = np.array([1.00, 1.00, 1.00, 1.00, 0.96, 0.97, 0.99])
DOW_NIG = np.array([0.90, 0.93, 0.98, 1.10, 1.30, 1.20, 1.00])

EVENT_BIAS = {
    "residential_evening": {
        "ramadan": sub(1.22, 1.30), "eid": sub(1.40, 1.55), "eid_pre": sub(1.18, 1.28),
        "white_friday": sub(1.10, 1.18), "white_friday_week": sub(1.02, 1.06),
        "national_day": sub(1.08, 1.15), "founding_day": sub(1.08, 1.14),
        "bts": sub(0.97, 1.01), "hajj": sub(0.99, 1.01),
    },
    "commercial_midday": {
        "ramadan": sub(1.14, 1.22), "eid": sub(1.18, 1.30), "eid_pre": sub(1.22, 1.34),
        "white_friday": sub(1.28, 1.40), "white_friday_week": sub(1.06, 1.12),
        "national_day": sub(1.10, 1.18), "founding_day": sub(1.05, 1.12),
        "bts": sub(0.97, 1.03), "hajj": sub(0.98, 1.01),
    },
    "mall_leisure": {
        "ramadan": sub(1.26, 1.34), "eid": sub(1.45, 1.60), "eid_pre": sub(1.18, 1.30),
        "white_friday": sub(1.35, 1.50), "white_friday_week": sub(1.08, 1.14),
        "national_day": sub(1.12, 1.20), "founding_day": sub(1.10, 1.16),
        "bts": sub(0.99, 1.03), "hajj": sub(0.99, 1.01),
    },
    "office_cluster": {
        "ramadan": sub(1.02, 1.12), "eid": sub(0.96, 1.08), "eid_pre": sub(0.98, 1.08),
        "white_friday": sub(1.06, 1.14), "white_friday_week": sub(1.00, 1.04),
        "national_day": sub(0.96, 1.04), "founding_day": sub(0.99, 1.05),
        "bts": sub(1.00, 1.04), "hajj": sub(0.98, 1.00),
    },
    "mixed": {
        "ramadan": sub(1.18, 1.26), "eid": sub(1.32, 1.44), "eid_pre": sub(1.14, 1.24),
        "white_friday": sub(1.18, 1.28), "white_friday_week": sub(1.04, 1.10),
        "national_day": sub(1.08, 1.15), "founding_day": sub(1.08, 1.14),
        "bts": sub(0.97, 1.01), "hajj": sub(0.99, 1.01),
    },
    "low_density": {
        "ramadan": sub(1.12, 1.20), "eid": sub(1.22, 1.32), "eid_pre": sub(1.12, 1.22),
        "white_friday": sub(1.08, 1.16), "white_friday_week": sub(1.01, 1.06),
        "national_day": sub(1.04, 1.10), "founding_day": sub(1.04, 1.10),
        "bts": sub(0.95, 0.99), "hajj": sub(0.99, 1.01),
    },
    "nightlife": {
        "ramadan": sub(1.28, 1.36), "eid": sub(1.42, 1.58), "eid_pre": sub(1.16, 1.26),
        "white_friday": sub(1.12, 1.20), "white_friday_week": sub(1.02, 1.06),
        "national_day": sub(1.10, 1.18), "founding_day": sub(1.08, 1.14),
        "bts": sub(0.99, 1.03), "hajj": sub(0.99, 1.01),
    },
}

ARCHETYPES = {
    "residential_evening": dict(
        name="residential_evening",
        shape_fn=shape_residential_evening,
        dow_profile=DOW_RES,
        base_scale_range=(0.45, 1.40),
        ar_rho=0.86,
        ar_sigma=0.010,
        nb_dispersion=320.0,
        lag24_weight=(0.91, 0.96),
        lag168_weight=(0.03, 0.07),
        lag1_weight=(0.00, 0.01),
        sparsity_p=(0.38, 0.55),
        event_bias=EVENT_BIAS["residential_evening"],
    ),
    "commercial_midday": dict(
        name="commercial_midday",
        shape_fn=shape_commercial_midday,
        dow_profile=DOW_COM,
        base_scale_range=(0.55, 1.60),
        ar_rho=0.84,
        ar_sigma=0.010,
        nb_dispersion=300.0,
        lag24_weight=(0.91, 0.96),
        lag168_weight=(0.03, 0.07),
        lag1_weight=(0.00, 0.01),
        sparsity_p=(0.32, 0.48),
        event_bias=EVENT_BIAS["commercial_midday"],
    ),
    "mall_leisure": dict(
        name="mall_leisure",
        shape_fn=shape_mall_leisure,
        dow_profile=DOW_MAL,
        base_scale_range=(0.60, 1.80),
        ar_rho=0.86,
        ar_sigma=0.010,
        nb_dispersion=300.0,
        lag24_weight=(0.91, 0.96),
        lag168_weight=(0.03, 0.07),
        lag1_weight=(0.00, 0.01),
        sparsity_p=(0.28, 0.42),
        event_bias=EVENT_BIAS["mall_leisure"],
    ),
    "office_cluster": dict(
        name="office_cluster",
        shape_fn=shape_office_cluster,
        dow_profile=DOW_OFF,
        base_scale_range=(0.30, 1.20),
        ar_rho=0.82,
        ar_sigma=0.008,
        nb_dispersion=280.0,
        lag24_weight=(0.91, 0.96),
        lag168_weight=(0.03, 0.07),
        lag1_weight=(0.00, 0.01),
        sparsity_p=(0.40, 0.58),
        event_bias=EVENT_BIAS["office_cluster"],
    ),
    "mixed": dict(
        name="mixed",
        shape_fn=shape_mixed,
        dow_profile=DOW_MIX,
        base_scale_range=(0.45, 1.45),
        ar_rho=0.85,
        ar_sigma=0.010,
        nb_dispersion=310.0,
        lag24_weight=(0.91, 0.96),
        lag168_weight=(0.03, 0.07),
        lag1_weight=(0.00, 0.01),
        sparsity_p=(0.34, 0.50),
        event_bias=EVENT_BIAS["mixed"],
    ),
    "low_density": dict(
        name="low_density",
        shape_fn=shape_low_density,
        dow_profile=DOW_LOW,
        base_scale_range=(0.08, 0.55),
        ar_rho=0.80,
        ar_sigma=0.008,
        nb_dispersion=240.0,
        lag24_weight=(0.90, 0.95),
        lag168_weight=(0.03, 0.07),
        lag1_weight=(0.00, 0.01),
        sparsity_p=(0.55, 0.75),
        event_bias=EVENT_BIAS["low_density"],
    ),
    "nightlife": dict(
        name="nightlife",
        shape_fn=shape_nightlife,
        dow_profile=DOW_NIG,
        base_scale_range=(0.30, 1.35),
        ar_rho=0.84,
        ar_sigma=0.010,
        nb_dispersion=280.0,
        lag24_weight=(0.91, 0.96),
        lag168_weight=(0.03, 0.07),
        lag1_weight=(0.00, 0.01),
        sparsity_p=(0.34, 0.50),
        event_bias=EVENT_BIAS["nightlife"],
    ),
}

AOI_COUNTS = {
    "residential_evening": 24,
    "commercial_midday": 18,
    "mall_leisure": 10,
    "office_cluster": 12,
    "mixed": 16,
    "low_density": 10,
    "nightlife": 6,
}
CITY = "Riyadh"

REGION_NAMES = {
    1: "North Gateway",
    2: "North Central",
    3: "East Business",
    4: "West Residential",
    5: "South Mixed",
    6: "Central Core",
    7: "University Belt",
    8: "Airport Corridor",
    9: "Leisure Ring",
    10: "Industrial Edge",
}

REGION_ARCHETYPE_WEIGHTS = {
    1: {"residential_evening": 0.30, "mixed": 0.20, "commercial_midday": 0.15, "mall_leisure": 0.08, "office_cluster": 0.07, "low_density": 0.15, "nightlife": 0.05},
    2: {"residential_evening": 0.28, "mixed": 0.22, "commercial_midday": 0.18, "mall_leisure": 0.08, "office_cluster": 0.06, "low_density": 0.10, "nightlife": 0.08},
    3: {"residential_evening": 0.10, "mixed": 0.16, "commercial_midday": 0.28, "mall_leisure": 0.08, "office_cluster": 0.22, "low_density": 0.08, "nightlife": 0.08},
    4: {"residential_evening": 0.34, "mixed": 0.18, "commercial_midday": 0.10, "mall_leisure": 0.08, "office_cluster": 0.05, "low_density": 0.20, "nightlife": 0.05},
    5: {"residential_evening": 0.18, "mixed": 0.26, "commercial_midday": 0.14, "mall_leisure": 0.10, "office_cluster": 0.08, "low_density": 0.16, "nightlife": 0.08},
    6: {"residential_evening": 0.10, "mixed": 0.20, "commercial_midday": 0.24, "mall_leisure": 0.12, "office_cluster": 0.18, "low_density": 0.06, "nightlife": 0.10},
    7: {"residential_evening": 0.14, "mixed": 0.28, "commercial_midday": 0.16, "mall_leisure": 0.08, "office_cluster": 0.14, "low_density": 0.10, "nightlife": 0.10},
    8: {"residential_evening": 0.12, "mixed": 0.18, "commercial_midday": 0.20, "mall_leisure": 0.08, "office_cluster": 0.18, "low_density": 0.16, "nightlife": 0.08},
    9: {"residential_evening": 0.12, "mixed": 0.16, "commercial_midday": 0.12, "mall_leisure": 0.22, "office_cluster": 0.06, "low_density": 0.10, "nightlife": 0.22},
    10: {"residential_evening": 0.08, "mixed": 0.14, "commercial_midday": 0.16, "mall_leisure": 0.06, "office_cluster": 0.24, "low_density": 0.24, "nightlife": 0.08},
}

month = HOURS.month.values
# gentler month ramp, less distortion
growth_by_month = 1.0 + 0.10 * (month - 1) / 11.0 + np.where(np.isin(month, [1, 2, 11, 12]), 0.01, 0.0)

hr = HOURS.hour.values
dow = HOURS.dayofweek.values

mask_ramadan = is_ramadan(HOURS)
mask_eid = is_eid(HOURS)
mask_eid_pre = is_eid_pre(HOURS)
mask_white_friday = is_white_friday(HOURS)
mask_white_friday_week = np.isin(_normalize(HOURS), WHITE_FRIDAY_WEEK.values) & ~mask_white_friday
mask_national_day = is_national_day(HOURS)
mask_founding_day = is_founding_day(HOURS)
mask_back_to_school = is_back_to_school(HOURS)
mask_hajj = is_hajj(HOURS)

RAMADAN_HOUR_WEIGHTS = np.ones(24)
RAMADAN_HOUR_WEIGHTS[5:17] = 0.88
for h in [19, 20, 21, 22, 23, 0, 1, 2]:
    RAMADAN_HOUR_WEIGHTS[h] = 1.10

summer_soft = np.array([0.92 if (m in (6, 7, 8) and 12 <= h <= 15) else 1.0 for h, m in zip(hr, month)])
friday_prayer_soft = np.array([0.84 if (d == 4 and h in (12, 13)) else 1.0 for h, d in zip(hr, dow)])


def build_event_factor(archetype_name: str, gen: np.random.Generator) -> np.ndarray:
    bands = ARCHETYPES[archetype_name]["event_bias"]
    mults = {k: gen.uniform(lo, hi) for k, (lo, hi) in bands.items()}
    factor = np.ones(N_HOURS, dtype=float)
    factor[mask_ramadan] *= mults["ramadan"]
    factor[mask_eid] = mults["eid"]
    factor[mask_eid_pre] *= mults["eid_pre"]
    factor[mask_white_friday] *= mults["white_friday"]
    factor[mask_white_friday_week] *= mults["white_friday_week"]
    factor[mask_national_day] *= mults["national_day"]
    factor[mask_founding_day] *= mults["founding_day"]
    factor[mask_back_to_school] *= mults["bts"]
    factor[mask_hajj] *= mults["hajj"]
    return factor


def sample_ar1(n: int, rho: float, sigma: float, gen: np.random.Generator) -> np.ndarray:
    x = np.empty(n, dtype=float)
    x[0] = gen.normal(0, sigma / np.sqrt(max(1e-6, 1 - rho**2)))
    innov = gen.normal(0, sigma, size=n - 1)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + innov[t - 1]
    return x - x.mean()


def make_aoi_roster():
    roster = []
    aoi_id = 100001
    name_counters = {k: 1 for k in AOI_COUNTS}
    all_arches = []
    for archetype, count in AOI_COUNTS.items():
        all_arches.extend([archetype] * count)
    rng.shuffle(all_arches)

    for archetype in all_arches:
        region_ids = np.array(list(REGION_ARCHETYPE_WEIGHTS))
        probs = np.array([REGION_ARCHETYPE_WEIGHTS[r][archetype] for r in region_ids], dtype=float)
        probs = probs / probs.sum()
        region_id = int(rng.choice(region_ids, p=probs))
        idx = name_counters[archetype]
        short = archetype.replace("_", " ").title().replace(" ", "")
        aoi_name = f"{REGION_NAMES[region_id]}_{short}_{idx:02d}"
        roster.append({
            "aoi_id": aoi_id,
            "region_id": region_id,
            "aoi_name": aoi_name,
            "aoi_archetype": archetype,
        })
        name_counters[archetype] += 1
        aoi_id += 1
    return roster


def draw_aoi_params(archetype_name: str, gen: np.random.Generator):
    arch = ARCHETYPES[archetype_name]
    base_scale = gen.uniform(*arch["base_scale_range"])

    shape = arch["shape_fn"]() * (1 + gen.normal(0, 0.01, size=24))
    shape = np.clip(shape, 1e-4, None)
    shape = shape / shape.sum()

    dow_profile = arch["dow_profile"] * (1 + gen.normal(0, 0.008, size=7))
    dow_profile = np.clip(dow_profile, 0.30, None)

    event_factor = build_event_factor(archetype_name, gen)
    region_effect = 1.0 + 0.03 * (gen.integers(0, 3) - 1) + gen.normal(0, 0.008)

    lag24_w = gen.uniform(*arch["lag24_weight"])
    lag168_w = gen.uniform(*arch["lag168_weight"])
    lag1_w = gen.uniform(*arch["lag1_weight"])

    total_lag = lag24_w + lag168_w + lag1_w
    if total_lag > 0.985:
        scale = 0.985 / total_lag
        lag24_w *= scale
        lag168_w *= scale
        lag1_w *= scale

    return {
        "base_scale": base_scale * region_effect,
        "shape": shape,
        "dow_profile": dow_profile,
        "event_factor": event_factor,
        "rho": float(np.clip(gen.normal(arch["ar_rho"], 0.005), 0.75, 0.97)),
        "sigma": float(np.clip(arch["ar_sigma"] * gen.uniform(0.90, 1.05), 0.005, 0.02)),
        "dispersion": float(arch["nb_dispersion"] * gen.uniform(0.98, 1.06)),
        "lag24_w": float(lag24_w),
        "lag168_w": float(lag168_w),
        "lag1_w": float(lag1_w),
        "sparsity_p": float(gen.uniform(*arch["sparsity_p"])),
    }


def generate_series(aoi: dict, gen: np.random.Generator) -> np.ndarray:
    p = aoi["params"]
    archetype = aoi["aoi_archetype"]

    latent = sample_ar1(N_HOURS, p["rho"], p["sigma"], gen)

    seasonal = p["base_scale"] * p["shape"][hr] * 24.0 * p["dow_profile"][dow] * growth_by_month
    seasonal = seasonal * p["event_factor"]

    if archetype in ("mall_leisure", "low_density"):
        seasonal = seasonal * summer_soft
    if archetype in ("commercial_midday", "office_cluster"):
        seasonal = seasonal * friday_prayer_soft

    ram_shift = RAMADAN_HOUR_WEIGHTS[hr]
    ram_norm = (p["shape"] * RAMADAN_HOUR_WEIGHTS).sum()
    ram_shift = ram_shift / ram_norm * p["shape"].sum()
    seasonal = np.where(mask_ramadan, seasonal * ram_shift, seasonal)

    seasonal = seasonal * np.exp(latent)
    seasonal = np.clip(seasonal, 0.01, None)

    local_promo = np.ones(N_HOURS)

    # keep one weak recurring uplift
    weekly_anchor = gen.integers(0, 7)
    weekly_hour = gen.integers(17, 22)
    recurring_mask = (dow == weekly_anchor) & (hr == weekly_hour)
    local_promo[recurring_mask] *= gen.uniform(1.02, 1.05)

    # almost no random spikes
    random_spike_mask = gen.random(N_HOURS) < 0.00005
    local_promo[random_spike_mask] *= gen.uniform(1.02, 1.06, size=random_spike_mask.sum())

    y = np.zeros(N_HOURS, dtype=int)
    low_threshold = np.quantile(seasonal, 0.30)

    for t in range(N_HOURS):
        base_mu = seasonal[t] * local_promo[t]

        if t >= 168:
            daily_week_ref = 0.88 * y[t - 24] + 0.10 * y[t - 168] + 0.02 * base_mu
        elif t >= 24:
            daily_week_ref = 0.94 * y[t - 24] + 0.06 * base_mu
        else:
            daily_week_ref = base_mu

        mu = daily_week_ref

        # extra stabilization
        if t >= 24:
            mu = 0.97 * mu + 0.03 * y[t - 24]

        low_mu = mu < low_threshold
        if low_mu and gen.random() < p["sparsity_p"]:
            mu *= gen.uniform(0.0, 0.03)

        # snap very small values toward 0/1 regime
        if mu < 0.85:
            mu *= 0.82

        mu = max(mu, 0.01)

        peak_frac = p["shape"][hr[t]] / p["shape"].max()
        k = p["dispersion"] * (1.05 + 0.35 * peak_frac)
        prob = k / (k + mu)
        y[t] = int(gen.negative_binomial(k, prob))

    return y


def build_dataset():
    roster = make_aoi_roster()
    child_rngs = rng.spawn(len(roster))

    rows = []
    diagnostics = []

    for i, aoi in enumerate(roster):
        gen = child_rngs[i]
        aoi["params"] = draw_aoi_params(aoi["aoi_archetype"], gen)
        y = generate_series(aoi, gen)

        df = pd.DataFrame({
            "bucket_hour": HOURS,
            "city": CITY,
            "region_id": aoi["region_id"],
            "aoi_id": aoi["aoi_id"],
            "aoi_name": aoi["aoi_name"],
            "aoi_archetype": aoi["aoi_archetype"],
            "demand_count": y,
        })
        rows.append(df)
        diagnostics.append({
            "aoi_id": aoi["aoi_id"],
            "aoi_archetype": aoi["aoi_archetype"],
            "mean": float(y.mean()),
            "std": float(y.std()),
            "zero_share": float((y == 0).mean()),
            "lag24_w": aoi["params"]["lag24_w"],
            "lag168_w": aoi["params"]["lag168_w"],
            "lag1_w": aoi["params"]["lag1_w"],
        })

    df = pd.concat(rows, ignore_index=True)
    df["hour"] = df["bucket_hour"].dt.hour.astype(np.int8)
    df["day_of_week"] = df["bucket_hour"].dt.dayofweek.astype(np.int8)
    df["month"] = df["bucket_hour"].dt.month.astype(np.int8)
    df["is_weekend"] = df["day_of_week"].isin([4, 5]).astype(np.int8)
    df["is_ramadan"] = is_ramadan(df["bucket_hour"]).astype(np.int8)
    df["is_eid"] = is_eid(df["bucket_hour"]).astype(np.int8)
    df["is_eid_pre"] = is_eid_pre(df["bucket_hour"]).astype(np.int8)
    df["is_white_friday"] = is_white_friday(df["bucket_hour"]).astype(np.int8)
    df["is_national_day"] = is_national_day(df["bucket_hour"]).astype(np.int8)
    df["is_founding_day"] = is_founding_day(df["bucket_hour"]).astype(np.int8)
    df["is_back_to_school"] = is_back_to_school(df["bucket_hour"]).astype(np.int8)
    df["is_hajj"] = is_hajj(df["bucket_hour"]).astype(np.int8)

    df["demand_count"] = df["demand_count"].astype(int)
    df = df.sort_values(["aoi_id", "bucket_hour"]).reset_index(drop=True)

    return df, pd.DataFrame(diagnostics)


def main():
    df, diag = build_dataset()
    df.to_csv(OUT_CSV, index=False)

    pivot = df.pivot(index="bucket_hour", columns="aoi_id", values="demand_count")
    corr = pivot.corr().values
    offdiag = corr[~np.eye(corr.shape[0], dtype=bool)]

    meta = {
        "seed": SEED,
        "city": CITY,
        "rows": int(len(df)),
        "hours": int(df["bucket_hour"].nunique()),
        "aois": int(df["aoi_id"].nunique()),
        "regions": int(df["region_id"].nunique()),
        "date_range": [str(df["bucket_hour"].min()), str(df["bucket_hour"].max())],
        "zero_share": float((df["demand_count"] == 0).mean()),
        "mean_demand": float(df["demand_count"].mean()),
        "median_demand": float(df["demand_count"].median()),
        "max_demand": int(df["demand_count"].max()),
        "cross_aoi_corr_mean": float(offdiag.mean()),
        "cross_aoi_corr_median": float(np.median(offdiag)),
        "per_archetype_summary": diag.groupby("aoi_archetype")[["mean", "std", "zero_share"]].mean().round(4).to_dict(),
        "notes": {
            "purpose": "Synthetic Saudi hourly AOI demand for forecasting experiments.",
            "strong_predictors": ["lag_24", "lag_168", "rolling means", "Saudi event flags", "hour/day/month"],
            "design_choice": "Aggressively tuned for higher predictability and higher exact-match forecasting accuracy."
        },
    }

    with open(OUT_META, "w") as f:
        json.dump(meta, f, indent=2)

    print("=" * 72)
    print("Saudi synthetic demand dataset generated")
    print("=" * 72)
    print(f"CSV:  {OUT_CSV}")
    print(f"META: {OUT_META}")
    print(f"Rows={len(df):,} | AOIs={df['aoi_id'].nunique()} | Hours={df['bucket_hour'].nunique()} | Regions={df['region_id'].nunique()}")
    print(f"Mean demand={df['demand_count'].mean():.3f} | Zero share={(df['demand_count']==0).mean():.3%} | Max={df['demand_count'].max()}")
    print(f"Cross-AOI corr mean={offdiag.mean():.4f} median={np.median(offdiag):.4f}")


if __name__ == "__main__":
    main()