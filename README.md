# AI-Driven Last-Mile Delivery Optimization

A senior project that combines **demand forecasting**, **route optimization**, and **travel-time estimation** to build an end-to-end intelligent last-mile logistics system.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Dataset](#dataset)
4. [Part 1 — Demand Forecasting (COMPLETED)](#part-1--demand-forecasting-completed)
5. [Part 2 — Route Optimizer & Fleet Assignment (REMAINING)](#part-2--route-optimizer--fleet-assignment-remaining)
6. [Part 3 — Travel-Time & Emissions Model (REMAINING)](#part-3--travel-time--emissions-model-remaining)
7. [How the Three Parts Connect](#how-the-three-parts-connect)
8. [Results So Far](#results-so-far)
9. [Repository Structure](#repository-structure)
10. [Tech Stack](#tech-stack)

---

## Project Overview

Last-mile delivery is the most expensive and least efficient leg of any logistics network — it accounts for up to 53% of total shipping costs. This project builds a three-component system that addresses last-mile delivery from prediction through planning through execution:

| Component | Role | Status |
|---|---|---|
| **Model 1 — Demand Forecasting** | Predicts how many deliveries will arrive per zone per hour | ✅ Complete |
| **Optimizer — Route & Fleet** | Decides which vehicle goes where and in what order | 🔲 Remaining |
| **Model 2 — Travel-Time Estimation** | Provides realistic, time-dependent travel times and emissions costs to the optimizer | 🔲 Remaining |

The core idea: **predict** incoming demand, **plan** optimal routes, and **adapt** those plans in real time using learned travel costs.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PLANNING WINDOW                              │
│                                                                     │
│   ┌─────────────────┐    demand forecast     ┌──────────────────┐  │
│   │   Model 1       │ ─────────────────────► │                  │  │
│   │  (XGBoost       │                        │   OR-Tools       │  │
│   │   Demand        │   actual orders        │   Optimizer      │  │
│   │   Forecast)     │ ─────────────────────► │   (VRP + Fleet   │  │
│   └─────────────────┘                        │    Assignment)   │  │
│                                              │                  │  │
│   ┌─────────────────┐   realistic travel     │                  │  │
│   │   Model 2       │   times + emissions    │                  │  │
│   │  (Travel-Time / │ ─────────────────────► │                  │  │
│   │   Emissions     │                        └──────────┬───────┘  │
│   │   Estimation)   │                                   │          │
│   └────────┬────────┘                                   │ routes   │
│            │ learns from                                │          │
│            │ executed routes                            ▼          │
│            └──────────────────────────────── EXECUTION + FEEDBACK  │
└─────────────────────────────────────────────────────────────────────┘
```

**Flow in plain words:**
1. Model 1 forecasts how many packages will need to be delivered in each zone in the upcoming hours.
2. The Optimizer receives the actual confirmed orders for the planning window and uses the forecast as a look-ahead signal. It produces concrete decisions: which vehicle serves which stops and in what sequence, subject to constraints (capacity, time windows, fleet size) and objectives (minimize distance, cost, and CO₂ emissions).
3. As routes are executed and new GPS/traffic data arrives, Model 2 provides more realistic, time-dependent travel times (and optionally energy/emissions costs) back to the Optimizer, enabling better route quality and faster re-optimization under disruptions.

---

## Dataset

**LaDe-P (Cainiao / Alibaba Last-Mile Delivery dataset)**

- **Source:** Public dataset from Cainiao's last-mile operations in China.
- **Coverage:** May 2022 – October 2022 (6 months).
- **City used:** Chongqing (`pickup_cq.csv` — ~190 MB raw).
- **Granularity:** Individual package-level pickup/delivery events with timestamps and zone identifiers.
- **Key fields:**
  - `city`, `region_id`, `aoi_id` — geographic hierarchy (city → region → area-of-interest)
  - Timestamps → parsed and bucketed into **hourly** windows
  - `demand_count` — engineered target: number of deliveries per AOI per hour

---

## Part 1 — Demand Forecasting (COMPLETED)

### Goal
Predict `demand_count` (number of deliveries) for each `(city, region_id, aoi_id, hour)` combination up to several hours ahead.

### Notebooks

| Notebook | Purpose |
|---|---|
| `LaDe.ipynb` | Raw data → feature engineering → `lade_hourly_features.csv` |
| `LaDe_XGBoost_Forecast.ipynb` | Model training, evaluation, and comparison |

### Preprocessing & Feature Engineering (`LaDe.ipynb`)

- Parsed timestamps (no year in raw data → prepended `2022`).
- Bucketed into **hourly** time windows (`bucket_hour`).
- Aggregated to `(city, region_id, aoi_id, bucket_hour)` and computed `demand_count`.
- Filled missing hours per AOI with `demand_count = 0` (continuous series).
- Built time features: `hour`, `day_of_week`, `month`, weekend flag.
- Built lag features: `lag_1`, `lag_2`, `lag_24`, `lag_48`, `lag_168` (same hour yesterday / last week).
- Built rolling features: `roll_24_mean`, `roll_168_mean` (24h and 7-day rolling averages).
- Output: **`lade_hourly_features.csv`** (~65 MB, ready for modeling).

### Model Training (`LaDe_XGBoost_Forecast.ipynb`)

- **Algorithm:** XGBoost regression (gradient boosted trees).
- **Split strategy:** Chronological 60 / 20 / 20 (train / val / test) — no data leakage.
- **Train set:** May 2022 – ~Aug 2022
- **Test set starts:** 2022-08-14 23:00
- **Test rows:** 122,100 observations
- **Variants trained:** standard XGBoost, `reg:squaredlogerror` (better for sparse/zero-heavy targets), zero-inflated (logistic for P(demand>0) + conditional XGBoost).

### Results

| Model | MAE (val) | RMSE (val) | sMAPE (val) | MAE (test) | RMSE (test) | sMAPE (test) |
|---|---|---|---|---|---|---|
| Baseline `lag_24` | 0.3579 | 0.8710 | 131.85% | 0.3089 | 0.8020 | 135.58% |
| Baseline `roll_24_mean` | 0.5912 | 1.0219 | 182.29% | 0.4908 | 0.9039 | 184.20% |
| **XGBoost** | **0.2931** | **0.6595** | 168.94% | **0.2595** | **0.6040** | 172.40% |
| XGBoost (squaredlogerror) | 0.2945 | 0.6605 | 167.40% | 0.2612 | 0.6049 | 169.70% |
| XGBoost (zero-inflated) | 0.3099 | 0.6792 | 177.08% | 0.2709 | 0.6252 | 181.89% |

**Key takeaway:** XGBoost achieves **16% lower MAE** and **25% lower RMSE** versus the best naive baseline (`lag_24`) on the test set. The high sMAPE figures are expected — many AOIs have near-zero demand in off-peak hours, making percentage-based errors inflate.

**Top features by importance:** `lag_24`, `lag_168`, `roll_24_mean`, `hour`, `roll_168_mean` — confirming that day-of-week and time-of-day periodicity are the strongest signals.

---

## Part 2 — Route Optimizer & Fleet Assignment (REMAINING)

### Goal
Given a planning window of confirmed orders (stops to serve), decide:
- Which vehicle is assigned to which stops.
- The exact sequence of stops each vehicle follows.
- Subject to hard constraints and minimizing cost/distance/emissions.

### Approach: OR-Tools Vehicle Routing Problem (VRP)

We will use **Google OR-Tools** — specifically its `routing` library — which solves variants of the Vehicle Routing Problem.

**Problem formulation:**

```
Minimize:   Σ (travel distance) + α × Σ (CO₂ emissions)

Subject to:
  - Each stop served by exactly one vehicle
  - Vehicle capacity constraints (volume / weight)
  - Time window constraints per stop (customer available slots)
  - Fleet size limit (number of vehicles available)
  - Depot start/end constraints
```

### Fleet Assignment Sub-problem
Before or jointly with routing, we need to decide how many vehicles of each type to deploy:
- Small electric vans (lower emissions, lower capacity, cheaper per km)
- Large diesel trucks (higher capacity, higher emissions, higher cost)

This can be modeled as a Mixed-Integer Program or handled as a pre-processing heuristic before VRP.

### Role of Model 1 Output
- The demand forecast for the next planning window can be used to **pre-position** vehicles or **warm-start** the optimizer with expected clusters of demand.
- If actual orders are lighter/heavier than forecast, the optimizer re-solves with updated data.

### Key Constraints to Model
- Time windows per delivery stop
- Vehicle load capacity
- Driver working hours
- Return to depot

---

## Part 3 — Travel-Time & Emissions Model (REMAINING)

### Goal
Replace the static (average-speed) travel time matrix in the optimizer with a **learned, time-dependent** travel time estimate. Optionally also estimate energy consumption / CO₂ per road segment and vehicle type.

### Why This Matters
The optimizer's solution quality is only as good as the travel time estimates it uses. Static averages ignore:
- Rush hour congestion (travel time on the same road at 8 AM vs 2 PM can differ 3×).
- Day-of-week patterns.
- Disruptions (accidents, weather).

A learned model makes the optimizer's cost function more realistic and makes re-optimization under disruptions faster and more accurate.

### Approach (planned)
- **Input features:** origin-destination pair, time of day, day of week, recent historical speed on segment, weather (optional).
- **Target:** actual travel time (seconds) for that O-D pair at that time.
- **Candidate models:** XGBoost (consistent with Part 1) or a lightweight neural network.
- **Emissions extension:** given vehicle type + speed profile, estimate CO₂ (g/km) using standard emission factor models (COPERT-style) or a learned surrogate.

### Integration with Optimizer
At each re-optimization event, the travel-time model provides an updated cost matrix to OR-Tools, enabling:
- More realistic initial route plans.
- Faster and better re-routing when a disruption occurs mid-route.

---

## How the Three Parts Connect

```
Hour H:     Model 1 forecasts demand for hours H+1 … H+K
Hour H+1:   Confirmed orders arrive for planning window
            → Optimizer solves VRP using:
                  - Actual confirmed stops
                  - Model 1 forecast as look-ahead
                  - Model 2 travel-time matrix (time-of-day aware)
            → Dispatch routes to drivers

During execution:
            - GPS traces collected
            - Actual travel times recorded
            - Model 2 updated (online or periodic retraining)

Disruption: → Optimizer re-solves with updated Model 2 costs
            → Revised routes pushed to affected drivers

End of day: → Execution data feeds back into Model 1 training
              and Model 2 training for next cycle
```

---

## Results So Far

| Milestone | Status |
|---|---|
| Data acquisition (LaDe-P Chongqing) | ✅ Done |
| Hourly AOI-level feature engineering | ✅ Done |
| Chronological train/val/test split | ✅ Done |
| Baseline models (`lag_24`, `roll_24_mean`) | ✅ Done |
| XGBoost demand forecast (3 variants) | ✅ Done |
| Model evaluation (MAE / RMSE / sMAPE) | ✅ Done |
| Feature importance analysis | ✅ Done |
| OR-Tools VRP optimizer | 🔲 Next |
| Fleet assignment module | 🔲 Pending |
| Travel-time estimation model | 🔲 Pending |
| End-to-end integration | 🔲 Pending |

---

## Repository Structure

```
senior project/
├── LaDe.ipynb                    # Part 1a: preprocessing + feature engineering
├── LaDe_XGBoost_Forecast.ipynb   # Part 1b: XGBoost model training + evaluation
├── lade_hourly_features.csv      # Engineered feature table (~65 MB, git-ignored)
├── pickup_cq.csv                 # Raw LaDe-P Chongqing data (~190 MB, git-ignored)
├── NOTEBOOKS.md                  # Short guide to running the notebooks
└── README.md                     # This file
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| Data processing | `pandas`, `numpy` |
| Demand forecasting | `XGBoost 3.2` |
| Route optimization | `OR-Tools` (planned) |
| Travel-time model | `XGBoost` / `scikit-learn` (planned) |
| Evaluation | `MAE`, `RMSE`, `sMAPE` |
| Environment | Google Colab (GPU/CPU), Python 3.x |
| Version control | Git |

---

## Quick-Start (Demand Forecasting)

```bash
# 1. Open in Google Colab or Jupyter
# 2. Run LaDe.ipynb — generates lade_hourly_features.csv
# 3. Run LaDe_XGBoost_Forecast.ipynb — trains and evaluates models

# Data: upload pickup_cq.csv to Google Drive, update path in first cell of LaDe.ipynb
```
