# Results: From Forecasts to Decisions

## Executive finding

The interval-aware policy increased the realised objective by 1.06% relative to point-forecast scheduling, while reducing mean peak demand by 2.87%. Peak demand fell in 3 of 3 buildings, but the combined objective improved in only 1 of 3. Under this scenario, uncertainty changed the cost–risk trade-off rather than improving every objective.

This is a deliberately small, reproducible study rather than a claim of deployment readiness. It demonstrates the full chain from measured building data, through leakage-safe probabilistic forecasting, to optimisation and realised decision evaluation.

## Research question

**Do calibrated probabilistic forecasts improve forecast-driven battery scheduling compared with point forecasts alone?**

## Data and buildings

The study used the cleaned electricity file from Building Data Genome Project 2. The deterministic selection rule chose a high-completeness office meter from each of three different sites.

| Building | Site | Subtype | Area_sqm | Coverage_pct |
| --- | --- | --- | --- | --- |
| Hog_office_Lavon | Hog | Office | 1530.30 | 100.00 |
| Lamb_office_Caitlin | Lamb | Office | 3365.00 | 100.00 |
| Robin_office_Lindsay | Robin | Office | 5358.00 | 100.00 |

## Evaluation protocol

- Day-ahead forecasts contain 24 hourly steps and are issued at 23:00.
- All lag and trailing-window features end at the forecast origin; no target-period load enters the feature set.
- Daily forecast origins are split chronologically: 70% training, 15% calibration, and 15% test.
- The held-out calibration period widens 5th–95th quantile forecasts using split conformal prediction for a nominal 90% interval.
- The decision test uses a simulated four-hour battery sized from each building's training median load, with 95% one-way efficiency and equal initial/final planned state of charge.
- The illustrative tariff is £0.16/kWh overnight, £0.24/kWh off-peak, and £0.38/kWh from 16:00–20:00; carbon intensity is 0.15/0.22/0.30 kgCO₂e/kWh by period, with a £0.12/kgCO₂e carbon price and £0.45/kW-day peak charge.
- The interval-aware controller uses lower/median/upper scenarios with 20%/60%/20% weights plus a 0.35 worst-case peak penalty. All scenario settings are held constant across buildings.

## Forecast performance

Metrics below are unweighted means across the three buildings. MASE below 1 indicates improvement over the in-sample 24-hour seasonal scale.

| method | MAE | RMSE | MASE |
| --- | --- | --- | --- |
| Gradient boosting | 33.963 | 78.690 | 0.893 |
| Seasonal naive | 39.790 | 105.487 | 1.124 |

### Prediction interval

| Nominal coverage | Empirical coverage | Normalised width |
| --- | --- | --- |
| 0.900 | 0.893 | 1.522 |

Building-level interval diagnostics are stored in `outputs/interval_metrics.csv`; this prevents an average coverage figure from hiding site-specific calibration differences.

## Decision performance

The realised objective combines energy charge, an explicit carbon-price term, and daily peak charge. Lower is better. The oracle uses the actual next-day load and is a performance bound, not a deployable method.

| method | Mean_objective | Mean_peak | Mean_emissions_kg | Saving_vs_no_battery_pct | Regret_vs_oracle |
| --- | --- | --- | --- | --- | --- |
| No battery | 672.166 | 146.046 | 479.985 | 0.000 | 15.807 |
| Point forecast | 660.672 | 149.615 | 473.985 | 1.710 | 4.313 |
| Interval-aware | 667.679 | 145.319 | 477.650 | 0.668 | 11.320 |
| Oracle | 656.359 | 143.470 | 472.856 | 2.352 | 0.000 |

Positive objective change below means the interval-aware policy was worse; positive peak reduction means it reduced the realised peak.

| Building | Point_objective | Interval_objective | Objective_change_pct | Peak_reduction_pct |
| --- | --- | --- | --- | --- |
| Hog_office_Lavon | 1293.336 | 1295.151 | 0.140 | 2.327 |
| Lamb_office_Caitlin | 284.845 | 284.701 | -0.051 | 0.395 |
| Robin_office_Lindsay | 403.836 | 423.185 | 4.791 | 6.952 |

**Answer to the research question:** The interval-aware policy increased the realised objective by 1.06% relative to point-forecast scheduling, while reducing mean peak demand by 2.87%. Peak demand fell in 3 of 3 buildings, but the combined objective improved in only 1 of 3. Under this scenario, uncertainty changed the cost–risk trade-off rather than improving every objective.

The result should be interpreted as evidence about this decision setup, not as a universal verdict on probabilistic forecasting. Forecast value depends on tariff structure, storage size, uncertainty quality, and the penalty for cautious scheduling.

## Limitations

1. Only three office meters are used; broader claims require more buildings and building types.
2. The model uses load history and calendar variables but not weather, occupancy, or known operational schedules.
3. Tariff, carbon, battery, and peak-charge settings are transparent experimental assumptions rather than site contracts or installed assets.
4. Each test day starts from the same state of charge. Multi-day control would require sequential state carry-over.
5. This is an observational forecasting benchmark and simulated control study, not a causal or field-deployment evaluation.
6. Split-conformal calibration treats hourly residual rows as exchangeable; a block- or day-level conformal method would better respect serial dependence.

## Application-ready description

**Project title:** *From Forecasts to Decisions: Uncertainty-Aware Energy Management for Commercial Buildings*

**CV bullet:** Built a leakage-safe day-ahead energy forecasting and optimisation pipeline on three real BDG2 office meters, combining gradient boosting, calibrated 90% prediction intervals, rolling-origin evaluation, and linear battery scheduling; reduced mean MASE from 1.124 to 0.893 and found that interval-aware control cut mean peak demand by 2.87% (0.39–6.95% across all buildings), with a 1.06% trade-off in the combined objective.

## Reproducibility record

- Random seed: `2027`
- Horizon: `24` hours
- Nominal interval coverage: `90%`
- Decision days evaluated: `327`
- Generated figures: `outputs/figures/forecast_example.png`, `outputs/figures/decision_comparison.png`
- Machine-readable tables: `outputs/forecast_metrics.csv`, `outputs/interval_metrics.csv`, `outputs/decision_results.csv`

## Citation

Miller, C., Kathirgamanathan, A., Picchetti, B. *et al.* (2020). The Building Data Genome Project 2, energy meter data from the ASHRAE Great Energy Predictor III competition. *Scientific Data, 7*, 368. <https://doi.org/10.1038/s41597-020-00712-x>
