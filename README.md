# From Forecasts to Decisions

**An uncertainty-aware energy-management study for commercial buildings**

This project tests a focused research question:

> Do calibrated probabilistic forecasts improve forecast-driven battery scheduling compared with point forecasts alone?

It uses hourly electricity readings from three offices in the open **Building Data Genome Project 2 (BDG2)**. A leakage-safe rolling-origin design compares a 24-hour seasonal-naive baseline with histogram gradient boosting. Quantile models and split-conformal calibration produce 90% prediction intervals. Those forecasts then drive a linear battery-scheduling model under an explicitly illustrative time-of-use tariff and carbon-intensity profile.

The completed numerical findings are in [reports/results.md](reports/results.md). Machine-readable results and figures are written to `outputs/`.

## Why this is useful

Many energy-forecasting projects stop at RMSE. This study connects predictive uncertainty to an operational decision, measures realised cost/peak outcomes, and reports when uncertainty helps—or does not help. That forecast-to-decision link is directly relevant to sustainable-AI research involving buildings, demand flexibility, optimisation, and uncertainty.

## Reproduce the study

1. Download the public BDG2 data:

   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts/fetch_bdg2.ps1
   ```

2. Install the Python dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Run the study and tests:

   ```powershell
   python run_study.py
   python -m unittest discover -s tests -v
   ```

The pipeline deterministically selects one high-completeness office meter from each of three sites, then records the selected buildings and data coverage in the report.

## Experimental design

- **Task:** day-ahead electricity forecasting, 24 hourly steps.
- **Features:** causal load lags, trailing means, and calendar encodings available at the forecast origin.
- **Split:** first 70% of daily origins for training, next 15% for interval calibration, final 15% for rolling-origin testing.
- **Forecasts:** seasonal naive; gradient-boosted point forecast; 5th, 50th, and 95th conditional quantiles.
- **Uncertainty:** held-out split-conformal widening of the 5–95% interval.
- **Decision:** linear battery schedule with charge/discharge/SOC limits and terminal-SOC equality.
- **Comparison:** no battery, point-forecast schedule, interval-aware risk schedule, and perfect-foresight oracle.
- **Metrics:** MAE, RMSE, MASE, empirical interval coverage/width, realised decision objective, peak demand, carbon, savings, and oracle regret.

## Important interpretation boundary

The electricity tariff, carbon profile, and battery are transparent experimental scenarios—not claims about the original sites' equipment or contracts. Consequently, the decision results test a method under stated assumptions; they do not estimate real site savings. The forecast results do use measured BDG2 electricity data.

## Data citation

Miller, C., Kathirgamanathan, A., Picchetti, B. *et al.* (2020). The Building Data Genome Project 2, energy meter data from the ASHRAE Great Energy Predictor III competition. *Scientific Data, 7*, 368. <https://doi.org/10.1038/s41597-020-00712-x>

Source repository: <https://github.com/buds-lab/building-data-genome-project-2>

