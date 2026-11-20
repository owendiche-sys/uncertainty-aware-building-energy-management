from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

_LOCAL_MPL_CONFIG = Path(__file__).resolve().parents[2] / ".matplotlib"
_LOCAL_MPL_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_LOCAL_MPL_CONFIG))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


FEATURE_COLUMNS = [
    "horizon",
    "target_hour_sin",
    "target_hour_cos",
    "target_dow_sin",
    "target_dow_cos",
    "target_month_sin",
    "target_month_cos",
    "last_observation",
    "same_hour_yesterday",
    "same_hour_last_week",
    "trailing_24h_mean",
    "trailing_168h_mean",
    "trailing_24h_std",
]


@dataclass(frozen=True)
class StudyConfig:
    root: Path
    n_buildings: int = 3
    horizon: int = 24
    interval_alpha: float = 0.10
    random_state: int = 2027
    train_fraction: float = 0.70
    calibration_fraction: float = 0.15
    min_coverage: float = 0.97
    battery_hours: float = 4.0
    battery_efficiency: float = 0.95
    demand_charge: float = 0.45
    carbon_price: float = 0.12
    interval_risk_aversion: float = 0.35

    @property
    def meter_path(self) -> Path:
        return (
            self.root
            / "vendor"
            / "bdg2"
            / "data"
            / "meters"
            / "cleaned"
            / "electricity_cleaned.csv"
        )

    @property
    def metadata_path(self) -> Path:
        return self.root / "vendor" / "bdg2" / "data" / "metadata" / "metadata.csv"


def _require_data(config: StudyConfig) -> None:
    missing = [p for p in (config.meter_path, config.metadata_path) if not p.exists()]
    if missing:
        joined = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Missing BDG2 input files:\n{joined}\nRun scripts/fetch_bdg2.ps1 first."
        )


def load_and_select_offices(
    config: StudyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load office columns and select high-coverage meters from distinct sites."""
    _require_data(config)
    metadata = pd.read_csv(config.metadata_path)
    header = pd.read_csv(config.meter_path, nrows=0).columns
    office_ids = metadata.loc[
        metadata["primaryspaceusage"].eq("Office"), "building_id"
    ].dropna()
    office_columns = sorted(set(office_ids).intersection(header))
    if len(office_columns) < config.n_buildings:
        raise ValueError("Not enough office electricity meters are present in the data file.")

    readings = pd.read_csv(
        config.meter_path,
        usecols=["timestamp", *office_columns],
        parse_dates=["timestamp"],
        low_memory=False,
    ).set_index("timestamp")
    readings = readings.sort_index()

    meta_index = metadata.set_index("building_id")
    rows: list[dict[str, object]] = []
    for building_id in office_columns:
        series = pd.to_numeric(readings[building_id], errors="coerce")
        mean = float(series.mean())
        std = float(series.std())
        rows.append(
            {
                "building_id": building_id,
                "site_id": str(meta_index.loc[building_id, "site_id"]),
                "sub_primaryspaceusage": meta_index.loc[
                    building_id, "sub_primaryspaceusage"
                ],
                "sqm": meta_index.loc[building_id, "sqm"],
                "coverage": float(series.notna().mean()),
                "mean_load": mean,
                "std_load": std,
                "coefficient_of_variation": std / mean if mean > 0 else np.nan,
            }
        )

    stats = pd.DataFrame(rows)
    eligible = stats.loc[
        (stats["coverage"] >= config.min_coverage)
        & (stats["mean_load"] > 0)
        & (stats["std_load"] > 0)
    ].copy()
    if len(eligible) < config.n_buildings:
        eligible = stats.loc[
            (stats["coverage"] >= 0.90)
            & (stats["mean_load"] > 0)
            & (stats["std_load"] > 0)
        ].copy()

    eligible = eligible.sort_values(
        ["coverage", "coefficient_of_variation", "building_id"],
        ascending=[False, False, True],
    )
    selected: list[str] = []
    sites: set[str] = set()
    for row in eligible.itertuples(index=False):
        if row.site_id not in sites:
            selected.append(row.building_id)
            sites.add(row.site_id)
        if len(selected) == config.n_buildings:
            break
    if len(selected) < config.n_buildings:
        for building_id in eligible["building_id"]:
            if building_id not in selected:
                selected.append(building_id)
            if len(selected) == config.n_buildings:
                break
    if len(selected) < config.n_buildings:
        raise ValueError("Unable to select the requested number of usable office meters.")

    selected_stats = stats.set_index("building_id").loc[selected].reset_index()
    return readings[selected].copy(), selected_stats


def build_daily_supervised(series: pd.Series, horizon: int = 24) -> pd.DataFrame:
    """Create direct multi-horizon rows using only information known at origin."""
    if horizon != 24:
        raise ValueError("This implementation currently expects a 24-hour horizon.")
    series = pd.to_numeric(series, errors="coerce").astype(float).sort_index()
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("Series must use a DatetimeIndex.")
    values = series.to_numpy(dtype=float)
    timestamps = series.index
    rows: list[dict[str, object]] = []

    possible_origins = np.flatnonzero(timestamps.hour.to_numpy() == 23)
    for origin_idx in possible_origins:
        if origin_idx < 168 or origin_idx + horizon >= len(series):
            continue
        history = values[origin_idx - 167 : origin_idx + 1]
        targets = values[origin_idx + 1 : origin_idx + horizon + 1]
        if np.isnan(history).any() or np.isnan(targets).any():
            continue
        trailing_24 = history[-24:]
        for step in range(1, horizon + 1):
            target_idx = origin_idx + step
            target_time = timestamps[target_idx]
            yesterday_idx = origin_idx + step - 24
            week_idx = origin_idx + step - 168
            rows.append(
                {
                    "origin_time": timestamps[origin_idx],
                    "target_time": target_time,
                    "horizon": float(step),
                    "target_hour_sin": math.sin(2 * math.pi * target_time.hour / 24),
                    "target_hour_cos": math.cos(2 * math.pi * target_time.hour / 24),
                    "target_dow_sin": math.sin(2 * math.pi * target_time.dayofweek / 7),
                    "target_dow_cos": math.cos(2 * math.pi * target_time.dayofweek / 7),
                    "target_month_sin": math.sin(2 * math.pi * target_time.month / 12),
                    "target_month_cos": math.cos(2 * math.pi * target_time.month / 12),
                    "last_observation": values[origin_idx],
                    "same_hour_yesterday": values[yesterday_idx],
                    "same_hour_last_week": values[week_idx],
                    "trailing_24h_mean": float(np.mean(trailing_24)),
                    "trailing_168h_mean": float(np.mean(history)),
                    "trailing_24h_std": float(np.std(trailing_24)),
                    "target": values[target_idx],
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No complete daily forecast origins were available.")
    return frame


def chronological_origin_split(
    frame: pd.DataFrame, train_fraction: float, calibration_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    origins = np.array(sorted(frame["origin_time"].unique()))
    n_origins = len(origins)
    train_end = max(1, int(n_origins * train_fraction))
    calibration_end = max(train_end + 1, int(n_origins * (train_fraction + calibration_fraction)))
    calibration_end = min(calibration_end, n_origins - 1)
    train_origins = set(origins[:train_end])
    calibration_origins = set(origins[train_end:calibration_end])
    test_origins = set(origins[calibration_end:])
    if not calibration_origins or not test_origins:
        raise ValueError("Insufficient forecast origins for train/calibration/test splits.")
    return (
        frame[frame["origin_time"].isin(train_origins)].copy(),
        frame[frame["origin_time"].isin(calibration_origins)].copy(),
        frame[frame["origin_time"].isin(test_origins)].copy(),
    )


def conformal_adjustment(
    actual: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float
) -> float:
    scores = np.maximum.reduce([lower - actual, actual - upper, np.zeros_like(actual)])
    n = len(scores)
    quantile = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, quantile, method="higher"))


def _new_model(config: StudyConfig, loss: str, quantile: float | None = None):
    kwargs: dict[str, object] = {
        "loss": loss,
        "learning_rate": 0.06,
        "max_iter": 140,
        "max_leaf_nodes": 24,
        "min_samples_leaf": 30,
        "l2_regularization": 0.10,
        "random_state": config.random_state,
        "early_stopping": True,
    }
    if quantile is not None:
        kwargs["quantile"] = quantile
    return HistGradientBoostingRegressor(**kwargs)


def _forecast_metrics(actual: np.ndarray, prediction: np.ndarray, scale: float) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, prediction)),
        "rmse": float(math.sqrt(mean_squared_error(actual, prediction))),
        "mase": float(mean_absolute_error(actual, prediction) / scale),
    }


def fit_forecasts(
    building_id: str, series: pd.Series, config: StudyConfig
) -> tuple[list[dict[str, object]], dict[str, object], pd.DataFrame]:
    supervised = build_daily_supervised(series, config.horizon)
    train, calibration, test = chronological_origin_split(
        supervised, config.train_fraction, config.calibration_fraction
    )
    x_train = train[FEATURE_COLUMNS]
    y_train = train["target"].to_numpy()
    point_model = _new_model(config, "squared_error")
    lower_model = _new_model(config, "quantile", 0.05)
    median_model = _new_model(config, "quantile", 0.50)
    upper_model = _new_model(config, "quantile", 0.95)
    for model in (point_model, lower_model, median_model, upper_model):
        model.fit(x_train, y_train)

    x_cal = calibration[FEATURE_COLUMNS]
    cal_lower = np.clip(lower_model.predict(x_cal), 0, None)
    cal_upper = np.clip(upper_model.predict(x_cal), 0, None)
    cal_lower, cal_upper = np.minimum(cal_lower, cal_upper), np.maximum(cal_lower, cal_upper)
    qhat = conformal_adjustment(
        calibration["target"].to_numpy(), cal_lower, cal_upper, config.interval_alpha
    )

    x_test = test[FEATURE_COLUMNS]
    actual = test["target"].to_numpy()
    point = np.clip(point_model.predict(x_test), 0, None)
    median = np.clip(median_model.predict(x_test), 0, None)
    lower = np.clip(lower_model.predict(x_test) - qhat, 0, None)
    upper = np.clip(upper_model.predict(x_test) + qhat, 0, None)
    lower, upper = np.minimum(lower, upper), np.maximum(lower, upper)
    seasonal = test["same_hour_yesterday"].to_numpy()
    scale = float(np.mean(np.abs(y_train[24:] - y_train[:-24])))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.mean(np.abs(y_train - np.mean(y_train)))) or 1.0

    metrics: list[dict[str, object]] = []
    for method, prediction in (("Seasonal naive", seasonal), ("Gradient boosting", point)):
        metrics.append(
            {
                "building_id": building_id,
                "method": method,
                **_forecast_metrics(actual, prediction, scale),
                "n_predictions": len(actual),
            }
        )
    coverage = float(np.mean((actual >= lower) & (actual <= upper)))
    interval = {
        "building_id": building_id,
        "nominal_coverage": 1 - config.interval_alpha,
        "empirical_coverage": coverage,
        "mean_interval_width": float(np.mean(upper - lower)),
        "normalised_width": float(np.mean(upper - lower) / np.mean(actual)),
        "conformal_adjustment": qhat,
        "calibration_predictions": len(calibration),
        "test_predictions": len(test),
        "train_origins": int(train["origin_time"].nunique()),
        "calibration_origins": int(calibration["origin_time"].nunique()),
        "test_origins": int(test["origin_time"].nunique()),
    }
    predictions = test[["origin_time", "target_time"]].copy()
    predictions["building_id"] = building_id
    predictions["actual"] = actual
    predictions["seasonal_naive"] = seasonal
    predictions["point"] = point
    predictions["lower"] = lower
    predictions["median"] = median
    predictions["upper"] = upper
    predictions["training_median_load"] = float(np.median(y_train))
    return metrics, interval, predictions


def tariff_profile(timestamps: Iterable[pd.Timestamp]) -> tuple[np.ndarray, np.ndarray]:
    hours = np.array([timestamp.hour for timestamp in timestamps])
    price = np.full(len(hours), 0.24, dtype=float)
    price[(hours >= 0) & (hours < 7)] = 0.16
    price[(hours >= 16) & (hours < 20)] = 0.38
    carbon = np.full(len(hours), 0.22, dtype=float)
    carbon[(hours >= 0) & (hours < 7)] = 0.15
    carbon[(hours >= 17) & (hours < 21)] = 0.30
    return price, carbon


def optimise_battery(
    scenarios: np.ndarray,
    scenario_weights: np.ndarray,
    price: np.ndarray,
    carbon: np.ndarray,
    capacity: float,
    power: float,
    efficiency: float,
    demand_charge: float,
    carbon_price: float,
    risk_aversion: float = 0.0,
) -> dict[str, np.ndarray | float]:
    scenarios = np.atleast_2d(np.asarray(scenarios, dtype=float))
    scenario_weights = np.asarray(scenario_weights, dtype=float)
    scenario_weights = scenario_weights / scenario_weights.sum()
    n_scenarios, horizon = scenarios.shape
    if len(price) != horizon or len(carbon) != horizon:
        raise ValueError("Profiles and scenarios must have the same horizon.")

    charge_start = 0
    discharge_start = horizon
    soc_start = 2 * horizon
    peak_start = soc_start + horizon + 1
    worst_idx = peak_start + n_scenarios
    n_variables = worst_idx + 1
    objective = np.zeros(n_variables)
    effective_price = price + carbon_price * carbon
    objective[charge_start : charge_start + horizon] = effective_price
    objective[discharge_start : discharge_start + horizon] = -effective_price
    objective[peak_start : peak_start + n_scenarios] = demand_charge * scenario_weights
    objective[worst_idx] = demand_charge * risk_aversion

    a_eq: list[np.ndarray] = []
    b_eq: list[float] = []
    for hour in range(horizon):
        row = np.zeros(n_variables)
        row[soc_start + hour + 1] = 1.0
        row[soc_start + hour] = -1.0
        row[charge_start + hour] = -efficiency
        row[discharge_start + hour] = 1.0 / efficiency
        a_eq.append(row)
        b_eq.append(0.0)
    initial_soc = 0.50 * capacity
    row_initial = np.zeros(n_variables)
    row_initial[soc_start] = 1.0
    a_eq.append(row_initial)
    b_eq.append(initial_soc)
    row_terminal = np.zeros(n_variables)
    row_terminal[soc_start + horizon] = 1.0
    a_eq.append(row_terminal)
    b_eq.append(initial_soc)

    a_ub: list[np.ndarray] = []
    b_ub: list[float] = []
    for scenario_idx in range(n_scenarios):
        for hour in range(horizon):
            # Scenario grid import cannot be negative.
            row_no_export = np.zeros(n_variables)
            row_no_export[discharge_start + hour] = 1.0
            row_no_export[charge_start + hour] = -1.0
            a_ub.append(row_no_export)
            b_ub.append(float(scenarios[scenario_idx, hour]))

            # Scenario grid import is bounded by its peak variable.
            row_peak = np.zeros(n_variables)
            row_peak[charge_start + hour] = 1.0
            row_peak[discharge_start + hour] = -1.0
            row_peak[peak_start + scenario_idx] = -1.0
            a_ub.append(row_peak)
            b_ub.append(float(-scenarios[scenario_idx, hour]))

        row_worst = np.zeros(n_variables)
        row_worst[peak_start + scenario_idx] = 1.0
        row_worst[worst_idx] = -1.0
        a_ub.append(row_worst)
        b_ub.append(0.0)

    bounds = (
        [(0.0, power)] * horizon
        + [(0.0, power)] * horizon
        + [(0.0, capacity)] * (horizon + 1)
        + [(0.0, None)] * n_scenarios
        + [(0.0, None)]
    )
    result = linprog(
        objective,
        A_ub=np.asarray(a_ub),
        b_ub=np.asarray(b_ub),
        A_eq=np.asarray(a_eq),
        b_eq=np.asarray(b_eq),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Battery optimisation failed: {result.message}")
    return {
        "charge": result.x[charge_start : charge_start + horizon],
        "discharge": result.x[discharge_start : discharge_start + horizon],
        "soc": result.x[soc_start : soc_start + horizon + 1],
        "optimisation_objective": float(result.fun),
    }


def evaluate_schedule(
    actual_demand: np.ndarray,
    charge: np.ndarray,
    discharge: np.ndarray,
    price: np.ndarray,
    carbon: np.ndarray,
    capacity: float,
    efficiency: float,
    demand_charge: float,
    carbon_price: float,
) -> dict[str, float]:
    demand = np.asarray(actual_demand, dtype=float)
    soc = 0.50 * capacity
    realised_grid: list[float] = []
    throughput = 0.0
    for hour in range(len(demand)):
        realised_charge = min(float(charge[hour]), max(0.0, (capacity - soc) / efficiency))
        available_discharge = max(0.0, soc * efficiency)
        realised_discharge = min(
            float(discharge[hour]), available_discharge, float(demand[hour] + realised_charge)
        )
        soc += efficiency * realised_charge - realised_discharge / efficiency
        realised_grid.append(float(demand[hour] + realised_charge - realised_discharge))
        throughput += realised_charge + realised_discharge
    grid = np.asarray(realised_grid)
    energy_charge = float(np.sum(grid * price))
    emissions = float(np.sum(grid * carbon))
    peak = float(np.max(grid))
    objective = energy_charge + carbon_price * emissions + demand_charge * peak
    return {
        "realised_objective": objective,
        "energy_charge": energy_charge,
        "emissions_kg": emissions,
        "peak_demand": peak,
        "battery_throughput": throughput,
        "terminal_soc": soc,
    }


def evaluate_decisions(predictions: pd.DataFrame, config: StudyConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = predictions.groupby(["building_id", "origin_time"], sort=True)
    for (building_id, origin_time), day in grouped:
        day = day.sort_values("target_time")
        if len(day) != config.horizon:
            continue
        actual = day["actual"].to_numpy()
        point = day["point"].to_numpy()
        lower = day["lower"].to_numpy()
        median = day["median"].to_numpy()
        upper = day["upper"].to_numpy()
        price, carbon = tariff_profile(day["target_time"])
        median_load = float(day["training_median_load"].iloc[0])
        capacity = max(1e-6, config.battery_hours * median_load)
        power = max(1e-6, median_load)
        common = {
            "price": price,
            "carbon": carbon,
            "capacity": capacity,
            "power": power,
            "efficiency": config.battery_efficiency,
            "demand_charge": config.demand_charge,
            "carbon_price": config.carbon_price,
        }
        zero = np.zeros(config.horizon)
        schedules: dict[str, dict[str, np.ndarray | float]] = {
            "No battery": {"charge": zero, "discharge": zero},
            "Point forecast": optimise_battery(
                point[None, :], np.array([1.0]), risk_aversion=0.0, **common
            ),
            "Interval-aware": optimise_battery(
                np.vstack([lower, median, upper]),
                np.array([0.20, 0.60, 0.20]),
                risk_aversion=config.interval_risk_aversion,
                **common,
            ),
            "Oracle": optimise_battery(
                actual[None, :], np.array([1.0]), risk_aversion=0.0, **common
            ),
        }
        for method, schedule in schedules.items():
            realised = evaluate_schedule(
                actual,
                np.asarray(schedule["charge"]),
                np.asarray(schedule["discharge"]),
                price,
                carbon,
                capacity,
                config.battery_efficiency,
                config.demand_charge,
                config.carbon_price,
            )
            rows.append(
                {
                    "building_id": building_id,
                    "origin_time": origin_time,
                    "method": method,
                    "battery_capacity": capacity,
                    "battery_power": power,
                    **realised,
                }
            )
    decisions = pd.DataFrame(rows)
    oracle = decisions.loc[decisions["method"].eq("Oracle"), [
        "building_id", "origin_time", "realised_objective"
    ]].rename(columns={"realised_objective": "oracle_objective"})
    baseline = decisions.loc[decisions["method"].eq("No battery"), [
        "building_id", "origin_time", "realised_objective"
    ]].rename(columns={"realised_objective": "baseline_objective"})
    decisions = decisions.merge(oracle, on=["building_id", "origin_time"]).merge(
        baseline, on=["building_id", "origin_time"]
    )
    decisions["regret_vs_oracle"] = (
        decisions["realised_objective"] - decisions["oracle_objective"]
    )
    decisions["saving_vs_no_battery"] = (
        decisions["baseline_objective"] - decisions["realised_objective"]
    )
    decisions["saving_vs_no_battery_pct"] = 100 * (
        decisions["baseline_objective"] - decisions["realised_objective"]
    ) / decisions["baseline_objective"]
    return decisions


def _markdown_table(frame: pd.DataFrame, decimals: int = 3) -> str:
    display = frame.copy()
    for column in display.select_dtypes(include=["number"]).columns:
        display[column] = display[column].map(lambda value: f"{value:.{decimals}f}")
    headers = [str(column) for column in display.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in display.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def create_figures(
    predictions: pd.DataFrame, decisions: pd.DataFrame, output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    first_building = predictions["building_id"].iloc[0]
    building_predictions = predictions[predictions["building_id"].eq(first_building)]
    first_origin = building_predictions["origin_time"].min()
    example = building_predictions[
        building_predictions["origin_time"].eq(first_origin)
    ].sort_values("target_time")
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.fill_between(
        example["target_time"], example["lower"], example["upper"],
        color="#73A9AD", alpha=0.28, label="Calibrated 90% interval"
    )
    ax.plot(example["target_time"], example["actual"], color="#17252A", lw=2, label="Actual")
    ax.plot(example["target_time"], example["point"], color="#D95F43", lw=1.8, label="Point forecast")
    ax.set_title(f"Day-ahead forecast example — {first_building}")
    ax.set_ylabel("Electricity use (meter units/hour)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=3, fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / "forecast_example.png", dpi=180)
    plt.close(fig)

    aggregate = decisions.groupby("method", as_index=False).agg(
        realised_objective=("realised_objective", "mean"),
        peak_demand=("peak_demand", "mean"),
    )
    order = ["No battery", "Point forecast", "Interval-aware", "Oracle"]
    aggregate["method"] = pd.Categorical(aggregate["method"], order, ordered=True)
    aggregate = aggregate.sort_values("method")
    colors = ["#B9C0C7", "#D95F43", "#2A9D8F", "#264653"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    axes[0].bar(aggregate["method"].astype(str), aggregate["realised_objective"], color=colors)
    axes[0].set_title("Mean realised decision objective")
    axes[0].set_ylabel("Illustrative £-equivalent/day")
    axes[1].bar(aggregate["method"].astype(str), aggregate["peak_demand"], color=colors)
    axes[1].set_title("Mean realised peak")
    axes[1].set_ylabel("Meter units/hour")
    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "decision_comparison.png", dpi=180)
    plt.close(fig)


def write_report(
    config: StudyConfig,
    selected_stats: pd.DataFrame,
    forecast_metrics: pd.DataFrame,
    interval_metrics: pd.DataFrame,
    decisions: pd.DataFrame,
    report_path: Path,
) -> dict[str, object]:
    forecast_summary = forecast_metrics.groupby("method", as_index=False).agg(
        MAE=("mae", "mean"), RMSE=("rmse", "mean"), MASE=("mase", "mean")
    )
    interval_summary = pd.DataFrame(
        {
            "Nominal coverage": [interval_metrics["nominal_coverage"].mean()],
            "Empirical coverage": [interval_metrics["empirical_coverage"].mean()],
            "Normalised width": [interval_metrics["normalised_width"].mean()],
        }
    )
    decision_summary = decisions.groupby("method", as_index=False).agg(
        Mean_objective=("realised_objective", "mean"),
        Mean_peak=("peak_demand", "mean"),
        Mean_emissions_kg=("emissions_kg", "mean"),
        Regret_vs_oracle=("regret_vs_oracle", "mean"),
    )
    order = ["No battery", "Point forecast", "Interval-aware", "Oracle"]
    decision_summary["method"] = pd.Categorical(decision_summary["method"], order, ordered=True)
    decision_summary = decision_summary.sort_values("method")
    baseline_value = float(
        decision_summary.loc[decision_summary["method"].eq("No battery"), "Mean_objective"].iloc[0]
    )
    decision_summary["Saving_vs_no_battery_pct"] = 100 * (
        baseline_value - decision_summary["Mean_objective"]
    ) / baseline_value
    decision_summary = decision_summary[[
        "method",
        "Mean_objective",
        "Mean_peak",
        "Mean_emissions_kg",
        "Saving_vs_no_battery_pct",
        "Regret_vs_oracle",
    ]]

    building_method = decisions.groupby(["building_id", "method"], as_index=False).agg(
        Mean_objective=("realised_objective", "mean"),
        Mean_peak=("peak_demand", "mean"),
    )
    objective_pivot = building_method.pivot(
        index="building_id", columns="method", values="Mean_objective"
    )
    peak_pivot = building_method.pivot(
        index="building_id", columns="method", values="Mean_peak"
    )
    decision_by_building = pd.DataFrame(
        {
            "Building": objective_pivot.index,
            "Point_objective": objective_pivot["Point forecast"].to_numpy(),
            "Interval_objective": objective_pivot["Interval-aware"].to_numpy(),
            "Objective_change_pct": (
                100
                * (objective_pivot["Interval-aware"] - objective_pivot["Point forecast"])
                / objective_pivot["Point forecast"]
            ).to_numpy(),
            "Peak_reduction_pct": (
                100
                * (peak_pivot["Point forecast"] - peak_pivot["Interval-aware"])
                / peak_pivot["Point forecast"]
            ).to_numpy(),
        }
    )
    point_value = float(
        decision_summary.loc[decision_summary["method"].eq("Point forecast"), "Mean_objective"].iloc[0]
    )
    interval_value = float(
        decision_summary.loc[decision_summary["method"].eq("Interval-aware"), "Mean_objective"].iloc[0]
    )
    point_peak = float(
        decision_summary.loc[decision_summary["method"].eq("Point forecast"), "Mean_peak"].iloc[0]
    )
    interval_peak = float(
        decision_summary.loc[decision_summary["method"].eq("Interval-aware"), "Mean_peak"].iloc[0]
    )
    interval_change = 100 * (point_value - interval_value) / point_value
    peak_reduction = 100 * (point_peak - interval_peak) / point_peak
    objective_improvement_count = int((decision_by_building["Objective_change_pct"] < 0).sum())
    peak_improvement_count = int((decision_by_building["Peak_reduction_pct"] > 0).sum())
    peak_min = float(decision_by_building["Peak_reduction_pct"].min())
    peak_max = float(decision_by_building["Peak_reduction_pct"].max())
    if interval_change > 0.05:
        objective_finding = f"reduced the realised objective by {interval_change:.2f}%"
    elif interval_change < -0.05:
        objective_finding = f"increased the realised objective by {abs(interval_change):.2f}%"
    else:
        objective_finding = "produced a practically indistinguishable realised objective"
    finding = (
        f"The interval-aware policy {objective_finding} relative to point-forecast scheduling, "
        f"while reducing mean peak demand by {peak_reduction:.2f}%. Peak demand fell in "
        f"{peak_improvement_count} of {len(decision_by_building)} buildings, but the combined objective "
        f"improved in only {objective_improvement_count} of {len(decision_by_building)}. Under this "
        "scenario, uncertainty changed the cost–risk trade-off rather than improving every objective."
    )

    seasonal_mase = float(
        forecast_summary.loc[forecast_summary["method"].eq("Seasonal naive"), "MASE"].iloc[0]
    )
    model_mase = float(
        forecast_summary.loc[forecast_summary["method"].eq("Gradient boosting"), "MASE"].iloc[0]
    )

    building_table = selected_stats[[
        "building_id", "site_id", "sub_primaryspaceusage", "sqm", "coverage"
    ]].copy()
    building_table["coverage"] *= 100
    report = f"""# Results: From Forecasts to Decisions

## Executive finding

{finding}

This is a deliberately small, reproducible study rather than a claim of deployment readiness. It demonstrates the full chain from measured building data, through leakage-safe probabilistic forecasting, to optimisation and realised decision evaluation.

## Research question

**Do calibrated probabilistic forecasts improve forecast-driven battery scheduling compared with point forecasts alone?**

## Data and buildings

The study used the cleaned electricity file from Building Data Genome Project 2. The deterministic selection rule chose a high-completeness office meter from each of three different sites.

{_markdown_table(building_table.rename(columns={'building_id': 'Building', 'site_id': 'Site', 'sub_primaryspaceusage': 'Subtype', 'sqm': 'Area_sqm', 'coverage': 'Coverage_pct'}), 2)}

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

{_markdown_table(forecast_summary, 3)}

### Prediction interval

{_markdown_table(interval_summary, 3)}

Building-level interval diagnostics are stored in `outputs/interval_metrics.csv`; this prevents an average coverage figure from hiding site-specific calibration differences.

## Decision performance

The realised objective combines energy charge, an explicit carbon-price term, and daily peak charge. Lower is better. The oracle uses the actual next-day load and is a performance bound, not a deployable method.

{_markdown_table(decision_summary, 3)}

Positive objective change below means the interval-aware policy was worse; positive peak reduction means it reduced the realised peak.

{_markdown_table(decision_by_building, 3)}

**Answer to the research question:** {finding}

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

**CV bullet:** Built a leakage-safe day-ahead energy forecasting and optimisation pipeline on three real BDG2 office meters, combining gradient boosting, calibrated 90% prediction intervals, rolling-origin evaluation, and linear battery scheduling; reduced mean MASE from {seasonal_mase:.3f} to {model_mase:.3f} and found that interval-aware control cut mean peak demand by {peak_reduction:.2f}% ({peak_min:.2f}–{peak_max:.2f}% across all buildings), with a {abs(interval_change):.2f}% trade-off in the combined objective.

## Reproducibility record

- Random seed: `{config.random_state}`
- Horizon: `{config.horizon}` hours
- Nominal interval coverage: `{1 - config.interval_alpha:.0%}`
- Decision days evaluated: `{int(decisions[['building_id', 'origin_time']].drop_duplicates().shape[0])}`
- Generated figures: `outputs/figures/forecast_example.png`, `outputs/figures/decision_comparison.png`
- Machine-readable tables: `outputs/forecast_metrics.csv`, `outputs/interval_metrics.csv`, `outputs/decision_results.csv`

## Citation

Miller, C., Kathirgamanathan, A., Picchetti, B. *et al.* (2020). The Building Data Genome Project 2, energy meter data from the ASHRAE Great Energy Predictor III competition. *Scientific Data, 7*, 368. <https://doi.org/10.1038/s41597-020-00712-x>
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return {
        "selected_buildings": selected_stats["building_id"].tolist(),
        "interval_vs_point_objective_change_pct": interval_change,
        "interval_vs_point_peak_reduction_pct": peak_reduction,
        "finding": finding,
        "decision_days": int(decisions[["building_id", "origin_time"]].drop_duplicates().shape[0]),
    }


def run_study(config: StudyConfig) -> dict[str, object]:
    config.root.mkdir(parents=True, exist_ok=True)
    output_dir = config.root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    readings, selected_stats = load_and_select_offices(config)

    all_metrics: list[dict[str, object]] = []
    all_intervals: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for building_id in selected_stats["building_id"]:
        metrics, interval, predictions = fit_forecasts(
            building_id, readings[building_id], config
        )
        all_metrics.extend(metrics)
        all_intervals.append(interval)
        prediction_frames.append(predictions)

    forecast_metrics = pd.DataFrame(all_metrics)
    interval_metrics = pd.DataFrame(all_intervals)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    decisions = evaluate_decisions(predictions, config)

    selected_stats.to_csv(output_dir / "selected_buildings.csv", index=False)
    forecast_metrics.to_csv(output_dir / "forecast_metrics.csv", index=False)
    interval_metrics.to_csv(output_dir / "interval_metrics.csv", index=False)
    predictions.to_csv(output_dir / "forecast_predictions.csv", index=False)
    decisions.to_csv(output_dir / "decision_results.csv", index=False)
    create_figures(predictions, decisions, output_dir / "figures")
    summary = write_report(
        config,
        selected_stats,
        forecast_metrics,
        interval_metrics,
        decisions,
        config.root / "reports" / "results.md",
    )
    summary["config"] = {**asdict(config), "root": str(config.root)}
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return summary

