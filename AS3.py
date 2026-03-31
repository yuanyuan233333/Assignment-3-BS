# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 18:44:56 2026

@author: Torra
"""

# Importing the libraries
import os

os.chdir('C:/Users/Torra/Documents/POLIMI/FINENG/ASSIGNMENTS/Assignment_3_RM')
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from utilities.backtest import backtest
from utilities.covariance_utilities import prepare_rolling_estimation_window
from utilities.portfolio_optimization import (
    mean_variance_portfolio,
    minimum_variance_portfolio,
)
from utilities.principal_component_analysis import (
    principal_component_analysis,
    align_eigenvectors_to_previous,
    pca_denoise_covariance,
)
from utilities.shrinkage import (
    constant_corr_shrinkage,
    market_factor_shrinkage,
)

# Reading the data
# Set the path to wherever your data folder is located relative to your execution directory
data_path = Path("C:/Users/Torra/Documents/POLIMI/FINENG/ASSIGNMENTS/Assignment_3_RM/data")  

last_prices = pd.read_csv(
    data_path / "sx5e_underlyings.csv", index_col="Date", parse_dates=True
)

# Compute daily returns
performance = last_prices.pct_change().iloc[1:]

print(f"Dataset period: {performance.index[0].date()} to {performance.index[-1].date()}")
print(f"Number of observations: {len(performance)}")
print(f"Number of assets: {performance.shape[1]}")

# Parameters
time_horizon = 3 * 252  # 2 years of data for estimation
# Parameter determining the minimum coverage of returns required for an asset to be included
# in the optimization process
min_coverage = 0.95

# Get the last actual day in the data for each month (monthly rebalancing)
rebalance_dates = pd.DatetimeIndex(
    performance.groupby(pd.Grouper(freq="ME"))
    .apply(lambda x: x.index[-1] if len(x) > 0 else None)
    .dropna()
    .values
)

print(f"Number of rebalance dates: {len(rebalance_dates)}")
print(f"First rebalance: {rebalance_dates[0].date()}")
print(f"Last rebalance: {rebalance_dates[-1].date()}")

#%% Calibration of Risk Aversion

calibration_date = None
calibration_window = None

for date in rebalance_dates:
    candidate_window = prepare_rolling_estimation_window(
        returns=performance,
        rebalance_date=date,
        lookback=time_horizon,
        min_coverage=min_coverage,
    )
    if candidate_window.shape[0] == time_horizon and candidate_window.shape[1] > 0:
        calibration_date = date
        calibration_window = candidate_window
        break

if calibration_date is None:
    raise RuntimeError("No full estimation window available for portfolio checks")

calibration_sample_cov = calibration_window.cov()
calibration_expected_returns = calibration_window.mean(axis=0).values
ones = np.ones(calibration_sample_cov.shape[0])

# --- Calibrate risk aversion to target a gross exposure of ~3 ---
target_exposure = 3.0
best_gamma = 1.0
min_diff = float('inf')

# Grid search for the optimal risk aversion parameter
gammas = np.logspace(-3, 2, 1000)
for gamma in gammas:
    w = mean_variance_portfolio(
        expected_returns=calibration_expected_returns,
        cov_matrix=calibration_sample_cov.values,
        risk_aversion=gamma,
    )
    exposure = np.abs(w).sum()
    if abs(exposure - target_exposure) < min_diff:
        min_diff = abs(exposure - target_exposure)
        best_gamma = gamma

mean_variance_risk_aversion = best_gamma
print(f"Calibrated Risk Aversion (gamma): {mean_variance_risk_aversion:.4f}")

# --- Generate calibration weights ---
calibration_mvp_weights = minimum_variance_portfolio(calibration_sample_cov.values)
calibration_mv_weights = mean_variance_portfolio(
    expected_returns=calibration_expected_returns,
    cov_matrix=calibration_sample_cov.values,
    risk_aversion=mean_variance_risk_aversion,
)

calibration_mvp_variance = float(
    calibration_mvp_weights @ calibration_sample_cov.values @ calibration_mvp_weights
)

np.testing.assert_allclose(calibration_mvp_weights.sum(), 1.0)
np.testing.assert_allclose(calibration_mv_weights.sum(), 1.0)

print(
    f"Closed-form checks on the {calibration_date.date()} rebalance window "
    f"({calibration_window.shape[1]} assets)"
)
print(pd.DataFrame(
    {
        "sum_weights": [calibration_mvp_weights.sum(), calibration_mv_weights.sum()],
        "in_sample_mean_return": [
            float(calibration_expected_returns @ calibration_mvp_weights),
            float(calibration_expected_returns @ calibration_mv_weights),
        ],
        "in_sample_volatility": [
            np.sqrt(calibration_mvp_variance),
            np.sqrt(
                float(
                    calibration_mv_weights
                    @ calibration_sample_cov.values
                    @ calibration_mv_weights
                )
            ),
        ],
        "gross_exposure": [
            np.abs(calibration_mvp_weights).sum(),
            np.abs(calibration_mv_weights).sum(),
        ],
    },
    index=["Minimum Variance", "Mean-Variance"],
))

#%% Rolling Estimations & Backtest Setup

constant_corr_results = {}
mkt_factor_results = {}
sample_cov_min_var_ptfs = {}
constant_corr_cov_min_var_ptfs = {}
mkt_factor_cov_min_var_ptfs = {}
sample_cov_mean_var_ptfs = {}
constant_corr_cov_mean_var_ptfs = {}
mkt_factor_cov_mean_var_ptfs = {}
window_diagnostics = {}

for rebalance_date in rebalance_dates:
    cur_performance, cur_window_diagnostics = prepare_rolling_estimation_window(
        returns=performance,
        rebalance_date=rebalance_date,
        lookback=time_horizon,
        min_coverage=min_coverage,
        return_diagnostics=True,
    )
    if (
        cur_window_diagnostics["row_count"] < time_horizon
        or cur_performance.shape[1] == 0
    ):
        continue

    print(
        f"Processing: {rebalance_date.date()} ({cur_performance.shape[1]} assets kept)"
    )
    reb_date = rebalance_date.to_pydatetime().date()

    # Market returns as equal-weighted average (mean across assets for each day)
    market_returns = cur_performance.mean(axis=1) 
    
    # Simple historical mean for expected returns
    estimated_expected_returns = cur_performance.mean(axis=0) 

    # Compute shrinkage estimators
    cur_constant_corr_results = constant_corr_shrinkage(returns=cur_performance)
    cur_mkt_factor_results = market_factor_shrinkage(
        returns=cur_performance, market_returns=market_returns
    )

    # Compute closed-form portfolios for each covariance estimator
    sample_cov = cur_constant_corr_results["sample_cov"]
    constant_corr_cov = cur_constant_corr_results["shrunk_cov"]
    mkt_factor_cov = cur_mkt_factor_results["shrunk_cov"]

    for cov, cov_min_var_ptfs, cov_mean_var_ptfs in zip(
        [sample_cov, constant_corr_cov, mkt_factor_cov],
        [
            sample_cov_min_var_ptfs,
            constant_corr_cov_min_var_ptfs,
            mkt_factor_cov_min_var_ptfs,
        ],
        [
            sample_cov_mean_var_ptfs,
            constant_corr_cov_mean_var_ptfs,
            mkt_factor_cov_mean_var_ptfs,
        ],
    ):
        aligned_expected_returns = estimated_expected_returns.values
        cov_min_var_ptfs[reb_date] = pd.Series(
            data=minimum_variance_portfolio(cov.values), index=cov.index
        )
        cov_mean_var_ptfs[reb_date] = pd.Series(
            data=mean_variance_portfolio(
                expected_returns=aligned_expected_returns,
                cov_matrix=cov.values,
                risk_aversion=mean_variance_risk_aversion,
            ),
            index=cov.index,
        )

    constant_corr_results[reb_date] = cur_constant_corr_results
    mkt_factor_results[reb_date] = cur_mkt_factor_results
    window_diagnostics[reb_date] = cur_window_diagnostics

#%% Backtest Minimum Variance Portfolios

constant_corr_cov_min_var_ptf = pd.DataFrame(constant_corr_cov_min_var_ptfs).T.fillna(0.0)
mkt_factor_cov_min_var_ptf = pd.DataFrame(mkt_factor_cov_min_var_ptfs).T.fillna(0.0)
sample_cov_min_var_ptf = pd.DataFrame(sample_cov_min_var_ptfs).T.fillna(0.0)

# Backtest portfolios
sample_cov_min_var_ptf_pfm = backtest(
    portfolios=sample_cov_min_var_ptf,
    returns=performance,
)
constant_corr_cov_min_var_ptf_pfm = backtest(
    portfolios=constant_corr_cov_min_var_ptf,
    returns=performance,
)
mkt_factor_cov_min_var_ptf_pfm = backtest(
    portfolios=mkt_factor_cov_min_var_ptf,
    returns=performance,
)

#%% Plot rolling annualized volatility
plt.figure(figsize=(14, 5))

(
    np.sqrt(252) * sample_cov_min_var_ptf_pfm.pct_change().ewm(span=time_horizon).std()
).plot(label="Sample Covariance", linewidth=1.5)

(
    np.sqrt(252)
    * constant_corr_cov_min_var_ptf_pfm.pct_change().ewm(span=time_horizon).std()
).plot(label="Constant Correlation Shrinkage", linewidth=1.5)

(
    np.sqrt(252)
    * mkt_factor_cov_min_var_ptf_pfm.pct_change().ewm(span=time_horizon).std()
).plot(label="Market Factor Shrinkage", linewidth=1.5)

plt.xlabel("Date")
plt.ylabel("Annualized Volatility")
plt.title("Rolling Annualized Volatility of Minimum Variance Portfolios")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

#%% Plot cumulative returns
plt.figure(figsize=(14, 5))

sample_cov_min_var_ptf_pfm.plot(label="Sample Covariance", linewidth=1.5)
constant_corr_cov_min_var_ptf_pfm.plot(
    label="Constant Correlation Shrinkage", linewidth=1.5
)
mkt_factor_cov_min_var_ptf_pfm.plot(label="Market Factor Shrinkage", linewidth=1.5)

plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.title("Cumulative Returns of Minimum Variance Portfolios")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

#%% Plot turnover
plt.figure(figsize=(14, 5))

sample_cov_min_var_ptf.diff().abs().sum(axis=1).plot(
    label="Sample Covariance", linewidth=1.5
)
constant_corr_cov_min_var_ptf.diff().abs().sum(axis=1).plot(
    label="Constant Correlation Shrinkage", linewidth=1.5
)
mkt_factor_cov_min_var_ptf.diff().abs().sum(axis=1).plot(
    label="Market Factor Shrinkage", linewidth=1.5
)

plt.xlabel("Date")
plt.ylabel("Turnover")
plt.title("Turnover of Minimum Variance Portfolios")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Print turnover statistics
print("Average Turnover (Min Var):")
print(
    f"  Sample Covariance: {sample_cov_min_var_ptf.diff().abs().sum(axis=1).mean():.2f}"
)
print(
    f"  Constant Corr Shrinkage: {constant_corr_cov_min_var_ptf.diff().abs().sum(axis=1).mean():.2f}"
)
print(
    f"  Market Factor Shrinkage: {mkt_factor_cov_min_var_ptf.diff().abs().sum(axis=1).mean():.2f}"
)

#%% Plot gross exposure
plt.figure(figsize=(14, 5))

sample_cov_min_var_ptf.abs().sum(axis=1).plot(label="Sample Covariance", linewidth=1.5)
constant_corr_cov_min_var_ptf.abs().sum(axis=1).plot(
    label="Constant Correlation Shrinkage", linewidth=1.5
)
mkt_factor_cov_min_var_ptf.abs().sum(axis=1).plot(
    label="Market Factor Shrinkage", linewidth=1.5
)

plt.xlabel("Date")
plt.ylabel("Gross Exposure")
plt.title("Gross Exposure of Minimum Variance Portfolios")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Print exposure statistics
print("Average Gross Exposure (Min Var):")
print(f"  Sample Covariance: {sample_cov_min_var_ptf.abs().sum(axis=1).mean():.2f}")
print(
    f"  Constant Corr Shrinkage: {constant_corr_cov_min_var_ptf.abs().sum(axis=1).mean():.2f}"
)
print(
    f"  Market Factor Shrinkage: {mkt_factor_cov_min_var_ptf.abs().sum(axis=1).mean():.2f}"
)

#%% Backtest Mean-Variance Portfolios

constant_corr_cov_mean_var_ptf = pd.DataFrame(constant_corr_cov_mean_var_ptfs).T.fillna(0.0)
mkt_factor_cov_mean_var_ptf = pd.DataFrame(mkt_factor_cov_mean_var_ptfs).T.fillna(0.0)
sample_cov_mean_var_ptf = pd.DataFrame(sample_cov_mean_var_ptfs).T.fillna(0.0)

# Backtest portfolios
sample_cov_mean_var_ptf_pfm = backtest(
    portfolios=sample_cov_mean_var_ptf,
    returns=performance,
)
constant_corr_cov_mean_var_ptf_pfm = backtest(
    portfolios=constant_corr_cov_mean_var_ptf,
    returns=performance,
)
mkt_factor_cov_mean_var_ptf_pfm = backtest(
    portfolios=mkt_factor_cov_mean_var_ptf,
    returns=performance,
)

#%% Plot rolling annualized volatility
plt.figure(figsize=(14, 5))

(
    np.sqrt(252) * sample_cov_mean_var_ptf_pfm.pct_change().ewm(span=time_horizon).std()
).plot(label="Sample Covariance", linewidth=1.5)

(
    np.sqrt(252)
    * constant_corr_cov_mean_var_ptf_pfm.pct_change().ewm(span=time_horizon).std()
).plot(label="Constant Correlation Shrinkage", linewidth=1.5)

(
    np.sqrt(252)
    * mkt_factor_cov_mean_var_ptf_pfm.pct_change().ewm(span=time_horizon).std()
).plot(label="Market Factor Shrinkage", linewidth=1.5)

plt.xlabel("Date")
plt.ylabel("Annualized Volatility")
plt.title("Rolling Annualized Volatility of Mean-Variance Portfolios")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

#%% Plot cumulative returns
plt.figure(figsize=(14, 5))

sample_cov_mean_var_ptf_pfm.plot(label="Sample Covariance", linewidth=1.5)
constant_corr_cov_mean_var_ptf_pfm.plot(
    label="Constant Correlation Shrinkage", linewidth=1.5
)
mkt_factor_cov_mean_var_ptf_pfm.plot(label="Market Factor Shrinkage", linewidth=1.5)

plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.title("Cumulative Returns of Mean-Variance Portfolios")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

#%% Plot turnover
plt.figure(figsize=(14, 5))

sample_cov_mean_var_ptf.diff().abs().sum(axis=1).plot(
    label="Sample Covariance", linewidth=1.5
)
constant_corr_cov_mean_var_ptf.diff().abs().sum(axis=1).plot(
    label="Constant Correlation Shrinkage", linewidth=1.5
)
mkt_factor_cov_mean_var_ptf.diff().abs().sum(axis=1).plot(
    label="Market Factor Shrinkage", linewidth=1.5
)

plt.xlabel("Date")
plt.ylabel("Turnover")
plt.title("Turnover of Mean-Variance Portfolios")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Print turnover statistics
print("Average Turnover (Mean Var):")
print(
    f"  Sample Covariance: {sample_cov_mean_var_ptf.diff().abs().sum(axis=1).mean():.2f}"
)
print(
    f"  Constant Corr Shrinkage: {constant_corr_cov_mean_var_ptf.diff().abs().sum(axis=1).mean():.2f}"
)
print(
    f"  Market Factor Shrinkage: {mkt_factor_cov_mean_var_ptf.diff().abs().sum(axis=1).mean():.2f}"
)

#%% Plot gross exposure
plt.figure(figsize=(14, 5))

sample_cov_mean_var_ptf.abs().sum(axis=1).plot(label="Sample Covariance", linewidth=1.5)
constant_corr_cov_mean_var_ptf.abs().sum(axis=1).plot(
    label="Constant Correlation Shrinkage", linewidth=1.5
)
mkt_factor_cov_mean_var_ptf.abs().sum(axis=1).plot(
    label="Market Factor Shrinkage", linewidth=1.5
)

plt.xlabel("Date")
plt.ylabel("Gross Exposure")
plt.title("Gross Exposure of Mean-Variance Portfolios")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Print exposure statistics
print("Average Gross Exposure (Mean Var):")
print(f"  Sample Covariance: {sample_cov_mean_var_ptf.abs().sum(axis=1).mean():.2f}")
print(
    f"  Constant Corr Shrinkage: {constant_corr_cov_mean_var_ptf.abs().sum(axis=1).mean():.2f}"
)
print(
    f"  Market Factor Shrinkage: {mkt_factor_cov_mean_var_ptf.abs().sum(axis=1).mean():.2f}"
)

#%% Plot shrinkage intensity over time
plt.figure(figsize=(14, 5))

pd.Series(
    {date: results["intensity"] for date, results in constant_corr_results.items()}
).plot(label="Constant Correlation Shrinkage", linewidth=1.5)

pd.Series(
    {date: results["intensity"] for date, results in mkt_factor_results.items()}
).plot(label="Market Factor Shrinkage", linewidth=1.5)

plt.xlabel("Date")
plt.ylabel("Shrinkage Intensity (δ)")
plt.title("Optimal Shrinkage Intensity Over Time")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Print statistics
const_corr_intensity = pd.Series(
    {date: results["intensity"] for date, results in constant_corr_results.items()}
)
mkt_factor_intensity = pd.Series(
    {date: results["intensity"] for date, results in mkt_factor_results.items()}
)

print("Constant Correlation Shrinkage Intensity:")
print(f"  Mean: {const_corr_intensity.mean():.2%}")
print(f"  Std: {const_corr_intensity.std():.2%}")
print("\nMarket Factor Shrinkage Intensity:")
print(f"  Mean: {mkt_factor_intensity.mean():.2%}")
print(f"  Std: {mkt_factor_intensity.std():.2%}")

#%% Condition number

sample_cov_cond = {
    date: np.linalg.cond(results["sample_cov"].values)
    for date, results in constant_corr_results.items()
}
constant_corr_cond = {
    date: np.linalg.cond(results["shrunk_cov"].values)
    for date, results in constant_corr_results.items()
}
mkt_factor_cond = {
    date: np.linalg.cond(results["shrunk_cov"].values)
    for date, results in mkt_factor_results.items()
}
eligible_asset_counts = pd.Series(
    {
        date: diagnostics["asset_count_after_filter"]
        for date, diagnostics in window_diagnostics.items()
    }
)

#%% Plot condition numbers
plt.figure(figsize=(14, 5))

pd.Series(sample_cov_cond).plot(label="Sample Covariance", linewidth=1.5)
pd.Series(constant_corr_cond).plot(
    label="Constant Correlation Shrinkage", linewidth=1.5
)
pd.Series(mkt_factor_cond).plot(label="Market Factor Shrinkage", linewidth=1.5)

plt.xlabel("Date")
plt.ylabel("Condition Number")
plt.yscale("log")
plt.title("Condition Number of Covariance Matrices Over Time (log scale)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Print statistics
print("Condition Number Statistics:")
print(f"  Sample Covariance - Mean: {np.mean(list(sample_cov_cond.values())):.1f}")
print(
    f"  Constant Corr Shrinkage - Mean: {np.mean(list(constant_corr_cond.values())):.1f}"
)
print(
    f"  Market Factor Shrinkage - Mean: {np.mean(list(mkt_factor_cond.values())):.1f}"
)
print(f"  Average number of eligible assets: {eligible_asset_counts.mean():.1f}")