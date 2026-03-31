"""
Portfolio optimization utilities.

Implements:
- Minimum variance portfolio (closed-form solution)
- Mean-variance portfolio (closed-form solution)

References:
- Markowitz, H. (1952). "Portfolio Selection." The Journal of Finance.
"""

import numpy as np
from utilities.covariance_utilities import (
    _validate_covariance_matrix,
)


def minimum_variance_portfolio(cov_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate the minimum variance portfolio weights given a covariance matrix.
    In particular the weights are given by:
    w = (Σ * 1) / (1^T * Σ * 1), i.e. the solution of the optimization problem:
    min_w w^T * Σ * w, subject to 1^T * w = 1.

    Parameters:
        cov_matrix (np.ndarray): Covariance matrix of asset returns.

    Returns:
        np.ndarray: Weights of the minimum variance portfolio.
    """
    cov_matrix = _validate_covariance_matrix(
        cov_matrix,
        name="cov_matrix",
        require_positive_definite=True,
        positive_definite_message=(
            "cov_matrix must be positive definite (symmetric with positive eigenvalues)"
        ),
    )

    n = cov_matrix.shape[0]
    
    # We solve the linear system Σ * x = 1, which gives us x = Σ^{-1} * 1
    # This is numerically much more stable than doing np.linalg.inv(cov_matrix) @ ones_vec
    min_var_ptf_numerator = np.linalg.solve(cov_matrix, np.ones(n))

    # The denominator 1^T * Σ^{-1} * 1 is simply the sum of the elements in our numerator
    # We normalize the unscaled weights so they sum to exactly 1
    min_var_ptf_weights = min_var_ptf_numerator / np.sum(min_var_ptf_numerator)

    return min_var_ptf_weights.flatten()


def mean_variance_portfolio(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_aversion: float = 1.0,
) -> np.ndarray:
    """
    Calculate the classic mean-variance portfolio weights given expected returns and a
    covariance matrix.

    In particular the weights solve:
    max_w mu^T * w - (gamma / 2) * w^T * Sigma * w, subject to 1^T * w = 1,
    where mu are the expected returns and gamma is the risk-aversion parameter.

    Parameters:
        expected_returns (np.ndarray): Expected returns vector.
        cov_matrix (np.ndarray): Covariance matrix of asset returns.
        risk_aversion (float): Risk-aversion parameter gamma. Must be strictly positive.

    Returns:
        np.ndarray: Weights of the mean-variance portfolio.
    """
    cov_matrix = _validate_covariance_matrix(
        cov_matrix,
        name="cov_matrix",
        require_positive_definite=True,
        positive_definite_message=(
            "cov_matrix must be positive definite (symmetric with positive eigenvalues)"
        ),
    )

    expected_returns = np.asarray(expected_returns, dtype=float)
    if expected_returns.ndim == 2 and 1 in expected_returns.shape:
        expected_returns = expected_returns.reshape(-1)
    elif expected_returns.ndim != 1:
        raise ValueError(
            "expected_returns must be one-dimensional or a single-column vector"
        )

    if expected_returns.shape[0] != cov_matrix.shape[0]:
        raise ValueError(
            "expected_returns and cov_matrix must refer to the same number of assets, "
            f"got {expected_returns.shape[0]} and {cov_matrix.shape[0]}"
        )

    if not np.isfinite(expected_returns).all():
        raise ValueError("expected_returns contains NaN or Inf values")

    if not np.isfinite(risk_aversion):
        raise ValueError("risk_aversion must be finite")

    if risk_aversion <= 0:
        raise ValueError(
            f"risk_aversion must be strictly positive, got {risk_aversion}"
        )

    n = cov_matrix.shape[0]
    ones_vec = np.ones(n)

    # Step 1: Solve for z1 = Σ^{-1} * 1 and z2 = Σ^{-1} * μ
    z1 = np.linalg.solve(cov_matrix, ones_vec)
    z2 = np.linalg.solve(cov_matrix, expected_returns)

    # Step 2: Calculate intermediate scalars
    A = np.sum(z1)  # Represents 1^T * Σ^{-1} * 1
    B = np.sum(z2)  # Represents 1^T * Σ^{-1} * μ

    # Step 3: Calculate the Lagrange multiplier (λ) for the budget constraint
    # Derived from setting 1^T * w = 1
    lambda_val = (risk_aversion - B) / A

    # Step 4: Calculate final weights: w = (1 / γ) * Σ^{-1} * (μ + λ * 1)
    mean_var_ptf_weights = (z2 + lambda_val * z1) / risk_aversion

    return mean_var_ptf_weights.flatten()

