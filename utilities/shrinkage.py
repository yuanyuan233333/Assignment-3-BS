from typing import Any, Dict

import numpy as np
import pandas as pd


def constant_corr_shrinkage(
    returns: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Calculate the shrinkage target, the optimal shrinkage intensity and the shrunk covariance
    matrix for the constant correlation shrinkage. No check is done on the input data.
    Implementation follows the Ledoit-Wolf (2003) paper "Honey, I Shrunk the Sample Covariance
    Matrix".

    Parameters:
        returns (pd.DataFrame): The returns matrix (T x N) where T is time periods and N is assets

    Returns:
        Dict[str, Any]: A dictionary containing the shrinkage target matrix ("target"), the optimal
            shrinkage intensity ("intensity"), the sample covariance matrix ("sample_cov") and the
            shrunk covariance matrix ("shrunk_cov").
    """

    cov_matrix = returns.cov()

    T, N = returns.shape  # T = time periods, N = number of assets

    # Convert to numpy arrays for efficiency
    returns_np = returns.values
    S = cov_matrix.values  # Sample covariance matrix

    # Extract standard deviations
    variances = np.diag(S)
    std_devs = np.sqrt(variances) #ERROR was variances, but we need standard deviations

    # Compute correlation matrix from covariance matrix
    std_outer = np.outer(std_devs, std_devs)
    corr_matrix = S / std_outer

    # Calculate average correlation
    avg_corr = (np.sum(corr_matrix) - N) / (N * (N - 1))  

    ## Target
    # Calculate target matrix (constant correlation)
    constant_corr_cov = avg_corr * std_outer  

    # Set diagonal elements to original variances
    np.fill_diagonal(constant_corr_cov, variances)

    target = pd.DataFrame(
        constant_corr_cov, index=cov_matrix.index, columns=cov_matrix.columns
    )

    ## Intensity
    sample_means = np.mean(returns_np, axis=0)

    # Center the returns: shape (T, N)
    centered_returns = returns_np - sample_means

    # Pre-compute useful quantities
    sqrt_ratio_matrix = np.outer(std_devs, 1 / std_devs)   # √(σᵢᵢ/σⱼⱼ), shape (N, N)
    centered_squares = centered_returns ** 2                 # y_t[i]², shape (T, N)

    # Deviation of outer products from sample covariance: y_t[i]*y_t[j] - S[i,j], shape (T, N, N)
    outer_diff = centered_returns[:, :, None] * centered_returns[:, None, :] - S[None, :, :]

    # Pi-hat: (1/T) * Σₜ ||y_t*y_tᵀ - S||²
    pi_hat = np.sum(outer_diff ** 2) / T

    # Rho-hat diagonal terms: (1/T) * Σₜ Σᵢ (y_t[i]² - S[i,i])²
    diag_terms = centered_squares - variances[None, :]       # (T, N)
    diag_rho = np.sum(diag_terms ** 2)

    # Rho-hat off-diagonal terms using einsum
    # For each (i,j) off-diagonal:
    # contribution = avg_corr * [√(σᵢᵢ/σⱼⱼ) * ϑ̂ᵢᵢ,ᵢⱼ + √(σⱼⱼ/σᵢᵢ) * ϑ̂ⱼⱼ,ᵢⱼ]
    # where ϑ̂ᵢᵢ,ᵢⱼ = Σₜ (y_t[i]² - S[i,i]) * (y_t[i]*y_t[j] - S[i,j])
    off_diag_rho = avg_corr * (
    np.einsum("ij,ti,tij->", sqrt_ratio_matrix.T, diag_terms, outer_diff) +
    np.einsum("ij,tj,tij->", sqrt_ratio_matrix,   diag_terms, outer_diff)
     )

    # Remove diagonal contributions (i==j) that einsum included
    # When i==j: sqrt_ratio[i,i]=1 and diag_terms[:,i]*outer_diff[:,i,i] = (y²-σ)*(y²-σ)
    # So diagonal contribution of both einsum = 2 * diag_rho
    off_diag_rho -= 2 * avg_corr * diag_rho

    # Combine diagonal and off-diagonal rho-hat
    rho_hat = (diag_rho + off_diag_rho) / T

    # Gamma-hat: ||F - S||² (Frobenius norm squared of target minus sample covariance)
    gamma_hat = np.sum((target.values - S) ** 2)

    # Optimal shrinkage intensity: δ* = κ/T = (π̂ - ρ̂) / (T * γ̂)
    # FIX: denominator must include factor T (was missing in original code)
    numerator = pi_hat - rho_hat
    denominator = T * gamma_hat

    if denominator == 0:
     intensity = 0.0
    else:
     intensity = max(0.0, min(1.0, numerator / denominator))
    return {
        "target": target,
        "intensity": intensity,
        "sample_cov": cov_matrix,
        "shrunk_cov": intensity*target+(1-intensity)*cov_matrix,
    }


def market_factor_shrinkage(
    returns: pd.DataFrame, market_returns: pd.Series
) -> Dict[str, Any]:
    """
    Calculate the shrinkage target, the optimal shrinkage intensity and the shrunk covariance
    matrix for the market factor shrinkage. No check is done on the input data.
    Implementation follows the Ledoit-Wolf (2002) paper "Improved estimation of the covariance
    matrix of stock returns with an application to portfolio selection".

    Parameters:
        returns (pd.DataFrame): The returns matrix (T x N) where T is time periods and N is assets
        market_returns (pd.Series): The market returns (T x 1)

    Returns:
        Dict[str, Any]: A dictionary containing the shrinkage target matrix ("target"), the optimal
            shrinkage intensity ("intensity"), the sample covariance matrix ("sample_cov") and the
            shrunk covariance matrix ("shrunk_cov").
    """

    # Align indices to ensure proper calculation
    aligned_data = pd.concat([returns, market_returns], axis=1, join="inner")
    returns_aligned = aligned_data.iloc[:, :-1]
    market_aligned = aligned_data.iloc[:, -1]

    T = returns_aligned.shape[0]

    ## Target
    # Calculate market variance
    market_variance = market_aligned.var() #fix before std

    # Calculate betas for all assets (vectorized)
    returns_np = returns_aligned.values
    market_np = market_aligned.values

    # Compute covariances between all assets and market at once
    # Stack returns and market, compute covariance matrix, extract asset-market covariances
    combined = np.column_stack([returns_np, market_np])
    cov_matrix_full = np.cov(combined.T)
    cov_with_market = cov_matrix_full[:-1, -1]  # Covariances of each asset with market
    betas = cov_with_market / market_variance

    # Calculate residual variances: Var(asset) - β² * Var(market) (vectorized)
    asset_variances = np.diag(cov_matrix_full[:-1, :-1])
    residual_variances = asset_variances - betas ** 2 * market_variance  # shape (N,)


    # Ensure residual variances are positive (vectorized)
    residual_variances = np.maximum(residual_variances, 1e-8)

    # Construct target matrix
    betas_outer = np.outer(betas, betas)
    residual_matrix = np.diag(residual_variances)

    # Final target matrix
    target = pd.DataFrame(
        market_variance * betas_outer + residual_matrix,  # σ²_mkt * β*βᵀ + Δ
        index=returns.columns,
        columns=returns.columns,
    )

    ## Intensity
    cov_matrix = returns_aligned.cov()
    S = cov_matrix.values

    # Center the returns
    sample_means = np.mean(returns_np, axis=0)
    centered_returns = returns_np - sample_means
    market_centered_returns = market_np - market_np.mean()

    # Calculate pi-hat and rho-hat
    # Pre-compute matrices for efficiency
    variances = np.diag(S)
    target_np = target.values

    pi_hat = 0.0
    rho_hat = 0.0

    for t in range(T):
        y_t = centered_returns[t, :]
        outer_prod = np.outer(y_t, y_t)
        diff = outer_prod - S #FIX before -

        # Pi-hat calculation
        pi_hat += np.sum(diff**2) # FIX: was = instead of +=

        # Rho-hat calculation
        m_t = market_centered_returns[t]

        # Diagonal terms: (y_t[i]**2 - S[i,i])**2
        diag_terms = y_t**2 - variances
        rho_hat += np.sum(diag_terms**2)

        # Off-diagonal terms
        # Create matrices: betas[j] * y_t[i] for all i,j
        betas_y_matrix = np.outer(y_t, betas)  # y_t[i] * betas[j]
        betas_y_matrix_T = betas_y_matrix.T  # y_t[j] * betas[i]

        # Calculate the rho-hat contribution for off-diagonal elements
        off_diag_term = (
            betas_y_matrix_T + betas_y_matrix - betas_outer * m_t
        ) * m_t * outer_prod - target_np * S

        # Sum only off-diagonal elements
        np.fill_diagonal(off_diag_term, 0)
        rho_hat += np.sum(off_diag_term)

    # FIX: both divisions by T must be outside the loop
    pi_hat /= T
    rho_hat /= T

    # Calculate gamma-hat: ||F - S||^2 where F is the target matrix
    gamma_hat = np.sum((target_np - S) ** 2)

    # Calculate optimal shrinkage intensity
    # k = (pi - rho) / gamma, but we need to be careful about numerical stability
    numerator = pi_hat - rho_hat
    denominator = gamma_hat

    if denominator == 0:
        intensity = 0.0
    else:
        intensity = max(0.0, min(1.0, numerator / (T * denominator)))

    return {
        "target": target,
        "intensity": intensity,
        "sample_cov": cov_matrix,
        "shrunk_cov": intensity * cov_matrix + (1 - intensity) * target,
    }
