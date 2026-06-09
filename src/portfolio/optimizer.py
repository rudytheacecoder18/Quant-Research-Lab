"""
PORTFOLIO OPTIMIZER
====================
Uses mathematical optimization (SciPy) to find the theoretically
EXACT optimal portfolio weights — not just the best among random guesses.

Monte Carlo finds approximate answers by guessing.
Optimization finds the EXACT answer by solving mathematically.

Two portfolios we solve for:
  1. Maximum Sharpe Portfolio — best risk-adjusted return
  2. Minimum Variance Portfolio — lowest possible risk

How optimization works (conceptually):
  - We define an objective function (e.g. "negative Sharpe Ratio")
  - We tell SciPy: "minimise this function"
  - SciPy adjusts the weights to find the minimum
  - Constraints: weights sum to 1, all weights >= 0

This is called "constrained optimisation" and is used everywhere in finance.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from src.analytics.returns import daily_returns


def _portfolio_stats(weights: np.ndarray, mean_returns: pd.Series, cov_matrix: pd.DataFrame, rf: float):
    """
    Helper: compute portfolio return, volatility, and Sharpe for a given weight vector.
    Used inside the optimizer as the function to minimise.
    """
    ret  = np.dot(weights, mean_returns)                # Weighted return
    var  = weights @ cov_matrix.values @ weights        # Variance (matrix form)
    vol  = np.sqrt(var)                                 # Standard deviation
    sharpe = (ret - rf) / vol
    return ret, vol, sharpe


def _neg_sharpe(weights, mean_returns, cov_matrix, rf):
    """Objective: negative Sharpe (we minimise this to maximise Sharpe)."""
    _, _, sharpe = _portfolio_stats(weights, mean_returns, cov_matrix, rf)
    return -sharpe


def _portfolio_variance(weights, mean_returns, cov_matrix, rf):
    """Objective: portfolio variance (minimise this for Min Variance portfolio)."""
    return weights @ cov_matrix.values @ weights


def optimize(
    prices          : pd.DataFrame,
    objective       : str   = "max_sharpe",   # or "min_variance"
    risk_free_rate  : float = 0.06,
    allow_short     : bool  = False,
) -> dict:
    """
    Find optimal portfolio weights using SciPy constrained optimization.

    Parameters
    ----------
    prices         : Price DataFrame
    objective      : "max_sharpe" or "min_variance"
    risk_free_rate : Annual risk-free rate
    allow_short    : If False (default), weights constrained to [0, 1]

    Returns
    -------
    dict with:
      - weights    : {ticker: optimal_weight}
      - Return     : Expected annual return
      - Volatility : Expected annual volatility
      - Sharpe     : Sharpe Ratio
      - success    : Whether optimization converged
    """
    tickers      = list(prices.columns)
    n            = len(tickers)
    daily_rets   = daily_returns(prices)
    mean_returns = daily_rets.mean() * 252
    cov_matrix   = daily_rets.cov() * 252

    # Starting guess: equal weights
    x0 = np.array([1.0 / n] * n)

    # Constraint: weights must sum to 1
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    # Bounds: each weight between 0 and 1 (no short-selling)
    bounds = tuple((0, 1) for _ in range(n)) if not allow_short else tuple((-1, 1) for _ in range(n))

    # Choose objective function
    obj_fn = _neg_sharpe if objective == "max_sharpe" else _portfolio_variance

    # Run the optimizer
    result = minimize(
        fun         = obj_fn,
        x0          = x0,
        args        = (mean_returns, cov_matrix, risk_free_rate),
        method      = "SLSQP",       # Sequential Least Squares Programming
        bounds      = bounds,
        constraints = constraints,
        options     = {"maxiter": 1000, "ftol": 1e-9},
    )

    if not result.success:
        print(f"⚠️  Optimizer did not converge: {result.message}")

    opt_weights = result.x
    ret, vol, sharpe = _portfolio_stats(opt_weights, mean_returns, cov_matrix, risk_free_rate)

    return {
        "weights"    : {t: round(float(w), 6) for t, w in zip(tickers, opt_weights)},
        "Return"     : round(float(ret),    4),
        "Volatility" : round(float(vol),    4),
        "Sharpe"     : round(float(sharpe), 4),
        "success"    : result.success,
        "objective"  : objective,
    }


def efficient_frontier_curve(
    prices         : pd.DataFrame,
    n_points       : int   = 50,
    risk_free_rate : float = 0.06,
) -> pd.DataFrame:
    """
    Compute the Efficient Frontier by solving for the minimum variance
    portfolio at each target return level.

    The Efficient Frontier is the set of portfolios that:
      - For a given volatility, maximise return
      - For a given return, minimise volatility

    Everything below/inside this curve is sub-optimal.

    Parameters
    ----------
    n_points : Number of points on the curve (more = smoother line)

    Returns
    -------
    DataFrame with columns: Return, Volatility
    """
    tickers      = list(prices.columns)
    n            = len(tickers)
    daily_rets   = daily_returns(prices)
    mean_returns = daily_rets.mean() * 252
    cov_matrix   = daily_rets.cov() * 252

    # Find return range: from Min Variance to Max possible
    min_var_result = optimize(prices, "min_variance", risk_free_rate)
    max_ret_result = optimize(prices, "max_sharpe",   risk_free_rate)

    # Clamp target range to realistic values
    ret_min = min_var_result["Return"]
    ret_max = max_ret_result["Return"] * 1.1   # Slightly above Max Sharpe

    target_returns = np.linspace(ret_min, ret_max, n_points)

    frontier_points = []

    for target in target_returns:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, t=target: np.dot(w, mean_returns) - t},
        ]
        bounds = tuple((0, 1) for _ in range(n))
        x0     = np.array([1.0 / n] * n)

        result = minimize(
            fun         = _portfolio_variance,
            x0          = x0,
            args        = (mean_returns, cov_matrix, risk_free_rate),
            method      = "SLSQP",
            bounds      = bounds,
            constraints = constraints,
            options     = {"maxiter": 500},
        )

        if result.success:
            vol = np.sqrt(result.fun)
            frontier_points.append({"Return": target, "Volatility": vol})

    return pd.DataFrame(frontier_points)
