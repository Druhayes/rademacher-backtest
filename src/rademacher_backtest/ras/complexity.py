"""Rademacher complexity estimation via Monte Carlo simulation.

Implements the core Rademacher complexity calculation from Section 8.3
of "Elements of Quantitative Investing".

Rademacher complexity measures the richness of a function class and is
used to bound the gap between empirical and true performance.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class RademacherComplexityEstimator:
    """Estimate Rademacher complexity via Monte Carlo simulation.

    Based on Section 8.3.1 of Elements of Quantitative Investing:

    R(X) = E_ε[sup_n (1/T) * Σ_t ε_t * x_{t,n}]

    where ε_t are iid Rademacher random variables (+1 or -1 with p=0.5)
    and X is a T×N matrix of standardized returns.

    The Rademacher complexity captures how well random noise can be fit
    by the best strategy in our strategy set. A higher complexity means
    more data-snooping risk.
    """

    def __init__(
        self,
        n_simulations: int = 20_000,
        random_seed: int | None = 42,
    ) -> None:
        """Initialize the estimator.

        Args:
            n_simulations: Number of Monte Carlo simulations for estimation.
                          More simulations = more accurate estimate.
            random_seed: Seed for reproducibility. None for random.
        """
        self.n_simulations = n_simulations
        self.rng = np.random.default_rng(random_seed)

    def estimate(self, X: NDArray[np.float64]) -> float:
        """Estimate Rademacher complexity of matrix X.

        Args:
            X: T×N matrix of standardized returns (z-scored)
               T = number of time periods
               N = number of strategies (for single portfolio, N=1)

        Returns:
            Estimated Rademacher complexity R_hat(X)

        Notes:
            For a single strategy (N=1), the complexity is approximately
            E[|Z|]/√T ≈ 0.8/√T for standard normal returns.

            For multiple strategies, complexity increases with N but
            decreases if strategies are correlated.
        """
        T, N = X.shape

        complexities: list[float] = []
        for _ in range(self.n_simulations):
            # Generate Rademacher vector: +1 or -1 with equal probability
            epsilon = self.rng.choice([-1.0, 1.0], size=T)

            # Compute (1/T) * ε^T @ X for each column, take supremum
            # This is the correlation of random signs with each strategy
            correlations = (epsilon @ X) / T  # Shape: (N,)
            sup_correlation = np.max(correlations)
            complexities.append(sup_correlation)

        # Return expected value (mean of simulations)
        return float(np.mean(complexities))

    def estimate_with_confidence(
        self,
        X: NDArray[np.float64],
        confidence: float = 0.95,
    ) -> tuple[float, float, float]:
        """Estimate Rademacher complexity with confidence interval.

        Args:
            X: T×N matrix of standardized returns
            confidence: Confidence level for interval (default 95%)

        Returns:
            Tuple of (mean, lower_bound, upper_bound)
        """
        T, N = X.shape
        complexities: list[float] = []

        for _ in range(self.n_simulations):
            epsilon = self.rng.choice([-1.0, 1.0], size=T)
            correlations = (epsilon @ X) / T
            complexities.append(np.max(correlations))

        complexities_arr = np.array(complexities)
        mean = float(np.mean(complexities_arr))
        std = float(np.std(complexities_arr, ddof=1))

        # Confidence interval using central limit theorem
        z_score = 1.96 if confidence == 0.95 else 2.576 if confidence == 0.99 else 1.645
        margin = z_score * std / np.sqrt(self.n_simulations)

        return mean, mean - margin, mean + margin

    def estimate_for_returns(
        self,
        returns: NDArray[np.float64],
    ) -> float:
        """Estimate complexity for a returns vector (single strategy).

        Convenience method that handles standardization.

        Args:
            returns: 1D array of returns (T,)

        Returns:
            Estimated Rademacher complexity
        """
        # Standardize returns
        mean = np.mean(returns)
        std = np.std(returns, ddof=1)
        if std > 0:
            z_scored = (returns - mean) / std
        else:
            z_scored = returns - mean

        # Reshape to T×1 matrix
        X = z_scored.reshape(-1, 1)
        return self.estimate(X)

    def estimate_for_strategies(
        self,
        returns_matrix: NDArray[np.float64],
    ) -> float:
        """Estimate complexity for multiple strategies.

        Args:
            returns_matrix: T×N matrix of returns for N strategies

        Returns:
            Estimated Rademacher complexity

        Notes:
            The complexity for correlated strategies is lower than for
            uncorrelated strategies. This reflects that testing similar
            strategies introduces less data-snooping risk.
        """
        T, N = returns_matrix.shape

        # Standardize each strategy column
        means = returns_matrix.mean(axis=0)
        stds = returns_matrix.std(axis=0, ddof=1)
        stds = np.where(stds > 0, stds, 1.0)  # Avoid division by zero

        z_scored = (returns_matrix - means) / stds
        return self.estimate(z_scored)


def massart_bound(N: int, T: int) -> float:
    """Compute Massart's theoretical upper bound for Rademacher complexity.

    From Section 8.3.2:
    R ≤ √(2 * log(N) / T)

    This provides a theoretical upper bound that is typically 15-20%
    higher than the empirical Monte Carlo estimate.

    Args:
        N: Number of strategies (columns in the return matrix)
        T: Number of time periods (rows in the return matrix)

    Returns:
        Upper bound on Rademacher complexity

    Notes:
        - For N=1, the bound is 0 (trivially tight)
        - For practical use, the Monte Carlo estimate is preferred
        - The bound assumes uncorrelated strategies; actual complexity
          is lower when strategies are correlated
    """
    if N <= 1:
        return 0.0
    return float(np.sqrt(2 * np.log(N) / T))


def theoretical_complexity_single(T: int) -> float:
    """Theoretical Rademacher complexity for a single strategy.

    For a single strategy with standardized returns, the expected
    Rademacher complexity is approximately:

    R ≈ E[|Z|] / √T ≈ √(2/π) / √T ≈ 0.798 / √T

    Args:
        T: Number of time periods

    Returns:
        Theoretical complexity for single strategy
    """
    return float(np.sqrt(2 / np.pi) / np.sqrt(T))
