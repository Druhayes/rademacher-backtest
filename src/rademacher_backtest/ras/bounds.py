"""Probabilistic bounds calculation for Sharpe Ratio and other metrics.

This module combines Rademacher complexity estimation with haircut
calculation to produce final bounds on performance metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from rademacher_backtest.config.settings import RASConfig
from rademacher_backtest.ras.complexity import RademacherComplexityEstimator
from rademacher_backtest.ras.haircut import HaircutCalculator, RASHaircut


@dataclass(frozen=True)
class SharpeRatioBounds:
    """Probabilistic bounds on Sharpe Ratio.

    Attributes:
        empirical_sharpe: Raw empirical Sharpe ratio (non-annualized)
        lower_bound: Conservative lower bound (non-annualized)
        haircut: Detailed haircut decomposition
        is_positive: Whether lower bound is positive
        annualized_empirical: Annualized empirical Sharpe
        annualized_lower_bound: Annualized lower bound
        rademacher_complexity: Estimated Rademacher complexity
    """

    empirical_sharpe: float
    lower_bound: float
    haircut: RASHaircut
    is_positive: bool
    annualized_empirical: float
    annualized_lower_bound: float
    rademacher_complexity: float

    @property
    def haircut_percentage(self) -> float:
        """Haircut as percentage of empirical Sharpe."""
        if abs(self.empirical_sharpe) < 1e-10:
            return 0.0
        return (self.haircut.total_haircut / abs(self.empirical_sharpe)) * 100

    @property
    def is_statistically_significant(self) -> bool:
        """Whether the strategy shows statistically significant positive returns."""
        return self.is_positive


class BoundsCalculator:
    """Calculate probabilistic bounds on performance metrics.

    Combines Rademacher complexity estimation with haircut calculation
    to produce RAS-adjusted performance bounds.
    """

    def __init__(self, config: RASConfig | None = None) -> None:
        """Initialize the calculator.

        Args:
            config: RAS configuration. Uses defaults if None.
        """
        self.config = config or RASConfig()
        self.complexity_estimator = RademacherComplexityEstimator(
            n_simulations=self.config.n_simulations,
            random_seed=self.config.random_seed,
        )
        self.haircut_calculator = HaircutCalculator(delta=self.config.delta)

    def calculate_sharpe_bounds(
        self,
        returns: NDArray[np.float64],
        annualization_factor: float | None = None,
    ) -> SharpeRatioBounds:
        """Calculate RAS-adjusted Sharpe Ratio bounds for a single strategy.

        Args:
            returns: Array of returns (e.g., daily returns)
            annualization_factor: Factor to annualize Sharpe. Default √252.

        Returns:
            SharpeRatioBounds with probabilistic guarantees

        Example:
            >>> config = RASConfig(delta=0.01)  # 99% confidence
            >>> calc = BoundsCalculator(config)
            >>> bounds = calc.calculate_sharpe_bounds(daily_returns)
            >>> print(f"Empirical: {bounds.annualized_empirical:.3f}")
            >>> print(f"Lower bound: {bounds.annualized_lower_bound:.3f}")
        """
        if annualization_factor is None:
            annualization_factor = self.config.annualization_factor

        T = len(returns)
        if T < 20:
            raise ValueError(f"Need at least 20 observations, got {T}")

        # Step 1: Calculate empirical Sharpe (non-annualized)
        mean_return = float(np.mean(returns))
        std_return = float(np.std(returns, ddof=1))

        if std_return < 1e-10:
            # Zero volatility - return trivial bounds
            return SharpeRatioBounds(
                empirical_sharpe=0.0,
                lower_bound=0.0,
                haircut=RASHaircut(0.0, 0.0, 0.0, self.config.delta, T),
                is_positive=False,
                annualized_empirical=0.0,
                annualized_lower_bound=0.0,
                rademacher_complexity=0.0,
            )

        empirical_sharpe = mean_return / std_return

        # Step 2: Standardize returns (z-score)
        z_scored = (returns - mean_return) / std_return

        # Step 3: Estimate Rademacher complexity
        X = z_scored.reshape(-1, 1)
        rademacher = self.complexity_estimator.estimate(X)

        # Step 4: Calculate haircut
        haircut = self.haircut_calculator.calculate_for_sharpe(
            rademacher_complexity=rademacher,
            T=T,
            N=1,
            sigma=1.0,  # Already standardized
        )

        # Step 5: Compute lower bound
        lower_bound = empirical_sharpe - haircut.total_haircut

        return SharpeRatioBounds(
            empirical_sharpe=empirical_sharpe,
            lower_bound=lower_bound,
            haircut=haircut,
            is_positive=lower_bound > 0,
            annualized_empirical=empirical_sharpe * annualization_factor,
            annualized_lower_bound=lower_bound * annualization_factor,
            rademacher_complexity=rademacher,
        )

    def calculate_multi_strategy_bounds(
        self,
        returns_matrix: NDArray[np.float64],
        annualization_factor: float | None = None,
    ) -> list[SharpeRatioBounds]:
        """Calculate bounds for multiple strategies simultaneously.

        The Rademacher complexity accounts for correlations between
        strategies, providing tighter bounds when strategies are
        correlated (less data-snooping risk).

        Args:
            returns_matrix: T×N matrix of returns for N strategies
            annualization_factor: Factor to annualize Sharpe

        Returns:
            List of SharpeRatioBounds, one per strategy
        """
        if annualization_factor is None:
            annualization_factor = self.config.annualization_factor

        T, N = returns_matrix.shape
        if T < 20:
            raise ValueError(f"Need at least 20 observations, got {T}")

        # Standardize each strategy
        means = returns_matrix.mean(axis=0)
        stds = returns_matrix.std(axis=0, ddof=1)
        stds = np.where(stds > 0, stds, 1.0)  # Avoid division by zero

        z_scored = (returns_matrix - means) / stds

        # Single Rademacher complexity for all strategies
        # This accounts for correlations between strategies
        rademacher = self.complexity_estimator.estimate(z_scored)

        bounds: list[SharpeRatioBounds] = []
        for n in range(N):
            mean_n = float(means[n])
            std_n = float(stds[n])

            if std_n < 1e-10:
                empirical_sharpe = 0.0
            else:
                empirical_sharpe = mean_n / std_n

            haircut = self.haircut_calculator.calculate_for_sharpe(
                rademacher_complexity=rademacher,
                T=T,
                N=N,  # Multiple testing adjustment
                sigma=1.0,
            )

            lower_bound = empirical_sharpe - haircut.total_haircut

            bounds.append(
                SharpeRatioBounds(
                    empirical_sharpe=empirical_sharpe,
                    lower_bound=lower_bound,
                    haircut=haircut,
                    is_positive=lower_bound > 0,
                    annualized_empirical=empirical_sharpe * annualization_factor,
                    annualized_lower_bound=lower_bound * annualization_factor,
                    rademacher_complexity=rademacher,
                )
            )

        return bounds

    def calculate_information_ratio_bounds(
        self,
        active_returns: NDArray[np.float64],
    ) -> SharpeRatioBounds:
        """Calculate bounds on Information Ratio.

        The Information Ratio is the Sharpe Ratio of active returns
        (portfolio return - benchmark return).

        Args:
            active_returns: Array of active returns (portfolio - benchmark)

        Returns:
            SharpeRatioBounds for the Information Ratio
        """
        # IR is mathematically equivalent to Sharpe of active returns
        return self.calculate_sharpe_bounds(active_returns)

    def sensitivity_analysis(
        self,
        returns: NDArray[np.float64],
        delta_values: list[float] | None = None,
    ) -> dict[float, SharpeRatioBounds]:
        """Perform sensitivity analysis across different confidence levels.

        Args:
            returns: Array of returns
            delta_values: List of delta values to test.
                         Default [0.01, 0.05, 0.10] for 99%, 95%, 90%

        Returns:
            Dictionary mapping delta to SharpeRatioBounds
        """
        if delta_values is None:
            delta_values = [0.01, 0.05, 0.10]

        results: dict[float, SharpeRatioBounds] = {}

        for delta in delta_values:
            config = RASConfig(
                delta=delta,
                n_simulations=self.config.n_simulations,
                random_seed=self.config.random_seed,
            )
            calc = BoundsCalculator(config)
            results[delta] = calc.calculate_sharpe_bounds(returns)

        return results
