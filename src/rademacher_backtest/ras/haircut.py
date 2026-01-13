"""RAS haircut calculation implementing Procedures 8.1 and 8.2.

The haircut represents the adjustment needed to convert empirical
performance metrics to conservative lower bounds with probabilistic
guarantees.

From "Elements of Quantitative Investing" Chapter 8:
True Performance >= Empirical Performance - Haircut
with probability >= 1 - δ
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RASHaircut:
    """RAS haircut decomposition.

    The total haircut is the sum of two terms:
    1. Data-snooping term: Accounts for testing multiple strategies
    2. Estimation term: Accounts for sampling error

    Attributes:
        data_snooping_term: 2 × Rademacher complexity
        estimation_term: Function of sample size, confidence, and N
        total_haircut: Sum of both terms
        delta: Confidence parameter (probability of bound failure)
        T: Sample size (number of time periods)
    """

    data_snooping_term: float
    estimation_term: float
    total_haircut: float
    delta: float
    T: int

    @property
    def confidence_level(self) -> float:
        """Confidence level as percentage (e.g., 99.0 for δ=0.01)."""
        return (1 - self.delta) * 100

    @property
    def annualized_haircut(self) -> float:
        """Annualized haircut (for Sharpe ratio)."""
        return self.total_haircut * np.sqrt(252)


class HaircutCalculator:
    """Calculate RAS haircut for performance metric adjustment.

    Implements Procedures 8.1 and 8.2 from Elements of Quantitative Investing.

    The haircut formula provides a lower bound on true performance:
    SR_true >= SR_empirical - haircut
    with probability >= 1 - δ
    """

    def __init__(self, delta: float = 0.01) -> None:
        """Initialize the calculator.

        Args:
            delta: Confidence parameter. Default 0.01 means 99% confidence
                   that true performance >= adjusted performance.
        """
        if not 0 < delta < 1:
            raise ValueError(f"delta must be between 0 and 1, got {delta}")
        self.delta = delta

    def calculate_for_signals(
        self,
        rademacher_complexity: float,
        T: int,
        max_ic: float = 1.0,
    ) -> RASHaircut:
        """Calculate haircut for bounded signals (IC ≤ 1).

        From Procedure 8.1:
        IC_true >= IC_empirical - 2R - max_ic × √(2×log(2/δ)/T)

        Args:
            rademacher_complexity: Estimated Rademacher complexity R_hat
            T: Sample size (number of time periods)
            max_ic: Maximum possible IC (typically 1.0)

        Returns:
            RASHaircut with decomposed terms
        """
        data_snooping = 2 * rademacher_complexity
        estimation = max_ic * np.sqrt(2 * np.log(2 / self.delta) / T)

        return RASHaircut(
            data_snooping_term=data_snooping,
            estimation_term=estimation,
            total_haircut=data_snooping + estimation,
            delta=self.delta,
            T=T,
        )

    def calculate_for_sharpe(
        self,
        rademacher_complexity: float,
        T: int,
        N: int = 1,
        sigma: float = 1.0,
    ) -> RASHaircut:
        """Calculate haircut for Sharpe Ratio (sub-Gaussian returns).

        From Procedure 8.2 and Theorem 8.4:
        SR_true >= SR_empirical - 2R - √(2×log(2N/δ)/T) - σ×√(2×log(2/δ)/T)

        For sub-Gaussian returns with proxy variance σ².

        The estimation error has two components:
        1. √(2×log(2N/δ)/T) - accounts for multiple testing
        2. σ×√(2×log(2/δ)/T) - accounts for sub-Gaussian tails

        Args:
            rademacher_complexity: Estimated Rademacher complexity R_hat
            T: Sample size (number of time periods)
            N: Number of strategies tested (default 1)
            sigma: Sub-Gaussian proxy standard deviation (default 1.0 for
                   standardized returns)

        Returns:
            RASHaircut with decomposed terms
        """
        data_snooping = 2 * rademacher_complexity

        # Estimation error for sub-Gaussian case (Equation 8.3)
        log_term_1 = np.sqrt(2 * np.log(2 * N / self.delta) / T)
        log_term_2 = sigma * np.sqrt(2 * np.log(2 / self.delta) / T)
        estimation = log_term_1 + log_term_2

        return RASHaircut(
            data_snooping_term=data_snooping,
            estimation_term=estimation,
            total_haircut=data_snooping + estimation,
            delta=self.delta,
            T=T,
        )

    def calculate_with_tuning(
        self,
        rademacher_complexity: float,
        T: int,
        N: int = 1,
        a: float = 2.0,
        b: float = 1.0,
    ) -> RASHaircut:
        """Calculate haircut with tunable parameters.

        From Section 8.3.2:
        "The bound will take the form SR >= SR_hat - a×R_hat - b×√(log/T)"

        Parameters a and b can be tuned via simulation for specific
        applications. The theoretical values (a=2, b≈1) are conservative.

        Args:
            rademacher_complexity: Estimated Rademacher complexity
            T: Sample size
            N: Number of strategies
            a: Multiplier for data-snooping term (default 2.0)
            b: Multiplier for estimation term (default 1.0)

        Returns:
            RASHaircut with tuned parameters
        """
        data_snooping = a * rademacher_complexity
        estimation = b * np.sqrt(2 * np.log(2 * N / self.delta) / T)

        return RASHaircut(
            data_snooping_term=data_snooping,
            estimation_term=estimation,
            total_haircut=data_snooping + estimation,
            delta=self.delta,
            T=T,
        )

    def calculate_minimum_sample(
        self,
        target_haircut: float,
        rademacher_complexity: float,
        N: int = 1,
    ) -> int:
        """Calculate minimum sample size for target haircut.

        Useful for determining how much data is needed to achieve
        a desired level of precision.

        Args:
            target_haircut: Desired total haircut
            rademacher_complexity: Expected Rademacher complexity
            N: Number of strategies

        Returns:
            Minimum sample size T
        """
        data_snooping = 2 * rademacher_complexity
        remaining = target_haircut - data_snooping

        if remaining <= 0:
            # Cannot achieve target with given complexity
            return int(1e9)  # Return very large number

        # Solve for T from estimation term
        # remaining = √(2×log(2N/δ)/T) + √(2×log(2/δ)/T)
        # Approximate: remaining ≈ 2×√(2×log(2N/δ)/T)
        log_factor = 2 * np.log(2 * N / self.delta)
        T = int(4 * log_factor / (remaining**2))

        return max(T, 100)  # Minimum 100 periods
