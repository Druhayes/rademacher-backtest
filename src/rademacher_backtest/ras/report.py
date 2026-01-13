"""RAS report generation for human-readable output.

Generates comprehensive reports explaining the RAS analysis results
in clear, interpretable format.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rademacher_backtest.ras.bounds import SharpeRatioBounds
from rademacher_backtest.ras.complexity import massart_bound


@dataclass
class RASReport:
    """Complete RAS analysis report.

    Contains all information needed to understand and interpret
    the RAS-adjusted performance metrics.

    Attributes:
        empirical_sharpe: Raw empirical Sharpe ratio (non-annualized)
        empirical_sharpe_annualized: Annualized empirical Sharpe
        adjusted_sharpe: RAS-adjusted lower bound (non-annualized)
        adjusted_sharpe_annualized: Annualized adjusted Sharpe
        rademacher_complexity: Estimated Rademacher complexity
        massart_upper_bound: Theoretical upper bound on complexity
        data_snooping_haircut: Haircut from data snooping (2×R)
        estimation_haircut: Haircut from estimation error
        total_haircut: Total haircut (non-annualized)
        total_haircut_annualized: Total annualized haircut
        confidence_level: Confidence level as percentage (e.g., 99.0)
        sample_size: Number of observations (T)
        num_strategies: Number of strategies tested (N)
        is_statistically_positive: Whether lower bound > 0
        interpretation: Human-readable interpretation
    """

    empirical_sharpe: float
    empirical_sharpe_annualized: float
    adjusted_sharpe: float
    adjusted_sharpe_annualized: float
    rademacher_complexity: float
    massart_upper_bound: float
    data_snooping_haircut: float
    estimation_haircut: float
    total_haircut: float
    total_haircut_annualized: float
    confidence_level: float
    sample_size: int
    num_strategies: int
    is_statistically_positive: bool
    interpretation: str

    def to_dict(self) -> dict:
        """Convert report to dictionary for serialization."""
        return {
            "empirical_sharpe": round(self.empirical_sharpe, 6),
            "empirical_sharpe_annualized": round(self.empirical_sharpe_annualized, 3),
            "adjusted_sharpe": round(self.adjusted_sharpe, 6),
            "adjusted_sharpe_annualized": round(self.adjusted_sharpe_annualized, 3),
            "rademacher_complexity": round(self.rademacher_complexity, 6),
            "massart_upper_bound": round(self.massart_upper_bound, 6),
            "data_snooping_haircut": round(self.data_snooping_haircut, 6),
            "estimation_haircut": round(self.estimation_haircut, 6),
            "total_haircut": round(self.total_haircut, 6),
            "total_haircut_annualized": round(self.total_haircut_annualized, 3),
            "confidence_level": round(self.confidence_level, 1),
            "sample_size": self.sample_size,
            "num_strategies": self.num_strategies,
            "is_statistically_positive": self.is_statistically_positive,
            "interpretation": self.interpretation,
        }

    def summary_lines(self) -> list[str]:
        """Generate summary lines for display."""
        lines = [
            "=" * 60,
            "RAS ANALYSIS REPORT",
            "=" * 60,
            "",
            "SHARPE RATIO ANALYSIS",
            "-" * 40,
            f"Empirical Sharpe (annualized):     {self.empirical_sharpe_annualized:>8.3f}",
            f"RAS-Adjusted Sharpe (annualized):  {self.adjusted_sharpe_annualized:>8.3f}",
            f"Total Haircut (annualized):        {self.total_haircut_annualized:>8.3f}",
            "",
            "HAIRCUT DECOMPOSITION",
            "-" * 40,
            f"Data-Snooping Term (2×R):          {self.data_snooping_haircut:>8.6f}",
            f"Estimation Error Term:             {self.estimation_haircut:>8.6f}",
            f"Total Haircut (daily):             {self.total_haircut:>8.6f}",
            "",
            "TECHNICAL DETAILS",
            "-" * 40,
            f"Rademacher Complexity (R̂):         {self.rademacher_complexity:>8.6f}",
            f"Massart Upper Bound:               {self.massart_upper_bound:>8.6f}",
            f"Sample Size (T):                   {self.sample_size:>8d}",
            f"Strategies Tested (N):             {self.num_strategies:>8d}",
            f"Confidence Level:                  {self.confidence_level:>7.1f}%",
            "",
            "INTERPRETATION",
            "-" * 40,
            self.interpretation,
            "",
            "=" * 60,
        ]
        return lines

    def __str__(self) -> str:
        """Return formatted string representation."""
        return "\n".join(self.summary_lines())


class RASReportGenerator:
    """Generate human-readable RAS reports."""

    def generate(
        self,
        bounds: SharpeRatioBounds,
        T: int,
        N: int = 1,
    ) -> RASReport:
        """Generate comprehensive RAS report from bounds.

        Args:
            bounds: Calculated Sharpe ratio bounds
            T: Sample size (number of time periods)
            N: Number of strategies tested

        Returns:
            Complete RASReport with interpretation
        """
        interpretation = self._generate_interpretation(bounds)
        massart = massart_bound(N, T)

        return RASReport(
            empirical_sharpe=bounds.empirical_sharpe,
            empirical_sharpe_annualized=bounds.annualized_empirical,
            adjusted_sharpe=bounds.lower_bound,
            adjusted_sharpe_annualized=bounds.annualized_lower_bound,
            rademacher_complexity=bounds.rademacher_complexity,
            massart_upper_bound=massart,
            data_snooping_haircut=bounds.haircut.data_snooping_term,
            estimation_haircut=bounds.haircut.estimation_term,
            total_haircut=bounds.haircut.total_haircut,
            total_haircut_annualized=bounds.haircut.total_haircut * np.sqrt(252),
            confidence_level=bounds.haircut.confidence_level,
            sample_size=T,
            num_strategies=N,
            is_statistically_positive=bounds.is_positive,
            interpretation=interpretation,
        )

    def _generate_interpretation(self, bounds: SharpeRatioBounds) -> str:
        """Generate natural language interpretation of results.

        Args:
            bounds: Sharpe ratio bounds

        Returns:
            Human-readable interpretation string
        """
        conf = bounds.haircut.confidence_level

        if bounds.is_positive:
            quality = self._assess_sharpe_quality(bounds.annualized_lower_bound)
            return (
                f"With {conf:.0f}% confidence, the true Sharpe Ratio is at least "
                f"{bounds.annualized_lower_bound:.3f} (annualized). "
                f"The strategy shows statistically significant positive "
                f"risk-adjusted returns. {quality}"
            )
        else:
            if bounds.annualized_empirical > 0:
                return (
                    f"With {conf:.0f}% confidence, we cannot rule out that the "
                    f"true Sharpe Ratio is zero or negative. The empirical "
                    f"Sharpe of {bounds.annualized_empirical:.3f} may be due to "
                    f"sampling variation or data snooping. The strategy does "
                    f"not demonstrate statistically significant positive "
                    f"risk-adjusted returns at this confidence level."
                )
            else:
                return (
                    f"The empirical Sharpe Ratio of {bounds.annualized_empirical:.3f} "
                    f"is negative, indicating the strategy underperformed on a "
                    f"risk-adjusted basis. With {conf:.0f}% confidence, the true "
                    f"Sharpe is at least {bounds.annualized_lower_bound:.3f}."
                )

    def _assess_sharpe_quality(self, sharpe: float) -> str:
        """Assess the quality of a Sharpe ratio.

        Args:
            sharpe: Annualized Sharpe ratio

        Returns:
            Quality assessment string
        """
        if sharpe >= 1.5:
            return "This represents excellent risk-adjusted performance."
        elif sharpe >= 1.0:
            return "This represents very good risk-adjusted performance."
        elif sharpe >= 0.5:
            return "This represents solid risk-adjusted performance."
        elif sharpe >= 0.25:
            return "This represents modest but positive risk-adjusted performance."
        else:
            return "This represents marginally positive risk-adjusted performance."

    def generate_comparison_report(
        self,
        bounds_list: list[SharpeRatioBounds],
        strategy_names: list[str],
        T: int,
    ) -> str:
        """Generate comparison report for multiple strategies.

        Args:
            bounds_list: List of bounds for each strategy
            strategy_names: Names for each strategy
            T: Sample size

        Returns:
            Formatted comparison report string
        """
        lines = [
            "=" * 70,
            "MULTI-STRATEGY RAS COMPARISON",
            "=" * 70,
            "",
            f"{'Strategy':<25} {'Empirical SR':>12} {'Adjusted SR':>12} {'Significant':>12}",
            "-" * 70,
        ]

        for name, bounds in zip(strategy_names, bounds_list, strict=True):
            sig = "Yes" if bounds.is_positive else "No"
            lines.append(
                f"{name:<25} {bounds.annualized_empirical:>12.3f} "
                f"{bounds.annualized_lower_bound:>12.3f} {sig:>12}"
            )

        lines.extend([
            "-" * 70,
            "",
            f"Sample size: {T} observations",
            f"Strategies compared: {len(bounds_list)}",
            f"Confidence level: {bounds_list[0].haircut.confidence_level:.0f}%",
            "",
            "Note: Rademacher complexity accounts for correlations between strategies.",
            "Testing correlated strategies incurs less data-snooping penalty.",
            "",
            "=" * 70,
        ])

        return "\n".join(lines)
