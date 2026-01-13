"""Risk analytics and drawdown analysis.

Provides detailed analysis of portfolio risk characteristics
including drawdown analysis and risk decomposition.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd


@dataclass
class DrawdownPeriod:
    """Information about a single drawdown period.

    Attributes:
        start_date: Date drawdown began
        trough_date: Date of maximum drawdown
        end_date: Date drawdown recovered (None if still underwater)
        max_drawdown: Maximum drawdown during period (negative)
        duration_days: Total days from start to recovery
        recovery_days: Days from trough to recovery
    """

    start_date: date
    trough_date: date
    end_date: date | None
    max_drawdown: float
    duration_days: int
    recovery_days: int | None


@dataclass
class DrawdownAnalysis:
    """Complete drawdown analysis results.

    Attributes:
        drawdown_series: Full drawdown time series
        max_drawdown: Maximum drawdown value
        current_drawdown: Current drawdown (last value)
        is_underwater: Whether currently in drawdown
        worst_periods: List of worst drawdown periods
        avg_drawdown: Average drawdown when underwater
        avg_recovery_days: Average recovery time
    """

    drawdown_series: pd.Series
    max_drawdown: float
    current_drawdown: float
    is_underwater: bool
    worst_periods: list[DrawdownPeriod]
    avg_drawdown: float
    avg_recovery_days: float | None


class RiskAnalyzer:
    """Analyze portfolio risk characteristics.

    Provides detailed risk metrics including drawdown analysis,
    volatility decomposition, and tail risk measures.
    """

    def analyze_drawdowns(
        self,
        returns: pd.Series,
        n_worst: int = 5,
    ) -> DrawdownAnalysis:
        """Perform comprehensive drawdown analysis.

        Args:
            returns: Series of returns
            n_worst: Number of worst drawdown periods to identify

        Returns:
            DrawdownAnalysis with complete results
        """
        # Calculate drawdown series
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = cumulative / rolling_max - 1

        max_dd = float(drawdowns.min())
        current_dd = float(drawdowns.iloc[-1])
        is_underwater = current_dd < 0

        # Identify drawdown periods
        periods = self._identify_drawdown_periods(cumulative, drawdowns)

        # Sort by severity and take worst N
        periods.sort(key=lambda p: p.max_drawdown)
        worst_periods = periods[:n_worst]

        # Calculate averages
        underwater = drawdowns[drawdowns < 0]
        avg_dd = float(underwater.mean()) if len(underwater) > 0 else 0.0

        recovery_days = [
            p.recovery_days for p in periods if p.recovery_days is not None
        ]
        avg_recovery = float(np.mean(recovery_days)) if recovery_days else None

        return DrawdownAnalysis(
            drawdown_series=drawdowns,
            max_drawdown=max_dd,
            current_drawdown=current_dd,
            is_underwater=is_underwater,
            worst_periods=worst_periods,
            avg_drawdown=avg_dd,
            avg_recovery_days=avg_recovery,
        )

    def _identify_drawdown_periods(
        self,
        cumulative: pd.Series,
        drawdowns: pd.Series,
    ) -> list[DrawdownPeriod]:
        """Identify individual drawdown periods.

        Args:
            cumulative: Cumulative return series
            drawdowns: Drawdown series

        Returns:
            List of DrawdownPeriod objects
        """
        periods: list[DrawdownPeriod] = []
        in_drawdown = False
        start_idx: int | None = None
        trough_idx: int | None = None
        trough_value = 0.0

        for i in range(len(drawdowns)):
            dd = drawdowns.iloc[i]
            current_date = (
                drawdowns.index[i].date()
                if hasattr(drawdowns.index[i], "date")
                else drawdowns.index[i]
            )

            if not in_drawdown and dd < 0:
                # Start of new drawdown
                in_drawdown = True
                start_idx = i
                trough_idx = i
                trough_value = dd

            elif in_drawdown:
                if dd < trough_value:
                    # New trough
                    trough_idx = i
                    trough_value = dd

                if dd >= 0:
                    # Recovery
                    start_date = drawdowns.index[start_idx].date() if hasattr(drawdowns.index[start_idx], "date") else drawdowns.index[start_idx]
                    trough_date = drawdowns.index[trough_idx].date() if hasattr(drawdowns.index[trough_idx], "date") else drawdowns.index[trough_idx]

                    period = DrawdownPeriod(
                        start_date=start_date,
                        trough_date=trough_date,
                        end_date=current_date,
                        max_drawdown=trough_value,
                        duration_days=i - start_idx,
                        recovery_days=i - trough_idx,
                    )
                    periods.append(period)
                    in_drawdown = False

        # Handle ongoing drawdown
        if in_drawdown and start_idx is not None and trough_idx is not None:
            start_date = drawdowns.index[start_idx].date() if hasattr(drawdowns.index[start_idx], "date") else drawdowns.index[start_idx]
            trough_date = drawdowns.index[trough_idx].date() if hasattr(drawdowns.index[trough_idx], "date") else drawdowns.index[trough_idx]

            period = DrawdownPeriod(
                start_date=start_date,
                trough_date=trough_date,
                end_date=None,  # Still ongoing
                max_drawdown=trough_value,
                duration_days=len(drawdowns) - 1 - start_idx,
                recovery_days=None,
            )
            periods.append(period)

        return periods

    def calculate_volatility_stats(
        self,
        returns: pd.Series,
        periods_per_year: int = 252,
    ) -> dict[str, float]:
        """Calculate various volatility statistics.

        Args:
            returns: Series of returns
            periods_per_year: Annualization factor

        Returns:
            Dictionary of volatility metrics
        """
        ann_factor = np.sqrt(periods_per_year)

        # Overall volatility
        overall_vol = float(returns.std() * ann_factor)

        # Downside volatility (only negative returns)
        downside = returns[returns < 0]
        downside_vol = float(downside.std() * ann_factor) if len(downside) > 0 else 0.0

        # Upside volatility
        upside = returns[returns > 0]
        upside_vol = float(upside.std() * ann_factor) if len(upside) > 0 else 0.0

        # Rolling volatility (20-day)
        rolling_vol = returns.rolling(20).std() * ann_factor
        max_rolling_vol = float(rolling_vol.max())
        min_rolling_vol = float(rolling_vol.min())

        return {
            "annualized_volatility": overall_vol,
            "downside_volatility": downside_vol,
            "upside_volatility": upside_vol,
            "max_rolling_vol_20d": max_rolling_vol,
            "min_rolling_vol_20d": min_rolling_vol,
            "volatility_of_volatility": float(rolling_vol.std()),
        }

    def calculate_tail_risk(
        self,
        returns: pd.Series,
        confidence_levels: list[float] | None = None,
    ) -> dict[str, float]:
        """Calculate tail risk metrics.

        Args:
            returns: Series of returns
            confidence_levels: VaR confidence levels (default [90, 95, 99])

        Returns:
            Dictionary of tail risk metrics
        """
        if confidence_levels is None:
            confidence_levels = [0.90, 0.95, 0.99]

        results: dict[str, float] = {}

        for conf in confidence_levels:
            alpha = 1 - conf
            var = float(returns.quantile(alpha))
            cvar = float(returns[returns <= var].mean()) if len(returns[returns <= var]) > 0 else var

            results[f"VaR_{int(conf*100)}"] = var
            results[f"CVaR_{int(conf*100)}"] = cvar

        # Additional tail metrics
        results["max_daily_loss"] = float(returns.min())
        results["max_daily_gain"] = float(returns.max())
        results["skewness"] = float(returns.skew())
        results["kurtosis"] = float(returns.kurtosis())

        return results

    def calculate_correlation_risk(
        self,
        returns_matrix: pd.DataFrame,
    ) -> dict[str, float]:
        """Calculate correlation-based risk metrics.

        Args:
            returns_matrix: DataFrame with asset returns as columns

        Returns:
            Dictionary of correlation metrics
        """
        corr = returns_matrix.corr()

        # Average pairwise correlation
        n = len(corr)
        if n > 1:
            upper_triangle = corr.where(
                np.triu(np.ones(corr.shape), k=1).astype(bool)
            )
            avg_corr = float(upper_triangle.stack().mean())
            max_corr = float(upper_triangle.stack().max())
            min_corr = float(upper_triangle.stack().min())
        else:
            avg_corr = max_corr = min_corr = 1.0

        return {
            "average_correlation": avg_corr,
            "max_correlation": max_corr,
            "min_correlation": min_corr,
            "num_assets": n,
        }
