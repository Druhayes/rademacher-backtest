"""Performance metrics calculation.

Calculates comprehensive performance statistics including returns,
risk-adjusted metrics, and distribution characteristics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics.

    All return metrics are in decimal form (0.10 = 10%).
    All ratio metrics are dimensionless.
    """

    # Returns
    total_return: float
    cagr: float
    best_year: float
    worst_year: float
    best_month: float
    worst_month: float

    # Risk-adjusted
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Volatility
    annualized_volatility: float
    downside_volatility: float

    # Drawdown
    max_drawdown: float
    max_drawdown_duration_days: int
    average_drawdown: float

    # Distribution
    skewness: float
    kurtosis: float
    var_95: float  # Value at Risk (5th percentile)
    cvar_95: float  # Conditional VaR (Expected Shortfall)

    # Win/Loss
    positive_periods: int
    negative_periods: int
    win_rate: float
    avg_win: float
    avg_loss: float
    win_loss_ratio: float

    def to_dict(self) -> dict:
        """Convert metrics to dictionary with formatted values."""
        return {
            "Total Return (%)": round(self.total_return * 100, 2),
            "CAGR (%)": round(self.cagr * 100, 2),
            "Best Year (%)": round(self.best_year * 100, 2),
            "Worst Year (%)": round(self.worst_year * 100, 2),
            "Best Month (%)": round(self.best_month * 100, 2),
            "Worst Month (%)": round(self.worst_month * 100, 2),
            "Sharpe Ratio": round(self.sharpe_ratio, 3),
            "Sortino Ratio": round(self.sortino_ratio, 3),
            "Calmar Ratio": round(self.calmar_ratio, 3),
            "Annualized Volatility (%)": round(self.annualized_volatility * 100, 2),
            "Downside Volatility (%)": round(self.downside_volatility * 100, 2),
            "Max Drawdown (%)": round(self.max_drawdown * 100, 2),
            "Max DD Duration (days)": self.max_drawdown_duration_days,
            "Average Drawdown (%)": round(self.average_drawdown * 100, 2),
            "Skewness": round(self.skewness, 3),
            "Kurtosis": round(self.kurtosis, 3),
            "VaR 95% (%)": round(self.var_95 * 100, 2),
            "CVaR 95% (%)": round(self.cvar_95 * 100, 2),
            "Win Rate (%)": round(self.win_rate * 100, 1),
            "Avg Win (%)": round(self.avg_win * 100, 3),
            "Avg Loss (%)": round(self.avg_loss * 100, 3),
            "Win/Loss Ratio": round(self.win_loss_ratio, 2),
        }

    def summary_lines(self) -> list[str]:
        """Generate formatted summary lines."""
        lines = [
            "=" * 50,
            "PERFORMANCE METRICS",
            "=" * 50,
            "",
            "RETURNS",
            "-" * 30,
            f"Total Return:        {self.total_return * 100:>10.2f}%",
            f"CAGR:                {self.cagr * 100:>10.2f}%",
            f"Best Year:           {self.best_year * 100:>10.2f}%",
            f"Worst Year:          {self.worst_year * 100:>10.2f}%",
            "",
            "RISK-ADJUSTED",
            "-" * 30,
            f"Sharpe Ratio:        {self.sharpe_ratio:>10.3f}",
            f"Sortino Ratio:       {self.sortino_ratio:>10.3f}",
            f"Calmar Ratio:        {self.calmar_ratio:>10.3f}",
            "",
            "VOLATILITY",
            "-" * 30,
            f"Ann. Volatility:     {self.annualized_volatility * 100:>10.2f}%",
            f"Downside Vol:        {self.downside_volatility * 100:>10.2f}%",
            "",
            "DRAWDOWN",
            "-" * 30,
            f"Max Drawdown:        {self.max_drawdown * 100:>10.2f}%",
            f"Max DD Duration:     {self.max_drawdown_duration_days:>10d} days",
            "",
            "DISTRIBUTION",
            "-" * 30,
            f"Skewness:            {self.skewness:>10.3f}",
            f"Kurtosis:            {self.kurtosis:>10.3f}",
            f"VaR (95%):           {self.var_95 * 100:>10.2f}%",
            f"CVaR (95%):          {self.cvar_95 * 100:>10.2f}%",
            "",
            "=" * 50,
        ]
        return lines

    def __str__(self) -> str:
        """Return formatted string."""
        return "\n".join(self.summary_lines())


class PerformanceCalculator:
    """Calculate comprehensive performance metrics from returns.

    Handles both daily and monthly return series and computes
    various performance statistics.
    """

    def __init__(
        self,
        risk_free_rate: pd.Series | None = None,
        periods_per_year: int = 252,
    ) -> None:
        """Initialize the calculator.

        Args:
            risk_free_rate: Optional risk-free rate series (in decimal form)
            periods_per_year: Number of periods per year (252 for daily)
        """
        self.rf = risk_free_rate
        self.periods_per_year = periods_per_year

    def calculate(self, returns: pd.Series) -> PerformanceMetrics:
        """Calculate all performance metrics from a returns series.

        Args:
            returns: Series of returns (daily or monthly, in decimal form)

        Returns:
            PerformanceMetrics with all calculated statistics
        """
        # Basic calculations
        n_periods = len(returns)
        if n_periods < 2:
            raise ValueError("Need at least 2 periods to calculate metrics")

        years = n_periods / self.periods_per_year

        # Excess returns for Sharpe
        if self.rf is not None:
            aligned_rf = self.rf.reindex(returns.index).fillna(0)
            excess_returns = returns - aligned_rf
        else:
            excess_returns = returns

        # Total return and CAGR
        cumulative = (1 + returns).prod()
        total_return = cumulative - 1
        cagr = cumulative ** (1 / years) - 1 if years > 0 else 0.0

        # Monthly/yearly aggregations
        monthly_returns = self._aggregate_to_monthly(returns)
        yearly_returns = self._aggregate_to_yearly(returns)

        best_month = float(monthly_returns.max()) if len(monthly_returns) > 0 else 0.0
        worst_month = float(monthly_returns.min()) if len(monthly_returns) > 0 else 0.0
        best_year = float(yearly_returns.max()) if len(yearly_returns) > 0 else 0.0
        worst_year = float(yearly_returns.min()) if len(yearly_returns) > 0 else 0.0

        # Volatility
        ann_vol = float(returns.std() * np.sqrt(self.periods_per_year))
        downside_returns = returns[returns < 0]
        downside_vol = float(
            downside_returns.std() * np.sqrt(self.periods_per_year)
            if len(downside_returns) > 0
            else 0.0
        )

        # Risk-adjusted ratios
        mean_excess = float(excess_returns.mean() * self.periods_per_year)
        sharpe = mean_excess / ann_vol if ann_vol > 0 else 0.0
        sortino = mean_excess / downside_vol if downside_vol > 0 else 0.0

        # Drawdown analysis
        max_dd, max_dd_duration, avg_dd = self._calculate_drawdowns(returns)
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

        # Distribution metrics
        skewness = float(returns.skew()) if len(returns) > 2 else 0.0
        kurtosis = float(returns.kurtosis()) if len(returns) > 3 else 0.0
        var_95 = float(returns.quantile(0.05))
        cvar_95 = float(returns[returns <= var_95].mean()) if len(returns[returns <= var_95]) > 0 else var_95

        # Win/Loss analysis
        positive = returns[returns > 0]
        negative = returns[returns < 0]
        n_positive = len(positive)
        n_negative = len(negative)
        win_rate = n_positive / n_periods if n_periods > 0 else 0.0
        avg_win = float(positive.mean()) if n_positive > 0 else 0.0
        avg_loss = float(negative.mean()) if n_negative > 0 else 0.0
        win_loss = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0

        return PerformanceMetrics(
            total_return=float(total_return),
            cagr=float(cagr),
            best_year=best_year,
            worst_year=worst_year,
            best_month=best_month,
            worst_month=worst_month,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            annualized_volatility=ann_vol,
            downside_volatility=downside_vol,
            max_drawdown=max_dd,
            max_drawdown_duration_days=max_dd_duration,
            average_drawdown=avg_dd,
            skewness=skewness,
            kurtosis=kurtosis,
            var_95=var_95,
            cvar_95=cvar_95,
            positive_periods=n_positive,
            negative_periods=n_negative,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            win_loss_ratio=win_loss,
        )

    def _aggregate_to_monthly(self, returns: pd.Series) -> pd.Series:
        """Aggregate daily returns to monthly."""
        return returns.resample("ME").apply(lambda x: (1 + x).prod() - 1).dropna()

    def _aggregate_to_yearly(self, returns: pd.Series) -> pd.Series:
        """Aggregate daily returns to yearly."""
        return returns.resample("YE").apply(lambda x: (1 + x).prod() - 1).dropna()

    def _calculate_drawdowns(
        self,
        returns: pd.Series,
    ) -> tuple[float, int, float]:
        """Calculate drawdown statistics.

        Args:
            returns: Series of returns

        Returns:
            Tuple of (max_drawdown, max_dd_duration, avg_drawdown)
        """
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = cumulative / rolling_max - 1

        max_dd = float(drawdowns.min())

        # Calculate max drawdown duration
        is_underwater = drawdowns < 0
        if not is_underwater.any():
            max_dd_duration = 0
        else:
            # Group consecutive underwater periods
            groups = (~is_underwater).cumsum()
            underwater_groups = is_underwater.groupby(groups).sum()
            max_dd_duration = int(underwater_groups.max())

        # Average drawdown (when underwater)
        underwater_values = drawdowns[drawdowns < 0]
        avg_dd = float(underwater_values.mean()) if len(underwater_values) > 0 else 0.0

        return max_dd, max_dd_duration, avg_dd

    def calculate_rolling_sharpe(
        self,
        returns: pd.Series,
        window: int = 252,
    ) -> pd.Series:
        """Calculate rolling Sharpe ratio.

        Args:
            returns: Daily returns series
            window: Rolling window size (default 252 for 1 year)

        Returns:
            Series of rolling Sharpe ratios
        """
        rolling_mean = returns.rolling(window).mean() * self.periods_per_year
        rolling_std = returns.rolling(window).std() * np.sqrt(self.periods_per_year)
        return rolling_mean / rolling_std
