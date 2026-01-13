"""Backtest result data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from rademacher_backtest.config.portfolio import PortfolioDefinition
    from rademacher_backtest.config.settings import BacktestConfig
    from rademacher_backtest.portfolio.portfolio import Holdings


@dataclass
class DailySnapshot:
    """Daily portfolio state snapshot.

    Captures the portfolio state at the end of each trading day.

    Attributes:
        date: The date of the snapshot
        portfolio_value: Total portfolio value
        holdings: Current holdings object
        weights: Current portfolio weights
        cash: Cash balance
    """

    date: date
    portfolio_value: float
    holdings: Holdings
    weights: dict[str, float]
    cash: float

    @property
    def equity_value(self) -> float:
        """Total equity value (excluding cash)."""
        return self.portfolio_value - self.cash


@dataclass
class BacktestResult:
    """Complete backtest results container.

    Contains all data generated during a backtest run, including
    daily values, returns, rebalancing information, and transaction costs.

    Attributes:
        config: The backtest configuration used
        portfolio_def: The portfolio definition
        daily_values: Series of daily portfolio values
        daily_returns: Series of daily returns
        monthly_returns: Series of monthly returns
        rebalance_dates: List of dates when rebalancing occurred
        transaction_costs: Series of transaction costs by date
        total_transaction_costs: Sum of all transaction costs
        snapshots: List of daily portfolio snapshots
    """

    config: BacktestConfig
    portfolio_def: PortfolioDefinition
    daily_values: pd.Series
    daily_returns: pd.Series
    monthly_returns: pd.Series
    rebalance_dates: list[date]
    transaction_costs: pd.Series
    total_transaction_costs: float
    snapshots: list[DailySnapshot] = field(default_factory=list)

    @property
    def start_date(self) -> date:
        """First date in the backtest."""
        return self.daily_values.index[0].date()

    @property
    def end_date(self) -> date:
        """Last date in the backtest."""
        return self.daily_values.index[-1].date()

    @property
    def initial_value(self) -> float:
        """Initial portfolio value."""
        return self.daily_values.iloc[0]

    @property
    def final_value(self) -> float:
        """Final portfolio value."""
        return self.daily_values.iloc[-1]

    @property
    def total_return(self) -> float:
        """Total return over the backtest period (as decimal)."""
        return self.final_value / self.initial_value - 1

    @property
    def num_trading_days(self) -> int:
        """Number of trading days in the backtest."""
        return len(self.daily_values)

    @property
    def num_rebalances(self) -> int:
        """Number of rebalancing events."""
        return len(self.rebalance_dates)

    @property
    def years(self) -> float:
        """Approximate number of years in the backtest."""
        return self.num_trading_days / 252

    @property
    def cumulative_returns(self) -> pd.Series:
        """Cumulative return series."""
        return (1 + self.daily_returns).cumprod()

    @property
    def normalized_values(self) -> pd.Series:
        """Portfolio values normalized to start at 100."""
        return (self.daily_values / self.initial_value) * 100

    def get_weights_history(self) -> pd.DataFrame:
        """Get historical weights as a DataFrame.

        Returns:
            DataFrame with dates as index and tickers as columns
        """
        data = {
            snapshot.date: snapshot.weights
            for snapshot in self.snapshots
        }
        df = pd.DataFrame.from_dict(data, orient="index")
        df.index = pd.to_datetime(df.index)
        return df.sort_index()

    def get_monthly_values(self) -> pd.Series:
        """Get month-end portfolio values.

        Returns:
            Series with month-end portfolio values
        """
        return self.daily_values.resample("ME").last()

    def get_yearly_returns(self) -> pd.Series:
        """Get annual returns.

        Returns:
            Series with yearly returns
        """
        return self.daily_values.resample("YE").last().pct_change().dropna()

    def get_drawdowns(self) -> pd.Series:
        """Calculate drawdown series.

        Returns:
            Series of drawdowns (negative values)
        """
        cumulative = (1 + self.daily_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        return cumulative / rolling_max - 1

    def get_max_drawdown(self) -> float:
        """Get maximum drawdown.

        Returns:
            Maximum drawdown as negative decimal
        """
        return self.get_drawdowns().min()

    def summary(self) -> dict:
        """Generate summary statistics dictionary.

        Returns:
            Dictionary of key backtest statistics
        """
        drawdowns = self.get_drawdowns()

        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "trading_days": self.num_trading_days,
            "years": round(self.years, 2),
            "initial_value": round(self.initial_value, 2),
            "final_value": round(self.final_value, 2),
            "total_return_pct": round(self.total_return * 100, 2),
            "num_rebalances": self.num_rebalances,
            "total_transaction_costs": round(self.total_transaction_costs, 2),
            "max_drawdown_pct": round(drawdowns.min() * 100, 2),
            "portfolio_name": self.portfolio_def.name,
        }

    def __repr__(self) -> str:
        """Return string representation of results."""
        return (
            f"BacktestResult("
            f"period={self.start_date} to {self.end_date}, "
            f"return={self.total_return:.2%}, "
            f"rebalances={self.num_rebalances})"
        )
