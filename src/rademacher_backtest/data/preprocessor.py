"""Data preprocessing utilities for returns calculation and date alignment."""

from __future__ import annotations

from datetime import date
from typing import Literal

import numpy as np
import pandas as pd


class DataPreprocessor:
    """Data preprocessing utilities for financial time series.

    Provides methods for calculating returns, aligning dates, and
    generating rebalancing schedules.
    """

    @staticmethod
    def calculate_returns(
        prices: pd.DataFrame,
        method: Literal["simple", "log"] = "simple",
    ) -> pd.DataFrame:
        """Calculate returns from prices.

        Args:
            prices: DataFrame with price data (dates as index, tickers as columns)
            method: 'simple' for arithmetic returns, 'log' for logarithmic returns

        Returns:
            DataFrame of returns with the same structure (first row dropped)
        """
        if method == "simple":
            returns = prices.pct_change()
        elif method == "log":
            returns = np.log(prices / prices.shift(1))
        else:
            raise ValueError(f"Unknown method: {method}. Use 'simple' or 'log'.")

        return returns.dropna()

    @staticmethod
    def calculate_portfolio_returns(
        asset_returns: pd.DataFrame,
        weights: dict[str, float],
    ) -> pd.Series:
        """Calculate portfolio returns from asset returns and weights.

        Args:
            asset_returns: DataFrame of individual asset returns
            weights: Dictionary mapping ticker to weight

        Returns:
            Series of portfolio returns
        """
        # Ensure weights are in the same order as columns
        weight_series = pd.Series(
            [weights[col] for col in asset_returns.columns],
            index=asset_returns.columns,
        )
        return (asset_returns * weight_series).sum(axis=1)

    @staticmethod
    def align_to_frequency(
        df: pd.DataFrame,
        frequency: Literal["D", "W", "ME", "QE", "YE"] = "ME",
    ) -> pd.DataFrame:
        """Resample DataFrame to specified frequency, taking last value.

        Args:
            df: DataFrame with DatetimeIndex
            frequency: Pandas frequency string
                - 'D': Daily
                - 'W': Weekly
                - 'ME': Month-end
                - 'QE': Quarter-end
                - 'YE': Year-end

        Returns:
            Resampled DataFrame
        """
        return df.resample(frequency).last()

    @staticmethod
    def get_rebalance_dates(
        start: date,
        end: date,
        frequency: Literal["daily", "weekly", "monthly", "quarterly", "annually"] = "monthly",
    ) -> list[date]:
        """Generate rebalancing dates based on frequency.

        Args:
            start: Start date
            end: End date
            frequency: Rebalancing frequency

        Returns:
            List of rebalancing dates (end of period)
        """
        freq_map = {
            "daily": "B",  # Business days
            "weekly": "W-FRI",  # Weekly on Friday
            "monthly": "ME",  # Month end
            "quarterly": "QE",  # Quarter end
            "annually": "YE",  # Year end
        }

        if frequency not in freq_map:
            raise ValueError(f"Unknown frequency: {frequency}")

        dates = pd.date_range(start, end, freq=freq_map[frequency])
        return [d.date() for d in dates]

    @staticmethod
    def get_trading_days(
        prices: pd.DataFrame,
        start: date,
        end: date,
    ) -> list[date]:
        """Extract trading days from price data within date range.

        Args:
            prices: DataFrame with DatetimeIndex
            start: Start date
            end: End date

        Returns:
            List of trading dates
        """
        mask = (prices.index.date >= start) & (prices.index.date <= end)
        return [d.date() for d in prices.index[mask]]

    @staticmethod
    def forward_fill_prices(prices: pd.DataFrame) -> pd.DataFrame:
        """Forward fill missing prices (e.g., for holidays).

        Args:
            prices: DataFrame with potential missing values

        Returns:
            DataFrame with forward-filled values
        """
        return prices.ffill()

    @staticmethod
    def detect_outliers(
        returns: pd.DataFrame,
        n_std: float = 5.0,
    ) -> pd.DataFrame:
        """Detect outlier returns based on standard deviation threshold.

        Args:
            returns: DataFrame of returns
            n_std: Number of standard deviations to use as threshold

        Returns:
            Boolean DataFrame indicating outliers
        """
        mean = returns.mean()
        std = returns.std()
        return (returns - mean).abs() > n_std * std

    @staticmethod
    def winsorize_returns(
        returns: pd.DataFrame,
        lower_percentile: float = 0.01,
        upper_percentile: float = 0.99,
    ) -> pd.DataFrame:
        """Winsorize returns to specified percentiles.

        Args:
            returns: DataFrame of returns
            lower_percentile: Lower percentile threshold (e.g., 0.01 for 1%)
            upper_percentile: Upper percentile threshold (e.g., 0.99 for 99%)

        Returns:
            DataFrame with winsorized returns
        """
        lower = returns.quantile(lower_percentile)
        upper = returns.quantile(upper_percentile)
        return returns.clip(lower=lower, upper=upper, axis=1)

    @staticmethod
    def standardize_returns(returns: pd.DataFrame) -> pd.DataFrame:
        """Standardize returns (z-score) for RAS analysis.

        Args:
            returns: DataFrame of returns

        Returns:
            DataFrame of standardized returns (mean=0, std=1)
        """
        mean = returns.mean()
        std = returns.std()
        # Avoid division by zero
        std = std.replace(0, 1)
        return (returns - mean) / std

    @staticmethod
    def align_dataframes(
        *dfs: pd.DataFrame,
        method: Literal["inner", "outer"] = "inner",
    ) -> list[pd.DataFrame]:
        """Align multiple DataFrames on their index.

        Args:
            *dfs: DataFrames to align
            method: 'inner' for intersection, 'outer' for union

        Returns:
            List of aligned DataFrames
        """
        if not dfs:
            return []

        # Get common index
        if method == "inner":
            common_index = dfs[0].index
            for df in dfs[1:]:
                common_index = common_index.intersection(df.index)
        else:  # outer
            common_index = dfs[0].index
            for df in dfs[1:]:
                common_index = common_index.union(df.index)

        common_index = common_index.sort_values()
        return [df.reindex(common_index) for df in dfs]
