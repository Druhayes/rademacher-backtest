"""Data loader Protocol definition.

This module defines the DataLoader Protocol that all data loaders must implement.
The Protocol pattern allows for flexible data source integration without tight
coupling to specific implementations.

Implementations:
- DataFrameLoader: In-memory pandas DataFrames (core package)
- CSVLoader: Load from CSV files (core package)
- PostgreSQLLoader: PostgreSQL database (requires 'postgres' extra)
"""

from __future__ import annotations

from datetime import date
from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class DataLoader(Protocol):
    """Protocol defining the interface for data loaders.

    All data loaders must implement these three methods to be compatible
    with the BacktestEngine. This allows users to create custom data loaders
    for any data source (APIs, databases, files, etc.).

    Example:
        >>> class MyCustomLoader:
        ...     def load_prices(self, tickers, start, end):
        ...         # Load from your custom source
        ...         return prices_dataframe
        ...     def load_risk_free_rate(self, start, end):
        ...         return rf_series
        ...     def load_fama_french(self, start, end):
        ...         return ff_dataframe
        >>>
        >>> loader = MyCustomLoader()
        >>> engine = BacktestEngine(config, portfolio, loader=loader)
    """

    def load_prices(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        """Load adjusted close prices for given tickers.

        Args:
            tickers: List of ticker symbols to load
            start: Start date for the data (inclusive)
            end: End date for the data (inclusive)

        Returns:
            DataFrame with DatetimeIndex and tickers as columns,
            containing adjusted close prices
        """
        ...

    def load_risk_free_rate(self, start: date, end: date) -> pd.Series:
        """Load daily risk-free rate.

        Args:
            start: Start date for the data (inclusive)
            end: End date for the data (inclusive)

        Returns:
            Series with DatetimeIndex and risk-free rate values in decimal form.
            Can return empty Series if risk-free rate data is not available.
        """
        ...

    def load_fama_french(self, start: date, end: date) -> pd.DataFrame:
        """Load Fama-French factor data.

        Args:
            start: Start date for the data (inclusive)
            end: End date for the data (inclusive)

        Returns:
            DataFrame with DatetimeIndex and factor columns
            (mkt, smb, hml, rmw, cma, mom, rf) in decimal form.
            Can return empty DataFrame if factor data is not available.
        """
        ...
