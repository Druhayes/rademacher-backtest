"""DataFrame-based data loader for in-memory financial data.

This module provides DataFrameLoader, the primary data loader for the
rademacher-backtest package. It implements the DataLoader Protocol and works
with pre-loaded pandas DataFrames, making it ideal for:
- Jupyter notebooks and interactive analysis
- Custom data sources (APIs, databases, files)
- Testing and prototyping
- Any scenario where data is already in memory

Usage:
    import pandas as pd
    import rademacher_backtest as rbt

    # Load your data into DataFrames
    prices = pd.DataFrame(...)  # DatetimeIndex, ticker columns

    # Create loader
    loader = rbt.DataFrameLoader(prices)

    # Use with backtest engine
    engine = rbt.BacktestEngine(config, portfolio, loader=loader)
"""

from __future__ import annotations

from datetime import date

import pandas as pd


class DataFrameLoader:
    """Memory-based data loader using pandas DataFrames.

    This loader implements the DataLoader Protocol and works with
    pre-loaded DataFrames. It's the recommended data source for most users.

    The loader performs filtering by date range and ticker symbols, but does
    not perform any I/O operations. All data must be provided at initialization.

    Args:
        prices: DataFrame with DatetimeIndex and ticker symbol columns.
                Values should be adjusted close prices.
        risk_free_rate: Optional Series with DatetimeIndex and daily risk-free
                       rate in decimal form (e.g., 0.0001 for 0.01%).
        fama_french: Optional DataFrame with DatetimeIndex and Fama-French
                    factor columns (mkt, smb, hml, rmw, cma, mom, rf) in
                    decimal form.

    Example:
        >>> import pandas as pd
        >>> import rademacher_backtest as rbt
        >>>
        >>> # Create sample data
        >>> dates = pd.date_range('2020-01-01', '2023-12-31')
        >>> prices = pd.DataFrame({
        ...     'SPY': [100, 101, 102, ...],
        ...     'AGG': [50, 50.1, 50.2, ...],
        ... }, index=dates)
        >>>
        >>> # Create loader
        >>> loader = rbt.DataFrameLoader(prices)
        >>>
        >>> # Load subset
        >>> result = loader.load_prices(
        ...     ['SPY'],
        ...     date(2020, 1, 1),
        ...     date(2020, 12, 31)
        ... )
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        risk_free_rate: pd.Series | None = None,
        fama_french: pd.DataFrame | None = None,
    ) -> None:
        """Initialize the DataFrame loader.

        Args:
            prices: DataFrame with DatetimeIndex and ticker columns
            risk_free_rate: Optional Series with DatetimeIndex (daily risk-free rate)
            fama_french: Optional DataFrame with Fama-French factors

        Raises:
            ValueError: If prices DataFrame is not properly formatted
        """
        # Validate prices DataFrame
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise ValueError(
                "prices DataFrame must have a DatetimeIndex. "
                f"Got {type(prices.index).__name__}. "
                "Use pd.to_datetime() to convert the index."
            )

        if prices.empty:
            raise ValueError("prices DataFrame cannot be empty")

        # Store data (make copies to avoid external mutations)
        self._prices = prices.copy()
        self._prices = self._prices.sort_index()

        # Store optional data
        if risk_free_rate is not None:
            if not isinstance(risk_free_rate.index, pd.DatetimeIndex):
                raise ValueError(
                    "risk_free_rate Series must have a DatetimeIndex"
                )
            self._risk_free_rate = risk_free_rate.copy()
            self._risk_free_rate = self._risk_free_rate.sort_index()
        else:
            self._risk_free_rate = None

        if fama_french is not None:
            if not isinstance(fama_french.index, pd.DatetimeIndex):
                raise ValueError(
                    "fama_french DataFrame must have a DatetimeIndex"
                )
            self._fama_french = fama_french.copy()
            self._fama_french = self._fama_french.sort_index()
        else:
            self._fama_french = None

    def load_prices(
        self,
        tickers: list[str],
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """Load adjusted close prices for specified tickers and date range.

        Args:
            tickers: List of ticker symbols to load (e.g., ['SPY', 'AGG'])
            start: Start date for the data (inclusive)
            end: End date for the data (inclusive)

        Returns:
            DataFrame with DatetimeIndex and tickers as columns,
            containing adjusted close prices for the specified date range

        Raises:
            ValueError: If any requested tickers are not available
        """
        # Check if all requested tickers are available
        available_tickers = set(self._prices.columns)
        missing_tickers = set(tickers) - available_tickers
        if missing_tickers:
            raise ValueError(
                f"Requested tickers not found in data: {sorted(missing_tickers)}. "
                f"Available tickers: {sorted(available_tickers)}"
            )

        # Filter by date range
        mask = (self._prices.index.date >= start) & (self._prices.index.date <= end)
        filtered = self._prices.loc[mask]

        if filtered.empty:
            return pd.DataFrame()

        # Return data for requested tickers in the specified order
        return filtered[tickers].sort_index()

    def load_risk_free_rate(self, start: date, end: date) -> pd.Series:
        """Load daily risk-free rate for specified date range.

        Args:
            start: Start date for the data (inclusive)
            end: End date for the data (inclusive)

        Returns:
            Series with DatetimeIndex and risk-free rate values in decimal form.
            Returns empty Series if risk-free rate data was not provided at init.
        """
        if self._risk_free_rate is None:
            return pd.Series(dtype=float)

        # Filter by date range
        mask = (
            (self._risk_free_rate.index.date >= start) &
            (self._risk_free_rate.index.date <= end)
        )
        filtered = self._risk_free_rate.loc[mask]

        return filtered.sort_index()

    def load_fama_french(self, start: date, end: date) -> pd.DataFrame:
        """Load Fama-French factor data for specified date range.

        Expected factors: mkt, smb, hml, rmw, cma, mom, rf

        Args:
            start: Start date for the data (inclusive)
            end: End date for the data (inclusive)

        Returns:
            DataFrame with DatetimeIndex and factor columns in decimal form.
            Returns empty DataFrame if Fama-French data was not provided at init.
        """
        if self._fama_french is None:
            return pd.DataFrame()

        # Filter by date range
        mask = (
            (self._fama_french.index.date >= start) &
            (self._fama_french.index.date <= end)
        )
        filtered = self._fama_french.loc[mask]

        return filtered.sort_index()

    @property
    def available_tickers(self) -> list[str]:
        """Get list of available ticker symbols.

        Returns:
            List of ticker symbols available in the prices data
        """
        return list(self._prices.columns)

    @property
    def date_range(self) -> tuple[date, date]:
        """Get the available date range in the prices data.

        Returns:
            Tuple of (min_date, max_date) for the price data
        """
        return (
            self._prices.index.min().date(),
            self._prices.index.max().date(),
        )

    def info(self) -> dict:
        """Get information about the loaded dataset.

        Returns:
            Dictionary with dataset metadata including available tickers,
            date range, and whether optional data is available
        """
        min_date, max_date = self.date_range

        return {
            "loader_type": "DataFrame (In-Memory)",
            "available_tickers": self.available_tickers,
            "num_tickers": len(self.available_tickers),
            "date_range": {
                "start": min_date.isoformat(),
                "end": max_date.isoformat(),
                "num_days": len(self._prices),
            },
            "optional_data": {
                "risk_free_rate": self._risk_free_rate is not None,
                "fama_french": self._fama_french is not None,
            },
            "status": "ready",
        }
