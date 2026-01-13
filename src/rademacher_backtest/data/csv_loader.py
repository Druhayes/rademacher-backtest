"""CSV data loader implementation for sample financial data.

This module provides CSVLoader, a DataLoader implementation that reads
sample financial data from CSV files. It's designed for testing, demos,
and offline analysis when a database connection is not available.

The loader expects the following CSV files in the data directory:
- sample_prices.csv: Daily adjusted close prices for multiple tickers
- sample_ff.csv: Fama-French factor returns (mkt, smb, hml, rmw, cma, mom, rf)
- sample_rf.csv: Daily risk-free rate data

Usage:
    from rademacher_backtest.data import CSVLoader
    
    loader = CSVLoader()
    prices = loader.load_prices(['SCHX', 'AGG'], start_date, end_date)
    factors = loader.load_fama_french(start_date, end_date)
    rf_rate = loader.load_risk_free_rate(start_date, end_date)
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pandas as pd


class CSVLoader:
    """CSV data loader for sample financial data.

    Loads price data, Fama-French factors, and risk-free rates from CSV files
    in the data directory. This loader is useful for testing, demos, and 
    offline analysis when a database connection is not available.
    """

    def __init__(self, data_dir: str | Path | None = None) -> None:
        """Initialize the CSV loader.

        Args:
            data_dir: Path to directory containing CSV files. If None, uses
                     the same directory as this module.
        """
        if data_dir is None:
            # Default to the directory containing this module
            data_dir = Path(__file__).parent
        self.data_dir = Path(data_dir)

        # Define expected file paths
        self.prices_file = self.data_dir / "sample_prices.csv"
        self.fama_french_file = self.data_dir / "sample_ff.csv"
        self.risk_free_file = self.data_dir / "sample_rf.csv"

    def _check_files_exist(self) -> None:
        """Check if required CSV files exist.

        Raises:
            FileNotFoundError: If any required CSV files are missing
        """
        missing_files = []
        for file_path in [self.prices_file, self.fama_french_file, self.risk_free_file]:
            if not file_path.exists():
                missing_files.append(str(file_path))

        if missing_files:
            raise FileNotFoundError(
                f"Required CSV files not found: {missing_files}. "
                f"Please ensure the data files exist in {self.data_dir}"
            )

    def load_prices(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        """Load adjusted close prices from sample_prices.csv.

        Args:
            tickers: List of ticker symbols (e.g., ['SCHX', 'AGG'])
            start: Start date for price data
            end: End date for price data

        Returns:
            DataFrame with DatetimeIndex and tickers as columns,
            containing adjusted close prices

        Raises:
            FileNotFoundError: If sample_prices.csv is not found
            ValueError: If requested tickers are not available in the data
        """
        self._check_files_exist()

        # Load the CSV file
        df = pd.read_csv(self.prices_file)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

        # Filter by date range
        df = df[(df.index.date >= start) & (df.index.date <= end)]

        if df.empty:
            return pd.DataFrame()

        # Check if all requested tickers are available
        available_tickers = set(df.columns)
        missing_tickers = set(tickers) - available_tickers
        if missing_tickers:
            raise ValueError(
                f"Requested tickers not available in sample data: {missing_tickers}. "
                f"Available tickers: {sorted(available_tickers)}"
            )

        # Return data for requested tickers in the specified order
        return df[tickers].sort_index()

    def load_risk_free_rate(self, start: date, end: date) -> pd.Series:
        """Load daily risk-free rate from sample_rf.csv.

        Args:
            start: Start date for the data
            end: End date for the data

        Returns:
            Series with DatetimeIndex and risk-free rate values (already in decimal form)

        Raises:
            FileNotFoundError: If sample_rf.csv is not found
        """
        self._check_files_exist()

        # Load the CSV file
        df = pd.read_csv(self.risk_free_file)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

        # Filter by date range
        df = df[(df.index.date >= start) & (df.index.date <= end)]

        if df.empty:
            return pd.Series(dtype=float)

        return df["rf"].sort_index()

    def load_fama_french(self, start: date, end: date) -> pd.DataFrame:
        """Load Fama-French factor data from sample_ff.csv.

        Loads the following factors:
        - mkt: Market excess return
        - smb: Small Minus Big (size factor)
        - hml: High Minus Low (value factor)
        - rmw: Robust Minus Weak (profitability factor)
        - cma: Conservative Minus Aggressive (investment factor)
        - mom: Momentum factor
        - rf: Risk-free rate

        Args:
            start: Start date for the data
            end: End date for the data

        Returns:
            DataFrame with factors as columns, values already in decimal form

        Raises:
            FileNotFoundError: If sample_ff.csv is not found
        """
        self._check_files_exist()

        # Load the CSV file
        df = pd.read_csv(self.fama_french_file)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

        # Filter by date range
        df = df[(df.index.date >= start) & (df.index.date <= end)]

        if df.empty:
            return pd.DataFrame()

        return df.sort_index()

    def get_available_tickers(self) -> list[str]:
        """Get list of available tickers in the sample data.

        Returns:
            List of ticker symbols available in sample_prices.csv

        Raises:
            FileNotFoundError: If sample_prices.csv is not found
        """
        self._check_files_exist()

        df = pd.read_csv(self.prices_file, nrows=1)  # Just read header
        return [col for col in df.columns if col != "date"]

    def get_data_range(self) -> tuple[date, date]:
        """Get the available date range in the sample data.

        Returns:
            Tuple of (min_date, max_date) across all data files

        Raises:
            FileNotFoundError: If required CSV files are not found
        """
        self._check_files_exist()

        # Check date range across all files
        date_ranges = []

        for file_path in [self.prices_file, self.fama_french_file, self.risk_free_file]:
            df = pd.read_csv(file_path, usecols=["date"])
            df["date"] = pd.to_datetime(df["date"])
            date_ranges.extend([df["date"].min().date(), df["date"].max().date()])

        return min(date_ranges), max(date_ranges)

    def info(self) -> dict:
        """Get information about the sample dataset.

        Returns:
            Dictionary with dataset metadata including available tickers,
            date range, and file locations
        """
        try:
            tickers = self.get_available_tickers()
            start_date, end_date = self.get_data_range()

            return {
                "loader_type": "CSV Sample Data",
                "data_directory": str(self.data_dir),
                "available_tickers": tickers,
                "date_range": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat(),
                },
                "files": {
                    "prices": str(self.prices_file),
                    "fama_french": str(self.fama_french_file),
                    "risk_free": str(self.risk_free_file),
                },
                "status": "ready",
            }
        except FileNotFoundError as e:
            return {
                "loader_type": "CSV Sample Data",
                "data_directory": str(self.data_dir),
                "status": "error",
                "error": str(e),
            }