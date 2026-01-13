"""PostgreSQL data loader implementation.

This module provides PostgreSQLLoader, an optional data loader that connects
to PostgreSQL databases. This loader requires the 'postgres' extra to be installed:

    pip install rademacher-backtest[postgres]

Usage:
    from rademacher_backtest.data.postgres_loader import PostgreSQLLoader

    loader = PostgreSQLLoader("postgresql://user:pass@host:5432/db")
    prices = loader.load_prices(['SPY', 'AGG'], start_date, end_date)
"""

from __future__ import annotations

import os
from datetime import date

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


class PostgreSQLLoader:
    """PostgreSQL data loader for financial data.

    Connects to PostgreSQL database and loads price data from the sfp table,
    Fama-French factors from the fama_french table, and interest rates from
    the rates table.
    """

    def __init__(self, connection_string: str | None = None) -> None:
        """Initialize the PostgreSQL loader.

        Args:
            connection_string: SQLAlchemy connection string. If None, uses
                              DATABASE_URL environment variable.
        """
        self._connection_string = connection_string or os.environ.get(
            "DATABASE_URL", "postgresql://localhost:5432/financial_data"
        )
        self._engine: Engine | None = None

    @property
    def engine(self) -> Engine:
        """Lazy initialization of SQLAlchemy engine."""
        if self._engine is None:
            self._engine = create_engine(self._connection_string)
        return self._engine

    def _check_connection(self) -> bool:
        """Check if database connection is available.

        Returns:
            True if connection is successful, False otherwise
        """
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    def load_prices(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        """Load adjusted close prices from the sfp table.

        Args:
            tickers: List of ticker symbols (e.g., ['SCHX', 'AGG'])
            start: Start date for price data
            end: End date for price data

        Returns:
            DataFrame with DatetimeIndex and tickers as columns,
            containing adjusted close prices

        Raises:
            ConnectionError: If database connection is not available
        """
        if not self._check_connection():
            raise ConnectionError(
                "Database connection not available. Please ensure PostgreSQL is running and "
                "DATABASE_URL environment variable is set with proper credentials."
            )

        query = text("""
            SELECT date, ticker, closeadj
            FROM sfp
            WHERE ticker = ANY(:tickers)
              AND date >= :start_date
              AND date <= :end_date
            ORDER BY date, ticker
        """)

        with self.engine.connect() as conn:
            df = pd.read_sql(
                query,
                conn,
                params={
                    "tickers": tickers,
                    "start_date": start,
                    "end_date": end,
                },
            )

        # Pivot to wide format: dates as index, tickers as columns
        if df.empty:
            return pd.DataFrame()

        pivoted = df.pivot(index="date", columns="ticker", values="closeadj")
        pivoted.index = pd.to_datetime(pivoted.index)
        pivoted = pivoted.sort_index()

        # Ensure all requested tickers are present
        missing = set(tickers) - set(pivoted.columns)
        if missing:
            raise ValueError(f"Missing tickers in database: {missing}")

        # Reorder columns to match input order
        return pivoted[tickers]

    def load_risk_free_rate(self, start: date, end: date) -> pd.Series:
        """Load daily risk-free rate from fama_french table.

        The rf column contains the daily risk-free rate in percentage terms.

        Args:
            start: Start date for the data
            end: End date for the data

        Returns:
            Series with DatetimeIndex and risk-free rate in decimal form

        Raises:
            ConnectionError: If database connection is not available
        """
        if not self._check_connection():
            raise ConnectionError(
                "Database connection not available. Please ensure PostgreSQL is running and "
                "DATABASE_URL environment variable is set with proper credentials."
            )

        query = text("""
            SELECT date, rf
            FROM fama_french
            WHERE date >= :start_date
              AND date <= :end_date
            ORDER BY date
        """)

        with self.engine.connect() as conn:
            df = pd.read_sql(
                query,
                conn,
                params={"start_date": start, "end_date": end},
            )

        if df.empty:
            return pd.Series(dtype=float)

        df["date"] = pd.to_datetime(df["date"])
        series = df.set_index("date")["rf"]
        # Convert from percentage to decimal
        return series / 100

    def load_fama_french(self, start: date, end: date) -> pd.DataFrame:
        """Load Fama-French factor data for attribution analysis.

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
            DataFrame with factors as columns, values in decimal form

        Raises:
            ConnectionError: If database connection is not available
        """
        if not self._check_connection():
            raise ConnectionError(
                "Database connection not available. Please ensure PostgreSQL is running and "
                "DATABASE_URL environment variable is set with proper credentials."
            )

        query = text("""
            SELECT date, mkt, smb, hml, rmw, cma, mom, rf
            FROM fama_french
            WHERE date >= :start_date
              AND date <= :end_date
            ORDER BY date
        """)

        with self.engine.connect() as conn:
            df = pd.read_sql(
                query,
                conn,
                params={"start_date": start, "end_date": end},
            )

        if df.empty:
            return pd.DataFrame()

        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        # Convert from percentage to decimal
        return df / 100

    def load_treasury_rates(self, start: date, end: date) -> pd.DataFrame:
        """Load Treasury yield curve data from rates table.

        Args:
            start: Start date for the data
            end: End date for the data

        Returns:
            DataFrame with various treasury maturities

        Raises:
            ConnectionError: If database connection is not available
        """
        if not self._check_connection():
            raise ConnectionError(
                "Database connection not available. Please ensure PostgreSQL is running and "
                "DATABASE_URL environment variable is set with proper credentials."
            )

        query = text("""
            SELECT *
            FROM rates
            WHERE date >= :start_date
              AND date <= :end_date
            ORDER BY date
        """)

        with self.engine.connect() as conn:
            df = pd.read_sql(
                query,
                conn,
                params={"start_date": start, "end_date": end},
            )

        if df.empty:
            return pd.DataFrame()

        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")

    def get_ticker_info(self, ticker: str) -> dict | None:
        """Get metadata for a specific ticker.

        Args:
            ticker: Ticker symbol to look up

        Returns:
            Dictionary with ticker metadata or None if not found

        Raises:
            ConnectionError: If database connection is not available
        """
        if not self._check_connection():
            raise ConnectionError(
                "Database connection not available. Please ensure PostgreSQL is running and "
                "DATABASE_URL environment variable is set with proper credentials."
            )

        query = text("""
            SELECT *
            FROM ticker_master
            WHERE ticker = :ticker
            LIMIT 1
        """)

        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"ticker": ticker})

        if df.empty:
            return None
        return df.iloc[0].to_dict()

    def get_data_range(self, ticker: str) -> tuple[date, date] | None:
        """Get the available date range for a ticker.

        Args:
            ticker: Ticker symbol to check

        Returns:
            Tuple of (min_date, max_date) or None if ticker not found

        Raises:
            ConnectionError: If database connection is not available
        """
        if not self._check_connection():
            raise ConnectionError(
                "Database connection not available. Please ensure PostgreSQL is running and "
                "DATABASE_URL environment variable is set with proper credentials."
            )

        query = text("""
            SELECT MIN(date) as min_date, MAX(date) as max_date
            FROM sfp
            WHERE ticker = :ticker
        """)

        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"ticker": ticker})

        if df.empty or df.iloc[0]["min_date"] is None:
            return None

        return (
            pd.to_datetime(df.iloc[0]["min_date"]).date(),
            pd.to_datetime(df.iloc[0]["max_date"]).date(),
        )
