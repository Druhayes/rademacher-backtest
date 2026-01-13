"""Pytest configuration and fixtures."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from rademacher_backtest.config.portfolio import SAMPLE_PORTFOLIO, AssetAllocation, PortfolioDefinition
from rademacher_backtest.config.settings import BacktestConfig, RASConfig


@pytest.fixture
def sample_portfolio() -> PortfolioDefinition:
    """Return the sample 60/40 portfolio."""
    return SAMPLE_PORTFOLIO


@pytest.fixture
def simple_portfolio() -> PortfolioDefinition:
    """Return a simple 2-asset portfolio for testing."""
    return PortfolioDefinition(
        name="Simple Test Portfolio",
        allocations=(
            AssetAllocation("SCHX", 0.60, "US Large Cap"),
            AssetAllocation("AGG", 0.40, "Aggregate Bond"),
        ),
    )


@pytest.fixture
def backtest_config() -> BacktestConfig:
    """Return default backtest configuration."""
    return BacktestConfig(
        start_date=date(2020, 1, 1),
        end_date=date(2023, 12, 31),
        initial_capital=100_000.0,
        transaction_cost_bps=10.0,
    )


@pytest.fixture
def ras_config() -> RASConfig:
    """Return default RAS configuration."""
    return RASConfig(
        delta=0.01,
        n_simulations=5000,  # Fewer for faster tests
        random_seed=42,
    )


@pytest.fixture
def sample_daily_returns() -> pd.Series:
    """Generate sample daily returns for testing."""
    np.random.seed(42)
    n_days = 1000
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")

    # Generate returns with realistic properties
    # Daily mean ~0.0003 (7.5% annual), std ~0.01 (16% annual)
    returns = np.random.normal(0.0003, 0.01, n_days)

    return pd.Series(returns, index=dates, name="returns")


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    """Generate sample price data for multiple assets."""
    np.random.seed(42)
    n_days = 1000
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")

    tickers = ["SCHX", "SCHA", "SCHF", "SCHE", "AGG"]

    # Generate correlated returns
    base_returns = np.random.normal(0.0003, 0.008, n_days)

    prices = {}
    for i, ticker in enumerate(tickers):
        # Add some idiosyncratic noise
        ticker_returns = base_returns + np.random.normal(0, 0.005, n_days)
        prices[ticker] = 100 * np.cumprod(1 + ticker_returns)

    return pd.DataFrame(prices, index=dates)


@pytest.fixture
def sample_returns_matrix() -> np.ndarray:
    """Generate sample returns matrix for RAS testing."""
    np.random.seed(42)
    T, N = 2000, 10

    # Generate correlated returns
    correlation = 0.3
    common_factor = np.random.randn(T)
    returns = np.zeros((T, N))

    for n in range(N):
        idiosyncratic = np.random.randn(T)
        returns[:, n] = np.sqrt(correlation) * common_factor + np.sqrt(1 - correlation) * idiosyncratic

    # Standardize
    returns = (returns - returns.mean(axis=0)) / returns.std(axis=0, ddof=1)

    return returns
