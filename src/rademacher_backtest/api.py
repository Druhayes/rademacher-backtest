"""High-level convenience API for rademacher-backtest.

This module provides simple functions that wrap the core engine for common use cases.
These functions are designed to be the primary entry point for most users.

Example:
    >>> import pandas as pd
    >>> import rademacher_backtest as rbt
    >>>
    >>> # Load your data
    >>> prices = pd.DataFrame(...)
    >>> loader = rbt.DataFrameLoader(prices)
    >>>
    >>> # Run backtest
    >>> result = rbt.backtest(
    ...     portfolio={'SPY': 0.6, 'AGG': 0.4},
    ...     loader=loader,
    ...     start_date='2020-01-01',
    ...     end_date='2023-12-31'
    ... )
    >>>
    >>> # Analyze with RAS
    >>> ras_result = rbt.analyze_ras(result.daily_returns)
    >>> print(ras_result.interpretation)
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from rademacher_backtest.backtest.engine import BacktestEngine
from rademacher_backtest.config.portfolio import AssetAllocation, PortfolioDefinition
from rademacher_backtest.config.settings import BacktestConfig, RASConfig
from rademacher_backtest.ras.bounds import BoundsCalculator
from rademacher_backtest.ras.report import RASReportGenerator

if TYPE_CHECKING:
    from rademacher_backtest.backtest.results import BacktestResult
    from rademacher_backtest.data.loader import DataLoader
    from rademacher_backtest.ras.report import RASReport


def backtest(
    portfolio: PortfolioDefinition | dict[str, float],
    loader: DataLoader,
    start_date: date | str,
    end_date: date | str,
    initial_capital: float = 100_000.0,
    transaction_cost_bps: float = 10.0,
    rebalance_frequency: str = "monthly",
) -> BacktestResult:
    """Run a backtest with sensible defaults.

    This is a simplified interface to the backtest engine. It handles
    common tasks like converting string dates and dict portfolios.

    Args:
        portfolio: Portfolio definition or dict of {ticker: weight}.
                  Weights should sum to 1.0.
        loader: Data loader (DataFrameLoader, CSVLoader, PostgreSQLLoader, etc.)
        start_date: Backtest start date (date object or 'YYYY-MM-DD' string)
        end_date: Backtest end date (date object or 'YYYY-MM-DD' string)
        initial_capital: Starting capital in dollars (default: $100,000)
        transaction_cost_bps: Transaction cost in basis points (default: 10 = 0.1%)
        rebalance_frequency: How often to rebalance ('monthly', 'quarterly', etc.)

    Returns:
        BacktestResult with all performance data

    Example:
        >>> import pandas as pd
        >>> import rademacher_backtest as rbt
        >>>
        >>> prices = pd.DataFrame({
        ...     'SPY': [100, 101, 102, 103],
        ...     'AGG': [50, 50.1, 50.2, 50.3]
        ... }, index=pd.date_range('2020-01-01', periods=4))
        >>>
        >>> loader = rbt.DataFrameLoader(prices)
        >>> result = rbt.backtest(
        ...     portfolio={'SPY': 0.6, 'AGG': 0.4},
        ...     loader=loader,
        ...     start_date='2020-01-01',
        ...     end_date='2020-01-04'
        ... )
        >>> print(f"Final value: ${result.final_value:,.2f}")
    """
    # Convert dict to PortfolioDefinition if needed
    if isinstance(portfolio, dict):
        portfolio = create_portfolio(portfolio)

    # Convert string dates to date objects if needed
    if isinstance(start_date, str):
        start_date = date.fromisoformat(start_date)
    if isinstance(end_date, str):
        end_date = date.fromisoformat(end_date)

    # Create config
    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        transaction_cost_bps=transaction_cost_bps,
        rebalance_frequency=rebalance_frequency,  # type: ignore
    )

    # Create engine and run
    engine = BacktestEngine(config, portfolio, loader=loader)
    return engine.run()


def analyze_ras(
    returns: pd.Series | np.ndarray,
    confidence: float = 0.99,
    n_simulations: int = 20_000,
    n_strategies: int = 1,
) -> RASReport:
    """Analyze returns using RAS (Rademacher Anti-Serum) methodology.

    This function provides statistically rigorous bounds on the Sharpe ratio
    by accounting for data snooping bias and multiple testing.

    Args:
        returns: Daily returns series or array
        confidence: Confidence level (0.99 = 99%, 0.95 = 95%, etc.)
        n_simulations: Number of Monte Carlo simulations for complexity estimation
        n_strategies: Number of strategies tested (for multiple testing correction)

    Returns:
        RASReport with statistical analysis including:
        - Empirical Sharpe ratio
        - RAS-adjusted Sharpe ratio (lower bound)
        - Statistical significance test
        - Detailed interpretation

    Example:
        >>> import numpy as np
        >>> import rademacher_backtest as rbt
        >>>
        >>> # Generate sample returns
        >>> np.random.seed(42)
        >>> returns = np.random.normal(0.001, 0.02, 252)  # 1 year daily
        >>>
        >>> # Analyze with RAS
        >>> ras_result = rbt.analyze_ras(returns, confidence=0.99)
        >>> print(f"Empirical Sharpe: {ras_result.empirical_sharpe_annualized:.3f}")
        >>> print(f"Adjusted Sharpe: {ras_result.adjusted_sharpe_annualized:.3f}")
        >>> print(f"Statistically positive: {ras_result.is_statistically_positive}")
    """
    # Convert to numpy array if Series
    if isinstance(returns, pd.Series):
        returns_array = returns.values
    else:
        returns_array = np.asarray(returns)

    # Create RAS config
    ras_config = RASConfig(
        delta=1 - confidence,
        n_simulations=n_simulations,
    )

    # Calculate bounds
    bounds_calc = BoundsCalculator(ras_config)
    bounds = bounds_calc.calculate_sharpe_bounds(returns_array)

    # Generate report
    report_gen = RASReportGenerator()
    return report_gen.generate(bounds, len(returns_array), N=n_strategies)


def create_portfolio(
    allocations: dict[str, float],
    name: str = "Custom Portfolio",
) -> PortfolioDefinition:
    """Create a portfolio from a simple dict of allocations.

    This is a convenience function to create PortfolioDefinition objects
    from dictionaries without manually creating AssetAllocation tuples.

    Args:
        allocations: Dictionary mapping ticker to weight
                    Example: {'SPY': 0.6, 'AGG': 0.4}
        name: Portfolio name (default: "Custom Portfolio")

    Returns:
        PortfolioDefinition ready to use with backtest engine

    Raises:
        ValueError: If weights don't sum to approximately 1.0

    Example:
        >>> import rademacher_backtest as rbt
        >>>
        >>> portfolio = rbt.create_portfolio({
        ...     'SPY': 0.60,
        ...     'AGG': 0.40
        ... }, name="60/40 Portfolio")
        >>>
        >>> print(portfolio.name)  # "60/40 Portfolio"
        >>> print(portfolio.tickers)  # ['SPY', 'AGG']
    """
    # Validate weights sum to 1
    total_weight = sum(allocations.values())
    if not (0.99 <= total_weight <= 1.01):
        raise ValueError(
            f"Portfolio weights must sum to 1.0, got {total_weight:.4f}. "
            f"Allocations: {allocations}"
        )

    # Create AssetAllocation tuples
    asset_allocations = tuple(
        AssetAllocation(
            ticker=ticker,
            weight=weight,
            name=ticker,  # Use ticker as name by default
        )
        for ticker, weight in allocations.items()
    )

    return PortfolioDefinition(
        name=name,
        allocations=asset_allocations,
    )
