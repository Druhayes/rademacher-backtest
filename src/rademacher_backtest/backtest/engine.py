"""Core backtesting engine implementation."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import pandas as pd

from rademacher_backtest.backtest.results import BacktestResult, DailySnapshot
from rademacher_backtest.data.loader import DataLoader
from rademacher_backtest.data.preprocessor import DataPreprocessor
from rademacher_backtest.portfolio.portfolio import Holdings
from rademacher_backtest.portfolio.rebalancer import MonthlyRebalancer

if TYPE_CHECKING:
    from rademacher_backtest.config.portfolio import PortfolioDefinition
    from rademacher_backtest.config.settings import BacktestConfig


class BacktestEngine:
    """Core backtesting engine with monthly rebalancing.

    Simulates a portfolio over historical data with periodic rebalancing
    and transaction cost modeling.
    """

    def __init__(
        self,
        config: BacktestConfig,
        portfolio: PortfolioDefinition,
        loader: DataLoader | None = None,
    ) -> None:
        """Initialize the backtest engine.

        Args:
            config: Backtest configuration
            portfolio: Portfolio definition with target weights
            loader: Data loader instance (DataFrameLoader, CSVLoader, PostgreSQLLoader, etc.)

        Raises:
            ValueError: If loader is None
        """
        self.config = config
        self.portfolio = portfolio
        if loader is None:
            raise ValueError(
                "A data loader is required. Create a DataFrameLoader, CSVLoader, or "
                "PostgreSQLLoader and pass it to BacktestEngine.\n"
                "Example: engine = BacktestEngine(config, portfolio, loader=DataFrameLoader(prices_df))"
            )
        self.loader = loader
        self.preprocessor = DataPreprocessor()

    def run(self) -> BacktestResult:
        """Execute the backtest.

        Returns:
            BacktestResult containing all backtest data
        """
        # 1. Load price data
        tickers = self.portfolio.tickers
        prices = self.loader.load_prices(
            tickers,
            self.config.start_date,
            self.config.end_date,
        )

        if prices.empty:
            raise ValueError("No price data loaded")

        # 2. Initialize rebalancer
        rebalancer = MonthlyRebalancer(
            target_weights=self.portfolio.weights,
            cost_bps=self.config.transaction_cost_bps,
        )

        # 3. Get rebalance dates
        rebalance_dates = self.preprocessor.get_rebalance_dates(
            self.config.start_date,
            self.config.end_date,
            self.config.rebalance_frequency,
        )
        rebalance_dates_set = set(rebalance_dates)

        # 4. Initialize portfolio
        first_prices = prices.iloc[0].to_dict()
        holdings = Holdings.from_weights(
            capital=self.config.initial_capital,
            weights=self.portfolio.weights,
            prices=first_prices,
        )

        # 5. Simulate day by day
        daily_values: dict[date, float] = {}
        transaction_costs: dict[date, float] = {}
        snapshots: list[DailySnapshot] = []
        actual_rebalance_dates: list[date] = []

        for idx, row in prices.iterrows():
            current_date = idx.date() if hasattr(idx, "date") else idx
            current_prices = row.to_dict()

            # Check if rebalance day
            if current_date in rebalance_dates_set:
                holdings, _, cost = rebalancer.rebalance(holdings, current_prices)
                transaction_costs[current_date] = cost
                actual_rebalance_dates.append(current_date)

            # Record daily value
            portfolio_value = holdings.total_value(current_prices)
            daily_values[current_date] = portfolio_value

            # Create snapshot
            snapshots.append(
                DailySnapshot(
                    date=current_date,
                    portfolio_value=portfolio_value,
                    holdings=holdings.copy(),
                    weights=holdings.weights(current_prices),
                    cash=holdings.cash,
                )
            )

        # 6. Calculate returns
        values_series = pd.Series(daily_values)
        values_series.index = pd.to_datetime(values_series.index)
        values_series = values_series.sort_index()

        daily_returns = values_series.pct_change().dropna()
        monthly_returns = values_series.resample("ME").last().pct_change().dropna()

        # 7. Build transaction costs series
        costs_series = pd.Series(transaction_costs)
        if not costs_series.empty:
            costs_series.index = pd.to_datetime(costs_series.index)
            costs_series = costs_series.sort_index()
        else:
            costs_series = pd.Series(dtype=float)

        return BacktestResult(
            config=self.config,
            portfolio_def=self.portfolio,
            daily_values=values_series,
            daily_returns=daily_returns,
            monthly_returns=monthly_returns,
            rebalance_dates=actual_rebalance_dates,
            transaction_costs=costs_series,
            total_transaction_costs=sum(transaction_costs.values()),
            snapshots=snapshots,
        )

    def run_buy_and_hold(self) -> BacktestResult:
        """Run a buy-and-hold backtest (no rebalancing).

        Useful for comparison with rebalanced portfolio.

        Returns:
            BacktestResult for buy-and-hold strategy
        """
        # Load price data
        tickers = self.portfolio.tickers
        prices = self.loader.load_prices(
            tickers,
            self.config.start_date,
            self.config.end_date,
        )

        if prices.empty:
            raise ValueError("No price data loaded")

        # Initialize portfolio once
        first_prices = prices.iloc[0].to_dict()
        holdings = Holdings.from_weights(
            capital=self.config.initial_capital,
            weights=self.portfolio.weights,
            prices=first_prices,
        )

        # Simulate without rebalancing
        daily_values: dict[date, float] = {}
        snapshots: list[DailySnapshot] = []

        for idx, row in prices.iterrows():
            current_date = idx.date() if hasattr(idx, "date") else idx
            current_prices = row.to_dict()

            portfolio_value = holdings.total_value(current_prices)
            daily_values[current_date] = portfolio_value

            snapshots.append(
                DailySnapshot(
                    date=current_date,
                    portfolio_value=portfolio_value,
                    holdings=holdings,  # Same holdings throughout
                    weights=holdings.weights(current_prices),
                    cash=holdings.cash,
                )
            )

        # Calculate returns
        values_series = pd.Series(daily_values)
        values_series.index = pd.to_datetime(values_series.index)
        values_series = values_series.sort_index()

        daily_returns = values_series.pct_change().dropna()
        monthly_returns = values_series.resample("ME").last().pct_change().dropna()

        return BacktestResult(
            config=self.config,
            portfolio_def=self.portfolio,
            daily_values=values_series,
            daily_returns=daily_returns,
            monthly_returns=monthly_returns,
            rebalance_dates=[],
            transaction_costs=pd.Series(dtype=float),
            total_transaction_costs=0.0,
            snapshots=snapshots,
        )


class BacktestRunner:
    """High-level backtest orchestrator.

    Provides a simplified interface for running backtests with
    sensible defaults.
    """

    def __init__(
        self,
        config: BacktestConfig,
        portfolio: PortfolioDefinition,
        loader: DataLoader,
    ) -> None:
        """Initialize the runner.

        Args:
            config: Backtest configuration
            portfolio: Portfolio definition
            loader: Data loader instance

        Raises:
            ValueError: If loader is None
        """
        self.config = config
        self.portfolio = portfolio
        self.engine = BacktestEngine(config, portfolio, loader=loader)

    def run(self) -> BacktestResult:
        """Run the backtest.

        Returns:
            BacktestResult containing all backtest data
        """
        return self.engine.run()

    def run_comparison(self) -> tuple[BacktestResult, BacktestResult]:
        """Run both rebalanced and buy-and-hold backtests.

        Returns:
            Tuple of (rebalanced_result, buy_and_hold_result)
        """
        rebalanced = self.engine.run()
        buy_hold = self.engine.run_buy_and_hold()
        return rebalanced, buy_hold
