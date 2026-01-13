"""Tests for backtest engine."""

from __future__ import annotations

from datetime import date
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from rademacher_backtest.backtest.engine import BacktestEngine, BacktestRunner
from rademacher_backtest.config.portfolio import SAMPLE_PORTFOLIO
from rademacher_backtest.config.settings import BacktestConfig


class TestBacktestEngine:
    """Test suite for BacktestEngine."""

    def test_initialization_requires_loader(
        self,
        backtest_config: BacktestConfig,
        simple_portfolio,
    ) -> None:
        """Test that loader is required."""
        with pytest.raises(ValueError, match="data loader is required"):
            BacktestEngine(backtest_config, simple_portfolio, loader=None)

    def test_initialization_with_loader(
        self,
        backtest_config: BacktestConfig,
        simple_portfolio,
    ) -> None:
        """Test engine initialization with loader."""
        mock_loader = Mock()
        engine = BacktestEngine(backtest_config, simple_portfolio, loader=mock_loader)

        assert engine.config == backtest_config
        assert engine.portfolio == simple_portfolio
        assert engine.loader is mock_loader
        assert engine.preprocessor is not None

    def test_initialization_custom_loader(
        self,
        backtest_config: BacktestConfig,
        simple_portfolio,
    ) -> None:
        """Test engine initialization with custom loader."""
        custom_loader = Mock()
        engine = BacktestEngine(backtest_config, simple_portfolio, loader=custom_loader)

        assert engine.loader is custom_loader

    def test_run_with_mock_data(
        self,
        simple_portfolio,
        sample_prices: pd.DataFrame,
    ) -> None:
        """Test backtest run with mocked price data."""
        # Use first two tickers from sample prices
        tickers = ["SCHX", "AGG"]
        prices = sample_prices[tickers].head(100)

        # Create mock loader
        mock_loader = Mock()
        mock_loader.load_prices.return_value = prices

        config = BacktestConfig(
            start_date=prices.index[0].date(),
            end_date=prices.index[-1].date(),
            initial_capital=100_000.0,
            transaction_cost_bps=10.0,
            rebalance_frequency="monthly",
        )

        engine = BacktestEngine(config, simple_portfolio, loader=mock_loader)
        result = engine.run()

        # Check result structure
        assert result.config == config
        assert result.portfolio_def == simple_portfolio
        assert len(result.daily_values) > 0
        assert len(result.daily_returns) > 0
        assert len(result.snapshots) == len(prices)

        # Check that loader was called correctly
        mock_loader.load_prices.assert_called_once()

        # Check portfolio value progression
        assert result.daily_values.iloc[0] > 0
        assert result.daily_values.iloc[-1] > 0

        # Check that transaction costs were recorded
        assert result.total_transaction_costs >= 0

    def test_run_buy_and_hold(
        self,
        simple_portfolio,
        sample_prices: pd.DataFrame,
    ) -> None:
        """Test buy-and-hold backtest (no rebalancing)."""
        tickers = ["SCHX", "AGG"]
        prices = sample_prices[tickers].head(100)

        mock_loader = Mock()
        mock_loader.load_prices.return_value = prices

        config = BacktestConfig(
            start_date=prices.index[0].date(),
            end_date=prices.index[-1].date(),
            initial_capital=100_000.0,
        )

        engine = BacktestEngine(config, simple_portfolio, loader=mock_loader)
        result = engine.run_buy_and_hold()

        # Should have no rebalancing
        assert len(result.rebalance_dates) == 0
        assert result.total_transaction_costs == 0.0

        # Should still have daily values
        assert len(result.daily_values) == len(prices)

    def test_rebalancing_occurs(
        self,
        simple_portfolio,
        sample_prices: pd.DataFrame,
    ) -> None:
        """Test that rebalancing occurs at expected times."""
        tickers = ["SCHX", "AGG"]

        # Use 6 months of data to ensure multiple rebalance dates
        prices = sample_prices[tickers].head(126)  # ~6 months

        mock_loader = Mock()
        mock_loader.load_prices.return_value = prices

        config = BacktestConfig(
            start_date=prices.index[0].date(),
            end_date=prices.index[-1].date(),
            initial_capital=100_000.0,
            rebalance_frequency="monthly",
        )

        engine = BacktestEngine(config, simple_portfolio, loader=mock_loader)
        result = engine.run()

        # Should have rebalanced at least once (likely 5-6 times for 6 months)
        assert len(result.rebalance_dates) >= 1

        # Transaction costs should be positive
        assert result.total_transaction_costs > 0

    def test_transaction_costs_accumulate(
        self,
        simple_portfolio,
        sample_prices: pd.DataFrame,
    ) -> None:
        """Test that transaction costs accumulate correctly."""
        tickers = ["SCHX", "AGG"]
        prices = sample_prices[tickers].head(100)

        mock_loader = Mock()
        mock_loader.load_prices.return_value = prices

        config = BacktestConfig(
            start_date=prices.index[0].date(),
            end_date=prices.index[-1].date(),
            initial_capital=100_000.0,
            transaction_cost_bps=10.0,
            rebalance_frequency="monthly",
        )

        engine = BacktestEngine(config, simple_portfolio, loader=mock_loader)
        result = engine.run()

        # Total costs should equal sum of individual costs
        if len(result.transaction_costs) > 0:
            assert np.isclose(
                result.total_transaction_costs,
                result.transaction_costs.sum()
            )

    def test_empty_price_data_raises_error(
        self,
        backtest_config: BacktestConfig,
        simple_portfolio,
    ) -> None:
        """Test that empty price data raises error."""
        mock_loader = Mock()
        mock_loader.load_prices.return_value = pd.DataFrame()

        engine = BacktestEngine(backtest_config, simple_portfolio, loader=mock_loader)

        with pytest.raises(ValueError, match="No price data loaded"):
            engine.run()

    def test_snapshots_track_holdings(
        self,
        simple_portfolio,
        sample_prices: pd.DataFrame,
    ) -> None:
        """Test that snapshots correctly track holdings over time."""
        tickers = ["SCHX", "AGG"]
        prices = sample_prices[tickers].head(50)

        mock_loader = Mock()
        mock_loader.load_prices.return_value = prices

        config = BacktestConfig(
            start_date=prices.index[0].date(),
            end_date=prices.index[-1].date(),
            initial_capital=100_000.0,
        )

        engine = BacktestEngine(config, simple_portfolio, loader=mock_loader)
        result = engine.run()

        # Check first snapshot
        first_snapshot = result.snapshots[0]
        assert first_snapshot.date == prices.index[0].date()
        assert first_snapshot.portfolio_value > 0
        assert len(first_snapshot.holdings.shares) > 0

        # Check that weights sum to approximately 1 (plus cash)
        total_weight = sum(first_snapshot.weights.values())
        cash_weight = first_snapshot.cash / first_snapshot.portfolio_value
        assert np.isclose(total_weight + cash_weight, 1.0, atol=0.01)


class TestBacktestRunner:
    """Test suite for BacktestRunner."""

    def test_initialization(
        self,
        backtest_config: BacktestConfig,
        simple_portfolio,
    ) -> None:
        """Test runner initialization."""
        mock_loader = Mock()
        runner = BacktestRunner(backtest_config, simple_portfolio, loader=mock_loader)

        assert runner.config == backtest_config
        assert runner.portfolio == simple_portfolio
        assert runner.engine is not None

    def test_run(
        self,
        simple_portfolio,
        sample_prices: pd.DataFrame,
    ) -> None:
        """Test runner run method."""
        tickers = ["SCHX", "AGG"]
        prices = sample_prices[tickers].head(100)

        mock_loader = Mock()
        mock_loader.load_prices.return_value = prices

        config = BacktestConfig(
            start_date=prices.index[0].date(),
            end_date=prices.index[-1].date(),
            initial_capital=100_000.0,
        )

        runner = BacktestRunner(config, simple_portfolio, loader=mock_loader)

        result = runner.run()

        assert result is not None
        assert len(result.daily_values) > 0

    def test_run_comparison(
        self,
        simple_portfolio,
        sample_prices: pd.DataFrame,
    ) -> None:
        """Test comparison between rebalanced and buy-and-hold."""
        tickers = ["SCHX", "AGG"]
        prices = sample_prices[tickers].head(100)

        mock_loader = Mock()
        mock_loader.load_prices.return_value = prices

        config = BacktestConfig(
            start_date=prices.index[0].date(),
            end_date=prices.index[-1].date(),
            initial_capital=100_000.0,
            rebalance_frequency="monthly",
        )

        runner = BacktestRunner(config, simple_portfolio, loader=mock_loader)

        rebalanced, buy_hold = runner.run_comparison()

        # Both should have results
        assert len(rebalanced.daily_values) > 0
        assert len(buy_hold.daily_values) > 0

        # Rebalanced should have transaction costs
        assert rebalanced.total_transaction_costs > 0

        # Buy-and-hold should have no transaction costs
        assert buy_hold.total_transaction_costs == 0

        # Same starting value
        assert np.isclose(
            rebalanced.daily_values.iloc[0],
            buy_hold.daily_values.iloc[0]
        )


@pytest.mark.integration
@pytest.mark.postgres
class TestBacktestEngineIntegration:
    """Integration tests using real PostgreSQL database."""

    def test_real_database_backtest(self) -> None:
        """Test backtest with real database connection."""
        # Import PostgreSQL loader (requires postgres extra)
        from rademacher_backtest.data.postgres_loader import PostgreSQLLoader
        from rademacher_backtest.config.portfolio import AssetAllocation, PortfolioDefinition

        portfolio = PortfolioDefinition(
            name="Test Portfolio",
            allocations=(
                AssetAllocation("SCHX", 0.60, "US Large Cap"),
                AssetAllocation("AGG", 0.40, "Aggregate Bond"),
            ),
        )

        # Short backtest period (1 year)
        config = BacktestConfig(
            start_date=date(2020, 1, 1),
            end_date=date(2020, 12, 31),
            initial_capital=100_000.0,
            transaction_cost_bps=10.0,
            rebalance_frequency="monthly",
        )

        # Create engine with real database loader
        loader = PostgreSQLLoader()
        engine = BacktestEngine(config, portfolio, loader=loader)

        # Run backtest
        result = engine.run()

        # Verify results
        assert len(result.daily_values) > 200  # At least 200 trading days
        assert len(result.daily_returns) > 0
        assert len(result.rebalance_dates) > 0  # Should have rebalanced

        # Check performance metrics are reasonable
        total_return = (result.daily_values.iloc[-1] / result.daily_values.iloc[0]) - 1
        assert -0.5 < total_return < 1.0  # Reasonable bounds for 1 year

        # Check transaction costs
        assert result.total_transaction_costs > 0
        assert result.total_transaction_costs < config.initial_capital * 0.01  # Less than 1% of capital

    def test_real_database_multiple_assets(self) -> None:
        """Test backtest with all 5 ETFs from database."""
        from rademacher_backtest.data.postgres_loader import PostgreSQLLoader

        # Use sample portfolio (5 assets)
        config = BacktestConfig(
            start_date=date(2015, 1, 1),
            end_date=date(2015, 12, 31),
            initial_capital=100_000.0,
            rebalance_frequency="monthly",
        )

        loader = PostgreSQLLoader()
        engine = BacktestEngine(config, SAMPLE_PORTFOLIO, loader=loader)
        result = engine.run()

        # Should have data for all assets
        assert len(result.daily_values) > 200

        # Check that all assets are represented in snapshots
        first_snapshot = result.snapshots[0]
        assert len(first_snapshot.holdings.shares) == 5

        # Weights should approximately match target
        target_weights = SAMPLE_PORTFOLIO.weights
        actual_weights = first_snapshot.weights

        for ticker in target_weights:
            # Allow some deviation due to discrete shares and cash
            assert abs(actual_weights[ticker] - target_weights[ticker]) < 0.05

    def test_real_database_buy_and_hold_vs_rebalanced(self) -> None:
        """Test buy-and-hold vs rebalanced with real data."""
        from rademacher_backtest.data.postgres_loader import PostgreSQLLoader
        from rademacher_backtest.config.portfolio import AssetAllocation, PortfolioDefinition

        portfolio = PortfolioDefinition(
            name="Test Portfolio",
            allocations=(
                AssetAllocation("SCHX", 0.60, "US Large Cap"),
                AssetAllocation("AGG", 0.40, "Aggregate Bond"),
            ),
        )

        config = BacktestConfig(
            start_date=date(2018, 1, 1),
            end_date=date(2018, 12, 31),
            initial_capital=100_000.0,
            rebalance_frequency="monthly",
        )

        loader = PostgreSQLLoader()
        engine = BacktestEngine(config, portfolio, loader=loader)

        rebalanced = engine.run()
        buy_hold = engine.run_buy_and_hold()

        # Both should have same length
        assert len(rebalanced.daily_values) == len(buy_hold.daily_values)

        # Rebalanced has costs
        assert rebalanced.total_transaction_costs > 0

        # Buy-and-hold has no costs
        assert buy_hold.total_transaction_costs == 0

        # Performance may differ (either could be better)
        rebalanced_return = (
            rebalanced.daily_values.iloc[-1] / rebalanced.daily_values.iloc[0]
        ) - 1
        buy_hold_return = (
            buy_hold.daily_values.iloc[-1] / buy_hold.daily_values.iloc[0]
        ) - 1

        # Both should be reasonable
        assert -0.5 < rebalanced_return < 1.0
        assert -0.5 < buy_hold_return < 1.0

    def test_real_database_different_rebalance_frequencies(self) -> None:
        """Test different rebalancing frequencies."""
        from rademacher_backtest.data.postgres_loader import PostgreSQLLoader
        from rademacher_backtest.config.portfolio import AssetAllocation, PortfolioDefinition

        portfolio = PortfolioDefinition(
            name="Test Portfolio",
            allocations=(
                AssetAllocation("SCHX", 0.50, "US Large Cap"),
                AssetAllocation("AGG", 0.50, "Aggregate Bond"),
            ),
        )

        base_config = BacktestConfig(
            start_date=date(2019, 1, 1),
            end_date=date(2019, 12, 31),
            initial_capital=100_000.0,
        )

        # Monthly rebalancing
        config_monthly = BacktestConfig(
            start_date=base_config.start_date,
            end_date=base_config.end_date,
            initial_capital=base_config.initial_capital,
            rebalance_frequency="monthly",
        )

        # Quarterly rebalancing
        config_quarterly = BacktestConfig(
            start_date=base_config.start_date,
            end_date=base_config.end_date,
            initial_capital=base_config.initial_capital,
            rebalance_frequency="quarterly",
        )

        loader = PostgreSQLLoader()
        engine_monthly = BacktestEngine(config_monthly, portfolio, loader=loader)
        engine_quarterly = BacktestEngine(config_quarterly, portfolio, loader=loader)

        result_monthly = engine_monthly.run()
        result_quarterly = engine_quarterly.run()

        # Monthly should rebalance more often
        assert len(result_monthly.rebalance_dates) > len(result_quarterly.rebalance_dates)

        # Monthly should have higher transaction costs
        assert result_monthly.total_transaction_costs > result_quarterly.total_transaction_costs

    def test_real_database_transaction_cost_impact(self) -> None:
        """Test impact of transaction costs on performance."""
        from rademacher_backtest.data.postgres_loader import PostgreSQLLoader
        from rademacher_backtest.config.portfolio import AssetAllocation, PortfolioDefinition

        portfolio = PortfolioDefinition(
            name="Test Portfolio",
            allocations=(
                AssetAllocation("SCHX", 0.60, "US Large Cap"),
                AssetAllocation("AGG", 0.40, "Aggregate Bond"),
            ),
        )

        # No transaction costs
        config_no_cost = BacktestConfig(
            start_date=date(2017, 1, 1),
            end_date=date(2017, 12, 31),
            initial_capital=100_000.0,
            transaction_cost_bps=0.0,
            rebalance_frequency="monthly",
        )

        # With transaction costs
        config_with_cost = BacktestConfig(
            start_date=date(2017, 1, 1),
            end_date=date(2017, 12, 31),
            initial_capital=100_000.0,
            transaction_cost_bps=10.0,
            rebalance_frequency="monthly",
        )

        loader = PostgreSQLLoader()
        engine_no_cost = BacktestEngine(config_no_cost, portfolio, loader=loader)
        engine_with_cost = BacktestEngine(config_with_cost, portfolio, loader=loader)

        result_no_cost = engine_no_cost.run()
        result_with_cost = engine_with_cost.run()

        # No cost version should have zero costs
        assert result_no_cost.total_transaction_costs == 0.0

        # With cost version should have positive costs
        assert result_with_cost.total_transaction_costs > 0

        # No cost version should have slightly better final value
        # (all else being equal)
        assert result_no_cost.daily_values.iloc[-1] >= result_with_cost.daily_values.iloc[-1]
