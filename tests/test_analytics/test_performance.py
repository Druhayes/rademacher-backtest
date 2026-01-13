"""Tests for performance metrics calculation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rademacher_backtest.analytics.performance import PerformanceCalculator, PerformanceMetrics


class TestPerformanceMetrics:
    """Test suite for PerformanceMetrics dataclass."""

    def test_initialization(self) -> None:
        """Test metrics initialization."""
        metrics = PerformanceMetrics(
            total_return=0.50,
            cagr=0.12,
            best_year=0.25,
            worst_year=-0.10,
            best_month=0.08,
            worst_month=-0.05,
            sharpe_ratio=1.2,
            sortino_ratio=1.5,
            calmar_ratio=1.0,
            annualized_volatility=0.12,
            downside_volatility=0.08,
            max_drawdown=-0.15,
            max_drawdown_duration_days=120,
            average_drawdown=-0.05,
            skewness=-0.3,
            kurtosis=2.5,
            var_95=-0.02,
            cvar_95=-0.03,
            positive_periods=600,
            negative_periods=400,
            win_rate=0.60,
            avg_win=0.005,
            avg_loss=-0.003,
            win_loss_ratio=1.67,
        )

        assert metrics.total_return == 0.50
        assert metrics.sharpe_ratio == 1.2

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        metrics = PerformanceMetrics(
            total_return=0.50,
            cagr=0.12,
            best_year=0.25,
            worst_year=-0.10,
            best_month=0.08,
            worst_month=-0.05,
            sharpe_ratio=1.2,
            sortino_ratio=1.5,
            calmar_ratio=1.0,
            annualized_volatility=0.12,
            downside_volatility=0.08,
            max_drawdown=-0.15,
            max_drawdown_duration_days=120,
            average_drawdown=-0.05,
            skewness=-0.3,
            kurtosis=2.5,
            var_95=-0.02,
            cvar_95=-0.03,
            positive_periods=600,
            negative_periods=400,
            win_rate=0.60,
            avg_win=0.005,
            avg_loss=-0.003,
            win_loss_ratio=1.67,
        )

        result_dict = metrics.to_dict()

        # Check that values are formatted correctly
        assert result_dict["Total Return (%)"] == 50.0
        assert result_dict["CAGR (%)"] == 12.0
        assert result_dict["Sharpe Ratio"] == 1.2

    def test_summary_lines(self) -> None:
        """Test summary lines generation."""
        metrics = PerformanceMetrics(
            total_return=0.50,
            cagr=0.12,
            best_year=0.25,
            worst_year=-0.10,
            best_month=0.08,
            worst_month=-0.05,
            sharpe_ratio=1.2,
            sortino_ratio=1.5,
            calmar_ratio=1.0,
            annualized_volatility=0.12,
            downside_volatility=0.08,
            max_drawdown=-0.15,
            max_drawdown_duration_days=120,
            average_drawdown=-0.05,
            skewness=-0.3,
            kurtosis=2.5,
            var_95=-0.02,
            cvar_95=-0.03,
            positive_periods=600,
            negative_periods=400,
            win_rate=0.60,
            avg_win=0.005,
            avg_loss=-0.003,
            win_loss_ratio=1.67,
        )

        lines = metrics.summary_lines()

        # Should have multiple lines
        assert len(lines) > 20

        # Should contain key sections
        summary_text = "\n".join(lines)
        assert "PERFORMANCE METRICS" in summary_text
        assert "RETURNS" in summary_text
        assert "RISK-ADJUSTED" in summary_text

    def test_str_representation(self) -> None:
        """Test string representation."""
        metrics = PerformanceMetrics(
            total_return=0.50,
            cagr=0.12,
            best_year=0.25,
            worst_year=-0.10,
            best_month=0.08,
            worst_month=-0.05,
            sharpe_ratio=1.2,
            sortino_ratio=1.5,
            calmar_ratio=1.0,
            annualized_volatility=0.12,
            downside_volatility=0.08,
            max_drawdown=-0.15,
            max_drawdown_duration_days=120,
            average_drawdown=-0.05,
            skewness=-0.3,
            kurtosis=2.5,
            var_95=-0.02,
            cvar_95=-0.03,
            positive_periods=600,
            negative_periods=400,
            win_rate=0.60,
            avg_win=0.005,
            avg_loss=-0.003,
            win_loss_ratio=1.67,
        )

        str_repr = str(metrics)

        # Should be the same as summary_lines joined
        assert str_repr == "\n".join(metrics.summary_lines())


class TestPerformanceCalculator:
    """Test suite for PerformanceCalculator."""

    def test_initialization_default(self) -> None:
        """Test calculator initialization with defaults."""
        calc = PerformanceCalculator()

        assert calc.rf is None
        assert calc.periods_per_year == 252

    def test_initialization_custom(self) -> None:
        """Test calculator initialization with custom parameters."""
        rf = pd.Series([0.0001] * 100)
        calc = PerformanceCalculator(risk_free_rate=rf, periods_per_year=12)

        assert calc.rf is not None
        assert calc.periods_per_year == 12

    def test_calculate_basic(self, sample_daily_returns: pd.Series) -> None:
        """Test basic metrics calculation."""
        calc = PerformanceCalculator()
        metrics = calc.calculate(sample_daily_returns)

        # Check that all metrics are present
        assert metrics.total_return is not None
        assert metrics.cagr is not None
        assert metrics.sharpe_ratio is not None
        assert metrics.annualized_volatility > 0
        assert metrics.max_drawdown <= 0  # Drawdown is negative

    def test_calculate_total_return(self) -> None:
        """Test total return calculation."""
        # Create simple returns: +10% each period for 3 periods
        # Total: (1.1)^3 - 1 = 0.331
        returns = pd.Series(
            [0.10, 0.10, 0.10],
            index=pd.date_range("2020-01-01", periods=3, freq="B")
        )

        calc = PerformanceCalculator(periods_per_year=252)
        metrics = calc.calculate(returns)

        expected_total = 1.1**3 - 1
        assert np.isclose(metrics.total_return, expected_total, rtol=0.001)

    def test_calculate_cagr(self) -> None:
        """Test CAGR calculation."""
        # 1 year of daily returns (252 periods)
        # 10% total return should give ~10% CAGR
        returns = pd.Series(
            [0.10 / 252] * 252,
            index=pd.date_range("2020-01-01", periods=252, freq="B")
        )

        calc = PerformanceCalculator(periods_per_year=252)
        metrics = calc.calculate(returns)

        # Should be close to 10%
        assert 0.08 < metrics.cagr < 0.12

    def test_calculate_volatility(self) -> None:
        """Test volatility calculation."""
        np.random.seed(42)

        # Daily returns with 1% std
        # Annualized: 1% * √252 ≈ 15.87%
        returns = pd.Series(
            np.random.normal(0, 0.01, 1000),
            index=pd.date_range("2020-01-01", periods=1000, freq="B")
        )

        calc = PerformanceCalculator(periods_per_year=252)
        metrics = calc.calculate(returns)

        # Should be close to 15-16%
        assert 0.14 < metrics.annualized_volatility < 0.18

    def test_calculate_sharpe_ratio(self) -> None:
        """Test Sharpe ratio calculation."""
        np.random.seed(42)

        # Daily Sharpe of 0.1 (annualized ~1.59)
        # Mean: 0.001, Std: 0.01
        returns = pd.Series(
            np.random.normal(0.001, 0.01, 1000),
            index=pd.date_range("2020-01-01", periods=1000, freq="B")
        )

        calc = PerformanceCalculator(periods_per_year=252)
        metrics = calc.calculate(returns)

        # Sharpe should be positive and reasonable
        assert 0.5 < metrics.sharpe_ratio < 2.5

    def test_calculate_sharpe_with_risk_free(self) -> None:
        """Test Sharpe ratio with risk-free rate."""
        np.random.seed(42)

        returns = pd.Series(np.random.normal(0.001, 0.01, 1000))
        returns.index = pd.date_range("2020-01-01", periods=1000, freq="B")

        # Risk-free rate: 0.02% daily (about 5% annualized)
        rf = pd.Series([0.0002] * 1000, index=returns.index)

        calc = PerformanceCalculator(risk_free_rate=rf, periods_per_year=252)
        metrics = calc.calculate(returns)

        # Sharpe should account for risk-free rate
        assert metrics.sharpe_ratio is not None

    def test_calculate_sortino_ratio(self) -> None:
        """Test Sortino ratio calculation."""
        np.random.seed(42)

        returns = pd.Series(
            np.random.normal(0.001, 0.01, 1000),
            index=pd.date_range("2020-01-01", periods=1000, freq="B")
        )

        calc = PerformanceCalculator(periods_per_year=252)
        metrics = calc.calculate(returns)

        # Sortino should be higher than Sharpe (uses downside vol)
        assert metrics.sortino_ratio > metrics.sharpe_ratio

    def test_calculate_max_drawdown(self) -> None:
        """Test maximum drawdown calculation."""
        # Create returns that produce a known drawdown
        # Start at 100, go to 120, drop to 90, recover to 110
        # Max DD: (90-120)/120 = -25%
        values = [100, 110, 120, 110, 100, 90, 95, 100, 105, 110]
        returns = pd.Series(
            [values[i] / values[i - 1] - 1 for i in range(1, len(values))],
            index=pd.date_range("2020-01-01", periods=9, freq="B")
        )

        calc = PerformanceCalculator(periods_per_year=252)
        metrics = calc.calculate(returns)

        # Max drawdown should be around -25%
        assert -0.30 < metrics.max_drawdown < -0.20

    def test_calculate_calmar_ratio(self) -> None:
        """Test Calmar ratio calculation."""
        np.random.seed(42)

        returns = pd.Series(
            np.random.normal(0.001, 0.01, 1000),
            index=pd.date_range("2020-01-01", periods=1000, freq="B")
        )

        calc = PerformanceCalculator(periods_per_year=252)
        metrics = calc.calculate(returns)

        # Calmar = CAGR / |max_drawdown|
        expected_calmar = metrics.cagr / abs(metrics.max_drawdown)

        assert np.isclose(metrics.calmar_ratio, expected_calmar, rtol=0.01)

    def test_calculate_skewness_kurtosis(self) -> None:
        """Test skewness and kurtosis calculation."""
        np.random.seed(42)

        # Normal distribution should have skew ~0, kurtosis ~0
        returns = pd.Series(
            np.random.normal(0, 0.01, 1000),
            index=pd.date_range("2020-01-01", periods=1000, freq="B")
        )

        calc = PerformanceCalculator(periods_per_year=252)
        metrics = calc.calculate(returns)

        # Should be close to 0
        assert -1.0 < metrics.skewness < 1.0
        assert -1.0 < metrics.kurtosis < 1.0

    def test_calculate_var_cvar(self) -> None:
        """Test VaR and CVaR calculation."""
        np.random.seed(42)

        returns = pd.Series(
            np.random.normal(0, 0.01, 1000),
            index=pd.date_range("2020-01-01", periods=1000, freq="B")
        )

        calc = PerformanceCalculator(periods_per_year=252)
        metrics = calc.calculate(returns)

        # VaR should be negative (5th percentile)
        assert metrics.var_95 < 0

        # CVaR should be more negative than VaR
        assert metrics.cvar_95 < metrics.var_95

    def test_calculate_win_loss_metrics(self) -> None:
        """Test win/loss ratio and related metrics."""
        # Create returns with known win rate
        # 4 positive: 0.01, 0.008, 0.012, 0.005
        # 4 negative: -0.005, -0.003, -0.007, -0.002
        returns = pd.Series(
            [0.01, -0.005, 0.008, -0.003, 0.012, -0.007, 0.005, -0.002],
            index=pd.date_range("2020-01-01", periods=8, freq="B")
        )

        calc = PerformanceCalculator(periods_per_year=252)
        metrics = calc.calculate(returns)

        # 4 positive, 4 negative = 50% win rate
        assert metrics.positive_periods == 4
        assert metrics.negative_periods == 4
        assert np.isclose(metrics.win_rate, 0.50)

        # Average win
        expected_avg_win = (0.01 + 0.008 + 0.012 + 0.005) / 4
        assert np.isclose(metrics.avg_win, expected_avg_win)

        # Average loss
        expected_avg_loss = (-0.005 - 0.003 - 0.007 - 0.002) / 4
        assert np.isclose(metrics.avg_loss, expected_avg_loss)

    def test_calculate_insufficient_data(self) -> None:
        """Test that insufficient data raises error."""
        returns = pd.Series([0.01])  # Only 1 period

        calc = PerformanceCalculator()

        with pytest.raises(ValueError, match="Need at least 2 periods"):
            calc.calculate(returns)

    def test_aggregate_to_monthly(self) -> None:
        """Test aggregation of daily returns to monthly."""
        # Create 60 days of returns (about 2 months)
        returns = pd.Series(
            [0.001] * 60,
            index=pd.date_range("2020-01-01", periods=60, freq="B"),
        )

        calc = PerformanceCalculator(periods_per_year=252)
        monthly = calc._aggregate_to_monthly(returns)

        # Should have about 2-3 monthly periods
        assert 1 <= len(monthly) <= 3

        # Each monthly return should be roughly (1.001)^20 - 1 ≈ 2%
        # (assuming ~20 trading days per month)
        assert all(0.01 < r < 0.05 for r in monthly)

    def test_aggregate_to_yearly(self) -> None:
        """Test aggregation of daily returns to yearly."""
        # Create 2 years of returns
        returns = pd.Series(
            [0.0004] * 504,  # 2 years * 252 days
            index=pd.date_range("2020-01-01", periods=504, freq="B"),
        )

        calc = PerformanceCalculator(periods_per_year=252)
        yearly = calc._aggregate_to_yearly(returns)

        # Should have 2 yearly periods
        assert len(yearly) >= 1

    def test_calculate_rolling_sharpe(self, sample_daily_returns: pd.Series) -> None:
        """Test rolling Sharpe ratio calculation."""
        calc = PerformanceCalculator(periods_per_year=252)

        rolling = calc.calculate_rolling_sharpe(sample_daily_returns, window=252)

        # Should have length = len(returns) - window + 1
        assert len(rolling) == len(sample_daily_returns)

        # First 251 should be NaN
        assert rolling.iloc[:251].isna().all()

        # Rest should be valid
        valid_sharpe = rolling.dropna()
        assert len(valid_sharpe) > 0


class TestPerformanceIntegration:
    """Integration tests for performance calculation."""

    def test_realistic_equity_returns(self) -> None:
        """Test metrics with realistic equity returns."""
        np.random.seed(42)

        # Simulate 5 years of daily returns
        # Equity-like: 8% annual, 15% vol, slight negative skew
        T = 252 * 5
        daily_mean = 0.08 / 252
        daily_std = 0.15 / np.sqrt(252)

        returns = pd.Series(
            np.random.normal(daily_mean, daily_std, T),
            index=pd.date_range("2020-01-01", periods=T, freq="B")
        )

        calc = PerformanceCalculator(periods_per_year=252)
        metrics = calc.calculate(returns)

        # Total return should be reasonable
        assert 0.0 < metrics.total_return < 2.0  # 0-200%

        # CAGR should be around 8%
        assert 0.0 < metrics.cagr < 0.20

        # Volatility should be around 15%
        assert 0.10 < metrics.annualized_volatility < 0.25

        # Sharpe should be positive for equity returns
        assert metrics.sharpe_ratio > 0

        # Max drawdown should be negative
        assert metrics.max_drawdown < 0

    def test_realistic_bond_returns(self) -> None:
        """Test metrics with realistic bond returns."""
        np.random.seed(42)

        # Simulate bond-like returns
        # 3% annual, 5% vol
        T = 252 * 3
        daily_mean = 0.03 / 252
        daily_std = 0.05 / np.sqrt(252)

        returns = pd.Series(
            np.random.normal(daily_mean, daily_std, T),
            index=pd.date_range("2020-01-01", periods=T, freq="B")
        )

        calc = PerformanceCalculator(periods_per_year=252)
        metrics = calc.calculate(returns)

        # Lower returns than equity
        assert 0.0 < metrics.cagr < 0.10

        # Lower volatility
        assert 0.02 < metrics.annualized_volatility < 0.10

        # Still positive Sharpe
        assert metrics.sharpe_ratio > 0

    def test_negative_returns_strategy(self) -> None:
        """Test metrics for a losing strategy."""
        np.random.seed(42)

        # Negative drift
        returns = pd.Series(
            np.random.normal(-0.0005, 0.01, 1000),
            index=pd.date_range("2020-01-01", periods=1000, freq="B")
        )

        calc = PerformanceCalculator(periods_per_year=252)
        metrics = calc.calculate(returns)

        # Should have negative total return and CAGR
        assert metrics.total_return < 0
        assert metrics.cagr < 0

        # Negative Sharpe
        assert metrics.sharpe_ratio < 0

        # Still have positive volatility
        assert metrics.annualized_volatility > 0

    def test_all_metrics_computed(self, sample_daily_returns: pd.Series) -> None:
        """Test that all metrics are computed without errors."""
        calc = PerformanceCalculator(periods_per_year=252)
        metrics = calc.calculate(sample_daily_returns)

        # Check that no metric is None
        for field in metrics.__dataclass_fields__:
            value = getattr(metrics, field)
            assert value is not None, f"Field {field} is None"

    @pytest.mark.integration
    @pytest.mark.postgres
    def test_real_backtest_performance(self) -> None:
        """Test performance calculation on real backtest results."""
        from datetime import date

        from rademacher_backtest.backtest.engine import BacktestEngine
        from rademacher_backtest.config.portfolio import AssetAllocation, PortfolioDefinition
        from rademacher_backtest.config.settings import BacktestConfig
        from rademacher_backtest.data.postgres_loader import PostgreSQLLoader

        # Create PostgreSQL loader
        loader = PostgreSQLLoader()

        # Simple portfolio
        portfolio = PortfolioDefinition(
            name="Test Portfolio",
            allocations=(
                AssetAllocation("SCHX", 0.60, "US Large Cap"),
                AssetAllocation("AGG", 0.40, "Aggregate Bond"),
            ),
        )

        # 1 year backtest
        config = BacktestConfig(
            start_date=date(2019, 1, 1),
            end_date=date(2019, 12, 31),
            initial_capital=100_000.0,
        )

        # Run backtest
        engine = BacktestEngine(config, portfolio, loader=loader)
        result = engine.run()

        # Calculate performance
        calc = PerformanceCalculator(periods_per_year=252)
        metrics = calc.calculate(result.daily_returns)

        # All metrics should be valid
        assert metrics.total_return is not None
        assert metrics.sharpe_ratio is not None
        assert metrics.max_drawdown < 0

        # Returns should be reasonable for 1 year
        assert -0.50 < metrics.total_return < 1.0

        # Volatility should be reasonable (60/40 portfolio)
        assert 0.03 < metrics.annualized_volatility < 0.20
