"""Tests for DataFrameLoader class."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from rademacher_backtest.data.dataframe_loader import DataFrameLoader


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    """Create sample price data for testing."""
    dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
    return pd.DataFrame(
        {
            "SCHX": [100 + i * 0.1 for i in range(len(dates))],
            "AGG": [50 + i * 0.05 for i in range(len(dates))],
            "SCHA": [75 + i * 0.075 for i in range(len(dates))],
        },
        index=dates,
    )


@pytest.fixture
def sample_risk_free() -> pd.Series:
    """Create sample risk-free rate data."""
    dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
    return pd.Series([0.0001] * len(dates), index=dates, name="rf")


@pytest.fixture
def sample_fama_french() -> pd.DataFrame:
    """Create sample Fama-French factor data."""
    dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
    return pd.DataFrame(
        {
            "mkt": [0.001] * len(dates),
            "smb": [0.0005] * len(dates),
            "hml": [0.0003] * len(dates),
            "rmw": [0.0002] * len(dates),
            "cma": [0.0001] * len(dates),
            "mom": [0.0004] * len(dates),
            "rf": [0.0001] * len(dates),
        },
        index=dates,
    )


class TestDataFrameLoaderInitialization:
    """Test DataFrameLoader initialization."""

    def test_initialization_with_prices_only(self, sample_prices):
        """Test initialization with just price data."""
        loader = DataFrameLoader(prices=sample_prices)
        assert loader is not None
        assert len(loader.available_tickers) == 3
        assert "SCHX" in loader.available_tickers

    def test_initialization_with_all_data(
        self, sample_prices, sample_risk_free, sample_fama_french
    ):
        """Test initialization with all optional data."""
        loader = DataFrameLoader(
            prices=sample_prices,
            risk_free_rate=sample_risk_free,
            fama_french=sample_fama_french,
        )
        assert loader is not None

    def test_initialization_fails_without_datetime_index(self):
        """Test that initialization fails if index is not DatetimeIndex."""
        df = pd.DataFrame({"SPY": [100, 101, 102]})  # No datetime index
        with pytest.raises(ValueError, match="DatetimeIndex"):
            DataFrameLoader(prices=df)

    def test_initialization_fails_with_empty_dataframe(self):
        """Test that initialization fails with empty DataFrame."""
        df = pd.DataFrame(index=pd.DatetimeIndex([]))
        with pytest.raises(ValueError, match="cannot be empty"):
            DataFrameLoader(prices=df)

    def test_initialization_sorts_index(self):
        """Test that data is sorted by index on initialization."""
        dates = pd.to_datetime(["2020-12-31", "2020-01-01", "2020-06-15"])
        df = pd.DataFrame({"SPY": [103, 101, 102]}, index=dates)

        loader = DataFrameLoader(prices=df)
        min_date, max_date = loader.date_range

        assert min_date == date(2020, 1, 1)
        assert max_date == date(2020, 12, 31)


class TestLoadPrices:
    """Test load_prices method."""

    def test_load_prices_single_ticker(self, sample_prices):
        """Test loading a single ticker."""
        loader = DataFrameLoader(prices=sample_prices)
        result = loader.load_prices(
            ["SCHX"],
            date(2020, 1, 1),
            date(2020, 12, 31),
        )

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["SCHX"]
        assert len(result) > 0

    def test_load_prices_multiple_tickers(self, sample_prices):
        """Test loading multiple tickers."""
        loader = DataFrameLoader(prices=sample_prices)
        result = loader.load_prices(
            ["SCHX", "AGG"],
            date(2020, 1, 1),
            date(2020, 12, 31),
        )

        assert list(result.columns) == ["SCHX", "AGG"]
        assert len(result) > 0

    def test_load_prices_preserves_ticker_order(self, sample_prices):
        """Test that ticker order is preserved."""
        loader = DataFrameLoader(prices=sample_prices)
        result = loader.load_prices(
            ["AGG", "SCHX"],  # Reverse alphabetical order
            date(2020, 1, 1),
            date(2020, 12, 31),
        )

        assert list(result.columns) == ["AGG", "SCHX"]

    def test_load_prices_filters_by_date_range(self, sample_prices):
        """Test that date filtering works correctly."""
        loader = DataFrameLoader(prices=sample_prices)
        result = loader.load_prices(
            ["SCHX"],
            date(2020, 6, 1),
            date(2020, 6, 30),
        )

        assert result.index.min().date() >= date(2020, 6, 1)
        assert result.index.max().date() <= date(2020, 6, 30)
        assert len(result) == 30  # June has 30 days

    def test_load_prices_empty_result_for_invalid_date_range(self, sample_prices):
        """Test that empty DataFrame is returned for out-of-range dates."""
        loader = DataFrameLoader(prices=sample_prices)
        result = loader.load_prices(
            ["SCHX"],
            date(2025, 1, 1),
            date(2025, 12, 31),
        )

        assert result.empty

    def test_load_prices_raises_for_missing_ticker(self, sample_prices):
        """Test that ValueError is raised for missing tickers."""
        loader = DataFrameLoader(prices=sample_prices)
        with pytest.raises(ValueError, match="not found in data"):
            loader.load_prices(
                ["INVALID_TICKER"],
                date(2020, 1, 1),
                date(2020, 12, 31),
            )

    def test_load_prices_raises_for_partially_missing_tickers(self, sample_prices):
        """Test error message lists all missing tickers."""
        loader = DataFrameLoader(prices=sample_prices)
        with pytest.raises(ValueError, match="MISSING1.*MISSING2"):
            loader.load_prices(
                ["SCHX", "MISSING1", "MISSING2"],
                date(2020, 1, 1),
                date(2020, 12, 31),
            )


class TestLoadRiskFreeRate:
    """Test load_risk_free_rate method."""

    def test_load_risk_free_rate_with_data(
        self, sample_prices, sample_risk_free
    ):
        """Test loading risk-free rate when data is provided."""
        loader = DataFrameLoader(
            prices=sample_prices,
            risk_free_rate=sample_risk_free,
        )
        result = loader.load_risk_free_rate(
            date(2020, 1, 1),
            date(2020, 12, 31),
        )

        assert isinstance(result, pd.Series)
        assert len(result) > 0

    def test_load_risk_free_rate_without_data(self, sample_prices):
        """Test loading risk-free rate when no data was provided."""
        loader = DataFrameLoader(prices=sample_prices)
        result = loader.load_risk_free_rate(
            date(2020, 1, 1),
            date(2020, 12, 31),
        )

        assert isinstance(result, pd.Series)
        assert result.empty

    def test_load_risk_free_rate_filters_by_date(
        self, sample_prices, sample_risk_free
    ):
        """Test that risk-free rate is filtered by date range."""
        loader = DataFrameLoader(
            prices=sample_prices,
            risk_free_rate=sample_risk_free,
        )
        result = loader.load_risk_free_rate(
            date(2020, 6, 1),
            date(2020, 6, 30),
        )

        assert result.index.min().date() >= date(2020, 6, 1)
        assert result.index.max().date() <= date(2020, 6, 30)


class TestLoadFamaFrench:
    """Test load_fama_french method."""

    def test_load_fama_french_with_data(
        self, sample_prices, sample_fama_french
    ):
        """Test loading Fama-French factors when data is provided."""
        loader = DataFrameLoader(
            prices=sample_prices,
            fama_french=sample_fama_french,
        )
        result = loader.load_fama_french(
            date(2020, 1, 1),
            date(2020, 12, 31),
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert "mkt" in result.columns
        assert "smb" in result.columns

    def test_load_fama_french_without_data(self, sample_prices):
        """Test loading Fama-French when no data was provided."""
        loader = DataFrameLoader(prices=sample_prices)
        result = loader.load_fama_french(
            date(2020, 1, 1),
            date(2020, 12, 31),
        )

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_load_fama_french_filters_by_date(
        self, sample_prices, sample_fama_french
    ):
        """Test that Fama-French data is filtered by date range."""
        loader = DataFrameLoader(
            prices=sample_prices,
            fama_french=sample_fama_french,
        )
        result = loader.load_fama_french(
            date(2020, 6, 1),
            date(2020, 6, 30),
        )

        assert result.index.min().date() >= date(2020, 6, 1)
        assert result.index.max().date() <= date(2020, 6, 30)


class TestProperties:
    """Test DataFrameLoader properties."""

    def test_available_tickers_property(self, sample_prices):
        """Test available_tickers property."""
        loader = DataFrameLoader(prices=sample_prices)
        tickers = loader.available_tickers

        assert isinstance(tickers, list)
        assert set(tickers) == {"SCHX", "AGG", "SCHA"}

    def test_date_range_property(self, sample_prices):
        """Test date_range property."""
        loader = DataFrameLoader(prices=sample_prices)
        min_date, max_date = loader.date_range

        assert min_date == date(2020, 1, 1)
        assert max_date == date(2020, 12, 31)

    def test_info_method_with_minimal_data(self, sample_prices):
        """Test info() method with just price data."""
        loader = DataFrameLoader(prices=sample_prices)
        info = loader.info()

        assert info["loader_type"] == "DataFrame (In-Memory)"
        assert info["num_tickers"] == 3
        assert info["optional_data"]["risk_free_rate"] is False
        assert info["optional_data"]["fama_french"] is False
        assert info["status"] == "ready"

    def test_info_method_with_all_data(
        self, sample_prices, sample_risk_free, sample_fama_french
    ):
        """Test info() method with all data provided."""
        loader = DataFrameLoader(
            prices=sample_prices,
            risk_free_rate=sample_risk_free,
            fama_french=sample_fama_french,
        )
        info = loader.info()

        assert info["optional_data"]["risk_free_rate"] is True
        assert info["optional_data"]["fama_french"] is True


class TestDataIsolation:
    """Test that loader doesn't modify external data."""

    def test_loader_makes_copy_of_data(self, sample_prices):
        """Test that loader makes a copy and external changes don't affect it."""
        original_prices = sample_prices.copy()
        loader = DataFrameLoader(prices=sample_prices)

        # Modify external DataFrame
        sample_prices.iloc[0, 0] = 9999.0

        # Loader should have original data
        result = loader.load_prices(
            ["SCHX"],
            date(2020, 1, 1),
            date(2020, 1, 1),
        )
        assert result.iloc[0, 0] == original_prices.iloc[0, 0]
        assert result.iloc[0, 0] != 9999.0
