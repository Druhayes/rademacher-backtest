"""Rademacher-Backtest: Backtesting with Statistical Rigor.

A library for backtesting investment strategies with the Rademacher Anti-Serum
(RAS) methodology for rigorous performance evaluation accounting for multiple
testing and data snooping bias.

Quick Start:
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

For more details, see the documentation and examples.
"""

__version__ = "1.0.0"

# Core data loaders
from rademacher_backtest.data.dataframe_loader import DataFrameLoader
from rademacher_backtest.data.loader import DataLoader

# Backtest engine and config
from rademacher_backtest.backtest.engine import BacktestEngine, BacktestRunner
from rademacher_backtest.backtest.results import BacktestResult, DailySnapshot
from rademacher_backtest.config.settings import BacktestConfig, RASConfig

# Portfolio definitions
from rademacher_backtest.config.portfolio import (
    AssetAllocation,
    PortfolioDefinition,
    SAMPLE_PORTFOLIO,
)

# High-level API
from rademacher_backtest.api import analyze_ras, backtest, create_portfolio

# Analytics
from rademacher_backtest.analytics.performance import (
    PerformanceCalculator,
    PerformanceMetrics,
)
from rademacher_backtest.analytics.risk import DrawdownAnalysis, RiskAnalyzer
from rademacher_backtest.ras.bounds import BoundsCalculator, SharpeRatioBounds
from rademacher_backtest.ras.report import RASReport, RASReportGenerator

# Optional loaders (graceful degradation)
try:
    from rademacher_backtest.data.csv_loader import CSVLoader
except ImportError:
    CSVLoader = None  # type: ignore

try:
    from rademacher_backtest.data.postgres_loader import PostgreSQLLoader
except ImportError:
    PostgreSQLLoader = None  # type: ignore

# Optional visualization (graceful degradation)
try:
    from rademacher_backtest.visualization.charts import (
        BacktestCharts,
        RASVisualization,
        ReturnHeatmap,
    )
except ImportError:
    BacktestCharts = None  # type: ignore
    RASVisualization = None  # type: ignore
    ReturnHeatmap = None  # type: ignore

__all__ = [
    # Version
    "__version__",
    # Core Data
    "DataFrameLoader",
    "DataLoader",
    # Engine & Config
    "BacktestEngine",
    "BacktestRunner",
    "BacktestResult",
    "DailySnapshot",
    "BacktestConfig",
    "RASConfig",
    # Portfolio
    "PortfolioDefinition",
    "AssetAllocation",
    "SAMPLE_PORTFOLIO",
    "create_portfolio",
    # High-level API
    "backtest",
    "analyze_ras",
    # Analytics
    "PerformanceCalculator",
    "PerformanceMetrics",
    "RiskAnalyzer",
    "DrawdownAnalysis",
    # RAS
    "BoundsCalculator",
    "SharpeRatioBounds",
    "RASReport",
    "RASReportGenerator",
    # Optional (may be None if not installed)
    "CSVLoader",
    "PostgreSQLLoader",
    "BacktestCharts",
    "RASVisualization",
    "ReturnHeatmap",
]
