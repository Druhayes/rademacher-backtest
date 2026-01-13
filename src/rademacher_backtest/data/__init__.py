"""Data loading and preprocessing module."""

from rademacher_backtest.data.loader import DataLoader
from rademacher_backtest.data.csv_loader import CSVLoader
from rademacher_backtest.data.dataframe_loader import DataFrameLoader
from rademacher_backtest.data.preprocessor import DataPreprocessor

# PostgreSQL loader is optional (requires postgres extra)
try:
    from rademacher_backtest.data.postgres_loader import PostgreSQLLoader
except ImportError:
    PostgreSQLLoader = None  # type: ignore

__all__ = [
    "DataLoader",
    "DataFrameLoader",
    "CSVLoader",
    "PostgreSQLLoader",
    "DataPreprocessor",
]
