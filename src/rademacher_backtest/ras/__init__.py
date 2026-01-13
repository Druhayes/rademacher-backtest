"""Rademacher Anti-Serum (RAS) methodology implementation.

This module implements the RAS backtesting protocol from Chapter 8 of
"Elements of Quantitative Investing" by Antti Ilmanen et al.

RAS provides:
- Rigorous statistical bounds on performance metrics
- Protection against data-snooping bias
- Handles serial dependencies in financial time series
"""

from rademacher_backtest.ras.bounds import BoundsCalculator, SharpeRatioBounds
from rademacher_backtest.ras.complexity import RademacherComplexityEstimator, massart_bound
from rademacher_backtest.ras.haircut import HaircutCalculator, RASHaircut
from rademacher_backtest.ras.report import RASReport, RASReportGenerator

__all__ = [
    "BoundsCalculator",
    "HaircutCalculator",
    "massart_bound",
    "RademacherComplexityEstimator",
    "RASHaircut",
    "RASReport",
    "RASReportGenerator",
    "SharpeRatioBounds",
]
