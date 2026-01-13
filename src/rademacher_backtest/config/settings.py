"""Configuration dataclasses for backtest and RAS settings."""

from dataclasses import dataclass
from datetime import date
from typing import Literal


@dataclass(frozen=True)
class BacktestConfig:
    """Immutable backtest configuration.

    Attributes:
        start_date: Start date for the backtest (default: SCHE inception)
        end_date: End date for the backtest
        initial_capital: Starting portfolio value in dollars
        rebalance_frequency: How often to rebalance the portfolio
        transaction_cost_bps: Round-trip transaction cost in basis points (10 = 0.1%)
        risk_free_rate_source: Source for risk-free rate data
    """

    start_date: date = date(2010, 1, 14)  # SCHE inception date
    end_date: date = date(2025, 12, 30)
    initial_capital: float = 100_000.0
    rebalance_frequency: Literal["daily", "weekly", "monthly", "quarterly", "annually"] = "monthly"
    transaction_cost_bps: float = 10.0  # 0.1% round-trip = 10 bps
    risk_free_rate_source: Literal["fama_french", "rates"] = "fama_french"

    @property
    def transaction_cost_rate(self) -> float:
        """Convert basis points to decimal rate."""
        return self.transaction_cost_bps / 10_000


@dataclass(frozen=True)
class RASConfig:
    """Rademacher Anti-Serum methodology configuration.

    Attributes:
        delta: Confidence parameter (1 - delta = confidence level)
               Default 0.01 means 99% confidence
        n_simulations: Number of Monte Carlo simulations for Rademacher complexity
        use_sub_gaussian: Use Theorem 8.4 bounds for sub-Gaussian returns
        random_seed: Seed for reproducible Rademacher simulations
        annualization_factor: Factor to annualize daily Sharpe (sqrt(252) for daily)
    """

    delta: float = 0.01  # 99% confidence
    n_simulations: int = 20_000
    use_sub_gaussian: bool = True
    random_seed: int = 42
    annualization_factor: float = 252**0.5  # sqrt(252) for daily data

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not 0 < self.delta < 1:
            raise ValueError(f"delta must be between 0 and 1, got {self.delta}")
        if self.n_simulations < 1000:
            raise ValueError(f"n_simulations should be >= 1000, got {self.n_simulations}")

    @property
    def confidence_level(self) -> float:
        """Return confidence level as percentage (e.g., 99.0 for delta=0.01)."""
        return (1 - self.delta) * 100
