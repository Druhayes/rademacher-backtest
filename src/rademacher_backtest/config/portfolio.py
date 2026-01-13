"""Portfolio definition dataclasses and sample portfolios."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AssetAllocation:
    """Single asset allocation within a portfolio.

    Attributes:
        ticker: Security ticker symbol (e.g., 'SCHX')
        weight: Target weight as decimal (e.g., 0.35 for 35%)
        name: Human-readable asset class name
    """

    ticker: str
    weight: float
    name: str = ""

    def __post_init__(self) -> None:
        """Validate allocation parameters."""
        if not 0 <= self.weight <= 1:
            raise ValueError(f"Weight must be between 0 and 1, got {self.weight}")
        if not self.ticker:
            raise ValueError("Ticker cannot be empty")


@dataclass(frozen=True)
class PortfolioDefinition:
    """Portfolio definition with target allocations.

    Attributes:
        name: Portfolio name/description
        allocations: Tuple of AssetAllocation objects
    """

    name: str
    allocations: tuple[AssetAllocation, ...]

    def __post_init__(self) -> None:
        """Validate that weights sum to 1.0."""
        total = sum(a.weight for a in self.allocations)
        if not (0.999 < total < 1.001):
            raise ValueError(f"Weights must sum to 1.0, got {total:.4f}")

    @property
    def tickers(self) -> list[str]:
        """Return list of ticker symbols."""
        return [a.ticker for a in self.allocations]

    @property
    def weights(self) -> dict[str, float]:
        """Return ticker to weight mapping."""
        return {a.ticker: a.weight for a in self.allocations}

    def get_weight(self, ticker: str) -> float:
        """Get weight for a specific ticker."""
        for alloc in self.allocations:
            if alloc.ticker == ticker:
                return alloc.weight
        raise KeyError(f"Ticker {ticker} not found in portfolio")


# Sample 60/40 Portfolio with Market Cap Weighted Equity
SAMPLE_PORTFOLIO = PortfolioDefinition(
    name="60/40 Market Cap Weighted",
    allocations=(
        AssetAllocation("SCHX", 0.35, "US Large Cap"),
        AssetAllocation("SCHA", 0.05, "US Small Cap"),
        AssetAllocation("SCHF", 0.15, "International Developed"),
        AssetAllocation("SCHE", 0.05, "Emerging Markets"),
        AssetAllocation("AGG", 0.40, "Aggregate Bond"),
    ),
)

# Alternative portfolios for comparison
EQUAL_WEIGHT_PORTFOLIO = PortfolioDefinition(
    name="Equal Weight",
    allocations=(
        AssetAllocation("SCHX", 0.20, "US Large Cap"),
        AssetAllocation("SCHA", 0.20, "US Small Cap"),
        AssetAllocation("SCHF", 0.20, "International Developed"),
        AssetAllocation("SCHE", 0.20, "Emerging Markets"),
        AssetAllocation("AGG", 0.20, "Aggregate Bond"),
    ),
)

US_HEAVY_PORTFOLIO = PortfolioDefinition(
    name="US Heavy 60/40",
    allocations=(
        AssetAllocation("SCHX", 0.25, "US Large Cap"),
        AssetAllocation("SCHA", 0.10, "US Small Cap"),
        AssetAllocation("SCHF", 0.15, "International Developed"),
        AssetAllocation("SCHE", 0.10, "Emerging Markets"),
        AssetAllocation("AGG", 0.40, "Aggregate Bond"),
    ),
)
