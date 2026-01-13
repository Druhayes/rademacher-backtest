"""Portfolio holdings management."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Self


@dataclass
class Holdings:
    """Current portfolio holdings tracking shares and cash.

    Attributes:
        shares: Dictionary mapping ticker to number of shares held
        cash: Cash balance in the portfolio
    """

    shares: dict[str, float] = field(default_factory=dict)
    cash: float = 0.0

    def total_value(self, prices: dict[str, float]) -> float:
        """Calculate total portfolio value at current prices.

        Args:
            prices: Dictionary mapping ticker to current price

        Returns:
            Total portfolio value (equity + cash)
        """
        equity_value = sum(
            self.shares.get(ticker, 0.0) * price
            for ticker, price in prices.items()
            if ticker in self.shares
        )
        return equity_value + self.cash

    def equity_value(self, prices: dict[str, float]) -> float:
        """Calculate total equity value (excluding cash).

        Args:
            prices: Dictionary mapping ticker to current price

        Returns:
            Total value of equity holdings
        """
        return sum(
            self.shares.get(ticker, 0.0) * price
            for ticker, price in prices.items()
            if ticker in self.shares
        )

    def position_value(self, ticker: str, price: float) -> float:
        """Calculate value of a single position.

        Args:
            ticker: Ticker symbol
            price: Current price

        Returns:
            Position value
        """
        return self.shares.get(ticker, 0.0) * price

    def weights(self, prices: dict[str, float]) -> dict[str, float]:
        """Calculate current portfolio weights.

        Args:
            prices: Dictionary mapping ticker to current price

        Returns:
            Dictionary mapping ticker to weight (as decimal)
        """
        total = self.total_value(prices)
        if total == 0:
            return {ticker: 0.0 for ticker in self.shares}

        return {
            ticker: (shares * prices.get(ticker, 0.0)) / total
            for ticker, shares in self.shares.items()
        }

    def weight_drift(
        self,
        prices: dict[str, float],
        target_weights: dict[str, float],
    ) -> dict[str, float]:
        """Calculate drift from target weights.

        Args:
            prices: Dictionary mapping ticker to current price
            target_weights: Dictionary mapping ticker to target weight

        Returns:
            Dictionary mapping ticker to weight deviation (current - target)
        """
        current = self.weights(prices)
        return {
            ticker: current.get(ticker, 0.0) - target_weights.get(ticker, 0.0)
            for ticker in set(current.keys()) | set(target_weights.keys())
        }

    def max_drift(
        self,
        prices: dict[str, float],
        target_weights: dict[str, float],
    ) -> float:
        """Calculate maximum absolute drift from target weights.

        Args:
            prices: Dictionary mapping ticker to current price
            target_weights: Dictionary mapping ticker to target weight

        Returns:
            Maximum absolute weight deviation
        """
        drift = self.weight_drift(prices, target_weights)
        return max(abs(d) for d in drift.values()) if drift else 0.0

    def copy(self) -> Self:
        """Create a copy of the holdings.

        Returns:
            New Holdings instance with same values
        """
        return Holdings(
            shares=self.shares.copy(),
            cash=self.cash,
        )

    @classmethod
    def from_weights(
        cls,
        capital: float,
        weights: dict[str, float],
        prices: dict[str, float],
    ) -> Self:
        """Create holdings from target weights and capital.

        Args:
            capital: Total capital to invest
            weights: Target weights for each ticker
            prices: Current prices for each ticker

        Returns:
            Holdings initialized to target weights
        """
        shares = {}
        remaining_cash = capital

        for ticker, weight in weights.items():
            if ticker not in prices:
                raise ValueError(f"Price not available for ticker: {ticker}")

            allocation = capital * weight
            ticker_shares = allocation / prices[ticker]
            shares[ticker] = ticker_shares
            remaining_cash -= allocation

        return cls(shares=shares, cash=remaining_cash)

    def __repr__(self) -> str:
        """Return string representation of holdings."""
        positions = ", ".join(
            f"{ticker}: {shares:.2f}"
            for ticker, shares in sorted(self.shares.items())
        )
        return f"Holdings(shares={{{positions}}}, cash={self.cash:.2f})"
