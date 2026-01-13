"""Portfolio rebalancing logic with transaction cost modeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rademacher_backtest.portfolio.portfolio import Holdings


@dataclass(frozen=True)
class RebalanceOrder:
    """Represents a single rebalancing trade.

    Attributes:
        ticker: Security ticker symbol
        shares: Number of shares to trade (positive=buy, negative=sell)
        price: Execution price per share
    """

    ticker: str
    shares: float
    price: float

    @property
    def value(self) -> float:
        """Absolute dollar value of the trade."""
        return abs(self.shares * self.price)

    @property
    def is_buy(self) -> bool:
        """Whether this is a buy order."""
        return self.shares > 0

    @property
    def is_sell(self) -> bool:
        """Whether this is a sell order."""
        return self.shares < 0


class MonthlyRebalancer:
    """Monthly rebalancing implementation with transaction costs.

    Rebalances portfolio to target weights at specified intervals,
    accounting for transaction costs.
    """

    def __init__(
        self,
        target_weights: dict[str, float],
        cost_bps: float = 10.0,
        min_trade_value: float = 100.0,
    ) -> None:
        """Initialize the rebalancer.

        Args:
            target_weights: Target portfolio weights (ticker -> weight)
            cost_bps: Round-trip transaction cost in basis points (10 = 0.1%)
            min_trade_value: Minimum trade value to avoid tiny trades
        """
        self.target_weights = target_weights
        self.cost_rate = cost_bps / 10_000  # Convert bps to decimal
        self.min_trade_value = min_trade_value

        # Validate weights sum to 1
        total = sum(target_weights.values())
        if not (0.999 < total < 1.001):
            raise ValueError(f"Target weights must sum to 1.0, got {total:.4f}")

    def calculate_orders(
        self,
        holdings: Holdings,
        prices: dict[str, float],
    ) -> list[RebalanceOrder]:
        """Calculate orders needed to rebalance to target weights.

        Args:
            holdings: Current portfolio holdings
            prices: Current prices for each ticker

        Returns:
            List of RebalanceOrder objects (sells first, then buys)
        """
        total_value = holdings.total_value(prices)
        if total_value <= 0:
            return []

        orders: list[RebalanceOrder] = []
        sells: list[RebalanceOrder] = []
        buys: list[RebalanceOrder] = []

        for ticker, target_weight in self.target_weights.items():
            if ticker not in prices:
                raise ValueError(f"Price not available for ticker: {ticker}")

            price = prices[ticker]
            current_shares = holdings.shares.get(ticker, 0.0)
            current_value = current_shares * price
            target_value = total_value * target_weight
            delta_value = target_value - current_value
            delta_shares = delta_value / price

            # Skip tiny trades
            if abs(delta_value) < self.min_trade_value:
                continue

            order = RebalanceOrder(
                ticker=ticker,
                shares=delta_shares,
                price=price,
            )

            if order.is_sell:
                sells.append(order)
            elif order.is_buy:
                buys.append(order)

        # Execute sells before buys to free up cash
        orders = sells + buys
        return orders

    def calculate_transaction_costs(self, orders: list[RebalanceOrder]) -> float:
        """Calculate total transaction costs for a set of orders.

        Args:
            orders: List of rebalancing orders

        Returns:
            Total transaction cost in dollars
        """
        total_turnover = sum(order.value for order in orders)
        return total_turnover * self.cost_rate

    def calculate_turnover(
        self,
        holdings: Holdings,
        prices: dict[str, float],
    ) -> float:
        """Calculate one-way turnover required to rebalance.

        Args:
            holdings: Current portfolio holdings
            prices: Current prices

        Returns:
            One-way turnover as fraction of portfolio value
        """
        total_value = holdings.total_value(prices)
        if total_value <= 0:
            return 0.0

        orders = self.calculate_orders(holdings, prices)
        total_trade_value = sum(order.value for order in orders)

        # One-way turnover (buys or sells, not both)
        return (total_trade_value / 2) / total_value

    def should_rebalance(
        self,
        holdings: Holdings,
        prices: dict[str, float],
        drift_threshold: float = 0.05,
    ) -> bool:
        """Determine if rebalancing is needed based on drift threshold.

        Args:
            holdings: Current portfolio holdings
            prices: Current prices
            drift_threshold: Maximum allowed drift before rebalancing

        Returns:
            True if any position has drifted beyond threshold
        """
        max_drift = holdings.max_drift(prices, self.target_weights)
        return max_drift > drift_threshold

    def execute_orders(
        self,
        holdings: Holdings,
        orders: list[RebalanceOrder],
        transaction_cost: float,
    ) -> Holdings:
        """Execute rebalancing orders and return new holdings.

        Args:
            holdings: Current portfolio holdings
            orders: List of orders to execute
            transaction_cost: Total transaction cost to deduct

        Returns:
            New Holdings object with updated positions
        """
        new_holdings = holdings.copy()
        new_holdings.cash -= transaction_cost

        for order in orders:
            current_shares = new_holdings.shares.get(order.ticker, 0.0)
            new_holdings.shares[order.ticker] = current_shares + order.shares
            # Adjust cash: selling adds cash, buying reduces cash
            new_holdings.cash -= order.shares * order.price

        return new_holdings

    def rebalance(
        self,
        holdings: Holdings,
        prices: dict[str, float],
    ) -> tuple[Holdings, list[RebalanceOrder], float]:
        """Perform a complete rebalance operation.

        Args:
            holdings: Current portfolio holdings
            prices: Current prices for each ticker

        Returns:
            Tuple of (new_holdings, orders, transaction_cost)
        """
        orders = self.calculate_orders(holdings, prices)
        cost = self.calculate_transaction_costs(orders)
        new_holdings = self.execute_orders(holdings, orders, cost)
        return new_holdings, orders, cost
