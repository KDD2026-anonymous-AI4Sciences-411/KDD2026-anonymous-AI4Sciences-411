"""
Limit Order Book (LOB) Implementation for SE-RL Framework
========================================================

This module implements a complete Limit Order Book system with:
- Price-time priority matching
- Limit orders and market orders
- Multi-agent order submission
- Price update mechanism (midpoint pricing)

As described in Section A.5.2 of the paper.

Author: AI Research Engineer
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import heapq
import logging
from collections import defaultdict
import time

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types supported by the LOB"""
    LIMIT = "limit"
    MARKET = "market"


class OrderSide(Enum):
    """Order sides"""
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """Represents a single order in the order book"""
    order_id: int
    agent_id: int
    side: OrderSide
    order_type: OrderType
    price: float
    quantity: float
    timestamp: float
    remaining_quantity: float = None
    status: str = "active"  # active, partially_filled, filled, cancelled

    def __post_init__(self):
        if self.remaining_quantity is None:
            self.remaining_quantity = self.quantity

    def __lt__(self, other):
        """For priority queue ordering"""
        if self.side == OrderSide.BUY:
            # Higher price has priority for buy orders
            if self.price != other.price:
                return self.price > other.price
        else:
            # Lower price has priority for sell orders
            if self.price != other.price:
                return self.price < other.price
        # Same price: earlier timestamp has priority (price-time priority)
        return self.timestamp < other.timestamp


@dataclass
class Trade:
    """Represents an executed trade"""
    trade_id: int
    buyer_id: int
    seller_id: int
    price: float
    quantity: float
    timestamp: float
    buyer_order_id: int
    seller_order_id: int


@dataclass
class LOBSnapshot:
    """Snapshot of the current LOB state"""
    timestamp: float
    best_bid: float
    best_ask: float
    mid_price: float
    spread: float
    bid_depth: List[Tuple[float, float]]  # (price, quantity) pairs
    ask_depth: List[Tuple[float, float]]  # (price, quantity) pairs
    total_bid_volume: float
    total_ask_volume: float
    imbalance: float  # (bid_volume - ask_volume) / (bid_volume + ask_volume)


class PriceLevel:
    """Represents a single price level in the order book"""

    def __init__(self, price: float):
        self.price = price
        self.orders: List[Order] = []
        self.total_quantity: float = 0.0

    def add_order(self, order: Order):
        """Add order to this price level (maintains time priority)"""
        self.orders.append(order)
        self.total_quantity += order.remaining_quantity

    def remove_order(self, order_id: int) -> Optional[Order]:
        """Remove order from this price level"""
        for i, order in enumerate(self.orders):
            if order.order_id == order_id:
                removed_order = self.orders.pop(i)
                self.total_quantity -= removed_order.remaining_quantity
                return removed_order
        return None

    def get_first_order(self) -> Optional[Order]:
        """Get first order at this price level (time priority)"""
        return self.orders[0] if self.orders else None

    def is_empty(self) -> bool:
        return len(self.orders) == 0


class LimitOrderBook:
    """
    Complete Limit Order Book implementation with price-time priority matching.

    Implements the matching mechanism described in Section A.5.2:
    - Orders are matched based on price-time priority
    - Price is updated as midpoint: P(t+1) = (p*_b + p*_s) / 2
    - Supports limit orders and market orders
    """

    def __init__(self, initial_price: float = 100.0, tick_size: float = 0.01):
        """
        Initialize the Limit Order Book.

        Args:
            initial_price: Initial mid-price
            tick_size: Minimum price increment
        """
        self.initial_price = initial_price
        self.tick_size = tick_size

        # Order book structure: price -> PriceLevel
        self.bids: Dict[float, PriceLevel] = {}  # Buy orders (descending by price)
        self.asks: Dict[float, PriceLevel] = {}  # Sell orders (ascending by price)

        # Sorted price levels for efficient best bid/ask lookup
        self.bid_prices: List[float] = []  # Max heap (negated for min-heap)
        self.ask_prices: List[float] = []  # Min heap

        # Order tracking
        self.orders: Dict[int, Order] = {}  # order_id -> Order
        self.order_counter: int = 0
        self.trade_counter: int = 0

        # Trade history
        self.trades: List[Trade] = []

        # Price history
        self.current_price: float = initial_price
        self.price_history: List[Tuple[float, float]] = [(0.0, initial_price)]  # (timestamp, price)

        # Statistics
        self.total_volume_traded: float = 0.0
        self.num_trades: int = 0

        logger.info(f"LOB initialized with initial price: {initial_price}, tick size: {tick_size}")

    def _round_price(self, price: float) -> float:
        """Round price to nearest tick size"""
        return round(price / self.tick_size) * self.tick_size

    def submit_order(self, agent_id: int, side: OrderSide, order_type: OrderType,
                     price: float, quantity: float, timestamp: float = None) -> Tuple[Order, List[Trade]]:
        """
        Submit an order to the order book.

        Args:
            agent_id: ID of the agent submitting the order
            side: Buy or Sell
            order_type: Limit or Market
            price: Order price (ignored for market orders)
            quantity: Order quantity
            timestamp: Order timestamp (uses current time if not provided)

        Returns:
            Tuple of (submitted order, list of trades executed)
        """
        if timestamp is None:
            timestamp = time.time()

        # Round price to tick size
        price = self._round_price(price)

        # Create order
        self.order_counter += 1
        order = Order(
            order_id=self.order_counter,
            agent_id=agent_id,
            side=side,
            order_type=order_type,
            price=price,
            quantity=quantity,
            timestamp=timestamp
        )

        # Process order based on type
        if order_type == OrderType.MARKET:
            trades = self._process_market_order(order)
        else:  # Limit order
            trades = self._process_limit_order(order)

        # Update price based on trades
        if trades:
            self._update_price(timestamp)

        return order, trades

    def _process_market_order(self, order: Order) -> List[Trade]:
        """Process a market order (immediate execution at best available price)"""
        trades = []

        if order.side == OrderSide.BUY:
            # Match against asks (sell orders)
            while order.remaining_quantity > 0 and self.ask_prices:
                best_ask_price = self.ask_prices[0]
                trades.extend(self._match_order(order, best_ask_price, OrderSide.SELL))
        else:
            # Match against bids (buy orders)
            while order.remaining_quantity > 0 and self.bid_prices:
                best_bid_price = -self.bid_prices[0]  # Negated for max-heap
                trades.extend(self._match_order(order, best_bid_price, OrderSide.BUY))

        # Market orders are filled or cancelled, never rest in book
        if order.remaining_quantity > 0:
            order.status = "partially_filled"
        else:
            order.status = "filled"

        return trades

    def _process_limit_order(self, order: Order) -> List[Trade]:
        """Process a limit order"""
        trades = []

        if order.side == OrderSide.BUY:
            # Try to match against asks at or below the limit price
            while order.remaining_quantity > 0 and self.ask_prices:
                best_ask_price = self.ask_prices[0]
                if best_ask_price <= order.price:
                    trades.extend(self._match_order(order, best_ask_price, OrderSide.SELL))
                else:
                    break
        else:
            # Try to match against bids at or above the limit price
            while order.remaining_quantity > 0 and self.bid_prices:
                best_bid_price = -self.bid_prices[0]
                if best_bid_price >= order.price:
                    trades.extend(self._match_order(order, best_bid_price, OrderSide.BUY))
                else:
                    break

        # If order has remaining quantity, add to order book
        if order.remaining_quantity > 0:
            self._add_order_to_book(order)
        else:
            order.status = "filled"

        return trades

    def _match_order(self, aggressor: Order, price_level: float,
                     resting_side: OrderSide) -> List[Trade]:
        """Match aggressor order against orders at a price level"""
        trades = []

        # Get the appropriate book
        book = self.asks if resting_side == OrderSide.SELL else self.bids

        if price_level not in book:
            return trades

        level = book[price_level]

        # Match against orders at this level (time priority)
        while aggressor.remaining_quantity > 0 and not level.is_empty():
            resting_order = level.get_first_order()

            # Determine trade quantity
            trade_qty = min(aggressor.remaining_quantity, resting_order.remaining_quantity)

            # Create trade
            self.trade_counter += 1
            if aggressor.side == OrderSide.BUY:
                trade = Trade(
                    trade_id=self.trade_counter,
                    buyer_id=aggressor.agent_id,
                    seller_id=resting_order.agent_id,
                    price=resting_order.price,
                    quantity=trade_qty,
                    timestamp=aggressor.timestamp,
                    buyer_order_id=aggressor.order_id,
                    seller_order_id=resting_order.order_id
                )
            else:
                trade = Trade(
                    trade_id=self.trade_counter,
                    buyer_id=resting_order.agent_id,
                    seller_id=aggressor.agent_id,
                    price=resting_order.price,
                    quantity=trade_qty,
                    timestamp=aggressor.timestamp,
                    buyer_order_id=resting_order.order_id,
                    seller_order_id=aggressor.order_id
                )

            trades.append(trade)
            self.trades.append(trade)
            self.total_volume_traded += trade_qty
            self.num_trades += 1

            # Update quantities
            aggressor.remaining_quantity -= trade_qty
            resting_order.remaining_quantity -= trade_qty
            level.total_quantity -= trade_qty

            # Remove filled resting order
            if resting_order.remaining_quantity <= 0:
                resting_order.status = "filled"
                level.orders.pop(0)
                if resting_order.order_id in self.orders:
                    del self.orders[resting_order.order_id]
            else:
                resting_order.status = "partially_filled"

        # Clean up empty price level
        if level.is_empty():
            del book[price_level]
            if resting_side == OrderSide.SELL:
                heapq.heappop(self.ask_prices)
            else:
                heapq.heappop(self.bid_prices)

        return trades

    def _add_order_to_book(self, order: Order):
        """Add order to the order book"""
        if order.side == OrderSide.BUY:
            if order.price not in self.bids:
                self.bids[order.price] = PriceLevel(order.price)
                heapq.heappush(self.bid_prices, -order.price)  # Max-heap
            self.bids[order.price].add_order(order)
        else:
            if order.price not in self.asks:
                self.asks[order.price] = PriceLevel(order.price)
                heapq.heappush(self.ask_prices, order.price)
            self.asks[order.price].add_order(order)

        self.orders[order.order_id] = order
        order.status = "active" if order.remaining_quantity == order.quantity else "partially_filled"

    def cancel_order(self, order_id: int) -> Optional[Order]:
        """Cancel an order by ID"""
        if order_id not in self.orders:
            return None

        order = self.orders[order_id]

        # Remove from appropriate book
        book = self.bids if order.side == OrderSide.BUY else self.asks
        if order.price in book:
            book[order.price].remove_order(order_id)
            if book[order.price].is_empty():
                del book[order.price]
                # Note: Not removing from heap for efficiency, will be cleaned during matching

        order.status = "cancelled"
        del self.orders[order_id]

        return order

    def _update_price(self, timestamp: float):
        """
        Update price based on best bid/ask (midpoint pricing).

        As per paper equation: P(t+1) = (p*_b + p*_s) / 2
        """
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()

        if best_bid is not None and best_ask is not None:
            # Midpoint pricing as specified in paper
            self.current_price = (best_bid + best_ask) / 2
        elif best_bid is not None:
            self.current_price = best_bid
        elif best_ask is not None:
            self.current_price = best_ask
        # Otherwise keep current price

        self.price_history.append((timestamp, self.current_price))

    def get_best_bid(self) -> Optional[float]:
        """Get the best (highest) bid price"""
        while self.bid_prices:
            price = -self.bid_prices[0]
            if price in self.bids and not self.bids[price].is_empty():
                return price
            heapq.heappop(self.bid_prices)
        return None

    def get_best_ask(self) -> Optional[float]:
        """Get the best (lowest) ask price"""
        while self.ask_prices:
            price = self.ask_prices[0]
            if price in self.asks and not self.asks[price].is_empty():
                return price
            heapq.heappop(self.ask_prices)
        return None

    def get_spread(self) -> Optional[float]:
        """Get the bid-ask spread"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid is not None and best_ask is not None:
            return best_ask - best_bid
        return None

    def get_mid_price(self) -> float:
        """Get the current mid price"""
        return self.current_price

    def get_depth(self, levels: int = 5) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """
        Get the order book depth up to specified levels.

        Returns:
            Tuple of (bid_depth, ask_depth) where each is a list of (price, quantity) tuples
        """
        bid_depth = []
        ask_depth = []

        # Get bid depth
        bid_prices_sorted = sorted([p for p in self.bids.keys()], reverse=True)[:levels]
        for price in bid_prices_sorted:
            if price in self.bids:
                bid_depth.append((price, self.bids[price].total_quantity))

        # Get ask depth
        ask_prices_sorted = sorted([p for p in self.asks.keys()])[:levels]
        for price in ask_prices_sorted:
            if price in self.asks:
                ask_depth.append((price, self.asks[price].total_quantity))

        return bid_depth, ask_depth

    def get_snapshot(self, timestamp: float = None) -> LOBSnapshot:
        """Get a complete snapshot of the current LOB state"""
        if timestamp is None:
            timestamp = time.time()

        best_bid = self.get_best_bid() or 0.0
        best_ask = self.get_best_ask() or float('inf')
        mid_price = self.current_price
        spread = best_ask - best_bid if best_ask < float('inf') else 0.0

        bid_depth, ask_depth = self.get_depth()

        total_bid_volume = sum(qty for _, qty in bid_depth)
        total_ask_volume = sum(qty for _, qty in ask_depth)

        total_volume = total_bid_volume + total_ask_volume
        imbalance = (total_bid_volume - total_ask_volume) / total_volume if total_volume > 0 else 0.0

        return LOBSnapshot(
            timestamp=timestamp,
            best_bid=best_bid,
            best_ask=best_ask if best_ask < float('inf') else 0.0,
            mid_price=mid_price,
            spread=spread,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            total_bid_volume=total_bid_volume,
            total_ask_volume=total_ask_volume,
            imbalance=imbalance
        )

    def get_vwap(self, lookback: int = 10) -> float:
        """Calculate Volume-Weighted Average Price from recent trades"""
        if not self.trades:
            return self.current_price

        recent_trades = self.trades[-lookback:]
        total_value = sum(t.price * t.quantity for t in recent_trades)
        total_volume = sum(t.quantity for t in recent_trades)

        if total_volume > 0:
            return total_value / total_volume
        return self.current_price

    def get_state_vector(self, depth_levels: int = 5) -> np.ndarray:
        """
        Get LOB state as a flattened vector for RL agent observation.

        Returns vector containing:
        - Best bid/ask prices (2)
        - Mid price (1)
        - Spread (1)
        - Bid depth at each level (depth_levels * 2: price + qty)
        - Ask depth at each level (depth_levels * 2: price + qty)
        - Imbalance (1)
        - VWAP (1)
        """
        snapshot = self.get_snapshot()
        bid_depth, ask_depth = self.get_depth(depth_levels)

        state = []

        # Basic prices
        state.extend([
            snapshot.best_bid / self.initial_price,  # Normalized
            snapshot.best_ask / self.initial_price if snapshot.best_ask > 0 else 0,
            snapshot.mid_price / self.initial_price,
            snapshot.spread / self.initial_price if snapshot.spread > 0 else 0,
        ])

        # Bid depth (pad if necessary)
        for i in range(depth_levels):
            if i < len(bid_depth):
                price, qty = bid_depth[i]
                state.extend([price / self.initial_price, qty / 1000])  # Normalize
            else:
                state.extend([0.0, 0.0])

        # Ask depth (pad if necessary)
        for i in range(depth_levels):
            if i < len(ask_depth):
                price, qty = ask_depth[i]
                state.extend([price / self.initial_price, qty / 1000])  # Normalize
            else:
                state.extend([0.0, 0.0])

        # Additional features
        state.extend([
            snapshot.imbalance,
            self.get_vwap() / self.initial_price
        ])

        return np.array(state, dtype=np.float32)

    def reset(self):
        """Reset the order book to initial state"""
        self.bids.clear()
        self.asks.clear()
        self.bid_prices.clear()
        self.ask_prices.clear()
        self.orders.clear()
        self.order_counter = 0
        self.trade_counter = 0
        self.trades.clear()
        self.current_price = self.initial_price
        self.price_history = [(0.0, self.initial_price)]
        self.total_volume_traded = 0.0
        self.num_trades = 0

        logger.debug("LOB reset to initial state")

    def get_statistics(self) -> Dict[str, Any]:
        """Get LOB statistics"""
        return {
            'num_bids': sum(len(level.orders) for level in self.bids.values()),
            'num_asks': sum(len(level.orders) for level in self.asks.values()),
            'total_bid_volume': sum(level.total_quantity for level in self.bids.values()),
            'total_ask_volume': sum(level.total_quantity for level in self.asks.values()),
            'num_trades': self.num_trades,
            'total_volume_traded': self.total_volume_traded,
            'current_price': self.current_price,
            'spread': self.get_spread(),
            'vwap': self.get_vwap()
        }


# Unit tests
if __name__ == "__main__":
    # Test LOB functionality
    lob = LimitOrderBook(initial_price=100.0, tick_size=0.01)

    # Submit some orders
    print("Testing LOB...")

    # Buy limit orders
    order1, trades1 = lob.submit_order(1, OrderSide.BUY, OrderType.LIMIT, 99.0, 100, 1.0)
    order2, trades2 = lob.submit_order(2, OrderSide.BUY, OrderType.LIMIT, 99.5, 50, 2.0)

    # Sell limit orders
    order3, trades3 = lob.submit_order(3, OrderSide.SELL, OrderType.LIMIT, 101.0, 100, 3.0)
    order4, trades4 = lob.submit_order(4, OrderSide.SELL, OrderType.LIMIT, 100.5, 50, 4.0)

    print(f"Best bid: {lob.get_best_bid()}")
    print(f"Best ask: {lob.get_best_ask()}")
    print(f"Spread: {lob.get_spread()}")

    # Submit crossing order
    order5, trades5 = lob.submit_order(5, OrderSide.BUY, OrderType.MARKET, 0, 30, 5.0)
    print(f"Market order trades: {len(trades5)}")

    snapshot = lob.get_snapshot()
    print(f"Snapshot: mid={snapshot.mid_price}, imbalance={snapshot.imbalance:.4f}")

    state = lob.get_state_vector()
    print(f"State vector shape: {state.shape}")

    stats = lob.get_statistics()
    print(f"Statistics: {stats}")

    print("LOB tests passed!")
