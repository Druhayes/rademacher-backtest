#!/usr/bin/env python3
"""Generate sample data for rademacher-backtest examples.

This script creates realistic-looking market data for demonstration purposes.
The data is synthetic but follows reasonable statistical properties.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_price_series(
    start_date: str,
    end_date: str,
    initial_price: float,
    annual_return: float,
    annual_volatility: float,
    seed: int,
) -> pd.Series:
    """Generate realistic price series using geometric Brownian motion."""
    np.random.seed(seed)

    # Create date range
    dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
    n_days = len(dates)

    # Daily parameters
    daily_return = annual_return / 252
    daily_volatility = annual_volatility / np.sqrt(252)

    # Generate returns using geometric Brownian motion
    returns = np.random.normal(daily_return, daily_volatility, n_days)

    # Convert to prices
    price_factors = np.exp(np.cumsum(returns))
    prices = initial_price * price_factors

    return pd.Series(prices, index=dates)


def main():
    """Generate all sample data files."""
    print("Generating sample data...")

    # Date range for all data
    start_date = "2010-01-04"  # First trading day of 2010
    end_date = "2024-12-31"

    # Generate price data for 5 ETFs
    # Using common ETFs: SPY (S&P 500), AGG (Bonds), GLD (Gold), VNQ (Real Estate), EEM (Emerging Markets)
    print("  - Generating price data...")

    etfs = {
        'SPY': {'initial': 100.0, 'return': 0.10, 'vol': 0.18, 'seed': 42},
        'AGG': {'initial': 100.0, 'return': 0.03, 'vol': 0.05, 'seed': 43},
        'GLD': {'initial': 100.0, 'return': 0.05, 'vol': 0.15, 'seed': 44},
        'VNQ': {'initial': 50.0, 'return': 0.08, 'vol': 0.22, 'seed': 45},
        'EEM': {'initial': 30.0, 'return': 0.04, 'vol': 0.25, 'seed': 46},
    }

    prices_data = {}
    for ticker, params in etfs.items():
        prices_data[ticker] = generate_price_series(
            start_date,
            end_date,
            params['initial'],
            params['return'],
            params['vol'],
            params['seed'],
        )

    prices_df = pd.DataFrame(prices_data)
    prices_df.index.name = 'date'
    prices_df.to_csv('examples/data/sample_prices.csv')
    print(f"    ✓ Created sample_prices.csv ({len(prices_df)} rows, {len(prices_df.columns)} tickers)")

    # Generate risk-free rate (using 3-month Treasury bill approximation)
    print("  - Generating risk-free rate...")
    dates = pd.date_range(start=start_date, end=end_date, freq='B')

    # Simulate declining interest rates over time (roughly matching 2010-2024 trend)
    base_rate = 0.02  # 2% starting rate
    trend = np.linspace(0, -0.015, len(dates))  # Declining to ~0.5%
    noise = np.random.RandomState(47).normal(0, 0.001, len(dates))

    annual_rf = base_rate + trend + noise
    annual_rf = np.maximum(annual_rf, 0.001)  # Floor at 0.1%

    # Convert to daily
    daily_rf = (1 + annual_rf) ** (1/252) - 1

    rf_series = pd.Series(daily_rf, index=dates, name='rf')
    rf_series.index.name = 'date'
    rf_series.to_csv('examples/data/sample_rf.csv', header=True)
    print(f"    ✓ Created sample_rf.csv ({len(rf_series)} rows)")

    # Generate Fama-French factors
    print("  - Generating Fama-French factors...")
    np.random.seed(48)

    # Market factor (similar to S&P 500 excess return)
    mkt = np.random.normal(0.04/252, 0.01, len(dates))

    # Size factor (SMB - Small Minus Big)
    smb = np.random.normal(0.02/252, 0.005, len(dates))

    # Value factor (HML - High Minus Low)
    hml = np.random.normal(0.03/252, 0.005, len(dates))

    # Profitability factor (RMW - Robust Minus Weak)
    rmw = np.random.normal(0.02/252, 0.004, len(dates))

    # Investment factor (CMA - Conservative Minus Aggressive)
    cma = np.random.normal(0.02/252, 0.004, len(dates))

    # Momentum factor
    mom = np.random.normal(0.05/252, 0.008, len(dates))

    ff_df = pd.DataFrame({
        'mkt': mkt,
        'smb': smb,
        'hml': hml,
        'rmw': rmw,
        'cma': cma,
        'mom': mom,
        'rf': daily_rf,
    }, index=dates)

    ff_df.index.name = 'date'
    ff_df.to_csv('examples/data/sample_ff.csv')
    print(f"    ✓ Created sample_ff.csv ({len(ff_df)} rows, {len(ff_df.columns)} factors)")

    print("\n✅ Sample data generation complete!")
    print(f"\nData summary:")
    print(f"  Date range: {start_date} to {end_date}")
    print(f"  Trading days: {len(dates)}")
    print(f"  ETFs: {', '.join(etfs.keys())}")
    print(f"\nFiles created in examples/data/:")
    print(f"  - sample_prices.csv")
    print(f"  - sample_rf.csv")
    print(f"  - sample_ff.csv")


if __name__ == "__main__":
    main()
