# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a quantitative finance backtesting engine implementing the **Rademacher Anti-Serum (RAS)** methodology from Chapter 8 of "Elements of Quantitative Investing". RAS provides statistically rigorous performance evaluation by accounting for data snooping bias and multiple testing corrections.

**Key Concept**: The RAS methodology addresses data snooping bias by calculating probabilistic lower bounds on performance metrics (especially Sharpe ratio) using Rademacher complexity theory. This prevents overfitting and provides confidence intervals that are mathematically sound.

## Development Commands

### Package Management (uv)
This project uses `uv` for dependency management. All Python commands should be run through `uv`:

```bash
# Install dependencies
uv sync                      # Install runtime dependencies
uv sync --extra dev          # Include development dependencies
uv sync --extra notebooks    # Include Jupyter support

# Add dependencies
uv add <package>            # Add runtime dependency
uv add --dev <package>      # Add dev dependency
```

### Running the Application

```bash
# Basic backtest with defaults
python main.py run

# Custom backtest parameters
python main.py run --start-date 2015-01-01 --end-date 2023-12-31 \
  --initial-capital 250000 --transaction-cost 15 --confidence 0.95 \
  --output results/

# View portfolio info
python main.py info

# Check database connectivity and data availability
python main.py check-data
```

### Testing

```bash
# Run all tests
uv run pytest

# Run with coverage report
uv run pytest --cov=src --cov-report=html

# Run specific test categories
uv run pytest tests/test_ras/              # RAS methodology tests
uv run pytest tests/test_backtest/         # Backtest engine tests
uv run pytest -m integration              # Integration tests (requires DB)

# Run single test file
uv run pytest tests/test_ras/test_bounds.py -v
```

### Code Quality

```bash
# Linting
uv run ruff check              # Check for issues
uv run ruff check --fix        # Auto-fix issues

# Formatting
uv run ruff format             # Format all code

# Type checking
uv run mypy src/               # Type check source code
```

## Architecture Overview

### Core Data Flow

1. **Data Loading** (`src/data/loader.py`)
   - `PostgreSQLLoader` connects to PostgreSQL database
   - Expected tables: `sfp` (price data), `fama_french` (factor data), `rates` (treasury yields)
   - Returns data as pandas DataFrames with DatetimeIndex

2. **Backtest Execution** (`src/backtest/engine.py`)
   - `BacktestEngine.run()` simulates portfolio over time
   - Monthly rebalancing by default (configurable)
   - Tracks holdings, transaction costs, daily snapshots
   - Returns `BacktestResult` with all performance data

3. **RAS Analysis** (3-step process in `src/ras/`)
   - **Complexity Estimation** (`complexity.py`): Monte Carlo estimation of Rademacher complexity
   - **Haircut Calculation** (`haircut.py`): Combines complexity with statistical penalty
   - **Bounds Calculation** (`bounds.py`): Produces final adjusted Sharpe ratio with confidence intervals

4. **Analytics** (`src/analytics/`)
   - `performance.py`: Standard metrics (CAGR, Sharpe, Sortino, Calmar)
   - `risk.py`: Risk metrics (volatility, VaR, max drawdown)
   - `attribution.py`: Factor-based attribution analysis

### Key Architectural Patterns

**Holdings and Rebalancing**:
- `Holdings` class (`src/portfolio/portfolio.py`) tracks shares + cash
- Rebalancing is transaction-cost aware (basis points configurable)
- Weight drift calculation determines when to rebalance

**RAS Methodology** (critical for understanding this codebase):
- Returns are **standardized** (z-scored) before complexity estimation
- Rademacher complexity: `E[sup_n (1/T) * Σ_t ε_t * x_{t,n}]` where ε are ±1 random signs
- Total haircut = 2×R_T(F) + sqrt(2log(2/δ)/T)
  - First term: overfitting penalty (Rademacher complexity)
  - Second term: estimation error (concentration inequality)
- Lower bound: `Empirical Sharpe - Total Haircut`

**Data Loader Protocol**:
- `DataLoader` is a Protocol (structural typing) in `src/data/loader.py`
- Allows swapping implementations (PostgreSQL, CSV, etc.)
- Tests use mock implementations via dependency injection

## Database Schema

The PostgreSQL database must have these tables:

- **`sfp`**: Stock/Fund Prices
  - Columns: `date`, `ticker`, `closeadj` (adjusted close)

- **`fama_french`**: Factor returns
  - Columns: `date`, `mkt`, `smb`, `hml`, `rmw`, `cma`, `mom`, `rf`
  - Values stored as percentages (converted to decimals by loader)

- **`rates`**: Treasury yield curve data

- **`ticker_master`**: Ticker metadata (optional)

Set `DATABASE_URL` environment variable in `.env` file.

## Configuration System

Two main config classes in `src/config/settings.py`:

1. **`BacktestConfig`**: Backtest parameters
   - Dates, capital, rebalancing frequency, transaction costs
   - `transaction_cost_bps`: Basis points (10 = 0.1% round-trip)

2. **`RASConfig`**: Statistical parameters
   - `delta`: Confidence parameter (1-δ = confidence level, e.g., 0.01 = 99%)
   - `n_simulations`: Monte Carlo iterations for Rademacher complexity (20k default)
   - `annualization_factor`: √252 for daily returns

Portfolio definitions in `src/config/portfolio.py` use dataclasses.

## Important Implementation Details

### Return Calculations
- Daily returns: `values.pct_change()`
- Monthly returns: `values.resample('ME').last().pct_change()`
- Sharpe ratio calculated on **non-annualized** returns, then annualized by multiplying by √252

### Transaction Costs
- Applied during rebalancing in `src/portfolio/rebalancer.py`
- Cost = `sum(|target_value - current_value|) * cost_rate`
- Deducted from cash balance

### Statistical Rigor
- All RAS calculations use **non-annualized** Sharpe ratios internally
- Annualization only for reporting (multiply by √252)
- This ensures statistical theory remains valid

### Testing Strategy
- Unit tests use synthetic data (numpy arrays)
- Integration tests marked with `@pytest.mark.integration`
- Use `pytest.approx()` for floating-point comparisons
- RAS tests verify theoretical bounds (e.g., complexity ≈ 0.8/√T for single strategy)

## Common Development Tasks

### Adding a New Performance Metric
1. Add calculation to `src/analytics/performance.py` or `src/analytics/risk.py`
2. Update `PerformanceMetrics` dataclass if needed
3. Add test in `tests/test_analytics/`
4. Update visualization in `src/visualization/charts.py` if displaying graphically

### Adding a New Data Loader
1. Create class implementing `DataLoader` protocol in `src/data/`
2. Implement `load_prices()`, `load_risk_free_rate()`, `load_fama_french()`
3. Update `BacktestEngine.__init__()` to accept new loader
4. Add tests with sample data

### Modifying Rebalancing Logic
1. Update `src/portfolio/rebalancer.py`
2. Rebalancer takes `Holdings` and returns new `Holdings` + transaction cost
3. Ensure transaction costs are properly calculated
4. Test with various portfolio drifts

### Changing RAS Parameters
- Never hardcode RAS parameters - always use `RASConfig`
- To test different confidence levels, use `BoundsCalculator.sensitivity_analysis()`
- Higher `n_simulations` = more accurate but slower (20k is good default)

## Data Validation

The codebase expects clean, aligned data:
- All tickers must have data for the same date range
- Missing data will raise `ValueError` in `PostgreSQLLoader.load_prices()`
- Use `python main.py check-data` to verify data availability before running backtests

## Python Version & Type Hints

- Requires Python 3.13+
- Uses modern type hints (`from __future__ import annotations`)
- Strict mypy configuration in `pyproject.toml`
- Use `NDArray[np.float64]` for numpy arrays, not `np.ndarray`
