# RAS Backtest Test Suite

Comprehensive test suite for the RAS Backtest application covering all core functionality including RAS methodology, backtesting engine, and performance analytics.

## Test Structure

```
tests/
├── conftest.py                          # Shared fixtures
├── test_ras/                           # RAS methodology tests
│   ├── test_complexity.py             # Rademacher complexity estimation
│   ├── test_haircut.py                # Haircut calculation
│   └── test_bounds.py                 # Probabilistic bounds
├── test_backtest/                      # Backtest engine tests
│   └── test_engine.py                 # Engine and runner tests
└── test_analytics/                     # Performance analytics tests
    └── test_performance.py            # Performance metrics calculation
```

## Test Coverage

### RAS Methodology (`test_ras/`)

#### `test_complexity.py` (21 tests)
- **RademacherComplexityEstimator**: Single/multiple strategy complexity estimation, reproducibility, Monte Carlo variation
- **Massart Bound**: Theoretical upper bounds, scaling with N and T
- **Theoretical Complexity**: Formula validation, 1/√T scaling
- **Integration Tests**: Empirical vs theoretical comparison, correlation benefits

#### `test_haircut.py` (19 tests)
- **RASHaircut Dataclass**: Initialization, confidence levels, annualization
- **HaircutCalculator**:
  - Bounded signals (Procedure 8.1)
  - Sharpe ratio (Procedure 8.2 & Theorem 8.4)
  - Tunable parameters
  - Minimum sample size calculation
  - Confidence level and sample size impact
- **Integration Tests**: Book procedures verification, decomposition validation

#### `test_bounds.py` (23 tests)
- **SharpeRatioBounds**: Haircut percentage, statistical significance
- **BoundsCalculator**:
  - Single strategy bounds
  - Multiple strategy bounds with correlation
  - Information Ratio bounds
  - Sensitivity analysis across confidence levels
  - Reproducibility with fixed seeds
- **Integration Tests**: Realistic scenarios, weak vs strong strategies, correlation benefits

### Backtest Engine (`test_backtest/`)

#### `test_engine.py` (20 tests)
- **BacktestEngine**: Initialization, rebalancing, transaction costs, snapshots
- **BacktestRunner**: Run methods, comparison functionality
- **Integration Tests** (6 tests - require database):
  - Real database backtest
  - Multiple asset portfolios
  - Buy-and-hold vs rebalanced comparison
  - Different rebalancing frequencies
  - Transaction cost impact

### Analytics (`test_analytics/`)

#### `test_performance.py` (18 tests)
- **PerformanceMetrics**: Dataclass, dictionary conversion, string representation
- **PerformanceCalculator**:
  - Total return, CAGR, volatility
  - Sharpe, Sortino, Calmar ratios
  - Maximum drawdown, drawdown duration
  - Skewness, kurtosis, VaR/CVaR
  - Win/loss metrics
  - Aggregation to monthly/yearly
  - Rolling Sharpe calculation
- **Integration Tests**: Realistic equity/bond/negative strategies

## Running Tests

### All Tests (excluding integration)
```bash
uv run pytest
```

### Specific Test Module
```bash
uv run pytest tests/test_ras/test_complexity.py -v
```

### Integration Tests Only (requires database)
```bash
uv run pytest -m integration
```

### Skip Integration Tests
```bash
uv run pytest -m "not integration"
```

### With Coverage Report
```bash
uv run pytest --cov=src --cov-report=html
```

## Test Fixtures (conftest.py)

### Configuration Fixtures
- `backtest_config`: Default backtest configuration
- `ras_config`: Default RAS configuration (5000 simulations for faster tests)
- `sample_portfolio`: Sample 60/40 portfolio
- `simple_portfolio`: Simple 2-asset portfolio

### Data Fixtures
- `sample_daily_returns`: 1000 days of simulated returns (7.5% annual, 16% vol)
- `sample_prices`: Multi-asset price data
- `sample_returns_matrix`: 2000×10 correlated returns matrix (standardized)

## Key Testing Principles

### Monte Carlo Variation
RAS methodology uses Monte Carlo simulation, which introduces random variation. Tests account for this by:
- Using fixed random seeds for reproducibility
- Allowing reasonable tolerances (10-30% depending on context)
- Testing ranges rather than exact values
- Verifying general trends rather than strict ordering

### Real Data Integration Tests
Integration tests (marked with `@pytest.mark.integration`) use the actual PostgreSQL database:
- Test with real ETF price data (SCHX, SCHA, SCHF, SCHE, AGG)
- Verify backtest engine with production data
- Require DATABASE_URL environment variable

### DatetimeIndex Requirement
Performance calculator tests require DatetimeIndex for proper aggregation (monthly/yearly). All test data uses business day frequency ("B").

## Test Statistics

- **Total Tests**: 101 unit tests + 6 integration tests
- **Test Coverage**: >90% of source code
- **Execution Time**: ~3 seconds (unit tests only)
- **Integration Time**: ~5-10 seconds (with database)

## Database Setup for Integration Tests

1. Copy example.env to .env:
```bash
cp example.env .env
```

2. Update DATABASE_URL in .env with your credentials:
```env
DATABASE_URL=postgresql://username:password@localhost:5432/econ_data
```

3. Run integration tests:
```bash
uv run pytest -m integration
```

## Continuous Integration

For CI/CD pipelines:
```bash
# Unit tests only (no database required)
uv run pytest -m "not integration" --cov=src --cov-report=xml

# With database in CI
export DATABASE_URL="postgresql://ci_user:ci_pass@db:5432/test_db"
uv run pytest --cov=src --cov-report=xml
```

## Known Limitations

### Monte Carlo Variability
- Rademacher complexity estimates vary between runs (even with seeds)
- Some tests use relaxed tolerances to accommodate this
- Tests focus on verifying methodology correctness rather than exact values

### Integration Test Dependencies
- Require PostgreSQL database with specific schema
- Need actual price data in `sfp` and `fama_french` tables
- May fail if database structure changes

### Performance Test Precision
- Some metrics (CAGR, Sharpe) sensitive to small variations in random data
- Tests use reasonable ranges rather than exact expected values

## Contributing

When adding new tests:
1. Use appropriate fixtures from conftest.py
2. Add DatetimeIndex to any returns/price data
3. Account for Monte Carlo variation in RAS tests
4. Mark database-dependent tests with `@pytest.mark.integration`
5. Follow existing naming conventions
6. Include docstrings explaining what's being tested
