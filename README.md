# rademacher-backtest

A Python library for backtesting investment strategies with **statistical rigor** using the **Rademacher Anti-Serum (RAS)** methodology.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Why RAS?

Traditional backtesting can be misleading due to:
- **Data snooping bias**: Testing multiple strategies inflates the chance of finding one that "worked" by luck
- **Overfitting**: Complex strategies may fit noise rather than signal
- **Estimation error**: Limited data gives noisy performance estimates

**RAS provides statistically rigorous bounds** on the Sharpe ratio, accounting for:
1. Finite sample size (estimation error)
2. Strategy complexity (overfitting penalty)
3. Multiple testing (data snooping correction)

With RAS, you can determine if a strategy's performance is **statistically significant** or just luck.

## 🚀 Quick Start

### Installation

```bash
# Core library (minimal dependencies)
pip install rademacher-backtest

# With visualization support
pip install rademacher-backtest[viz]

# With PostgreSQL support
pip install rademacher-backtest[postgres]

# With CLI interface
pip install rademacher-backtest[cli]

# Everything
pip install rademacher-backtest[all]
```

### Basic Example

```python
import pandas as pd
import rademacher_backtest as rbt

# Load your price data (DataFrame with DatetimeIndex and ticker columns)
prices = pd.read_csv('prices.csv', index_col='date', parse_dates=True)

# Create a data loader
loader = rbt.DataFrameLoader(prices)

# Run a backtest with a simple 60/40 portfolio
result = rbt.backtest(
    portfolio={'SPY': 0.60, 'AGG': 0.40},
    loader=loader,
    start_date='2015-01-01',
    end_date='2023-12-31',
    initial_capital=100_000.0,
)

# Calculate performance metrics
perf = rbt.PerformanceCalculator().calculate(result.daily_returns)
print(f"CAGR: {perf.cagr:.2f}%")
print(f"Sharpe Ratio: {perf.sharpe_ratio:.3f}")
print(f"Max Drawdown: {perf.max_drawdown:.2f}%")

# Apply RAS methodology for statistical rigor
ras = rbt.analyze_ras(result.daily_returns, confidence=0.99)
print(f"\nEmpirical Sharpe: {ras.empirical_sharpe_annualized:.3f}")
print(f"RAS-Adjusted Sharpe: {ras.adjusted_sharpe_annualized:.3f}")
print(f"Statistically Positive: {ras.is_statistically_positive}")
```

## 📊 What is RAS?

RAS adjusts the empirical Sharpe ratio with two penalties:

```
RAS-Adjusted Sharpe = Empirical Sharpe - Estimation Haircut - Complexity Haircut
```

- **Estimation Haircut**: Accounts for sampling error (decreases with more data: ~1/√T)
- **Complexity Haircut**: Accounts for overfitting and multiple testing

The result is a **statistically rigorous lower bound** on the true Sharpe ratio at your chosen confidence level (e.g., 99%).

### Statistical Significance Test

RAS provides a formal hypothesis test:

- **Null Hypothesis**: Strategy has zero or negative Sharpe ratio
- **Decision**: If RAS-adjusted Sharpe > 0, reject null at specified confidence level

This tells you whether performance is **genuinely positive** or could be explained by luck.

## 🎯 Features

### Backtesting Engine
- **Flexible data sources**: DataFrames (primary), CSV files, PostgreSQL database
- **Realistic simulation**: Transaction costs, rebalancing schedules
- **Multiple asset classes**: Stocks, bonds, commodities, alternatives
- **Daily granularity**: Full historical tracking of portfolio evolution

### Performance Analytics
- **Standard metrics**: CAGR, Sharpe ratio, Sortino ratio, Calmar ratio
- **Risk analysis**: Volatility, Value-at-Risk (VaR), maximum drawdown
- **Factor attribution**: Fama-French factor exposure analysis
- **Visualizations**: Charts, heatmaps, drawdown plots (with `[viz]` extra)

### RAS Methodology
- **Statistical testing**: Rigorous significance tests for strategy performance
- **Confidence bounds**: 95%, 99%, or custom confidence levels
- **Multiple testing correction**: Accounts for testing N strategies
- **Complexity estimation**: Monte Carlo estimation of Rademacher complexity
- **Clear interpretation**: Actionable reports for investment decisions

## 📖 Examples

### Custom Portfolio Comparison

```python
import rademacher_backtest as rbt

# Define multiple strategies
strategies = {
    'Conservative': {'SPY': 0.20, 'AGG': 0.80},
    'Balanced': {'SPY': 0.60, 'AGG': 0.40},
    'Aggressive': {'SPY': 0.80, 'AGG': 0.20},
}

# Run all backtests
results = {}
for name, portfolio in strategies.items():
    results[name] = rbt.backtest(
        portfolio=portfolio,
        loader=loader,
        start_date='2015-01-01',
        end_date='2023-12-31',
    )

# Compare with RAS (including multiple testing correction!)
n_strategies = len(strategies)
for name, result in results.items():
    ras = rbt.analyze_ras(
        result.daily_returns,
        confidence=0.99,
        n_strategies=n_strategies,  # Important: corrects for multiple testing
    )

    stat = "✅" if ras.is_statistically_positive else "❌"
    print(f"{name:12s}: Sharpe={ras.empirical_sharpe_annualized:.3f}, "
          f"RAS-Adj={ras.adjusted_sharpe_annualized:.3f} {stat}")
```

### Using CSV Data

```python
# Load from CSV files (one file per ticker)
loader = rbt.CSVLoader('./data')  # Directory with SPY.csv, AGG.csv, etc.

result = rbt.backtest(
    portfolio={'SPY': 0.60, 'AGG': 0.40},
    loader=loader,
    start_date='2020-01-01',
    end_date='2023-12-31',
)
```

### Using PostgreSQL

```python
# Requires: pip install rademacher-backtest[postgres]
from rademacher_backtest.data.postgres_loader import PostgreSQLLoader

loader = PostgreSQLLoader('postgresql://user:pass@localhost/db')

result = rbt.backtest(
    portfolio={'SPY': 0.60, 'AGG': 0.40},
    loader=loader,
    start_date='2020-01-01',
    end_date='2023-12-31',
)
```

### Visualization (with `[viz]` extra)

```python
# Requires: pip install rademacher-backtest[viz]

# Create charts
charts = rbt.BacktestCharts()

# Cumulative returns
fig = charts.cumulative_returns(result.daily_values)
charts.save_figure(fig, 'cumulative_returns.png')

# Drawdown analysis
fig = charts.drawdown(result.daily_returns)
charts.save_figure(fig, 'drawdowns.png')

# Monthly returns heatmap
heatmap = rbt.ReturnHeatmap()
fig = heatmap.plot(result.daily_returns)
charts.save_figure(fig, 'monthly_heatmap.png')

# RAS decomposition
ras_viz = rbt.RASVisualization()
fig = ras_viz.haircut_decomposition(ras)
charts.save_figure(fig, 'ras_analysis.png')
```

## 📚 Documentation & Examples

Comprehensive Jupyter notebook examples are available in the [`examples/`](examples/) directory:

1. **[01_basic_backtest.ipynb](examples/01_basic_backtest.ipynb)** - Quick start guide with simple DataFrame usage
2. **[02_custom_portfolio.ipynb](examples/02_custom_portfolio.ipynb)** - Comparing multiple portfolio strategies
3. **[03_ras_analysis.ipynb](examples/03_ras_analysis.ipynb)** - Deep dive into RAS methodology
4. **[04_visualization.ipynb](examples/04_visualization.ipynb)** - Creating publication-quality charts

Sample data is provided in [`examples/data/`](examples/data/).

## 🔧 CLI Interface

If installed with the `[cli]` extra, you can use the command-line interface:

```bash
# Install with CLI support
pip install rademacher-backtest[cli]

# Run a backtest from command line
ras-backtest run --loader-type csv --csv-dir ./data

# Get help
ras-backtest --help
ras-backtest run --help

# View portfolio information
ras-backtest info
```

## 🧮 Theoretical Background

### The Multiple Testing Problem

When testing many strategies, false discoveries become likely:
- Test 1 strategy at α = 5%: **5% chance** of false positive
- Test 20 strategies at α = 5%: **~65% chance** of at least one false positive
- Test 100 strategies at α = 5%: **~99% chance** of false discovery

RAS corrects for this using the Rademacher complexity framework.

### Rademacher Complexity

RAS estimates how much a strategy family can "memorize" random noise:

```
R_T(F) = E[sup_{f∈F} (1/T) Σ σᵢ f(xᵢ)]
```

Where σᵢ are i.i.d. Rademacher variables (±1 with probability 1/2).

### Statistical Guarantee

With probability at least 1-δ:

```
True Sharpe ≥ Empirical Sharpe - 2×R_T(F) - √(2log(2/δ)/T)
```

This provides a **mathematically rigorous lower bound** accounting for both overfitting and estimation error.

## 🔬 API Reference

### High-Level API

```python
# Main functions
rbt.backtest(portfolio, loader, start_date, end_date, **kwargs) -> BacktestResult
rbt.analyze_ras(returns, confidence=0.99, n_strategies=1, **kwargs) -> RASReport
rbt.create_portfolio(allocations: dict, name: str) -> PortfolioDefinition

# Data loaders
rbt.DataFrameLoader(prices, risk_free_rate=None, fama_french=None)
rbt.CSVLoader(directory_path)
rbt.PostgreSQLLoader(database_url)  # Requires [postgres] extra

# Analytics
rbt.PerformanceCalculator().calculate(returns) -> PerformanceMetrics
rbt.RiskAnalyzer()  # For drawdown and risk analysis

# Visualization (requires [viz] extra)
rbt.BacktestCharts()
rbt.RASVisualization()
rbt.ReturnHeatmap()
```

### Core Engine (Advanced)

```python
from rademacher_backtest import BacktestEngine, BacktestConfig, RASConfig

config = BacktestConfig(
    start_date=date(2015, 1, 1),
    end_date=date(2023, 12, 31),
    initial_capital=100_000.0,
    transaction_cost_bps=10.0,
    rebalance_frequency='monthly',
)

ras_config = RASConfig(
    delta=0.01,  # 1 - confidence_level
    n_simulations=20_000,
)

engine = BacktestEngine(config, portfolio, loader=loader)
result = engine.run()
```

## 📦 Package Structure

```
rademacher-backtest/
├── Core dependencies: pandas, numpy, scipy, statsmodels, pydantic
├── Optional extras:
│   ├── [viz]: matplotlib, seaborn
│   ├── [postgres]: sqlalchemy, psycopg2-binary
│   ├── [cli]: click, rich, python-dotenv
│   ├── [notebooks]: jupyter, ipykernel
│   └── [all]: all of the above
```

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🔍 References

### Academic Papers
- Bailey, D. H., & López de Prado, M. (2012). "The Sharpe Ratio Efficient Frontier"
- Bailey, D. H., et al. (2014). "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality"
- Bailey, D. H., et al. (2014). "Pseudomathematics and Financial Charlatanism: The Effects of Backtest Overfitting on Out-of-Sample Performance"

### Books
- López de Prado, M. (2018). *Advances in Financial Machine Learning*, Chapter 8: "The Multiple Testing Problem"
- López de Prado, M. (2018). *Elements of Quantitative Investing*, Chapter 8 (RAS methodology source)

### Theory
- Bartlett, P. L., & Mendelson, S. (2002). "Rademacher and Gaussian Complexities: Risk Bounds and Structural Results"

## 💬 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/rademacher-backtest/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/rademacher-backtest/discussions)
- **Documentation**: See [`examples/`](examples/) notebooks for comprehensive guides

---

**Ready to backtest with statistical rigor?** Install now: `pip install rademacher-backtest`
