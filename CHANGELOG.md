# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-01-12

### 🎉 Major Release - PyPI Package

This release transforms the RAS backtest engine into a library-first PyPI package with flexible data input options.

### ✨ Added

#### Core Features
- **DataFrameLoader**: New primary data loader accepting pandas DataFrames directly
  - No I/O operations required - works with in-memory data
  - Supports optional risk-free rate and Fama-French factors
  - Validates data formats and handles missing data gracefully

- **High-Level API** (`rademacher_backtest.api`):
  - `backtest()`: Simplified interface for running backtests
  - `analyze_ras()`: RAS analysis on returns with one function call
  - `create_portfolio()`: Convert dict allocations to PortfolioDefinition

- **Flexible Data Sources**:
  - `DataFrameLoader`: In-memory DataFrames (primary, zero dependencies)
  - `CSVLoader`: Load from CSV files (included in core)
  - `PostgreSQLLoader`: PostgreSQL database support (optional `[postgres]` extra)

- **Optional Dependency Groups**:
  - `[viz]`: Visualization with matplotlib and seaborn
  - `[postgres]`: PostgreSQL database support
  - `[cli]`: Command-line interface with rich output
  - `[notebooks]`: Jupyter notebook support
  - `[all]`: Install everything

#### Documentation & Examples
- **Jupyter Notebooks**: 4 comprehensive example notebooks
  - `01_basic_backtest.ipynb`: Quick start with DataFrames
  - `02_custom_portfolio.ipynb`: Portfolio comparison
  - `03_ras_analysis.ipynb`: RAS methodology deep dive
  - `04_visualization.ipynb`: Advanced charts and plots

- **Sample Data**: Realistic synthetic data (2010-2024)
  - 5 ETFs: SPY, AGG, GLD, VNQ, EEM
  - Risk-free rates (3-month Treasury approximation)
  - Fama-French factors

- **PyPI-Ready README**: Comprehensive documentation for library usage

#### CLI Enhancements
- Multi-loader support: `--loader-type` option (csv, postgres)
- Graceful degradation for optional dependencies
- Clear error messages when extras not installed

### 🔄 Changed

#### Breaking Changes
- **Package Restructure**:
  - `src/` → `src/rademacher_backtest/`
  - `main.py` → `src/rademacher_backtest/cli.py`
  - Import path changed: `from src.` → `from rademacher_backtest.`

- **BacktestEngine**: No longer uses PostgreSQL by default
  - **Before**: `BacktestEngine(config, portfolio)` (PostgreSQL assumed)
  - **After**: `BacktestEngine(config, portfolio, loader=loader)` (explicit loader required)
  - Clear error message if loader not provided

- **BacktestConfig**: Removed `database_url` parameter
  - Database connection now handled by loader, not config
  - Config focused solely on backtest parameters

- **Data Loader Protocol**: Split into separate files
  - `loader.py`: Protocol definition only
  - `postgres_loader.py`: PostgreSQL implementation (optional extra)
  - `dataframe_loader.py`: In-memory DataFrame loader (core)
  - `csv_loader.py`: CSV file loader (core)

#### Dependencies
- **Core Package**: Minimal dependencies (no database, no visualization)
  - pandas, numpy, scipy, statsmodels, pydantic
  - All data operations work with DataFrames

- **Optional Extras**: Database and visualization now optional
  - Install only what you need
  - Smaller package size for most users
  - Graceful imports (no errors if extras not installed)

#### API Design
- **Library-First**: Primary usage through Python imports, not CLI
- **Dict Support**: Portfolios can be specified as simple dicts
- **String Dates**: Date parameters accept both `date` objects and ISO strings

### 📝 Migration Guide

#### For Existing Users

**Old code (database-coupled)**:
```python
from src.backtest.engine import BacktestEngine
from src.config.settings import BacktestConfig

config = BacktestConfig(database_url="postgresql://...")
engine = BacktestEngine(config, portfolio)  # PostgreSQL by default
result = engine.run()
```

**New code (library-first, DataFrames)**:
```python
import rademacher_backtest as rbt

# Option 1: High-level API with DataFrames
loader = rbt.DataFrameLoader(prices_df)
result = rbt.backtest(
    portfolio={'SPY': 0.6, 'AGG': 0.4},
    loader=loader,
    start_date='2020-01-01',
    end_date='2023-12-31'
)

# Option 2: PostgreSQL (requires [postgres] extra)
from rademacher_backtest.data.postgres_loader import PostgreSQLLoader
loader = PostgreSQLLoader("postgresql://...")
config = rbt.BacktestConfig(...)  # No database_url parameter
engine = rbt.BacktestEngine(config, portfolio, loader=loader)
result = engine.run()
```

#### Installation Changes

**Before**: Required PostgreSQL and all dependencies
```bash
git clone repo
cd ras-backtest
uv sync
```

**After**: Install from PyPI with only needed extras
```bash
# Core only (DataFrame support, no database)
pip install rademacher-backtest

# With PostgreSQL support
pip install rademacher-backtest[postgres]

# Everything
pip install rademacher-backtest[all]
```

#### Import Changes

All imports must be updated:
```python
# Old
from src.backtest.engine import BacktestEngine
from src.ras.bounds import BoundsCalculator

# New
from rademacher_backtest import BacktestEngine
from rademacher_backtest.ras.bounds import BoundsCalculator

# Or use the high-level API
import rademacher_backtest as rbt
```

### 🐛 Fixed
- Import errors when optional dependencies not installed (now graceful)
- CLI version number updated to 1.0.0
- PostgreSQL imports now conditional on extra being installed

### 🧪 Testing
- Added comprehensive DataFrameLoader tests (23 test cases)
- Marked PostgreSQL tests with `@pytest.mark.postgres` marker
- All core tests pass without database dependencies
- Integration tests still available for optional features

### 📦 Package Info
- **Name**: `rademacher-backtest`
- **Version**: 1.0.0
- **Python**: 3.11+
- **License**: MIT
- **PyPI**: Ready for publication

### 🙏 Acknowledgments
- RAS methodology from Chapter 8 of *Elements of Quantitative Investing*
- Implementation based on academic work by Bailey & López de Prado

---

## [0.1.0] - 2025-01-XX (Pre-release)

### Initial Development Version
- Core backtesting engine
- PostgreSQL data loader
- RAS methodology implementation
- CLI interface
- Basic visualization
- Sample portfolio configuration

**Note**: This was a development version not published to PyPI.

---

## Migration Notes

### For Contributors & Developers

#### Development Setup
**Old**:
```bash
git clone repo
uv sync --extra dev
```

**New** (still recommended for development):
```bash
git clone repo
cd rademacher-backtest
uv sync --all-extras  # Install everything for development
```

#### Running Tests
No changes to test commands:
```bash
pytest
pytest --cov=rademacher_backtest
```

#### Building Package
**New** (for PyPI distribution):
```bash
python -m build
```

### Breaking Changes Summary

| Component | Before | After |
|-----------|--------|-------|
| Package name | ras-backtest | rademacher-backtest |
| Import path | `from src.*` | `from rademacher_backtest.*` |
| Data input | PostgreSQL only | DataFrame, CSV, or PostgreSQL |
| BacktestEngine | Auto PostgreSQL | Requires explicit loader |
| BacktestConfig | Has database_url | No database_url field |
| CLI command | `python main.py` | `ras-backtest` (if cli extra installed) |
| Dependencies | All included | Core + optional extras |

### Upgrade Checklist

- [ ] Update all imports from `src.*` to `rademacher_backtest.*`
- [ ] Remove `database_url` from BacktestConfig initialization
- [ ] Create explicit data loader (DataFrameLoader, CSVLoader, or PostgreSQLLoader)
- [ ] Pass loader to BacktestEngine constructor
- [ ] Install optional extras if needed (`[viz]`, `[postgres]`, `[cli]`)
- [ ] Update environment variables (no more DATABASE_URL in config)
- [ ] Test that code works with new API

---

**Questions?** Open an issue on GitHub or see the [migration examples](examples/) for detailed usage patterns.
