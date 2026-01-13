#!/usr/bin/env python3
"""RAS Backtest - Main CLI entry point.

A comprehensive backtesting engine implementing the Rademacher Anti-Serum (RAS)
methodology from Chapter 8 of "Elements of Quantitative Investing".

Usage:
    python main.py run                    # Run backtest with defaults
    python main.py run --output results/  # Save results to directory
    python main.py info                   # Show portfolio and config info
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import click
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from rademacher_backtest.analytics.performance import PerformanceCalculator
from rademacher_backtest.backtest.engine import BacktestEngine
from rademacher_backtest.config.portfolio import SAMPLE_PORTFOLIO
from rademacher_backtest.config.settings import BacktestConfig, RASConfig
from rademacher_backtest.data.csv_loader import CSVLoader
from rademacher_backtest.ras.bounds import BoundsCalculator
from rademacher_backtest.ras.report import RASReportGenerator
from rademacher_backtest.visualization.charts import BacktestCharts, RASVisualization, ReturnHeatmap

console = Console()


def create_summary_table(
    backtest_summary: dict,
    performance_dict: dict,
    ras_report: dict,
) -> Table:
    """Create rich table with backtest summary."""
    table = Table(title="Backtest Summary", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="dim", width=30)
    table.add_column("Value", justify="right")

    # Backtest info
    table.add_row("[bold]BACKTEST INFO[/bold]", "")
    table.add_row("Portfolio", backtest_summary["portfolio_name"])
    table.add_row("Period", f"{backtest_summary['start_date']} to {backtest_summary['end_date']}")
    table.add_row("Trading Days", f"{backtest_summary['trading_days']:,}")
    table.add_row("Years", f"{backtest_summary['years']:.1f}")
    table.add_row("Rebalances", str(backtest_summary["num_rebalances"]))
    table.add_row("Transaction Costs", f"${backtest_summary['total_transaction_costs']:,.2f}")
    table.add_row("", "")

    # Returns
    table.add_row("[bold]RETURNS[/bold]", "")
    table.add_row("Initial Value", f"${backtest_summary['initial_value']:,.2f}")
    table.add_row("Final Value", f"${backtest_summary['final_value']:,.2f}")
    table.add_row("Total Return", f"{backtest_summary['total_return_pct']:.2f}%")
    table.add_row("CAGR", f"{performance_dict['CAGR (%)']:.2f}%")
    table.add_row("", "")

    # Risk
    table.add_row("[bold]RISK METRICS[/bold]", "")
    table.add_row("Volatility (Ann.)", f"{performance_dict['Annualized Volatility (%)']:.2f}%")
    table.add_row("Max Drawdown", f"{performance_dict['Max Drawdown (%)']:.2f}%")
    table.add_row("VaR (95%)", f"{performance_dict['VaR 95% (%)']:.2f}%")
    table.add_row("", "")

    # Risk-Adjusted
    table.add_row("[bold]RISK-ADJUSTED[/bold]", "")
    table.add_row("Sharpe Ratio", f"{performance_dict['Sharpe Ratio']:.3f}")
    table.add_row("Sortino Ratio", f"{performance_dict['Sortino Ratio']:.3f}")
    table.add_row("Calmar Ratio", f"{performance_dict['Calmar Ratio']:.3f}")
    table.add_row("", "")

    # RAS Analysis
    table.add_row("[bold]RAS ANALYSIS[/bold]", "")
    table.add_row("Empirical Sharpe", f"{ras_report['empirical_sharpe_annualized']:.3f}")
    table.add_row("RAS-Adjusted Sharpe", f"{ras_report['adjusted_sharpe_annualized']:.3f}")
    table.add_row("Total Haircut", f"{ras_report['total_haircut_annualized']:.3f}")
    table.add_row("Confidence Level", f"{ras_report['confidence_level']:.0f}%")
    table.add_row(
        "Statistically Positive",
        "[green]Yes[/green]" if ras_report["is_statistically_positive"] else "[red]No[/red]",
    )

    return table


@click.group()
@click.version_option(version="1.0.0")
def cli() -> None:
    """RAS Backtest - Backtesting with Rademacher Anti-Serum methodology."""
    pass


@cli.command()
@click.option(
    "--loader-type",
    type=click.Choice(["csv", "postgres"]),
    default="csv",
    help="Data source type (csv or postgres)",
)
@click.option(
    "--csv-dir",
    type=click.Path(exists=True),
    default=None,
    help="Directory containing CSV data files (for csv loader)",
)
@click.option(
    "--database-url",
    type=str,
    default=None,
    envvar="DATABASE_URL",
    help="PostgreSQL connection string (for postgres loader)",
)
@click.option(
    "--start-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default="2010-01-14",
    help="Backtest start date (YYYY-MM-DD)",
)
@click.option(
    "--end-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default="2025-12-30",
    help="Backtest end date (YYYY-MM-DD)",
)
@click.option(
    "--initial-capital",
    type=float,
    default=100_000.0,
    help="Initial portfolio value",
)
@click.option(
    "--transaction-cost",
    type=float,
    default=10.0,
    help="Transaction cost in basis points (10 = 0.1%)",
)
@click.option(
    "--confidence",
    type=float,
    default=0.99,
    help="RAS confidence level (e.g., 0.99 for 99%)",
)
@click.option(
    "--output",
    type=click.Path(),
    default=None,
    help="Output directory for results and charts",
)
@click.option(
    "--no-charts",
    is_flag=True,
    help="Skip chart generation",
)
def run(
    loader_type: str,
    csv_dir: str | None,
    database_url: str | None,
    start_date,
    end_date,
    initial_capital: float,
    transaction_cost: float,
    confidence: float,
    output: str | None,
    no_charts: bool,
) -> None:
    """Run the backtest with RAS analysis."""
    console.print(
        Panel.fit(
            "[bold blue]RAS Backtest Engine[/bold blue]\nRademacher Anti-Serum Methodology",
            border_style="blue",
        )
    )

    # Create data loader based on type
    if loader_type == "csv":
        if csv_dir is None:
            csv_dir = "./data"  # Default CSV directory
        try:
            loader = CSVLoader(csv_dir)
        except Exception as e:
            console.print(f"[red]Error loading CSV data: {e}[/red]")
            console.print(f"[dim]CSV directory: {csv_dir}[/dim]")
            sys.exit(1)
    elif loader_type == "postgres":
        try:
            from rademacher_backtest.data.postgres_loader import PostgreSQLLoader
        except ImportError:
            console.print(
                "[red]PostgreSQL support not installed.[/red]\n"
                "[dim]Install with: pip install rademacher-backtest[postgres][/dim]"
            )
            sys.exit(1)

        try:
            loader = PostgreSQLLoader(database_url)
        except Exception as e:
            console.print(f"[red]Error connecting to database: {e}[/red]")
            sys.exit(1)
    else:
        console.print(f"[red]Unknown loader type: {loader_type}[/red]")
        sys.exit(1)

    # Create configurations
    backtest_config = BacktestConfig(
        start_date=start_date.date()
        if hasattr(start_date, "date")
        else date.fromisoformat(str(start_date)[:10]),
        end_date=end_date.date()
        if hasattr(end_date, "date")
        else date.fromisoformat(str(end_date)[:10]),
        initial_capital=initial_capital,
        transaction_cost_bps=transaction_cost,
    )

    ras_config = RASConfig(
        delta=1 - confidence,
        n_simulations=20_000,
    )

    portfolio = SAMPLE_PORTFOLIO

    # Output directory setup
    if output:
        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Step 1: Run backtest
        task = progress.add_task("Running backtest...", total=None)

        try:
            engine = BacktestEngine(backtest_config, portfolio, loader=loader)
            result = engine.run()
            progress.update(task, description="[green]Backtest complete!")
        except Exception as e:
            console.print(f"[red]Error running backtest: {e}[/red]")
            sys.exit(1)

        # Step 2: Calculate performance metrics
        progress.update(task, description="Calculating performance metrics...")

        try:
            perf_calc = PerformanceCalculator()
            metrics = perf_calc.calculate(result.daily_returns)
        except Exception as e:
            console.print(f"[red]Error calculating metrics: {e}[/red]")
            sys.exit(1)

        # Step 3: RAS analysis
        progress.update(task, description="Running RAS analysis...")

        try:
            bounds_calc = BoundsCalculator(ras_config)
            bounds = bounds_calc.calculate_sharpe_bounds(result.daily_returns.values)

            report_gen = RASReportGenerator()
            ras_report = report_gen.generate(bounds, len(result.daily_returns), N=1)
        except Exception as e:
            console.print(f"[red]Error in RAS analysis: {e}[/red]")
            sys.exit(1)

        # Step 4: Generate charts (optional)
        if not no_charts and output_path:
            progress.update(task, description="Generating charts...")

            try:
                charts = BacktestCharts()

                # Cumulative returns
                fig = charts.cumulative_returns(result.daily_values)
                charts.save_figure(fig, output_path / "cumulative_returns.png")

                # Drawdown
                fig = charts.drawdown(result.daily_returns)
                charts.save_figure(fig, output_path / "drawdowns.png")

                # Monthly heatmap
                heatmap = ReturnHeatmap()
                fig = heatmap.plot(result.daily_returns)
                charts.save_figure(fig, output_path / "monthly_returns.png")

                # RAS analysis
                ras_viz = RASVisualization()
                fig = ras_viz.haircut_decomposition(ras_report)
                charts.save_figure(fig, output_path / "ras_analysis.png")

            except Exception as e:
                console.print(f"[yellow]Warning: Could not generate charts: {e}[/yellow]")

        progress.update(task, description="[green]Analysis complete!")

    # Display results
    console.print()
    table = create_summary_table(
        result.summary(),
        metrics.to_dict(),
        ras_report.to_dict(),
    )
    console.print(table)

    # RAS interpretation
    console.print()
    console.print(
        Panel(
            ras_report.interpretation,
            title="[bold]RAS Interpretation[/bold]",
            border_style="green" if ras_report.is_statistically_positive else "yellow",
        )
    )

    # Output path info
    if output_path:
        console.print()
        console.print(f"[dim]Results saved to: {output_path.absolute()}[/dim]")


@cli.command()
def info() -> None:
    """Display portfolio and configuration information."""
    console.print(
        Panel.fit(
            "[bold blue]Portfolio Information[/bold blue]",
            border_style="blue",
        )
    )

    # Portfolio info
    table = Table(title=SAMPLE_PORTFOLIO.name, show_header=True, header_style="bold cyan")
    table.add_column("Ticker", style="bold")
    table.add_column("Asset Class")
    table.add_column("Weight", justify="right")

    for alloc in SAMPLE_PORTFOLIO.allocations:
        table.add_row(alloc.ticker, alloc.name, f"{alloc.weight:.0%}")

    console.print(table)

    # Default config info
    console.print()
    config = BacktestConfig()
    console.print("[bold]Default Configuration:[/bold]")
    console.print(f"  Start Date: {config.start_date}")
    console.print(f"  End Date: {config.end_date}")
    console.print(f"  Initial Capital: ${config.initial_capital:,.2f}")
    console.print(f"  Rebalance Frequency: {config.rebalance_frequency}")
    console.print(
        f"  Transaction Cost: {config.transaction_cost_bps} bps ({config.transaction_cost_rate:.2%})"
    )

    ras_config = RASConfig()
    console.print()
    console.print("[bold]RAS Configuration:[/bold]")
    console.print(f"  Confidence Level: {ras_config.confidence_level:.0f}%")
    console.print(f"  Monte Carlo Simulations: {ras_config.n_simulations:,}")


@cli.command()
@click.option(
    "--database-url",
    type=str,
    default=None,
    envvar="DATABASE_URL",
    help="PostgreSQL connection string",
)
def check_data(database_url: str | None) -> None:
    """Check PostgreSQL database connection and available data.

    Note: This command requires the 'postgres' extra to be installed.
    Install with: pip install rademacher-backtest[postgres]
    """
    console.print("[bold]Checking database connection...[/bold]")

    try:
        from rademacher_backtest.data.postgres_loader import PostgreSQLLoader
    except ImportError:
        console.print(
            "[red]PostgreSQL support not installed.[/red]\n"
            "[dim]Install with: pip install rademacher-backtest[postgres][/dim]"
        )
        sys.exit(1)

    try:
        loader = PostgreSQLLoader(database_url)

        # Check each ticker
        tickers = SAMPLE_PORTFOLIO.tickers
        table = Table(title="Data Availability", show_header=True)
        table.add_column("Ticker")
        table.add_column("Start Date")
        table.add_column("End Date")
        table.add_column("Status")

        all_ok = True
        for ticker in tickers:
            try:
                date_range = loader.get_data_range(ticker)
                if date_range:
                    start, end = date_range
                    table.add_row(ticker, str(start), str(end), "[green]OK[/green]")
                else:
                    table.add_row(ticker, "-", "-", "[red]Not Found[/red]")
                    all_ok = False
            except Exception as e:
                table.add_row(ticker, "-", "-", f"[red]Error: {e}[/red]")
                all_ok = False

        console.print(table)

        if all_ok:
            console.print("\n[green]All data available! Ready to run backtest.[/green]")
        else:
            console.print("\n[yellow]Some data is missing. Check database.[/yellow]")

    except Exception as e:
        console.print(f"[red]Database connection failed: {e}[/red]")
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
