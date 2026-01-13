"""Visualization charts for backtest results and RAS analysis.

Provides matplotlib-based visualizations for performance analysis,
drawdowns, monthly returns, and RAS haircut decomposition.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

if TYPE_CHECKING:
    from rademacher_backtest.ras.report import RASReport


class BacktestCharts:
    """Generate backtest visualization charts."""

    def __init__(
        self,
        figsize: tuple[int, int] = (12, 6),
        style: str = "seaborn-v0_8-whitegrid",
    ) -> None:
        """Initialize chart generator.

        Args:
            figsize: Default figure size (width, height)
            style: Matplotlib style to use
        """
        self.figsize = figsize
        try:
            plt.style.use(style)
        except OSError:
            plt.style.use("seaborn-v0_8")

    def cumulative_returns(
        self,
        values: pd.Series,
        benchmark: pd.Series | None = None,
        title: str = "Cumulative Portfolio Performance",
        normalize: bool = True,
    ) -> plt.Figure:
        """Plot cumulative returns chart.

        Args:
            values: Portfolio value series
            benchmark: Optional benchmark series for comparison
            title: Chart title
            normalize: Whether to normalize to 100 at start

        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        if normalize:
            portfolio = (values / values.iloc[0]) * 100
            label_suffix = " (Starting = 100)"
        else:
            portfolio = values
            label_suffix = ""

        ax.plot(
            portfolio.index,
            portfolio.values,
            label="Portfolio",
            linewidth=2,
            color="#2E86AB",
        )

        if benchmark is not None:
            if normalize:
                bench = (benchmark / benchmark.iloc[0]) * 100
            else:
                bench = benchmark
            ax.plot(
                bench.index,
                bench.values,
                label="Benchmark",
                linewidth=1.5,
                linestyle="--",
                color="#A23B72",
            )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel(f"Value{label_suffix}")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig

    def drawdown(
        self,
        returns: pd.Series,
        title: str = "Underwater Chart (Drawdowns)",
    ) -> plt.Figure:
        """Plot drawdown/underwater chart.

        Args:
            returns: Series of returns
            title: Chart title

        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = (cum_returns / rolling_max - 1) * 100

        ax.fill_between(
            drawdowns.index,
            drawdowns.values,
            0,
            color="#E74C3C",
            alpha=0.3,
        )
        ax.plot(
            drawdowns.index,
            drawdowns.values,
            color="#E74C3C",
            linewidth=1,
        )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown (%)")
        ax.grid(True, alpha=0.3)

        # Add max drawdown annotation
        min_idx = drawdowns.idxmin()
        min_val = drawdowns.min()
        ax.annotate(
            f"Max: {min_val:.1f}%",
            xy=(min_idx, min_val),
            xytext=(10, -20),
            textcoords="offset points",
            fontsize=10,
            color="#E74C3C",
            fontweight="bold",
        )

        fig.tight_layout()
        return fig

    def rolling_sharpe(
        self,
        returns: pd.Series,
        window: int = 252,
        title: str = "Rolling Sharpe Ratio (1 Year)",
    ) -> plt.Figure:
        """Plot rolling Sharpe ratio.

        Args:
            returns: Series of returns
            window: Rolling window size
            title: Chart title

        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        rolling_mean = returns.rolling(window).mean() * 252
        rolling_std = returns.rolling(window).std() * np.sqrt(252)
        rolling_sharpe = rolling_mean / rolling_std

        ax.plot(
            rolling_sharpe.index,
            rolling_sharpe.values,
            color="#2E86AB",
            linewidth=1.5,
        )
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.axhline(
            y=rolling_sharpe.mean(),
            color="#27AE60",
            linestyle="--",
            linewidth=1,
            label=f"Average: {rolling_sharpe.mean():.2f}",
        )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Sharpe Ratio")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig

    def portfolio_weights(
        self,
        weights_df: pd.DataFrame,
        title: str = "Portfolio Weights Over Time",
    ) -> plt.Figure:
        """Plot stacked area chart of portfolio weights.

        Args:
            weights_df: DataFrame with dates as index, tickers as columns
            title: Chart title

        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Resample to monthly for cleaner visualization
        weights_monthly = weights_df.resample("ME").last()

        ax.stackplot(
            weights_monthly.index,
            weights_monthly.T.values,
            labels=weights_monthly.columns,
            alpha=0.8,
        )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Weight")
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig

    def save_figure(
        self,
        fig: plt.Figure,
        filepath: str | Path,
        dpi: int = 150,
    ) -> None:
        """Save figure to file.

        Args:
            fig: Figure to save
            filepath: Output path
            dpi: Resolution
        """
        fig.savefig(filepath, dpi=dpi, bbox_inches="tight")


class ReturnHeatmap:
    """Monthly returns heatmap visualization."""

    def __init__(
        self,
        figsize: tuple[int, int] = (14, 8),
        cmap: str = "RdYlGn",
    ) -> None:
        """Initialize heatmap generator.

        Args:
            figsize: Figure size
            cmap: Colormap for heatmap
        """
        self.figsize = figsize
        self.cmap = cmap

    def plot(
        self,
        returns: pd.Series,
        title: str = "Monthly Returns Heatmap (%)",
    ) -> plt.Figure:
        """Generate monthly returns heatmap.

        Args:
            returns: Series of daily returns
            title: Chart title

        Returns:
            Matplotlib Figure object
        """
        # Aggregate to monthly returns
        monthly = returns.resample("ME").apply(
            lambda x: (1 + x).prod() - 1
        ) * 100

        # Create year x month matrix
        df = pd.DataFrame({
            "year": monthly.index.year,
            "month": monthly.index.month,
            "return": monthly.values,
        })
        pivot = df.pivot(index="year", columns="month", values="return")
        pivot.columns = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
        ]

        # Add annual returns column
        annual = returns.resample("YE").apply(lambda x: (1 + x).prod() - 1) * 100
        annual_df = pd.DataFrame({"Year": annual.values}, index=annual.index.year)

        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot heatmap
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".1f",
            cmap=self.cmap,
            center=0,
            ax=ax,
            cbar_kws={"label": "Return (%)"},
            linewidths=0.5,
        )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Month")
        ax.set_ylabel("Year")

        fig.tight_layout()
        return fig

    def plot_with_annual(
        self,
        returns: pd.Series,
        title: str = "Monthly Returns Heatmap (%)",
    ) -> plt.Figure:
        """Generate heatmap with annual returns column.

        Args:
            returns: Series of daily returns
            title: Chart title

        Returns:
            Matplotlib Figure object
        """
        # Aggregate to monthly and annual returns
        monthly = returns.resample("ME").apply(
            lambda x: (1 + x).prod() - 1
        ) * 100

        annual = returns.resample("YE").apply(
            lambda x: (1 + x).prod() - 1
        ) * 100

        # Create matrix with months + annual column
        df = pd.DataFrame({
            "year": monthly.index.year,
            "month": monthly.index.month,
            "return": monthly.values,
        })
        pivot = df.pivot(index="year", columns="month", values="return")
        pivot.columns = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
        ]

        # Add annual column
        pivot["Year"] = annual.groupby(annual.index.year).first().values

        fig, ax = plt.subplots(figsize=(self.figsize[0] + 1, self.figsize[1]))

        sns.heatmap(
            pivot,
            annot=True,
            fmt=".1f",
            cmap=self.cmap,
            center=0,
            ax=ax,
            cbar_kws={"label": "Return (%)"},
            linewidths=0.5,
        )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Year")

        fig.tight_layout()
        return fig


class RASVisualization:
    """RAS-specific visualizations."""

    def __init__(self, figsize: tuple[int, int] = (14, 5)) -> None:
        """Initialize RAS visualization.

        Args:
            figsize: Default figure size
        """
        self.figsize = figsize

    def haircut_decomposition(
        self,
        ras_report: RASReport,
        title: str = "RAS Sharpe Ratio Analysis",
    ) -> plt.Figure:
        """Visualize haircut decomposition.

        Shows empirical Sharpe, haircut components, and adjusted Sharpe
        as a waterfall-style chart.

        Args:
            ras_report: RAS analysis report
            title: Chart title

        Returns:
            Matplotlib Figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)

        # Left: Waterfall chart of haircut
        categories = [
            "Empirical\nSharpe",
            "Data\nSnooping",
            "Estimation\nError",
            "Adjusted\nSharpe",
        ]
        values = [
            ras_report.empirical_sharpe_annualized,
            -ras_report.data_snooping_haircut * np.sqrt(252),
            -ras_report.estimation_haircut * np.sqrt(252),
            ras_report.adjusted_sharpe_annualized,
        ]
        colors = ["#27AE60", "#E74C3C", "#F39C12", "#2E86AB"]

        bars = axes[0].bar(categories, values, color=colors, edgecolor="black", linewidth=0.5)
        axes[0].axhline(y=0, color="black", linestyle="-", linewidth=0.5)

        # Add value labels
        for bar, val in zip(bars, values, strict=True):
            height = bar.get_height()
            axes[0].annotate(
                f"{val:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3 if height >= 0 else -15),
                textcoords="offset points",
                ha="center",
                va="bottom" if height >= 0 else "top",
                fontsize=10,
                fontweight="bold",
            )

        axes[0].set_title("Sharpe Ratio Adjustment", fontweight="bold")
        axes[0].set_ylabel("Annualized Sharpe Ratio")
        axes[0].grid(True, alpha=0.3, axis="y")

        # Right: Pie chart of haircut sources
        haircut_values = [
            ras_report.data_snooping_haircut,
            ras_report.estimation_haircut,
        ]
        haircut_labels = ["Data Snooping\n(2 × R)", "Estimation\nError"]
        colors_pie = ["#E74C3C", "#F39C12"]

        wedges, texts, autotexts = axes[1].pie(
            haircut_values,
            labels=haircut_labels,
            autopct="%1.1f%%",
            colors=colors_pie,
            startangle=90,
            explode=(0.02, 0.02),
        )
        axes[1].set_title("Haircut Source Breakdown", fontweight="bold")

        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
        fig.tight_layout()
        return fig

    def confidence_comparison(
        self,
        bounds_dict: dict,
        title: str = "Sharpe Ratio Bounds at Different Confidence Levels",
    ) -> plt.Figure:
        """Compare bounds at different confidence levels.

        Args:
            bounds_dict: Dict mapping delta to SharpeRatioBounds
            title: Chart title

        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        deltas = sorted(bounds_dict.keys())
        empirical = bounds_dict[deltas[0]].annualized_empirical
        lower_bounds = [bounds_dict[d].annualized_lower_bound for d in deltas]
        confidence_levels = [(1 - d) * 100 for d in deltas]

        # Plot bounds
        ax.barh(
            range(len(deltas)),
            lower_bounds,
            color="#2E86AB",
            alpha=0.7,
            label="Lower Bound",
        )

        # Add empirical line
        ax.axvline(
            x=empirical,
            color="#27AE60",
            linestyle="--",
            linewidth=2,
            label=f"Empirical: {empirical:.3f}",
        )
        ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)

        # Labels
        ax.set_yticks(range(len(deltas)))
        ax.set_yticklabels([f"{c:.0f}%" for c in confidence_levels])
        ax.set_ylabel("Confidence Level")
        ax.set_xlabel("Annualized Sharpe Ratio")
        ax.set_title(title, fontweight="bold")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3, axis="x")

        fig.tight_layout()
        return fig

    def summary_dashboard(
        self,
        ras_report: RASReport,
        performance_metrics: dict,
        title: str = "RAS Backtest Summary",
    ) -> plt.Figure:
        """Create summary dashboard with key metrics.

        Args:
            ras_report: RAS analysis report
            performance_metrics: Dictionary of performance metrics
            title: Dashboard title

        Returns:
            Matplotlib Figure object
        """
        fig = plt.figure(figsize=(14, 8))

        # Create grid
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # Panel 1: Key metrics
        ax1 = fig.add_subplot(gs[0, 0])
        metrics_text = [
            f"CAGR: {performance_metrics.get('CAGR (%)', 0):.2f}%",
            f"Volatility: {performance_metrics.get('Annualized Volatility (%)', 0):.2f}%",
            f"Max Drawdown: {performance_metrics.get('Max Drawdown (%)', 0):.2f}%",
            f"",
            f"Empirical Sharpe: {ras_report.empirical_sharpe_annualized:.3f}",
            f"RAS-Adjusted Sharpe: {ras_report.adjusted_sharpe_annualized:.3f}",
        ]
        ax1.text(
            0.1, 0.9, "\n".join(metrics_text),
            transform=ax1.transAxes,
            fontsize=11,
            verticalalignment="top",
            fontfamily="monospace",
        )
        ax1.axis("off")
        ax1.set_title("Key Metrics", fontweight="bold")

        # Panel 2: RAS Analysis
        ax2 = fig.add_subplot(gs[0, 1])
        ras_text = [
            f"Confidence Level: {ras_report.confidence_level:.0f}%",
            f"Sample Size: {ras_report.sample_size:,}",
            f"",
            f"Rademacher Complexity: {ras_report.rademacher_complexity:.6f}",
            f"Data Snooping Haircut: {ras_report.data_snooping_haircut:.6f}",
            f"Estimation Haircut: {ras_report.estimation_haircut:.6f}",
            f"",
            f"Statistically Positive: {'Yes' if ras_report.is_statistically_positive else 'No'}",
        ]
        ax2.text(
            0.1, 0.9, "\n".join(ras_text),
            transform=ax2.transAxes,
            fontsize=11,
            verticalalignment="top",
            fontfamily="monospace",
        )
        ax2.axis("off")
        ax2.set_title("RAS Analysis", fontweight="bold")

        # Panel 3: Interpretation
        ax3 = fig.add_subplot(gs[0, 2])
        # Word wrap the interpretation
        interpretation = ras_report.interpretation
        wrapped = "\n".join([
            interpretation[i:i+40]
            for i in range(0, len(interpretation), 40)
        ])
        ax3.text(
            0.1, 0.9, wrapped,
            transform=ax3.transAxes,
            fontsize=10,
            verticalalignment="top",
            wrap=True,
        )
        ax3.axis("off")
        ax3.set_title("Interpretation", fontweight="bold")

        # Panel 4-6: Haircut chart (spans bottom row)
        ax4 = fig.add_subplot(gs[1, :])
        categories = ["Empirical", "- Data Snooping", "- Estimation", "= Adjusted"]
        values = [
            ras_report.empirical_sharpe_annualized,
            -ras_report.data_snooping_haircut * np.sqrt(252),
            -ras_report.estimation_haircut * np.sqrt(252),
            ras_report.adjusted_sharpe_annualized,
        ]
        colors = ["#27AE60", "#E74C3C", "#F39C12", "#2E86AB"]

        bars = ax4.bar(categories, values, color=colors, edgecolor="black")
        ax4.axhline(y=0, color="black", linewidth=0.5)
        ax4.set_ylabel("Annualized Sharpe Ratio")
        ax4.set_title("Sharpe Ratio Haircut Decomposition", fontweight="bold")
        ax4.grid(True, alpha=0.3, axis="y")

        for bar, val in zip(bars, values, strict=True):
            ax4.annotate(
                f"{val:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, val),
                xytext=(0, 5 if val >= 0 else -15),
                textcoords="offset points",
                ha="center",
                fontweight="bold",
            )

        fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
        return fig
