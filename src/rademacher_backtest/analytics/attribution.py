"""Factor attribution analysis using Fama-French factors.

Performs regression-based attribution to decompose portfolio returns
into factor exposures and alpha.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FactorAttribution:
    """Factor attribution results from regression analysis.

    Attributes:
        alpha: Annualized alpha (intercept)
        alpha_tstat: t-statistic for alpha
        alpha_pvalue: p-value for alpha
        factor_betas: Beta exposures to each factor
        factor_tstats: t-statistics for each beta
        factor_pvalues: p-values for each beta
        r_squared: R-squared of the regression
        adj_r_squared: Adjusted R-squared
        residual_volatility: Annualized volatility of residuals
        information_ratio: Alpha / residual volatility
        factor_contributions: Attribution to each factor
    """

    alpha: float
    alpha_tstat: float
    alpha_pvalue: float
    factor_betas: dict[str, float]
    factor_tstats: dict[str, float]
    factor_pvalues: dict[str, float]
    r_squared: float
    adj_r_squared: float
    residual_volatility: float
    information_ratio: float
    factor_contributions: dict[str, float]

    def is_alpha_significant(self, threshold: float = 0.05) -> bool:
        """Check if alpha is statistically significant.

        Args:
            threshold: p-value threshold (default 5%)

        Returns:
            True if alpha p-value < threshold
        """
        return self.alpha_pvalue < threshold

    def significant_factors(self, threshold: float = 0.05) -> list[str]:
        """Get list of statistically significant factors.

        Args:
            threshold: p-value threshold (default 5%)

        Returns:
            List of factor names with significant exposures
        """
        return [
            factor
            for factor, pval in self.factor_pvalues.items()
            if pval < threshold
        ]

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "alpha": round(self.alpha, 6),
            "alpha_tstat": round(self.alpha_tstat, 3),
            "alpha_pvalue": round(self.alpha_pvalue, 4),
            "factor_betas": {k: round(v, 4) for k, v in self.factor_betas.items()},
            "factor_tstats": {k: round(v, 3) for k, v in self.factor_tstats.items()},
            "r_squared": round(self.r_squared, 4),
            "adj_r_squared": round(self.adj_r_squared, 4),
            "residual_volatility": round(self.residual_volatility, 4),
            "information_ratio": round(self.information_ratio, 3),
        }

    def summary_lines(self) -> list[str]:
        """Generate formatted summary lines."""
        lines = [
            "=" * 60,
            "FACTOR ATTRIBUTION ANALYSIS",
            "=" * 60,
            "",
            "ALPHA",
            "-" * 40,
            f"Annualized Alpha:    {self.alpha * 100:>10.3f}%",
            f"t-statistic:         {self.alpha_tstat:>10.3f}",
            f"p-value:             {self.alpha_pvalue:>10.4f}",
            f"Significant:         {'Yes' if self.is_alpha_significant() else 'No':>10}",
            "",
            "FACTOR EXPOSURES",
            "-" * 40,
            f"{'Factor':<10} {'Beta':>10} {'t-stat':>10} {'Signif':>10}",
        ]

        for factor in self.factor_betas:
            beta = self.factor_betas[factor]
            tstat = self.factor_tstats[factor]
            pval = self.factor_pvalues[factor]
            sig = "Yes" if pval < 0.05 else "No"
            lines.append(f"{factor:<10} {beta:>10.3f} {tstat:>10.2f} {sig:>10}")

        lines.extend([
            "",
            "MODEL FIT",
            "-" * 40,
            f"R-squared:           {self.r_squared:>10.4f}",
            f"Adj. R-squared:      {self.adj_r_squared:>10.4f}",
            f"Residual Volatility: {self.residual_volatility * 100:>9.2f}%",
            f"Information Ratio:   {self.information_ratio:>10.3f}",
            "",
            "=" * 60,
        ])

        return lines

    def __str__(self) -> str:
        """Return formatted string."""
        return "\n".join(self.summary_lines())


class FactorAttributor:
    """Perform Fama-French factor attribution analysis.

    Uses OLS regression to decompose portfolio returns into
    systematic factor exposures and idiosyncratic alpha.
    """

    # Standard Fama-French factor names
    FACTOR_NAMES = ["mkt", "smb", "hml", "rmw", "cma", "mom"]

    def __init__(
        self,
        periods_per_year: int = 252,
        factors_to_use: list[str] | None = None,
    ) -> None:
        """Initialize the attributor.

        Args:
            periods_per_year: Annualization factor (252 for daily)
            factors_to_use: List of factor names to include.
                           Default uses all 6 factors.
        """
        self.periods_per_year = periods_per_year
        self.factors_to_use = factors_to_use or self.FACTOR_NAMES

    def attribute(
        self,
        portfolio_returns: pd.Series,
        factors: pd.DataFrame,
    ) -> FactorAttribution:
        """Perform factor attribution regression.

        R_p - R_f = alpha + beta_mkt*MKT + beta_smb*SMB + ... + epsilon

        Args:
            portfolio_returns: Series of portfolio returns (decimal form)
            factors: DataFrame with factor returns (decimal form)
                    Must include 'rf' column for risk-free rate

        Returns:
            FactorAttribution with regression results
        """
        # Align dates
        aligned = pd.concat(
            [portfolio_returns.rename("portfolio"), factors],
            axis=1,
        ).dropna()

        if len(aligned) < 30:
            raise ValueError(f"Need at least 30 observations, got {len(aligned)}")

        # Excess portfolio return
        rf = aligned["rf"] if "rf" in aligned.columns else 0.0
        y = aligned["portfolio"] - rf

        # Select factors to use
        available_factors = [
            f for f in self.factors_to_use if f in aligned.columns
        ]
        if not available_factors:
            raise ValueError(f"No factors found. Available: {list(aligned.columns)}")

        X = aligned[available_factors]

        # Add constant for alpha
        X_with_const = pd.DataFrame({"const": 1.0}, index=X.index)
        X_with_const = pd.concat([X_with_const, X], axis=1)

        # OLS regression using numpy
        X_matrix = X_with_const.values
        y_vector = y.values

        # Solve normal equations: beta = (X'X)^-1 X'y
        XtX = X_matrix.T @ X_matrix
        Xty = X_matrix.T @ y_vector
        betas = np.linalg.solve(XtX, Xty)

        # Calculate residuals and statistics
        y_hat = X_matrix @ betas
        residuals = y_vector - y_hat
        n, k = X_matrix.shape

        # R-squared
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_vector - y_vector.mean())**2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k)

        # Standard errors
        mse = ss_res / (n - k)
        var_beta = mse * np.linalg.inv(XtX)
        se_beta = np.sqrt(np.diag(var_beta))

        # t-statistics and p-values
        t_stats = betas / se_beta
        # Two-tailed p-values (approximation using normal distribution)
        from scipy import stats
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - k))

        # Extract results
        alpha_daily = betas[0]
        alpha = alpha_daily * self.periods_per_year  # Annualize
        alpha_tstat = t_stats[0]
        alpha_pvalue = p_values[0]

        factor_betas = dict(zip(available_factors, betas[1:], strict=True))
        factor_tstats = dict(zip(available_factors, t_stats[1:], strict=True))
        factor_pvalues = dict(zip(available_factors, p_values[1:], strict=True))

        # Residual volatility (annualized)
        residual_vol = float(np.std(residuals, ddof=k) * np.sqrt(self.periods_per_year))

        # Information ratio
        ir = alpha / residual_vol if residual_vol > 0 else 0.0

        # Factor contributions (beta * factor mean return, annualized)
        factor_contributions = {
            factor: factor_betas[factor] * float(X[factor].mean()) * self.periods_per_year
            for factor in available_factors
        }

        return FactorAttribution(
            alpha=alpha,
            alpha_tstat=float(alpha_tstat),
            alpha_pvalue=float(alpha_pvalue),
            factor_betas=factor_betas,
            factor_tstats={k: float(v) for k, v in factor_tstats.items()},
            factor_pvalues={k: float(v) for k, v in factor_pvalues.items()},
            r_squared=float(r_squared),
            adj_r_squared=float(adj_r_squared),
            residual_volatility=residual_vol,
            information_ratio=ir,
            factor_contributions=factor_contributions,
        )

    def attribute_capm(
        self,
        portfolio_returns: pd.Series,
        market_returns: pd.Series,
        risk_free_rate: pd.Series | None = None,
    ) -> FactorAttribution:
        """Perform simple CAPM attribution (single factor).

        Args:
            portfolio_returns: Series of portfolio returns
            market_returns: Series of market returns
            risk_free_rate: Optional risk-free rate series

        Returns:
            FactorAttribution with single-factor results
        """
        # Build factors DataFrame
        factors = pd.DataFrame({"mkt": market_returns})
        if risk_free_rate is not None:
            factors["rf"] = risk_free_rate
        else:
            factors["rf"] = 0.0

        # Use only market factor
        attributor = FactorAttributor(
            periods_per_year=self.periods_per_year,
            factors_to_use=["mkt"],
        )
        return attributor.attribute(portfolio_returns, factors)
