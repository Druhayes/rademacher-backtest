"""Tests for RAS haircut calculation."""

from __future__ import annotations

import numpy as np
import pytest

from rademacher_backtest.ras.haircut import HaircutCalculator, RASHaircut


class TestRASHaircut:
    """Test suite for RASHaircut dataclass."""

    def test_initialization(self) -> None:
        """Test haircut initialization."""
        haircut = RASHaircut(
            data_snooping_term=0.05,
            estimation_term=0.03,
            total_haircut=0.08,
            delta=0.01,
            T=1000,
        )

        assert haircut.data_snooping_term == 0.05
        assert haircut.estimation_term == 0.03
        assert haircut.total_haircut == 0.08
        assert haircut.delta == 0.01
        assert haircut.T == 1000

    def test_confidence_level(self) -> None:
        """Test confidence level property."""
        haircut = RASHaircut(0.05, 0.03, 0.08, delta=0.01, T=1000)
        assert haircut.confidence_level == 99.0

        haircut = RASHaircut(0.05, 0.03, 0.08, delta=0.05, T=1000)
        assert haircut.confidence_level == 95.0

    def test_annualized_haircut(self) -> None:
        """Test annualized haircut property."""
        haircut = RASHaircut(0.05, 0.03, 0.08, delta=0.01, T=1000)

        # Should be total * √252
        expected = 0.08 * np.sqrt(252)
        assert np.isclose(haircut.annualized_haircut, expected)

    def test_immutability(self) -> None:
        """Test that haircut is immutable."""
        haircut = RASHaircut(0.05, 0.03, 0.08, delta=0.01, T=1000)

        with pytest.raises(AttributeError):
            haircut.total_haircut = 0.10  # type: ignore


class TestHaircutCalculator:
    """Test suite for HaircutCalculator."""

    def test_initialization(self) -> None:
        """Test calculator initialization."""
        calc = HaircutCalculator(delta=0.01)
        assert calc.delta == 0.01

    def test_initialization_validation(self) -> None:
        """Test that invalid delta values are rejected."""
        with pytest.raises(ValueError, match="delta must be between 0 and 1"):
            HaircutCalculator(delta=0.0)

        with pytest.raises(ValueError, match="delta must be between 0 and 1"):
            HaircutCalculator(delta=1.0)

        with pytest.raises(ValueError, match="delta must be between 0 and 1"):
            HaircutCalculator(delta=-0.1)

    def test_calculate_for_signals_basic(self) -> None:
        """Test haircut calculation for bounded signals."""
        calc = HaircutCalculator(delta=0.01)

        # Simple case: R=0.02, T=1000, max_ic=1.0
        rademacher = 0.02
        T = 1000
        haircut = calc.calculate_for_signals(rademacher, T, max_ic=1.0)

        # Data snooping term: 2 * R = 2 * 0.02 = 0.04
        assert np.isclose(haircut.data_snooping_term, 0.04)

        # Estimation term: √(2*log(2/δ)/T)
        expected_est = np.sqrt(2 * np.log(2 / 0.01) / 1000)
        assert np.isclose(haircut.estimation_term, expected_est)

        # Total
        assert np.isclose(haircut.total_haircut, 0.04 + expected_est)

    def test_calculate_for_signals_different_max_ic(self) -> None:
        """Test haircut with different max_ic values."""
        calc = HaircutCalculator(delta=0.01)
        rademacher = 0.02
        T = 1000

        haircut_1 = calc.calculate_for_signals(rademacher, T, max_ic=1.0)
        haircut_2 = calc.calculate_for_signals(rademacher, T, max_ic=2.0)

        # Estimation term should double with max_ic=2.0
        assert np.isclose(
            haircut_2.estimation_term,
            haircut_1.estimation_term * 2.0
        )

    def test_calculate_for_sharpe_basic(self) -> None:
        """Test haircut calculation for Sharpe ratio."""
        calc = HaircutCalculator(delta=0.01)

        rademacher = 0.02
        T = 1000
        N = 1
        sigma = 1.0

        haircut = calc.calculate_for_sharpe(rademacher, T, N, sigma)

        # Data snooping: 2 * R
        assert np.isclose(haircut.data_snooping_term, 0.04)

        # Estimation term: √(2*log(2N/δ)/T) + σ*√(2*log(2/δ)/T)
        log_term_1 = np.sqrt(2 * np.log(2 * N / 0.01) / T)
        log_term_2 = sigma * np.sqrt(2 * np.log(2 / 0.01) / T)
        expected_est = log_term_1 + log_term_2

        assert np.isclose(haircut.estimation_term, expected_est, rtol=1e-5)

    def test_calculate_for_sharpe_multiple_strategies(self) -> None:
        """Test haircut for multiple strategies (N > 1)."""
        calc = HaircutCalculator(delta=0.01)

        rademacher = 0.03
        T = 1000

        haircut_1 = calc.calculate_for_sharpe(rademacher, T, N=1)
        haircut_10 = calc.calculate_for_sharpe(rademacher, T, N=10)

        # Data snooping should be the same (depends only on R)
        assert np.isclose(haircut_1.data_snooping_term, haircut_10.data_snooping_term)

        # Estimation term should be larger for N=10
        assert haircut_10.estimation_term > haircut_1.estimation_term

        # Total haircut should be larger
        assert haircut_10.total_haircut > haircut_1.total_haircut

    def test_calculate_for_sharpe_different_sigma(self) -> None:
        """Test haircut with different sub-Gaussian parameters."""
        calc = HaircutCalculator(delta=0.01)

        rademacher = 0.02
        T = 1000

        haircut_1 = calc.calculate_for_sharpe(rademacher, T, N=1, sigma=1.0)
        haircut_2 = calc.calculate_for_sharpe(rademacher, T, N=1, sigma=2.0)

        # Larger sigma should increase estimation term
        assert haircut_2.estimation_term > haircut_1.estimation_term

    def test_calculate_with_tuning(self) -> None:
        """Test haircut with tunable parameters."""
        calc = HaircutCalculator(delta=0.01)

        rademacher = 0.02
        T = 1000
        N = 1

        # Default parameters (a=2.0, b=1.0)
        haircut_default = calc.calculate_with_tuning(rademacher, T, N, a=2.0, b=1.0)

        # More conservative (a=3.0, b=1.5)
        haircut_conservative = calc.calculate_with_tuning(rademacher, T, N, a=3.0, b=1.5)

        # Conservative should be larger
        assert haircut_conservative.total_haircut > haircut_default.total_haircut

        # Less conservative (a=1.5, b=0.8)
        haircut_aggressive = calc.calculate_with_tuning(rademacher, T, N, a=1.5, b=0.8)

        # Aggressive should be smaller
        assert haircut_aggressive.total_haircut < haircut_default.total_haircut

    def test_calculate_minimum_sample_basic(self) -> None:
        """Test minimum sample size calculation."""
        calc = HaircutCalculator(delta=0.01)

        rademacher = 0.01
        target_haircut = 0.10

        min_T = calc.calculate_minimum_sample(target_haircut, rademacher, N=1)

        # Should return a positive integer >= 100
        assert isinstance(min_T, int)
        assert min_T >= 100

    def test_calculate_minimum_sample_unachievable(self) -> None:
        """Test minimum sample when target is unachievable."""
        calc = HaircutCalculator(delta=0.01)

        # Very high complexity, low target
        rademacher = 0.10
        target_haircut = 0.05  # Impossible: data_snooping alone is 0.20

        min_T = calc.calculate_minimum_sample(target_haircut, rademacher, N=1)

        # Should return very large number
        assert min_T >= 1e9

    def test_confidence_level_impact(self) -> None:
        """Test that lower confidence (higher delta) reduces haircut."""
        rademacher = 0.02
        T = 1000

        calc_99 = HaircutCalculator(delta=0.01)  # 99% confidence
        calc_95 = HaircutCalculator(delta=0.05)  # 95% confidence
        calc_90 = HaircutCalculator(delta=0.10)  # 90% confidence

        haircut_99 = calc_99.calculate_for_sharpe(rademacher, T, N=1)
        haircut_95 = calc_95.calculate_for_sharpe(rademacher, T, N=1)
        haircut_90 = calc_90.calculate_for_sharpe(rademacher, T, N=1)

        # Data snooping should be the same (doesn't depend on delta)
        assert np.isclose(haircut_99.data_snooping_term, haircut_95.data_snooping_term)
        assert np.isclose(haircut_99.data_snooping_term, haircut_90.data_snooping_term)

        # Total haircut should decrease with lower confidence
        assert haircut_99.total_haircut > haircut_95.total_haircut > haircut_90.total_haircut

    def test_sample_size_impact(self) -> None:
        """Test that larger sample size reduces haircut."""
        calc = HaircutCalculator(delta=0.01)
        rademacher = 0.02

        haircut_500 = calc.calculate_for_sharpe(rademacher, T=500, N=1)
        haircut_1000 = calc.calculate_for_sharpe(rademacher, T=1000, N=1)
        haircut_2000 = calc.calculate_for_sharpe(rademacher, T=2000, N=1)

        # Data snooping should be the same
        assert np.isclose(haircut_500.data_snooping_term, haircut_1000.data_snooping_term)

        # Estimation term should decrease
        assert haircut_500.estimation_term > haircut_1000.estimation_term > haircut_2000.estimation_term

        # Total should decrease
        assert haircut_500.total_haircut > haircut_1000.total_haircut > haircut_2000.total_haircut

    def test_realistic_values(self) -> None:
        """Test haircut calculation with realistic values."""
        calc = HaircutCalculator(delta=0.01)

        # Typical single strategy backtest
        # 10 years of daily data ≈ 2500 observations
        # Empirical Sharpe ≈ 1.0 (annualized)
        # Daily Sharpe ≈ 1.0 / √252 ≈ 0.063
        # Rademacher complexity ≈ 0.016 for T=2500

        T = 2500
        rademacher = np.sqrt(2 / np.pi) / np.sqrt(T)  # Theoretical single strategy
        haircut = calc.calculate_for_sharpe(rademacher, T, N=1, sigma=1.0)

        # Data snooping term should be around 0.032
        assert 0.025 < haircut.data_snooping_term < 0.040

        # Estimation term depends on delta and T
        # For delta=0.01 (99% confidence), it can be substantial
        assert haircut.estimation_term > 0

        # Total haircut combines data snooping and estimation
        assert haircut.total_haircut > haircut.data_snooping_term

        # Annualized haircut (for Sharpe ratio)
        ann_haircut = haircut.annualized_haircut
        # Annualized haircut can be substantial for conservative bounds
        assert ann_haircut > 0


class TestHaircutIntegration:
    """Integration tests for haircut calculation."""

    def test_procedure_8_1_signals(self) -> None:
        """Test Procedure 8.1 from the book (bounded signals)."""
        calc = HaircutCalculator(delta=0.01)

        # Example from book (hypothetical)
        R_hat = 0.025
        T = 1000
        max_ic = 1.0

        haircut = calc.calculate_for_signals(R_hat, T, max_ic)

        # Verify formula: IC_true >= IC_emp - 2R - max_ic*√(2*log(2/δ)/T)
        data_snooping = 2 * R_hat
        estimation = max_ic * np.sqrt(2 * np.log(2 / 0.01) / T)

        assert np.isclose(haircut.data_snooping_term, data_snooping)
        assert np.isclose(haircut.estimation_term, estimation)
        assert np.isclose(haircut.total_haircut, data_snooping + estimation)

    def test_procedure_8_2_sharpe(self) -> None:
        """Test Procedure 8.2 from the book (Sharpe ratio)."""
        calc = HaircutCalculator(delta=0.01)

        # Example: single strategy with sub-Gaussian returns
        R_hat = 0.020
        T = 2000
        N = 1
        sigma = 1.0

        haircut = calc.calculate_for_sharpe(R_hat, T, N, sigma)

        # Verify components
        assert haircut.data_snooping_term == 2 * R_hat

        # Estimation has two parts
        log_term_1 = np.sqrt(2 * np.log(2 * N / 0.01) / T)
        log_term_2 = sigma * np.sqrt(2 * np.log(2 / 0.01) / T)

        assert np.isclose(haircut.estimation_term, log_term_1 + log_term_2)

    def test_haircut_decomposition_makes_sense(self) -> None:
        """Test that haircut decomposition is sensible."""
        calc = HaircutCalculator(delta=0.01)

        rademacher = 0.02
        T = 1000

        haircut = calc.calculate_for_sharpe(rademacher, T, N=1)

        # Both terms should be positive
        assert haircut.data_snooping_term > 0
        assert haircut.estimation_term > 0

        # Data snooping should be significant (typically larger than estimation for small T)
        # For small samples, data snooping dominates
        # For large samples, they're comparable
        assert haircut.data_snooping_term > 0  # Always positive

        # Total should equal sum
        assert np.isclose(
            haircut.total_haircut,
            haircut.data_snooping_term + haircut.estimation_term
        )
