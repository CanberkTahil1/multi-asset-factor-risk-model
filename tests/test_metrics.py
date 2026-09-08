"""SPEC.md 1's identity and SPEC.md 6.6's Sharpe statistics, every value by hand. W6-P2."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from mafrm.backtest import metrics

EXACT = dict(rel=1e-9, abs=1e-12)


# ---------------------------------------------------------------------------
# SPEC.md 1 -- the identity
# ---------------------------------------------------------------------------


def test_identity_terms_by_hand() -> None:
    # Four periods, one year: gross 1%, 3%, -1%, 5%; cost 0.5% each period;
    # forecast vol 2% per period throughout.
    gross = np.array([0.01, 0.03, -0.01, 0.05])
    net = gross - 0.005
    forecast = np.full(4, 0.02)
    out = metrics.identity_terms(gross, net, forecast, periods_per_year=4.0, aggregation=2.0)
    mu_g = 0.02 * 4  # mean 2% x 4
    tc = 0.005 * 4
    sigma_f = 0.02 * 2.0  # rms 2% x (4 / aggregation) = x sqrt(4) at the naive aggregation
    sigma_r = float(np.std(net, ddof=1)) * 2.0
    b = sigma_r / sigma_f
    assert out.gross_return == pytest.approx(mu_g, **EXACT)
    assert out.cost_drag == pytest.approx(tc, **EXACT)
    assert out.forecast_volatility == pytest.approx(sigma_f, **EXACT)
    assert out.bias_ratio == pytest.approx(b, **EXACT)
    assert out.sharpe_paper == pytest.approx(mu_g / sigma_f, **EXACT)
    assert out.sharpe_real == pytest.approx((mu_g - tc) / sigma_r, **EXACT)
    assert out.risk_model_term == pytest.approx(mu_g / sigma_f * (1 - 1 / b), **EXACT)
    assert out.cost_term == pytest.approx(tc / (b * sigma_f), **EXACT)
    assert out.risk_model_term + out.cost_term == pytest.approx(out.gap, **EXACT)


def test_identity_cost_term_is_zero_when_nothing_is_charged() -> None:
    gross = np.array([0.01, 0.02, -0.01, 0.03, 0.0])
    out = metrics.identity_terms(
        gross, gross, np.full(5, 0.015), periods_per_year=12.0, aggregation=3.0
    )
    assert out.cost_drag == 0.0 and out.cost_term == 0.0
    assert out.gap == pytest.approx(out.risk_model_term, **EXACT)


def test_on_los_scale_the_paper_sharpe_is_eta_times_the_per_period_one_and_gross_sharpe_times_b() -> (
    None
):
    rng = np.random.default_rng(20260825)
    gross = 0.004 + 0.02 * rng.standard_normal(60)
    forecast = np.full(60, 0.018)
    est = metrics.annualised_sharpe(gross, q=12)
    out = metrics.identity_terms(gross, gross, forecast, periods_per_year=12.0, aggregation=est.eta)
    # sigma annualised by 12 / eta, mu by 12: SR_paper = eta * mu_pp / sigma_f_pp.
    assert out.sharpe_paper == pytest.approx(est.eta * gross.mean() / 0.018, **EXACT)
    # Cost-free: SR_real is the net (= gross) Sharpe on the same scale, so gross SR x B = SR_paper.
    assert out.sharpe_real == pytest.approx(est.annualised, **EXACT)
    assert est.annualised * out.bias_ratio == pytest.approx(out.sharpe_paper, **EXACT)
    with pytest.raises(metrics.MetricsError, match="aggregation"):
        metrics.identity_terms(gross, gross, forecast, periods_per_year=12.0, aggregation=0.0)


def test_identity_refuses_misaligned_or_nonpositive_inputs() -> None:
    with pytest.raises(metrics.MetricsError, match="align"):
        metrics.identity_terms(
            np.zeros(3), np.zeros(4), np.ones(3), periods_per_year=12.0, aggregation=1.0
        )
    with pytest.raises(metrics.MetricsError, match="positive"):
        metrics.identity_terms(
            np.array([0.01, 0.02]),
            np.array([0.01, 0.02]),
            np.array([0.01, 0.0]),
            periods_per_year=12.0,
            aggregation=1.0,
        )


# ---------------------------------------------------------------------------
# Lo (2002)
# ---------------------------------------------------------------------------


def test_sharpe_and_autocorrelations_by_hand() -> None:
    r = np.array([0.01, 0.03, 0.02])
    assert metrics.sharpe_ratio(r) == pytest.approx(0.02 / 0.01, **EXACT)
    # d = (-0.01, 0.01, 0); rho_1 = (d1 d0 + d2 d1) / sum d^2 = (-1e-4 + 0) / 2e-4 = -0.5
    assert metrics.autocorrelations(r, lags=1).tolist() == pytest.approx([-0.5], **EXACT)


def test_aggregation_factor_is_sqrt_q_when_uncorrelated_and_lo_eq_20_otherwise() -> None:
    r = np.array([0.01, 0.03, 0.02])
    # q = 2: eta = 2 / sqrt(2 + 2 * 1 * rho_1) = 2 / sqrt(2 - 1) = 2
    assert metrics.aggregation_factor(r, q=2) == pytest.approx(2.0, **EXACT)
    # q = 1: no autocorrelation enters; eta = 1
    assert metrics.aggregation_factor(r, q=1) == pytest.approx(1.0, **EXACT)
    rng = np.random.default_rng(20260825)
    white = rng.standard_normal(20_000)
    assert metrics.aggregation_factor(white, q=12) == pytest.approx(math.sqrt(12), rel=0.02)


def test_lo_standard_error_at_zero_lags_by_hand() -> None:
    r = np.array([0.01, 0.03, 0.02])
    mu, t = 0.02, 3
    d = r - mu  # (-0.01, 0.01, 0)
    var = float(np.mean(d**2))  # 2e-4 / 3
    sigma = math.sqrt(var)
    g = np.column_stack([d, d**2 - var])
    s = g.T @ g / t
    grad = np.array([1 / sigma, -mu / (2 * sigma**3)])
    expected = math.sqrt(float(grad @ s @ grad) / t)
    assert metrics.lo_standard_error(r, lags=0) == pytest.approx(expected, **EXACT)


def test_lo_standard_error_recovers_the_iid_closed_form_on_normal_returns() -> None:
    rng = np.random.default_rng(20260825)
    r = 0.002 + 0.01 * rng.standard_normal(50_000)
    sr = metrics.sharpe_ratio(r)
    closed = math.sqrt((1 + sr**2 / 2) / r.size)
    assert metrics.lo_standard_error(r, lags=0) == pytest.approx(closed, rel=0.03)
    # Adding Bartlett lags to white noise changes the estimate only by sampling noise.
    assert metrics.lo_standard_error(r, lags=11) == pytest.approx(closed, rel=0.05)


def test_annualised_sharpe_scales_both_the_ratio_and_its_error_by_eta() -> None:
    rng = np.random.default_rng(7)
    r = 0.001 + 0.02 * rng.standard_normal(300)
    est = metrics.annualised_sharpe(r, q=12)
    assert est.q == 12 and est.lags == 11
    assert est.annualised == pytest.approx(est.per_period * est.eta, **EXACT)
    assert est.standard_error == pytest.approx(est.standard_error_per_period * est.eta, **EXACT)
    assert est.iid_annualised == pytest.approx(est.per_period * math.sqrt(12), **EXACT)


# ---------------------------------------------------------------------------
# Bailey & Lopez de Prado (2014)
# ---------------------------------------------------------------------------


def test_expected_maximum_sharpe_by_hand() -> None:
    g = metrics.EULER_MASCHERONI
    n, v = 10, 0.04
    z1 = stats.norm.ppf(1 - 1 / n)
    z2 = stats.norm.ppf(1 - 1 / (n * math.e))
    expected = 0.2 * ((1 - g) * z1 + g * z2)
    assert metrics.expected_maximum_sharpe(trials_variance=v, trials=n) == pytest.approx(
        expected, **EXACT
    )
    assert metrics.expected_maximum_sharpe(trials_variance=0.0, trials=n) == 0.0


def test_deflated_sharpe_by_hand_against_the_published_formula() -> None:
    rng = np.random.default_rng(20260825)
    r = 0.003 + 0.02 * rng.standard_normal(120)
    out = metrics.deflated_sharpe(r, trials_variance=0.01, trials=38)
    sr = metrics.sharpe_ratio(r)
    sr0 = metrics.expected_maximum_sharpe(trials_variance=0.01, trials=38)
    skew = stats.skew(r, bias=False)
    kurt = stats.kurtosis(r, fisher=False, bias=False)
    z = (sr - sr0) * math.sqrt(119) / math.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2)
    assert out.probability == pytest.approx(stats.norm.cdf(z), **EXACT)
    assert out.trials == 38 and out.periods == 120
    assert out.expected_maximum == pytest.approx(sr0, **EXACT)


def test_deflated_sharpe_falls_as_the_trial_count_rises() -> None:
    rng = np.random.default_rng(3)
    r = 0.003 + 0.02 * rng.standard_normal(120)
    few = metrics.deflated_sharpe(r, trials_variance=0.01, trials=2).probability
    many = metrics.deflated_sharpe(r, trials_variance=0.01, trials=200).probability
    assert many < few


# ---------------------------------------------------------------------------
# The trial count
# ---------------------------------------------------------------------------


def test_read_trial_count_reads_the_consolidated_line_and_refuses_duplicates(
    tmp_path: Path,
) -> None:
    path = tmp_path / "experiments.md"
    path.write_text(
        "| Category | Count | Feeds |\n|---|---|---|\n| **Total evaluated** | **238** | **`N` = 38** |\n",
        encoding="utf-8",
    )
    assert metrics.read_trial_count(path) == 38
    path.write_text(path.read_text() * 2, encoding="utf-8")
    with pytest.raises(metrics.MetricsError, match="exactly one"):
        metrics.read_trial_count(path)
    with pytest.raises(metrics.MetricsError, match="does not exist"):
        metrics.read_trial_count(tmp_path / "missing.md")


def test_the_repositorys_running_total_is_readable() -> None:
    n = metrics.read_trial_count(Path(__file__).resolve().parents[1] / "experiments.md")
    assert n >= 10
