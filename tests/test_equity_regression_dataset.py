"""SPEC.md 15.6's regression on the REAL cache: data-contract tests (``dataset``-marked).

Deselected by ``make test`` and the fast CI workflow; run weekly after ``make
data``. Each test skips, naming what is missing, when the SIC tables are not
cached. The two constraint tests SPEC.md 15.6.1 asks for on every date, the
holdout boundary, and the undated tables' shape.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mafrm.config import load
from mafrm.data import cache, edgar, french, sic
from mafrm.factors import equity_regression


@pytest.fixture(scope="module")
def factors() -> equity_regression.FactorReturns:
    manifest = cache.Manifest.load()
    for source, name in (
        (french.SOURCE, sic.SICCODES_NAME),
        (edgar.SOURCE, sic.CURRENT_NAME),
        (edgar.SOURCE, sic.FIRST_10K_NAME),
    ):
        try:
            manifest.latest(source=source, name=name)
        except cache.CacheError as exc:
            pytest.skip(f"no cached {source}/{name}: {exc}")
    built, _exposures, _inputs = equity_regression.build(load())
    return built


@pytest.mark.dataset
def test_the_real_siccodes_file_is_the_published_scheme() -> None:
    cfg = load().model.data.ken_french_siccodes49
    table = sic.load_siccodes()
    assert sorted(table["industry"].unique()) == list(range(1, cfg.industries + 1))
    assert sic.industry_of(2911, table, unassigned=cfg.unassigned_industry) == 30  # Oil
    assert sic.industry_of(6798, table, unassigned=cfg.unassigned_industry) == 48  # Fin (REIT)


@pytest.mark.dataset
def test_nothing_reaches_the_holdout(factors: equity_regression.FactorReturns) -> None:
    boundary = pd.Timestamp(load().require_holdout_start())
    assert factors.returns.index.max() < boundary
    assert factors.residuals.index.max() < boundary


@pytest.mark.dataset
def test_the_constraint_binds_and_the_identity_closes_on_every_date(
    factors: equity_regression.FactorReturns,
) -> None:
    """SPEC.md 15.6.1's two legs. The regression raises on a failing date, so the
    summary carrying a row for every date is itself the test; this re-derives both
    from the stored terms rather than trusting the raise."""
    s = factors.summary[factors.summary["names"].notna()]
    assert len(s) > 4000
    assert float(s["industry_term"].abs().max()) < factors.tolerance
    closing = s["cap_weighted_return"] - (
        s["country"] + s["industry_term"] + s["style_term"] + s["cap_weighted_specific"]
    )
    assert float(closing.abs().max()) < factors.tolerance
    # And from the factor returns themselves: the cap-weighted sum of the industry
    # returns is zero when weighted by the industries' cap shares, which the
    # summary's industry_term records; here the weaker, weight-free check that an
    # absent industry is NaN, never zero.
    members = factors.members.loc[s.index].to_numpy()
    industry_returns = factors.returns.loc[s.index, list(factors.industry_columns)].to_numpy()
    assert np.isnan(industry_returns[members == 0]).all()
    assert np.isfinite(industry_returns[members > 0]).all()


@pytest.mark.dataset
def test_the_country_factor_tracks_the_cap_weighted_market(
    factors: equity_regression.FactorReturns,
) -> None:
    reg = load().model.equity_regression.registrations
    s = factors.summary[factors.summary["names"].notna()]
    corr = float(s["country"].corr(s["cap_weighted_return"]))
    assert corr >= reg.row_286_country_market_correlation_min
    assert np.isfinite(factors.returns["COUNTRY"]).sum() == len(s)


@pytest.mark.dataset
def test_a_missing_sic_is_excluded_not_other(factors: equity_regression.FactorReturns) -> None:
    unassigned = load().model.data.ken_french_siccodes49.unassigned_industry
    ind = factors.industries
    no_sic = ind[ind["status"] == sic.NO_SIC]
    assert ind.loc[no_sic.index, "industry"].isna().all()
    other = ind[ind["industry"] == unassigned]
    assert (other["status"] == "ok").all()


@pytest.mark.dataset
def test_the_history_table_is_dated_and_only_drifted_names_move(
    factors: equity_regression.FactorReturns,
) -> None:
    """SPEC.md 15.6.3: the per-filing headers stop before the boundary, every drifted
    name has at least one usable filing, and no undrifted name's industry ever moves."""
    boundary = pd.Timestamp(load().require_holdout_start())
    history = sic.load_history()
    assert history.index.max() < boundary
    drifted = set(sic.drifted_tickers(factors.drift))
    usable = history[history["status"] == "ok"]
    assert drifted <= set(usable["ticker"].unique())
    current = factors.industries["industry"]
    for ticker in factors.industries_pit.columns:
        if ticker in drifted or pd.isna(current.get(ticker, pd.NA)):
            continue
        column = factors.industries_pit[ticker].dropna()
        assert (column == current[ticker]).all(), ticker
    assert set(factors.reclassified.index) <= drifted
