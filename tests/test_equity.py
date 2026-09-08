"""SPEC.md 15.4 and 15.5's descriptors on hand-built panels.

Every expected value below was worked by hand from the inputs before the code
ran (CLAUDE.md, definition of done). The two properties the module exists to
deliver -- a cap-weighted portfolio with zero exposure to every style factor,
and NLSIZE / RESVOL orthogonal to what they were regressed on -- are asserted
on a synthetic panel to floating precision, not approximately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mafrm import config as config_mod
from mafrm.factors import equity

# ---------------------------------------------------------------------------
# Rolling stage
# ---------------------------------------------------------------------------


def _panel(values: list[list[float]], start: str = "2020-01-01") -> pd.DataFrame:
    arr = np.asarray(values, dtype=float)
    return pd.DataFrame(
        arr,
        index=pd.bdate_range(start, periods=arr.shape[0], name="date"),
        columns=pd.Index([f"T{i}" for i in range(arr.shape[1])], name="ticker"),
    )


def test_windowed_sums_by_hand_oldest_first_weights_and_the_valid_count() -> None:
    """Column [1, 2, 3, 4], weights [0.25, 0.75]: 0.25*1 + 0.75*2 = 1.75, 2.75, 3.75."""
    values = np.array([[1.0], [2.0], [3.0], [4.0]])
    sums, counts = equity.windowed_sums(values, np.array([0.25, 0.75]))
    assert np.isnan(sums[0, 0])
    assert sums[1:, 0].tolist() == [1.75, 2.75, 3.75]
    assert counts[:, 0].tolist() == [0.0, 2.0, 2.0, 2.0]
    # A hole: the missing observation contributes zero and the count says so.
    holed = np.array([[1.0], [2.0], [np.nan], [4.0]])
    sums, counts = equity.windowed_sums(holed, np.array([0.25, 0.75]))
    assert sums[2, 0] == 0.25 * 2.0 and counts[2, 0] == 1.0
    assert sums[3, 0] == 0.75 * 4.0 and counts[3, 0] == 1.0


def test_beta_and_hsigma_by_hand_equal_weights() -> None:
    """x = [1, 2, 3], y = 2x + [0.1, -0.2, 0.1] -> beta 2, residual sd sqrt(0.02) = 0.141421."""
    market = pd.Series([1.0, 2.0, 3.0], index=pd.bdate_range("2020-01-01", periods=3))
    excess = _panel([[2.1], [3.8], [6.1]])
    beta, hsigma = equity.rolling_beta_hsigma(excess, market, window=3, halflife=10**12)
    assert np.isnan(beta.iloc[0, 0]) and np.isnan(beta.iloc[1, 0])
    assert beta.iloc[2, 0] == pytest.approx(2.0, abs=1e-9)
    assert hsigma.iloc[2, 0] == pytest.approx(np.sqrt(0.02), abs=1e-9)


def test_beta_by_hand_with_a_one_day_halflife() -> None:
    """Window 2, half-life 1: weights [1/3, 2/3]. x = [1, 3], y = [1, 5].

    x_bar = 7/3, y_bar = 11/3; cov = (1/3)(-4/3)(-8/3) + (2/3)(2/3)(4/3) = 16/9;
    var_x = 8/9; beta = 2. Two points fit exactly, so HSIGMA = 0.
    """
    market = pd.Series([1.0, 3.0], index=pd.bdate_range("2020-01-01", periods=2))
    excess = _panel([[1.0], [5.0]])
    beta, hsigma = equity.rolling_beta_hsigma(excess, market, window=2, halflife=1)
    assert beta.iloc[1, 0] == pytest.approx(2.0, abs=1e-12)
    assert hsigma.iloc[1, 0] == pytest.approx(0.0, abs=1e-6)  # sqrt of a rounding residual


def test_beta_needs_a_complete_window_of_name_and_market() -> None:
    market = pd.Series([1.0, np.nan, 3.0, 2.0], index=pd.bdate_range("2020-01-01", periods=4))
    excess = _panel([[1.0], [2.0], [3.0], [np.nan]])
    beta, _ = equity.rolling_beta_hsigma(excess, market, window=2, halflife=1)
    assert beta.iloc[1:, 0].isna().all()  # market hole, then the name's hole


def test_dastd_by_hand() -> None:
    """Weights [1/3, 2/3] on y = [1, 4]: mean 3, E[y^2] = 11, var 2, sd sqrt(2)."""
    out = equity.rolling_dastd(_panel([[1.0], [4.0]]), window=2, halflife=1)
    assert out.iloc[1, 0] == pytest.approx(np.sqrt(2.0), abs=1e-12)
    holed = equity.rolling_dastd(_panel([[1.0], [np.nan], [4.0]]), window=2, halflife=1)
    assert holed.iloc[:, 0].isna().all()


def test_rstr_is_lagged_by_exactly_lag_sessions() -> None:
    """Window 2, half-life 1 (weights [1/3, 2/3]), lag 1 on [a, b, c, d]."""
    a, b, c, d = 0.01, 0.02, -0.03, 0.04
    out = equity.rstr(_panel([[a], [b], [c], [d]]), window=2, halflife=1, lag=1)
    assert out.iloc[:2, 0].isna().all()
    assert out.iloc[2, 0] == pytest.approx(a / 3 + 2 * b / 3)
    assert out.iloc[3, 0] == pytest.approx(b / 3 + 2 * c / 3)
    unlagged = equity.rstr(_panel([[a], [b], [c], [d]]), window=2, halflife=1, lag=0)
    assert unlagged.iloc[3, 0] == pytest.approx(c / 3 + 2 * d / 3)


def test_cmra_by_hand_two_months_of_two_sessions() -> None:
    """[0.1, 0.2, -0.3, 0.05]: Z(1) = -0.25, Z(2) = 0.05 -> range 0.30 on the last session."""
    out = equity.cmra(_panel([[0.1], [0.2], [-0.3], [0.05]]), months=2, days_per_month=2)
    assert out.iloc[:3, 0].isna().all()
    assert out.iloc[3, 0] == pytest.approx(0.30)
    holed = equity.cmra(_panel([[0.1], [np.nan], [-0.3], [0.05]]), months=2, days_per_month=2)
    assert holed.iloc[:, 0].isna().all()


def test_turnover_by_hand() -> None:
    """V = [10, 20, 30, 40], S = [100, 100, 200, 200] -> q = [0.1, 0.2, 0.15, 0.2].

    STOM (1 month of 2 sessions) = ln(0.3) then ln(0.35); STOQ (2 months) on
    the last session = ln((0.1 + 0.2 + 0.15 + 0.2) / 2) = ln(0.325).
    """
    volume = _panel([[10.0], [20.0], [30.0], [40.0]])
    shares = _panel([[100.0], [100.0], [200.0], [200.0]])
    out = equity.turnover(volume, shares, months={"stom": 1, "stoq": 2}, days_per_month=2)
    assert out["stom"].iloc[1, 0] == pytest.approx(np.log(0.3))
    assert out["stom"].iloc[3, 0] == pytest.approx(np.log(0.35))
    assert out["stoq"].iloc[3, 0] == pytest.approx(np.log(0.325))
    assert out["stoq"].iloc[:3, 0].isna().all()
    zero = equity.turnover(
        _panel([[0.0], [0.0]]), _panel([[1.0], [1.0]]), months={"stom": 1}, days_per_month=2
    )
    assert zero["stom"].iloc[:, 0].isna().all()  # invalid bars, not ln(0)


def test_total_returns_by_hand_and_lncap() -> None:
    close = _panel([[10.0], [11.0], [np.nan], [12.0]])
    dividends = _panel([[0.0], [0.5], [0.0], [0.0]])
    r = equity.total_returns(close, dividends)
    assert np.isnan(r.iloc[0, 0])
    assert r.iloc[1, 0] == pytest.approx(0.15)  # (11 + 0.5) / 10 - 1
    assert np.isnan(r.iloc[2, 0]) and np.isnan(r.iloc[3, 0])
    size = equity.lncap(_panel([[np.e], [0.0]]))
    assert size.iloc[0, 0] == pytest.approx(1.0) and np.isnan(size.iloc[1, 0])


def test_cap_weighted_market_uses_the_previous_sessions_cap() -> None:
    """Caps [100, 300] on day 0, returns [0.01, 0.03] on day 1 -> 0.25*0.01 + 0.75*0.03 = 0.025."""
    excess = _panel([[0.05, 0.05], [0.01, 0.03]])
    cap = _panel([[100.0, 300.0], [999.0, 999.0]])
    universe = pd.DataFrame(True, index=excess.index, columns=excess.columns)
    market, names = equity.cap_weighted_market_excess(excess, cap, universe)
    assert np.isnan(market.iloc[0]) and names.iloc[0] == 0
    assert market.iloc[1] == pytest.approx(0.025) and names.iloc[1] == 2
    universe.iloc[1, 1] = False
    market, names = equity.cap_weighted_market_excess(excess, cap, universe)
    assert market.iloc[1] == pytest.approx(0.01) and names.iloc[1] == 1


# ---------------------------------------------------------------------------
# Cross-sectional stage
# ---------------------------------------------------------------------------


def test_winsorize_by_hand_one_pass_equal_weights() -> None:
    """[0, 1, 2, 3, 100]: mean 21.2, sample sd 44.0647; at 1 sd the 100 clips to 65.2647."""
    raw = _panel([[0.0, 1.0, 2.0, 3.0, 100.0]])
    out = equity.winsorize(raw, bound=1.0)
    assert out.iloc[0].tolist()[:4] == [0.0, 1.0, 2.0, 3.0]
    assert out.iloc[0, 4] == pytest.approx(21.2 + np.sqrt(7766.8 / 4), abs=1e-9)
    assert out.iloc[0, 4] == pytest.approx(65.2647, abs=1e-3)
    # one pass: the clipped row is not re-clipped against its own new sd
    again = equity.winsorize(out, bound=1.0)
    assert not again.equals(out)


def test_standardize_cap_weighted_mean_equal_weighted_sd_by_hand() -> None:
    """x = [1, 2, 3], cap = [1, 1, 2]: mu = 2.25 (cap-weighted), sigma = 1 (equal-weighted)."""
    out = equity.standardize(_panel([[1.0, 2.0, 3.0]]), _panel([[1.0, 1.0, 2.0]]))
    assert out.iloc[0].tolist() == pytest.approx([-1.25, -0.25, 0.75])
    # the cap-weighted portfolio has zero exposure; the equal-weighted one does not
    assert np.dot([0.25, 0.25, 0.5], out.iloc[0]) == pytest.approx(0.0, abs=1e-15)
    assert out.iloc[0].mean() == pytest.approx(-0.25)
    # a cell without a cap is not standardized, and does not enter the mean
    partial = equity.standardize(_panel([[1.0, 2.0, 3.0]]), _panel([[1.0, np.nan, 2.0]]))
    assert np.isnan(partial.iloc[0, 1])
    # mu = (1 + 6) / 3 = 7/3, sigma over [1, 3] = sqrt(2)
    assert partial.iloc[0, 0] == pytest.approx((1 - 7 / 3) / np.sqrt(2))


def test_orthogonalize_by_hand_equal_weights() -> None:
    """y = [1, 2, 4] on x = [1, 2, 3]: slope 1.5, intercept -2/3, residual [1/6, -1/3, 1/6]."""
    y, x, w = _panel([[1.0, 2.0, 4.0]]), _panel([[1.0, 2.0, 3.0]]), _panel([[1.0, 1.0, 1.0]])
    resid = equity.orthogonalize(y, [x], w)
    assert resid.iloc[0].tolist() == pytest.approx([1 / 6, -1 / 3, 1 / 6])


def test_orthogonalize_uses_the_weights_it_is_given() -> None:
    """Three collinear points y = 2x - 1 under weights [1, 2, 3] fit exactly: residual zero
    (two points would leave no degree of freedom and the row is refused as missing)."""
    exact = equity.orthogonalize(
        _panel([[1.0, 3.0, 5.0]]), [_panel([[1.0, 2.0, 3.0]])], _panel([[1.0, 2.0, 3.0]])
    )
    assert exact.iloc[0].to_numpy() == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)
    two = equity.orthogonalize(_panel([[1.0, 5.0]]), [_panel([[1.0, 3.0]])], _panel([[1.0, 2.0]]))
    assert two.iloc[0].isna().all()
    rng = np.random.default_rng(config_mod.load().seed)
    y = _panel([rng.normal(size=12).tolist()])
    x1 = _panel([rng.normal(size=12).tolist()])
    x2 = _panel([rng.normal(size=12).tolist()])
    cap = _panel([np.exp(rng.normal(size=12)).tolist()])
    resid = equity.orthogonalize(y, [x1, x2], cap).iloc[0].to_numpy()
    w = cap.iloc[0].to_numpy() / cap.iloc[0].sum()
    assert np.dot(w, resid) == pytest.approx(0.0, abs=1e-12)
    assert np.dot(w, resid * x1.iloc[0].to_numpy()) == pytest.approx(0.0, abs=1e-12)
    assert np.dot(w, resid * x2.iloc[0].to_numpy()) == pytest.approx(0.0, abs=1e-12)
    # equal-weighted, it is NOT orthogonal -- that is the difference the report shows
    assert abs(np.mean(resid * x1.iloc[0].to_numpy())) > 1e-6


# ---------------------------------------------------------------------------
# The whole cross-sectional stage on a synthetic panel
# ---------------------------------------------------------------------------


def _synthetic_raw(
    n_dates: int = 6, n_names: int = 40
) -> tuple[equity.RawDescriptors, pd.DataFrame]:
    rng = np.random.default_rng(config_mod.load().seed)
    idx = pd.bdate_range("2020-01-01", periods=n_dates, name="date")
    cols = pd.Index([f"N{i:02d}" for i in range(n_names)], name="ticker")

    def frame(scale: float, loc: float = 0.0) -> pd.DataFrame:
        return pd.DataFrame(
            rng.normal(loc, scale, size=(n_dates, n_names)), index=idx, columns=cols
        )

    cap = pd.DataFrame(
        np.exp(rng.normal(3.0, 1.2, size=(n_dates, n_names))), index=idx, columns=cols
    )
    frames = {
        "beta": frame(0.3, 1.0),
        "hsigma": frame(0.005, 0.02),
        "dastd": frame(0.005, 0.02),
        "rstr": frame(0.2),
        "cmra": frame(0.1, 0.3),
        "stom": frame(0.5, -2.0),
        "stoq": frame(0.5, -2.0),
        "stoa": frame(0.5, -2.0),
        "lncap": np.log(cap),
    }
    frames["lncap"].iloc[2, 3] = np.nan  # one missing SIZE
    frames["rstr"].iloc[0, 0] = np.nan  # one missing MOMENTUM
    market = pd.Series(rng.normal(0, 0.01, size=n_dates), index=idx)
    raw = equity.RawDescriptors(
        frames=frames, market_excess=market, market_names=pd.Series(n_names, index=idx)
    )
    return raw, cap


def _exposures(bound: float = 3.0) -> equity.Exposures:
    raw, cap = _synthetic_raw()
    universe = pd.DataFrame(True, index=cap.index, columns=cap.columns)
    universe.iloc[1, 5] = False  # one name out of the universe on one date
    return equity.exposures_from_raw(
        raw,
        cap=cap,
        universe=universe,
        descriptors=config_mod.load().model.equity_descriptors,
        winsor_bound=bound,
    )


def test_a_cap_weighted_portfolio_has_zero_exposure_to_every_style_factor() -> None:
    """SPEC.md 15.5: the property the cap-weighted centring exists to produce."""
    exp = _exposures()
    zero = exp.cap_weighted_exposure()
    assert list(zero.columns) == list(equity.FACTORS)
    assert zero.notna().all().all()
    assert np.abs(zero.to_numpy()).max() < 1e-12
    # ...and the equal-weighted portfolio does NOT, which is what makes the test bite
    ew = pd.DataFrame({f: exp.exposures[f].mean(axis=1) for f in equity.FACTORS})
    assert np.abs(ew.to_numpy()).max() > 1e-3


def test_every_exposure_is_scaled_by_the_equal_weighted_sd() -> None:
    exp = _exposures()
    for factor in equity.FACTORS:
        sd = exp.exposures[factor].std(axis=1, ddof=1)
        assert sd.to_numpy() == pytest.approx(np.ones(len(sd)), abs=1e-12), factor


def test_nlsize_is_orthogonal_to_size_and_resvol_to_beta_and_size_regression_weighted() -> None:
    """The two orthogonalizations SPEC.md 15.4 calls not optional, in the REGRESSION's metric.

    W7-P2b ruling 2: the weights are sqrt(cap), the cross-sectional
    regression's. RESVOL is orthogonalized and re-standardized, so its
    sqrt-cap-weighted moment with BETA and SIZE is zero to floating precision
    (re-standardization is an affine map, which preserves it). NLSIZE is
    winsorized AFTER its orthogonalization (SPEC.md 15.5's one exception), so
    exactness holds at the residual stage and the shipped exposure is
    orthogonal up to the clipping -- asserted small, and far smaller than the
    un-orthogonalized cube's.
    """
    exp = _exposures()
    cap = np.sqrt(exp.cap.to_numpy())  # the regression weights

    def cw_moment(y: np.ndarray, x: np.ndarray, i: int) -> float:
        """The weighted COVARIANCE, which an affine re-standardization preserves at zero."""
        valid = np.isfinite(y[i]) & np.isfinite(x[i]) & np.isfinite(cap[i])
        w = cap[i, valid] / cap[i, valid].sum()
        yc = y[i, valid] - np.dot(w, y[i, valid])
        xc = x[i, valid] - np.dot(w, x[i, valid])
        return float(np.dot(w, yc * xc))

    resvol = exp.exposures["RESVOL"].to_numpy()
    for x_name in ("BETA", "SIZE"):
        x = exp.exposures[x_name].to_numpy()
        for i in range(resvol.shape[0]):
            assert cw_moment(resvol, x, i) == pytest.approx(0.0, abs=1e-12), (x_name, i)

    size = exp.exposures["SIZE"]
    residual = equity.orthogonalize(
        exp.pre_orthogonal["NLSIZE"], [size], exp.cap.pow(0.5)
    ).to_numpy()
    nlsize = exp.exposures["NLSIZE"].to_numpy()
    skipped = equity.standardize(exp.pre_orthogonal["NLSIZE"], exp.cap).to_numpy()
    for i in range(nlsize.shape[0]):
        assert cw_moment(residual, size.to_numpy(), i) == pytest.approx(0.0, abs=1e-12), i
        assert abs(cw_moment(nlsize, size.to_numpy(), i)) < 0.05, i
        # the "skipped orthogonalization" control: the standardized cube is NOT orthogonal,
        # and by a clear margin over the clipping trace the shipped NLSIZE carries
        assert abs(cw_moment(skipped, size.to_numpy(), i)) > 3 * abs(
            cw_moment(nlsize, size.to_numpy(), i)
        ), i
    # Before orthogonalization the cube of SIZE is strongly correlated with SIZE.
    assert abs(exp.pre_orthogonal["NLSIZE"].iloc[0].corr(size.iloc[0])) > 0.5


def test_composites_are_weighted_sums_of_standardized_descriptors_then_restandardized() -> None:
    exp = _exposures()
    lq = config_mod.load().model.equity_descriptors.liquidity_composite
    s = exp.standardized
    combo = lq.stom * s["stom"] + lq.stoq * s["stoq"] + lq.stoa * s["stoa"]
    expected = equity.standardize(combo, exp.cap)
    pd.testing.assert_frame_equal(exp.exposures["LIQUIDITY"], expected)
    rv = config_mod.load().model.equity_descriptors.resvol_composite
    combo_rv = rv.dastd * s["dastd"] + rv.cmra * s["cmra"] + rv.hsigma * s["hsigma"]
    pd.testing.assert_frame_equal(
        exp.pre_orthogonal["RESVOL"], equity.standardize(combo_rv, exp.cap)
    )


def test_missing_cells_and_out_of_universe_names_stay_missing_and_are_counted() -> None:
    exp = _exposures()
    assert np.isnan(exp.exposures["SIZE"].iloc[2, 3]) and np.isnan(
        exp.exposures["NLSIZE"].iloc[2, 3]
    )
    assert np.isnan(exp.exposures["RESVOL"].iloc[2, 3])  # SIZE is one of its regressors
    assert np.isnan(exp.exposures["MOMENTUM"].iloc[0, 0])
    assert exp.exposures["BETA"].iloc[1, 5:6].isna().all()  # out of the universe on date 1
    assert exp.counts.loc[exp.counts.index[1], "universe"] == 39
    assert exp.counts.loc[exp.counts.index[2], "SIZE"] == 39
    assert exp.counts.loc[exp.counts.index[0], "MOMENTUM"] == 39
    assert exp.counts["BETA"].iloc[0] == 40


def test_the_winsorization_bound_is_read_not_assumed() -> None:
    tight = _exposures(bound=0.5)
    loose = _exposures(bound=3.0)
    assert tight.winsor_bound == 0.5 and loose.winsor_bound == 3.0
    assert not tight.exposures["BETA"].equals(loose.exposures["BETA"])


# ---------------------------------------------------------------------------
# The rolling stage end to end, under the real config's windows
# ---------------------------------------------------------------------------


def test_raw_descriptors_first_valid_sessions_follow_the_configured_windows() -> None:
    cfg = config_mod.load()
    desc = cfg.model.equity_descriptors
    rng = np.random.default_rng(cfg.seed)
    n, k = 600, 8
    idx = pd.bdate_range("2018-01-01", periods=n, name="date")
    cols = pd.Index([f"N{i}" for i in range(k)], name="ticker")
    excess = pd.DataFrame(rng.normal(0, 0.01, size=(n, k)), index=idx, columns=cols)
    rf = pd.Series(0.0001, index=idx)
    volume = pd.DataFrame(1e6, index=idx, columns=cols)
    shares = pd.DataFrame(1e8, index=idx, columns=cols)
    cap = shares * 50.0
    universe = pd.DataFrame(True, index=idx, columns=cols)
    raw = equity.raw_descriptors(
        excess=excess,
        risk_free=rf,
        volume=volume,
        shares=shares,
        cap=cap,
        universe=universe,
        descriptors=desc,
        days_per_month=cfg.model.data.trading_days_per_month,
    )
    assert set(raw.frames) == set(equity.DESCRIPTORS)
    # the market needs a previous cap, so it starts on row 1; BETA's 252 window is
    # complete on row 252, RSTR's on row lag + window - 1 = 524, CMRA's on 251.
    assert np.isnan(raw.market_excess.iloc[0]) and np.isfinite(raw.market_excess.iloc[1])
    assert raw.frames["beta"]["N0"].first_valid_index() == idx[desc.beta.window]
    assert raw.frames["hsigma"]["N0"].first_valid_index() == idx[desc.hsigma.window]
    assert raw.frames["dastd"]["N0"].first_valid_index() == idx[desc.dastd.window - 1]
    assert (
        raw.frames["rstr"]["N0"].first_valid_index()
        == idx[desc.momentum.lag + desc.momentum.window - 1]
    )
    assert (
        raw.frames["cmra"]["N0"].first_valid_index()
        == idx[desc.cmra_months * cfg.model.data.trading_days_per_month - 1]
    )
    assert (
        raw.frames["stoa"]["N0"].first_valid_index()
        == idx[desc.liquidity_horizons_months.stoa * cfg.model.data.trading_days_per_month - 1]
    )
    # constant turnover: STOM = ln(21 x 0.01) on every complete window
    assert raw.frames["stom"]["N0"].dropna().to_numpy() == pytest.approx(
        np.log(cfg.model.data.trading_days_per_month * 1e6 / 1e8)
    )
    assert raw.frames["lncap"].iloc[0, 0] == pytest.approx(np.log(5e9))
