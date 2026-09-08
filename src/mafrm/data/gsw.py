"""Constant-maturity zero-coupon Treasury returns from the Fed's fitted curves.

SPEC.md 3.2. The Fed publishes daily Svensson parameters for the fitted nominal
zero curve back to 1961 (Gurkaynak, Sack & Wright 2006, file ``feds200628``) and
for the TIPS real curve back to 1999 (GSW 2010, ``feds200805``). Six numbers per
day -- ``BETA0..BETA3, TAU1, TAU2`` -- give the continuously-compounded zero
yield at any maturity, which is what makes a genuine constant-maturity total
return series constructible rather than approximated from an ETF.

This module does the mathematics and reads from the cache. It never fetches:
``mafrm.data.loaders`` owns the network (CLAUDE.md invariant 1) and writes the
parsed curve into ``data/raw``; everything here starts from those bytes.

**The roll-down.** A constant-maturity holding is not a buy-and-hold bond. Over
one day the position ages by ``Delta = 1/252`` of a year and is rebalanced back
to maturity ``n``, so the day-``t+1`` curve must be read at ``n - Delta``, not at
``n``::

    P_t(n)       = exp(-n . y_t(n)/100)
    r_{t+1}(n)   = exp( n . y_t(n)/100 - (n-Delta) . y_{t+1}(n-Delta)/100 ) - 1
    excess_{t+1} = r_{t+1}(n) - rf_t . Delta

Reading the second term at ``n`` instead of ``n - Delta`` silently deletes the
roll, which at the 30y point on a 2% slope is worth a few basis points a day --
about 0.5% a year, in the direction that makes the bond sleeve look worse.

**Two traps in the file that SPEC.md does not name**, both found by inspection
and both handled explicitly here:

1. Before 1980-01-02 the Fed fitted Nelson-Siegel, not Svensson: ``BETA3`` is 0
   and ``TAU2`` carries the sentinel ``-999.99``. Evaluating ``exp(-n/-999.99)``
   overflows to ``inf`` and ``0 * inf`` is ``nan``, which would silently destroy
   4,620 rows -- 27% of the sample, and the whole of the Volcker era. Any date
   whose ``TAU2`` is missing or non-positive drops the fourth term.
2. The files carry a row for every weekday including market holidays, on which
   every field is ``NA``. Left in, they accrue 261 days of carry and roll per
   year against a ``Delta`` of 1/252 -- a 3.5% overstatement of both. They are
   dropped at parse time, which leaves ~250.5 observations a year.

**Where the published curve stops.** ``SVENY30`` begins 1985-11-25 and
``SVENY10`` 1971-08-16, because the Fed does not publish a fitted yield beyond
the longest bond actually outstanding. The parameters would happily evaluate at
30y in 1961; that number would be an extrapolation of a curve fitted to bonds of
seven years and less, not an observation. Each maturity is therefore masked to
the dates on which the Fed publishes it. The 1961 start in ``universe.yaml``
holds for the 2y and 5y points only.
"""

from __future__ import annotations

import io
import math
from typing import Final, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from mafrm.config import CurveConfig, load
from mafrm.data import cache

__all__ = [
    "PARAMETER_COLUMNS",
    "CurveKind",
    "GswError",
    "breakeven_inflation",
    "constant_maturity_return",
    "curve_yields",
    "load_curve",
    "load_curve_unrestricted",
    "load_risk_free",
    "parse_fed_curve_csv",
    "parse_fred_csv",
    "published_mask",
    "svensson_yield",
]

#: The six fitted parameters, present in both Fed files under these exact names.
PARAMETER_COLUMNS: Final[tuple[str, ...]] = ("BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2")

#: The header row is the first line beginning with this. Both files carry a
#: multi-row preamble -- 9 lines in feds200628, 18 in feds200805 -- so the row
#: index is found, never assumed. SPEC.md 3.2, point 3.
_HEADER_MARKER: Final[str] = "Date,BETA0,"

#: Strings the Fed uses for a missing observation.
_NA_VALUES: Final[tuple[str, ...]] = ("NA", "ND", "NaN", ".")

CurveKind = Literal["nominal", "real"]
Frequency = Literal["daily", "monthly"]


class GswError(RuntimeError):
    """A curve file, or a request against it, is not what this module expects."""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_fed_curve_csv(text: str, *, yield_prefix: str) -> pd.DataFrame:
    """Parse a Fed fitted-curve CSV into a dated frame of parameters and yields.

    The multi-row header is handled by locating the line that starts with
    ``Date,BETA0,`` rather than by a ``skiprows`` count, because the two files
    disagree on the preamble length and the Fed has changed it before.

    Rows whose parameters are entirely missing are dropped: those are market
    holidays, carried in the file as blank weekdays.
    """
    lines = text.splitlines()
    header_rows = [i for i, line in enumerate(lines) if line.startswith(_HEADER_MARKER)]
    if len(header_rows) != 1:
        raise GswError(
            f"expected exactly one header row starting {_HEADER_MARKER!r}, found "
            f"{len(header_rows)}. The Fed's file layout has changed; do not guess skiprows."
        )

    frame = pd.read_csv(
        io.StringIO("\n".join(lines[header_rows[0] :])),
        na_values=list(_NA_VALUES),
        keep_default_na=True,
        low_memory=False,
    )

    missing = [column for column in ("Date", *PARAMETER_COLUMNS) if column not in frame.columns]
    if missing:
        raise GswError(f"curve file is missing required column(s): {missing}")
    yield_columns = [column for column in frame.columns if column.startswith(yield_prefix)]
    if not yield_columns:
        raise GswError(f"curve file has no published yield columns matching {yield_prefix!r}")

    frame["Date"] = pd.to_datetime(frame["Date"], errors="raise")
    frame = frame.set_index("Date").sort_index()
    if frame.index.has_duplicates:
        raise GswError("curve file contains duplicate dates")

    # Market holidays: every field blank. Dropping them is what makes Delta=1/252
    # honest -- see the module docstring.
    blank = frame[list(PARAMETER_COLUMNS)].isna().all(axis=1)

    # ...and one holiday the parameter test does not catch. On 2008-03-21, Good
    # Friday, the Fed wrote a degenerate fit (BETA0 = 8.3e-14) but published no
    # yield at any tenor. A row the Fed publishes nothing for is a day the curve
    # was not observed, whatever the parameter columns contain: pricing across
    # it produces a spurious one-day return spanning a four-day weekend. One row
    # in 65 years, worth 0.05%/yr on a 20y par bond -- a third of the ladder's
    # acceptance tolerance, decided by a single day.
    unpublished = frame[yield_columns].isna().all(axis=1)
    frame = frame.loc[~(blank | unpublished)]
    if frame["BETA0"].isna().any():
        raise GswError("curve file has rows with a partially missing parameter set")
    if (frame["TAU1"] <= 0).any():
        raise GswError("curve file has a non-positive TAU1, which the Svensson form cannot use")
    frame.index.name = "date"
    return frame


def parse_fred_csv(text: str, series_id: str) -> pd.Series:
    """Parse a fredgraph CSV export into a dated percent-per-annum series."""
    frame = pd.read_csv(io.StringIO(text), na_values=list(_NA_VALUES), keep_default_na=True)
    date_column = frame.columns[0]
    if series_id not in frame.columns:
        raise GswError(
            f"FRED export has no column {series_id!r}; columns are {list(frame.columns)}"
        )
    frame[date_column] = pd.to_datetime(frame[date_column], errors="raise")
    series = frame.set_index(date_column)[series_id].astype(float).sort_index()
    series.index.name = "date"
    series.name = series_id
    return series.dropna()


# ---------------------------------------------------------------------------
# The Svensson form
# ---------------------------------------------------------------------------


def _decay_terms(
    n: npt.NDArray[np.float64], tau: npt.NDArray[np.float64], tiny: npt.NDArray[np.bool_]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """``(1-exp(-n/tau))/(n/tau)`` and ``exp(-n/tau)``, with their ``n -> 0`` limits.

    Both tend to 1 as ``n -> 0``; the ratio is 0/0 there and must not be
    evaluated (SPEC.md 3.2, point 2). A non-positive or missing ``tau`` yields
    ``nan``, which the caller drops -- it marks the Nelson-Siegel regime.
    """
    safe_tau = np.where(np.isfinite(tau) & (tau > 0.0), tau, np.nan)
    # The placeholder 1.0 only keeps the arithmetic finite where `tiny` holds;
    # both results are then replaced by their limits, which are 1 and 1.
    x = np.where(tiny, 1.0, n / safe_tau)
    ratio = np.where(tiny, 1.0, (1.0 - np.exp(-x)) / x)
    decay = np.where(tiny, 1.0, np.exp(-x))
    return ratio, decay


def svensson_yield(
    beta0: npt.ArrayLike,
    beta1: npt.ArrayLike,
    beta2: npt.ArrayLike,
    beta3: npt.ArrayLike,
    tau1: npt.ArrayLike,
    tau2: npt.ArrayLike,
    n: npt.ArrayLike,
    *,
    small_n: float | None = None,
) -> npt.NDArray[np.float64]:
    """Continuously-compounded zero yield in percent at maturity ``n`` years.

    Svensson (1994) as published by the Fed; SPEC.md 3.2::

        y(n) = b0 + b1.[1-exp(-n/t1)]/(n/t1)
                  + b2.{[1-exp(-n/t1)]/(n/t1) - exp(-n/t1)}
                  + b3.{[1-exp(-n/t2)]/(n/t2) - exp(-n/t2)}

    Below ``small_n`` the two ratio terms are replaced by their limit of 1, so
    ``y(0) = b0 + b1`` exactly rather than ``nan``. Where ``tau2`` is missing or
    non-positive -- the pre-1980 Nelson-Siegel regime, flagged in the file by
    ``TAU2 = -999.99`` -- the fourth term is dropped rather than evaluated.

    Broadcasts over all seven arguments, so one call prices the whole history.
    """
    if small_n is None:
        small_n = load().model.data.svensson_small_n

    b0, b1, b2, b3, t1, t2, maturity = (
        np.asarray(value, dtype=float)
        for value in np.broadcast_arrays(beta0, beta1, beta2, beta3, tau1, tau2, n)
    )
    tiny = np.abs(maturity) < small_n

    ratio1, decay1 = _decay_terms(maturity, t1, tiny)
    ratio2, decay2 = _decay_terms(maturity, t2, tiny)

    fourth = b3 * (ratio2 - decay2)
    nelson_siegel = ~np.isfinite(t2) | (t2 <= 0.0)
    fourth = np.where(nelson_siegel, 0.0, fourth)

    result: npt.NDArray[np.float64] = b0 + b1 * ratio1 + b2 * (ratio1 - decay1) + fourth
    return result


def curve_yields(params: pd.DataFrame, n: float, *, small_n: float | None = None) -> pd.Series:
    """Evaluate the fitted curve at one maturity for every date in ``params``."""
    values = svensson_yield(
        params["BETA0"].to_numpy(),
        params["BETA1"].to_numpy(),
        params["BETA2"].to_numpy(),
        params["BETA3"].to_numpy(),
        params["TAU1"].to_numpy(),
        params["TAU2"].to_numpy(),
        n,
        small_n=small_n,
    )
    return pd.Series(values, index=params.index, name=f"y{n:g}")


def published_mask(params: pd.DataFrame, n: float, *, yield_prefix: str) -> pd.Series:
    """Dates on which the Fed publishes a fitted yield covering maturity ``n``.

    The parameters evaluate anywhere; the Fed only publishes out to the longest
    bond outstanding. Beyond that the curve is extrapolation, not observation,
    so each maturity is restricted to the dates its published column exists.
    """
    tenor = math.ceil(n - 1e-12)
    column = f"{yield_prefix}{tenor:02d}"
    if column not in params.columns:
        raise GswError(
            f"maturity {n}y needs published column {column!r} to bound its history, and the "
            f"curve file does not have it. Available: "
            f"{[c for c in params.columns if c.startswith(yield_prefix)][:3]}..."
        )
    mask = params[column].notna()
    mask.name = f"published_{tenor}y"
    return mask


# ---------------------------------------------------------------------------
# Constant-maturity returns
# ---------------------------------------------------------------------------


def constant_maturity_return(
    n: float,
    freq: Frequency = "daily",
    *,
    params: pd.DataFrame | None = None,
    kind: CurveKind = "nominal",
    risk_free_pct: pd.Series | None = None,
    delta: float | None = None,
    small_n: float | None = None,
    yield_prefix: str | None = None,
    mask_unpublished: bool = True,
) -> pd.DataFrame:
    """Total and excess returns for a constant-maturity zero-coupon bond.

    SPEC.md 3.2. Each row is dated ``t+1`` and holds the return earned from the
    close of ``t`` to the close of ``t+1``::

        log(1+r_{t+1}) = [ n.y_t(n) - (n-Delta).y_{t+1}(n-Delta) ] / 100

    decomposed, exactly and by construction, into three additive log components::

        carry           = Delta.y_t(n) / 100
        roll_down       = (n-Delta).[y_t(n) - y_t(n-Delta)] / 100
        duration_effect = (n-Delta).[y_t(n-Delta) - y_{t+1}(n-Delta)] / 100

    ``carry + roll_down + duration_effect == log(1+total_return)`` is asserted in
    the tests; it is the cheapest available check that the ``n - Delta``
    evaluation was not quietly dropped.

    ``freq='monthly'`` compounds the daily series within each calendar month --
    never resampled from a vendor's monthly bar, per CLAUDE.md invariant 2.
    """
    config = load().model.data
    if delta is None:
        delta = config.roll_down_step
    curve: CurveConfig = config.nominal_curve if kind == "nominal" else config.real_curve
    if yield_prefix is None:
        yield_prefix = curve.yield_prefix
    if params is None:
        params = load_curve(kind)
    if n <= delta:
        raise GswError(f"maturity {n} must exceed the roll-down step {delta}")

    y_n = curve_yields(params, n, small_n=small_n)
    y_rolled = curve_yields(params, n - delta, small_n=small_n)

    start_yield = y_n.shift(1)  # y_t(n), dated at t+1
    start_rolled = y_rolled.shift(1)  # y_t(n-Delta), dated at t+1

    carry = delta * start_yield / 100.0
    roll_down = (n - delta) * (start_yield - start_rolled) / 100.0
    duration_effect = (n - delta) * (start_rolled - y_rolled) / 100.0
    log_total = carry + roll_down + duration_effect

    frame = pd.DataFrame(
        {
            "total_return": np.expm1(log_total),
            "carry": carry,
            "roll_down": roll_down,
            "duration_effect": duration_effect,
            "start_yield_pct": start_yield,
            "end_yield_pct": y_n,
        }
    )

    if mask_unpublished:
        published = published_mask(params, n, yield_prefix=yield_prefix)
        frame = frame.where(published & published.shift(1))

    if risk_free_pct is not None:
        rf = risk_free_pct.reindex(params.index).shift(1)
        frame["excess_return"] = frame["total_return"] - rf * delta / 100.0
    else:
        frame["excess_return"] = np.nan

    frame = frame.iloc[1:]
    if freq == "monthly":
        frame = _to_monthly(frame)
    elif freq != "daily":
        raise GswError(f"unknown freq {freq!r}: expected 'daily' or 'monthly'")
    frame.attrs["maturity_years"] = n
    frame.attrs["curve"] = kind
    return frame


def _to_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    """Compound the daily series inside each calendar month.

    Simple returns compound, log components add, yields are taken at the ends of
    the month. CLAUDE.md failure mode 1: this is done explicitly rather than by
    ``.resample().last()``, which would silently take a price where a compounded
    return is meant.
    """
    grouper = pd.Grouper(freq="ME")
    compounded = daily[["total_return", "excess_return"]].add(1.0).groupby(grouper).prod().sub(1.0)
    summed = daily[["carry", "roll_down", "duration_effect"]].groupby(grouper).sum(min_count=1)
    ends = daily[["start_yield_pct", "end_yield_pct"]].groupby(grouper).last()
    monthly = pd.concat([compounded, summed, ends], axis=1)
    # A month in which every day was masked out compounds to 0.0, not to nothing.
    empty = daily["total_return"].groupby(grouper).count() == 0
    monthly.loc[empty, ["total_return", "excess_return"]] = np.nan
    return monthly[list(daily.columns)]


# ---------------------------------------------------------------------------
# Cache access and derived series
# ---------------------------------------------------------------------------


def _cache_name(kind: CurveKind) -> str:
    return "nominal_curve" if kind == "nominal" else "real_curve"


def load_curve(kind: CurveKind = "nominal") -> pd.DataFrame:
    """Read the most recent cached pull of one Fed curve, **truncated at the holdout**.

    Never fetches. ``mafrm.data.loaders`` owns the network (CLAUDE.md invariant
    1); if nothing is cached this raises and tells you to run ``make data``.
    The truncation is :func:`mafrm.data.cache.read`'s -- see
    :mod:`mafrm.data.holdout` for why it is the default.
    """
    manifest = cache.Manifest.load()
    entry = manifest.latest(source="fed", name=_cache_name(kind))
    return cache.read(entry, manifest=manifest)


def load_curve_unrestricted(kind: CurveKind = "nominal", *, reason: str) -> pd.DataFrame:
    """The same curve over its full published history. See :func:`load_curve`.

    For data-layer work that asks what the Fed actually published -- the W1-P3
    TLT cross-check, which validates a vendor price series against our
    construction and reports over the whole cached window.
    """
    manifest = cache.Manifest.load()
    entry = manifest.latest(source="fed", name=_cache_name(kind))
    return cache.read_unrestricted(entry, manifest=manifest, reason=reason)


def load_risk_free() -> pd.Series:
    """Read the selected short rate, in percent per annum. SPEC.md 3.2.

    The selected leg is ``data.risk_free.selected`` -- Ken French's daily
    one-month bill since W1-P4, which spans the whole 1961- GSW sample with no
    splice, so excess returns now begin with the curve rather than with
    ``DGS1MO``'s 2001-07-31 start.

    Ken French publishes that rate in percent per **trading day** while this
    function's contract, inherited from ``DGS1MO``, is percent per **annum**;
    :func:`mafrm.data.french.as_annual_percent` converts. The two units differ by
    a factor of about 252 and both are small plausible numbers, which is why the
    conversion is a named function with a hand-computed test rather than a
    multiplication written inline here.

    Falls back to the interim FRED series when the selected leg is marked
    ``implemented: false``, so the config stays the single switch.
    """
    from mafrm.data import french  # local import: gsw is imported by french's callers

    config = load().model.data.risk_free
    if config.selected == "ken_french_daily_rf" and config.ken_french_daily_rf.implemented:
        return french.load_risk_free_annual_percent()

    interim = config.interim_fred
    manifest = cache.Manifest.load()
    entry = manifest.latest(source="fred", name=interim.fred_series.lower())
    frame = cache.read(entry, manifest=manifest)
    # The FRED loader now writes the API shape (a `value` column plus real-time
    # bounds); the pre-W1-P4 cache entry was a single column named for the series.
    column = "value" if "value" in frame.columns else interim.fred_series
    return frame[column].astype(float)


def breakeven_inflation(
    *,
    nominal: pd.DataFrame | None = None,
    real: pd.DataFrame | None = None,
    n: float | None = None,
) -> pd.Series:
    """Nominal minus real fitted zero yield, in percent. SPEC.md 3.2.

    The cleanest available breakeven-inflation factor: both legs come from the
    same estimator on the same day, so the difference carries no splice.
    """
    config = load().model.data
    if n is None:
        n = config.real_curve.maturities_years[0]
    if nominal is None:
        nominal = load_curve("nominal")
    if real is None:
        real = load_curve("real")

    nominal_y = curve_yields(nominal, n).where(
        published_mask(nominal, n, yield_prefix=config.nominal_curve.yield_prefix)
    )
    real_y = curve_yields(real, n).where(
        published_mask(real, n, yield_prefix=config.real_curve.yield_prefix)
    )
    breakeven = (nominal_y - real_y).dropna()
    breakeven.name = f"breakeven_{n:g}y_pct"
    return breakeven
