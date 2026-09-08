"""External validation of Model A's factor set against published series.

SPEC.md 6.5, last row, as amended by 6.5.1: *"Regress each self-built factor on
the corresponding AQR Century-of-Factor-Premia and TSMOM series. Report R^2,
beta, alpha t-stat, rolling correlation."* This module is the measurement half;
:mod:`mafrm.factors.validation_report` renders it.

**Three of Model A's six factors have no comparand and that is the result.** The
mapping in ``config/model.yaml`` was written factor by factor from a printed
column listing of the three AQR workbooks and the Ken French daily research
table, before any regression was run. ``rates_slope``, ``credit`` and ``dollar``
have no published analogue in the files this project holds; they are recorded
with the reason rather than regressed against the nearest available column,
which is the failure this module exists not to commit.

**The comparison is monthly, and the frequency change is where a session of this
shape goes quietly wrong.** All three AQR workbooks are monthly -- the source
URLs end ``-Monthly.xlsx`` and the manifest records 1,196 / 497 / 1,780 rows
against the factor panel's 4,146 complete daily dates -- so these regressions run
at 200-213 observations, not 4,146. An alpha t-statistic at n = 213 is a
different instrument from one at n = 4,146, and every record this module emits
carries its own ``observations``, its own window and its own units so that a
reader cannot assume otherwise.

Two specifics on the aggregation, because CLAUDE.md failure mode 1 sits directly
in the path:

* **Daily returns compound, they do not average.** :func:`monthly_factor_returns`
  compounds the four decimal-return factors in log space and *sums* the two
  basis-point yield-change factors, which is the correct aggregation for each and
  is asserted as an exact round trip rather than trusted.
* **AQR's month-ends are the last business day** -- 1926-07-30, 1877-02-28,
  1985-01-31 in the three files -- so ``.resample("ME").last()`` would stamp on
  the calendar month end and misalign. The join key is the calendar *month*, the
  index carried through is AQR's own observed date, and the joined count is
  asserted against an expectation computed from the two coverages rather than
  taken on trust.

CLAUDE.md invariant 1: this module makes no network call. It reads the cache
through :mod:`mafrm.data.aqr`, :mod:`mafrm.data.french` and
:mod:`mafrm.data.cache`, exactly as every other non-loader module does.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from mafrm.config import Comparand, Config, ValidationConfig, load
from mafrm.data import aqr, cache, french, prices
from mafrm.factors import betas, macro

__all__ = [
    "AlphaConfound",
    "ComparandFit",
    "ComparandPair",
    "DailyAnchorCheck",
    "MonthlyPanel",
    "PlaceboRow",
    "ValidationError",
    "ValidationResult",
    "alpha_confound",
    "assert_comparands_are_distinct",
    "build",
    "century_volatility_summary",
    "comparand_series",
    "comparand_volatility_pct_per_year",
    "compare_comparands",
    "daily_anchor_check",
    "duplicate_comparand_check",
    "fit_comparand",
    "monthly_factor_returns",
    "newey_west_plug_in_lags",
    "placebo",
    "rolling_correlation",
]

#: Basis points in one unit of decimal return. The same unit conversion
#: :mod:`mafrm.factors.macro` applies, named here rather than imported private.
_BPS_PER_UNIT = 1e4

#: Months in a year. Used only to annualise a monthly intercept for display.
_MONTHS_PER_YEAR = 12

#: The unit every regression in this module runs in. Both sides are put here
#: before fitting, exactly as W2-P2 does at daily frequency, so a coefficient on
#: a rate factor reads in years and one on a return factor is dimensionless.
_REGRESSION_UNIT = "basis points per month"

#: Two series count as the same when they agree to this, after standardization.
#: A floating-point tolerance, not a tunable -- the same status as
#: ``numerics.psd_eigenvalue_floor``. Standardized rather than raw because an
#: affine rescaling of a comparand (a unit change, say) would tautologise the
#: placebo exactly as thoroughly as a literal copy; z-scoring both sides over
#: their common window catches that too. Century's `Commodities Market` against
#: Commodities-for-the-Long-Run's `Excess return of equal-weight commodities
#: portfolio` agrees to 2.3e-12 raw, twelve orders inside this.
_DUPLICATE_TOLERANCE = 1e-9


class ValidationError(ValueError):
    """A comparand cannot be built, or an alignment did not produce what it must."""


def newey_west_plug_in_lags(observations: int) -> int:
    """Newey & West (1994) deterministic plug-in bandwidth.

    ``lag = floor(4 * (T/100)^(2/9))``. A published RULE with ``T`` supplied at
    runtime, not a number: at the 200-213 months these regressions run on it
    gives 4, and at W2-P2's 4,146 daily observations it would give 9. Borrowing
    ``covariance.volatility_newey_west_lags.short`` instead -- a daily
    covariance-pipeline parameter -- would be a magic number wearing a config
    key's name.
    """
    if observations < 1:
        raise ValidationError(f"newey_west_plug_in_lags: T must be positive, got {observations}")
    return int(math.floor(4.0 * (observations / 100.0) ** (2.0 / 9.0)))  # noqa: RUF046


# ---------------------------------------------------------------------------
# Daily -> monthly
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MonthlyPanel:
    """Model A's factors aggregated to months, one column per factor.

    ``frame`` is indexed by ``pd.Period`` at monthly frequency because the join
    key against a published monthly series is the calendar month and nothing
    else. ``observed_month_end`` records, per factor, the trading date each
    monthly observation actually closed on -- which is what makes the
    aggregation auditable rather than merely plausible.
    """

    frame: pd.DataFrame
    units: Mapping[str, str]
    observed_month_end: pd.DataFrame
    #: Boundary months dropped because the factor's history opened or closed
    #: inside them. Never an interior month.
    dropped: Mapping[str, tuple[pd.Period, ...]]

    def series(self, factor: str) -> pd.Series:
        """One factor's monthly series, ``NaN`` months removed."""
        if factor not in self.frame.columns:
            raise ValidationError(f"{factor!r} is not in the monthly panel")
        return self.frame[factor].dropna()

    @property
    def observations(self) -> dict[str, int]:
        return {str(name): int(self.frame[name].notna().sum()) for name in self.frame.columns}


def monthly_factor_returns(panel: macro.FactorPanel) -> MonthlyPanel:
    """Compound the return factors, sum the basis-point factors, per calendar month.

    Each factor is aggregated over **its own** trading dates. That is deliberate:
    the six factors have different burn-ins (``credit`` opens 2008-04-14 against
    ``equity``'s 2007-04-02) and different source calendars, and forcing them
    onto one intersection would throw away 13 months of the two factors that have
    them for no gain -- every comparison here is one factor against one
    comparand, never one factor against another.

    Only a **boundary** month is ever dropped: the first month if the factor's
    history opens after that month's first date on the panel calendar, the last
    if it closes before that month's last. An interior month in which a factor
    did not trade on some panel date is kept, because compounding over the dates
    a factor actually has is the correct monthly return for it -- and dropping
    those would have cost ``equity`` six good months.
    """
    orthogonal = panel.orthogonal
    index = pd.DatetimeIndex(orthogonal.index)
    period = index.to_period("M")
    stamps = pd.Series(index, index=period)
    panel_first = stamps.groupby(level=0).min()
    panel_last = stamps.groupby(level=0).max()

    columns: dict[str, pd.Series] = {}
    closes: dict[str, pd.Series] = {}
    dropped: dict[str, tuple[pd.Period, ...]] = {}
    for name in orthogonal.columns:
        key = str(name)
        series = orthogonal[name].dropna()
        if series.empty:
            raise ValidationError(f"{key}: the orthogonalized factor is empty")
        own = pd.DatetimeIndex(series.index).to_period("M")
        if "basis points" in panel.units[key]:
            # A yield CHANGE is additive: the month's move is the sum of the
            # daily moves, not their compound.
            aggregate = series.groupby(own).sum()
        else:
            # A RETURN compounds. Summed in log space and returned to simple,
            # which is exact rather than a small-return approximation.
            aggregate = np.expm1(np.log1p(series).groupby(own).sum())
        close = pd.Series(pd.DatetimeIndex(series.index), index=own).groupby(level=0).max()
        first = pd.Series(pd.DatetimeIndex(series.index), index=own).groupby(level=0).min()

        cut: list[pd.Period] = []
        lo, hi = aggregate.index.min(), aggregate.index.max()
        if first.loc[lo] > panel_first.loc[lo]:
            cut.append(lo)
        if hi != lo and close.loc[hi] < panel_last.loc[hi]:
            cut.append(hi)
        columns[key] = aggregate.drop(cut)
        closes[key] = close.drop(cut)
        dropped[key] = tuple(cut)

    frame = pd.DataFrame(columns)
    frame.index.name = "month"
    ends = pd.DataFrame(closes)
    ends.index.name = "month"
    return MonthlyPanel(
        frame=frame, units=dict(panel.units), observed_month_end=ends, dropped=dropped
    )


def monthly_round_trip(panel: macro.FactorPanel, monthly: MonthlyPanel) -> dict[str, float]:
    """Residual of the aggregation, per factor, over the months it retained.

    The assertion the aggregation owes: a compounded monthly series must
    reproduce the daily total over exactly the dates it covers. Anything but zero
    here is an aggregation defect, and it is measured rather than commented.
    """
    out: dict[str, float] = {}
    for name in monthly.frame.columns:
        key = str(name)
        kept = set(monthly.series(key).index)
        series = panel.orthogonal[key].dropna()
        mask = pd.DatetimeIndex(series.index).to_period("M").isin(kept)
        daily = series[mask]
        month = monthly.series(key)
        if "basis points" in monthly.units[key]:
            out[key] = float(daily.sum() - month.sum())
        else:
            out[key] = float(np.log1p(daily).sum() - np.log1p(month).sum())
    return out


# ---------------------------------------------------------------------------
# The published comparands
# ---------------------------------------------------------------------------


def comparand_series(comparand: Comparand) -> pd.Series:
    """One published monthly series, on **its own observed date index**.

    The index is AQR's, not a generated month end. AQR stamps the last business
    day and this project's calendar module stamps the last *traded* day; the two
    usually agree and the day they do not is the day a generated index would
    silently declare two different months contemporaneous.
    """
    frame = aqr.load_dataset(comparand.dataset)
    if comparand.column not in frame.columns:
        raise ValidationError(
            f"{comparand.dataset} has no {comparand.column!r} column; got {list(frame.columns)}"
        )
    series = pd.to_numeric(frame[comparand.column], errors="coerce").dropna()
    if series.empty:
        raise ValidationError(f"{comparand.dataset}:{comparand.column} is empty after coercion")
    series.index = pd.DatetimeIndex(series.index)
    return series.rename(f"{comparand.dataset}:{comparand.column}")


@dataclass(frozen=True)
class ComparandPair:
    """Whether two comparands are, in fact, the same published series.

    **The rule this class enforces was written after it was violated.** W2-P3's
    placebo gate -- each factor's R^2 against its own comparand must beat its
    R^2 against every other's -- was introduced precisely because it needs no
    invented constant. A duplicated comparand would have made it pass by
    *tautology* while looking like an independent check: `commodity` would have
    been compared against a second copy of its own series and "won". Two of the
    three candidate comparands turned out to be one series, and it was caught by
    somebody noticing rather than by an assertion. See ``experiments.md``, the
    flattering-direction artifact family.

    So: **before any comparand enters a placebo matrix, every pair is checked.**
    """

    left: str
    right: str
    months: int
    max_abs_difference: float
    max_abs_standardized_difference: float
    correlation: float

    @property
    def is_same_series(self) -> bool:
        """Identical to floating point after standardization, over their overlap."""
        if self.months == 0:
            return False
        return self.max_abs_standardized_difference < _DUPLICATE_TOLERANCE

    def render(self) -> str:
        return (
            f"{self.left} vs {self.right}: {self.months} shared months, "
            f"max|diff| {self.max_abs_difference:.2e}, standardized "
            f"{self.max_abs_standardized_difference:.2e}, corr {self.correlation:+.6f}"
        )


def _standardize(values: pd.Series) -> pd.Series:
    spread = float(values.std())
    if spread <= 0.0:
        raise ValidationError("a comparand with zero variance cannot be compared")
    return (values - float(values.mean())) / spread


def compare_comparands(left: Comparand, right: Comparand) -> ComparandPair:
    """Measure whether two comparands are the same series, over their overlap."""
    a, b = comparand_series(left), comparand_series(right)
    joined = pd.concat({"left": a, "right": b}, axis=1, join="inner").dropna()
    label_a = f"{left.dataset}:{left.column}"
    label_b = f"{right.dataset}:{right.column}"
    if joined.empty:
        return ComparandPair(label_a, label_b, 0, float("nan"), float("nan"), float("nan"))
    raw = float((joined["left"] - joined["right"]).abs().max())
    standardized = float((_standardize(joined["left"]) - _standardize(joined["right"])).abs().max())
    return ComparandPair(
        left=label_a,
        right=label_b,
        months=len(joined),
        max_abs_difference=raw,
        max_abs_standardized_difference=standardized,
        correlation=float(joined["left"].corr(joined["right"])),
    )


def assert_comparands_are_distinct(
    comparands: Sequence[Comparand],
) -> tuple[ComparandPair, ...]:
    """Every configured comparand is a different series. Raises if any two are not.

    The general rule W2-P3's near-miss implies: **a 2.3e-12 identity should be
    caught by an assertion, not by someone noticing.** Run before the placebo,
    every time, because a placebo built on a duplicated comparand reports a pass
    that means nothing.
    """
    pairs = tuple(
        compare_comparands(left, right)
        for index, left in enumerate(comparands)
        for right in comparands[index + 1 :]
    )
    duplicates = [pair for pair in pairs if pair.is_same_series]
    if duplicates:
        raise ValidationError(
            "two configured comparands are the same series, which would make the placebo gate "
            "pass by tautology: " + "; ".join(pair.render() for pair in duplicates)
        )
    return pairs


def comparand_volatility_pct_per_year(comparand: Comparand, *, months: int = 12) -> float:
    """Annualised volatility of a published comparand over its full history.

    Reported because it answers, by measurement, a question a reader will
    otherwise answer by assumption: whether AQR's Century series are scaled to a
    constant volatility target. They are not -- see
    ``reports/factor_validation.md``.
    """
    series = comparand_series(comparand)
    return float(series.std() * math.sqrt(months) * 100.0)


def century_volatility_summary(*, months: int = 12) -> dict[str, float]:
    """Annualised volatility across every Century of Factor Premia column.

    Answers by measurement a question a reader would otherwise answer by
    assumption -- whether these series are scaled to a common volatility target.
    Several AQR datasets are, so the question is live; this one is not.
    """
    frame = aqr.load_dataset("century_of_factor_premia")
    vols = frame.apply(lambda column: column.dropna().std() * math.sqrt(months) * 100.0)
    values = vols.to_numpy(dtype=float)
    return {
        "columns": float(len(values)),
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
        "near_ten_percent": float(np.sum((values >= 9.0) & (values <= 11.0))),
    }


def duplicate_comparand_check(left: Comparand, right: Comparand) -> tuple[int, float]:
    """Whether two configured comparands are in fact the same published series.

    Written because they were. Century of Factor Premia's ``Commodities Market``
    and Commodities-for-the-Long-Run's ``Excess return of equal-weight
    commodities portfolio`` agree to 2.3e-12 over all 1,187 overlapping months,
    so listing both as commodity comparands would have been a second reading of
    one number -- and would have let ``commodity`` tie with itself in the
    placebo. Returns ``(overlapping months, max absolute difference)``.
    """
    a, b = comparand_series(left), comparand_series(right)
    joined = pd.concat({"left": a, "right": b}, axis=1, join="inner").dropna()
    if joined.empty:
        return 0, float("nan")
    return len(joined), float((joined["left"] - joined["right"]).abs().max())


# ---------------------------------------------------------------------------
# The regression
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComparandFit:
    """One self-built factor regressed on one published series.

    SPEC.md 6.5's own list, plus the three things it does not name and without
    which the row cannot be read: the monthly ``observations``, the window, and
    the units of both sides.
    """

    factor: str
    comparand: str
    dataset: str
    column: str
    gate: str
    observations: int
    start: pd.Period
    end: pd.Period
    #: The comparand's own observed dates at the ends of the joined window.
    observed_start: pd.Timestamp
    observed_end: pd.Timestamp
    newey_west_lags: int
    r_squared: float
    correlation: float
    beta: float
    beta_standard_error: float
    #: Monthly intercept in basis points; annualised for display only.
    alpha_bps_per_month: float
    alpha_t_statistic: float
    factor_units: str
    comparand_units: str
    beta_units: str
    #: First two moments of both sides over the joined window, in the regression
    #: unit. Carried because beta = corr x sd(factor)/sd(comparand), so a beta
    #: away from 1 is either a correlation effect or a scale effect and the two
    #: have completely different readings -- see reports/factor_validation.md.
    factor_sd_bps_per_month: float
    comparand_sd_bps_per_month: float
    factor_mean_bps_per_month: float
    comparand_mean_bps_per_month: float

    @property
    def scale_ratio(self) -> float:
        """``sd(factor) / sd(comparand)``. The scale half of the beta decomposition."""
        return self.factor_sd_bps_per_month / self.comparand_sd_bps_per_month

    @property
    def mean_gap_bps_per_month(self) -> float:
        return self.factor_mean_bps_per_month - self.comparand_mean_bps_per_month

    @property
    def implied_duration_bracket(self) -> tuple[float, float, float]:
        """``(low, central, high)`` years implied by a yield-change-on-return fit.

        Only meaningful for the rate factor. The two regression directions
        bracket the answer and the volatility ratio sits between them, because
        each regression attenuates toward the axis carrying the noise:

        * ``|R^2/beta|`` -- regress the comparand's return on the yield change;
        * ``sd(comparand)/sd(factor)`` -- the orthogonal estimate, which is the
          geometric mean of the two and assumes noise in neither;
        * ``1/|beta|`` -- the reciprocal of the fitted slope.
        """
        low = abs(self.r_squared / self.beta)
        high = abs(1.0 / self.beta)
        return low, self.comparand_sd_bps_per_month / self.factor_sd_bps_per_month, high

    @property
    def alpha_pct_per_year(self) -> float:
        return self.alpha_bps_per_month * _MONTHS_PER_YEAR / _BPS_PER_UNIT * 100.0

    @property
    def window(self) -> str:
        return f"{self.start}..{self.end}"


def _to_basis_points(series: pd.Series, units: str) -> pd.Series:
    """Put a monthly series in basis points, leaving one already there alone."""
    return series if "basis points" in units else series * _BPS_PER_UNIT


def _join(factor: pd.Series, comparand: pd.Series, *, boundary: date, label: str) -> pd.DataFrame:
    """Align a monthly factor onto the comparand's **observed** date index.

    The join key is the calendar month; the surviving index is the comparand's
    own date. Two things are asserted rather than trusted, because both are ways
    this session could produce a plausible table that is wrong:

    * the right edge is ``holdout_start``, always -- the comparand's coverage may
      bind the left edge or an *earlier* right edge, never a later one, and all
      three AQR files end after the boundary (2026-02-27, 2026-05-29, 2025-05-30
      against 2025-01-01);
    * the joined count equals the count computed from the two coverages, so a
      month lost to a stamp mismatch cannot pass as a month the sources did not
      share.
    """
    limit = pd.Timestamp(boundary)
    published = comparand[comparand.index < limit]
    if published.empty:
        raise ValidationError(f"{label}: the comparand has no observation before {limit.date()}")
    observed = pd.DatetimeIndex(published.index)
    keyed = pd.DataFrame(
        {"comparand": published.to_numpy(dtype=float), "observed": observed},
        index=observed.to_period("M"),
    )
    if keyed.index.has_duplicates:
        raise ValidationError(f"{label}: the comparand carries two observations in one month")

    joined = keyed.join(factor.rename("factor"), how="inner").dropna()

    expected = len(set(factor.index) & set(keyed.index))
    if len(joined) != expected:
        raise ValidationError(
            f"{label}: the join produced {len(joined)} months, expected {expected} from the two "
            "coverages. A month was lost to a stamp mismatch rather than to coverage."
        )
    if joined.empty:
        raise ValidationError(f"{label}: the factor and the comparand share no month")
    if pd.DatetimeIndex(joined["observed"]).max() >= limit:
        raise ValidationError(
            f"{label}: an observation lands on or after the holdout boundary {limit.date()}"
        )
    return joined.sort_index()


def fit_comparand(
    monthly: MonthlyPanel,
    comparand: Comparand,
    *,
    factor: str | None = None,
    config: Config | None = None,
) -> ComparandFit:
    """Regress one monthly factor on one published monthly series.

    The self-built factor is the **dependent** variable, per SPEC.md 6.5's own
    wording ("regress each self-built factor on the corresponding ... series"),
    so ``beta`` is the factor's loading on the published analogue and ``alpha``
    is the part of the factor the analogue does not carry.
    """
    settings = config or load()
    name = factor if factor is not None else comparand.factor
    boundary = settings.require_holdout_start()

    units = monthly.units[name]
    left = _to_basis_points(monthly.series(name), units)
    right = comparand_series(comparand) * _BPS_PER_UNIT
    label = f"{name} vs {comparand.dataset}:{comparand.column}"
    joined = _join(left, right, boundary=boundary, label=label)

    lags = newey_west_plug_in_lags(len(joined))
    stamped = pd.DatetimeIndex(joined["observed"])
    fit = betas.full_sample_fit(
        pd.Series(joined["factor"].to_numpy(dtype=float), index=stamped, name=name),
        pd.DataFrame({"comparand": joined["comparand"].to_numpy(dtype=float)}, index=stamped),
        fit_intercept=True,
        newey_west_lags=lags,
        label=name,
    )
    beta = float(fit.beta["comparand"])
    correlation = float(np.sign(beta) * math.sqrt(max(fit.r_squared, 0.0)))
    return ComparandFit(
        factor=name,
        comparand=f"{comparand.dataset}:{comparand.column}",
        dataset=comparand.dataset,
        column=comparand.column,
        gate=comparand.gate,
        observations=len(joined),
        start=pd.Period(joined.index.min()),
        end=pd.Period(joined.index.max()),
        observed_start=pd.Timestamp(stamped.min()),
        observed_end=pd.Timestamp(stamped.max()),
        newey_west_lags=lags,
        r_squared=float(fit.r_squared),
        correlation=correlation,
        beta=beta,
        beta_standard_error=float(fit.standard_error["comparand"]),
        alpha_bps_per_month=float(fit.alpha),
        alpha_t_statistic=float(fit.alpha_t_statistic),
        factor_units=(
            "basis points of yield per month" if "basis points" in units else _REGRESSION_UNIT
        ),
        comparand_units=_REGRESSION_UNIT,
        beta_units="years^-1" if "basis points" in units else "dimensionless",
        factor_sd_bps_per_month=float(joined["factor"].std()),
        comparand_sd_bps_per_month=float(joined["comparand"].std()),
        factor_mean_bps_per_month=float(joined["factor"].mean()),
        comparand_mean_bps_per_month=float(joined["comparand"].mean()),
    )


def rolling_correlation(
    monthly: MonthlyPanel,
    comparand: Comparand,
    *,
    config: Config | None = None,
) -> pd.Series:
    """Rolling correlation of a factor with its comparand, on the configured window.

    The window is a CHOICE (36 months), recorded as one in ``config/model.yaml``,
    and it is always printed beside
    :attr:`ValidationConfig.rolling_correlation_standard_error` -- 0.174 at 36
    points -- because a drift from 0.55 to 0.75 inside a band that wide is not a
    change in anything.
    """
    settings = config or load()
    validation = settings.model.factors.validation
    units = monthly.units[comparand.factor]
    left = _to_basis_points(monthly.series(comparand.factor), units)
    right = comparand_series(comparand) * _BPS_PER_UNIT
    joined = _join(
        left,
        right,
        boundary=settings.require_holdout_start(),
        label=f"rolling {comparand.factor}",
    )
    window = validation.rolling_correlation_window_months
    rolled = joined["factor"].rolling(window).corr(joined["comparand"]).dropna()
    rolled.index = pd.PeriodIndex(joined.index[-len(rolled) :], freq="M")
    return rolled.rename(f"{comparand.factor}~{comparand.column}")


# ---------------------------------------------------------------------------
# The placebo -- what replaces the withdrawn 0.3 gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlaceboRow:
    """One factor's R^2 against every mapped comparand, its own included.

    The gate is **row-wise**: the window and the dependent variable are held
    fixed and only the comparand varies, so the comparison is a like-for-like
    one. Reading the matrix down a column compares different windows and is not
    what the gate asserts.
    """

    factor: str
    own: str
    r_squared: Mapping[str, float]
    observations: Mapping[str, int]

    @property
    def own_r_squared(self) -> float:
        return self.r_squared[self.own]

    @property
    def best_other(self) -> tuple[str, float]:
        others = {k: v for k, v in self.r_squared.items() if k != self.own}
        name = max(others, key=lambda k: others[k])
        return name, others[name]

    @property
    def passed(self) -> bool:
        """Own comparand beats every other. No constant, and it can fail."""
        _, best = self.best_other
        return self.own_r_squared > best

    @property
    def margin(self) -> float:
        _, best = self.best_other
        return self.own_r_squared - best


def placebo(monthly: MonthlyPanel, *, config: Config | None = None) -> tuple[PlaceboRow, ...]:
    """Regress each mapped factor on every mapped factor's comparand.

    This is what replaces SPEC.md 6.5's withdrawn 0.3 correlation gate, and the
    replacement is not a lowering. 0.3 was written before anyone had looked at
    what the comparands contain, and SPEC.md 6.5's only published figures --
    0.6-0.8 strong, 0.2 wrong -- are stated for a *trend* factor against AQR
    TSMOM, which maps to nothing in Model A's factor set. It was a threshold for
    a comparand that does not exist. The placebo is falsifiable without any
    constant at all: if ``commodity`` explains AQR's Fixed income Market as well
    as it explains AQR's commodity index, the mapping is doing no work.
    """
    settings = config or load()
    validation = settings.model.factors.validation
    # BEFORE any regression. A placebo built on a duplicated comparand reports a
    # pass that means nothing, and this project has already come within one
    # config line of exactly that.
    assert_comparands_are_distinct(validation.comparands)
    rows: list[PlaceboRow] = []
    for factor in validation.placebo_factors:
        scores: dict[str, float] = {}
        counts: dict[str, int] = {}
        for other in validation.comparands:
            fit = fit_comparand(monthly, other, factor=factor, config=settings)
            scores[other.factor] = fit.r_squared
            counts[other.factor] = fit.observations
        rows.append(PlaceboRow(factor=factor, own=factor, r_squared=scores, observations=counts))
    return tuple(rows)


# ---------------------------------------------------------------------------
# The one measured prior -- experiments.md row 41
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DailyAnchorCheck:
    """Our own SPY excess return against the equity factor, at daily frequency.

    The equity factor **is** Ken French ``Mkt-RF``, so it cannot be validated
    against Ken French -- that regression returns R^2 = 1 and proves nothing
    (SPEC.md 4.1.1 ruling 2). What ruling 2 left is our own SPY excess return,
    which it converted from a candidate input into a comparand, and which
    experiments.md row 41 has already measured. That makes this the only row in
    the W2-P3 table with a sharp prior rather than a guessed threshold.
    """

    label: str
    observations: int
    start: pd.Timestamp
    end: pd.Timestamp
    correlation: float
    mean_gap_pct_per_year: float
    #: Fisher-z 95% interval on ``correlation`` at this ``observations``.
    correlation_low: float
    correlation_high: float

    @property
    def fisher_standard_error(self) -> float:
        return 1.0 / math.sqrt(self.observations - 3)

    def reproduces(self, registered: float) -> bool:
        """Whether ``registered`` lies inside this window's sampling interval."""
        return self.correlation_low <= registered <= self.correlation_high


def _daily_equity_legs(config: Config) -> tuple[pd.Series, pd.Series, pd.Series]:
    """SPY excess, Ken French ``Mkt-RF``, and their difference, on shared dates."""
    data = config.model.data
    ticker = data.cross_checks.spy_versus_market.ticker.lower()
    manifest = cache.Manifest.load()
    raw = cache.read(manifest.latest(source="yfinance", name=f"{ticker}_prices"), manifest=manifest)
    actions = cache.read(
        manifest.latest(source="yfinance", name=f"{ticker}_actions"), manifest=manifest
    )
    ours = prices.total_return(raw, actions)
    table = french.load_daily_factors()
    common = ours.index.intersection(table.index)
    if len(common) == 0:
        raise ValidationError("SPY and the Ken French table share no dates")
    rf_column = data.risk_free.ken_french_daily_rf.rf_column
    # Both published in percent per trading day.
    spy = ours.loc[common] - table.loc[common, rf_column] / 100.0
    market = table.loc[common, config.model.factors.macro.equity.column] / 100.0
    return spy, market, spy - market


def _anchor(
    label: str, spy: pd.Series, market: pd.Series, mask: pd.Series, *, config: Config
) -> DailyAnchorCheck:
    left, right = spy[mask], market[mask]
    n = len(left)
    if n < 4:
        raise ValidationError(f"{label}: {n} observations cannot support a correlation interval")
    correlation = float(left.corr(right))
    z = math.atanh(correlation)
    error = 1.0 / math.sqrt(n - 3)
    index = pd.DatetimeIndex(left.index)
    return DailyAnchorCheck(
        label=label,
        observations=n,
        start=pd.Timestamp(index.min()),
        end=pd.Timestamp(index.max()),
        correlation=correlation,
        mean_gap_pct_per_year=float(
            (left - right).mean() * config.model.data.trading_days_per_year * 100.0
        ),
        correlation_low=float(math.tanh(z - 1.96 * error)),
        correlation_high=float(math.tanh(z + 1.96 * error)),
    )


def daily_anchor_check(*, config: Config | None = None) -> tuple[DailyAnchorCheck, ...]:
    """Row 41's measurement, on row 41's window and on W2-P3's in-sample window.

    Both are reported because the falsifier is about **reproduction**, not about
    agreement between two different windows. Row 41 ran on the full common
    history; W2-P3 may not look past ``holdout_start``. Reporting only the
    in-sample number and comparing it to 0.9786 would charge a window difference
    to sampling error, which is the error this project's own instruction --
    *every comparison states its own window* -- exists to prevent.
    """
    settings = config or load()
    spy, market, _ = _daily_equity_legs(settings)
    index = pd.DatetimeIndex(spy.index)
    start = pd.Timestamp(settings.model.sample.start)
    boundary = pd.Timestamp(settings.require_holdout_start())

    # Row 41's own window, truncated at the holdout boundary. Row 41 itself ran
    # to the end of the cache; this cannot, and the truncation is stated rather
    # than absorbed.
    full = pd.Series(index < boundary, index=index)
    in_sample = pd.Series((index >= start) & (index < boundary), index=index)
    pre_sample = pd.Series(index < start, index=index)
    return (
        _anchor("row 41 window, truncated at holdout", spy, market, full, config=settings),
        _anchor("W2-P3 in-sample", spy, market, in_sample, config=settings),
        _anchor("pre-sample only", spy, market, pre_sample, config=settings),
    )


# ---------------------------------------------------------------------------
# The four flagged alphas -- answered here or handed on with a control
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AlphaConfound:
    """Whether a published comparand accounts for one flagged W2-P2 intercept.

    W2-P2 flagged four assets with persistently non-zero alphas against 0.59
    expected by chance, and rows 79-82 identified all four as government carry
    and roll-down -- a term premium a factor set built from yield *changes*
    cannot span. A validation session is the natural place to ask whether the
    published analogues carry it. The test adds the comparand to the six-factor
    regression and measures what happens to the intercept.
    """

    asset: str
    observations: int
    alpha_pct_per_year: float
    alpha_t_statistic: float
    augmented_alpha_pct_per_year: float
    augmented_alpha_t_statistic: float
    comparand_beta: float
    delta_r_squared: float

    @property
    def shift_pct_per_year(self) -> float:
        return self.augmented_alpha_pct_per_year - self.alpha_pct_per_year

    @property
    def explained(self) -> bool:
        """Whether adding the comparand removes the flag.

        Deliberately strict and deliberately simple: the intercept has to stop
        being distinguishable from zero at the same threshold W2-P2 flagged it
        with. A shrinkage that leaves ``t = 4`` is not an explanation.
        """
        limit = load().model.factors.macro.alpha_flag.abs_t_statistic
        return abs(self.augmented_alpha_t_statistic) < limit


def alpha_confound(
    monthly: MonthlyPanel,
    *,
    assets: tuple[str, ...] | None = None,
    config: Config | None = None,
) -> tuple[AlphaConfound, ...]:
    """Do the published comparands explain W2-P2's four flagged alphas?

    Runs the six-factor regression at monthly frequency with and without AQR's
    ``Fixed income Market`` -- the only comparand in the mapping that carries a
    bond term premium at all -- for each flagged asset.
    """
    settings = config or load()
    validation = settings.model.factors.validation
    boundary = settings.require_holdout_start()
    flagged = assets if assets is not None else _flagged_assets(settings)

    comparand = validation.by_factor(macro.LEVEL)
    published = comparand_series(comparand) * _BPS_PER_UNIT

    factor_columns = [str(name) for name in monthly.frame.columns]
    design = pd.DataFrame(
        {
            name: _to_basis_points(monthly.frame[name], monthly.units[name])
            for name in factor_columns
        }
    )
    returns = betas.asset_excess_returns(config=settings)

    out: list[AlphaConfound] = []
    for asset in flagged:
        series = returns[asset].dropna()
        index = pd.DatetimeIndex(series.index)
        window = (index >= pd.Timestamp(settings.model.sample.start)) & (
            index < pd.Timestamp(boundary)
        )
        series = series[window]
        own = pd.DatetimeIndex(series.index).to_period("M")
        month = np.expm1(np.log1p(series).groupby(own).sum()) * _BPS_PER_UNIT

        base = pd.concat({"asset": month}, axis=1).join(design, how="inner").dropna()
        keyed = pd.Series(
            published.to_numpy(dtype=float),
            index=pd.DatetimeIndex(published.index).to_period("M"),
            name="comparand",
        )
        augmented = base.join(keyed, how="inner").dropna()
        if len(augmented) != len(base):
            raise ValidationError(
                f"{asset}: the comparand does not cover the {len(base)} months the six-factor "
                f"regression runs on ({len(augmented)} matched). The two arms would not be "
                "comparable."
            )

        stamps = base.index.to_timestamp()
        plain = betas.full_sample_fit(
            pd.Series(base["asset"].to_numpy(dtype=float), index=stamps, name=asset),
            pd.DataFrame(
                base[factor_columns].to_numpy(dtype=float), index=stamps, columns=factor_columns
            ),
            fit_intercept=True,
            newey_west_lags=newey_west_plug_in_lags(len(base)),
            label=asset,
        )
        columns = [*factor_columns, "comparand"]
        wide = betas.full_sample_fit(
            pd.Series(augmented["asset"].to_numpy(dtype=float), index=stamps, name=asset),
            pd.DataFrame(augmented[columns].to_numpy(dtype=float), index=stamps, columns=columns),
            fit_intercept=True,
            newey_west_lags=newey_west_plug_in_lags(len(augmented)),
            label=asset,
        )
        scale = _MONTHS_PER_YEAR / _BPS_PER_UNIT * 100.0
        out.append(
            AlphaConfound(
                asset=asset,
                observations=len(base),
                alpha_pct_per_year=float(plain.alpha) * scale,
                alpha_t_statistic=float(plain.alpha_t_statistic),
                augmented_alpha_pct_per_year=float(wide.alpha) * scale,
                augmented_alpha_t_statistic=float(wide.alpha_t_statistic),
                comparand_beta=float(wide.beta["comparand"]),
                delta_r_squared=float(wide.r_squared - plain.r_squared),
            )
        )
    return tuple(out)


def _flagged_assets(settings: Config) -> tuple[str, ...]:
    """The four synthetic government zeros. experiments.md rows 79-80.

    Read from the frozen universe by construction rather than listed, so adding
    a curve point cannot leave this behind.
    """
    return tuple(
        asset.id
        for asset in settings.universe.assets
        if asset.construction == "synthetic_gsw_zero_curve"
    )


# ---------------------------------------------------------------------------
# Everything, built once
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationResult:
    """The whole W2-P3 table, built once so no two sections can disagree."""

    settings: ValidationConfig
    panel: macro.FactorPanel
    monthly: MonthlyPanel
    round_trip: Mapping[str, float]
    fits: tuple[ComparandFit, ...]
    placebo_rows: tuple[PlaceboRow, ...]
    rolling: Mapping[str, pd.Series]
    anchors: tuple[DailyAnchorCheck, ...]
    confounds: tuple[AlphaConfound, ...]
    comparand_pairs: tuple[ComparandPair, ...]
    duplicate_months: int
    duplicate_max_difference: float

    def fit(self, factor: str) -> ComparandFit:
        for item in self.fits:
            if item.factor == factor:
                return item
        raise ValidationError(f"no fit for {factor!r}")

    @property
    def sign_gates(self) -> tuple[ComparandFit, ...]:
        return tuple(item for item in self.fits if item.gate == "sign")

    @property
    def sign_gate_passed(self) -> bool:
        """Every sign-test comparand's beta carries the predicted sign."""
        for item in self.sign_gates:
            expected = self.settings.by_factor(item.factor).expected_beta_sign
            if expected is None or math.copysign(1.0, item.beta) != expected:
                return False
        return True

    @property
    def placebo_passed(self) -> bool:
        return all(row.passed for row in self.placebo_rows)

    @property
    def anchor_passed(self) -> bool:
        """Row 41 reproduced on row 41's own window, to within sampling error."""
        registered = self.settings.daily_anchor.registered_correlation
        return self.anchors[0].reproduces(registered)

    @property
    def passed(self) -> bool:
        return self.anchor_passed and self.placebo_passed and self.sign_gate_passed

    @property
    def unexplained_confounds(self) -> tuple[AlphaConfound, ...]:
        return tuple(item for item in self.confounds if not item.explained)


def build(config: Config | None = None) -> ValidationResult:
    """Every measurement in W2-P3, from the cache."""
    settings = config or load()
    validation = settings.model.factors.validation
    panel = macro.macro_factor_panel(config=settings)
    monthly = monthly_factor_returns(panel)

    fits = tuple(fit_comparand(monthly, item, config=settings) for item in validation.comparands)
    rolling = {
        item.factor: rolling_correlation(monthly, item, config=settings)
        for item in validation.comparands
    }
    months, difference = duplicate_comparand_check(
        Comparand(
            factor=macro.COMMODITY,
            dataset="century_of_factor_premia",
            column="Commodities Market",
            gate="placebo",
            expected_beta_sign=None,
        ),
        validation.by_factor(macro.COMMODITY),
    )
    return ValidationResult(
        comparand_pairs=assert_comparands_are_distinct(validation.comparands),
        settings=validation,
        panel=panel,
        monthly=monthly,
        round_trip=monthly_round_trip(panel, monthly),
        fits=fits,
        placebo_rows=placebo(monthly, config=settings),
        rolling=rolling,
        anchors=daily_anchor_check(config=settings),
        confounds=alpha_confound(monthly, config=settings),
        duplicate_months=months,
        duplicate_max_difference=difference,
    )
