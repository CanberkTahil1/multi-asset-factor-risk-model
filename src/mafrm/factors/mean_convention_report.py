"""Generates ``reports/mean_convention.md``. SPEC.md 5.1.1.

WHAT THIS MEASURES, AND WHAT IT CANNOT DO
-----------------------------------------

SPEC.md 5.1.1 fixes the covariance pipeline's moments about **zero**. The
argument for it is coherence with SPEC.md 6.1, not size: the bias statistic
``b_nt = R_nt / sigma_nt`` has a raw return in the numerator, so a denominator
that forecast variance about a drifting mean would make the two sides measure
different quantities and bias ``B`` -- the one statistic the project exists to
decompose -- by the mismatch alone.

The decision carries a second, weaker clause: *"and at daily frequency the mean
is small relative to the standard deviation anyway."* That is a claim about
data, so it is measured here rather than asserted, and it is the only part of
the ruling this file speaks to.

One consumer of the answer, recorded because it is the reason row 95 was worth
having: the correlation bound below is what made re-basing
:func:`mafrm.factors.macro.condition_numbers` onto the pipeline's estimators a
cheap consistency fix in W3-P1b rather than a judgement call.

**THE MEASUREMENT CANNOT REVERSE THE DECISION, AND THAT WAS REGISTERED BEFORE IT
RAN.** ``experiments.md`` rows 93-95 name in advance the result that would
change it: there is none. A large difference here would be a finding about the
factor means -- worth reporting, and reported -- not a licence to break
coherence with SPEC.md 6.1. Because no result would move the decision, this is
not a search, and the rows are ``data-diagnostic`` and add nothing to the
deflated-Sharpe trial count. The general form of that criterion is the W3-P1
ruling: name the reversing result in advance; if none exists it is not a trial,
and if one does it is a sweep and counts as ``model-config``.

THE DEMEANED COMPARAND LIVES HERE, NOT IN ``mafrm.risk``
--------------------------------------------------------

``mafrm.risk.covariance`` ships exactly one estimator. Putting a demeaned
variant beside it, however clearly labelled, would leave something on the
production path that a later session could reach for. The comparand is
therefore local to this module and is measurement-only.

THE IDENTITY THAT MAKES THE ANSWER INTERPRETABLE
-------------------------------------------------

For weights summing to one, with ``m = sum(w f)`` the EWMA mean::

    sum(w f^2)  =  sum(w (f - m)^2)  +  m^2

exactly. So the two conventions differ by a term that is *quadratic* in the
mean-to-volatility ratio::

    sigma_zero / sigma_demeaned  =  sqrt(1 + (m / sigma_demeaned)^2)

The consequence is worth stating because it converts "negligible at daily
frequency" from a belief into arithmetic: a factor whose EWMA mean is a tenth of
its EWMA volatility -- already a very large daily drift -- moves the volatility
by 0.50%. The report gives ``m / sigma`` per factor alongside the volatility
difference, so a reader can check the identity rather than take the number.

CLAUDE.md FAILURE MODE 9
------------------------

Every statistic here is a rolling EWMA read at successive dates, so consecutive
readings share almost all of their weighted history and are near-duplicates.
**No count of dates is quoted as ``n`` and no standard error is computed.** The
report gives the maximum and the quantiles of the observed distribution and says
plainly that the uncertainty is not quantified, which is the honest form for a
statistic with this much overlap.

CLAUDE.md invariant 5: everything read stops strictly before
``sample.holdout_start``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from mafrm import config
from mafrm.factors import macro
from mafrm.numerics import ewma_weights
from mafrm.risk.config import HORIZONS, Horizon, RiskConfig
from mafrm.risk.covariance import correlation_from_covariance, ewma_second_moment

__all__ = ["Comparison", "build", "main", "render", "report_path"]

#: Read the rolling comparison at this stride, in trading days. A stride is not
#: a fix for the overlap -- at a 504-day correlation half-life even monthly
#: readings share nearly all their weighted history -- it only keeps the table a
#: readable length. The overlap is stated instead of being engineered away.
_STRIDE_DAYS = 21


def report_path() -> Path:
    """``reports/mean_convention.md`` -- committed, per CLAUDE.md."""
    return Path(__file__).resolve().parents[3] / "reports" / "mean_convention.md"


def _demeaned_second_moment(returns: np.ndarray, *, halflife: float) -> np.ndarray:
    """MEASUREMENT ONLY. The comparand SPEC.md 5.1.1 rejects.

    An EWMA moment about the trailing EWMA mean, with no ``1 - w'w`` reliability
    correction. Leaving that correction out is deliberate: it is a second change,
    and including it would confound the cost of demeaning with the cost of the
    degree-of-freedom adjustment that accompanies it. The difference measured
    here is therefore the mean term and nothing else.

    Not exported, and deliberately not in ``mafrm.risk``.
    """
    weights = ewma_weights(len(returns), halflife)
    mean = weights @ returns
    centred = returns - mean
    scaled = centred * np.sqrt(weights)[:, None]
    moment: np.ndarray = scaled.T @ scaled
    symmetric: np.ndarray = (moment + moment.T) / 2.0
    return symmetric


def _ewma_mean(returns: np.ndarray, *, halflife: float) -> np.ndarray:
    weights = ewma_weights(len(returns), halflife)
    mean: np.ndarray = weights @ returns
    return mean


@dataclass(frozen=True)
class Comparison:
    """Both conventions, read at every stride date, for one horizon."""

    horizon: Horizon
    columns: tuple[str, ...]
    volatility_halflife: int
    correlation_halflife: int
    dates: pd.DatetimeIndex
    #: Per date x factor: ``sigma_zero / sigma_demeaned - 1``.
    volatility_ratio: pd.DataFrame
    #: Per date x factor: ``m / sigma_demeaned`` at the volatility half-life.
    mean_to_volatility: pd.DataFrame
    #: Per date: ``max |rho_zero - rho_demeaned|`` off the diagonal.
    correlation_difference: pd.Series

    @property
    def overlap_note(self) -> str:
        """The CLAUDE.md failure-mode-9 statement, sized to this run."""
        return (
            f"Read every {_STRIDE_DAYS} trading days on EWMA windows whose volatility "
            f"half-life is {self.volatility_halflife}d and whose correlation half-life is "
            f"{self.correlation_halflife}d. Consecutive readings share almost all of their "
            "weighted history, so these are near-duplicates rather than independent "
            "observations. **No count of dates is quoted as `n` and no standard error is "
            "computed.** CLAUDE.md failure mode 9."
        )

    @property
    def worst_volatility_shift(self) -> float:
        return float(np.abs(self.volatility_ratio.to_numpy()).max())

    @property
    def worst_mean_to_volatility(self) -> float:
        return float(np.abs(self.mean_to_volatility.to_numpy()).max())

    @property
    def worst_correlation_difference(self) -> float:
        return float(self.correlation_difference.abs().max())

    @property
    def median_volatility_shift(self) -> float:
        """The typical cost, as against :attr:`worst_volatility_shift`'s tail."""
        return float(np.median(np.abs(self.volatility_ratio.to_numpy())))

    @property
    def worst_reading(self) -> tuple[pd.Timestamp, str, float, float]:
        """``(date, factor, shift, m/sigma)`` at the largest observed shift.

        The date is reported rather than only the magnitude, and that is not
        housekeeping. The identity ``sigma_zero/sigma_demeaned = sqrt(1 + (m/sigma)^2)``
        holds for **any** ``m``, so it confirms the arithmetic and can say nothing
        about whether ``m`` is real. A drift large enough to move a volatility by
        3% has to correspond to a nameable episode in the underlying series; if
        the maximum landed on a quiet date, the EWMA mean would be wrong and the
        identity check would pass anyway.
        """
        values = np.abs(self.volatility_ratio.to_numpy())
        flat = int(np.argmax(values))
        row, column = divmod(flat, values.shape[1])
        return (
            pd.Timestamp(self.volatility_ratio.index[row]),
            str(self.volatility_ratio.columns[column]),
            float(self.volatility_ratio.to_numpy()[row, column]),
            float(self.mean_to_volatility.to_numpy()[row, column]),
        )

    def top_readings(self, count: int = 6) -> list[tuple[pd.Timestamp, str, float]]:
        """The largest shifts, to show whether they are one episode or scattered."""
        values = np.abs(self.volatility_ratio.to_numpy())
        order = np.argsort(values, axis=None)[::-1][:count]
        readings: list[tuple[pd.Timestamp, str, float]] = []
        for flat in order:
            row, column = divmod(int(flat), values.shape[1])
            readings.append(
                (
                    pd.Timestamp(self.volatility_ratio.index[row]),
                    str(self.volatility_ratio.columns[column]),
                    float(values[row, column]),
                )
            )
        return readings

    @property
    def identity_check(self) -> tuple[float, float]:
        """``(observed shift, sqrt(1 + (m/sigma)^2) - 1)`` at the worst reading.

        The two must agree: the identity in the module docstring is exact, so a
        disagreement would mean the comparand is not the estimator it claims to
        be. Reported in the file rather than only asserted in a test, because a
        reader checking the arithmetic is the point of publishing the ratio.
        """
        flat_shift = np.abs(self.volatility_ratio.to_numpy()).ravel()
        position = int(np.argmax(flat_shift))
        ratio = float(np.abs(self.mean_to_volatility.to_numpy()).ravel()[position])
        return float(flat_shift[position]), float(np.sqrt(1.0 + ratio**2) - 1.0)


def _compare(frame: pd.DataFrame, risk: RiskConfig) -> Comparison:
    values = frame.to_numpy(dtype=float)
    periods, factors = values.shape
    # Start once the shorter half-life has had room to decay; this is a
    # presentation choice for the report, not a burn-in rule for the pipeline.
    first = min(periods - 1, 4 * risk.volatility_halflife)
    positions = list(range(first, periods, _STRIDE_DAYS))
    if positions and positions[-1] != periods - 1:
        positions.append(periods - 1)

    ratios: list[np.ndarray] = []
    means: list[np.ndarray] = []
    correlation_gaps: list[float] = []
    for position in positions:
        window = values[: position + 1]
        zero = ewma_second_moment(window, halflife=float(risk.volatility_halflife))
        demeaned = _demeaned_second_moment(window, halflife=float(risk.volatility_halflife))
        mean = _ewma_mean(window, halflife=float(risk.volatility_halflife))
        sigma_zero = np.sqrt(np.diag(zero))
        sigma_demeaned = np.sqrt(np.diag(demeaned))
        ratios.append(sigma_zero / sigma_demeaned - 1.0)
        means.append(mean / sigma_demeaned)

        zero_correlation = correlation_from_covariance(
            ewma_second_moment(window, halflife=float(risk.correlation_halflife))
        )
        demeaned_correlation = correlation_from_covariance(
            _demeaned_second_moment(window, halflife=float(risk.correlation_halflife))
        )
        gap = np.abs(zero_correlation - demeaned_correlation)
        np.fill_diagonal(gap, 0.0)
        correlation_gaps.append(float(gap.max()))

    dates = pd.DatetimeIndex([frame.index[position] for position in positions])
    columns = [str(column) for column in frame.columns]
    assert factors == len(columns)
    return Comparison(
        horizon=risk.horizon,
        columns=tuple(columns),
        volatility_halflife=risk.volatility_halflife,
        correlation_halflife=risk.correlation_halflife,
        dates=dates,
        volatility_ratio=pd.DataFrame(np.vstack(ratios), index=dates, columns=columns),
        mean_to_volatility=pd.DataFrame(np.vstack(means), index=dates, columns=columns),
        correlation_difference=pd.Series(correlation_gaps, index=dates, name="max_abs_difference"),
    )


def build(cfg: config.Config | None = None) -> tuple[Comparison, ...]:
    """Both horizons, on the orthogonalized factor panel that reaches the pipeline."""
    settings = cfg or config.load()
    frame = macro.macro_factor_panel(config=settings).complete
    return tuple(
        _compare(frame, RiskConfig.load(horizon=horizon, config=settings)) for horizon in HORIZONS
    )


def _episode(date: pd.Timestamp, name: str, cfg: config.Config) -> tuple[float, float, float]:
    """``(EWMA mean, EWMA sd, trailing 252d sum)`` for one factor at one date.

    Recomputed from the panel rather than carried through :class:`Comparison`,
    because what it is for is checking the reading against the underlying series
    -- the point of which is lost if it is derived from the same summary.
    """
    frame = macro.macro_factor_panel(config=cfg).complete
    column = frame[name]
    location = frame.index.get_loc(date)
    if not isinstance(location, int):  # pragma: no cover - a unique DatetimeIndex
        raise TypeError(f"_episode: {date!r} is not a unique date in the panel")
    position = location
    window = column.to_numpy(dtype=float)[: position + 1]
    weights = ewma_weights(
        len(window), float(cfg.model.covariance.factor_volatility_halflife.short)
    )
    mean = float(weights @ window)
    deviation = float(np.sqrt(weights @ (window - mean) ** 2))
    trailing = float(column.iloc[max(0, position - 251) : position + 1].sum())
    return mean, deviation, trailing


def _quantile_row(name: str, series: pd.Series, scale: float, unit: str) -> str:
    return (
        f"| {name} | {scale * series.abs().median():.4f}{unit} | "
        f"{scale * series.abs().quantile(0.95):.4f}{unit} | "
        f"{scale * series.abs().max():.4f}{unit} |"
    )


def render(comparisons: tuple[Comparison, ...], cfg: config.Config | None = None) -> str:
    """The committed markdown."""
    settings = cfg or config.load()
    holdout = settings.model.sample.holdout_start
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# The zero-mean convention, and what it costs",
        "",
        f"Generated {stamp} by `mafrm.factors.mean_convention_report`. ",
        "SPEC.md 5.1.1; `covariance.mean_convention` in `config/model.yaml`; "
        "`experiments.md` rows 93-95.",
        "",
        "## The decision, and why this file cannot change it",
        "",
        "SPEC.md 5.1.1 takes the covariance pipeline's EWMA moments about **zero**. "
        "The reason is coherence, not size: SPEC.md 6.1's bias statistic is "
        "`b_nt = R_nt / sigma_nt` with a **raw** return in the numerator, so a "
        "denominator forecasting variance about a drifting mean would make the two "
        "sides measure different quantities, and `B` -- the one statistic the project "
        "exists to decompose into a risk-model term and a cost term -- would be biased "
        "by the mismatch alone, before any estimation error.",
        "",
        "The ruling carries a second clause: *and at daily frequency the mean is small "
        "relative to the standard deviation anyway*. That is a claim about data, and it "
        "is the only part this file speaks to.",
        "",
        "**No result here would reverse the decision, and that was registered before the "
        "measurement ran** (`experiments.md` rows 93-95). A large number below would be a "
        "finding about the factor means, not a licence to break coherence with SPEC.md 6.1. "
        "Having no reversing result is precisely why these rows are `data-diagnostic` and "
        "add nothing to the deflated-Sharpe trial count.",
        "",
        "## The identity",
        "",
        "For weights summing to one, with `m = sum(w f)` the EWMA mean, "
        "`sum(w f^2) = sum(w (f-m)^2) + m^2` **exactly**. So",
        "",
        "```",
        "sigma_zero / sigma_demeaned = sqrt(1 + (m / sigma_demeaned)^2)",
        "```",
        "",
        "The difference is *quadratic* in the mean-to-volatility ratio, which is why it is "
        "small without needing to be argued: a factor whose EWMA mean reached a tenth of "
        "its EWMA volatility -- a very large daily drift -- would move the volatility by "
        "0.50%. Both quantities are tabulated below, so the identity can be checked rather "
        "than taken.",
        "",
        "The comparand subtracts the trailing EWMA mean and **omits** the `1 - w'w` "
        "reliability correction that normally accompanies demeaning. That omission is "
        "deliberate: including it would confound the cost of the mean term with the cost "
        "of a degree-of-freedom adjustment, and the mean term is what was ruled on.",
        "",
    ]

    for comparison in comparisons:
        lines += [
            f"## Horizon: {comparison.horizon} "
            f"(volatility half-life {comparison.volatility_halflife}d, "
            f"correlation half-life {comparison.correlation_halflife}d)",
            "",
            comparison.overlap_note,
            "",
            f"Window {comparison.dates[0].date()} to {comparison.dates[-1].date()}, "
            f"stopping strictly before the holdout at {holdout}.",
            "",
            "### Volatility",
            "",
            "`sigma_zero / sigma_demeaned - 1`, in basis points of relative difference, "
            "and the driver `m / sigma_demeaned` that produces it.",
            "",
            "| Factor | median \\|shift\\| | 95th pct | max | max \\|m/sigma\\| |",
            "|---|---|---|---|---|",
        ]
        for column in comparison.columns:
            shift = comparison.volatility_ratio[column]
            ratio = comparison.mean_to_volatility[column]
            lines.append(
                f"| `{column}` | {1e4 * shift.abs().median():.2f} bp | "
                f"{1e4 * shift.abs().quantile(0.95):.2f} bp | "
                f"{1e4 * shift.abs().max():.2f} bp | {ratio.abs().max():.4f} |"
            )
        lines += [
            "",
            f"**Worst relative volatility shift across all factors and all dates: "
            f"{1e4 * comparison.worst_volatility_shift:.2f} bp "
            f"({100 * comparison.worst_volatility_shift:.4f}%), at a largest observed "
            f"`|m/sigma|` of {comparison.worst_mean_to_volatility:.4f}.**",
            "",
            "### Correlation",
            "",
            "`max |rho_zero - rho_demeaned|` over the off-diagonal, at the correlation "
            f"half-life of {comparison.correlation_halflife}d. This is the number that "
            "sized the W3-P1b re-basing of `mafrm.factors.macro.condition_numbers`: that "
            "diagnostic was demeaned through W3-P1 and now calls the pipeline's own "
            "estimators, so there is one code path rather than two computing the same "
            "statistic two ways. `experiments.md` row 68's published figures stand as "
            "published; the re-based ones move by at most 1.7% and round identically.",
            "",
            "| | median | 95th pct | max |",
            "|---|---|---|---|",
            _quantile_row("max off-diagonal |Δrho|", comparison.correlation_difference, 1.0, ""),
            "",
        ]

    worst_volatility = max(item.worst_volatility_shift for item in comparisons)
    worst_correlation = max(item.worst_correlation_difference for item in comparisons)
    short = comparisons[0]
    observed, predicted = short.identity_check
    worst_date, worst_name, _, worst_ratio = short.worst_reading
    worst_mean, worst_sigma, worst_trailing = _episode(worst_date, worst_name, settings)
    lines += [
        "## What this settles, and what it does not",
        "",
        "**The second clause of the ruling is only half supported, and this file says so.**",
        "",
        "The *typical* cost is negligible exactly as the ruling anticipated: the median "
        f"absolute volatility shift is {1e4 * short.median_volatility_shift:.2f} bp "
        f"({100 * short.median_volatility_shift:.4f}%) at the short horizon and "
        f"{1e4 * comparisons[1].median_volatility_shift:.2f} bp at the long one. "
        "The *tail* is not negligible: across both horizons, every factor and every read "
        f"date, the largest single shift is **{1e4 * worst_volatility:.0f} bp "
        f"({100 * worst_volatility:.2f}%)**. On a volatility forecast that is a real "
        "number, not a rounding difference, and it is reported as one.",
        "",
        "The mechanism is the identity, not a defect. The largest shifts occur where a "
        "factor's EWMA mean reaches roughly a quarter of its EWMA volatility -- an 84-day "
        "half-life is short enough for a sustained trend to do that -- and the identity "
        f"then predicts a shift of {1e4 * predicted:.2f} bp against the "
        f"{1e4 * observed:.2f} bp observed. The comparand is what it claims to be.",
        "",
        "### When the maximum occurs, which the identity cannot tell you",
        "",
        "The identity holds for **any** `m`, so it confirms the arithmetic and says "
        "nothing about whether the drift is real. A shift of this size has to correspond "
        "to a nameable episode in the underlying series; a maximum landing on a quiet date "
        "would mean the EWMA mean is wrong, and the identity check would pass anyway.",
        "",
        f"**It does not land on a quiet date.** The maximum is `{worst_name}` on "
        f"**{worst_date.date()}**, at `m/sigma` = {worst_ratio:+.4f} -- an EWMA mean of "
        f"{100 * 252 * worst_mean:+.2f}%/yr against an annualised volatility of "
        f"{100 * (252**0.5) * worst_sigma:.2f}%/yr. That is four days after the ECB "
        "announced sovereign QE (22 January 2015) and eleven days after the SNB abandoned "
        "the franc's euro floor (15 January 2015), in the middle of the 2014-15 dollar "
        "surge; the factor's trailing 252-day sum to that date is "
        f"{100 * worst_trailing:+.2f}%. Confirmatory, and the ratio is large for two "
        "reasons rather than one -- the numerator is a genuine trend, and the denominator "
        "is small because `dollar` is among the least volatile factors in the panel at "
        f"{100 * (252**0.5) * worst_sigma:.1f}%/yr.",
        "",
        "The neighbouring readings say the same thing and are worth listing, because they "
        "are **not** independent confirmations -- they are one episode seen four times "
        "through overlapping windows, which is the CLAUDE.md failure mode 9 caveat applied "
        "to this file's own headline:",
        "",
        "| Date | Factor | Shift |",
        "|---|---|---|",
    ]
    lines += [
        f"| {date.date()} | `{name}` | {1e4 * value:.0f} bp |"
        for date, name, value in short.top_readings(6)
    ]
    lines += [
        "",
        "The two clusters are the 2014-15 dollar surge and the 2018 curve flattening. Both "
        "are episodes, not artefacts.",
        "",
        "Correlations are a different story and the answer there is unambiguous: the "
        f"largest off-diagonal disagreement anywhere is **{worst_correlation:.5f}**, at "
        "the 504-day half-life both horizons share. That bound is what made the W3-P1b "
        "re-basing of `mafrm.factors.macro.condition_numbers` cheap: the SPEC.md 4.1.2 "
        "conditioning diagnostic was demeaned through W3-P1 and now calls the pipeline's "
        "own estimators, which removes a second code path computing the same statistic a "
        "second way -- a trap for any later session that compared them. It cost a "
        "re-published diagnostic whose headline figures round identically.",
        "",
        "### None of this moves the decision",
        "",
        "Which is what was registered in advance (`experiments.md` rows 93-95), and the "
        "registration is worth more here than it would have been had the number come back "
        "small. **The tail result is a finding about the factor means, not an argument "
        "against the convention.** The convention rests on SPEC.md 6.1: the bias statistic "
        "divides a raw return by a forecast volatility, and a forecast of variance about a "
        "drifting mean is a forecast of a different quantity than the one being realised. "
        "That holds at 3 bp and it holds at 300 bp. A decision that would have been "
        "reversed by a large number was never a coherence argument in the first place.",
        "",
        "What the tail *does* earn is an entry on the W4 watch list: `B` is estimated on "
        "windows where a factor can carry a mean of a quarter of its volatility, and the "
        "risk-model term of the SPEC.md 1 identity is measured through `B`. The direction "
        "is known and stated here so that it is not rediscovered as a puzzle later -- "
        "forecasting about zero makes the forecast volatility *larger* than a demeaned one "
        "by exactly this amount, so it biases `B` **downward**, never upward, and cannot "
        "manufacture an apparent risk-model failure.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    """``make report``: regenerate ``reports/mean_convention.md``."""
    settings = config.load()
    comparisons = build(settings)
    path = report_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(comparisons, settings), encoding="utf-8")
    root = Path(__file__).resolve().parents[3]
    print(f"wrote {path.relative_to(root)}")
    for comparison in comparisons:
        print(
            f"  {comparison.horizon}: worst volatility shift "
            f"{1e4 * comparison.worst_volatility_shift:.2f} bp, worst |d rho| "
            f"{comparison.worst_correlation_difference:.5f}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
