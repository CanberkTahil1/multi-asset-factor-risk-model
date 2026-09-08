"""The TLT cross-check and the generated validation report. **VALIDATION PATH.**

SPEC.md 3.2, amended 2026-08-26. This module scores the synthetic Treasury series
against TLT and writes ``reports/gsw_validation.md``.

It is deliberately separate from :mod:`mafrm.data.gsw`. That module builds the
production series -- constant-maturity zeros, which are what the factor models
consume -- and must not know that a par-coupon ladder exists. This one imports
both, because validating an ETF requires a comparand with an ETF's cashflow
structure. ``tests/test_ladder.py`` asserts the direction of that dependency.

The gate is the ladder gap, not the daily correlation. A no-roll control moved
the daily correlation by 1e-7, so it cannot falsify the roll-down it was written
to test; the same control moves the ladder gap by 0.43%/yr against a 0.15%/yr
tolerance. Section 3 of the generated report carries the full argument.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from mafrm.config import load
from mafrm.data import cache
from mafrm.data.gsw import (
    constant_maturity_return,
    curve_yields,
    load_curve_unrestricted,
    load_risk_free,
    published_mask,
)
from mafrm.data.ladder import ladder_returns, par_bond_returns, seasoned_ladder_returns

__all__ = ["CrossCheck", "cross_check_tlt", "validation_markdown"]


@dataclass(frozen=True)
class CrossCheck:
    """TLT against the synthetic Treasury series. SPEC.md 3.2, amended.

    ``ladder_*`` fields carry the gate; every ``correlation_*`` field is a
    reported diagnostic with no pass/fail attached to it.
    """

    maturity_years: float
    observations: int
    start: pd.Timestamp
    end: pd.Timestamp
    tlt_mean_annual_pct: float

    # -- the gate: par-coupon ladder, the comparand with TLT's own structure --
    ladder_gap_pct: float
    ladder_beta: float
    ladder_mean_annual_pct: float
    ladder_gap_pct_uniform: float
    ladder_beta_uniform: float
    #: Power evidence: the same gap with the roll-down's sign flipped.
    ladder_gap_pct_roll_flipped: float
    ladder_roll_pct: float
    ladder_correlations: tuple[tuple[str, float], ...]

    # -- diagnostics on the single 20y zero, reported but not gated --
    zero_correlations: tuple[tuple[str, float], ...]
    zero_mean_annual_pct: float
    zero_gap_pct: float
    beta: float
    alpha_annual_pct: float
    synthetic_vol_annual_pct: float
    tlt_vol_annual_pct: float
    zero_roll_pct: float
    zero_gap_pct_no_roll: float
    zero_gap_pct_roll_flipped: float
    zero_alpha_pct_no_roll: float
    zero_alpha_pct_roll_flipped: float
    #: TLT against the Fed's own published yield change: no code of ours.
    correlation_daily_published_yield: float
    #: Alternatives ruled out.
    lead_lag: tuple[tuple[int, float], ...]
    correlation_excluding_ex_dividend: float
    ex_dividend_days: int
    subperiod_correlations: tuple[tuple[str, float], ...]

    @property
    def gate_passed(self) -> bool:
        config = load().model.data.tlt_cross_check
        return (
            abs(self.ladder_gap_pct) <= config.max_abs_gap_pct_per_year
            and config.beta_min <= self.ladder_beta <= config.beta_max
        )


#: Why this module reads past the holdout boundary. The W1-P3 cross-check asks
#: whether a VENDOR price series (TLT) still moves one-for-one with a
#: curve-derived comparand over everything cached -- a question about the data,
#: not about the model -- and its reported n, window and -0.192%/yr gap are
#: pinned by experiments.md and reports/gsw_validation.md. Nothing here reaches a
#: factor, a covariance or a strategy. See mafrm.data.holdout.
_REASON = "data-layer vendor cross-check: TLT vs the GSW ladder over the full cached history"


def _tlt_total_return() -> pd.Series:
    from mafrm.data.prices import total_return

    config = load().model.data.tlt_cross_check
    manifest = cache.Manifest.load()
    ticker = config.ticker.lower()
    prices_frame = cache.read_unrestricted(
        manifest.latest(source="yfinance", name=f"{ticker}_prices"),
        manifest=manifest,
        reason=_REASON,
    )
    actions = cache.read_unrestricted(
        manifest.latest(source="yfinance", name=f"{ticker}_actions"),
        manifest=manifest,
        reason=_REASON,
    )
    return total_return(prices_frame, actions)


def _tlt_ex_dividend_dates() -> pd.Index:
    config = load().model.data.tlt_cross_check
    manifest = cache.Manifest.load()
    actions = cache.read_unrestricted(
        manifest.latest(source="yfinance", name=f"{config.ticker.lower()}_actions"),
        manifest=manifest,
        reason=_REASON,
    )
    return actions.index[actions["Dividends"] > 0]


_HORIZONS: Final[tuple[tuple[str, str], ...]] = (
    ("daily", ""),
    ("weekly", "W-FRI"),
    ("monthly", "ME"),
    ("quarterly", "QE"),
)


def _correlations(left: pd.Series, right: pd.Series) -> tuple[tuple[str, float], ...]:
    """Correlation of two daily return series at four compounding horizons."""
    out = []
    for label, rule in _HORIZONS:
        if not rule:
            out.append((label, float(left.corr(right))))
            continue
        a = (1.0 + left).resample(rule).prod() - 1.0
        b = (1.0 + right).resample(rule).prod() - 1.0
        out.append((label, float(a.corr(b))))
    return tuple(out)


def cross_check_tlt(*, params: pd.DataFrame | None = None, n: float | None = None) -> CrossCheck:
    """Run the SPEC.md 3.2 gate and every diagnostic needed to read its result.

    The gate compares TLT against a par-coupon ladder (VALIDATION ONLY -- see
    :mod:`mafrm.data.ladder`). The 20y zero is carried alongside because it is
    the production series, and because the contrast between the two is what
    showed the original correlation gate to be the wrong instrument.
    """
    config = load().model.data
    if params is None:
        params = load_curve_unrestricted("nominal", reason=_REASON)
    if n is None:
        n = config.tlt_cross_check.maturity_years
    delta = config.roll_down_step
    prefix = config.nominal_curve.yield_prefix
    annual = 252.0

    zero = constant_maturity_return(n, params=params, delta=delta)
    ladder = ladder_returns(params)
    uniform = ladder_returns(
        params, weights=(1.0,) * len(config.validation_ladder.maturities_years)
    )

    y_n = curve_yields(params, n)
    published = published_mask(params, n, yield_prefix=prefix)
    both = published & published.shift(1)
    no_roll = np.expm1((n * y_n.shift(1) - (n - delta) * y_n) / 100.0).where(both)
    published_change = -params[f"{prefix}{math.ceil(n - 1e-12):02d}"].diff()

    joined = pd.concat(
        {
            "zero": zero["total_return"],
            "zero_roll": zero["roll_down"],
            "ladder": ladder["total_return"],
            "ladder_roll": ladder["roll_down"],
            "uniform": uniform["total_return"],
            "tlt": _tlt_total_return(),
            "no_roll": no_roll,
            "published_change": published_change,
        },
        axis=1,
        # Stated explicitly: sort=True is today's default for an
        # all-DatetimeIndex concat, and pandas 3 deprecates the implicit
        # behaviour ahead of flipping it to sort=False. An unsorted union here
        # would reorder every series feeding the W1-P3 cross-check.
        sort=True,
    ).loc[pd.Timestamp(config.tlt_cross_check.start) :]
    joined = joined.dropna()

    def annualise(series: pd.Series) -> float:
        return float(series.mean() * annual * 100.0)

    def gap(series: pd.Series) -> float:
        return float((joined["tlt"].mean() - series.mean()) * annual * 100.0)

    def slope(series: pd.Series) -> float:
        return float(np.polyfit(series, joined["tlt"], 1)[0])

    def intercept(series: pd.Series) -> float:
        return float(np.polyfit(series, joined["tlt"], 1)[1] * annual * 100.0)

    ladder_flipped = np.expm1(np.log1p(joined["ladder"]) - 2.0 * joined["ladder_roll"])
    zero_flipped = np.expm1(np.log1p(joined["zero"]) - 2.0 * joined["zero_roll"])

    ex_dividend = joined.index.isin(_tlt_ex_dividend_dates())
    clean = joined.loc[~ex_dividend]

    subperiods = []
    years = pd.DatetimeIndex(joined.index).year
    for first in range(int(years[0]), int(years[-1]) + 1, 4):
        window = joined[(years >= first) & (years <= first + 3)]
        if len(window) > 100:
            last = min(first + 3, int(years[-1]))
            subperiods.append((f"{first}-{last}", float(window["zero"].corr(window["tlt"]))))

    beta, alpha = np.polyfit(joined["zero"], joined["tlt"], 1)
    return CrossCheck(
        maturity_years=n,
        observations=len(joined),
        start=joined.index[0],
        end=joined.index[-1],
        tlt_mean_annual_pct=annualise(joined["tlt"]),
        ladder_gap_pct=gap(joined["ladder"]),
        ladder_beta=slope(joined["ladder"]),
        ladder_mean_annual_pct=annualise(joined["ladder"]),
        ladder_gap_pct_uniform=gap(joined["uniform"]),
        ladder_beta_uniform=slope(joined["uniform"]),
        ladder_gap_pct_roll_flipped=gap(ladder_flipped),
        ladder_roll_pct=annualise(joined["ladder_roll"]),
        ladder_correlations=_correlations(joined["ladder"], joined["tlt"]),
        zero_correlations=_correlations(joined["zero"], joined["tlt"]),
        zero_mean_annual_pct=annualise(joined["zero"]),
        zero_gap_pct=gap(joined["zero"]),
        beta=float(beta),
        alpha_annual_pct=float(alpha * annual * 100.0),
        synthetic_vol_annual_pct=float(joined["zero"].std() * math.sqrt(annual) * 100.0),
        tlt_vol_annual_pct=float(joined["tlt"].std() * math.sqrt(annual) * 100.0),
        zero_roll_pct=annualise(joined["zero_roll"]),
        zero_gap_pct_no_roll=gap(joined["no_roll"]),
        zero_gap_pct_roll_flipped=gap(zero_flipped),
        zero_alpha_pct_no_roll=intercept(joined["no_roll"]),
        zero_alpha_pct_roll_flipped=intercept(zero_flipped),
        correlation_daily_published_yield=float(joined["published_change"].corr(joined["tlt"])),
        lead_lag=tuple(
            (lag, float(joined["tlt"].corr(joined["zero"].shift(lag)))) for lag in (-2, -1, 0, 1, 2)
        ),
        correlation_excluding_ex_dividend=float(clean["zero"].corr(clean["tlt"])),
        ex_dividend_days=int(ex_dividend.sum()),
        subperiod_correlations=tuple(subperiods),
    )


def seasoning_sweep(params: pd.DataFrame, benchmark: pd.Series) -> pd.DataFrame:
    """Bound the par-bond idealisation: fresh par bonds against seasoned ones.

    A par ladder strikes a fresh bond at par every day; a real index holds bonds
    issued years ago carrying whatever coupon was par then, trading at a premium
    or discount today. Same maturity, same curve, different coupon -- so a
    different present-value weighting across cashflows and a different effective
    duration. This measures the difference, with no reference to any observed
    residual.
    """
    config = load().model.data
    par = ladder_returns(params)["total_return"]
    rows = []
    for seasoning in config.tlt_cross_check.seasoning_sweep_years:
        seasoned = seasoned_ladder_returns(params, seasoning)
        joined = pd.concat({"par": par, "seasoned": seasoned, "bench": benchmark}, axis=1).loc[
            pd.Timestamp(config.tlt_cross_check.start) :
        ]
        joined = joined.dropna()
        rows.append(
            {
                "seasoning_years": f"{seasoning:g}",
                "seasoned_return_pct": float(joined["seasoned"].mean() * 252 * 100),
                "par_minus_seasoned_pct": float(
                    (joined["par"].mean() - joined["seasoned"].mean()) * 252 * 100
                ),
                "benchmark_minus_seasoned_pct": float(
                    (joined["bench"].mean() - joined["seasoned"].mean()) * 252 * 100
                ),
            }
        )
    return pd.DataFrame(rows).set_index("seasoning_years")


#: Coupon-date phases swept to bound the "next coupon is a full period away"
#: assumption. A real bond's next coupon is uniformly distributed in (0, 1/f].
_PHASES: Final[tuple[float, ...]] = (0.5, 0.4, 0.3, 0.25, 0.2, 0.1, 0.05)


def coupon_phase_bound(params: pd.DataFrame) -> float:
    """Mean cost, in %/yr, of assuming the next coupon is a full period away.

    Averaged over the sweep of phases, which approximates a portfolio whose
    coupon dates are spread uniformly through the period -- which is what any
    real index holds.
    """
    config = load().model.data
    ladder = config.validation_ladder
    start = pd.Timestamp(config.tlt_cross_check.start)

    def at_phase(phase: float | None) -> pd.Series:
        blended: pd.Series | None = None
        for weight, maturity in zip(
            ladder.normalised_weights, ladder.maturities_years, strict=True
        ):
            leg = (
                par_bond_returns(
                    params,
                    maturity,
                    frequency=ladder.coupon_frequency,
                    delta=config.roll_down_step,
                    first_coupon=phase,
                )["total_return"]
                * weight
            )
            blended = leg if blended is None else blended.add(leg, fill_value=0.0)
        assert blended is not None
        return blended

    base = at_phase(None)
    deltas = []
    for phase in _PHASES:
        joined = pd.concat({"a": at_phase(phase), "b": base}, axis=1).loc[start:].dropna()
        deltas.append(float((joined["a"].mean() - joined["b"].mean()) * 252 * 100))
    return float(np.mean(deltas))


def _series_inventory(params: pd.DataFrame) -> pd.DataFrame:
    """Per-maturity summary of what the constructed series actually contains."""
    config = load().model.data
    try:
        risk_free: pd.Series | None = load_risk_free()
    except cache.CacheError:
        risk_free = None
    rows = []
    for n in config.nominal_curve.maturities_years:
        frame = constant_maturity_return(
            n, params=params, delta=config.roll_down_step, risk_free_pct=risk_free
        )
        total = frame["total_return"].dropna()
        rows.append(
            {
                "maturity": f"{n:g}y",
                "obs": len(total),
                "start": total.index[0].date().isoformat(),
                "end": total.index[-1].date().isoformat(),
                "ann_return_pct": float(total.mean() * 252 * 100),
                "ann_vol_pct": float(total.std() * math.sqrt(252) * 100),
                "vol_per_year_of_maturity": float(total.std() * math.sqrt(252) * 100 / n),
                "roll_bp_per_day": float(frame["roll_down"].mean() * 1e4),
                "roll_pct_per_year": float(frame["roll_down"].mean() * 252 * 100),
                "carry_pct_per_year": float(frame["carry"].mean() * 252 * 100),
                "excess_obs": int(frame["excess_return"].notna().sum()),
            }
        )
    return pd.DataFrame(rows).set_index("maturity")


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a frame as a GitHub markdown table without pulling in tabulate."""
    header = [str(frame.index.name or "")] + [str(c) for c in frame.columns]
    rows = [
        [str(index)]
        + [f"{value:,.3f}" if isinstance(value, float) else str(value) for value in row]
        for index, row in zip(frame.index, frame.to_numpy(), strict=True)
    ]
    widths = [max(len(cell) for cell in column) for column in zip(header, *rows, strict=True)]

    def line(cells: list[str]) -> str:
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths, strict=True)) + " |"

    return "\n".join(
        [
            line(header),
            "|" + "|".join("-" * (w + 2) for w in widths) + "|",
            *(line(r) for r in rows),
        ]
    )


def _corr_row(label: str, pairs: tuple[tuple[str, float], ...]) -> str:
    return f"| {label} | " + " | ".join(f"{value:.4f}" for _, value in pairs) + " |"


def validation_markdown() -> str:
    """Regenerate ``reports/gsw_validation.md`` from the current cache."""
    config = load().model.data
    params = load_curve_unrestricted("nominal", reason=_REASON)
    check = cross_check_tlt(params=params)
    inventory = _series_inventory(params)
    gate = config.tlt_cross_check
    manifest = cache.Manifest.load()
    entry = manifest.latest(source="fed", name="nominal_curve")
    tenor = f"{config.nominal_curve.yield_prefix}{math.ceil(check.maturity_years):02d}"
    duration = check.beta * check.maturity_years
    ladder_top = max(config.validation_ladder.maturities_years)
    ladder_bottom = min(config.validation_ladder.maturities_years)
    longest = max(config.nominal_curve.maturities_years)
    longest_roll = constant_maturity_return(longest, params=params, delta=config.roll_down_step)[
        "roll_down"
    ].mean()
    verdict = "PASS" if check.gate_passed else "FAIL"
    gap_ok = "pass" if abs(check.ladder_gap_pct) <= gate.max_abs_gap_pct_per_year else "FAIL"
    beta_ok = "pass" if gate.beta_min <= check.ladder_beta <= gate.beta_max else "FAIL"
    band = f"[{gate.beta_min:.2f}, {gate.beta_max:.2f}]"
    tolerance = f"{gate.max_abs_gap_pct_per_year:.2f}%/yr"
    ladder_label = f"{ladder_bottom:g}-{ladder_top:g}y"
    moved = abs(check.ladder_gap_pct_roll_flipped - check.ladder_gap_pct)
    power = moved / gate.max_abs_gap_pct_per_year
    inside = abs(check.ladder_gap_pct) <= gate.max_abs_gap_pct_per_year
    excess = abs(check.ladder_gap_pct) - gate.max_abs_gap_pct_per_year
    tolerance_verdict = (
        f"The observed {check.ladder_gap_pct:+.3f}%/yr is inside that, which says the fund "
        "tracks its own asset class and our construction tracks the fund."
        if inside
        else (
            f"The observed {check.ladder_gap_pct:+.3f}%/yr is **outside it by {excess:.3f}%/yr**. "
            "The sign is right -- the fee-free synthetic beats the fund -- and the magnitude is "
            "fee plus something, not fee times something. Section 6 says what that something is "
            "and why the excess is not evidence of a roll error."
        )
    )
    daily_ladder = dict(check.ladder_correlations)["daily"]
    weekly_ladder = dict(check.ladder_correlations)["weekly"]
    daily_zero = dict(check.zero_correlations)["daily"]
    tracking = gate.published_tracking_difference_pct_per_year
    fee = gate.expense_ratio_pct_per_year
    ladder_minus_index = -check.ladder_gap_pct + tracking
    sweep = seasoning_sweep(params, _tlt_total_return())
    phase_cost = coupon_phase_bound(params)
    realistic = sweep.loc[["3", "4", "5", "6", "7"], "par_minus_seasoned_pct"]
    idealisation = float(realistic.mean())
    idealisation_lo = float(sweep["par_minus_seasoned_pct"].min())
    idealisation_hi = float(sweep["par_minus_seasoned_pct"].max())
    accrual_years = check.observations / 252.0
    span_years = (check.end - check.start).days / 365.25
    carry_shortfall = (span_years - accrual_years) / span_years * 3.75
    # Each correction is signed as its EFFECT ON THE GAP (ladder less index).
    # Negative closes the gap, positive widens it.
    #   seasoning: a seasoned replica earns `-idealisation` more -> widens
    #   phase:     realistic coupon dates earn `phase_cost` less  -> closes
    #   accrual:   full calendar carry earns `carry_shortfall` more -> widens
    effect_seasoning = -idealisation
    effect_phase = phase_cost
    effect_accrual = carry_shortfall
    net_effect = effect_seasoning + effect_phase + effect_accrual
    unexplained = ladder_minus_index + net_effect
    gate_rows = "\n".join(
        [
            f"| TLT less ladder, {ladder_label} 3:1 tilt "
            f"| **{check.ladder_gap_pct:+.3f}%/yr** | within {tolerance} | {gap_ok} |",
            f"| Beta of TLT on the ladder | **{check.ladder_beta:.4f}** | {band} | {beta_ok} |",
            f"| Same, uniform weights (sensitivity) "
            f"| {check.ladder_gap_pct_uniform:+.3f}%/yr | -- "
            f"| beta {check.ladder_beta_uniform:.4f} |",
        ]
    )
    lead_lag = ", ".join(f"{lag:+d}: {value:+.4f}" for lag, value in check.lead_lag if lag != 0)
    subperiods = ", ".join(f"{label} {value:.4f}" for label, value in check.subperiod_correlations)
    corr_header = "| series | " + " | ".join(label for label, _ in check.zero_correlations) + " |"
    corr_rule = "|---" * (len(check.zero_correlations) + 1) + "|"

    return f"""# GSW constant-maturity Treasury series -- validation

Generated by `python -m mafrm.data.gsw_report`. Do not edit by hand.

Curve pull: `{entry.path}`, downloaded {entry.downloaded_at}, {entry.rows:,} rows,
covering {entry.first_date} to {entry.last_date}. File SHA-256 `{entry.sha256[:16]}...`,
upstream bytes `{(entry.source_sha256 or "n/a")[:16]}...`.

## 1. The acceptance gate (SPEC.md 3.2, amended 2026-08-26)

The gate is the mean-return gap against a par-coupon ladder, not a daily
correlation. Section 3 records why the original gate was replaced.

> **Primary gate:** `|TLT - coupon ladder| <= {tolerance}` with `beta` in {band}.

| Quantity | Value | Gate | Verdict |
|---|---|---|---|
{gate_rows}

**{verdict}.** {check.observations:,} overlapping days, {check.start.date()} to
{check.end.date()}. TLT returned {check.tlt_mean_annual_pct:+.3f}%/yr against the
ladder's {check.ladder_mean_annual_pct:+.3f}%/yr.

**Why {tolerance} is the right tolerance.** It is TLT's expense ratio. A synthetic
built from the Fed's curve pays no fee, so a correctly specified replica should
beat the fund by about its fee and by nothing else. {tolerance_verdict}

**The gate has power.** Flipping the sign of the ladder's roll-down moves the gap
from {check.ladder_gap_pct:+.3f}%/yr to {check.ladder_gap_pct_roll_flipped:+.3f}%/yr --
a movement of **{moved:.3f}%/yr, about {power:.1f}x the tolerance**, and a change of
sign. Compare the 1e-7 the superseded correlation gate moved by. The ladder's own
roll is {check.ladder_roll_pct:+.3f}%/yr.

## 2. Reported diagnostics -- not gated

Correlation of daily total returns against TLT, compounded to each horizon.

{corr_header}
{corr_rule}
{_corr_row(f"{check.maturity_years:g}y zero (production series)", check.zero_correlations)}
{_corr_row("coupon ladder " + ladder_label + " (validation only)", check.ladder_correlations)}

| Statistic | Value |
|---|---|
| Mean difference, TLT less {check.maturity_years:g}y zero | {check.zero_gap_pct:+.3f}%/yr |
| Beta of TLT on the {check.maturity_years:g}y zero | {check.beta:.4f} (implied duration {duration:.1f}y) |
| Duration-neutral alpha on the zero | {check.alpha_annual_pct:+.3f}%/yr |
| Annualised vol, {check.maturity_years:g}y zero | {check.synthetic_vol_annual_pct:.2f}% |
| Annualised vol, TLT | {check.tlt_vol_annual_pct:.2f}% |

**Mechanism for the correlation shortfall.** The Fed fits its curve to end-of-day
bond quotes, about 3:30pm ET; TLT prints at the 4:00pm equity close. That half
hour is in the ETF and in nothing constructible from the curve, and it is
compounded by TLT's close being a traded price carrying a premium or discount to
NAV. Both are daily observation noise, which is why correlation rises with
horizon for both comparands.

**Alternatives ruled out.**

- *Date misalignment:* cross-correlation of TLT against the lagged zero is
  {lead_lag} -- essentially zero at every non-contemporaneous lag, so the
  shortfall is a within-day effect, not an off-by-one on dates.
- *Dividend handling:* excluding the {check.ex_dividend_days} ex-dividend days
  leaves the daily correlation at {check.correlation_excluding_ex_dividend:.4f}.
- *A period effect:* four-year sub-periods give {subperiods}.

**The ceiling on single-point constructions.** Correlating TLT against the first
difference of the Fed's own published `{tenor}` column -- no Svensson evaluation,
no roll, no return construction -- gives
**{check.correlation_daily_published_yield:.4f}**, within 0.001 of what the {check.maturity_years:g}y
zero achieves. Nothing built at the {check.maturity_years:g}y point can beat it.

> **CORRECTED 2026-08-26.** An earlier version of this report read that figure as
> a ceiling on *every* series derivable from the Fed's file: "nothing downstream
> of the Fed's file can beat it", and concluded that the 0.98 gate "cannot be met
> by any series derived from the Fed's fitted curve". ~~Both statements~~ are
> **wrong as written**. The bound is specific to constructions read off a single
> point of the curve at the {check.maturity_years:g}y tenor, at daily frequency.
> The coupon ladder is built from the same file and does better --
> {daily_ladder:.4f} daily and
> {weekly_ladder:.4f} weekly, the latter clearing 0.98
> -- because it matches TLT's cashflow profile rather than a single key rate. The
> original claim generalised from one comparand to all of them.

## 3. Why the gate was amended

The original SPEC.md 3.2 gate was "daily correlation against TLT > 0.98", with
the instruction that failing it means the roll-down is wrong. It was replaced
because it cannot detect the roll-down at all.

**The no-roll control.** Deleting the roll entirely -- reading day `t+1` at `n`
instead of `n - Delta` -- moves the daily correlation by **1e-7**, from
{daily_zero:.7f} to essentially the same number. The
roll is a slow drift in the mean and contributes almost nothing to daily
covariance, so a correlation gate is insensitive to the one thing it was written
to falsify.

**The mean is not insensitive.** On the {check.maturity_years:g}y zero the same three
variants give:

| Roll treatment | TLT less synthetic | Alpha (TLT intercept) | Economic reading |
|---|---|---|---|
| Included (as shipped) | {check.zero_gap_pct:+.3f}%/yr | {check.alpha_annual_pct:+.3f}%/yr | TLT underperforms a fee-free replica -- possible |
| Removed | {check.zero_gap_pct_no_roll:+.3f}%/yr | {check.zero_alpha_pct_no_roll:+.3f}%/yr | TLT beats a fee-free replica -- impossible |
| Sign flipped | {check.zero_gap_pct_roll_flipped:+.3f}%/yr | {check.zero_alpha_pct_roll_flipped:+.3f}%/yr | TLT beats a fee-free replica by more -- impossible |

Only the shipped treatment produces an economically possible result. That is the
argument the amended gate encodes: score the mean, against a comparand whose
structure matches the fund's, with a tolerance set by the fund's fee.

**Why the ladder and not the zero.** A {check.maturity_years:g}y zero has all its
duration at one point and earns {check.zero_roll_pct:+.3f}%/yr of roll; TLT holds
coupon bonds spanning {ladder_label} with a regression-implied
duration of {duration:.1f}y and earns {check.ladder_roll_pct:+.3f}%/yr. Scoring one
against the other conflates a roll differential, a cash-drag artefact of
beta-scaling, and a key-rate mismatch. **The ladder is a validation comparand
only** -- see `src/mafrm/data/ladder.py`. The production factor inputs remain the
constant-maturity zeros, because a blended ladder's loading on the level, slope
and curvature of the curve is a mixture rather than a clean key rate, which is
the property the macro factor model needs its inputs not to have.

## 4. Named findings in the Fed's files

None of the three is documented in SPEC.md; all were found by inspection and all
three would have passed silently.

**Finding 1 -- the `TAU2 = -999.99` sentinel destroys the pre-1980 sample.**
Before 1980-01-02 the Fed fitted Nelson-Siegel, not Svensson: `BETA3` is 0 and
`TAU2` carries the sentinel `-999.99`. Evaluating the fourth term at a negative
tau gives `exp(-n / -999.99)`, which overflows to `inf`, and `0 * inf` is `nan`.
Left unhandled this silently deletes **4,620 rows -- 27% of the sample**, and the
lost block is precisely the Volcker era, the highest-rate-volatility regime in
the record and the single most informative window for a term-structure risk
model. Handled by dropping the fourth term wherever `TAU2` is missing or
non-positive.

**Finding 2 -- blank weekday rows inflate carry and roll by 3.5%.** Both curve
files carry a row for every weekday including market holidays, on which every
field is `NA`. Retained, they yield 261 observations a year against a `Delta` of
1/252, so a constant-maturity series accrues **3.5% more carry and roll than
time actually passed** -- roughly 18bp a year at a 5% yield, in the direction
that flatters the bond sleeve. Dropped at parse time, which leaves ~250.5
observations a year and makes `Delta = 1/252` honest.

**Finding 3 -- one holiday the parameter test does not catch.** On 2008-03-21,
Good Friday, the Fed wrote a degenerate fit (`BETA0 = 8.3e-14`) but published no
yield at any tenor. Finding 2's filter drops rows whose *parameters* are all
blank; this row has parameters, so it survived, and anything pricing across it
carried a four-day weekend as a single `Delta` step. One row in 65 years, and it
moved the acceptance gap by 0.08%/yr -- **more than half the tolerance, decided by
a single day**. Rows with no published yield at any tenor are now dropped at parse
time alongside the blank-parameter rows, and the ladder masks each maturity to its
published dates exactly as the zeros do.

## 5. Series inventory

Built from the same parameters, masked to the dates the Fed publishes each tenor.
`config/universe.yaml` now records these per-tenor starts: 1961-06-14 for the 2y
and 5y, 1971-08-16 for the 10y, 1985-11-25 for the 30y. The Fed does not publish
a fitted yield beyond the longest bond outstanding, and the parameters would
happily extrapolate there.

{_markdown_table(inventory.round(3))}

Annualised volatility per year of maturity is about 1% at every point, which is
the duration identity a zero-coupon series must satisfy: a 30y zero moves 30
times the yield change. The +18.13% of 1987-12-01 is a 55bp rally, not a defect.

The {longest:g}y roll-down is **negative** ({float(longest_roll) * 1e4:+.3f} bp/day,
{float(longest_roll) * 252 * 100:+.2f}%/yr). That is not an error: the Svensson form
has a flat asymptote and the fitted curve is on average very slightly inverted
past about 25 years, so a constant-maturity {longest:g}y zero rolls up the curve
rather than down. The 2y, 5y and 10y points all roll down positively. Anyone who
assumes roll is positive by construction, or who checks only the long point, will
read this as a bug.

## 6. The unexplained residual

The gate fails at {check.ladder_gap_pct:+.3f}%/yr against a {tolerance} tolerance.
This section is the accounting for why, and it does not end in a reconciliation.

**The quantity to explain is not the fund's.** iShares publishes TLT's tracking
difference against its own benchmark, the ICE US Treasury 20+ Year Bond Index, at
{tracking:+.2f}%/yr since inception ({-0.13:+.2f}% 1yr, {-0.09:+.2f}% 5yr, {-0.10:+.2f}% 10yr)
against a {fee:.2f}% expense ratio. **The fund beats its fee**, via securities
lending, so there is no ETF implementation drag left to absorb anything. Chaining:

| Leg | Value | Source |
|---|---|---|
| ladder less TLT | {-check.ladder_gap_pct:+.3f}%/yr | measured here |
| TLT less index | {tracking:+.3f}%/yr | iShares fact sheet |
| **ladder less index** | **{ladder_minus_index:+.3f}%/yr** | **the quantity to explain** |

The synthetic ladder outperforms the real bond index by about {ladder_minus_index:.2f}%/yr.
That is ours to account for.

**Bound 1 -- the par idealisation, measured independently.** A par ladder strikes
a fresh bond at par every day; the index holds bonds issued years ago carrying
whatever coupon was par then. Same maturity, same curve, different coupon, so a
different present-value weighting and a different effective duration. Seasoning
is capped per leg at `30 - m`: a bond with `m` years left cannot have been issued
more than 30 years ago.

{_markdown_table(sweep.round(3))}

**The prediction registered before this ran was +0.06%/yr, rising monotonically in
seasoning, range +0.03% to +0.15%. It is refuted.** The effect is non-monotone,
spans {idealisation_lo:+.3f}% to {idealisation_hi:+.3f}%/yr across the sweep, and at the
seasonings the 20+ bucket plausibly averages (3-7 years) it is
**{idealisation:+.3f}%/yr** -- an order of magnitude too small, and not reliably the
right sign. Four-year sub-periods run from -0.26% to +0.13%/yr, which says this is
a noisy regime-dependent quantity that averages to roughly nothing, not a
systematic drag. The par-versus-seasoned idealisation does not explain the gap.

**Bound 2 -- the coupon-date phase.** The ladder assumes the next coupon is a full
period away; a real bond's next coupon is uniformly distributed inside the period.
Sweeping the phase and averaging gives **{phase_cost:+.3f}%/yr** -- the right sign,
and about a quarter of what is needed.

**Bound 3 -- calendar time against `Delta = 1/252`.** The ladder accrues
{accrual_years:.2f} years of carry over a {span_years:.2f}-year window, because there
are about 250 observations a year rather than 252. At a ~3.75% carry that
*understates* the ladder by roughly {carry_shortfall:.3f}%/yr, so correcting it
widens the gap rather than closing it. Reported because it is real and because
it points the wrong way.

Each correction below is signed as its **effect on the gap**: negative closes it,
positive widens it.

| Component | Effect on the gap | Independent source |
|---|---|---|
| Observed, ladder less index | {ladder_minus_index:+.3f}%/yr | measured + iShares fact sheet |
| Par bonds to seasoned bonds | {effect_seasoning:+.3f}%/yr | measured, Bound 1 |
| Full period to realistic coupon phase | {effect_phase:+.3f}%/yr | measured, Bound 2 |
| `Delta = 1/252` to calendar accrual | {effect_accrual:+.3f}%/yr | measured, Bound 3 |
| Net of the three corrections | {net_effect:+.3f}%/yr | |
| **UNEXPLAINED, after corrections** | **{unexplained:+.3f}%/yr** | |

For reference, the two published inputs: TLT's expense ratio is {fee:.2f}%/yr and its
tracking difference against the index is {tracking:+.2f}%/yr, both from the iShares
fact sheet. Neither is a free parameter here.

**The corrections make the puzzle slightly worse, not better.** Only the coupon
phase closes any of the gap, and the other two widen it by more than it closes.

**Verdict: unresolved, and the tolerance is not widened.** The pre-registered
decision rule was that a measured idealisation near +0.10%/yr would justify
setting the tolerance to fee plus idealisation, and that anything materially away
from it would not. It landed at {idealisation:+.3f}%/yr. Widening the tolerance now
would be fitting it to the residual it is supposed to test, which is the one move
`experiments.md` exists to prevent.

**What has been ruled out.** The roll-down, three ways: its sign and magnitude
survive an impossibility argument (Section 3), and the ladder's own roll is
{check.ladder_roll_pct:+.3f}%/yr against the coupon-roll term structure. Duration:
beta is {check.ladder_beta:.4f}, inside the gate band. ETF implementation drag:
the fund beats its fee. The weighting scheme: uniform against the 3:1 tilt moves
the gap by {abs(check.ladder_gap_pct_uniform - check.ladder_gap_pct):.3f}%/yr. The par
idealisation and the coupon phase: bounded above.

**What is left, and its sign is consistent.** The leading suspect is the curve
itself, and it points the right way, which is worth stating rather than leaving as
a shrug.

GSW fit their zero curve to a filtered set of Treasuries, deliberately excluding
on-the-run and first off-the-run issues **because those trade rich on liquidity**
and would distort the fit. So the fitted curve sits at *off-the-run* yields. The
index holds the actual outstanding stock, on-the-runs included -- bonds trading at
richer prices, and therefore lower yields, than the curve the ladder is priced
off. A ladder discounted at off-the-run yields earns more carry than the stock the
index actually owns. **Positive, and order ten basis points a year** -- the sign
and the magnitude the residual has.

That is a mechanism with a direction, not a placeholder. It is still not a
measurement: confirming it needs bond-level CUSIP prices to compare the fitted
curve against realised trades issue by issue, which this project does not have and
is not going to acquire. A secondary contributor points the same way unquantified:
the index is market-value weighted across the real stock where this ladder is
eleven equally-spaced maturities.

**Unexplained, suspect named, sign consistent, untestable within scope** is where
this stops. That is a stronger statement than "unexplained", and it is as far as
the available data carries it.

**This is an open question, stated as one.** It does not block the data layer: the
production series are the constant-maturity zeros, the roll is verified by
construction and by the impossibility argument, and the residual is a property of
the *validation comparand*, not of the series the factor models consume.

## 7. Risk-free leg

The selected risk-free leg is `{config.risk_free.selected}` -- Ken French's daily
one-month Treasury bill, daily from {config.risk_free.ken_french_daily_rf.history_start} --
chosen because it covers the whole 1961- sample with **no splice**. A spliced short rate
would put a construction seam inside the excess-return series at exactly the point where
the long history is the differentiator (SPEC.md 3.1).

Its loader landed in W1-P4, so the excess-return column now begins with the curve rather
than with {config.risk_free.interim_fred.fred_series}'s
{config.risk_free.interim_fred.history_start} start -- roughly 40 additional years of
excess returns on the government sleeve.

**Units.** Ken French publishes the rate as percent per *trading day*;
{config.risk_free.interim_fred.fred_series}, which it replaces, is percent per *annum*.
The two differ by a factor of about {config.trading_days_per_year} and both are
small plausible numbers, so the substitution would be silent.
`mafrm.data.french.as_annual_percent` converts against `data.trading_days_per_year`,
chosen so the daily accrual `rf_annual * Delta / 100` reproduces exactly the fraction
Ken French published for that day. {config.risk_free.interim_fred.fred_series} is retained
as the comparand for that conversion: the two legs are compared over their overlap in
`tests/test_data_contracts.py`, where a units error would show as a factor-of-252 gap
rather than as the small real spread between an overnight bill return and a one-month
constant-maturity yield.
"""


def _main(argv: Sequence[str]) -> int:
    if list(argv) not in ([], ["report"]):
        print("usage: python -m mafrm.data.gsw_report", file=sys.stderr)
        return 2
    target = Path(__file__).resolve().parents[3] / "reports" / "gsw_validation.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(validation_markdown(), encoding="utf-8")
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
