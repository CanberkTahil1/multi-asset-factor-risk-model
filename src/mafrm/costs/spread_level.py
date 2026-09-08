"""The spread LEVEL, and how SPEC.md 7.1's ``a_i`` is built from it. SPEC.md 3.4.1.

EDGE supplies time variation and cannot supply level for this sleeve. W1-P5c
measured a per-window noise of ~23bp at 63 daily bars irrespective of the true
spread -- more than every calm-period spread in the universe -- and showed the
estimator is unbiased (it recovers 1/5/20/50bp), so this is variance, not bias,
and no averaging within the window structure fixes it. SPEC.md 3.4.1 records
the consequence as DESIGN rather than quarantine: there is no constant, and no
per-ticker constant, that turns a series with signal-to-noise 0.04 into a level,
because scaling multiplies signal and noise alike. Nothing in this module takes
an EDGE level and the question is not reopened here.

The level is therefore the issuer-disclosed 30-day median bid/ask spread,
committed as a dated static file (``costs.spread_level.file``) whose rows carry
the URL they were read from and the date they were read on. A row without a
value is a placeholder and :meth:`SpreadLevels.require` refuses it by name. No
default exists, and ``costs.flat_spread_assumption_bps`` is not one: SPEC.md
3.4.1 forbids it as a level source.

**The combination rule** (operator ruling, 2026-09-03, W5-P1):

    s_i(t) = s_issuer,i + max(0, s_EDGE,i(t) - baseline_i)
    a_i(t) = s_i(t) / 2 + commission

Level from the issuer, shape from EDGE as an ADDITIVE widening in basis points,
no negatives, agreement at baseline. A multiplicative rule was ruled and
withdrawn the same day: it would have divided by the calm-period EDGE level,
which for the liquid names is a clipped estimate that is often exactly zero --
the division-by-zero class SPEC.md 7.4 names -- and which
``costs.spread_vs_volatility.minimum_baseline_for_ratio_bps`` already records
as not a statement about spreads. SPEC.md 3.4.1 consequence 1 says the same:
a level offset cancels in a widening, which is why the crisis statistic is in
bp rather than a multiple. The additive form keeps every principle of the
ruling and drops the one thing that was wrong.

``baseline_i`` is the in-sample median of the CLIPPED production series. A
median is robust to the crisis tail, so it is the calm level with no date
selection and no judgement about which periods are calm -- and it reads
nothing at or after the holdout boundary, because the series it is taken from
is truncated there by :func:`mafrm.data.cache.read`.

**The direction of the error that matters.** An over-stated spread widens the
L1 no-trade region, suppresses turnover, lowers measured cost drag and flatters
the cost term of the SPEC.md 1 identity. That failure would not look like an
error. It is the fourth instance of the flattering-direction pattern in this
project, and it is why the level is refused rather than defaulted.

**Anachronism, stated.** The issuer figure is a 2026 disclosure applied to
2009-2024. It is the same anachronism SPEC.md 3.4.1 accepted when it made
issuer medians the permanent level source, because no historical disclosed
series exists. Every cost figure carries it as a caveat.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

__all__ = [
    "IssuerLevel",
    "SecondSource",
    "SpreadBand",
    "SpreadLevelError",
    "SpreadLevels",
    "SpreadSeries",
    "calm_baseline",
    "half_spread",
    "load_spread_levels",
    "spread_series",
    "widening",
]

#: One basis point as a proportion. A unit, not a tunable.
_BPS = 1e-4


class SpreadLevelError(ValueError):
    """The level file cannot supply a spread for an asset, or is malformed."""


@dataclass(frozen=True)
class IssuerLevel:
    """One issuer-disclosed 30-day median bid/ask spread, with its provenance."""

    asset: str
    ticker: str
    source_url: str
    disclosure_date: date
    #: FULL bid/ask spread, basis points of price. The code halves it.
    median_spread_bps: float

    @property
    def levels_bps(self) -> tuple[float, ...]:
        return (self.median_spread_bps,)


@dataclass(frozen=True)
class SpreadBand:
    """A sourced [low, high] band for an asset with no disclosed median.

    Swept and reported as a range, the treatment the impact coefficient gets.
    Bounds must be sourced like a point value; they are never invented.
    """

    asset: str
    ticker: str
    source_url: str
    disclosure_date: date
    low_bps: float
    high_bps: float
    #: The figure as the issuer displayed it, when the band is a rounding
    #: interval of that figure rather than a sourced range; ``None`` otherwise.
    displayed_bps: float | None = None
    #: A second published figure that does not enter the band, with its URL
    #: and why it is recorded -- SPY's ~0.6bp from SSGA's capital-markets note.
    second_source: SecondSource | None = None

    @property
    def levels_bps(self) -> tuple[float, ...]:
        return (self.low_bps, self.high_bps)


@dataclass(frozen=True)
class SecondSource:
    """A second published figure, recorded beside a level and never used as one."""

    url: str
    value_bps: float
    note: str


@dataclass(frozen=True)
class SpreadLevels:
    """Every row of the level file, placeholders included, keyed by asset id."""

    path: Path
    entries: Mapping[str, IssuerLevel | SpreadBand | None]

    @property
    def placeholders(self) -> tuple[str, ...]:
        return tuple(asset for asset, entry in self.entries.items() if entry is None)

    @property
    def supplied(self) -> tuple[str, ...]:
        return tuple(asset for asset, entry in self.entries.items() if entry is not None)

    def require(self, asset: str) -> IssuerLevel | SpreadBand:
        """The level for ``asset``, or a refusal that names it.

        This is the only way a level leaves this module, and there is no
        default. SPEC.md 3.4.1 forbids the EDGE level and the flat assumption
        as sources; an unsupplied row is therefore a stop, not a fallback.
        """
        if asset not in self.entries:
            raise SpreadLevelError(f"{self.path}: no row for {asset!r}")
        entry = self.entries[asset]
        if entry is None:
            raise SpreadLevelError(
                f"{self.path}: the row for {asset!r} is a placeholder (median_spread_bps is "
                "null). The cost model will not run on a placeholder, and there is no default: "
                "SPEC.md 3.4.1 forbids the EDGE level and flat_spread_assumption_bps as level "
                "sources. Read the issuer's disclosed 30-day median from its fund page and "
                "record it with its URL and date, or supply a sourced band_bps."
            )
        return entry


def _text(node: Mapping[str, Any], key: str, path: str) -> str:
    value = node.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SpreadLevelError(f"{path}.{key}: expected a non-empty string, got {value!r}")
    return value.strip()


def _date(node: Mapping[str, Any], key: str, path: str) -> date:
    value = node.get(key)
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise SpreadLevelError(f"{path}.{key}: expected an ISO date, got {value!r}") from exc
    raise SpreadLevelError(f"{path}.{key}: expected an ISO date, got {value!r}")


def _positive_bps(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SpreadLevelError(f"{path}: expected a number in basis points, got {value!r}")
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise SpreadLevelError(f"{path}: a spread is finite and non-negative, got {number}")
    return number


def _parse_second_source(raw: Mapping[str, Any], path: str) -> SecondSource | None:
    node = raw.get("second_source")
    if node is None:
        return None
    if not isinstance(node, Mapping):
        raise SpreadLevelError(f"{path}.second_source: expected a mapping")
    return SecondSource(
        url=_text(node, "url", f"{path}.second_source"),
        value_bps=_positive_bps(node.get("value_bps"), f"{path}.second_source.value_bps"),
        note=_text(node, "note", f"{path}.second_source"),
    )


def _parse_entry(
    raw: Any, index: int, *, display_resolution_bps: float | None
) -> tuple[str, IssuerLevel | SpreadBand | None]:
    path = f"entries[{index}]"
    if not isinstance(raw, Mapping):
        raise SpreadLevelError(f"{path}: expected a mapping, got {type(raw).__name__}")
    asset = _text(raw, "asset", path)
    ticker = _text(raw, "ticker", path)
    value = raw.get("median_spread_bps")
    band = raw.get("band_bps")

    if value is None and band is None:
        # A placeholder. Its provenance fields must be null too: a URL and a
        # date beside a missing value would read as though something had been
        # looked up and lost.
        for key in ("source_url", "disclosure_date"):
            if raw.get(key) is not None:
                raise SpreadLevelError(
                    f"{path}.{key}: set while median_spread_bps is null -- a placeholder row "
                    "carries no provenance; either supply the value or clear the field"
                )
        return asset, None
    if value is not None and band is not None:
        raise SpreadLevelError(f"{path}: give median_spread_bps OR band_bps, not both")

    source_url = _text(raw, "source_url", path)
    disclosure_date = _date(raw, "disclosure_date", path)
    second = _parse_second_source(raw, path)
    if value is not None:
        displayed = _positive_bps(value, f"{path}.median_spread_bps")
        if display_resolution_bps is None:
            return asset, IssuerLevel(
                asset=asset,
                ticker=ticker,
                source_url=source_url,
                disclosure_date=disclosure_date,
                median_spread_bps=displayed,
            )
        # The interval ruling: a figure displayed at resolution r is the
        # rounding interval [v - r/2, v + r/2) of the full spread, floored at
        # zero. Same object as a sourced band, so the same code path downstream.
        half = display_resolution_bps / 2.0
        return asset, SpreadBand(
            asset=asset,
            ticker=ticker,
            source_url=source_url,
            disclosure_date=disclosure_date,
            low_bps=max(0.0, displayed - half),
            high_bps=displayed + half,
            displayed_bps=displayed,
            second_source=second,
        )
    if not isinstance(band, list) or len(band) != 2:
        raise SpreadLevelError(f"{path}.band_bps: expected [low, high], got {band!r}")
    low = _positive_bps(band[0], f"{path}.band_bps[0]")
    high = _positive_bps(band[1], f"{path}.band_bps[1]")
    if not low < high:
        raise SpreadLevelError(f"{path}.band_bps: low must be below high, got [{low}, {high}]")
    return asset, SpreadBand(
        asset=asset,
        ticker=ticker,
        source_url=source_url,
        disclosure_date=disclosure_date,
        low_bps=low,
        high_bps=high,
        second_source=second,
    )


def load_spread_levels(
    path: Path, *, expected: Mapping[str, str], display_resolution_bps: float | None = None
) -> SpreadLevels:
    """Parse the level file and check it against the universe's cost tickers.

    ``expected`` maps every asset id that carries a cost to the ticker whose
    bars price it (``Config.cost_ticker``). The file must hold exactly those
    assets, each with that ticker: a row for an asset the universe does not
    trade, or a curve point keyed to the wrong proxy, is refused.

    ``display_resolution_bps`` is ``costs.spread_level.display_resolution_bps``:
    when set, every point value is read as a rounding interval of that width
    and returned as a :class:`SpreadBand` (the interval ruling, SPEC.md 7.1.2).
    """
    if display_resolution_bps is not None and display_resolution_bps <= 0.0:
        raise SpreadLevelError(
            f"display_resolution_bps must be positive or None, got {display_resolution_bps}"
        )
    if not path.is_file():
        raise SpreadLevelError(f"spread level file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise SpreadLevelError(f"{path}: expected a mapping at the top level")
    if raw.get("schema_version") != 1:
        raise SpreadLevelError(
            f"{path}: schema_version must be 1, got {raw.get('schema_version')!r}"
        )
    raw_entries = raw.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise SpreadLevelError(f"{path}: entries must be a non-empty list")

    entries: dict[str, IssuerLevel | SpreadBand | None] = {}
    tickers: dict[str, str] = {}
    for i, item in enumerate(raw_entries):
        asset, entry = _parse_entry(item, i, display_resolution_bps=display_resolution_bps)
        if asset in entries:
            raise SpreadLevelError(f"{path}: duplicate row for {asset!r}")
        entries[asset] = entry
        tickers[asset] = _text(item, "ticker", f"entries[{i}]")

    unknown = sorted(set(entries) - set(expected))
    if unknown:
        raise SpreadLevelError(f"{path}: rows for assets that carry no cost: {unknown}")
    missing = sorted(set(expected) - set(entries))
    if missing:
        raise SpreadLevelError(f"{path}: no row for {missing}")
    for asset, ticker in expected.items():
        if tickers[asset] != ticker:
            raise SpreadLevelError(
                f"{path}: {asset!r} is priced by {ticker} (its ticker or its tradable proxy) "
                f"but the row says {tickers[asset]}"
            )
    return SpreadLevels(path=path, entries={asset: entries[asset] for asset in expected})


# ---------------------------------------------------------------------------
# Shape from EDGE, level from the issuer
# ---------------------------------------------------------------------------


def calm_baseline(production: pd.Series) -> float:
    """The in-sample median of the CLIPPED production EDGE series, proportional.

    Robust to the crisis tail, so it is the calm level with no date selection.
    The series must already be truncated at the holdout boundary; this function
    does not know where that is and must not be handed anything that crosses it.
    """
    values = production.dropna()
    if values.empty:
        raise SpreadLevelError("calm_baseline: no EDGE estimates to take a median of")
    if (values < 0.0).any():
        raise SpreadLevelError(
            "calm_baseline: negative estimates -- pass the CLIPPED production series, not "
            "the signed one (mafrm.costs.spread.SpreadPanel.production)"
        )
    return float(np.median(values.to_numpy(dtype="float64")))


def widening(production: pd.Series, baseline: float) -> pd.Series:
    """``max(0, s_EDGE(t) - baseline)`` -- the shape, in the same units as its input.

    ``NaN`` stays ``NaN``: a month the estimator has no window for is not a
    month with zero widening.
    """
    if not np.isfinite(baseline) or baseline < 0.0:
        raise SpreadLevelError(
            f"widening: baseline must be finite and non-negative, got {baseline}"
        )
    out = (production - baseline).clip(lower=0.0)
    out.name = "widening"
    return out


def half_spread(full_spread: pd.Series, *, commission: float) -> pd.Series:
    """SPEC.md 7.1's ``a_i``: half the full spread, plus commission."""
    if not np.isfinite(commission) or commission < 0.0:
        raise SpreadLevelError(
            f"half_spread: commission must be finite and non-negative, got {commission}"
        )
    out = full_spread / 2.0 + commission
    out.name = "half_spread"
    return out


@dataclass(frozen=True)
class SpreadSeries:
    """One asset's spread through time at one level. Proportional units throughout."""

    #: The issuer level used, proportional (5bp is 0.0005).
    level: float
    #: The calm baseline the widening was measured from, proportional.
    baseline: float
    #: ``max(0, s_EDGE(t) - baseline)``.
    widening: pd.Series
    #: ``level + widening``.
    full: pd.Series
    #: ``full / 2 + commission`` -- SPEC.md 7.1's ``a_i``.
    half: pd.Series


def spread_series(
    *,
    level_bps: float,
    production: pd.Series,
    commission: float,
) -> SpreadSeries:
    """The combination rule, end to end, for one asset at one issuer level.

    ``level_bps`` comes from :class:`IssuerLevel` (or one end of a
    :class:`SpreadBand`); ``production`` is the clipped monthly EDGE series,
    in-sample only. At every month where EDGE sits at or below its baseline the
    full spread equals the issuer level exactly -- agreement at baseline is a
    property of the arithmetic, and a test pins it.
    """
    level = _positive_bps(level_bps, "level_bps") * _BPS
    baseline = calm_baseline(production)
    shape = widening(production, baseline)
    full = level + shape
    full.name = "full_spread"
    return SpreadSeries(
        level=level,
        baseline=baseline,
        widening=shape,
        full=full,
        half=half_spread(full, commission=commission),
    )
