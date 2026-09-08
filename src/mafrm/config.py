"""Typed, validated loaders for ``config/model.yaml`` and ``config/universe.yaml``.

CLAUDE.md invariant 6 forbids magic numbers in code: every tunable in this
project is read from those two files through this module. Nothing here supplies
a default. A missing or mistyped key raises :class:`ConfigError` at load time,
because a config that silently falls back to a plausible value is exactly the
failure this project cannot detect later.

CLAUDE.md invariant 5 -- never read anything on or after ``HOLDOUT_START`` -- is
enforceable only once that date is set. It is deliberately ``null`` in the
committed config; :meth:`Config.require_holdout_start` raises until a human
pins it.
"""

from __future__ import annotations

import itertools
import math
import os
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, Generic, Literal, TypeVar, cast

import yaml

__all__ = [
    "SEED",
    "AdvConfig",
    "AlphaFlagConfig",
    "AqrConfig",
    "AqrDataset",
    "Asset",
    "Comparand",
    "ConditionNumberConfig",
    "Config",
    "ConfigError",
    "CostConfig",
    "CovarianceConfig",
    "CrisisWindow",
    "CrossCheckConfig",
    "DailyAnchor",
    "DataConfig",
    "DurationCheckConfig",
    "DurationCheckInstrument",
    "EdgeSpreadConfig",
    "EigenfactorConfig",
    "EquityUniverseConfig",
    "EquityUniverseRegistrations",
    "EtfConfig",
    "ExemptPair",
    "FactorsConfig",
    "FredConfig",
    "FredSeries",
    "HalflifeSensitivityConfig",
    "Horizons",
    "HybridConfig",
    "MacroAssetFactorConfig",
    "MacroBetaConfig",
    "MacroDollarConfig",
    "MacroEquityConfig",
    "MacroFactorConfig",
    "MacroRatesConfig",
    "MeanConvention",
    "ModelConfig",
    "NumericsConfig",
    "OptimizerAlphaConfig",
    "OptimizerConfig",
    "OptimizerConstraint",
    "OptimizerConstraintsConfig",
    "OptimizerVerificationConfig",
    "OrthogonalizationConfig",
    "RatePcaConfig",
    "Row83Config",
    "SeriesClassification",
    "SpecificRiskConfig",
    "SpreadVersusVolatilityConfig",
    "SpyVersusMarketConfig",
    "StooqConfig",
    "TltVersusLadderConfig",
    "UniverseConfig",
    "ValidationBatteryConfig",
    "ValidationConfig",
    "ValidationLadderConfig",
    "VolatilityRegimeAdjustmentConfig",
    "load",
    "load_model",
    "load_universe",
]

# Repo root: src/mafrm/config.py -> src/mafrm -> src -> <root>.
_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

#: Tolerance for descriptor-composite weights summing to one.
_WEIGHT_SUM_TOL: Final[float] = 1e-9

#: How the EWMA moments in SPEC.md 5.1 are centred. Only ``"zero"`` is
#: supported and that is deliberate: see SPEC.md 5.1.1 and the reasoning at
#: ``covariance.mean_convention`` in ``config/model.yaml``. The alias exists so
#: that the decision has a name in the type system, not so that it has options.
MeanConvention = Literal["zero"]

_MEAN_CONVENTIONS: Final[frozenset[str]] = frozenset({"zero"})

#: ``factors.macro.orthogonalization.expanding_window_start``, shipped value.
#: The Gram-Schmidt opens its expanding window at ``model.sample.start``, so no
#: hedge ratio is estimated on history that precedes the model window.
#: SPEC.md 4.1.3, experiments.md row 69.
_SAMPLE_START: Final[str] = "sample.start"

#: The recognised values of that key. Two, deliberately: the alternative that
#: W2-P1 ran is named rather than deleted, so switching back is a config edit
#: with a row in ``experiments.md`` behind it rather than a silent code change.
_ORTHOGONALIZATION_WINDOW_STARTS: Final[frozenset[str]] = frozenset({_SAMPLE_START, "full_history"})

#: ``factors.validation.comparands.*.gate``. ``"placebo"`` means the comparand is
#: another estimate of the same object and the placebo gate scores it;
#: ``"sign"`` means it is a transform whose coefficient is a product of
#: unpublished quantities, so only the sign is falsifiable. SPEC.md 6.5.1.
_SIGN_GATE: Final[str] = "sign"
_PLACEBO_GATE: Final[str] = "placebo"
_VALIDATION_GATES: Final[frozenset[str]] = frozenset({_SIGN_GATE, _PLACEBO_GATE})

#: ``factors.validation.newey_west_lag_rule``. Newey & West (1994) deterministic
#: plug-in bandwidth, ``floor(4 * (T/100)^(2/9))``, with ``T`` supplied at
#: runtime. A rule rather than a number: the count is a function of the sample
#: the regression actually ran on, which at monthly frequency is not the count
#: the daily covariance pipeline uses.
_NEWEY_WEST_1994: Final[str] = "newey_west_1994_plug_in"


class ConfigError(ValueError):
    """A config file is missing a required key, or a key has the wrong type."""


# ---------------------------------------------------------------------------
# Validation primitives. Each raises rather than defaulting.
# ---------------------------------------------------------------------------


def _as_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{path}: expected a mapping, got {type(value).__name__}")
    return value


def _require(node: dict[str, Any], key: str, path: str) -> Any:
    if key not in node:
        raise ConfigError(f"{path}.{key} is required but missing")
    return node[key]


def _child(node: dict[str, Any], key: str, path: str) -> dict[str, Any]:
    return _as_mapping(_require(node, key, path), f"{path}.{key}")


def _int(node: dict[str, Any], key: str, path: str) -> int:
    value = _require(node, key, path)
    # bool is a subclass of int; a stray `true` must not silently become 1.
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"{path}.{key}: expected an int, got {value!r}")
    return int(value)


def _float(node: dict[str, Any], key: str, path: str) -> float:
    value = _require(node, key, path)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path}.{key}: expected a number, got {value!r}")
    return float(value)


def _str(node: dict[str, Any], key: str, path: str) -> str:
    value = _require(node, key, path)
    if not isinstance(value, str):
        raise ConfigError(f"{path}.{key}: expected a string, got {value!r}")
    return value


def _opt_str(node: dict[str, Any], key: str, path: str) -> str | None:
    value = _require(node, key, path)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError(f"{path}.{key}: expected a string or null, got {value!r}")
    return value


def _opt_float(node: dict[str, Any], key: str, path: str) -> float | None:
    value = _require(node, key, path)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path}.{key}: expected a number or null, got {value!r}")
    return float(value)


def _bool(node: dict[str, Any], key: str, path: str) -> bool:
    value = _require(node, key, path)
    if not isinstance(value, bool):
        raise ConfigError(f"{path}.{key}: expected a bool, got {value!r}")
    return value


def _date(node: dict[str, Any], key: str, path: str) -> date:
    value = _require(node, key, path)
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise ConfigError(f"{path}.{key}: expected an ISO date string, got {value!r}")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ConfigError(f"{path}.{key}: {value!r} is not an ISO date") from exc


def _opt_date(node: dict[str, Any], key: str, path: str) -> date | None:
    if _require(node, key, path) is None:
        return None
    return _date(node, key, path)


def _float_list(node: dict[str, Any], key: str, path: str) -> tuple[float, ...]:
    value = _require(node, key, path)
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{path}.{key}: expected a non-empty list, got {value!r}")
    out: list[float] = []
    for i, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ConfigError(f"{path}.{key}[{i}]: expected a number, got {item!r}")
        out.append(float(item))
    return tuple(out)


def _str_list(node: dict[str, Any], key: str, path: str) -> tuple[str, ...]:
    value = _require(node, key, path)
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{path}.{key}: expected a non-empty list, got {value!r}")
    for i, item in enumerate(value):
        if not isinstance(item, str):
            raise ConfigError(f"{path}.{key}[{i}]: expected a string, got {item!r}")
    return tuple(value)


def _check_weights_sum_to_one(weights: dict[str, float], path: str) -> None:
    total = sum(weights.values())
    if abs(total - 1.0) > _WEIGHT_SUM_TOL:
        raise ConfigError(f"{path}: composite weights must sum to 1.0, got {total!r}")


# ---------------------------------------------------------------------------
# model.yaml
# ---------------------------------------------------------------------------

T = TypeVar("T", int, float)


@dataclass(frozen=True)
class Horizons(Generic[T]):
    """A parameter with a separate short-horizon and long-horizon value."""

    short: T
    long: T


def _int_horizons(node: dict[str, Any], key: str, path: str) -> Horizons[int]:
    child = _child(node, key, path)
    sub = f"{path}.{key}"
    return Horizons(short=_int(child, "short", sub), long=_int(child, "long", sub))


def _float_horizons(node: dict[str, Any], key: str, path: str) -> Horizons[float]:
    child = _child(node, key, path)
    sub = f"{path}.{key}"
    return Horizons(short=_float(child, "short", sub), long=_float(child, "long", sub))


@dataclass(frozen=True)
class SampleConfig:
    """Backtest window bounds, including the untouchable holdout (invariant 5)."""

    start: date
    holdout_start: date | None
    holdout_end: date | None


@dataclass(frozen=True)
class CurveConfig:
    """One Fed fitted zero curve. SPEC.md 3.2."""

    url: str
    yield_prefix: str
    maturities_years: tuple[float, ...]


@dataclass(frozen=True)
class KenFrenchSiccodes:
    """Ken French's Fama-French 49 industry definitions. SPEC.md 15.6.1 ruling 1 (W7-P3).

    A URL, the member file inside the zip, and two facts about the scheme the
    parser asserts: the industry count and the index of "Other", which takes
    every SIC code in none of the listed ranges -- French's rule.
    """

    description: str
    url: str
    member: str
    industries: int
    unassigned_industry: int


@dataclass(frozen=True)
class KenFrenchRiskFree:
    """The SELECTED short rate. Implemented in W1-P4; the excess-return leg.

    ``units`` is the field that earns its place. Ken French publishes ``RF`` as
    percent per *trading day*; ``DGS1MO``, which this replaces, is percent per
    *annum*. Both are small plausible numbers, so the substitution would be
    silent. :func:`mafrm.data.french.as_annual_percent` converts explicitly.
    """

    description: str
    history_start: date
    url: str
    #: ``pandas_datareader.famafrench`` dataset key for the daily factor file.
    dataset: str
    #: Column holding the one-month bill inside that table.
    rf_column: str
    #: Publisher's units. Only ``percent_per_trading_day`` is understood.
    units: str
    implemented: bool


@dataclass(frozen=True)
class InterimFredRiskFree:
    """The short rate actually cached and read today. SPEC.md 3.2."""

    fred_series: str
    url_template: str
    history_start: date
    implemented: bool

    @property
    def url(self) -> str:
        return self.url_template.format(series_id=self.fred_series)


@dataclass(frozen=True)
class RiskFreeConfig:
    """The excess-return leg: what was chosen, and what is wired up today.

    ``selected`` names Ken French's daily one-month bill, chosen because it
    covers the whole 1961- GSW sample with no splice. It is not implemented --
    ``interim_fred`` is what :func:`mafrm.data.gsw.load_risk_free` reads, so
    excess returns currently begin 2001-07-31 rather than 1961.
    """

    selected: str
    ken_french_daily_rf: KenFrenchRiskFree
    interim_fred: InterimFredRiskFree


@dataclass(frozen=True)
class ValidationLadderConfig:
    """The par-coupon ladder TLT is scored against. VALIDATION ONLY.

    Never a factor input. The government sleeve's production series are the
    constant-maturity zeros in :class:`CurveConfig`; this exists so that an ETF
    is compared against something with an ETF's cashflow structure.
    """

    maturities_years: tuple[float, ...]
    weights: tuple[float, ...]
    coupon_frequency: int

    @property
    def normalised_weights(self) -> tuple[float, ...]:
        total = sum(self.weights)
        return tuple(w / total for w in self.weights)


@dataclass(frozen=True)
class TltCrossCheckConfig:
    """The SPEC.md 3.2 acceptance test, amended 2026-08-26.

    ``max_abs_gap_pct_per_year`` and the beta band are the gate.
    ``min_correlation`` is retained only as a reported diagnostic: a no-roll
    control moved the daily correlation by 1e-7, so it cannot falsify the
    roll-down it was written to test.
    """

    ticker: str
    start: date
    min_correlation: float
    maturity_years: float
    max_abs_gap_pct_per_year: float
    beta_min: float
    beta_max: float
    #: iShares fact sheet: TLT against the ICE US Treasury 20+ Year Bond Index.
    published_tracking_difference_pct_per_year: float
    expense_ratio_pct_per_year: float
    seasoning_sweep_years: tuple[float, ...]


@dataclass(frozen=True)
class EtfConfig:
    """How ETF bars are requested and what is kept. SPEC.md 3.5.

    ``required_response_columns`` is asserted on the *response*, not on the
    cache, and it includes ``Adj Close`` even though that column is deliberately
    discarded: its disappearance is the signal that yfinance flipped
    ``auto_adjust`` again and the "raw" closes are no longer raw.
    """

    interval: str
    auto_adjust: bool
    period: str
    price_columns: tuple[str, ...]
    action_columns: tuple[str, ...]
    required_response_columns: tuple[str, ...]


SeriesClassification = Literal["market_observed", "revised_statistic"]


@dataclass(frozen=True)
class FredSeries:
    """One FRED series, and whether ALFRED vintages are required for it.

    The distinction is what vintages are actually *for* (SPEC.md 3.5). A
    ``revised_statistic`` is estimated, published, and then restated as source
    data arrives -- payrolls, GDP, CPI -- so using today's value as a factor is
    look-ahead through revision, and the initial-release series is mandatory.

    A ``market_observed`` series is a price, a yield or an index computed from
    quotes actually printed that day. There is no estimate to restate and the
    value is final when published, so absent or partial vintage coverage is a
    fact about when ALFRED adopted the series rather than a defect in the data.
    Vintages are still fetched where they exist -- they cost one request and
    they are the only way to answer "was this revised after all?" -- but they
    are not required.
    """

    id: str
    name: str
    description: str
    classification: SeriesClassification

    @property
    def requires_vintages(self) -> bool:
        """Only a revised statistic can suffer look-ahead through revision."""
        return self.classification == "revised_statistic"


@dataclass(frozen=True)
class FredConfig:
    """FRED/ALFRED access. The API, never the keyless CSV endpoint."""

    api_base_url: str
    api_key_env_var: str
    #: FRED's documented real-time bounds. Required by ``output_type=4``, which
    #: rejects the default real-time period outright.
    vintage_realtime_start: date
    vintage_realtime_end: str
    #: FRED's documented maximum vintages per ``output_type=4`` request.
    vintage_chunk_size: int
    series: tuple[FredSeries, ...]

    def by_id(self, series_id: str) -> FredSeries:
        for series in self.series:
            if series.id == series_id:
                return series
        raise KeyError(f"no FRED series {series_id!r} in model.data.fred.series")

    @property
    def revised_statistics(self) -> tuple[FredSeries, ...]:
        """Series whose initial release is mandatory. Currently empty by design."""
        return tuple(s for s in self.series if s.requires_vintages)

    @property
    def market_observed(self) -> tuple[FredSeries, ...]:
        return tuple(s for s in self.series if not s.requires_vintages)


@dataclass(frozen=True)
class AqrDataset:
    """One AQR workbook's layout. Every field is observed, none is guessed."""

    name: str
    url: str
    description: str
    sheet: str
    #: Zero-based header row. The banner height differs per file.
    header_row: int
    #: Header text of the date column, or ``None`` when AQR left the cell blank.
    date_column: str | None
    required_columns: tuple[str, ...]
    #: Whether AQR maintains the file. Only a maintained file can drift; a
    #: static paper companion whose end date never moves is behaving correctly.
    expected_to_update: bool


@dataclass(frozen=True)
class AqrConfig:
    """The three AQR workbooks and the contracts they are held to.

    There is deliberately no staleness limit here. AQR data is a validation
    comparand, not a production input, and what a comparand owes is coverage of
    the window it is compared over -- see ``data.aqr`` in ``config/model.yaml``.
    """

    user_agent_is_browser: bool
    datasets: tuple[AqrDataset, ...]

    @property
    def maintained(self) -> tuple[AqrDataset, ...]:
        """Datasets the release-drift detector applies to."""
        return tuple(d for d in self.datasets if d.expected_to_update)

    def by_name(self, name: str) -> AqrDataset:
        for dataset in self.datasets:
            if dataset.name == name:
                return dataset
        raise KeyError(f"no AQR dataset {name!r} in model.data.aqr.datasets")


@dataclass(frozen=True)
class SpyVersusMarketConfig:
    """SPY's own total return against Ken French's CRSP market excess return.

    The strongest independence available in this project: CRSP is not Yahoo, the
    dividend treatment is CRSP's rather than ours, and the two series share no
    code path. Every tolerance is derived from what it must discriminate rather
    than fitted to what was measured -- see the comments in ``config/model.yaml``.
    """

    ticker: str
    benchmark_column: str
    min_correlation: float
    max_abs_mean_gap_pct_per_year: float
    max_abs_daily_difference_pct: float


@dataclass(frozen=True)
class TltVersusLadderConfig:
    """TLT's Yahoo closes against the curve-derived par-coupon ladder.

    Gated on the regression beta only, which is a question about the *prices*.
    The mean gap is a question about the *construction* and is W1-P3's open
    residual; both take their bounds from :class:`TltCrossCheckConfig`.
    """

    ticker: str


@dataclass(frozen=True)
class CrossCheckConfig:
    """The independent corroboration the sleeve actually has."""

    spy_versus_market: SpyVersusMarketConfig
    tlt_versus_ladder: TltVersusLadderConfig


@dataclass(frozen=True)
class StooqConfig:
    """The independent price cross-check. Never a production input.

    There is deliberately no drift tolerance here: none has been validated
    against real Stooq data, and CLAUDE.md invariant 9 forbids inventing one.
    """

    url_template: str
    symbol_template: str
    cross_check_tickers: tuple[str, ...]


@dataclass(frozen=True)
class DataConfig:
    """Everything the data layer needs that is not a universe member."""

    trading_days_per_year: int
    #: 252 / 12. Calendar arithmetic used ONLY by
    #: :func:`mafrm.risk.horizon.to_monthly`, downstream of the whole covariance
    #: pipeline. Deliberately not on ``RiskConfig``; see the key in
    #: ``config/model.yaml`` and the W3-P2 ruling.
    trading_days_per_month: int
    svensson_small_n: float
    curve_tau2_missing_sentinel: float
    nominal_curve: CurveConfig
    real_curve: CurveConfig
    risk_free: RiskFreeConfig
    validation_ladder: ValidationLadderConfig
    #: Non-universe tickers fetched only to isolate the SPY spread residual.
    #: They never reach a factor model, a covariance matrix or the optimizer.
    spread_structure_control: tuple[str, ...]
    tlt_cross_check: TltCrossCheckConfig
    etf: EtfConfig
    fred: FredConfig
    aqr: AqrConfig
    cross_checks: CrossCheckConfig
    stooq: StooqConfig
    ken_french_siccodes49: KenFrenchSiccodes

    @property
    def roll_down_step(self) -> float:
        """Delta, the one-trading-day roll in years. SPEC.md 3.2."""
        return 1.0 / self.trading_days_per_year


@dataclass(frozen=True)
class EffectiveSampleSize:
    """EWMA effective sample size ``T_eff = 2*tau/ln(2)`` and its check table."""

    formula: str
    reference: dict[int, int]


@dataclass(frozen=True)
class NumericsConfig:
    #: What a non-PSD eigenvalue is REPAIRED to (CLAUDE.md invariant 4).
    psd_eigenvalue_floor: float
    #: Roundings a repaired matrix accumulates per entry that do NOT depend on
    #: its size: the scaling by the floored spectrum, and the symmetrisation.
    #: ``assert_psd`` detects on ``-(size + this) * eps * lambda_max`` -- a bound
    #: in units of machine epsilon, so it is scale-relative. W4-P2b ruling.
    psd_reconstruction_roundings: int
    #: Largest permitted ``max|A - A.T|`` before a matrix is called asymmetric.
    symmetry_absolute_tolerance: float
    effective_sample_size: EffectiveSampleSize


@dataclass(frozen=True)
class MacroEquityConfig:
    """The equity factor's source series. SPEC.md 4.1.1, settled 2026-08-30."""

    source: str
    #: Column in the Ken French daily table, published in percent per trading day.
    column: str


@dataclass(frozen=True)
class RatePcaConfig:
    """How the level and slope PCs are extracted and normalized. SPEC.md 4.1.1.

    ``level_mean_loading_bps`` and ``slope_loading_difference_bps`` are what make
    an exposure readable: they rescale each eigenvector so that the level PC's
    mean loading across the tenors, and the slope PC's long-minus-short loading
    difference, are one basis point. Both targets are quadratic in the
    eigenvector's sign, so they also FIX the sign -- there is no separate sign
    convention to get wrong, and a test pins that.

    ``min_normalizer`` is a guard rather than a tunable. Both rescalings divide
    by a loading combination; a degenerate eigenvector drives that denominator to
    zero and would emit a factor scaled by an arbitrarily large number instead of
    failing.
    """

    estimator: str
    expanding_min_window: int
    level_mean_loading_bps: float
    slope_loading_difference_bps: float
    min_normalizer: float


@dataclass(frozen=True)
class MacroRatesConfig:
    """Curve points the rate factors are built from. SPEC.md 4.1."""

    tenors_years: tuple[float, ...]
    pca: RatePcaConfig


@dataclass(frozen=True)
class MacroAssetFactorConfig:
    """A factor that is one universe member's return, net of something.

    ``excess_over`` is ``"cash"`` for both members that use this shape. It is a
    key rather than an assumption because SPEC.md 4.1 originally specified the
    credit factor as an excess over a duration-matched Treasury, and SPEC.md
    4.1.1 replaced that with an excess over cash plus an orthogonalization. The
    key records which of the two this build is running.
    """

    asset_id: str
    excess_over: str


@dataclass(frozen=True)
class MacroDollarConfig:
    """The dollar factor. SPEC.md 4.1: "sign so positive = USD strength"."""

    fred_series: str
    #: +1 because the broad dollar index rises as the dollar strengthens.
    sign: float


@dataclass(frozen=True)
class ExemptPair:
    """One targeted pair excused from the level criterion, by name and with a reason.

    Deliberately not a threshold. A pair that cannot meet the criterion is
    exempted **visibly**, carrying why and what follow-up it owes, rather than by
    moving the number until it fits. See SPEC.md 4.1.2.
    """

    target: str
    regressor: str
    reason: str
    owed: str

    @property
    def pair(self) -> frozenset[str]:
        return frozenset((self.target, self.regressor))


@dataclass(frozen=True)
class ConditionNumberConfig:
    """The conditioning diagnostic. SPEC.md 4.1.2 -- reported, never gated.

    ``halflife_from`` and ``crisis_windows_from`` are pointers to the keys the
    values are actually read from, held so that the coupling is visible in the
    config rather than only in code. Nothing here duplicates a number that lives
    somewhere else in the file.
    """

    halflife_from: str
    min_observations: int
    crisis_windows_from: str


@dataclass(frozen=True)
class OrthogonalizationConfig:
    """Sequential expanding-window Gram-Schmidt on the factor series. SPEC.md 4.1.

    ``against`` is the spec's own pairing -- slope on level, credit on equity and
    level, commodity on dollar -- and not a general dependency graph. A factor
    absent from it is never modified, which is what makes ``commodity`` regressed
    on the later-ordered ``dollar`` well defined; the parser enforces exactly
    that condition rather than leaving it to a comment.

    ``fit_intercept`` and ``subtract_intercept`` are separate keys because they
    are separate decisions. Fitting the intercept is what makes the residual
    uncorrelated with its regressors rather than merely orthogonal in the
    uncentred sense. NOT subtracting it is what preserves the factor's own mean
    return, which is a risk premium an orthogonalization has no business
    removing.
    """

    order: tuple[str, ...]
    against: dict[str, tuple[str, ...]]
    expanding_min_window: int
    #: Where the expanding window opens. ``"sample.start"`` (shipped, W2-P2) or
    #: ``"full_history"`` (what W2-P1 ran). SPEC.md 4.1.3; experiments.md row 69.
    #: A pointer, not a date -- the value it names lives in ``model.sample``.
    expanding_window_start: str
    fit_intercept: bool
    subtract_intercept: bool
    #: Level criterion for the pairs ``against`` targets. SPEC.md 4.1.2 -- scoped
    #: to those pairs only; the all-pairs version of this bar was withdrawn as
    #: invalid, because meeting it would drive the factor correlation matrix
    #: toward diagonal and leave the eigenfactor adjustment nothing to correct.
    targeted_pair_max_abs_correlation: float
    exempt_pairs: tuple[ExemptPair, ...]

    def regressors_for(self, factor: str) -> tuple[str, ...]:
        """Factors ``factor`` is projected off. Empty when it is left alone."""
        return self.against.get(factor, ())

    @property
    def targeted_pairs(self) -> tuple[frozenset[str], ...]:
        """The pairs the scheme acts on, and therefore the pairs it can be judged on."""
        return tuple(
            frozenset((target, regressor))
            for target, regressors in self.against.items()
            for regressor in regressors
        )

    def is_exempt(self, pair: frozenset[str]) -> bool:
        return any(item.pair == pair for item in self.exempt_pairs)

    @property
    def burns_in_inside_the_sample(self) -> bool:
        """True when the Gram-Schmidt sees nothing before ``model.sample.start``."""
        return self.expanding_window_start == _SAMPLE_START


@dataclass(frozen=True)
class MacroBetaConfig:
    """Model A's rolling exposure regression. SPEC.md 4.1.

    ``window`` and ``halflife`` are the Barra BETA descriptor's, deliberately
    duplicated rather than aliased to ``equity_descriptors.beta``: they are the
    same numbers for the same published reason, but one is a multi-asset
    time-series beta and the other a cross-sectional equity descriptor.
    :attr:`window_halflife` is what the drift test compares.

    ``fit_intercept`` is specified by SPEC.md 4.1's own equation, which carries
    ``alpha_i``. It is a key rather than a literal because CLAUDE.md invariant 6
    requires the value to be in the config before code reads it.
    """

    window: int
    halflife: int
    fit_intercept: bool

    @property
    def window_halflife(self) -> WindowHalflife:
        return WindowHalflife(window=self.window, halflife=self.halflife)


@dataclass(frozen=True)
class AlphaFlagConfig:
    """When a Model A intercept counts as distinguishable from zero. SPEC.md 4.1.

    ``newey_west_lags_from`` is a POINTER to the key the lag count is actually
    read from, held in the same style as
    :attr:`ConditionNumberConfig.halflife_from` so the coupling is visible in the
    config rather than only in code. Nothing here duplicates a number that lives
    elsewhere in the file.
    """

    abs_t_statistic: float
    newey_west_lags_from: str


@dataclass(frozen=True)
class DurationCheckInstrument:
    """One pre-registered duration prediction. experiments.md rows 64-67.

    ``predicted_duration`` is NEGATIVE -- it is the regression slope on the
    normalized level factor, which reads as minus the effective duration.
    """

    id: str
    kind: str
    predicted_duration: float
    maturity_years: float | None
    ticker: str | None
    predicted_min: float | None
    predicted_max: float | None

    def band(self, tolerance_fraction: float) -> tuple[float, float]:
        """The registered acceptance band, low first.

        An explicit ``predicted_min``/``predicted_max`` pair wins: it means the
        prediction was registered AS a range and must not be widened again by
        applying the fractional tolerance on top of it.
        """
        if self.predicted_min is not None and self.predicted_max is not None:
            return (self.predicted_min, self.predicted_max)
        half = abs(self.predicted_duration) * tolerance_fraction
        return (self.predicted_duration - half, self.predicted_duration + half)


@dataclass(frozen=True)
class DurationCheckConfig:
    """Pre-registered duration predictions for the level-factor normalization.

    Registered in experiments.md rows 64-67 BEFORE the factor existed. This block
    is the machine-readable copy so the report regenerates without transcription;
    it is not a tolerance to be tuned, and editing a value to match a measurement
    is the manipulation the experiment log exists to prevent.
    """

    tolerance_fraction: float
    instruments: tuple[DurationCheckInstrument, ...]


@dataclass(frozen=True)
class MacroFactorConfig:
    """Model A, the macro factor set. SPEC.md 4.1 as amended by 4.1.1."""

    equity: MacroEquityConfig
    rates: MacroRatesConfig
    credit: MacroAssetFactorConfig
    commodity: MacroAssetFactorConfig
    dollar: MacroDollarConfig
    orthogonalization: OrthogonalizationConfig
    beta: MacroBetaConfig
    min_r_squared_flag: float
    #: Fraction of defined dates a statistic must breach its criterion on before
    #: it is called "persistent". Shared by the R-squared and alpha flags.
    persistence_fraction: float
    alpha_flag: AlphaFlagConfig
    condition_number: ConditionNumberConfig
    duration_check: DurationCheckConfig


@dataclass(frozen=True)
class Comparand:
    """One published series a self-built factor is regressed on. SPEC.md 6.5.

    ``gate`` is ``"placebo"`` or ``"sign"`` and it is a statement about what the
    comparison can support, not a severity dial. ``"sign"`` marks a comparand
    that is a *transform* of the factor rather than another estimate of it --
    the coefficient is a product of unpublished quantities, so only its sign is
    falsifiable and :attr:`expected_beta_sign` carries the prediction.
    """

    factor: str
    dataset: str
    column: str
    gate: str
    #: Set only when ``gate == "sign"``. A convention, like ``dollar.sign``.
    expected_beta_sign: float | None

    @property
    def is_sign_test(self) -> bool:
        return self.gate == _SIGN_GATE


@dataclass(frozen=True)
class DailyAnchor:
    """experiments.md row 41, transcribed. The one measured prior in W2-P3.

    Every other expectation in the validation table is a placebo or a sign. This
    one is a number somebody already measured, so the daily equity comparand
    test has a sharp falsifier: reproduce it **on row 41's own window** to within
    sampling error. Editing a value here to match a rerun is the manipulation
    the experiment log exists to detect.
    """

    comparand: str
    registered_correlation: float
    registered_mean_gap_pct_per_year: float
    registered_observations: int
    source_row: int


@dataclass(frozen=True)
class ValidationConfig:
    """W2-P3's comparand mapping, stated before it was implemented.

    ``unmapped`` is the part that matters most. Three of Model A's six factors
    have no published analogue in the files this project holds, and they are
    named here with the reason rather than regressed against the nearest
    available column.
    """

    comparands: tuple[Comparand, ...]
    unmapped: dict[str, str]
    tsmom_applicable: bool
    rolling_correlation_window_months: int
    newey_west_lag_rule: str
    placebo_gate: bool
    daily_anchor: DailyAnchor

    def by_factor(self, factor: str) -> Comparand:
        for item in self.comparands:
            if item.factor == factor:
                return item
        raise ConfigError(f"no comparand configured for factor {factor!r}")

    @property
    def mapped(self) -> tuple[str, ...]:
        return tuple(item.factor for item in self.comparands)

    @property
    def placebo_factors(self) -> tuple[str, ...]:
        """Factors the placebo gate is scored on -- every mapped factor."""
        return self.mapped

    @property
    def rolling_correlation_standard_error(self) -> float:
        """Fisher-z standard error at the configured window, ``1/sqrt(w-3)``.

        A formula, not a parameter. Printed beside every rolling correlation so a
        drift inside two of these is not read as a change in anything.
        """
        return 1.0 / math.sqrt(self.rolling_correlation_window_months - 3)


@dataclass(frozen=True)
class StatisticalFactorConfig:
    """Model B, the statistical factor model. SPEC.md 4.2, as ruled in 4.2.1-4.2.4.

    ``expanding_min_window`` HAS NO KEY OF ITS OWN in ``config/model.yaml``: it
    is copied from ``factors.macro.rates.pca.expanding_min_window`` by
    :func:`_parse_statistical`, so that exactly one number in the file says how
    long an expanding-window PCA must run before it is trusted.

    That reuse is permitted and the opposite direction is not, which is worth
    stating here rather than only at the key. SPEC.md 5.1.2 forbids importing a
    day count into a **risk-side** estimator, because ``risk/`` has to stay
    asset-class agnostic (CLAUDE.md invariant 10) and a burn-in there must be a
    bound on ``K/T_eff``. Model B lives in ``factors/``, where the parameter
    already is, and means the same thing by it. Reuse *within* ``factors/`` is
    not the violation; importing *into* ``risk/`` is.
    """

    #: ``"sample_correlation"``. Equally weighted, never the EWMA correlation --
    #: Marchenko-Pastur is derived for equally-weighted iid observations and
    #: ``lambda_+`` takes that ``T`` directly. SPEC.md 4.2.1.
    estimator: str
    #: Copied from the rates PCA's key. See the class docstring.
    expanding_min_window: int
    #: SPEC.md 4.2, Lopez de Prado ch. 2. ABSOLUTE, in eigenvalue units.
    kde_bandwidth: float
    #: Grid resolution of the ``sigma^2`` fit. A discretisation, not a dial.
    kde_grid_points: int
    #: ``(low, high)`` for the fitted noise variance. Derived from the estimator
    #: being a CORRELATION matrix, where unit variance bounds it above by 1.
    noise_variance_bounds: tuple[float, float]
    #: How many leading eigenvectors the detoned variant removes. SPEC.md 4.2
    #: says the first one.
    detoned_components: int
    #: Both variants are always built. Neither is chosen.
    variants: tuple[str, ...]
    #: Trading days between rebalances in the minimum-variance comparand run.
    comparand_rebalance_days: int


@dataclass(frozen=True)
class Row83Config:
    """experiments.md row 83's criteria, ruled 2026-08-31 before the panel existed.

    Every field is a RULE or a DERIVED tenor. There is no magnitude here that
    anyone chose, and that is the point: row 83 was registered on 2026-08-30,
    five sessions before the hybrid existed, and a threshold picked once the
    loadings were within reach would have thrown away everything the early
    registration bought.
    """

    #: ``"largest_absolute"``. Gold must hold the largest absolute loading on
    #: the residual PC. A rank, so it is scale-free and has a computable null of
    #: ``1/N`` per component.
    material_loading_rule: str
    #: ``"real_beats_nominal"``. A PLACEBO, not a significance bar:
    #: ``|rho(PC, d.real)| > |rho(PC, d.nominal)|``. Null rate 1/2.
    comparand_rule: str
    #: ``"daily_change_bps"``. The comparand is first-differenced; correlating a
    #: return series against a yield LEVEL is a spurious-regression setup.
    comparand_transform: str
    #: The asset whose loading is made positive before the sign clause is read.
    #: A PC's sign is unidentified, so the clause is stated sign-invariantly.
    orientation_asset: str
    #: ``-1``: gold rises when real yields fall.
    expected_correlation_sign: int
    #: DERIVED from ``data.real_curve.maturities_years``, which has exactly one
    #: entry. Not a key of its own -- the universe states the tenor once.
    tenor_years: float


@dataclass(frozen=True)
class HybridConfig:
    """The hybrid. SPEC.md 4.3, as ruled in 4.3.1-4.3.3.

    ``expanding_min_window`` HAS NO KEY OF ITS OWN, for the same reason
    :class:`StatisticalFactorConfig` has none: one number in
    ``config/model.yaml`` says how long an expanding-window PCA must run before
    it is trusted, and it is stated at the rates PCA. A key of that name under
    ``factors.hybrid`` is REJECTED by :func:`_parse_hybrid` rather than ignored.

    ``component_counts`` is ``(1, 2)`` because SPEC.md 4.3 says "the top 1-2
    residual PCs" and that is a range over MODELS rather than over noise. Both
    are built and neither is selected -- the same treatment ``a = 1.0`` /
    ``a = 1.4`` and ``denoised`` / ``detoned`` get.
    """

    #: ``"sample_correlation"``. Equally weighted; the residuals already carry
    #: the regression's own EWMA decay and weighting them again would compound
    #: two schemes with no stated meaning for the product.
    estimator: str
    #: Copied from ``factors.macro.rates.pca.expanding_min_window``.
    expanding_min_window: int
    #: ``(1, 2)``. Both built, neither chosen.
    component_counts: tuple[int, ...]
    #: ``"duration_effect"``: the regressand for universe members constructed as
    #: ``synthetic_gsw_zero_curve``. W2-P3's mechanical control, fixed before any
    #: residual PC was seen -- experiments.md rows 79-82 and 89.
    government_zero_return_leg: str
    #: Row 83's criteria.
    row_83: Row83Config


@dataclass(frozen=True)
class FactorsConfig:
    """Factor construction. The equity module joins this in week 7."""

    macro: MacroFactorConfig
    statistical: StatisticalFactorConfig
    hybrid: HybridConfig
    validation: ValidationConfig


@dataclass(frozen=True)
class CovarianceConfig:
    """Factor covariance pipeline parameters. USE4 Table 4.1."""

    factor_volatility_halflife: Horizons[int]
    factor_correlation_halflife: Horizons[int]
    volatility_newey_west_lags: Horizons[int]
    correlation_newey_west_lags: Horizons[int]
    #: ``"zero"``: EWMA moments are taken about zero, never about a trailing
    #: mean. A specification decision, not a dial -- see the reasoning at the key
    #: in ``config/model.yaml`` and SPEC.md 5.1.1. ``"sample"`` is rejected at
    #: load time rather than supported, so the alternative cannot be reached by
    #: editing the config.
    mean_convention: MeanConvention
    #: SPEC.md 5's pipeline stages in application order. ``make model`` fails on
    #: any stage named here that ``mafrm.risk.covariance`` does not implement.
    stages: tuple[str, ...]


@dataclass(frozen=True)
class VolatilityRegimeAdjustmentConfig:
    """USE4 section 4.4. One half-life, shared by the factor and specific VRA."""

    halflife: Horizons[int]


@dataclass(frozen=True)
class EigenfactorConfig:
    """Eigenfactor risk adjustment. USE4 section 4.3."""

    monte_carlo_trials: int
    #: Both 1.0 (attribution-facing) and 1.4 (optimizer-facing) are always run.
    scaling_a: tuple[float, ...]


@dataclass(frozen=True)
class SpecificRiskConfig:
    """Specific-risk parameters. USE4 Table 4.1 and section 4.5."""

    ewma_halflife: Horizons[int]
    newey_west_lags: Horizons[int]
    autocorrelation_halflife: Horizons[int]
    bayesian_shrinkage_q: Horizons[float]


@dataclass(frozen=True)
class HalflifeSensitivityConfig:
    """SPEC.md 5.1's half-life sweep grid. W4-P2b.

    One field, because the sweep has exactly one dial. SPEC.md 5.1 asks for a
    sensitivity chart "across half-lives from 21 to 504 days" and this is the
    grid it is drawn on; what varies is the FACTOR VOLATILITY half-life, with
    the correlation half-life held at its pinned 504 (CLAUDE.md's parameter
    table marks that identical-across-horizons value deliberate).

    The grid is not a set of candidates. SPEC.md 5.1.3 classifies the sweep
    ``data-diagnostic`` precisely because the shipped values are the published
    USE4 constants and no sweep result may change them, so this is a set of
    points a curve is drawn through rather than a set of options.
    """

    #: Trading days, strictly increasing. Validated in :func:`_parse_validation_battery`
    #: for shape and in ``mafrm.factors.halflife_report`` for the two cross-node
    #: facts it cannot see from here: that every point is an integer number of
    #: trading months, and that both shipped half-lives are ON the grid.
    grid: tuple[int, ...]


@dataclass(frozen=True)
class BaselConfig:
    """Basel Committee (1996) traffic light. All four numbers are Basel's own."""

    level: float
    window_days: int
    green_max_breaches: int
    yellow_max_breaches: int


@dataclass(frozen=True)
class BatteryTestsConfig:
    """SPEC.md 6.5's battery. W4-P3, under the five rulings of 2026-09-03.

    Two published lists, Basel's four constants, and five POINTERS to keys that
    already exist. The ``*_from`` fields are documentation of coupling in the
    ``ConditionNumberConfig.halflife_from`` style: the value is read from the key
    they name, never from here, and :func:`_check_pointer` refuses a pointer to a
    key that does not exist so a rename cannot leave one dangling.
    """

    #: Ruling 1: both reported, neither selected.
    var_levels: tuple[float, ...]
    #: Ruling 3: a month, a quarter, a year -- converted with
    #: ``data.trading_days_per_month``, like ``rolling_window_months``.
    ljung_box_lags_months: tuple[int, ...]
    basel: BaselConfig
    #: Ruling 4: the Acerbi-Szekely null is simulated with the eigenfactor ``M``.
    monte_carlo_trials_from: str
    newey_west_lags_from: str
    rank_correlation_block_days_from: str
    t_statistic_threshold_from: str
    t_statistic_window_from: str


@dataclass(frozen=True)
class SecondOrderTauGridConfig:
    """W4-P3's forward registration for W8-P1, run in W8-P1 (SPEC.md 6.4.5).

    The split-window simulation at every point of SPEC.md 5.1.3's grid, with
    the seed varied across points. Every number is a registration written
    before the run; none is a tunable.
    """

    grid_from: str
    #: The registered joint bias at the shortest grid point, in VARIANCE.
    joint_at_shortest_hypothesis: float
    joint_at_shortest_lower: float
    joint_at_shortest_upper: float
    #: Row 177's measured ``rho`` leg, which every grid point must reproduce.
    correlation_leg_registered: float


@dataclass(frozen=True)
class SecondOrderConfig:
    """The W4-P3 disjointness simulation (SPEC.md 6.4, 5.4.2)."""

    monte_carlo_trials_from: str
    #: ``experiments.md`` row 179's registered falsifier. Not a tunable.
    interaction_share_falsifier: float
    #: Rows 175-178's registered agreement band against a closed form.
    closed_form_relative_band: float
    #: Row 109's MEASURED equivalent equal-weight ``T`` for the correlation
    #: estimator; row 177's lower comparand.
    row_109_equivalent_correlation_t: int
    tau_grid: SecondOrderTauGridConfig


@dataclass(frozen=True)
class KtScalingRegistrations:
    """experiments.md row 164 on the equity panel, as registered (W4-P2, boundaries W4-P3)."""

    row_164_macro_c1_share_of_excess: float
    row_164_macro_c1_share_of_naive_gap: float
    row_164_macro_eigen_share_of_excess: float
    row_164_residual_subset: str


@dataclass(frozen=True)
class KtScalingConfig:
    """SPEC.md 15.1's K/T scaling chart (W8-P1), under the operator rulings of 2026-09-06.

    Every field is a reading of a ruling, a pointer to an existing key, or a
    registered baseline. The tracking test uses ``validation.chi_square_level``'s
    exact interval and adds no constant of its own.
    """

    family4_variant: str
    #: Model -> ``"track"`` or ``"not_track"``, the registered direction per cluster.
    expected_tracking: dict[str, str]
    factor_comparand_window: str
    naive_comparand_window: str
    x_axis: str
    interval_periods: str
    interval_level_from: str
    subperiod_block_months_from: str
    equity_sweep_grid_from: str
    equity_sweep_horizon: str
    book_sweep_reversing_halflife: int
    book_sweep_reference_bias: float
    registrations: KtScalingRegistrations


@dataclass(frozen=True)
class ValidationBatteryConfig:
    """SPEC.md 6.1 and 6.2. The bias statistic and the four portfolio families.

    Named ``...Battery`` because :class:`ValidationLadderConfig` and
    ``factors.validation`` already exist and mean something else -- W2-P3's
    external comparand table. This one is the risk-model validation of SPEC.md 6.
    """

    #: SPEC.md 6.1's ``+-4`` clip on standardized returns. Applies to this
    #: statistic only; SPEC.md 5.4's cross-sectional bias is deliberately
    #: unclipped and does not read this (SPEC.md 5.4.1 ruling 2).
    standardized_return_clip: float
    #: SPEC.md 6.1's rolling window, in MONTHS. Converted with
    #: ``data.trading_days_per_month`` rather than restated in days.
    rolling_window_months: int
    #: SPEC.md 6.1's ``1 +- z/sqrt(T)``. A DISPLAY CONVENTION -- see the config
    #: comment and SPEC.md 6.1.1: the chi-square interval is the one that decides.
    normal_band_z: float
    #: Coverage of the exact interval, read off ``normal_band_z``'s own 95%.
    chi_square_level: float
    #: SPEC.md 6.2 family 2. Drawn from the top-level ``seed``.
    random_portfolios: int
    #: SPEC.md 5.1's half-life sweep. W4-P2b.
    halflife_sensitivity: HalflifeSensitivityConfig
    #: SPEC.md 6.5's battery. W4-P3.
    battery: BatteryTestsConfig
    #: The disjointness simulation. W4-P3.
    second_order: SecondOrderConfig
    kt_scaling: KtScalingConfig


@dataclass(frozen=True)
class WindowHalflife:
    window: int
    halflife: int


@dataclass(frozen=True)
class LaggedWindowHalflife:
    window: int
    halflife: int
    #: Trading days skipped before the window starts. RSTR sums t = lag ... lag+window.
    lag: int


@dataclass(frozen=True)
class ResvolComposite:
    """Residual Volatility composite weights. CNE5 Descriptor Details."""

    dastd: float
    cmra: float
    hsigma: float


@dataclass(frozen=True)
class LiquidityComposite:
    """Liquidity composite weights over share turnover. CNE5 Descriptor Details."""

    stom: float
    stoq: float
    stoa: float


@dataclass(frozen=True)
class LiquidityHorizons:
    """The three turnover horizons in MONTHS of ``data.trading_days_per_month``. CNE5."""

    stom: int
    stoq: int
    stoa: int


@dataclass(frozen=True)
class Winsorization:
    """SPEC.md 15.4.1 ruling 2: +/- ``bound_sd`` equal-weighted sd, one pass, a READING.

    ``sensitivity_bounds_sd`` are the two bounds the report re-runs the
    correlation matrix and VIFs at (a ``data-diagnostic``); the bound itself is
    chosen by convention, not by that measurement.
    """

    bound_sd: float
    sensitivity_bounds_sd: tuple[float, ...]


@dataclass(frozen=True)
class EquityDescriptorRegistrations:
    """experiments.md rows 278-281, as registered. Read by the report, never tuned."""

    row_278_spy_correlation_min: float
    row_279_raw_nlsize_size_min_abs: float
    row_279_raw_resvol_beta_min_abs: float
    row_279_orthogonalized_max_abs: float
    row_280_correlation_move_max: float
    row_280_vif_move_max: float
    row_281_vif_max: float


@dataclass(frozen=True)
class EquityDescriptorConfig:
    """CNE5 descriptor windows and composite weights. SPEC.md section 15.4."""

    beta: WindowHalflife
    momentum: LaggedWindowHalflife
    dastd: WindowHalflife
    hsigma: WindowHalflife
    resvol_composite: ResvolComposite
    liquidity_composite: LiquidityComposite
    liquidity_horizons_months: LiquidityHorizons
    #: CMRA's T = 1 ... cmra_months trailing monthly blocks (CNE5: 12).
    cmra_months: int
    winsorization: Winsorization
    registrations: EquityDescriptorRegistrations


@dataclass(frozen=True)
class EquityUniverseRegistrations:
    """experiments.md rows 265 and 267, as registered. Read by the report, never tuned."""

    row_265_failed_after_retry_max: int
    row_265_empty_expected: int
    row_265_empty_tolerance: int
    row_267_no_data_excess_max: int


@dataclass(frozen=True)
class EquityUniverseConfig:
    """The approximate point-in-time S&P 500 universe. SPEC.md 15.3, rulings at 15.3.1.

    ``min_dollar_adv`` is ``None`` while no published threshold exists (ruling
    1): the screen is implemented and drops nothing until a sourced figure is
    written here. ``membership_count_band`` is the acceptance band on the
    reconstructed member count on every session of the estimation sample.
    """

    constituents_url: str
    changes_url: str
    licence: str
    licence_url: str
    reference_dir: str
    intervals_file: str
    membership_file: str
    min_history_days: int
    min_dollar_adv: float | None
    membership_count_band: tuple[int, int]
    registrations: EquityUniverseRegistrations


@dataclass(frozen=True)
class Row272Registration:
    """experiments.md row 272: the point-in-time join checked on one public cover page."""

    ticker: str
    cik: int
    in_force_on: date
    expected_form: str
    expected_filed: date
    expected_shares: int
    split_ex_date: date
    split_ratio: int
    pre_split_filed: date
    pre_split_shares: int


@dataclass(frozen=True)
class Row276Registration:
    """experiments.md row 276: the predecessor-CIK route checked on one re-incorporated name."""

    ticker: str
    predecessor_cik: int
    in_force_on: date
    expected_filed: date
    #: The cover page's figure (W7-P2b ruling 4); the registration's transcription
    #: error is recorded in experiments.md row 276, not here.
    expected_shares: int


@dataclass(frozen=True)
class EquityShareRegistrations:
    """experiments.md rows 269-272 and 274-277, as registered. Read by the report, never tuned."""

    row_269_current_by_ticker_min_pct: float
    row_269_departed_by_name_min_pct: float
    row_270_approximated_share_band_pct: tuple[float, float]
    row_271_no_cap_first_session_min: int
    row_271_no_cap_last_session_max: int
    row_272: Row272Registration
    row_274_max_dropped_universe_filings: int
    row_275_jumps_inside_band_max: int
    row_276: Row276Registration
    row_277_no_cap_dollar_volume_share_band_pct: tuple[float, float]


@dataclass(frozen=True)
class ConsistencyScreen:
    """SPEC.md 15.4.3 ruling (i): drop a filed count more than ``ratio`` off the name's median.

    ``empty_band`` is the (low, high) band of consecutive-count ratios that the
    2026-09-05 pull found empty among universe names -- ``ratio`` is its
    geometric midpoint -- and that a dataset-marked test asserts empty on every
    re-pull. A jump inside it re-opens the ratio; the ratio is never moved to
    fit.
    """

    ratio: float
    empty_band: tuple[float, float]


@dataclass(frozen=True)
class EquityIndustryRegistrations:
    """experiments.md rows 283-284, as registered. Read by the report, never tuned."""

    row_283_industry_coverage_min_pct: float
    row_283_other_max_pct: float
    row_284_sic_differs_max_pct: float
    row_284_industry_differs_max_pct: float
    row_290_reclassified_cells_band_pct: tuple[float, float]
    row_290_r2_move_max: float
    row_290_t_frequency_move_max_pp: float
    row_290_baseline_r2_weighted: float
    row_290_baseline_t_frequency_pct: Mapping[str, float]


@dataclass(frozen=True)
class EquityIndustryConfig:
    """The per-name SIC code from EDGAR's submissions endpoint. SPEC.md 15.6.1 ruling 2 (W7-P3).

    Two URL templates, the form-type vocabulary of "the first in-sample 10-K",
    and the registrations. The User-Agent is ``equity_shares.user_agent_env``.
    """

    submissions_url: str
    header_url: str
    first_filing_forms: tuple[str, ...]
    #: SPEC.md 15.6.3's point-in-time rule for the industry exposure, one reading.
    point_in_time: str
    registrations: EquityIndustryRegistrations

    def submissions_url_for(self, name: str) -> str:
        return self.submissions_url.format(name=name)

    def header_url_for(self, cik: int, accession: str) -> str:
        return self.header_url.format(
            cik=int(cik), accession=accession, accession_nodash=accession.replace("-", "")
        )


@dataclass(frozen=True)
class EquityRegressionRegistrations:
    """experiments.md rows 286, 288 and 289, as registered. Read by the report, never tuned."""

    row_286_country_market_correlation_min: float
    row_286_gap_sd_ratio_max: float
    row_288_thin_industries_median_max: int
    row_288_style_condition_max: float
    row_288_full_condition_median_max: float
    row_289_t_frequency_min_pct: float


@dataclass(frozen=True)
class EquityRegressionConfig:
    """SPEC.md 15.6's daily WLS cross-section (W7-P3; rulings at 15.6.1).

    ``weight_exponent`` is USE4's 0.5 (w proportional to sqrt(cap), CLAUDE.md
    failure mode 8); ``constraint`` and ``thin_industry_rule`` are readings of
    the rulings held in one place; ``r_squared_target`` and
    ``t_stat_threshold`` are SPEC.md 15.7's; ``identity_tolerance`` is the
    numerical tolerance of the two constraint tests.
    """

    weight_exponent: float
    constraint: str
    thin_industry_rule: str
    thin_industry_report_max_members: int
    r_squared_target: float
    t_stat_threshold: float
    identity_tolerance: float
    registrations: EquityRegressionRegistrations


@dataclass(frozen=True)
class EquityRiskRegistrations:
    """experiments.md rows 291-296, as registered (W7-P4). Read by the report, never tuned."""

    row_291_factors: int
    row_291_family4_pre_eigen_min: float
    row_291_family4_pre_eigen_max: float
    row_291_macro_family4_pre_eigen: float
    row_292_never_present: tuple[str, ...]
    row_293_macro_amplitude: float
    row_293_return_to_one_min_share: float
    row_293_firing_k_over_t_min: float
    row_294_floor_percentile: float
    row_295_random_inside_min_share: float
    row_298_floor_percentile: float
    row_298_baseline_singleton_percentile: float
    row_298_baseline_singleton_weight_share: float
    row_298_baseline_family4_daily: float
    row_298_baseline_family4_monthly: float
    row_298_baseline_b_specific: float
    row_298_baseline_b_factor: float
    row_298_baseline_specified_family2: float
    row_298_baseline_random_inside: float


@dataclass(frozen=True)
class EquityRiskConfig:
    """The equity model through the UNCHANGED risk pipeline (W7-P4; SPEC.md 15.7.1).

    Every field is a reading of an operator ruling or a construction held in one
    place, a definition (``size_buckets``), or a unit. The risk-model numbers
    themselves are the shared ``covariance`` / ``eigenfactor`` /
    ``specific_risk`` / ``validation`` blocks, read through ``RiskConfig``.
    """

    never_present_industries: str
    absent_industry_return: str
    size_buckets: int
    bucket_rule: str
    residual_history: str
    short_history_rule: str
    singleton_treatment: str
    singleton_specific_risk: str
    forecast_grid: str
    missing_return: str
    comparand_subset: str
    basis_points_per_unit: float
    registrations: EquityRiskRegistrations


@dataclass(frozen=True)
class EquityShareConfig:
    """Point-in-time shares outstanding from SEC EDGAR. SPEC.md 15.4.1 (W7-P2a).

    Source locations, the XBRL taxonomy/tag/unit of the cover-page fact, the
    first calendar quarter requested, and the NAME of the environment variable
    carrying the User-Agent EDGAR requires. Nothing here is tuned; the
    point-in-time rule is a single named reading so that the code and the
    config cannot drift apart on it.
    """

    frames_url: str
    taxonomy: str
    tag: str
    unit: str
    index_url: str
    company_tickers_url: str
    #: ``YYYYQq``, the first instant frame requested, inclusive.
    first_period: str
    user_agent_env: str
    point_in_time: str
    registrations: EquityShareRegistrations
    consistency_screen: ConsistencyScreen
    #: Hand tables in the config directory (SPEC.md 15.4.3 rulings (i) and (ii)).
    overrides_file: str
    published_totals_file: str

    def frame_url(self, period: str) -> str:
        """The frames URL for one calendar-quarter instant period, e.g. ``CY2019Q1I``."""
        return self.frames_url.format(
            taxonomy=self.taxonomy, tag=self.tag, unit=self.unit, period=period
        )

    def quarter_index_url(self, year: int, quarter: int) -> str:
        return self.index_url.format(year=year, quarter=quarter)


@dataclass(frozen=True)
class SquareRootPrefactor:
    """Square-root impact prefactor Y, per execution regime. SPEC.md section 7.2."""

    patient: float
    urgent: float


@dataclass(frozen=True)
class GammaTradeSweep:
    """Trading-cost aversion. Swept, not fixed -- the choice must be documented.

    ``sweep_step`` makes the grid a rule rather than a choice made at run time:
    1.0 to 3.0 in half-steps is five points (operator ruling, 2026-09-03,
    W5-P1). Characterising the cost model across them is ``data-diagnostic``;
    selecting one on backtest performance is ``strategy-config`` and counts.
    """

    sweep_min: float
    sweep_max: float
    sweep_step: float

    @property
    def points(self) -> tuple[float, ...]:
        """The grid, inclusive at both ends. Integer arithmetic, so no drift."""
        n = round((self.sweep_max - self.sweep_min) / self.sweep_step)
        return tuple(self.sweep_min + k * self.sweep_step for k in range(n + 1))


@dataclass(frozen=True)
class EdgeSpreadConfig:
    """EDGE effective-spread estimation window. Ardia, Guidotti & Kroencke (2024).

    ``min_observations`` is the estimator's published minimum sample and
    ``require_full_window`` decides whether a window shorter than ``window`` may
    emit an estimate at all. They are separate keys because they answer
    different questions: the first is what the method permits, the second is
    what this project is willing to plot next to a full-window estimate.

    ``signed_estimates`` and ``clip_negative_to_zero`` are likewise separate
    because they are different decisions. The first asks the estimator for its
    actual output rather than its absolute value; the second imposes the
    non-negativity a spread has, once, at the production boundary. Signing
    without clipping is a legitimate configuration -- it is what an averaging
    diagnostic wants. Clipping without signing is not, and is rejected in
    :func:`_parse_costs`, because on an already-non-negative series it is a
    silent no-op that reads as though the bias had been handled.
    """

    window: int
    step: str
    min_observations: int
    require_full_window: bool
    signed_estimates: bool
    clip_negative_to_zero: bool

    @property
    def min_periods(self) -> int:
        """Rows a window must hold before an estimate is emitted."""
        return self.window if self.require_full_window else self.min_observations


@dataclass(frozen=True)
class AdvConfig:
    """Average daily dollar volume. SPEC.md 3.4 -- a MEDIAN, never a mean."""

    window: int


@dataclass(frozen=True)
class CrisisWindow:
    """One episode shaded on reports/spread_vs_vol.png. ``end`` is inclusive."""

    label: str
    start: date
    end: date


@dataclass(frozen=True)
class SpreadVersusVolatilityConfig:
    """Inputs to reports/spread_vs_vol.png. SPEC.md 3.4."""

    realised_volatility_window: int
    minimum_baseline_for_ratio_bps: float
    crisis_windows: tuple[CrisisWindow, ...]


@dataclass(frozen=True)
class SpreadLevelConfig:
    """Where the issuer-disclosed spread levels live and how EDGE's shape joins them.

    SPEC.md 3.4.1: the level is a dated static file, committed, not a loader.
    ``shape_baseline`` names the calm level the EDGE widening is measured from;
    ``"in_sample_median"`` is the only value the parser accepts, because the
    alternative -- a window at the disclosure date -- would read the holdout.
    See ``costs.spread_level`` in ``config/model.yaml`` for the rule.
    """

    file: Path
    shape_baseline: str
    #: Resolution at which the issuer displays the figure, in bp; a point value
    #: in the file is read as a rounding interval of this width. ``None`` = exact.
    display_resolution_bps: float | None


@dataclass(frozen=True)
class TradableProxy:
    """The ETF whose bars supply a synthetic curve point's spread, ADV and volatility.

    A modelling choice, not a universe member (operator ruling 0, 2026-09-03,
    W5-P1). ``caveat`` is carried beside every cost figure for the asset: the
    proxy's cost is the cost of trading the ETF, not the exposure the risk model
    priced.
    """

    asset: str
    ticker: str
    caveat: str


@dataclass(frozen=True)
class CalibrationPoint:
    """One published (participation, cost) pair. SPEC.md 7.2's table."""

    participation_of_adv: float
    cost_bps: float


@dataclass(frozen=True)
class CalibrationRegime:
    source: str
    points: tuple[CalibrationPoint, ...]


@dataclass(frozen=True)
class CalibrationCrossCheck:
    participation_of_adv: float
    expected_cost_bps: float


@dataclass(frozen=True)
class CalibrationAnchors:
    """The published figures the two ``Y`` values are backed out of. SPEC.md 7.2."""

    daily_volatility: float
    patient: CalibrationRegime
    urgent: CalibrationRegime
    cross_check: CalibrationCrossCheck


@dataclass(frozen=True)
class AumGridRule:
    """SPEC.md 10.4's candidate-AUM grid, as a RULE rather than two endpoints.

    Log-spaced between the AUM at which the largest position's trade is
    ``low_participation_of_adv`` of its proxy's ADV and the AUM at which it is
    ``high_participation_of_adv`` (operator ruling, 2026-09-03, W5-P2). The
    endpoints are derived from the ADV data once a portfolio exists --
    :func:`mafrm.costs.capacity.aum_grid_bounds` -- and nothing is invented.
    ``points`` is display resolution: both marked points on the curve are
    closed-form and read off no grid.
    """

    spacing: str
    low_participation_of_adv: float
    high_participation_of_adv: float
    points: int


@dataclass(frozen=True)
class CapacityConfig:
    """SPEC.md 10.4, rules only. No input to the curve lives here (W5-P2).

    ``minimum_net_alpha`` is parameterised and only the spec's ``0.0`` special
    case is accepted -- any other value is refused until one is sourced.
    ``gross_alpha`` is refused unless ``None``: it is swept in W6-P1 under a
    principle written before that session opens, never picked as a point.
    """

    minimum_net_alpha: float
    gross_alpha: None
    aum_grid: AumGridRule


@dataclass(frozen=True)
class CostConfig:
    square_root_prefactor: SquareRootPrefactor
    total_cost_exponent: float
    gamma_trade: GammaTradeSweep
    edge_spread: EdgeSpreadConfig
    flat_spread_assumption_bps: float
    adv: AdvConfig
    spread_vs_volatility: SpreadVersusVolatilityConfig
    #: SPEC.md 7.1 -- commission per one-way trade, proportional. Zero is the
    #: absence of a published figure, not a value.
    commission: float
    #: SPEC.md 7.1's ``c_i``. Zero unless short-sale asymmetry is modelled.
    buy_sell_asymmetry: float
    #: SPEC.md 7.1's ``sigma_i`` window: trailing one year, in trading days.
    volatility_window: int
    spread_level: SpreadLevelConfig
    tradable_proxies: tuple[TradableProxy, ...]
    calibration_anchors: CalibrationAnchors
    capacity: CapacityConfig

    def proxy_for(self, asset_id: str) -> TradableProxy | None:
        for proxy in self.tradable_proxies:
            if proxy.asset == asset_id:
                return proxy
        return None


@dataclass(frozen=True)
class NoLookaheadConfig:
    """SPEC.md 11's mechanical no-look-ahead test. ``model.backtest.no_lookahead``."""

    #: SPEC.md 11: "across >= 50 values of t". The SPEC's constant, not a choice.
    minimum_evaluation_dates: int


@dataclass(frozen=True)
class ReconciliationConfig:
    """SPEC.md 7.4's engine cross-check against ``bt``. ``model.backtest.reconciliation``."""

    #: bt's own default initial capital. A normalisation; see config/model.yaml.
    initial_nav: float
    #: Size-independent roundings per asset per accounting step -- the ``c`` of
    #: ``c * eps * T`` is ``2 * (assets + accounting_roundings)``.
    accounting_roundings: int
    #: The task's fixed strategy on the two synthetic assets.
    synthetic_weights: tuple[float, ...]
    #: The real-cache strategy. Only ``"equal_weight"`` is accepted.
    real_strategy: str

    def coefficient(self, assets: int) -> int:
        """``c = 2 * (assets + accounting_roundings)``: both engines round."""
        if assets < 1:
            raise ConfigError(f"reconciliation coefficient: assets must be >= 1, got {assets}")
        return 2 * (assets + self.accounting_roundings)

    def tolerance(self, *, assets: int, steps: int) -> float:
        """``c * eps * T``, relative on NAV. ``steps`` is the daily step count ``T``."""
        if steps < 1:
            raise ConfigError(f"reconciliation tolerance: steps must be >= 1, got {steps}")
        eps: float = sys.float_info.epsilon  # numpy.finfo(float64).eps, without the import
        return self.coefficient(assets) * eps * steps


@dataclass(frozen=True)
class FinancingConfig:
    """The financing leg, operator ruling 1 of W6-P3 (SPEC.md 9.3). ``model.backtest.financing``.

    Cash earns and pays the risk-free rate both ways with no spread. On the
    engine's EXCESS-return price panel the accrual is ``rf - rf = 0`` by
    identity, so the engine carries cash at zero and the leg is REPORTED in
    nominal bp/yr per cell. Only the ruled convention is accepted.
    """

    #: Only ``"risk_free_both_ways"``.
    convention: str
    #: Only ``"data.risk_free"`` -- the Ken French daily bill rate the excess returns use.
    rate_source: str
    #: Only ``0.0``: a spread over RF would be an invented number (CLAUDE.md invariant 9).
    borrowing_spread: float


@dataclass(frozen=True)
class BacktestConfig:
    """``model.backtest``: SPEC.md 11's test constant and SPEC.md 7.4's reconciliation rules."""

    no_lookahead: NoLookaheadConfig
    reconciliation: ReconciliationConfig
    financing: FinancingConfig


@dataclass(frozen=True)
class OptimizerAlphaConfig:
    """SPEC.md 8.5.1 ruling 1: alpha-hat is RSTR, fixed, in return units. ``optimizer.alpha``."""

    #: Only ``"rstr"``. SPEC.md 15.4's momentum descriptor.
    construction: str
    #: Resolved from ``equity_descriptors.momentum`` through the ``*_from`` pointers.
    window: int
    halflife: int
    lag: int
    #: Only ``"normalised"``: the weights sum to one, so RSTR is a rate (return/day).
    weights: str
    #: Only ``"demean"``: equal-weighted centring, NO scaling to unit variance.
    cross_sectional: str
    #: Days per rebalance period; the rate is scaled to it. ``data.trading_days_per_month``.
    horizon_days: int


@dataclass(frozen=True)
class OptimizerTrackingErrorTarget:
    """Ruling 3: a MEASURED anchor times a multiple. ``model.optimizer.tracking_error_target``."""

    #: Only ``"equal_weight_realised_volatility"``.
    anchor: str
    #: The W6-P2 band. Strictly increasing, positive.
    multiples: tuple[float, ...]
    #: The one W6-P1 runs; must be on ``multiples``.
    verification_multiple: float


@dataclass(frozen=True)
class OptimizerRobustification:
    """SPEC.md 8.2's two amplitudes. Ruling 4: both zero, as absences."""

    return_forecast_rho: float
    covariance_varrho: float


@dataclass(frozen=True)
class OptimizerConstraint:
    """One limit and how it is imposed. ``priority is None`` means HARD."""

    name: str
    #: ``None`` switches the constraint off entirely.
    bound: float | None
    #: ``None`` -> hard constraint whose multiplier is observed; a number -> hinge
    #: penalty ``priority * pos(violation)`` in the objective (SPEC.md 8.1).
    priority: float | None

    @property
    def active(self) -> bool:
        return self.bound is not None

    @property
    def hard(self) -> bool:
        return self.active and self.priority is None


@dataclass(frozen=True)
class OptimizerConstraintsConfig:
    """Ruling 5. ``model.optimizer.constraints``."""

    long_only: bool
    fully_invested: bool
    adv_participation: OptimizerConstraint
    position_box: OptimizerConstraint
    turnover: OptimizerConstraint
    tracking_error: OptimizerConstraint
    #: SPEC.md 8.1: priorities from the observed multipliers "at around the 80th percentile".
    lagrange_percentile: float

    @property
    def limits(self) -> tuple[OptimizerConstraint, ...]:
        return (self.adv_participation, self.position_box, self.turnover, self.tracking_error)

    def by_name(self, name: str) -> OptimizerConstraint:
        for limit in self.limits:
            if limit.name == name:
                return limit
        raise ConfigError(f"optimizer.constraints: no constraint named {name!r}")


@dataclass(frozen=True)
class OptimizerSolverConfig:
    primary: str
    fallback: str
    #: The primary solver's own default feasibility tolerance, mirrored so that
    #: "binding" means "within the solver's tolerance of the bound" -- the
    #: comparand's default, not a chosen number (the bt ``initial_nav`` pattern).
    feasibility_tolerance: float


@dataclass(frozen=True)
class OptimizerBookSize:
    """The NAV the impact term and the ADV cap need. SPEC.md 8.5.1's third construction."""

    #: ``"aum_grid_endpoints_equal_weight"`` (a band from the ruled grid) or
    #: ``"fixed"`` (a ruled ``nav_dollars``). Never both.
    rule: str
    nav_dollars: float | None


@dataclass(frozen=True)
class OptimizerVerificationConfig:
    """Ruling 6: ONE verification configuration. ``model.optimizer.verification``."""

    alpha: str
    cost_regime: str
    eigenfactor_scaling: float
    horizon: str
    #: ``"eigen"`` (pre-VRA, SPEC.md 6.2.2), ``"vra"`` or ``"specified"``.
    risk_variant: str
    gamma_trade: float
    tracking_error_multiple: float
    book_size: OptimizerBookSize
    spread_ends: tuple[str, ...]
    control_cost_free: bool


@dataclass(frozen=True)
class OptimizerGridBookSize:
    """The reference cell's NAV. ``model.optimizer.grid.book_size``."""

    #: ``"aum_grid_equal_weight_at_anchor"`` (``A = p N min_i median V_i`` at the
    #: cost model's lower calibration anchor) or ``"fixed"`` (a ruled ``nav_dollars``).
    rule: str
    participation_of_adv: float | None
    nav_dollars: float | None


@dataclass(frozen=True)
class OptimizerGridBandsConfig:
    """W6-P3's one-dimensional bands off the reference cell. ``model.optimizer.grid.bands``.

    Every list here is the band LESS the reference's own value, derived at parse
    time from the rules the reference reads: the bands never restate a number.
    """

    reference_variant: str
    reference_treatment: str
    reference_regime: str
    #: The issuer half-spread ends the reference does not run at.
    spread_ends: tuple[str, ...]
    #: ``optimizer.tracking_error_target.multiples`` less the reference multiple.
    tracking_error_multiples: tuple[float, ...]
    #: Only ``"aum_grid_endpoints_equal_weight"``: A_low and A_high, W6-P1's rule.
    book_size_rule: str
    #: ``costs.gamma_trade``'s five points less the reference's.
    gamma_trades: tuple[float, ...]
    #: The per-asset bound as a multiple of ``1/N`` (SPEC.md 9.1 ruling 5: 2).
    position_box_equal_weight_multiple: int
    #: Horizons other than the reference's.
    horizons: tuple[str, ...]
    #: Risk-aversion modes other than the reference's.
    risk_aversions: tuple[str, ...]

    def position_box(self, assets: int) -> float:
        """``multiple / N``: the ruled bound, derived from the universe size."""
        if assets < 1:
            raise ConfigError(f"position_box: assets must be >= 1, got {assets}")
        return self.position_box_equal_weight_multiple / assets


@dataclass(frozen=True)
class OptimizerGridResolveConfig:
    """The ``optimal_inaccurate`` re-solve. ``model.optimizer.grid.resolve``."""

    #: Only ``"fallback"``: the same problem, the other solver.
    solver: str
    #: Only ``"max"`` (operator ruling 6, W6-P3): the worst date decides.
    criterion: str
    #: The L1 distance above which a cell re-runs as new rows; registered at SPEC.md 9.2.
    l1_threshold: float


@dataclass(frozen=True)
class OptimizerGridCapacityConfig:
    """SPEC.md 10.4's re-optimisation. ``model.optimizer.grid.capacity``."""

    aum_grid_from: str
    #: Only ``"hard"``: the ADV cap is imposed as a constraint, not a hinge.
    adv_participation: str
    relaxation_ladder: tuple[str, ...]
    regimes: tuple[str, ...]
    #: Only ``False`` (operator ruling 4): a Sharpe per AUM point makes every point a trial.
    report_sharpe: bool
    rescaling_comparand: str


@dataclass(frozen=True)
class OptimizerGridDiagnosticsConfig:
    """SPEC.md 10.1-10.3's diagnostics. ``model.optimizer.grid.diagnostics``."""

    decision_price: str
    residual: str
    return_identity: str
    variance_test: str
    interval_from: str
    #: Only ``"equal_weight"`` (operator ruling 3).
    brinson_benchmark: str
    multi_period_linking: str


@dataclass(frozen=True)
class OptimizerGridConfig:
    """SPEC.md 9's grid under the W6-P2 rulings (SPEC.md 9.1). ``model.optimizer.grid``."""

    horizon: str
    alpha: str
    eigenfactor_scaling: float
    gamma_trade: float
    tracking_error_multiple: float
    spread_end: str
    #: ``"tracking_error_constraint"`` (SPEC.md 8.4's second step) or ``"spec_initialisation"``.
    risk_aversion: str
    #: ``"recovered_multiplier_two_pass"`` or ``"spec_initialisation"``.
    misalignment_pricing: str
    book_size: OptimizerGridBookSize
    covariance_variants: tuple[str, ...]
    dense_window_rule: str
    #: DERIVED: ``round(2 tau / ln 2)`` at the horizon's factor volatility half-life.
    dense_window: int
    cost_treatments: tuple[str, ...]
    #: The FULL spread behind treatment C, in basis points; the half-spread is half of it.
    flat_full_spread_bps: float
    time_varying_regimes: tuple[str, ...]
    cost_free_drops_adv_cap: bool
    hinge_priorities: Mapping[str, float]
    relaxation_ladder: tuple[str, ...]
    trial_count_source: str
    #: DERIVED: Lo (2002)'s ``q``, periods per year at the rebalance horizon.
    sharpe_aggregation_periods: int
    #: W6-P3 (SPEC.md 9.3).
    bands: OptimizerGridBandsConfig
    resolve: OptimizerGridResolveConfig
    capacity: OptimizerGridCapacityConfig
    diagnostics: OptimizerGridDiagnosticsConfig

    @property
    def flat_half_spread(self) -> float:
        """Treatment C's ``a_i``, proportional: half the full spread (ruling 7, W6-P2)."""
        return 0.5 * self.flat_full_spread_bps * 1e-4


@dataclass(frozen=True)
class OptimizerConfig:
    """``model.optimizer``: SPEC.md 8 under the W6-P1 rulings (SPEC.md 8.5)."""

    alpha: OptimizerAlphaConfig
    tracking_error_target: OptimizerTrackingErrorTarget
    #: ``None`` -> SPEC.md 8.4's ``IR / (2 TE_target)`` at each rebalance; a float pins it.
    gamma_risk: float | None
    robustification: OptimizerRobustification
    gamma_hold: float
    #: ``"msci"`` or ``"none"``.
    misalignment_penalty: str
    #: Only ``"zero"``: absolute risk.
    benchmark: str
    constraints: OptimizerConstraintsConfig
    #: Hard constraints in the order they are dropped on solver failure; then prior weights.
    relaxation_ladder: tuple[str, ...]
    solver: OptimizerSolverConfig
    verification: OptimizerVerificationConfig
    grid: OptimizerGridConfig


@dataclass(frozen=True)
class ModelConfig:
    """Everything in ``config/model.yaml``."""

    schema_version: int
    seed: int
    sample: SampleConfig
    data: DataConfig
    numerics: NumericsConfig
    factors: FactorsConfig
    covariance: CovarianceConfig
    volatility_regime_adjustment: VolatilityRegimeAdjustmentConfig
    eigenfactor: EigenfactorConfig
    specific_risk: SpecificRiskConfig
    validation: ValidationBatteryConfig
    equity_descriptors: EquityDescriptorConfig
    equity_universe: EquityUniverseConfig
    equity_shares: EquityShareConfig
    equity_industries: EquityIndustryConfig
    equity_regression: EquityRegressionConfig
    equity_risk: EquityRiskConfig
    costs: CostConfig
    backtest: BacktestConfig
    optimizer: OptimizerConfig


def _parse_sample(root: dict[str, Any]) -> SampleConfig:
    node = _child(root, "sample", "model")
    sample = SampleConfig(
        start=_date(node, "start", "model.sample"),
        holdout_start=_opt_date(node, "holdout_start", "model.sample"),
        holdout_end=_opt_date(node, "holdout_end", "model.sample"),
    )
    if sample.holdout_start is not None and sample.holdout_start <= sample.start:
        raise ConfigError("model.sample.holdout_start must be after model.sample.start")
    if (
        sample.holdout_start is not None
        and sample.holdout_end is not None
        and sample.holdout_end <= sample.holdout_start
    ):
        raise ConfigError("model.sample.holdout_end must be after model.sample.holdout_start")
    return sample


def _parse_curve(node: dict[str, Any], key: str, path: str) -> CurveConfig:
    child = _child(node, key, path)
    where = f"{path}.{key}"
    maturities = _float_list(child, "maturities_years", where)
    if not maturities or any(m <= 0 for m in maturities):
        raise ConfigError(f"{where}.maturities_years: expected positive maturities")
    return CurveConfig(
        url=_str(child, "url", where),
        yield_prefix=_str(child, "yield_prefix", where),
        maturities_years=maturities,
    )


def _parse_etf(node: dict[str, Any]) -> EtfConfig:
    where = "model.data.etf"
    child = _child(node, "etf", "model.data")
    # CLAUDE.md invariant 2 is enforced at the config boundary as well as at the
    # call site: a "1wk" written here would otherwise reach yfinance as a default.
    interval = _str(child, "interval", where)
    if interval != "1d":
        raise ConfigError(
            f"{where}.interval must be '1d' (CLAUDE.md invariant 2: Yahoo mis-applies "
            f"dividends to '1wk' and '1mo' bars), got {interval!r}"
        )
    if _bool(child, "auto_adjust", where):
        raise ConfigError(
            f"{where}.auto_adjust must be false. SPEC.md 3.5: adjusted closes are "
            "rewritten backwards, so this project caches raw bars and adjusts itself."
        )
    return EtfConfig(
        interval=interval,
        auto_adjust=False,
        period=_str(child, "period", where),
        price_columns=_str_list(child, "price_columns", where),
        action_columns=_str_list(child, "action_columns", where),
        required_response_columns=_str_list(child, "required_response_columns", where),
    )


def _series_classification(node: dict[str, Any], path: str) -> SeriesClassification:
    value = _str(node, "classification", path)
    if value == "market_observed":
        return "market_observed"
    if value == "revised_statistic":
        return "revised_statistic"
    raise ConfigError(
        f"{path}.classification: expected 'market_observed' or 'revised_statistic', "
        f"got {value!r}. The distinction decides whether ALFRED vintages are "
        "required, so it must be stated rather than defaulted."
    )


def _parse_fred(node: dict[str, Any]) -> FredConfig:
    where = "model.data.fred"
    child = _child(node, "fred", "model.data")
    base_url = _str(child, "api_base_url", where)
    if "fredgraph" in base_url:
        raise ConfigError(
            f"{where}.api_base_url points at the keyless fredgraph CSV endpoint, which "
            "serves only the current vintage. ALFRED vintages need the API (SPEC.md 3.5)."
        )
    chunk_size = _int(child, "vintage_chunk_size", where)
    if chunk_size <= 0:
        raise ConfigError(f"{where}.vintage_chunk_size must be positive, got {chunk_size}")
    raw_series = _require(child, "series", where)
    if not isinstance(raw_series, list) or not raw_series:
        raise ConfigError(f"{where}.series: expected a non-empty list")

    series: list[FredSeries] = []
    seen: set[str] = set()
    for index, item in enumerate(raw_series):
        path = f"{where}.series[{index}]"
        entry = _as_mapping(item, path)
        series_id = _str(entry, "id", path)
        if series_id in seen:
            raise ConfigError(f"{where}.series: duplicate id {series_id!r}")
        seen.add(series_id)
        series.append(
            FredSeries(
                id=series_id,
                name=_str(entry, "name", path),
                description=_str(entry, "description", path),
                classification=_series_classification(entry, path),
            )
        )
    return FredConfig(
        api_base_url=base_url.rstrip("/"),
        api_key_env_var=_str(child, "api_key_env_var", where),
        vintage_realtime_start=_date(child, "vintage_realtime_start", where),
        # Kept as text: FRED's documented upper bound is 9999-12-31, which is
        # outside pandas' representable range and is a sentinel, not a date.
        vintage_realtime_end=_str(child, "vintage_realtime_end", where),
        vintage_chunk_size=chunk_size,
        series=tuple(series),
    )


def _parse_aqr(node: dict[str, Any]) -> AqrConfig:
    where = "model.data.aqr"
    child = _child(node, "aqr", "model.data")
    raw_datasets = _child(child, "datasets", where)
    datasets: list[AqrDataset] = []
    for name in sorted(raw_datasets):
        path = f"{where}.datasets.{name}"
        entry = _as_mapping(raw_datasets[name], path)
        header_row = _int(entry, "header_row", path)
        if header_row < 0:
            raise ConfigError(f"{path}.header_row must be non-negative, got {header_row}")
        datasets.append(
            AqrDataset(
                name=name,
                url=_str(entry, "url", path),
                description=_str(entry, "description", path),
                sheet=_str(entry, "sheet", path),
                header_row=header_row,
                date_column=_opt_str(entry, "date_column", path),
                required_columns=_str_list(entry, "required_columns", path),
                expected_to_update=_bool(entry, "expected_to_update", path),
            )
        )
    if not datasets:
        raise ConfigError(f"{where}.datasets: expected at least one workbook")
    return AqrConfig(
        user_agent_is_browser=_bool(child, "user_agent_is_browser", where),
        datasets=tuple(datasets),
    )


def _parse_cross_checks(node: dict[str, Any]) -> CrossCheckConfig:
    where = "model.data.cross_checks"
    child = _child(node, "cross_checks", "model.data")

    spy = _child(child, "spy_versus_market", where)
    spy_where = f"{where}.spy_versus_market"
    correlation = _float(spy, "min_correlation", spy_where)
    if not 0.0 < correlation < 1.0:
        raise ConfigError(f"{spy_where}.min_correlation must lie in (0, 1)")
    gap = _float(spy, "max_abs_mean_gap_pct_per_year", spy_where)
    daily = _float(spy, "max_abs_daily_difference_pct", spy_where)
    if gap <= 0 or daily <= 0:
        raise ConfigError(f"{spy_where}: tolerances must be positive")

    tlt = _child(child, "tlt_versus_ladder", where)
    return CrossCheckConfig(
        spy_versus_market=SpyVersusMarketConfig(
            ticker=_str(spy, "ticker", spy_where),
            benchmark_column=_str(spy, "benchmark_column", spy_where),
            min_correlation=correlation,
            max_abs_mean_gap_pct_per_year=gap,
            max_abs_daily_difference_pct=daily,
        ),
        tlt_versus_ladder=TltVersusLadderConfig(
            ticker=_str(tlt, "ticker", f"{where}.tlt_versus_ladder")
        ),
    )


def _parse_stooq(node: dict[str, Any]) -> StooqConfig:
    where = "model.data.stooq"
    child = _child(node, "stooq", "model.data")
    url_template = _str(child, "url_template", where)
    if "{symbol}" not in url_template:
        raise ConfigError(f"{where}.url_template must contain {{symbol}}")
    symbol_template = _str(child, "symbol_template", where)
    if "{ticker}" not in symbol_template:
        raise ConfigError(f"{where}.symbol_template must contain {{ticker}}")
    return StooqConfig(
        url_template=url_template,
        symbol_template=symbol_template,
        cross_check_tickers=_str_list(child, "cross_check_tickers", where),
    )


def _parse_data(root: dict[str, Any]) -> DataConfig:
    node = _child(root, "data", "model")
    trading_days = _int(node, "trading_days_per_year", "model.data")
    if trading_days <= 0:
        raise ConfigError("model.data.trading_days_per_year must be positive")
    trading_days_month = _int(node, "trading_days_per_month", "model.data")
    if trading_days_month <= 0:
        raise ConfigError("model.data.trading_days_per_month must be positive")
    # The two calendar constants must agree, and the check is here rather than
    # in a comment because they are 40 lines apart in the YAML and a later edit
    # to one would not obviously implicate the other.
    if trading_days % 12 != 0 or trading_days_month != trading_days // 12:
        raise ConfigError(
            f"model.data: trading_days_per_month ({trading_days_month}) must be "
            f"trading_days_per_year / 12 ({trading_days} / 12). They are the same calendar "
            "expressed two ways and a horizon scaling built from a mismatched pair would be "
            "silently wrong in every consumer."
        )

    rf_node = _child(node, "risk_free", "model.data")
    kf_node = _child(rf_node, "ken_french_daily_rf", "model.data.risk_free")
    fred_node = _child(rf_node, "interim_fred", "model.data.risk_free")
    url_template = _str(fred_node, "url_template", "model.data.risk_free.interim_fred")
    if "{series_id}" not in url_template:
        raise ConfigError("model.data.risk_free.interim_fred.url_template must contain {series_id}")
    selected = _str(rf_node, "selected", "model.data.risk_free")
    if selected not in rf_node:
        raise ConfigError(f"model.data.risk_free.selected names {selected!r}, which is not a key")
    # The units string is validated rather than carried, because it is the one
    # field whose misreading is silent: Ken French's RF is percent per TRADING
    # DAY and DGS1MO is percent per ANNUM, a factor of ~252 apart, and both look
    # like ordinary small numbers. mafrm.data.french.as_annual_percent converts.
    kf_units = _str(kf_node, "units", "model.data.risk_free.ken_french_daily_rf")
    if kf_units != "percent_per_trading_day":
        raise ConfigError(
            "model.data.risk_free.ken_french_daily_rf.units: only "
            f"'percent_per_trading_day' is understood, got {kf_units!r}. A different unit "
            "needs a different conversion in mafrm.data.french, not a different string here."
        )

    ladder_node = _child(node, "validation_ladder", "model.data")
    ladder_maturities = _float_list(ladder_node, "maturities_years", "model.data.validation_ladder")
    ladder_weights = _float_list(ladder_node, "weights", "model.data.validation_ladder")
    if len(ladder_maturities) != len(ladder_weights):
        raise ConfigError(
            "model.data.validation_ladder: maturities_years and weights must be the same length"
        )
    if not ladder_maturities or any(w < 0 for w in ladder_weights) or sum(ladder_weights) <= 0:
        raise ConfigError("model.data.validation_ladder: weights must be non-negative and positive")

    tlt_node = _child(node, "tlt_cross_check", "model.data")
    min_correlation = _float(tlt_node, "min_correlation", "model.data.tlt_cross_check")
    if not 0.0 < min_correlation < 1.0:
        raise ConfigError("model.data.tlt_cross_check.min_correlation must lie in (0, 1)")

    return DataConfig(
        trading_days_per_year=trading_days,
        trading_days_per_month=trading_days_month,
        svensson_small_n=_float(node, "svensson_small_n", "model.data"),
        curve_tau2_missing_sentinel=_float(node, "curve_tau2_missing_sentinel", "model.data"),
        nominal_curve=_parse_curve(node, "nominal_curve", "model.data"),
        real_curve=_parse_curve(node, "real_curve", "model.data"),
        risk_free=RiskFreeConfig(
            selected=selected,
            ken_french_daily_rf=KenFrenchRiskFree(
                description=_str(
                    kf_node, "description", "model.data.risk_free.ken_french_daily_rf"
                ),
                history_start=_date(
                    kf_node, "history_start", "model.data.risk_free.ken_french_daily_rf"
                ),
                url=_str(kf_node, "url", "model.data.risk_free.ken_french_daily_rf"),
                dataset=_str(kf_node, "dataset", "model.data.risk_free.ken_french_daily_rf"),
                rf_column=_str(kf_node, "rf_column", "model.data.risk_free.ken_french_daily_rf"),
                units=kf_units,
                implemented=_bool(
                    kf_node, "implemented", "model.data.risk_free.ken_french_daily_rf"
                ),
            ),
            interim_fred=InterimFredRiskFree(
                fred_series=_str(fred_node, "fred_series", "model.data.risk_free.interim_fred"),
                url_template=url_template,
                history_start=_date(
                    fred_node, "history_start", "model.data.risk_free.interim_fred"
                ),
                implemented=_bool(fred_node, "implemented", "model.data.risk_free.interim_fred"),
            ),
        ),
        validation_ladder=ValidationLadderConfig(
            maturities_years=ladder_maturities,
            weights=ladder_weights,
            coupon_frequency=_int(ladder_node, "coupon_frequency", "model.data.validation_ladder"),
        ),
        spread_structure_control=tuple(
            _str_list(
                _child(node, "spread_structure_control", "model.data"),
                "tickers",
                "model.data.spread_structure_control",
            )
        ),
        tlt_cross_check=TltCrossCheckConfig(
            ticker=_str(tlt_node, "ticker", "model.data.tlt_cross_check"),
            start=_date(tlt_node, "start", "model.data.tlt_cross_check"),
            min_correlation=min_correlation,
            maturity_years=_float(tlt_node, "maturity_years", "model.data.tlt_cross_check"),
            max_abs_gap_pct_per_year=_float(
                tlt_node, "max_abs_gap_pct_per_year", "model.data.tlt_cross_check"
            ),
            beta_min=_float(tlt_node, "beta_min", "model.data.tlt_cross_check"),
            beta_max=_float(tlt_node, "beta_max", "model.data.tlt_cross_check"),
            published_tracking_difference_pct_per_year=_float(
                tlt_node, "published_tracking_difference_pct_per_year", "model.data.tlt_cross_check"
            ),
            expense_ratio_pct_per_year=_float(
                tlt_node, "expense_ratio_pct_per_year", "model.data.tlt_cross_check"
            ),
            seasoning_sweep_years=_float_list(
                tlt_node, "seasoning_sweep_years", "model.data.tlt_cross_check"
            ),
        ),
        etf=_parse_etf(node),
        fred=_parse_fred(node),
        aqr=_parse_aqr(node),
        cross_checks=_parse_cross_checks(node),
        stooq=_parse_stooq(node),
        ken_french_siccodes49=_parse_siccodes(node),
    )


def _parse_siccodes(node: dict[str, Any]) -> KenFrenchSiccodes:
    sub = _child(node, "ken_french_siccodes49", "model.data")
    path = "model.data.ken_french_siccodes49"
    industries = _int(sub, "industries", path)
    unassigned = _int(sub, "unassigned_industry", path)
    if industries < 2:
        raise ConfigError(f"{path}.industries must be at least 2, got {industries}")
    if not 1 <= unassigned <= industries:
        raise ConfigError(
            f"{path}.unassigned_industry must be one of the {industries} industries, "
            f"got {unassigned}"
        )
    return KenFrenchSiccodes(
        description=_str(sub, "description", path),
        url=_str(sub, "url", path),
        member=_str(sub, "member", path),
        industries=industries,
        unassigned_industry=unassigned,
    )


def _parse_numerics(root: dict[str, Any]) -> NumericsConfig:
    node = _child(root, "numerics", "model")
    ess_node = _child(node, "effective_sample_size", "model.numerics")
    raw_reference = _require(ess_node, "reference", "model.numerics.effective_sample_size")
    ref_path = "model.numerics.effective_sample_size.reference"
    if not isinstance(raw_reference, dict):
        raise ConfigError(f"{ref_path}: expected a mapping, got {type(raw_reference).__name__}")
    # Unlike every other node, this one is keyed by half-life in days, not by name.
    reference: dict[int, int] = {}
    for raw_key, raw_value in raw_reference.items():
        if not isinstance(raw_key, int) or isinstance(raw_key, bool):
            raise ConfigError(f"{ref_path}: half-life keys must be ints, got {raw_key!r}")
        if not isinstance(raw_value, int) or isinstance(raw_value, bool):
            raise ConfigError(f"{ref_path}[{raw_key}]: expected an int, got {raw_value!r}")
        reference[int(raw_key)] = int(raw_value)
    floor = _float(node, "psd_eigenvalue_floor", "model.numerics")
    if floor <= 0.0:
        raise ConfigError("model.numerics.psd_eigenvalue_floor must be positive")
    roundings = _int(node, "psd_reconstruction_roundings", "model.numerics")
    if roundings < 0:
        raise ConfigError(
            "model.numerics.psd_reconstruction_roundings must be non-negative: it counts "
            "floating-point roundings, and the size-dependent term is added to it rather "
            f"than replaced by it. Got {roundings!r}"
        )
    symmetry = _float(node, "symmetry_absolute_tolerance", "model.numerics")
    if symmetry <= 0.0:
        raise ConfigError("model.numerics.symmetry_absolute_tolerance must be positive")
    return NumericsConfig(
        psd_eigenvalue_floor=floor,
        psd_reconstruction_roundings=roundings,
        symmetry_absolute_tolerance=symmetry,
        effective_sample_size=EffectiveSampleSize(
            formula=_str(ess_node, "formula", "model.numerics.effective_sample_size"),
            reference=reference,
        ),
    )


#: PCA forms this project knows how to run. The normalization in SPEC.md 4.1.1
#: is only interpretable on the covariance form -- standardising each tenor to
#: unit variance rescales the long end, whose changes are genuinely smaller,
#: until a loading no longer means basis points.
_PCA_ESTIMATORS: Final[frozenset[str]] = frozenset({"covariance"})

#: Model B's permitted input estimator. SPEC.md 4.2.1 rules out the EWMA
#: correlation on the derivation of Marchenko-Pastur, so the set has one member
#: and the alternative is rejected at load time rather than left reachable.
_STATISTICAL_ESTIMATORS: Final[frozenset[str]] = frozenset({"sample_correlation"})

#: SPEC.md 4.2's two variants. Both are always built; neither is chosen.
_STATISTICAL_VARIANTS: Final[frozenset[str]] = frozenset({"denoised", "detoned"})

#: The residual-PCA estimators implemented. SPEC.md 4.3.3: the CORRELATION, so
#: that a residual panel whose volatilities span an order of magnitude does not
#: return its most volatile member as a "factor" (experiments.md row 96).
_HYBRID_ESTIMATORS: Final[frozenset[str]] = frozenset({"sample_correlation"})

#: Row 83's rules, ruled 2026-08-31. Each is rank- or comparison-based.
_MATERIAL_LOADING_RULES: Final[frozenset[str]] = frozenset({"largest_absolute"})
_HYBRID_COMPARAND_RULES: Final[frozenset[str]] = frozenset({"real_beats_nominal"})
_HYBRID_COMPARAND_TRANSFORMS: Final[frozenset[str]] = frozenset({"daily_change_bps"})

#: The universe ``construction`` whose regressand is the duration leg alone.
_GOVERNMENT_ZERO_CONSTRUCTION: Final[str] = "synthetic_gsw_zero_curve"

#: The only leg of SPEC.md 3.2's three-way decomposition the control names.
_HYBRID_RETURN_LEGS: Final[frozenset[str]] = frozenset({"duration_effect"})

#: Duration-check instrument shapes. Each requires a different identifying key,
#: enforced below rather than discovered as a ``None`` in the report.
_DURATION_CHECK_KINDS: Final[frozenset[str]] = frozenset({"synthetic_zero", "etf_excess"})


def _parse_rate_pca(node: dict[str, Any], path: str) -> RatePcaConfig:
    child = _child(node, "pca", path)
    where = f"{path}.pca"

    estimator = _str(child, "estimator", where)
    if estimator not in _PCA_ESTIMATORS:
        raise ConfigError(
            f"{where}.estimator: {estimator!r} is not implemented; expected one of "
            f"{sorted(_PCA_ESTIMATORS)}. The 1bp loading normalization is defined on the "
            "covariance form only."
        )

    min_window = _int(child, "expanding_min_window", where)
    if min_window < 2:
        raise ConfigError(
            f"{where}.expanding_min_window must be at least 2 for a covariance to exist, "
            f"got {min_window}"
        )

    normalizer_floor = _float(child, "min_normalizer", where)
    if normalizer_floor <= 0.0:
        raise ConfigError(f"{where}.min_normalizer must be positive, got {normalizer_floor}")

    level = _float(child, "level_mean_loading_bps", where)
    slope = _float(child, "slope_loading_difference_bps", where)
    for key, value in (("level_mean_loading_bps", level), ("slope_loading_difference_bps", slope)):
        if value == 0.0:
            raise ConfigError(
                f"{where}.{key} must be nonzero: it is the size the loading is scaled TO, and "
                "zero would ask for a factor with no loading at all"
            )

    return RatePcaConfig(
        estimator=estimator,
        expanding_min_window=min_window,
        level_mean_loading_bps=level,
        slope_loading_difference_bps=slope,
        min_normalizer=normalizer_floor,
    )


def _parse_macro_rates(node: dict[str, Any], path: str) -> MacroRatesConfig:
    child = _child(node, "rates", path)
    where = f"{path}.rates"

    tenors = _float_list(child, "tenors_years", where)
    if len(tenors) < 2:
        raise ConfigError(
            f"{where}.tenors_years: need at least 2 tenors -- one to average for the level "
            f"normalization and two to difference for the slope -- got {list(tenors)}"
        )
    if any(tenor <= 0.0 for tenor in tenors):
        raise ConfigError(f"{where}.tenors_years must all be positive, got {list(tenors)}")
    if list(tenors) != sorted(set(tenors)):
        raise ConfigError(
            f"{where}.tenors_years must be strictly ascending and unique, got {list(tenors)}. "
            "The slope normalization differences the LAST against the FIRST, so the order is "
            "load-bearing rather than cosmetic."
        )

    pca = _parse_rate_pca(child, where)
    if pca.expanding_min_window <= len(tenors):
        raise ConfigError(
            f"{where}.pca.expanding_min_window ({pca.expanding_min_window}) must exceed the "
            f"number of tenors ({len(tenors)}), or the covariance matrix is singular by "
            "construction and its eigenvectors are arbitrary"
        )
    return MacroRatesConfig(tenors_years=tenors, pca=pca)


def _parse_asset_factor(node: dict[str, Any], key: str, path: str) -> MacroAssetFactorConfig:
    child = _child(node, key, path)
    where = f"{path}.{key}"
    return MacroAssetFactorConfig(
        asset_id=_str(child, "asset_id", where),
        excess_over=_str(child, "excess_over", where),
    )


def _parse_orthogonalization(node: dict[str, Any], path: str) -> OrthogonalizationConfig:
    child = _child(node, "orthogonalization", path)
    where = f"{path}.orthogonalization"

    order = _str_list(child, "order", where)
    if len(set(order)) != len(order):
        raise ConfigError(f"{where}.order repeats a factor: {list(order)}")

    raw_against = _child(child, "against", where)
    against: dict[str, tuple[str, ...]] = {}
    for target in raw_against:
        target_path = f"{where}.against.{target}"
        if not isinstance(target, str):
            raise ConfigError(f"{where}.against: keys must be factor names, got {target!r}")
        if target not in order:
            raise ConfigError(f"{target_path}: {target!r} is not in {where}.order")
        names = _str_list(raw_against, target, f"{where}.against")
        if not names:
            raise ConfigError(
                f"{target_path}: an empty regressor list is not 'leave it alone' -- omit the key"
            )
        if target in names:
            raise ConfigError(f"{target_path}: cannot orthogonalize {target!r} against itself")
        if len(set(names)) != len(names):
            raise ConfigError(f"{target_path} repeats a regressor: {list(names)}")
        against[target] = names

    # A regressor must be settled by the time it is used: either it comes earlier
    # in the processing order, or it is never orthogonalized at all. This is what
    # makes `commodity` regressed on the later-ordered `dollar` unambiguous, and
    # it is the check that would fire if a later session added `dollar` to
    # `against` without moving it in `order`.
    position = {name: index for index, name in enumerate(order)}
    for target, names in against.items():
        for name in names:
            if name not in position:
                raise ConfigError(f"{where}.against.{target}: {name!r} is not in {where}.order")
            if position[name] >= position[target] and name in against:
                raise ConfigError(
                    f"{where}.against.{target}: {name!r} is itself orthogonalized and does not "
                    f"precede {target!r} in {where}.order, so which version of it to use is "
                    "undefined. Move it earlier in the order or drop its own `against` entry."
                )

    min_window = _int(child, "expanding_min_window", where)
    if min_window < 2:
        raise ConfigError(
            f"{where}.expanding_min_window must be at least 2 to fit a slope, got {min_window}"
        )

    # A pointer, and deliberately a closed set of two. The alternative is NAMED
    # rather than deleted, so that switching back is a config edit with a row in
    # experiments.md rather than a code change nobody logs.
    window_start = _str(child, "expanding_window_start", where)
    if window_start not in _ORTHOGONALIZATION_WINDOW_STARTS:
        raise ConfigError(
            f"{where}.expanding_window_start must be one of "
            f"{sorted(_ORTHOGONALIZATION_WINDOW_STARTS)}, got {window_start!r}. It names where "
            "the expanding window opens, not a date -- see experiments.md row 69."
        )

    fit_intercept = _bool(child, "fit_intercept", where)
    subtract_intercept = _bool(child, "subtract_intercept", where)
    if subtract_intercept and not fit_intercept:
        raise ConfigError(
            f"{where}: subtract_intercept is true while fit_intercept is false, so there is no "
            "intercept to subtract"
        )

    limit = _float(child, "targeted_pair_max_abs_correlation", where)
    if not 0.0 < limit <= 1.0:
        raise ConfigError(
            f"{where}.targeted_pair_max_abs_correlation must be in (0, 1], got {limit}"
        )

    raw_exempt = _require(child, "exempt_pairs", where)
    if not isinstance(raw_exempt, list):
        raise ConfigError(f"{where}.exempt_pairs: expected a list (write [] for none)")
    exempt: list[ExemptPair] = []
    for index, entry in enumerate(raw_exempt):
        item_path = f"{where}.exempt_pairs[{index}]"
        item = _as_mapping(entry, item_path)
        pair = ExemptPair(
            target=_str(item, "target", item_path),
            regressor=_str(item, "regressor", item_path),
            reason=_str(item, "reason", item_path),
            owed=_str(item, "owed", item_path),
        )
        # An exemption is only meaningful for a pair the scheme actually targets;
        # exempting an untargeted pair would be claiming the gate covers ground
        # SPEC.md 4.1.2 explicitly took away from it.
        if pair.regressor not in against.get(pair.target, ()):
            raise ConfigError(
                f"{item_path}: {pair.target!r} is not orthogonalized against {pair.regressor!r}, "
                f"so there is nothing to exempt. Only targeted pairs are gated at all."
            )
        if not pair.reason.strip() or not pair.owed.strip():
            raise ConfigError(
                f"{item_path}: an exemption must carry a reason and the follow-up it owes. "
                "An unexplained exemption is a lowered threshold wearing a different name."
            )
        exempt.append(pair)

    return OrthogonalizationConfig(
        order=order,
        against=against,
        expanding_min_window=min_window,
        expanding_window_start=window_start,
        fit_intercept=fit_intercept,
        subtract_intercept=subtract_intercept,
        targeted_pair_max_abs_correlation=limit,
        exempt_pairs=tuple(exempt),
    )


def _parse_condition_number(node: dict[str, Any], path: str) -> ConditionNumberConfig:
    child = _child(node, "condition_number", path)
    where = f"{path}.condition_number"
    minimum = _int(child, "min_observations", where)
    if minimum < 2:
        raise ConfigError(f"{where}.min_observations must be at least 2, got {minimum}")
    return ConditionNumberConfig(
        halflife_from=_str(child, "halflife_from", where),
        min_observations=minimum,
        crisis_windows_from=_str(child, "crisis_windows_from", where),
    )


def _parse_duration_check(node: dict[str, Any], path: str) -> DurationCheckConfig:
    child = _child(node, "duration_check", path)
    where = f"{path}.duration_check"

    tolerance = _float(child, "tolerance_fraction", where)
    if not 0.0 < tolerance < 1.0:
        raise ConfigError(
            f"{where}.tolerance_fraction is a fraction of the prediction and must be in (0, 1), "
            f"got {tolerance}"
        )

    raw = _require(child, "instruments", where)
    if not isinstance(raw, list) or not raw:
        raise ConfigError(f"{where}.instruments: expected a non-empty list")

    instruments: list[DurationCheckInstrument] = []
    seen: set[str] = set()
    for index, entry in enumerate(raw):
        item = _as_mapping(entry, f"{where}.instruments[{index}]")
        item_path = f"{where}.instruments[{index}]"
        identifier = _str(item, "id", item_path)
        if identifier in seen:
            raise ConfigError(f"{item_path}: duplicate instrument id {identifier!r}")
        seen.add(identifier)

        kind = _str(item, "kind", item_path)
        if kind not in _DURATION_CHECK_KINDS:
            raise ConfigError(
                f"{item_path}.kind: {kind!r} is not one of {sorted(_DURATION_CHECK_KINDS)}"
            )
        has_maturity = "maturity_years" in item
        maturity = _opt_float(item, "maturity_years", item_path) if has_maturity else None
        ticker = _opt_str(item, "ticker", item_path) if "ticker" in item else None
        if kind == "synthetic_zero" and maturity is None:
            raise ConfigError(f"{item_path}: kind 'synthetic_zero' needs maturity_years")
        if kind == "etf_excess" and ticker is None:
            raise ConfigError(f"{item_path}: kind 'etf_excess' needs ticker")

        low = _opt_float(item, "predicted_min", item_path) if "predicted_min" in item else None
        high = _opt_float(item, "predicted_max", item_path) if "predicted_max" in item else None
        if (low is None) != (high is None):
            raise ConfigError(
                f"{item_path}: predicted_min and predicted_max must be given together or not "
                "at all -- half a band is not a band"
            )
        if low is not None and high is not None and low >= high:
            raise ConfigError(
                f"{item_path}: predicted_min {low} must be below predicted_max {high}"
            )

        instruments.append(
            DurationCheckInstrument(
                id=identifier,
                kind=kind,
                predicted_duration=_float(item, "predicted_duration", item_path),
                maturity_years=maturity,
                ticker=ticker,
                predicted_min=low,
                predicted_max=high,
            )
        )

    return DurationCheckConfig(tolerance_fraction=tolerance, instruments=tuple(instruments))


def _parse_validation(node: dict[str, Any], path: str) -> ValidationConfig:
    """``factors.validation``. SPEC.md 6.5 as amended by 6.5.1.

    The parser enforces the two things a later edit could quietly break: that a
    factor is never both mapped and listed as having no analogue, and that a
    ``"sign"`` comparand carries the sign it predicts. Neither is left to a
    comment, because the mapping is the deliverable.
    """
    where = f"{path}.validation"
    child = _child(node, "validation", path)

    raw_comparands = _as_mapping(_require(child, "comparands", where), f"{where}.comparands")
    comparands: list[Comparand] = []
    for factor in sorted(raw_comparands):
        entry = _as_mapping(raw_comparands[factor], f"{where}.comparands.{factor}")
        at = f"{where}.comparands.{factor}"
        gate = _str(entry, "gate", at)
        if gate not in _VALIDATION_GATES:
            raise ConfigError(
                f"{at}.gate: expected one of {sorted(_VALIDATION_GATES)}, got {gate!r}"
            )
        sign: float | None = None
        if gate == _SIGN_GATE:
            sign = _float(entry, "expected_beta_sign", at)
            if sign not in (-1.0, 1.0):
                raise ConfigError(
                    f"{at}.expected_beta_sign is a convention, not a magnitude: "
                    f"expected -1 or +1, got {sign}"
                )
        elif "expected_beta_sign" in entry:
            raise ConfigError(
                f"{at}.expected_beta_sign is set on a {gate!r} comparand. A sign prediction "
                "belongs to a sign test; carrying one here would imply a falsifier the gate "
                "does not apply."
            )
        comparands.append(
            Comparand(
                factor=factor,
                dataset=_str(entry, "dataset", at),
                column=_str(entry, "column", at),
                gate=gate,
                expected_beta_sign=sign,
            )
        )
    if not comparands:
        raise ConfigError(f"{where}.comparands is empty; W2-P3 has nothing to regress")

    raw_unmapped = _as_mapping(_require(child, "unmapped", where), f"{where}.unmapped")
    unmapped: dict[str, str] = {}
    for factor in sorted(raw_unmapped):
        reason = raw_unmapped[factor]
        if not isinstance(reason, str) or not reason.strip():
            raise ConfigError(
                f"{where}.unmapped.{factor}: a factor with no comparand owes a stated reason, "
                "sourced to the column listing rather than to recollection"
            )
        unmapped[factor] = " ".join(reason.split())

    overlap = sorted({item.factor for item in comparands} & set(unmapped))
    if overlap:
        raise ConfigError(
            f"{where}: {overlap} are both mapped and listed as having no analogue. A factor is "
            "one or the other, and the mapping is the deliverable."
        )

    window = _int(child, "rolling_correlation_window_months", where)
    if window < 4:
        raise ConfigError(
            f"{where}.rolling_correlation_window_months: the Fisher-z standard error is "
            f"1/sqrt(w-3), which is undefined at w <= 3; got {window}"
        )

    rule = _str(child, "newey_west_lag_rule", where)
    if rule != _NEWEY_WEST_1994:
        raise ConfigError(
            f"{where}.newey_west_lag_rule: {rule!r} is not implemented. The only rule this "
            f"project uses is {_NEWEY_WEST_1994!r} -- Newey & West (1994), "
            "lag = floor(4*(T/100)^(2/9)) with T supplied at runtime."
        )

    anchor_at = f"{where}.daily_anchor"
    anchor_node = _child(child, "daily_anchor", where)
    return ValidationConfig(
        comparands=tuple(comparands),
        unmapped=unmapped,
        tsmom_applicable=_bool(child, "tsmom_applicable", where),
        rolling_correlation_window_months=window,
        newey_west_lag_rule=rule,
        placebo_gate=_bool(child, "placebo_gate", where),
        daily_anchor=DailyAnchor(
            comparand=_str(anchor_node, "comparand", anchor_at),
            registered_correlation=_float(anchor_node, "registered_correlation", anchor_at),
            registered_mean_gap_pct_per_year=_float(
                anchor_node, "registered_mean_gap_pct_per_year", anchor_at
            ),
            registered_observations=_int(anchor_node, "registered_observations", anchor_at),
            source_row=_int(anchor_node, "source_row", anchor_at),
        ),
    )


def _parse_factors(root: dict[str, Any]) -> FactorsConfig:
    node = _child(root, "factors", "model")
    macro = _child(node, "macro", "model.factors")
    path = "model.factors.macro"

    equity_node = _child(macro, "equity", path)
    dollar_node = _child(macro, "dollar", path)
    dollar_sign = _float(dollar_node, "sign", f"{path}.dollar")
    if dollar_sign not in (-1.0, 1.0):
        raise ConfigError(
            f"{path}.dollar.sign is a convention, not a scale: expected -1 or +1, got {dollar_sign}"
        )

    flag = _float(macro, "min_r_squared_flag", path)
    if not 0.0 <= flag <= 1.0:
        raise ConfigError(f"{path}.min_r_squared_flag is an R-squared and must be in [0, 1]")

    persistence = _float(macro, "persistence_fraction", path)
    if not 0.0 < persistence <= 1.0:
        raise ConfigError(
            f"{path}.persistence_fraction is a share of the defined dates and must be in (0, 1], "
            f"got {persistence}. Zero would flag every asset on a single date's breach."
        )

    beta_node = _child(macro, "beta", path)
    beta_window = _window_halflife(macro, "beta", path)

    # Parsed once and used twice: the macro rates block, and Model B's burn-in,
    # which is the SAME number by the W3-P5 ruling rather than a copy of it.
    rates = _parse_macro_rates(macro, path)

    alpha_node = _child(macro, "alpha_flag", path)
    alpha_t = _float(alpha_node, "abs_t_statistic", f"{path}.alpha_flag")
    if alpha_t <= 0.0:
        raise ConfigError(
            f"{path}.alpha_flag.abs_t_statistic is a critical value and must be positive, "
            f"got {alpha_t}"
        )

    return FactorsConfig(
        macro=MacroFactorConfig(
            equity=MacroEquityConfig(
                source=_str(equity_node, "source", f"{path}.equity"),
                column=_str(equity_node, "column", f"{path}.equity"),
            ),
            rates=rates,
            credit=_parse_asset_factor(macro, "credit", path),
            commodity=_parse_asset_factor(macro, "commodity", path),
            dollar=MacroDollarConfig(
                fred_series=_str(dollar_node, "fred_series", f"{path}.dollar"),
                sign=dollar_sign,
            ),
            orthogonalization=_parse_orthogonalization(macro, path),
            beta=MacroBetaConfig(
                window=beta_window.window,
                halflife=beta_window.halflife,
                fit_intercept=_bool(beta_node, "fit_intercept", f"{path}.beta"),
            ),
            min_r_squared_flag=flag,
            persistence_fraction=persistence,
            alpha_flag=AlphaFlagConfig(
                abs_t_statistic=alpha_t,
                newey_west_lags_from=_str(alpha_node, "newey_west_lags_from", f"{path}.alpha_flag"),
            ),
            condition_number=_parse_condition_number(macro, path),
            duration_check=_parse_duration_check(macro, path),
        ),
        statistical=_parse_statistical(node, "model.factors", rates_pca=rates.pca),
        hybrid=_parse_hybrid(
            node, "model.factors", rates_pca=rates.pca, curves=_curve_tenors(root)
        ),
        validation=_parse_validation(node, "model.factors"),
    )


def _parse_statistical(
    node: dict[str, Any], path: str, *, rates_pca: RatePcaConfig
) -> StatisticalFactorConfig:
    """Model B's parameters. SPEC.md 4.2 and the W3-P5 rulings at 4.2.1-4.2.4.

    ``rates_pca`` is passed in rather than looked up so that the reuse of its
    ``expanding_min_window`` is visible in the call site: the burn-in is ONE
    number in ``config/model.yaml``, stated at the rates PCA, and Model B reads
    it rather than declaring a second one that could drift. A key of that name
    under ``factors.statistical`` is REJECTED here rather than ignored, because
    an ignored key is exactly how the two would silently diverge.
    """
    child = _child(node, "statistical", path)
    where = f"{path}.statistical"

    estimator = _str(child, "estimator", where)
    if estimator not in _STATISTICAL_ESTIMATORS:
        raise ConfigError(
            f"{where}.estimator: {estimator!r} is not implemented; expected one of "
            f"{sorted(_STATISTICAL_ESTIMATORS)}. SPEC.md 4.2.1 -- Marchenko-Pastur is derived "
            "for T equally-weighted iid observations, so the EWMA correlation is not an "
            "available input here and is refused rather than supported."
        )

    if "expanding_min_window" in child:
        raise ConfigError(
            f"{where}.expanding_min_window must not be set: Model B reads the burn-in from "
            f"{path}.macro.rates.pca.expanding_min_window so that one number in this file "
            "states how long an expanding-window PCA must run before it is trusted. Two keys "
            "for one concept is how they drift apart."
        )

    bandwidth = _float(child, "kde_bandwidth", where)
    if bandwidth <= 0.0:
        raise ConfigError(
            f"{where}.kde_bandwidth is a kernel width in eigenvalue units and must be "
            f"positive, got {bandwidth}"
        )

    grid_points = _int(child, "kde_grid_points", where)
    if grid_points < 2:
        raise ConfigError(
            f"{where}.kde_grid_points must be at least 2 to span an interval, got {grid_points}"
        )

    bounds = _float_list(child, "noise_variance_bounds", where)
    if len(bounds) != 2:
        raise ConfigError(f"{where}.noise_variance_bounds must be [low, high], got {list(bounds)}")
    low, high = bounds
    if not 0.0 < low < high:
        raise ConfigError(
            f"{where}.noise_variance_bounds must satisfy 0 < low < high, got {list(bounds)}"
        )
    if high > 1.0:
        raise ConfigError(
            f"{where}.noise_variance_bounds upper bound is {high}: the fit runs on a "
            "CORRELATION matrix, where every variable has unit variance, so a noise variance "
            "above 1 is not a wider prior -- it is outside the estimator's own normalization."
        )

    components = _int(child, "detoned_components", where)
    if components < 1:
        raise ConfigError(
            f"{where}.detoned_components must be at least 1 -- SPEC.md 4.2 removes the first "
            f"eigenvector -- got {components}"
        )

    variants = _str_list(child, "variants", where)
    if set(variants) != _STATISTICAL_VARIANTS:
        raise ConfigError(
            f"{where}.variants must be exactly {sorted(_STATISTICAL_VARIANTS)}, got "
            f"{list(variants)}. Both are always built, as two model variants -- the same "
            "treatment the eigenfactor a = 1.0 / a = 1.4 pair gets. Dropping one here would "
            "be choosing between them, which is a `model-config` trial and owes a counted row."
        )

    rebalance = _int(child, "comparand_rebalance_days", where)
    if rebalance < 1:
        raise ConfigError(
            f"{where}.comparand_rebalance_days must be at least 1 trading day, got {rebalance}"
        )

    return StatisticalFactorConfig(
        estimator=estimator,
        expanding_min_window=rates_pca.expanding_min_window,
        kde_bandwidth=bandwidth,
        kde_grid_points=grid_points,
        noise_variance_bounds=(low, high),
        detoned_components=components,
        variants=variants,
        comparand_rebalance_days=rebalance,
    )


def _curve_tenors(root: dict[str, Any]) -> tuple[float, tuple[float, ...]]:
    """``(real tenor, nominal tenors)``, through the same parser ``_parse_data`` uses.

    Row 83's comparand tenor is NOT a key under ``factors.hybrid``. It is the
    single entry of ``data.real_curve.maturities_years``, because SPEC.md 3.1
    puts exactly one real point in the frozen universe, and a second place to
    state it is a second place for it to drift.
    """
    node = _child(root, "data", "model")
    real = _parse_curve(node, "real_curve", "model.data")
    nominal = _parse_curve(node, "nominal_curve", "model.data")
    if len(real.maturities_years) != 1:
        raise ConfigError(
            "model.data.real_curve.maturities_years must have exactly one entry -- SPEC.md 3.1 "
            f"puts one real point in the frozen universe -- got {list(real.maturities_years)}. "
            "Row 83's comparand tenor is read from it, so more than one leaves the placebo "
            "undefined rather than merely over-specified."
        )
    return real.maturities_years[0], nominal.maturities_years


def _parse_hybrid(
    node: dict[str, Any],
    path: str,
    *,
    rates_pca: RatePcaConfig,
    curves: tuple[float, tuple[float, ...]],
) -> HybridConfig:
    """The hybrid's parameters. SPEC.md 4.3 and the W3-P6 rulings at 4.3.1-4.3.3.

    ``rates_pca`` is passed in for the same reason :func:`_parse_statistical`
    takes it: the expanding-window burn-in is ONE number in
    ``config/model.yaml``, stated at the rates PCA, and a key of that name here
    is rejected rather than ignored.
    """
    child = _child(node, "hybrid", path)
    where = f"{path}.hybrid"
    tenor, nominal_tenors = curves

    estimator = _str(child, "estimator", where)
    if estimator not in _HYBRID_ESTIMATORS:
        raise ConfigError(
            f"{where}.estimator: {estimator!r} is not implemented; expected one of "
            f"{sorted(_HYBRID_ESTIMATORS)}. SPEC.md 4.3.3 -- the PCA runs on the residual "
            "CORRELATION, because a covariance PCA on residuals whose volatilities span an "
            "order of magnitude returns the most volatile asset as its leading eigenvector."
        )

    if "expanding_min_window" in child:
        raise ConfigError(
            f"{where}.expanding_min_window must not be set: the hybrid reads the burn-in from "
            f"{path}.macro.rates.pca.expanding_min_window so that one number in this file "
            "states how long an expanding-window PCA must run before it is trusted. Two keys "
            "for one concept is how they drift apart."
        )

    counts = tuple(int(value) for value in _float_list(child, "component_counts", where))
    if counts != (1, 2):
        raise ConfigError(
            f"{where}.component_counts must be [1, 2], got {list(counts)}. SPEC.md 4.3 asks for "
            '"the top 1-2 residual PCs" and that is a range over MODELS, not over noise: both '
            "are built and neither is selected. Dropping one here would be choosing between "
            "them, which is a `model-config` trial and owes a counted row in experiments.md."
        )

    leg = _str(child, "government_zero_return_leg", where)
    if leg not in _HYBRID_RETURN_LEGS:
        raise ConfigError(
            f"{where}.government_zero_return_leg: {leg!r} is not a leg this module strips to; "
            f"expected one of {sorted(_HYBRID_RETURN_LEGS)}. W2-P3 fixed the control as the "
            "duration leg alone (experiments.md rows 79-82), before any residual PC existed."
        )

    row_node = _child(child, "row_83", where)
    row_where = f"{where}.row_83"

    loading_rule = _str(row_node, "material_loading_rule", row_where)
    if loading_rule not in _MATERIAL_LOADING_RULES:
        raise ConfigError(
            f"{row_where}.material_loading_rule: {loading_rule!r} is not implemented; expected "
            f"one of {sorted(_MATERIAL_LOADING_RULES)}. A magnitude bar is deliberately not "
            "available here -- row 83 was registered on 2026-08-30 and a threshold chosen once "
            "the loadings were within reach would discard what the early registration bought."
        )

    comparand_rule = _str(row_node, "comparand_rule", row_where)
    if comparand_rule not in _HYBRID_COMPARAND_RULES:
        raise ConfigError(
            f"{row_where}.comparand_rule: {comparand_rule!r} is not implemented; expected one "
            f"of {sorted(_HYBRID_COMPARAND_RULES)}. The criterion is a PLACEBO, not a "
            "significance bar: at n ~ 4,100 a 5% test passes at |rho| > 0.031, and the 0.3 bar "
            "SPEC.md 6.5.2 withdrew is not reimported."
        )

    transform = _str(row_node, "comparand_transform", row_where)
    if transform not in _HYBRID_COMPARAND_TRANSFORMS:
        raise ConfigError(
            f"{row_where}.comparand_transform: {transform!r} is not implemented; expected one "
            f"of {sorted(_HYBRID_COMPARAND_TRANSFORMS)}. The comparand is first-differenced: "
            "correlating a daily return series against a yield LEVEL is a spurious-regression "
            "setup."
        )

    sign = _int(row_node, "expected_correlation_sign", row_where)
    if sign not in (-1, 1):
        raise ConfigError(
            f"{row_where}.expected_correlation_sign is a direction, not a scale: expected -1 "
            f"or +1, got {sign}"
        )

    if tenor not in nominal_tenors:
        raise ConfigError(
            f"{row_where}: the placebo compares the real and nominal curves at the SAME tenor, "
            f"and {tenor:g}y is not in model.data.nominal_curve.maturities_years "
            f"{list(nominal_tenors)}. A placebo run at two different tenors would compare "
            "curve shape as well as curve type, which is not the comparison row 83 registered."
        )

    return HybridConfig(
        estimator=estimator,
        expanding_min_window=rates_pca.expanding_min_window,
        component_counts=counts,
        government_zero_return_leg=leg,
        row_83=Row83Config(
            material_loading_rule=loading_rule,
            comparand_rule=comparand_rule,
            comparand_transform=transform,
            orientation_asset=_str(row_node, "orientation_asset", row_where),
            expected_correlation_sign=sign,
            tenor_years=tenor,
        ),
    )


def _parse_covariance(root: dict[str, Any]) -> CovarianceConfig:
    node = _child(root, "covariance", "model")
    path = "model.covariance"
    convention = _str(node, "mean_convention", path)
    if convention not in _MEAN_CONVENTIONS:
        raise ConfigError(
            f"{path}.mean_convention: only {sorted(_MEAN_CONVENTIONS)} is supported, got "
            f"{convention!r}. SPEC.md 5.1.1 -- the zero-mean convention is a coherence "
            "requirement against SPEC.md 6.1's bias statistic, whose numerator is a raw "
            "return, and it is deliberately not reachable by editing this key."
        )
    stages = _str_list(node, "stages", path)
    if not stages:
        raise ConfigError(f"{path}.stages must name at least one stage")
    if len(set(stages)) != len(stages):
        raise ConfigError(f"{path}.stages contains a duplicate: {stages}")
    return CovarianceConfig(
        factor_volatility_halflife=_int_horizons(node, "factor_volatility_halflife", path),
        factor_correlation_halflife=_int_horizons(node, "factor_correlation_halflife", path),
        volatility_newey_west_lags=_int_horizons(node, "volatility_newey_west_lags", path),
        correlation_newey_west_lags=_int_horizons(node, "correlation_newey_west_lags", path),
        mean_convention=cast(MeanConvention, convention),
        stages=stages,
    )


def _parse_vra(root: dict[str, Any]) -> VolatilityRegimeAdjustmentConfig:
    node = _child(root, "volatility_regime_adjustment", "model")
    return VolatilityRegimeAdjustmentConfig(
        halflife=_int_horizons(node, "halflife", "model.volatility_regime_adjustment")
    )


def _parse_eigenfactor(root: dict[str, Any]) -> EigenfactorConfig:
    node = _child(root, "eigenfactor", "model")
    trials = _int(node, "monte_carlo_trials", "model.eigenfactor")
    # CLAUDE.md parameter table: not published; any value in 1000-3000.
    if not 1000 <= trials <= 3000:
        raise ConfigError(
            f"model.eigenfactor.monte_carlo_trials must be in [1000, 3000], got {trials}"
        )
    return EigenfactorConfig(
        monte_carlo_trials=trials,
        scaling_a=_float_list(node, "scaling_a", "model.eigenfactor"),
    )


def _parse_validation_battery(root: dict[str, Any]) -> ValidationBatteryConfig:
    node = _child(root, "validation", "model")
    path = "model.validation"
    clip = _float(node, "standardized_return_clip", path)
    if not clip > 0.0:
        raise ConfigError(f"{path}.standardized_return_clip must be positive, got {clip}")
    months = _int(node, "rolling_window_months", path)
    if months < 2:
        raise ConfigError(
            f"{path}.rolling_window_months must be at least 2 -- a standard deviation at "
            f"ddof=1 needs two observations, got {months}"
        )
    z = _float(node, "normal_band_z", path)
    if not z > 0.0:
        raise ConfigError(f"{path}.normal_band_z must be positive, got {z}")
    level = _float(node, "chi_square_level", path)
    if not 0.0 < level < 1.0:
        raise ConfigError(f"{path}.chi_square_level must be in (0, 1), got {level}")
    count = _int(node, "random_portfolios", path)
    if count < 2:
        raise ConfigError(f"{path}.random_portfolios must be at least 2, got {count}")
    return ValidationBatteryConfig(
        standardized_return_clip=clip,
        rolling_window_months=months,
        normal_band_z=z,
        chi_square_level=level,
        random_portfolios=count,
        halflife_sensitivity=_parse_halflife_sensitivity(node, path),
        battery=_parse_battery_tests(root, node, path),
        second_order=_parse_second_order(root, node, path),
        kt_scaling=_parse_kt_scaling(root, node, path),
    )


def resolve(model: ModelConfig, pointer: str) -> Any:
    """Dereference a ``*_from`` pointer such as ``"eigenfactor.monte_carlo_trials"``.

    Walks attribute by attribute from the model config, so the value comes from
    the key the pointer names and never from a second copy. A pointer that does
    not resolve is a defect in ``config/model.yaml`` and raises.
    """
    node: Any = model
    for part in pointer.split("."):
        if not hasattr(node, part):
            raise ConfigError(f"config.resolve: {pointer!r} does not resolve at {part!r}")
        node = getattr(node, part)
    return node


def _check_pointer(root: dict[str, Any], pointer: str, where: str) -> None:
    """A ``*_from`` key must name a key that exists, so a rename cannot leave it dangling."""
    node: Any = root
    for part in pointer.split("."):
        if not isinstance(node, dict) or part not in node:
            raise ConfigError(
                f"{where} = {pointer!r} points at a key that is not in config/model.yaml"
            )
        node = node[part]


def _int_list(node: dict[str, Any], key: str, path: str) -> tuple[int, ...]:
    value = _require(node, key, path)
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{path}.{key}: expected a non-empty list, got {value!r}")
    out: list[int] = []
    for i, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise ConfigError(f"{path}.{key}[{i}]: expected an integer, got {item!r}")
        out.append(item)
    return tuple(out)


def _parse_battery_tests(
    root: dict[str, Any], parent: dict[str, Any], parent_path: str
) -> BatteryTestsConfig:
    """SPEC.md 6.5's battery. Shape checks here; the values are the rulings'."""
    node = _child(parent, "battery", parent_path)
    path = f"{parent_path}.battery"
    levels = _float_list(node, "var_levels", path)
    for level in levels:
        if not 0.5 < level < 1.0:
            raise ConfigError(f"{path}.var_levels must lie in (0.5, 1), got {level}")
    if len(set(levels)) != len(levels):
        raise ConfigError(f"{path}.var_levels repeats a level: {list(levels)}")
    months = _int_list(node, "ljung_box_lags_months", path)
    if any(month < 1 for month in months) or any(
        later <= earlier for earlier, later in itertools.pairwise(months)
    ):
        raise ConfigError(f"{path}.ljung_box_lags_months must be positive and increasing")
    basel_node = _child(node, "basel", path)
    basel_path = f"{path}.basel"
    basel = BaselConfig(
        level=_float(basel_node, "level", basel_path),
        window_days=_int(basel_node, "window_days", basel_path),
        green_max_breaches=_int(basel_node, "green_max_breaches", basel_path),
        yellow_max_breaches=_int(basel_node, "yellow_max_breaches", basel_path),
    )
    if basel.level not in levels:
        raise ConfigError(
            f"{basel_path}.level = {basel.level} must be one of var_levels {list(levels)} so the "
            "breach series is shared with Kupiec and Christoffersen at that level"
        )
    if basel.window_days < 1 or not 0 <= basel.green_max_breaches < basel.yellow_max_breaches:
        raise ConfigError(f"{basel_path}: window and zone bounds are inconsistent")
    pointers = {
        name: _str(node, name, path)
        for name in (
            "monte_carlo_trials_from",
            "newey_west_lags_from",
            "rank_correlation_block_days_from",
            "t_statistic_threshold_from",
            "t_statistic_window_from",
        )
    }
    for name, pointer in pointers.items():
        _check_pointer(root, pointer, f"{path}.{name}")
    return BatteryTestsConfig(
        var_levels=levels, ljung_box_lags_months=months, basel=basel, **pointers
    )


def _parse_second_order(
    root: dict[str, Any], parent: dict[str, Any], parent_path: str
) -> SecondOrderConfig:
    node = _child(parent, "second_order", parent_path)
    path = f"{parent_path}.second_order"
    pointer = _str(node, "monte_carlo_trials_from", path)
    _check_pointer(root, pointer, f"{path}.monte_carlo_trials_from")
    share = _float(node, "interaction_share_falsifier", path)
    if not 0.0 < share < 1.0:
        raise ConfigError(f"{path}.interaction_share_falsifier must be in (0, 1), got {share}")
    band = _float(node, "closed_form_relative_band", path)
    if not 0.0 < band < 1.0:
        raise ConfigError(f"{path}.closed_form_relative_band must be in (0, 1), got {band}")
    equivalent = _int(node, "row_109_equivalent_correlation_t", path)
    if equivalent < 2:
        raise ConfigError(
            f"{path}.row_109_equivalent_correlation_t must exceed 1, got {equivalent}"
        )
    grid_node = _child(node, "tau_grid", path)
    grid_path = f"{path}.tau_grid"
    grid_pointer = _str(grid_node, "grid_from", grid_path)
    _check_pointer(root, grid_pointer, f"{grid_path}.grid_from")
    hypothesis = _float(grid_node, "joint_at_shortest_hypothesis", grid_path)
    lower = _float(grid_node, "joint_at_shortest_lower", grid_path)
    upper = _float(grid_node, "joint_at_shortest_upper", grid_path)
    if not 0.0 < lower <= hypothesis <= upper:
        raise ConfigError(
            f"{grid_path}: the registered band must satisfy 0 < lower <= hypothesis <= upper, "
            f"got {lower}, {hypothesis}, {upper}"
        )
    leg = _float(grid_node, "correlation_leg_registered", grid_path)
    if not leg > 0.0:
        raise ConfigError(f"{grid_path}.correlation_leg_registered must be positive, got {leg}")
    return SecondOrderConfig(
        monte_carlo_trials_from=pointer,
        interaction_share_falsifier=share,
        closed_form_relative_band=band,
        row_109_equivalent_correlation_t=equivalent,
        tau_grid=SecondOrderTauGridConfig(
            grid_from=grid_pointer,
            joint_at_shortest_hypothesis=hypothesis,
            joint_at_shortest_lower=lower,
            joint_at_shortest_upper=upper,
            correlation_leg_registered=leg,
        ),
    )


_KT_WINDOWS: Final[tuple[str, ...]] = ("correlation", "volatility")
_KT_TRACKING: Final[tuple[str, ...]] = ("track", "not_track")
_KT_HORIZONS: Final[tuple[str, ...]] = ("short", "long")


def _parse_kt_scaling(
    root: dict[str, Any], parent: dict[str, Any], parent_path: str
) -> KtScalingConfig:
    """SPEC.md 15.1.1's rulings, checked for the readings they may take and for dangling "
    "pointers."""
    node = _child(parent, "kt_scaling", parent_path)
    path = f"{parent_path}.kt_scaling"
    variant = _str(node, "family4_variant", path)
    tracking_node = _child(node, "expected_tracking", path)
    tracking: dict[str, str] = {}
    for model in ("equity", "macro"):
        value = _str(tracking_node, model, f"{path}.expected_tracking")
        if value not in _KT_TRACKING:
            raise ConfigError(
                f"{path}.expected_tracking.{model} must be one of {list(_KT_TRACKING)}, "
                f"got {value!r}"
            )
        tracking[model] = value
    factor_window = _str(node, "factor_comparand_window", path)
    naive_window = _str(node, "naive_comparand_window", path)
    for key, value in (
        ("factor_comparand_window", factor_window),
        ("naive_comparand_window", naive_window),
    ):
        if value not in _KT_WINDOWS:
            raise ConfigError(f"{path}.{key} must be one of {list(_KT_WINDOWS)}, got {value!r}")
    x_axis = _str(node, "x_axis", path)
    if x_axis != "realised_kish_mean":
        raise ConfigError(f"{path}.x_axis must be 'realised_kish_mean' (ruling 2), got {x_axis!r}")
    periods = _str(node, "interval_periods", path)
    if periods != "months":
        raise ConfigError(f"{path}.interval_periods must be 'months' (ruling 1), got {periods!r}")
    pointers = {
        key: _str(node, key, path)
        for key in ("interval_level_from", "subperiod_block_months_from", "equity_sweep_grid_from")
    }
    for key, pointer in pointers.items():
        _check_pointer(root, pointer, f"{path}.{key}")
    horizon = _str(node, "equity_sweep_horizon", path)
    if horizon not in _KT_HORIZONS:
        raise ConfigError(
            f"{path}.equity_sweep_horizon must be one of {list(_KT_HORIZONS)}, got {horizon!r}"
        )
    reversing = _int(node, "book_sweep_reversing_halflife", path)
    if reversing < 1:
        raise ConfigError(f"{path}.book_sweep_reversing_halflife must be positive, got {reversing}")
    reference = _float(node, "book_sweep_reference_bias", path)
    if not reference > 0.0:
        raise ConfigError(f"{path}.book_sweep_reference_bias must be positive, got {reference}")
    reg_node = _child(node, "registrations", path)
    reg_path = f"{path}.registrations"
    subset = _str(reg_node, "row_164_residual_subset", reg_path)
    if subset != "always_present":
        raise ConfigError(
            f"{reg_path}.row_164_residual_subset must be 'always_present' (SPEC.md 15.7.1 "
            f"construction 7), got {subset!r}"
        )
    registrations = KtScalingRegistrations(
        row_164_macro_c1_share_of_excess=_float(
            reg_node, "row_164_macro_c1_share_of_excess", reg_path
        ),
        row_164_macro_c1_share_of_naive_gap=_float(
            reg_node, "row_164_macro_c1_share_of_naive_gap", reg_path
        ),
        row_164_macro_eigen_share_of_excess=_float(
            reg_node, "row_164_macro_eigen_share_of_excess", reg_path
        ),
        row_164_residual_subset=subset,
    )
    return KtScalingConfig(
        family4_variant=variant,
        expected_tracking=tracking,
        factor_comparand_window=factor_window,
        naive_comparand_window=naive_window,
        x_axis=x_axis,
        interval_periods=periods,
        interval_level_from=pointers["interval_level_from"],
        subperiod_block_months_from=pointers["subperiod_block_months_from"],
        equity_sweep_grid_from=pointers["equity_sweep_grid_from"],
        equity_sweep_horizon=horizon,
        book_sweep_reversing_halflife=reversing,
        book_sweep_reference_bias=reference,
        registrations=registrations,
    )


def _parse_halflife_sensitivity(
    parent: dict[str, Any], parent_path: str
) -> HalflifeSensitivityConfig:
    """SPEC.md 5.1's sweep grid. Shape only -- the cross-node checks live at the caller.

    Strictly increasing is not a tidiness preference: the grid is the x-axis of
    ``reports/halflife_sensitivity.png`` and of SPEC.md 5.1.3's ``K/T_eff``
    companion panel, and a repeated point would silently build the same history
    twice under two names while an out-of-order one would draw a curve that
    doubles back. Both are defects the chart itself would not show.
    """
    node = _child(parent, "halflife_sensitivity", parent_path)
    path = f"{parent_path}.halflife_sensitivity"
    raw = node.get("grid")
    if not isinstance(raw, list) or not raw:
        raise ConfigError(f"{path}.grid must be a non-empty list of trading-day half-lives")
    grid: list[int] = []
    for position, entry in enumerate(raw):
        if isinstance(entry, bool) or not isinstance(entry, int):
            raise ConfigError(f"{path}.grid[{position}] must be an integer, got {entry!r}")
        if entry < 1:
            raise ConfigError(f"{path}.grid[{position}] must be positive, got {entry}")
        grid.append(entry)
    if len(grid) < 2:
        raise ConfigError(f"{path}.grid needs at least two points to be a sweep, got {grid}")
    if any(later <= earlier for earlier, later in itertools.pairwise(grid)):
        raise ConfigError(f"{path}.grid must be strictly increasing, got {grid}")
    return HalflifeSensitivityConfig(grid=tuple(grid))


def _parse_specific_risk(root: dict[str, Any]) -> SpecificRiskConfig:
    node = _child(root, "specific_risk", "model")
    path = "model.specific_risk"
    return SpecificRiskConfig(
        ewma_halflife=_int_horizons(node, "ewma_halflife", path),
        newey_west_lags=_int_horizons(node, "newey_west_lags", path),
        autocorrelation_halflife=_int_horizons(node, "autocorrelation_halflife", path),
        bayesian_shrinkage_q=_float_horizons(node, "bayesian_shrinkage_q", path),
    )


def _window_halflife(node: dict[str, Any], key: str, path: str) -> WindowHalflife:
    child = _child(node, key, path)
    sub = f"{path}.{key}"
    return WindowHalflife(window=_int(child, "window", sub), halflife=_int(child, "halflife", sub))


def _parse_equity_descriptors(root: dict[str, Any]) -> EquityDescriptorConfig:
    node = _child(root, "equity_descriptors", "model")
    path = "model.equity_descriptors"

    momentum_node = _child(node, "momentum", path)
    momentum = LaggedWindowHalflife(
        window=_int(momentum_node, "window", f"{path}.momentum"),
        halflife=_int(momentum_node, "halflife", f"{path}.momentum"),
        lag=_int(momentum_node, "lag", f"{path}.momentum"),
    )

    resvol_node = _child(node, "resvol_composite", path)
    resvol = ResvolComposite(
        dastd=_float(resvol_node, "dastd", f"{path}.resvol_composite"),
        cmra=_float(resvol_node, "cmra", f"{path}.resvol_composite"),
        hsigma=_float(resvol_node, "hsigma", f"{path}.resvol_composite"),
    )
    _check_weights_sum_to_one(
        {"dastd": resvol.dastd, "cmra": resvol.cmra, "hsigma": resvol.hsigma},
        f"{path}.resvol_composite",
    )

    liquidity_node = _child(node, "liquidity_composite", path)
    liquidity = LiquidityComposite(
        stom=_float(liquidity_node, "stom", f"{path}.liquidity_composite"),
        stoq=_float(liquidity_node, "stoq", f"{path}.liquidity_composite"),
        stoa=_float(liquidity_node, "stoa", f"{path}.liquidity_composite"),
    )
    _check_weights_sum_to_one(
        {"stom": liquidity.stom, "stoq": liquidity.stoq, "stoa": liquidity.stoa},
        f"{path}.liquidity_composite",
    )

    horizons_node = _child(node, "liquidity_horizons_months", path)
    horizons_path = f"{path}.liquidity_horizons_months"
    horizons = LiquidityHorizons(
        stom=_int(horizons_node, "stom", horizons_path),
        stoq=_int(horizons_node, "stoq", horizons_path),
        stoa=_int(horizons_node, "stoa", horizons_path),
    )
    if not 0 < horizons.stom <= horizons.stoq <= horizons.stoa:
        raise ConfigError(f"{horizons_path}: expected 0 < stom <= stoq <= stoa, got {horizons}")
    cmra_months = _int(node, "cmra_months", path)
    if cmra_months <= 0:
        raise ConfigError(f"{path}.cmra_months must be positive, got {cmra_months}")

    winsor_node = _child(node, "winsorization", path)
    winsor_path = f"{path}.winsorization"
    bound = _float(winsor_node, "bound_sd", winsor_path)
    if bound <= 0.0:
        raise ConfigError(f"{winsor_path}.bound_sd must be positive, got {bound}")
    sensitivity = _float_list(winsor_node, "sensitivity_bounds_sd", winsor_path)
    if any(b <= 0.0 for b in sensitivity) or bound in sensitivity:
        raise ConfigError(
            f"{winsor_path}.sensitivity_bounds_sd must be positive and differ from bound_sd, "
            f"got {list(sensitivity)} against {bound}"
        )
    winsorization = Winsorization(bound_sd=bound, sensitivity_bounds_sd=sensitivity)

    reg_node = _child(node, "registrations", path)
    reg_path = f"{path}.registrations"
    registrations = EquityDescriptorRegistrations(
        row_278_spy_correlation_min=_float(reg_node, "row_278_spy_correlation_min", reg_path),
        row_279_raw_nlsize_size_min_abs=_float(
            reg_node, "row_279_raw_nlsize_size_min_abs", reg_path
        ),
        row_279_raw_resvol_beta_min_abs=_float(
            reg_node, "row_279_raw_resvol_beta_min_abs", reg_path
        ),
        row_279_orthogonalized_max_abs=_float(reg_node, "row_279_orthogonalized_max_abs", reg_path),
        row_280_correlation_move_max=_float(reg_node, "row_280_correlation_move_max", reg_path),
        row_280_vif_move_max=_float(reg_node, "row_280_vif_move_max", reg_path),
        row_281_vif_max=_float(reg_node, "row_281_vif_max", reg_path),
    )

    return EquityDescriptorConfig(
        beta=_window_halflife(node, "beta", path),
        momentum=momentum,
        dastd=_window_halflife(node, "dastd", path),
        hsigma=_window_halflife(node, "hsigma", path),
        resvol_composite=resvol,
        liquidity_composite=liquidity,
        liquidity_horizons_months=horizons,
        cmra_months=cmra_months,
        winsorization=winsorization,
        registrations=registrations,
    )


def _parse_equity_universe(root: dict[str, Any]) -> EquityUniverseConfig:
    node = _child(root, "equity_universe", "model")
    path = "model.equity_universe"
    min_history = _int(node, "min_history_days", path)
    if min_history <= 0:
        raise ConfigError(f"{path}.min_history_days must be positive, got {min_history}")
    min_adv = _opt_float(node, "min_dollar_adv", path)
    if min_adv is not None and min_adv <= 0:
        raise ConfigError(
            f"{path}.min_dollar_adv must be positive or null (null = no published "
            f"threshold, SPEC.md 15.3.1 ruling 1), got {min_adv}"
        )
    band = _float_list(node, "membership_count_band", path)
    if len(band) != 2 or band[0] >= band[1] or any(b != int(b) for b in band):
        raise ConfigError(
            f"{path}.membership_count_band must be two integers [low, high] with low < high, "
            f"got {list(band)}"
        )
    reg_node = _child(node, "registrations", path)
    reg_path = f"{path}.registrations"
    registrations = EquityUniverseRegistrations(
        row_265_failed_after_retry_max=_int(reg_node, "row_265_failed_after_retry_max", reg_path),
        row_265_empty_expected=_int(reg_node, "row_265_empty_expected", reg_path),
        row_265_empty_tolerance=_int(reg_node, "row_265_empty_tolerance", reg_path),
        row_267_no_data_excess_max=_int(reg_node, "row_267_no_data_excess_max", reg_path),
    )
    for name, value in (
        ("row_265_failed_after_retry_max", registrations.row_265_failed_after_retry_max),
        ("row_265_empty_expected", registrations.row_265_empty_expected),
        ("row_265_empty_tolerance", registrations.row_265_empty_tolerance),
        ("row_267_no_data_excess_max", registrations.row_267_no_data_excess_max),
    ):
        if value < 0:
            raise ConfigError(f"{reg_path}.{name} must be non-negative, got {value}")
    return EquityUniverseConfig(
        constituents_url=_str(node, "constituents_url", path),
        changes_url=_str(node, "changes_url", path),
        licence=_str(node, "licence", path),
        licence_url=_str(node, "licence_url", path),
        reference_dir=_str(node, "reference_dir", path),
        intervals_file=_str(node, "intervals_file", path),
        membership_file=_str(node, "membership_file", path),
        min_history_days=min_history,
        min_dollar_adv=min_adv,
        membership_count_band=(int(band[0]), int(band[1])),
        registrations=registrations,
    )


_PERIOD_RE = re.compile(r"^(\d{4})Q([1-4])$")
_POINT_IN_TIME_READINGS: Final[tuple[str, ...]] = ("known_from_filing_date",)


def _parse_equity_shares(root: dict[str, Any]) -> EquityShareConfig:
    node = _child(root, "equity_shares", "model")
    path = "model.equity_shares"
    first_period = _str(node, "first_period", path)
    if _PERIOD_RE.match(first_period) is None:
        raise ConfigError(f"{path}.first_period must look like 2009Q2, got {first_period!r}")
    reading = _str(node, "point_in_time", path)
    if reading not in _POINT_IN_TIME_READINGS:
        raise ConfigError(
            f"{path}.point_in_time must be one of {list(_POINT_IN_TIME_READINGS)}, got {reading!r}"
        )
    frames_url = _str(node, "frames_url", path)
    for placeholder in ("{taxonomy}", "{tag}", "{unit}", "{period}"):
        if placeholder not in frames_url:
            raise ConfigError(f"{path}.frames_url must contain {placeholder}")
    index_url = _str(node, "index_url", path)
    for placeholder in ("{year}", "{quarter}"):
        if placeholder not in index_url:
            raise ConfigError(f"{path}.index_url must contain {placeholder}")
    user_agent_env = _str(node, "user_agent_env", path)
    if not user_agent_env.isidentifier():
        raise ConfigError(f"{path}.user_agent_env must be an environment variable NAME")
    reg_node = _child(node, "registrations", path)
    reg_path = f"{path}.registrations"
    band = _float_list(reg_node, "row_270_approximated_share_band_pct", reg_path)
    if len(band) != 2 or band[0] >= band[1]:
        raise ConfigError(f"{reg_path}.row_270_approximated_share_band_pct must be [low, high]")
    r272_node = _child(reg_node, "row_272", reg_path)
    r272_path = f"{reg_path}.row_272"
    row_272 = Row272Registration(
        ticker=_str(r272_node, "ticker", r272_path),
        cik=_int(r272_node, "cik", r272_path),
        in_force_on=_date(r272_node, "in_force_on", r272_path),
        expected_form=_str(r272_node, "expected_form", r272_path),
        expected_filed=_date(r272_node, "expected_filed", r272_path),
        expected_shares=_int(r272_node, "expected_shares", r272_path),
        split_ex_date=_date(r272_node, "split_ex_date", r272_path),
        split_ratio=_int(r272_node, "split_ratio", r272_path),
        pre_split_filed=_date(r272_node, "pre_split_filed", r272_path),
        pre_split_shares=_int(r272_node, "pre_split_shares", r272_path),
    )
    r276_node = _child(reg_node, "row_276", reg_path)
    r276_path = f"{reg_path}.row_276"
    row_276 = Row276Registration(
        ticker=_str(r276_node, "ticker", r276_path),
        predecessor_cik=_int(r276_node, "predecessor_cik", r276_path),
        in_force_on=_date(r276_node, "in_force_on", r276_path),
        expected_filed=_date(r276_node, "expected_filed", r276_path),
        expected_shares=_int(r276_node, "expected_shares", r276_path),
    )
    dv_band = _float_list(reg_node, "row_277_no_cap_dollar_volume_share_band_pct", reg_path)
    if len(dv_band) != 2 or dv_band[0] >= dv_band[1]:
        raise ConfigError(
            f"{reg_path}.row_277_no_cap_dollar_volume_share_band_pct must be [low, high]"
        )
    registrations = EquityShareRegistrations(
        row_269_current_by_ticker_min_pct=_float(
            reg_node, "row_269_current_by_ticker_min_pct", reg_path
        ),
        row_269_departed_by_name_min_pct=_float(
            reg_node, "row_269_departed_by_name_min_pct", reg_path
        ),
        row_270_approximated_share_band_pct=(band[0], band[1]),
        row_271_no_cap_first_session_min=_int(
            reg_node, "row_271_no_cap_first_session_min", reg_path
        ),
        row_271_no_cap_last_session_max=_int(reg_node, "row_271_no_cap_last_session_max", reg_path),
        row_272=row_272,
        row_274_max_dropped_universe_filings=_int(
            reg_node, "row_274_max_dropped_universe_filings", reg_path
        ),
        row_275_jumps_inside_band_max=_int(reg_node, "row_275_jumps_inside_band_max", reg_path),
        row_276=row_276,
        row_277_no_cap_dollar_volume_share_band_pct=(dv_band[0], dv_band[1]),
    )
    screen_node = _child(node, "consistency_screen", path)
    screen_path = f"{path}.consistency_screen"
    ratio = _float(screen_node, "ratio", screen_path)
    if ratio <= 1.0:
        raise ConfigError(f"{screen_path}.ratio must exceed 1, got {ratio}")
    empty_band = _float_list(screen_node, "empty_band", screen_path)
    if len(empty_band) != 2 or not (1.0 < empty_band[0] < ratio < empty_band[1]):
        raise ConfigError(
            f"{screen_path}.empty_band must be [low, high] with 1 < low < ratio < high, "
            f"got {list(empty_band)} against ratio {ratio}"
        )
    screen = ConsistencyScreen(ratio=ratio, empty_band=(empty_band[0], empty_band[1]))
    return EquityShareConfig(
        registrations=registrations,
        frames_url=frames_url,
        taxonomy=_str(node, "taxonomy", path),
        tag=_str(node, "tag", path),
        unit=_str(node, "unit", path),
        index_url=index_url,
        company_tickers_url=_str(node, "company_tickers_url", path),
        first_period=first_period,
        user_agent_env=user_agent_env,
        point_in_time=reading,
        consistency_screen=screen,
        overrides_file=_str(node, "overrides_file", path),
        published_totals_file=_str(node, "published_totals_file", path),
    )


_CONSTRAINT_READINGS: Final[tuple[str, ...]] = ("cap_weighted_industry_sum_to_zero",)
_INDUSTRY_POINT_IN_TIME_READINGS: Final[tuple[str, ...]] = ("drifted_names_by_filing_date",)
_THIN_INDUSTRY_READINGS: Final[tuple[str, ...]] = ("none",)


def _parse_equity_industries(root: dict[str, Any]) -> EquityIndustryConfig:
    node = _child(root, "equity_industries", "model")
    path = "model.equity_industries"
    submissions_url = _str(node, "submissions_url", path)
    if "{name}" not in submissions_url:
        raise ConfigError(f"{path}.submissions_url must contain {{name}}")
    header_url = _str(node, "header_url", path)
    for placeholder in ("{cik}", "{accession_nodash}", "{accession}"):
        if placeholder not in header_url:
            raise ConfigError(f"{path}.header_url must contain {placeholder}")
    forms = _str_list(node, "first_filing_forms", path)
    if not forms:
        raise ConfigError(f"{path}.first_filing_forms must name at least one form")
    reg_node = _child(node, "registrations", path)
    reg_path = f"{path}.registrations"
    band = _float_list(reg_node, "row_290_reclassified_cells_band_pct", reg_path)
    if len(band) != 2 or band[0] >= band[1]:
        raise ConfigError(f"{reg_path}.row_290_reclassified_cells_band_pct must be [low, high]")
    bt_node = _child(reg_node, "row_290_baseline_t_frequency_pct", reg_path)
    baseline_t = {
        str(k): _float(bt_node, str(k), f"{reg_path}.row_290_baseline_t_frequency_pct")
        for k in bt_node
    }
    registrations = EquityIndustryRegistrations(
        row_283_industry_coverage_min_pct=_float(
            reg_node, "row_283_industry_coverage_min_pct", reg_path
        ),
        row_283_other_max_pct=_float(reg_node, "row_283_other_max_pct", reg_path),
        row_284_sic_differs_max_pct=_float(reg_node, "row_284_sic_differs_max_pct", reg_path),
        row_284_industry_differs_max_pct=_float(
            reg_node, "row_284_industry_differs_max_pct", reg_path
        ),
        row_290_reclassified_cells_band_pct=(band[0], band[1]),
        row_290_r2_move_max=_float(reg_node, "row_290_r2_move_max", reg_path),
        row_290_t_frequency_move_max_pp=_float(
            reg_node, "row_290_t_frequency_move_max_pp", reg_path
        ),
        row_290_baseline_r2_weighted=_float(reg_node, "row_290_baseline_r2_weighted", reg_path),
        row_290_baseline_t_frequency_pct=baseline_t,
    )
    point_in_time = _str(node, "point_in_time", path)
    if point_in_time not in _INDUSTRY_POINT_IN_TIME_READINGS:
        raise ConfigError(
            f"{path}.point_in_time must be one of {list(_INDUSTRY_POINT_IN_TIME_READINGS)}, "
            f"got {point_in_time!r}"
        )
    return EquityIndustryConfig(
        submissions_url=submissions_url,
        header_url=header_url,
        first_filing_forms=tuple(forms),
        point_in_time=point_in_time,
        registrations=registrations,
    )


def _parse_equity_regression(root: dict[str, Any]) -> EquityRegressionConfig:
    node = _child(root, "equity_regression", "model")
    path = "model.equity_regression"
    exponent = _float(node, "weight_exponent", path)
    if not 0.0 <= exponent <= 1.0:
        raise ConfigError(f"{path}.weight_exponent must lie in [0, 1], got {exponent}")
    constraint = _str(node, "constraint", path)
    if constraint not in _CONSTRAINT_READINGS:
        raise ConfigError(
            f"{path}.constraint must be one of {list(_CONSTRAINT_READINGS)}, got {constraint!r}"
        )
    thin = _str(node, "thin_industry_rule", path)
    if thin not in _THIN_INDUSTRY_READINGS:
        raise ConfigError(
            f"{path}.thin_industry_rule must be one of {list(_THIN_INDUSTRY_READINGS)} "
            f"(SPEC.md 15.6.1 ruling 3), got {thin!r}"
        )
    max_members = _int(node, "thin_industry_report_max_members", path)
    if max_members < 1:
        raise ConfigError(f"{path}.thin_industry_report_max_members must be positive")
    target = _float(node, "r_squared_target", path)
    if not 0.0 < target < 1.0:
        raise ConfigError(f"{path}.r_squared_target must lie in (0, 1), got {target}")
    t_threshold = _float(node, "t_stat_threshold", path)
    if t_threshold <= 0.0:
        raise ConfigError(f"{path}.t_stat_threshold must be positive")
    tolerance = _float(node, "identity_tolerance", path)
    if not 0.0 < tolerance < 1e-6:
        raise ConfigError(f"{path}.identity_tolerance must be a small positive number")
    reg_node = _child(node, "registrations", path)
    reg_path = f"{path}.registrations"
    registrations = EquityRegressionRegistrations(
        row_286_country_market_correlation_min=_float(
            reg_node, "row_286_country_market_correlation_min", reg_path
        ),
        row_286_gap_sd_ratio_max=_float(reg_node, "row_286_gap_sd_ratio_max", reg_path),
        row_288_thin_industries_median_max=_int(
            reg_node, "row_288_thin_industries_median_max", reg_path
        ),
        row_288_style_condition_max=_float(reg_node, "row_288_style_condition_max", reg_path),
        row_288_full_condition_median_max=_float(
            reg_node, "row_288_full_condition_median_max", reg_path
        ),
        row_289_t_frequency_min_pct=_float(reg_node, "row_289_t_frequency_min_pct", reg_path),
    )
    return EquityRegressionConfig(
        weight_exponent=exponent,
        constraint=constraint,
        thin_industry_rule=thin,
        thin_industry_report_max_members=max_members,
        r_squared_target=target,
        t_stat_threshold=t_threshold,
        identity_tolerance=tolerance,
        registrations=registrations,
    )


_EQUITY_RISK_READINGS: Final[dict[str, tuple[str, ...]]] = {
    "never_present_industries": ("dropped",),
    "absent_industry_return": ("zero_fill",),
    "bucket_rule": ("equal_count_screened_cap",),
    "residual_history": ("own_observed_gaps_removed",),
    "short_history_rule": ("structural_below_arithmetic_floor",),
    "singleton_treatment": ("held_out_of_target",),
    "singleton_specific_risk": ("time_series", "structural"),
    "forecast_grid": ("month_end",),
    "missing_return": ("zero_after_last_bar",),
    "comparand_subset": ("always_present",),
}


def _parse_equity_risk(root: dict[str, Any]) -> EquityRiskConfig:
    node = _child(root, "equity_risk", "model")
    path = "model.equity_risk"
    readings: dict[str, str] = {}
    for key, allowed in _EQUITY_RISK_READINGS.items():
        value = _str(node, key, path)
        if value not in allowed:
            raise ConfigError(
                f"{path}.{key} must be one of {list(allowed)} (SPEC.md 15.7.1), got {value!r}"
            )
        readings[key] = value
    buckets = _int(node, "size_buckets", path)
    if buckets != 10:
        raise ConfigError(
            f"{path}.size_buckets is the definition of a decile and must be 10, got {buckets}"
        )
    scale = _float(node, "basis_points_per_unit", path)
    if scale != 10000.0:
        raise ConfigError(f"{path}.basis_points_per_unit is a unit and must be 10000, got {scale}")
    reg_node = _child(node, "registrations", path)
    reg_path = f"{path}.registrations"
    registrations = EquityRiskRegistrations(
        row_291_factors=_int(reg_node, "row_291_factors", reg_path),
        row_291_family4_pre_eigen_min=_float(reg_node, "row_291_family4_pre_eigen_min", reg_path),
        row_291_family4_pre_eigen_max=_float(reg_node, "row_291_family4_pre_eigen_max", reg_path),
        row_291_macro_family4_pre_eigen=_float(
            reg_node, "row_291_macro_family4_pre_eigen", reg_path
        ),
        row_292_never_present=_str_list(reg_node, "row_292_never_present", reg_path),
        row_293_macro_amplitude=_float(reg_node, "row_293_macro_amplitude", reg_path),
        row_293_return_to_one_min_share=_float(
            reg_node, "row_293_return_to_one_min_share", reg_path
        ),
        row_293_firing_k_over_t_min=_float(reg_node, "row_293_firing_k_over_t_min", reg_path),
        row_294_floor_percentile=_float(reg_node, "row_294_floor_percentile", reg_path),
        row_295_random_inside_min_share=_float(
            reg_node, "row_295_random_inside_min_share", reg_path
        ),
        row_298_floor_percentile=_float(reg_node, "row_298_floor_percentile", reg_path),
        row_298_baseline_singleton_percentile=_float(
            reg_node, "row_298_baseline_singleton_percentile", reg_path
        ),
        row_298_baseline_singleton_weight_share=_float(
            reg_node, "row_298_baseline_singleton_weight_share", reg_path
        ),
        row_298_baseline_family4_daily=_float(reg_node, "row_298_baseline_family4_daily", reg_path),
        row_298_baseline_family4_monthly=_float(
            reg_node, "row_298_baseline_family4_monthly", reg_path
        ),
        row_298_baseline_b_specific=_float(reg_node, "row_298_baseline_b_specific", reg_path),
        row_298_baseline_b_factor=_float(reg_node, "row_298_baseline_b_factor", reg_path),
        row_298_baseline_specified_family2=_float(
            reg_node, "row_298_baseline_specified_family2", reg_path
        ),
        row_298_baseline_random_inside=_float(reg_node, "row_298_baseline_random_inside", reg_path),
    )
    low, high = (
        registrations.row_291_family4_pre_eigen_min,
        registrations.row_291_family4_pre_eigen_max,
    )
    if not low < high:
        raise ConfigError(f"{reg_path}: row 291's band is inverted")
    if registrations.row_291_factors < 1:
        raise ConfigError(f"{reg_path}.row_291_factors must be positive")
    return EquityRiskConfig(
        never_present_industries=readings["never_present_industries"],
        absent_industry_return=readings["absent_industry_return"],
        size_buckets=buckets,
        bucket_rule=readings["bucket_rule"],
        residual_history=readings["residual_history"],
        short_history_rule=readings["short_history_rule"],
        singleton_treatment=readings["singleton_treatment"],
        singleton_specific_risk=readings["singleton_specific_risk"],
        forecast_grid=readings["forecast_grid"],
        missing_return=readings["missing_return"],
        comparand_subset=readings["comparand_subset"],
        basis_points_per_unit=scale,
        registrations=registrations,
    )


def _parse_crisis_windows(node: dict[str, Any], path: str) -> tuple[CrisisWindow, ...]:
    """Parse the shaded episodes, rejecting an inverted or empty window."""
    raw = _require(node, "crisis_windows", path)
    if not isinstance(raw, list) or not raw:
        raise ConfigError(f"{path}.crisis_windows: expected a non-empty list, got {raw!r}")
    windows = []
    for i, item in enumerate(raw):
        sub = f"{path}.crisis_windows[{i}]"
        mapping = _as_mapping(item, sub)
        window = CrisisWindow(
            label=_str(mapping, "label", sub),
            start=_date(mapping, "start", sub),
            end=_date(mapping, "end", sub),
        )
        if window.end <= window.start:
            raise ConfigError(f"{sub}: end must be after start, got {window.start}..{window.end}")
        windows.append(window)
    return tuple(windows)


def _parse_tradable_proxies(node: dict[str, Any], path: str) -> tuple[TradableProxy, ...]:
    raw = _require(node, "tradable_proxies", path.rsplit(".", 1)[0])
    if not isinstance(raw, list):
        raise ConfigError(f"{path}: expected a list")
    proxies: list[TradableProxy] = []
    seen: set[str] = set()
    for i, item in enumerate(raw):
        sub = f"{path}[{i}]"
        entry = _as_mapping(item, sub)
        proxy = TradableProxy(
            asset=_str(entry, "asset", sub),
            ticker=_str(entry, "ticker", sub),
            caveat=_str(entry, "caveat", sub).strip(),
        )
        if proxy.asset in seen:
            raise ConfigError(f"{path}: duplicate proxy for {proxy.asset!r}")
        if not proxy.caveat:
            raise ConfigError(f"{sub}.caveat: the proxy's caveat must be stated, not empty")
        seen.add(proxy.asset)
        proxies.append(proxy)
    return tuple(proxies)


def _parse_calibration_regime(node: dict[str, Any], path: str) -> CalibrationRegime:
    raw_points = _require(node, "points", path)
    if not isinstance(raw_points, list) or not raw_points:
        raise ConfigError(f"{path}.points: expected a non-empty list")
    points: list[CalibrationPoint] = []
    for i, item in enumerate(raw_points):
        sub = f"{path}.points[{i}]"
        entry = _as_mapping(item, sub)
        point = CalibrationPoint(
            participation_of_adv=_float(entry, "participation_of_adv", sub),
            cost_bps=_float(entry, "cost_bps", sub),
        )
        if point.participation_of_adv <= 0.0 or point.cost_bps <= 0.0:
            raise ConfigError(f"{sub}: participation and cost must both be positive")
        points.append(point)
    return CalibrationRegime(source=_str(node, "source", path), points=tuple(points))


def _parse_calibration_anchors(node: dict[str, Any], path: str) -> CalibrationAnchors:
    sigma = _float(node, "daily_volatility", path)
    if sigma <= 0.0:
        raise ConfigError(f"{path}.daily_volatility: must be positive, got {sigma}")
    check_node = _child(node, "cross_check", path)
    return CalibrationAnchors(
        daily_volatility=sigma,
        patient=_parse_calibration_regime(_child(node, "patient", path), f"{path}.patient"),
        urgent=_parse_calibration_regime(_child(node, "urgent", path), f"{path}.urgent"),
        cross_check=CalibrationCrossCheck(
            participation_of_adv=_float(check_node, "participation_of_adv", f"{path}.cross_check"),
            expected_cost_bps=_float(check_node, "expected_cost_bps", f"{path}.cross_check"),
        ),
    )


def _parse_capacity(node: dict[str, Any], path: str) -> CapacityConfig:
    """SPEC.md 10.4's rules. Refuses every input the curve would need (W5-P2)."""
    alpha_min = _float(node, "minimum_net_alpha", path)
    if alpha_min != 0.0:
        raise ConfigError(
            f"{path}.minimum_net_alpha: only the SPEC.md 10.4 special case alpha_min = 0 is "
            f"implemented, and no source supplies another value; got {alpha_min}. A non-zero "
            "threshold is refused until one is sourced (operator ruling, 2026-09-03, W5-P2)."
        )
    gross_alpha = _opt_float(node, "gross_alpha", path)
    if gross_alpha is not None:
        raise ConfigError(
            f"{path}.gross_alpha: alpha_g is NEVER a point value; got {gross_alpha}. It is swept "
            "and reported as a band under a principle written before W6-P1 opens (SPEC.md "
            "10.4.1, experiments.md W5-P2 forward registration). A point alpha_g would be the "
            "first invented parameter to reach the headline (CLAUDE.md invariant 9)."
        )
    grid_node = _child(node, "aum_grid", path)
    grid_path = f"{path}.aum_grid"
    grid = AumGridRule(
        spacing=_str(grid_node, "spacing", grid_path),
        low_participation_of_adv=_float(grid_node, "low_participation_of_adv", grid_path),
        high_participation_of_adv=_float(grid_node, "high_participation_of_adv", grid_path),
        points=_int(grid_node, "points", grid_path),
    )
    if grid.spacing != "log":
        raise ConfigError(
            f"{grid_path}.spacing: the rule is log-spaced (SPEC.md 10.4.1), got {grid.spacing!r}"
        )
    if not 0.0 < grid.low_participation_of_adv < grid.high_participation_of_adv:
        raise ConfigError(
            f"{grid_path}: need 0 < low_participation_of_adv < high_participation_of_adv, got "
            f"{grid.low_participation_of_adv} and {grid.high_participation_of_adv}"
        )
    if grid.high_participation_of_adv > 1.0:
        raise ConfigError(
            f"{grid_path}.high_participation_of_adv: a trade larger than a day's ADV is outside "
            f"the square-root law's calibration range, got {grid.high_participation_of_adv}"
        )
    if grid.points < 2:
        raise ConfigError(f"{grid_path}.points: a grid needs at least 2 points, got {grid.points}")
    return CapacityConfig(minimum_net_alpha=alpha_min, gross_alpha=None, aum_grid=grid)


def _parse_costs(root: dict[str, Any]) -> CostConfig:
    node = _child(root, "costs", "model")
    path = "model.costs"

    prefactor_node = _child(node, "square_root_prefactor", path)
    gamma_node = _child(node, "gamma_trade", path)
    edge_node = _child(node, "edge_spread", path)
    adv_node = _child(node, "adv", path)
    svv_node = _child(node, "spread_vs_volatility", path)

    edge = EdgeSpreadConfig(
        window=_int(edge_node, "window", f"{path}.edge_spread"),
        step=_str(edge_node, "step", f"{path}.edge_spread"),
        min_observations=_int(edge_node, "min_observations", f"{path}.edge_spread"),
        require_full_window=_bool(edge_node, "require_full_window", f"{path}.edge_spread"),
        signed_estimates=_bool(edge_node, "signed_estimates", f"{path}.edge_spread"),
        clip_negative_to_zero=_bool(edge_node, "clip_negative_to_zero", f"{path}.edge_spread"),
    )
    if edge.clip_negative_to_zero and not edge.signed_estimates:
        raise ConfigError(
            f"{path}.edge_spread: clip_negative_to_zero is true while signed_estimates is "
            "false. With unsigned estimates bidask returns |estimate|, which is already "
            "non-negative, so clipping would do nothing while appearing to guard against "
            "negative spreads. Set signed_estimates true (see experiments.md rows 54-58)."
        )
    if edge.step != "monthly":
        raise ConfigError(
            f"{path}.edge_spread.step: only 'monthly' is implemented, got {edge.step!r}"
        )
    if edge.min_observations < 3:
        raise ConfigError(
            f"{path}.edge_spread.min_observations: EDGE is undefined below 3 observations, "
            f"got {edge.min_observations}"
        )
    if edge.window < edge.min_observations:
        raise ConfigError(
            f"{path}.edge_spread: window {edge.window} is shorter than min_observations "
            f"{edge.min_observations}, so no window could ever produce an estimate"
        )

    crisis_windows = _parse_crisis_windows(svv_node, f"{path}.spread_vs_volatility")

    gamma = GammaTradeSweep(
        sweep_min=_float(gamma_node, "sweep_min", f"{path}.gamma_trade"),
        sweep_max=_float(gamma_node, "sweep_max", f"{path}.gamma_trade"),
        sweep_step=_float(gamma_node, "sweep_step", f"{path}.gamma_trade"),
    )
    if gamma.sweep_min > gamma.sweep_max:
        raise ConfigError(f"{path}.gamma_trade: sweep_min must not exceed sweep_max")
    if gamma.sweep_step <= 0.0:
        raise ConfigError(f"{path}.gamma_trade: sweep_step must be positive")
    span = (gamma.sweep_max - gamma.sweep_min) / gamma.sweep_step
    if abs(span - round(span)) > 1e-9:
        raise ConfigError(
            f"{path}.gamma_trade: sweep_step {gamma.sweep_step} does not divide the range "
            f"{gamma.sweep_min}..{gamma.sweep_max} into whole steps"
        )

    exponent = _float(node, "total_cost_exponent", path)
    if exponent <= 1.0:
        raise ConfigError(
            f"{path}.total_cost_exponent: must exceed 1 for the impact term to be convex "
            f"(SPEC.md 7.1, 'the 3/2 exponent'), got {exponent}"
        )

    commission = _float(node, "commission", path)
    if commission < 0.0:
        raise ConfigError(f"{path}.commission: must be non-negative, got {commission}")
    volatility_window = _int(node, "volatility_window", path)
    if volatility_window < 2:
        raise ConfigError(
            f"{path}.volatility_window: a standard deviation needs at least 2 observations, "
            f"got {volatility_window}"
        )

    level_node = _child(node, "spread_level", path)
    resolution = _opt_float(level_node, "display_resolution_bps", f"{path}.spread_level")
    if resolution is not None and resolution <= 0.0:
        raise ConfigError(
            f"{path}.spread_level.display_resolution_bps: must be positive or null, "
            f"got {resolution}"
        )
    level = SpreadLevelConfig(
        file=Path(_str(level_node, "file", f"{path}.spread_level")),
        shape_baseline=_str(level_node, "shape_baseline", f"{path}.spread_level"),
        display_resolution_bps=resolution,
    )
    if level.shape_baseline != "in_sample_median":
        raise ConfigError(
            f"{path}.spread_level.shape_baseline: only 'in_sample_median' is implemented -- a "
            "baseline at the disclosure date would read the holdout (CLAUDE.md invariant 5); "
            f"got {level.shape_baseline!r}"
        )

    proxies = _parse_tradable_proxies(node, f"{path}.tradable_proxies")
    anchors = _parse_calibration_anchors(
        _child(node, "calibration_anchors", path), f"{path}.calibration_anchors"
    )
    capacity = _parse_capacity(_child(node, "capacity", path), f"{path}.capacity")

    return CostConfig(
        square_root_prefactor=SquareRootPrefactor(
            patient=_float(prefactor_node, "patient", f"{path}.square_root_prefactor"),
            urgent=_float(prefactor_node, "urgent", f"{path}.square_root_prefactor"),
        ),
        total_cost_exponent=exponent,
        gamma_trade=gamma,
        edge_spread=edge,
        flat_spread_assumption_bps=_float(node, "flat_spread_assumption_bps", path),
        adv=AdvConfig(window=_int(adv_node, "window", f"{path}.adv")),
        commission=commission,
        buy_sell_asymmetry=_float(node, "buy_sell_asymmetry", path),
        volatility_window=volatility_window,
        spread_level=level,
        tradable_proxies=proxies,
        calibration_anchors=anchors,
        capacity=capacity,
        spread_vs_volatility=SpreadVersusVolatilityConfig(
            realised_volatility_window=_int(
                svv_node, "realised_volatility_window", f"{path}.spread_vs_volatility"
            ),
            minimum_baseline_for_ratio_bps=_float(
                svv_node, "minimum_baseline_for_ratio_bps", f"{path}.spread_vs_volatility"
            ),
            crisis_windows=crisis_windows,
        ),
    )


#: SPEC.md 11: the no-look-ahead test runs "across >= 50 values of t".
_SPEC_MINIMUM_LOOKAHEAD_DATES = 50


def _parse_backtest(root: dict[str, Any]) -> BacktestConfig:
    node = _child(root, "backtest", "model")
    path = "model.backtest"

    look_node = _child(node, "no_lookahead", path)
    look_path = f"{path}.no_lookahead"
    minimum = _int(look_node, "minimum_evaluation_dates", look_path)
    if minimum < _SPEC_MINIMUM_LOOKAHEAD_DATES:
        raise ConfigError(
            f"{look_path}.minimum_evaluation_dates: SPEC.md 11 requires at least "
            f"{_SPEC_MINIMUM_LOOKAHEAD_DATES} values of t, got {minimum}"
        )

    rec_node = _child(node, "reconciliation", path)
    rec_path = f"{path}.reconciliation"
    initial_nav = _float(rec_node, "initial_nav", rec_path)
    if not math.isfinite(initial_nav) or initial_nav <= 0.0:
        raise ConfigError(f"{rec_path}.initial_nav: must be a positive NAV, got {initial_nav}")
    roundings = _int(rec_node, "accounting_roundings", rec_path)
    if roundings < 1:
        raise ConfigError(
            f"{rec_path}.accounting_roundings: a rounding count is a positive integer, got "
            f"{roundings}"
        )
    weights = _float_list(rec_node, "synthetic_weights", rec_path)
    if len(weights) < 2 or any(w < 0.0 for w in weights) or sum(weights) > 1.0 + 1e-12:
        raise ConfigError(
            f"{rec_path}.synthetic_weights: expected at least two non-negative weights summing "
            f"to at most 1 (the rest is cash), got {list(weights)}"
        )
    real_strategy = _str(rec_node, "real_strategy", rec_path)
    if real_strategy != "equal_weight":
        raise ConfigError(
            f"{rec_path}.real_strategy: only 'equal_weight' is implemented -- the absence of "
            f"a choice (operator ruling 2, W5-P3) -- got {real_strategy!r}"
        )
    fin_node = _child(node, "financing", path)
    fin_path = f"{path}.financing"
    convention = _str(fin_node, "convention", fin_path)
    if convention != "risk_free_both_ways":
        raise ConfigError(
            f"{fin_path}.convention: only 'risk_free_both_ways' is ruled (W6-P3 ruling 1: cash "
            f"earns and pays RF, no spread); got {convention!r}"
        )
    rate_source = _str(fin_node, "rate_source", fin_path)
    if rate_source != "data.risk_free":
        raise ConfigError(
            f"{fin_path}.rate_source: the leg is the risk-free rate the excess returns already "
            f"use, 'data.risk_free'; got {rate_source!r}"
        )
    spread = _float(fin_node, "borrowing_spread", fin_path)
    if spread != 0.0:
        raise ConfigError(
            f"{fin_path}.borrowing_spread: a spread over RF is an invented number (CLAUDE.md "
            f"invariant 9); only 0.0 is accepted, got {spread}"
        )
    return BacktestConfig(
        no_lookahead=NoLookaheadConfig(minimum_evaluation_dates=minimum),
        reconciliation=ReconciliationConfig(
            initial_nav=initial_nav,
            accounting_roundings=roundings,
            synthetic_weights=weights,
            real_strategy=real_strategy,
        ),
        financing=FinancingConfig(
            convention=convention, rate_source=rate_source, borrowing_spread=spread
        ),
    )


def _deref(root: dict[str, Any], pointer: str, where: str) -> Any:
    """Follow a ``*_from`` pointer through the RAW mapping and return the value it names."""
    _check_pointer(root, pointer, where)
    node: Any = root
    for part in pointer.split("."):
        node = node[part]
    return node


_OPTIMIZER_LIMITS: Final[tuple[str, ...]] = (
    "adv_participation",
    "position_box",
    "turnover",
    "tracking_error",
)
_RISK_VARIANTS: Final[frozenset[str]] = frozenset({"eigen", "vra", "specified"})
_BOOK_SIZE_GRID: Final[str] = "aum_grid_endpoints_equal_weight"
_BOOK_SIZE_FIXED: Final[str] = "fixed"
_GRID_BOOK_SIZE_ANCHOR: Final[str] = "aum_grid_equal_weight_at_anchor"
_GRID_VARIANTS: Final[tuple[str, ...]] = (
    "sample_equal_weight",
    "ewma",
    "newey_west",
    "eigenfactor_a1.0",
    "eigenfactor_a1.4",
    "volatility_regime",
    "ledoit_wolf",
)
_GRID_TREATMENTS: Final[tuple[str, ...]] = (
    "none",
    "gross_then_net",
    "flat_spread",
    "time_varying",
)
_GRID_RISK_AVERSION: Final[frozenset[str]] = frozenset(
    {"tracking_error_constraint", "spec_initialisation"}
)
_GRID_MISALIGNMENT: Final[frozenset[str]] = frozenset(
    {"recovered_multiplier_two_pass", "spec_initialisation"}
)
_GRID_DENSE_WINDOW_RULE: Final[str] = "ewma_effective_sample_size"


def _parse_optimizer_limit(
    node: dict[str, Any], name: str, path: str, *, bound_key: str, maximum: float
) -> OptimizerConstraint:
    limit_node = _child(node, name, path)
    limit_path = f"{path}.{name}"
    bound = _opt_float(limit_node, bound_key, limit_path)
    priority = _opt_float(limit_node, "priority", limit_path)
    if bound is not None and not (0.0 < bound <= maximum):
        raise ConfigError(
            f"{limit_path}.{bound_key}: must be in (0, {maximum:g}] -- {maximum:g} is the "
            f"mathematical maximum on the simplex -- or null to switch the limit off; got {bound}"
        )
    if priority is not None and not (math.isfinite(priority) and priority > 0.0):
        raise ConfigError(
            f"{limit_path}.priority: a hinge priority is a positive number, or null for a HARD "
            f"constraint whose multiplier is observed (SPEC.md 8.1); got {priority}"
        )
    if bound is None and priority is not None:
        raise ConfigError(f"{limit_path}: a priority without a bound penalises nothing")
    return OptimizerConstraint(name=name, bound=bound, priority=priority)


def _parse_optimizer_grid(
    root: dict[str, Any],
    node: dict[str, Any],
    path: str,
    *,
    tracking: OptimizerTrackingErrorTarget,
) -> OptimizerGridConfig:
    """``model.optimizer.grid`` -- SPEC.md 9 under the W6-P2 rulings (SPEC.md 9.1)."""
    grid_node = _child(node, "grid", path)
    grid_path = f"{path}.grid"
    horizon = _str(grid_node, "horizon", grid_path)
    if horizon not in ("short", "long"):
        raise ConfigError(f"{grid_path}.horizon: 'short' or 'long', got {horizon!r}")
    alpha = _str(grid_node, "alpha", grid_path)
    if alpha != "rstr":
        raise ConfigError(
            f"{grid_path}.alpha: the grid trades the ruled alpha-hat, 'rstr'; got {alpha!r}"
        )
    scaling = _float(grid_node, "eigenfactor_scaling", grid_path)
    published = [float(a) for a in _child(root, "eigenfactor", "model")["scaling_a"]]
    if scaling not in published:
        raise ConfigError(
            f"{grid_path}.eigenfactor_scaling: {scaling} is not a published a; "
            f"eigenfactor.scaling_a declares {published}"
        )
    gamma_trade = _float(grid_node, "gamma_trade", grid_path)
    sweep = _child(_child(root, "costs", "model"), "gamma_trade", "model.costs")
    lo, hi, step = (
        float(sweep["sweep_min"]),
        float(sweep["sweep_max"]),
        float(sweep["sweep_step"]),
    )
    steps = round((gamma_trade - lo) / step)
    if not (lo <= gamma_trade <= hi) or abs(lo + steps * step - gamma_trade) > 1e-12:
        raise ConfigError(
            f"{grid_path}.gamma_trade: {gamma_trade} is not on costs.gamma_trade's grid "
            f"{lo}..{hi} step {step}"
        )
    te_multiple = _float(grid_node, "tracking_error_multiple", grid_path)
    if te_multiple not in tracking.multiples:
        raise ConfigError(
            f"{grid_path}.tracking_error_multiple: {te_multiple} is not on the band "
            f"{list(tracking.multiples)}"
        )
    spread_end = _str(grid_node, "spread_end", grid_path)
    if spread_end not in ("low", "high"):
        raise ConfigError(f"{grid_path}.spread_end: 'low' or 'high', got {spread_end!r}")
    risk_aversion = _str(grid_node, "risk_aversion", grid_path)
    if risk_aversion not in _GRID_RISK_AVERSION:
        raise ConfigError(
            f"{grid_path}.risk_aversion: one of {sorted(_GRID_RISK_AVERSION)}, got "
            f"{risk_aversion!r}"
        )
    pricing = _str(grid_node, "misalignment_pricing", grid_path)
    if pricing not in _GRID_MISALIGNMENT:
        raise ConfigError(
            f"{grid_path}.misalignment_pricing: one of {sorted(_GRID_MISALIGNMENT)}, got "
            f"{pricing!r}"
        )
    if risk_aversion == "spec_initialisation" and pricing != "spec_initialisation":
        raise ConfigError(
            f"{grid_path}.misalignment_pricing: with risk_aversion 'spec_initialisation' there "
            "is no recovered multiplier to price the penalty at; use 'spec_initialisation'"
        )

    # -- the book size -------------------------------------------------------
    book_node = _child(grid_node, "book_size", grid_path)
    book_path = f"{grid_path}.book_size"
    rule = _str(book_node, "rule", book_path)
    participation = _opt_float(book_node, "participation_of_adv", book_path)
    nav = _opt_float(book_node, "nav_dollars", book_path)
    anchors_node = _child(_child(root, "costs", "model"), "calibration_anchors", "model.costs")
    anchored = [
        float(point["participation_of_adv"])
        for regime in ("patient", "urgent")
        for point in anchors_node[regime]["points"]
    ]
    if rule == _GRID_BOOK_SIZE_ANCHOR:
        if nav is not None:
            raise ConfigError(
                f"{book_path}: rule {rule!r} derives the NAV from the ADV data; nav_dollars "
                f"must be null. Set rule: '{_BOOK_SIZE_FIXED}' to use a ruled NAV instead"
            )
        if participation is None or participation != min(anchored):
            raise ConfigError(
                f"{book_path}.participation_of_adv: must equal the SMALLEST anchored "
                f"participation of the square-root law, {min(anchored)} "
                f"(costs.calibration_anchors) -- the reference book trades where the cost "
                f"model is calibrated (SPEC.md 9.1); got {participation}"
            )
    elif rule == _BOOK_SIZE_FIXED:
        if nav is None or not (math.isfinite(nav) and nav > 0.0):
            raise ConfigError(f"{book_path}: rule 'fixed' needs a positive nav_dollars, got {nav}")
        if participation is not None:
            raise ConfigError(f"{book_path}: rule 'fixed' takes no participation_of_adv")
    else:
        raise ConfigError(
            f"{book_path}.rule: '{_GRID_BOOK_SIZE_ANCHOR}' or '{_BOOK_SIZE_FIXED}', got {rule!r}"
        )

    # -- axis 1 ----------------------------------------------------------------
    variants = _str_list(grid_node, "covariance_variants", grid_path)
    if tuple(variants) != _GRID_VARIANTS:
        raise ConfigError(
            f"{grid_path}.covariance_variants: SPEC.md 9's seven, in its order, "
            f"{list(_GRID_VARIANTS)}; got {list(variants)}"
        )
    window_rule = _str(grid_node, "dense_window_rule", grid_path)
    if window_rule != _GRID_DENSE_WINDOW_RULE:
        raise ConfigError(
            f"{grid_path}.dense_window_rule: only '{_GRID_DENSE_WINDOW_RULE}' -- the window is "
            f"DERIVED from the EWMA half-life, never chosen (SPEC.md 9.1 ruling 2); got "
            f"{window_rule!r}"
        )
    halflives = _child(_child(root, "covariance", "model"), "factor_volatility_halflife", "model")
    halflife = halflives[horizon]
    if not isinstance(halflife, int) or isinstance(halflife, bool) or halflife < 1:
        raise ConfigError(
            f"model.covariance.factor_volatility_halflife.{horizon}: expected a positive integer"
        )
    # T_eff = 2 tau / ln 2 (CLAUDE.md parameter table; mafrm.numerics.effective_sample_size).
    dense_window = round(2.0 * halflife / math.log(2.0))

    # -- axis 2 ----------------------------------------------------------------
    treatments = _str_list(grid_node, "cost_treatments", grid_path)
    if tuple(treatments) != _GRID_TREATMENTS:
        raise ConfigError(
            f"{grid_path}.cost_treatments: SPEC.md 9's four, in its order, "
            f"{list(_GRID_TREATMENTS)}; got {list(treatments)}"
        )
    flat_pointer = _str(grid_node, "flat_spread_from", grid_path)
    flat = _deref(root, flat_pointer, f"{grid_path}.flat_spread_from")
    if isinstance(flat, bool) or not isinstance(flat, int | float) or not flat > 0.0:
        raise ConfigError(f"{grid_path}.flat_spread_from -> {flat_pointer}: expected a positive")
    regimes = _str_list(grid_node, "time_varying_regimes", grid_path)
    if tuple(regimes) != ("patient", "urgent"):
        raise ConfigError(
            f"{grid_path}.time_varying_regimes: SPEC.md 9's treatment D runs BOTH regimes, "
            f"['patient', 'urgent']; got {list(regimes)}"
        )
    drops_cap = _bool(grid_node, "cost_free_drops_adv_cap", grid_path)

    # -- hinges and the ladder ---------------------------------------------------
    hinge_node = _child(grid_node, "hinge_priorities", grid_path)
    hinge_path = f"{grid_path}.hinge_priorities"
    priorities: dict[str, float] = {}
    for name in hinge_node:
        if name not in _OPTIMIZER_LIMITS:
            raise ConfigError(f"{hinge_path}: {name!r} is not a constraint")
        priority = _float(hinge_node, str(name), hinge_path)
        if not (math.isfinite(priority) and priority > 0.0):
            raise ConfigError(f"{hinge_path}.{name}: a hinge priority is positive, got {priority}")
        priorities[str(name)] = priority
    ladder = _str_list(grid_node, "relaxation_ladder", grid_path)
    if len(set(ladder)) != len(ladder):
        raise ConfigError(f"{grid_path}.relaxation_ladder: duplicate entries in {list(ladder)}")
    for name in ladder:
        if name not in _OPTIMIZER_LIMITS:
            raise ConfigError(f"{grid_path}.relaxation_ladder: {name!r} is not a constraint")
        if name in priorities:
            raise ConfigError(
                f"{grid_path}.relaxation_ladder: {name!r} is a hinge here; only a HARD "
                "constraint belongs on the ladder"
            )
    if risk_aversion == "tracking_error_constraint" and "tracking_error" not in ladder:
        raise ConfigError(
            f"{grid_path}.relaxation_ladder: in constraint mode the tracking-error bound is "
            "the hard limit that can empty the feasible set, so it must be on the ladder"
        )

    source = _str(grid_node, "trial_count_source", grid_path)
    if source != "experiments.md":
        raise ConfigError(
            f"{grid_path}.trial_count_source: the deflated Sharpe's N is read from "
            f"experiments.md's running total, nowhere else; got {source!r}"
        )
    data_node = _child(root, "data", "model")
    per_year, per_month = data_node["trading_days_per_year"], data_node["trading_days_per_month"]
    if not isinstance(per_year, int) or not isinstance(per_month, int) or per_month < 1:
        raise ConfigError("model.data: trading_days_per_year and _per_month must be integers")
    if per_year % per_month != 0:
        raise ConfigError(
            f"model.data: trading_days_per_year {per_year} is not a whole number of "
            f"{per_month}-day periods, so Lo's q is not an integer"
        )
    bands = _parse_grid_bands(
        grid_node,
        grid_path,
        tracking=tracking,
        reference_spread_end=spread_end,
        reference_multiple=te_multiple,
        reference_gamma_trade=gamma_trade,
        reference_horizon=horizon,
        reference_risk_aversion=risk_aversion,
        regimes=regimes,
        sweep=(lo, hi, step),
    )
    resolve = _parse_grid_resolve(grid_node, grid_path)
    capacity_cfg = _parse_grid_capacity(grid_node, grid_path, risk_aversion=risk_aversion)
    diagnostics = _parse_grid_diagnostics(grid_node, grid_path)
    return OptimizerGridConfig(
        horizon=horizon,
        alpha=alpha,
        eigenfactor_scaling=scaling,
        gamma_trade=gamma_trade,
        tracking_error_multiple=te_multiple,
        spread_end=spread_end,
        risk_aversion=risk_aversion,
        misalignment_pricing=pricing,
        book_size=OptimizerGridBookSize(
            rule=rule, participation_of_adv=participation, nav_dollars=nav
        ),
        covariance_variants=variants,
        dense_window_rule=window_rule,
        dense_window=int(dense_window),
        cost_treatments=treatments,
        flat_full_spread_bps=float(flat),
        time_varying_regimes=regimes,
        cost_free_drops_adv_cap=drops_cap,
        hinge_priorities=priorities,
        relaxation_ladder=ladder,
        trial_count_source=source,
        sharpe_aggregation_periods=per_year // per_month,
        bands=bands,
        resolve=resolve,
        capacity=capacity_cfg,
        diagnostics=diagnostics,
    )


_BANDS_BOOK_SIZE_RULE: Final[str] = "aum_grid_endpoints_equal_weight"


def _parse_grid_bands(
    grid_node: dict[str, Any],
    grid_path: str,
    *,
    tracking: OptimizerTrackingErrorTarget,
    reference_spread_end: str,
    reference_multiple: float,
    reference_gamma_trade: float,
    reference_horizon: str,
    reference_risk_aversion: str,
    regimes: tuple[str, ...],
    sweep: tuple[float, float, float],
) -> OptimizerGridBandsConfig:
    """``model.optimizer.grid.bands`` -- W6-P3, every list derived less the reference."""
    node = _child(grid_node, "bands", grid_path)
    path = f"{grid_path}.bands"
    ref = _child(node, "reference", path)
    variant = _str(ref, "variant", f"{path}.reference")
    if variant not in _GRID_VARIANTS:
        raise ConfigError(f"{path}.reference.variant: not one of SPEC.md 9's seven: {variant!r}")
    treatment = _str(ref, "treatment", f"{path}.reference")
    if treatment not in _GRID_TREATMENTS:
        raise ConfigError(f"{path}.reference.treatment: not one of SPEC.md 9's four: {treatment!r}")
    regime = _str(ref, "regime", f"{path}.reference")
    if regime not in regimes:
        raise ConfigError(f"{path}.reference.regime: one of {list(regimes)}, got {regime!r}")

    ends = _str_list(node, "spread_ends", path)
    for end in ends:
        if end not in ("low", "high"):
            raise ConfigError(f"{path}.spread_ends: 'low' or 'high', got {end!r}")
        if end == reference_spread_end:
            raise ConfigError(
                f"{path}.spread_ends: {end!r} is the reference cell's own end; a band is the "
                "OTHER end"
            )
    pointer = _str(node, "tracking_error_multiples_from", path)
    if pointer != "optimizer.tracking_error_target.multiples":
        raise ConfigError(
            f"{path}.tracking_error_multiples_from: the band is SPEC.md 8.5.1 ruling 3's "
            f"multiples and nothing else; got {pointer!r}"
        )
    multiples = tuple(m for m in tracking.multiples if m != reference_multiple)
    if len(multiples) != len(tracking.multiples) - 1:
        raise ConfigError(f"{path}: the reference multiple {reference_multiple} is not on the band")
    book_rule = _str(node, "book_size_rule", path)
    if book_rule != _BANDS_BOOK_SIZE_RULE:
        raise ConfigError(
            f"{path}.book_size_rule: only '{_BANDS_BOOK_SIZE_RULE}' (A_low and A_high, "
            f"W6-P1's rule); got {book_rule!r}"
        )
    gamma_pointer = _str(node, "gamma_trade_from", path)
    if gamma_pointer != "costs.gamma_trade":
        raise ConfigError(
            f"{path}.gamma_trade_from: SPEC.md 7.2.1's five-point rule, 'costs.gamma_trade'; "
            f"got {gamma_pointer!r}"
        )
    lo, hi, step = sweep
    count = round((hi - lo) / step)
    points = tuple(lo + k * step for k in range(count + 1))
    gammas = tuple(g for g in points if abs(g - reference_gamma_trade) > 1e-12)
    if len(gammas) != len(points) - 1:
        raise ConfigError(
            f"{path}: the reference gamma_trade {reference_gamma_trade} is off the sweep"
        )
    multiple = _int(node, "position_box_equal_weight_multiple", path)
    if multiple < 2:
        raise ConfigError(
            f"{path}.position_box_equal_weight_multiple: 1x forces equal weight and leaves the "
            f"optimizer no choice; the ruled value is 2 (SPEC.md 9.1 ruling 5), got {multiple}"
        )
    horizons = _str_list(node, "horizons", path)
    for h in horizons:
        if h not in ("short", "long"):
            raise ConfigError(f"{path}.horizons: 'short' or 'long', got {h!r}")
        if h == reference_horizon:
            raise ConfigError(f"{path}.horizons: {h!r} is the reference's own horizon")
    aversions = _str_list(node, "risk_aversions", path)
    for mode in aversions:
        if mode not in _GRID_RISK_AVERSION:
            raise ConfigError(
                f"{path}.risk_aversions: one of {sorted(_GRID_RISK_AVERSION)}, got {mode!r}"
            )
        if mode == reference_risk_aversion:
            raise ConfigError(f"{path}.risk_aversions: {mode!r} is the reference's own mode")
    return OptimizerGridBandsConfig(
        reference_variant=variant,
        reference_treatment=treatment,
        reference_regime=regime,
        spread_ends=ends,
        tracking_error_multiples=multiples,
        book_size_rule=book_rule,
        gamma_trades=gammas,
        position_box_equal_weight_multiple=multiple,
        horizons=horizons,
        risk_aversions=aversions,
    )


def _parse_grid_resolve(grid_node: dict[str, Any], grid_path: str) -> OptimizerGridResolveConfig:
    node = _child(grid_node, "resolve", grid_path)
    path = f"{grid_path}.resolve"
    solver = _str(node, "solver", path)
    if solver != "fallback":
        raise ConfigError(f"{path}.solver: the re-solve uses the FALLBACK solver; got {solver!r}")
    criterion = _str(node, "criterion", path)
    if criterion != "max":
        raise ConfigError(
            f"{path}.criterion: the MAX per-date L1 decides (operator ruling 6, W6-P3: the "
            f"median hides the worst date); got {criterion!r}"
        )
    threshold = _float(node, "l1_threshold", path)
    if not (math.isfinite(threshold) and threshold > 0.0):
        raise ConfigError(f"{path}.l1_threshold: a positive distance, got {threshold}")
    return OptimizerGridResolveConfig(solver=solver, criterion=criterion, l1_threshold=threshold)


def _parse_grid_capacity(
    grid_node: dict[str, Any], grid_path: str, *, risk_aversion: str
) -> OptimizerGridCapacityConfig:
    node = _child(grid_node, "capacity", grid_path)
    path = f"{grid_path}.capacity"
    pointer = _str(node, "aum_grid_from", path)
    if pointer != "costs.capacity.aum_grid":
        raise ConfigError(
            f"{path}.aum_grid_from: the ruled grid is 'costs.capacity.aum_grid'; got {pointer!r}"
        )
    adv = _str(node, "adv_participation", path)
    if adv != "hard":
        raise ConfigError(
            f"{path}.adv_participation: SPEC.md 10.4 re-optimises WITH the ADV constraint "
            f"binding, so it is 'hard'; got {adv!r}"
        )
    ladder = _str_list(node, "relaxation_ladder", path)
    if len(set(ladder)) != len(ladder):
        raise ConfigError(f"{path}.relaxation_ladder: duplicate entries in {list(ladder)}")
    for name in ladder:
        if name not in _OPTIMIZER_LIMITS:
            raise ConfigError(f"{path}.relaxation_ladder: {name!r} is not a constraint")
    if "adv_participation" not in ladder:
        raise ConfigError(
            f"{path}.relaxation_ladder: a hard ADV cap can empty the feasible set and must be "
            "on the ladder"
        )
    if risk_aversion == "tracking_error_constraint" and "tracking_error" not in ladder:
        raise ConfigError(f"{path}.relaxation_ladder: constraint mode needs 'tracking_error' on it")
    regimes = _str_list(node, "regimes", path)
    if tuple(regimes) != ("patient", "urgent"):
        raise ConfigError(
            f"{path}.regimes: SPEC.md 10.4.1 -- both Y regimes, ['patient', 'urgent']; got "
            f"{list(regimes)}"
        )
    report_sharpe = _bool(node, "report_sharpe", path)
    if report_sharpe:
        raise ConfigError(
            f"{path}.report_sharpe: a Sharpe per AUM point makes every point a strategy-config "
            "trial and N ~150 (operator ruling 4, W6-P3); only false is accepted"
        )
    comparand = _str(node, "rescaling_comparand", path)
    if comparand != "reference_cell_structure":
        raise ConfigError(
            f"{path}.rescaling_comparand: only 'reference_cell_structure'; got {comparand!r}"
        )
    return OptimizerGridCapacityConfig(
        aum_grid_from=pointer,
        adv_participation=adv,
        relaxation_ladder=ladder,
        regimes=regimes,
        report_sharpe=report_sharpe,
        rescaling_comparand=comparand,
    )


_DIAGNOSTICS_PINNED: Final[dict[str, tuple[str, str]]] = {
    "decision_price": ("implementation_shortfall", "rebalance_close"),
    "residual": ("implementation_shortfall", "priced_minus_charged"),
    "return_identity": ("attribution", "specific_is_the_remainder"),
    "variance_test": ("attribution", "bias_statistic_per_component"),
    "interval_from": ("attribution", "validation.chi_square_level"),
    "brinson_benchmark": ("attribution", "equal_weight"),
    "multi_period_linking": ("attribution", "carino"),
}


def _parse_grid_diagnostics(
    grid_node: dict[str, Any], grid_path: str
) -> OptimizerGridDiagnosticsConfig:
    """Every key is pinned to the one implemented, ruled reading; a change fails loudly."""
    node = _child(grid_node, "diagnostics", grid_path)
    path = f"{grid_path}.diagnostics"
    values: dict[str, str] = {}
    for key, (block, pinned) in _DIAGNOSTICS_PINNED.items():
        sub = _child(node, block, path)
        got = _str(sub, key, f"{path}.{block}")
        if got != pinned:
            raise ConfigError(
                f"{path}.{block}.{key}: only {pinned!r} is implemented and ruled (W6-P3); got "
                f"{got!r}"
            )
        values[key] = got
    return OptimizerGridDiagnosticsConfig(**values)


def _parse_optimizer(root: dict[str, Any]) -> OptimizerConfig:
    node = _child(root, "optimizer", "model")
    path = "model.optimizer"

    # -- ruling 1: alpha ----------------------------------------------------
    alpha_node = _child(node, "alpha", path)
    alpha_path = f"{path}.alpha"
    construction = _str(alpha_node, "construction", alpha_path)
    if construction != "rstr":
        raise ConfigError(
            f"{alpha_path}.construction: alpha-hat is RSTR by ruling (SPEC.md 8.5.1), never "
            f"another signal and never tuned; got {construction!r}"
        )
    resolved: dict[str, int] = {}
    for key in ("window", "halflife", "lag"):
        pointer = _str(alpha_node, f"{key}_from", alpha_path)
        value = _deref(root, pointer, f"{alpha_path}.{key}_from")
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ConfigError(f"{alpha_path}.{key}_from -> {pointer}: expected a positive integer")
        resolved[key] = value
    weights = _str(alpha_node, "weights", alpha_path)
    if weights != "normalised":
        raise ConfigError(
            f"{alpha_path}.weights: only 'normalised' -- RSTR must be a RATE so that alpha-hat is "
            f"in return units and commensurate with the cost term (SPEC.md 8.5.1); got {weights!r}"
        )
    centring = _str(alpha_node, "cross_sectional", alpha_path)
    if centring != "demean":
        raise ConfigError(
            f"{alpha_path}.cross_sectional: only 'demean'. Scaling to unit variance would destroy "
            f"the return units SPEC.md 8.4 presumes (SPEC.md 8.5.1); got {centring!r}"
        )
    horizon_pointer = _str(alpha_node, "horizon_days_from", alpha_path)
    horizon_days = _deref(root, horizon_pointer, f"{alpha_path}.horizon_days_from")
    if not isinstance(horizon_days, int) or isinstance(horizon_days, bool) or horizon_days < 1:
        raise ConfigError(f"{alpha_path}.horizon_days_from: expected a positive integer")
    alpha = OptimizerAlphaConfig(
        construction=construction,
        window=resolved["window"],
        halflife=resolved["halflife"],
        lag=resolved["lag"],
        weights=weights,
        cross_sectional=centring,
        horizon_days=horizon_days,
    )
    if alpha.lag >= alpha.window:
        raise ConfigError(
            f"{alpha_path}: lag {alpha.lag} must be shorter than window {alpha.window}"
        )

    # -- ruling 3: TE target ------------------------------------------------
    te_node = _child(node, "tracking_error_target", path)
    te_path = f"{path}.tracking_error_target"
    anchor = _str(te_node, "anchor", te_path)
    if anchor != "equal_weight_realised_volatility":
        raise ConfigError(
            f"{te_path}.anchor: TE_target is a MEASURED anchor -- the equal-weight portfolio's "
            f"realised in-sample volatility -- not a constant (SPEC.md 8.5.1); got {anchor!r}"
        )
    multiples = _float_list(te_node, "multiples", te_path)
    if any(m <= 0.0 for m in multiples) or list(multiples) != sorted(set(multiples)):
        raise ConfigError(
            f"{te_path}.multiples: strictly increasing positive values, got {list(multiples)}"
        )
    verification_multiple = _float(te_node, "verification_multiple", te_path)
    if verification_multiple not in multiples:
        raise ConfigError(
            f"{te_path}.verification_multiple: {verification_multiple} is not on the band "
            f"{list(multiples)}"
        )
    tracking = OptimizerTrackingErrorTarget(
        anchor=anchor, multiples=multiples, verification_multiple=verification_multiple
    )

    gamma_risk = _opt_float(node, "gamma_risk", path)
    if gamma_risk is not None and not (math.isfinite(gamma_risk) and gamma_risk > 0.0):
        raise ConfigError(
            f"{path}.gamma_risk: null for SPEC.md 8.4's IR/(2 TE_target), or a positive pin; "
            f"got {gamma_risk}"
        )

    # -- ruling 4: absences -------------------------------------------------
    rob_node = _child(node, "robustification", path)
    rob_path = f"{path}.robustification"
    rho = _float(rob_node, "return_forecast_rho", rob_path)
    varrho = _float(rob_node, "covariance_varrho", rob_path)
    if rho < 0.0 or not math.isfinite(rho):
        raise ConfigError(f"{rob_path}.return_forecast_rho: must be >= 0, got {rho}")
    if not (0.0 <= varrho < 1.0):
        raise ConfigError(
            f"{rob_path}.covariance_varrho: SPEC.md 8.2 has varrho in [0, 1), got {varrho}"
        )
    gamma_hold = _float(node, "gamma_hold", path)
    if gamma_hold < 0.0 or not math.isfinite(gamma_hold):
        raise ConfigError(f"{path}.gamma_hold: must be >= 0, got {gamma_hold}")
    misalignment = _str(node, "misalignment_penalty", path)
    if misalignment not in ("msci", "none"):
        raise ConfigError(
            f"{path}.misalignment_penalty: 'msci' (psi_mis = lambda * sigma^2(alpha_perp), "
            f"COMPUTED) or 'none' (attribution only); got {misalignment!r}"
        )
    benchmark = _str(node, "benchmark", path)
    if benchmark != "zero":
        raise ConfigError(
            f"{path}.benchmark: only 'zero' -- w^b = 0 is ABSOLUTE risk, matching sigma_f in "
            f"SPEC.md 1's identity (ruling 4); got {benchmark!r}"
        )

    # -- ruling 5: constraints ----------------------------------------------
    con_node = _child(node, "constraints", path)
    con_path = f"{path}.constraints"
    long_only = _bool(con_node, "long_only", con_path)
    fully_invested = _bool(con_node, "fully_invested", con_path)
    adv_limit = _parse_optimizer_limit(
        con_node, "adv_participation", con_path, bound_key="cap", maximum=1.0
    )
    anchors_node = _child(_child(root, "costs", "model"), "calibration_anchors", "model.costs")
    anchored = [
        float(point["participation_of_adv"])
        for regime in ("patient", "urgent")
        for point in anchors_node[regime]["points"]
    ]
    if adv_limit.bound is None or adv_limit.bound != max(anchored):
        raise ConfigError(
            f"{con_path}.adv_participation.cap: must equal the LARGEST anchored participation of "
            f"the square-root law, {max(anchored)} (costs.calibration_anchors) -- 'do not trade "
            f"where "
            f"the cost model is not calibrated' (SPEC.md 8.5.1 ruling 5); got {adv_limit.bound}"
        )
    box_limit = _parse_optimizer_limit(
        con_node, "position_box", con_path, bound_key="upper", maximum=1.0
    )
    turnover_limit = _parse_optimizer_limit(
        con_node, "turnover", con_path, bound_key="bound", maximum=2.0
    )
    te_limit = _parse_optimizer_limit(
        con_node, "tracking_error", con_path, bound_key="bound", maximum=math.inf
    )
    percentile = _float(con_node, "lagrange_percentile", con_path)
    if not (0.0 < percentile < 1.0):
        raise ConfigError(
            f"{con_path}.lagrange_percentile: a probability in (0, 1), got {percentile}"
        )
    constraints = OptimizerConstraintsConfig(
        long_only=long_only,
        fully_invested=fully_invested,
        adv_participation=adv_limit,
        position_box=box_limit,
        turnover=turnover_limit,
        tracking_error=te_limit,
        lagrange_percentile=percentile,
    )

    ladder = _str_list(node, "relaxation_ladder", path)
    if len(set(ladder)) != len(ladder):
        raise ConfigError(f"{path}.relaxation_ladder: duplicate entries in {list(ladder)}")
    for name in ladder:
        if name not in _OPTIMIZER_LIMITS:
            raise ConfigError(
                f"{path}.relaxation_ladder: {name!r} is not a constraint; expected one of "
                f"{list(_OPTIMIZER_LIMITS)}"
            )
        if not constraints.by_name(name).hard:
            raise ConfigError(
                f"{path}.relaxation_ladder: {name!r} is not a HARD constraint. Only a hard "
                "constraint can make the problem infeasible, so only one belongs on the ladder"
            )

    solver_node = _child(node, "solver", path)
    solver_path = f"{path}.solver"
    solver = OptimizerSolverConfig(
        primary=_str(solver_node, "primary", solver_path),
        fallback=_str(solver_node, "fallback", solver_path),
        feasibility_tolerance=_float(solver_node, "feasibility_tolerance", solver_path),
    )
    if solver.primary == solver.fallback:
        raise ConfigError(
            f"{solver_path}: primary and fallback must differ, both {solver.primary!r}"
        )
    if not (0.0 < solver.feasibility_tolerance < 1e-2):
        raise ConfigError(
            f"{solver_path}.feasibility_tolerance: a solver tolerance, got "
            f"{solver.feasibility_tolerance}"
        )

    # -- ruling 6: the verification run -------------------------------------
    ver_node = _child(node, "verification", path)
    ver_path = f"{path}.verification"
    ver_alpha = _str(ver_node, "alpha", ver_path)
    if ver_alpha not in ("rstr", "zero"):
        raise ConfigError(f"{ver_path}.alpha: 'rstr' or 'zero', got {ver_alpha!r}")
    regime = _str(ver_node, "cost_regime", ver_path)
    if regime not in ("patient", "urgent"):
        raise ConfigError(
            f"{ver_path}.cost_regime: 'patient' or 'urgent' (SPEC.md 7.2), got {regime!r}"
        )
    scaling = _float(ver_node, "eigenfactor_scaling", ver_path)
    published = [float(a) for a in _child(root, "eigenfactor", "model")["scaling_a"]]
    if scaling not in published:
        raise ConfigError(
            f"{ver_path}.eigenfactor_scaling: {scaling} is not a published a; "
            f"eigenfactor.scaling_a "
            f"declares {published}"
        )
    horizon = _str(ver_node, "horizon", ver_path)
    if horizon not in ("short", "long"):
        raise ConfigError(f"{ver_path}.horizon: 'short' or 'long', got {horizon!r}")
    variant = _str(ver_node, "risk_variant", ver_path)
    if variant not in _RISK_VARIANTS:
        raise ConfigError(
            f"{ver_path}.risk_variant: one of {sorted(_RISK_VARIANTS)}, got {variant!r}"
        )
    gamma_trade = _float(ver_node, "gamma_trade", ver_path)
    sweep = _child(_child(root, "costs", "model"), "gamma_trade", "model.costs")
    lo, hi, step = (
        float(sweep["sweep_min"]),
        float(sweep["sweep_max"]),
        float(sweep["sweep_step"]),
    )
    steps = round((gamma_trade - lo) / step)
    if not (lo <= gamma_trade <= hi) or abs(lo + steps * step - gamma_trade) > 1e-12:
        raise ConfigError(
            f"{ver_path}.gamma_trade: {gamma_trade} is not on costs.gamma_trade's grid "
            f"{lo}..{hi} step {step}"
        )
    te_multiple = _float(ver_node, "tracking_error_multiple", ver_path)
    if te_multiple not in tracking.multiples:
        raise ConfigError(
            f"{ver_path}.tracking_error_multiple: {te_multiple} is not on the band "
            f"{list(tracking.multiples)}"
        )
    book_node = _child(ver_node, "book_size", ver_path)
    book_path = f"{ver_path}.book_size"
    rule = _str(book_node, "rule", book_path)
    nav = _opt_float(book_node, "nav_dollars", book_path)
    if rule == _BOOK_SIZE_GRID:
        if nav is not None:
            raise ConfigError(
                f"{book_path}: rule {rule!r} derives the band from the ADV data; nav_dollars must "
                f"be "
                f"null. Set rule: '{_BOOK_SIZE_FIXED}' to use a ruled NAV instead"
            )
    elif rule == _BOOK_SIZE_FIXED:
        if nav is None or not (math.isfinite(nav) and nav > 0.0):
            raise ConfigError(f"{book_path}: rule 'fixed' needs a positive nav_dollars, got {nav}")
    else:
        raise ConfigError(
            f"{book_path}.rule: '{_BOOK_SIZE_GRID}' or '{_BOOK_SIZE_FIXED}', got {rule!r}"
        )
    ends = _str_list(ver_node, "spread_ends", ver_path)
    if not ends or len(set(ends)) != len(ends) or any(e not in ("low", "high") for e in ends):
        raise ConfigError(
            f"{ver_path}.spread_ends: a non-empty subset of ['low', 'high'], got {list(ends)}"
        )
    control_cost_free = _bool(ver_node, "control_cost_free", ver_path)
    verification = OptimizerVerificationConfig(
        alpha=ver_alpha,
        cost_regime=regime,
        eigenfactor_scaling=scaling,
        horizon=horizon,
        risk_variant=variant,
        gamma_trade=gamma_trade,
        tracking_error_multiple=te_multiple,
        book_size=OptimizerBookSize(rule=rule, nav_dollars=nav),
        spread_ends=ends,
        control_cost_free=control_cost_free,
    )

    grid = _parse_optimizer_grid(root, node, path, tracking=tracking)

    return OptimizerConfig(
        alpha=alpha,
        tracking_error_target=tracking,
        gamma_risk=gamma_risk,
        robustification=OptimizerRobustification(return_forecast_rho=rho, covariance_varrho=varrho),
        gamma_hold=gamma_hold,
        misalignment_penalty=misalignment,
        benchmark=benchmark,
        constraints=constraints,
        relaxation_ladder=ladder,
        solver=solver,
        verification=verification,
        grid=grid,
    )


def _parse_model(raw: Any) -> ModelConfig:
    root = _as_mapping(raw, "model")
    return ModelConfig(
        schema_version=_int(root, "schema_version", "model"),
        seed=_int(root, "seed", "model"),
        sample=_parse_sample(root),
        data=_parse_data(root),
        numerics=_parse_numerics(root),
        factors=_parse_factors(root),
        covariance=_parse_covariance(root),
        volatility_regime_adjustment=_parse_vra(root),
        eigenfactor=_parse_eigenfactor(root),
        specific_risk=_parse_specific_risk(root),
        validation=_parse_validation_battery(root),
        equity_descriptors=_parse_equity_descriptors(root),
        equity_universe=_parse_equity_universe(root),
        equity_shares=_parse_equity_shares(root),
        equity_industries=_parse_equity_industries(root),
        equity_regression=_parse_equity_regression(root),
        equity_risk=_parse_equity_risk(root),
        costs=_parse_costs(root),
        backtest=_parse_backtest(root),
        optimizer=_parse_optimizer(root),
    )


# ---------------------------------------------------------------------------
# universe.yaml
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Asset:
    """One frozen universe member. SPEC.md section 3.1."""

    id: str
    #: Vendor symbol, or ``None`` for the synthetic curve-derived series.
    ticker: str | None
    sleeve: str
    construction: str
    source: str
    history_start: date
    splice: bool
    rationale: str
    #: Constant maturity in years for the synthetic bond series; ``None`` otherwise.
    maturity_years: float | None = None
    #: Required whenever ``splice`` is true.
    splice_note: str | None = None


@dataclass(frozen=True)
class UniverseConfig:
    """Everything in ``config/universe.yaml``."""

    schema_version: int
    frozen_date: date
    common_start: date
    assets: tuple[Asset, ...]

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(a.id for a in self.assets)

    @property
    def tickers(self) -> tuple[str, ...]:
        """Vendor symbols only; synthetic series are excluded."""
        return tuple(a.ticker for a in self.assets if a.ticker is not None)

    @property
    def yfinance_tickers(self) -> tuple[str, ...]:
        """Symbols pulled from yfinance, in universe order.

        Filtered on ``source`` rather than on "has a ticker", because
        :attr:`tickers` also returns ``DTWEXBGS`` -- a FRED index level that has
        a symbol but no bars, and asking Yahoo for it returns an empty frame the
        cache would then refuse.
        """
        return tuple(
            a.ticker
            for a in self.assets
            if a.ticker is not None and a.source.split("+")[0].strip() == "yfinance"
        )

    def by_id(self, asset_id: str) -> Asset:
        for asset in self.assets:
            if asset.id == asset_id:
                return asset
        raise KeyError(f"no asset {asset_id!r} in the frozen universe")


def _parse_asset(raw: Any, index: int) -> Asset:
    path = f"universe.assets[{index}]"
    node = _as_mapping(raw, path)
    splice = _bool(node, "splice", path)
    splice_note = node.get("splice_note")
    if splice and not isinstance(splice_note, str):
        raise ConfigError(f"{path}.splice_note is required when splice is true")
    if splice_note is not None and not isinstance(splice_note, str):
        raise ConfigError(f"{path}.splice_note: expected a string, got {splice_note!r}")
    return Asset(
        id=_str(node, "id", path),
        ticker=_opt_str(node, "ticker", path),
        sleeve=_str(node, "sleeve", path),
        construction=_str(node, "construction", path),
        source=_str(node, "source", path),
        history_start=_date(node, "history_start", path),
        splice=splice,
        rationale=_str(node, "rationale", path),
        maturity_years=_opt_float(node, "maturity_years", path)
        if "maturity_years" in node
        else None,
        splice_note=splice_note,
    )


def _parse_universe(raw: Any) -> UniverseConfig:
    root = _as_mapping(raw, "universe")
    raw_assets = _require(root, "assets", "universe")
    if not isinstance(raw_assets, list) or not raw_assets:
        raise ConfigError("universe.assets: expected a non-empty list")

    assets = tuple(_parse_asset(item, i) for i, item in enumerate(raw_assets))

    seen: set[str] = set()
    for asset in assets:
        if asset.id in seen:
            raise ConfigError(f"universe.assets: duplicate id {asset.id!r}")
        seen.add(asset.id)

    # SPEC.md 3.1: three splices maximum, half a day budgeted each.
    n_splices = sum(a.splice for a in assets)
    if n_splices > 3:
        raise ConfigError(f"universe.assets: at most 3 splices allowed, found {n_splices}")

    return UniverseConfig(
        schema_version=_int(root, "schema_version", "universe"),
        frozen_date=_date(root, "frozen_date", "universe"),
        common_start=_date(root, "common_start", "universe"),
        assets=assets,
    )


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def config_dir() -> Path:
    """Directory holding the two YAML files. Override with ``MAFRM_CONFIG_DIR``."""
    override = os.environ.get("MAFRM_CONFIG_DIR")
    return Path(override) if override else _REPO_ROOT / "config"


def _read_yaml(path: Path) -> Any:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_model(path: Path | None = None) -> ModelConfig:
    """Parse and validate ``config/model.yaml``."""
    return _parse_model(_read_yaml(path or config_dir() / "model.yaml"))


def load_universe(path: Path | None = None) -> UniverseConfig:
    """Parse and validate ``config/universe.yaml``."""
    return _parse_universe(_read_yaml(path or config_dir() / "universe.yaml"))


@dataclass(frozen=True)
class Config:
    """Both config files, parsed and cross-validated."""

    model: ModelConfig
    universe: UniverseConfig

    @property
    def seed(self) -> int:
        """The single project seed. Pass to ``np.random.default_rng(seed)``."""
        return self.model.seed

    def cost_ticker(self, asset_id: str) -> str:
        """The vendor symbol whose bars price trading ``asset_id``.

        The asset's own ticker for an ETF, its ``costs.tradable_proxies`` entry
        for a synthetic curve point (operator ruling 0, 2026-09-03, W5-P1).
        Raises for anything else -- there is no tradable instrument behind the
        dollar index, and a cost for it would be a cost for nothing.
        """
        asset = self.universe.by_id(asset_id)
        proxy = self.model.costs.proxy_for(asset_id)
        if proxy is not None:
            return proxy.ticker
        if asset.ticker is not None and asset.source.split("+")[0].strip() == "yfinance":
            return asset.ticker
        raise ConfigError(
            f"no tradable instrument for {asset_id!r}: it has no yfinance ticker and no entry "
            "in model.costs.tradable_proxies, so it cannot carry a spread, an ADV or an "
            "impact cost"
        )

    @property
    def cost_tickers(self) -> tuple[str, ...]:
        """Every symbol the cost model reads bars for, proxies included, deduplicated."""
        out: list[str] = []
        for asset in self.universe.assets:
            try:
                ticker = self.cost_ticker(asset.id)
            except ConfigError:
                continue
            if ticker not in out:
                out.append(ticker)
        return tuple(out)

    def require_holdout_start(self) -> date:
        """The holdout boundary, or raise if it has not been pinned yet.

        CLAUDE.md invariant 5. Any code that filters data against the holdout
        must call this rather than reading ``model.sample.holdout_start``
        directly, so that an unset holdout fails loudly instead of silently
        letting the holdout into the training window.
        """
        holdout_start = self.model.sample.holdout_start
        if holdout_start is None:
            raise ConfigError(
                "model.sample.holdout_start is not set. CLAUDE.md invariant 5 cannot be "
                "enforced until it is pinned in config/model.yaml. Set it (and "
                "holdout_end) before running anything that touches the holdout."
            )
        return holdout_start


#: Seated ONLY by :func:`mafrm.data.holdout.evaluating_holdout`, the one
#: sanctioned crossing of the holdout boundary (CLAUDE.md invariant 5, SPEC.md 9
#: and 12: the holdout is evaluated once). It is a module-level override rather
#: than a threaded argument because ``cache.read`` takes no config -- the
#: boundary is enforced at the point data enters the process, so moving it has
#: to move there too. Nothing else may write it, and a test pins the caller.
_OVERRIDE: Config | None = None


@lru_cache(maxsize=1)
def _load_cached() -> Config:
    """The parsed config files. :func:`load` is the entry point."""
    config = Config(model=load_model(), universe=load_universe())
    if config.universe.common_start != config.model.sample.start:
        raise ConfigError(
            f"universe.common_start ({config.universe.common_start}) must equal "
            f"model.sample.start ({config.model.sample.start})"
        )
    _check_tradable_proxies(config)
    return config


def load() -> Config:
    """Load, validate and cache both config files.

    Cached because the parsed result is immutable and the seed is read at import
    time. Call ``load.cache_clear()`` in tests that rewrite the YAML.

    Returns :data:`_OVERRIDE` while :func:`mafrm.data.holdout.evaluating_holdout`
    is active, which is the only thing that ever sets it.
    """
    return _OVERRIDE if _OVERRIDE is not None else _load_cached()


def _clear_config_cache() -> None:
    """``load.cache_clear()``, kept as the documented API after the split."""
    _load_cached.cache_clear()


load.cache_clear = _clear_config_cache  # type: ignore[attr-defined]


def _check_tradable_proxies(config: Config) -> None:
    """A proxy must stand in for a synthetic member, never shadow a real ticker."""
    for proxy in config.model.costs.tradable_proxies:
        try:
            asset = config.universe.by_id(proxy.asset)
        except KeyError as exc:
            raise ConfigError(
                f"model.costs.tradable_proxies: {proxy.asset!r} is not in the frozen universe"
            ) from exc
        if asset.ticker is not None:
            raise ConfigError(
                f"model.costs.tradable_proxies: {proxy.asset!r} already trades as "
                f"{asset.ticker}; a proxy is only for a synthetic curve point"
            )


#: The single project seed (CLAUDE.md, code conventions). Never call
#: ``np.random.seed()``; pass this to ``np.random.default_rng(SEED)`` explicitly
#: at every stochastic call site, or the eigenfactor Monte Carlo is not
#: reproducible.
SEED: Final[int] = load().seed
