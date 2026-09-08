"""The boundary object for :mod:`mafrm.risk`. SPEC.md 15.2.

SPEC.md 15.2 specifies the interface this package presents::

    def build_covariance(factor_returns, config: RiskConfig) -> np.ndarray

:class:`RiskConfig` is that ``RiskConfig``, and it is the mechanism by which
CLAUDE.md invariant 10 is kept rather than merely intended. It is a flat bundle
of half-lives, lag counts, tolerances and a seed. **There is no field on it that
names a source, an instrument, a calendar or a model.** A function that takes
only this and a ``T x K`` frame cannot condition on what the columns are,
because nothing it can reach knows.

Every value is read from ``config/model.yaml`` through :mod:`mafrm.config`;
nothing here supplies a default, so a missing key fails at load time rather than
silently taking a plausible value (CLAUDE.md invariants 6 and 9).

The one structural decision worth stating: a ``RiskConfig`` is built **for one
horizon**. SPEC.md 5.1 runs both -- short is optimizer-facing, long is
attribution-facing -- and they are two model variants rather than a setting with
a default, so the horizon is chosen at construction and then fixed. Nothing
downstream has to remember to pass it, and nothing can accidentally mix a short
volatility half-life with a long regime half-life.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, TypeVar

from mafrm.config import Config, Horizons, MeanConvention, ModelConfig, load

__all__ = ["HORIZONS", "Horizon", "RiskConfig"]

#: SPEC.md 5.1 runs both. Short is the optimizer-facing model, long the
#: attribution-facing one; they are variants, not alternatives to choose between.
Horizon = Literal["short", "long"]

HORIZONS: tuple[Horizon, ...] = ("short", "long")

#: Matches the value restriction on ``Horizons``'s own TypeVar in
#: :mod:`mafrm.config`; an unconstrained one is not a valid argument to it.
_T = TypeVar("_T", int, float)


def _pick(values: Horizons[_T], horizon: Horizon) -> _T:
    """One horizon out of a short/long pair, typed. ``getattr`` would erase the type."""
    return values.short if horizon == "short" else values.long


@dataclass(frozen=True)
class RiskConfig:
    """Every number the covariance and specific-risk pipelines need, for one horizon."""

    #: ``"short"`` or ``"long"``. Fixed at construction; see the module docstring.
    horizon: Horizon

    # -- SPEC.md 5.1, EWMA with separated volatility and correlation ----------
    #: USE4 Table 4.1. 84d short, 252d long. Volatilities mean-revert fast.
    volatility_halflife: int
    #: USE4 Table 4.1. 504d at BOTH horizons -- deliberate, not a copy-paste
    #: error. Correlations are noisier and need stability.
    correlation_halflife: int
    #: How the EWMA moments are centred. Always ``"zero"``; SPEC.md 5.1.1.
    mean_convention: MeanConvention

    # -- SPEC.md 5.2, Newey-West ---------------------------------------------
    volatility_newey_west_lags: int
    correlation_newey_west_lags: int

    # -- SPEC.md 5.3, eigenfactor risk adjustment ----------------------------
    eigenfactor_monte_carlo_trials: int
    #: Both 1.0 and 1.4 are always run, as two variants. CLAUDE.md table.
    eigenfactor_scaling: tuple[float, ...]

    # -- SPEC.md 5.4, volatility regime adjustment ---------------------------
    #: One half-life shared by the factor and specific VRA. USE4 is explicit
    #: that equal half-lives for both components is deliberate.
    volatility_regime_halflife: int

    # -- SPEC.md 5.5, specific risk ------------------------------------------
    specific_volatility_halflife: int
    specific_newey_west_lags: int
    specific_autocorrelation_halflife: int
    bayesian_shrinkage_q: float

    # -- SPEC.md 5, the pipeline itself --------------------------------------
    #: Stage names in application order. Order is load-bearing.
    stages: tuple[str, ...]

    # -- Numerical thresholds -------------------------------------------------
    psd_eigenvalue_floor: float
    psd_reconstruction_roundings: int
    symmetry_absolute_tolerance: float

    #: Passed explicitly to ``np.random.default_rng``; never a global seed.
    seed: int

    #: WHICH published ``a`` this config builds. ``None`` until a caller says.
    #:
    #: There is deliberately **no default**. SPEC.md 5.3 publishes two values --
    #: ``a = 1.0`` attribution-facing, which is what USE4 runs in production, and
    #: ``a = 1.4`` optimizer-facing -- and CLAUDE.md's parameter table says run
    #: both. A default here would let a later session inherit one without
    #: noticing, and picking one *because it flattered a result* is a
    #: ``model-config`` trial that would then go unlogged against the deflated
    #: Sharpe. So the pipeline refuses to run the eigenfactor stage until
    #: :meth:`for_scaling` has been called, and the refusal names both values.
    eigenfactor_variant: float | None = None

    def for_scaling(self, scaling: float) -> RiskConfig:
        """This config, fixed to one published eigenfactor ``a``.

        Refuses any value not in ``eigenfactor.scaling_a``. CLAUDE.md invariant
        9: 1.0 and 1.4 are published, anything between them is invented, and an
        invented risk-scaling constant is exactly the kind of number that gets
        tuned until a bias statistic looks right.
        """
        if scaling not in self.eigenfactor_scaling:
            raise ValueError(
                f"RiskConfig.for_scaling: {scaling} is not a published eigenfactor scaling. "
                f"config/model.yaml declares {list(self.eigenfactor_scaling)} -- 1.0 is USE4's "
                "production, attribution-facing value and 1.4 is the optimizer-facing one. "
                "A value in between is not a compromise, it is an invented constant."
            )
        return replace(self, eigenfactor_variant=float(scaling))

    @classmethod
    def from_model(cls, model: ModelConfig, *, horizon: Horizon) -> RiskConfig:
        """Project ``config/model.yaml`` onto one horizon."""
        if horizon not in HORIZONS:
            raise ValueError(f"RiskConfig: horizon must be one of {HORIZONS}, got {horizon!r}")
        covariance = model.covariance
        specific = model.specific_risk
        numerics = model.numerics
        return cls(
            horizon=horizon,
            volatility_halflife=_pick(covariance.factor_volatility_halflife, horizon),
            correlation_halflife=_pick(covariance.factor_correlation_halflife, horizon),
            mean_convention=covariance.mean_convention,
            volatility_newey_west_lags=_pick(covariance.volatility_newey_west_lags, horizon),
            correlation_newey_west_lags=_pick(covariance.correlation_newey_west_lags, horizon),
            eigenfactor_monte_carlo_trials=model.eigenfactor.monte_carlo_trials,
            eigenfactor_scaling=model.eigenfactor.scaling_a,
            volatility_regime_halflife=_pick(model.volatility_regime_adjustment.halflife, horizon),
            specific_volatility_halflife=_pick(specific.ewma_halflife, horizon),
            specific_newey_west_lags=_pick(specific.newey_west_lags, horizon),
            specific_autocorrelation_halflife=_pick(specific.autocorrelation_halflife, horizon),
            bayesian_shrinkage_q=_pick(specific.bayesian_shrinkage_q, horizon),
            stages=covariance.stages,
            psd_eigenvalue_floor=numerics.psd_eigenvalue_floor,
            psd_reconstruction_roundings=numerics.psd_reconstruction_roundings,
            symmetry_absolute_tolerance=numerics.symmetry_absolute_tolerance,
            seed=model.seed,
        )

    @classmethod
    def load(cls, *, horizon: Horizon, config: Config | None = None) -> RiskConfig:
        """Read ``config/model.yaml`` and project it onto one horizon."""
        return cls.from_model((config or load()).model, horizon=horizon)
