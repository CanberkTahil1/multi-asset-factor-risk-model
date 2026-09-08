"""Covariance pipeline, specific risk and bias-statistic validation.

CLAUDE.md invariant 10: this package MUST NOT know what asset class it is
looking at. It takes ``(factor_returns, exposures, residuals, config)`` and
returns matrices. That is what lets the week-7 cross-sectional module reuse it
whole. See SPEC.md sections 5, 6 and 15.2.

The corollary, which is the half that will actually get violated: **nothing here
may import from ``mafrm.factors``.** Reaching sideways for a helper that already
exists always looks harmless at the moment it is done, and it is how the
asset-class assumption gets in. Shared numerics live in :mod:`mafrm.numerics`,
beneath both packages, so there is no edge between them in either direction.
``tests/test_risk_architecture.py`` asserts this mechanically.
"""

from mafrm.risk.checks import PsdCheck, PsdError, assert_psd
from mafrm.risk.config import HORIZONS, Horizon, RiskConfig
from mafrm.risk.covariance import (
    CovarianceBuild,
    CovarianceError,
    build_covariance,
    implemented_stages,
    missing_stages,
    run_pipeline,
)
from mafrm.risk.regime import (
    ForecastHistory,
    RegimeError,
    RegimeMultiplier,
    cross_sectional_bias,
    forecast_history,
    regime_multiplier,
)

__all__ = [
    "HORIZONS",
    "CovarianceBuild",
    "CovarianceError",
    "ForecastHistory",
    "Horizon",
    "PsdCheck",
    "PsdError",
    "RegimeError",
    "RegimeMultiplier",
    "RiskConfig",
    "assert_psd",
    "build_covariance",
    "cross_sectional_bias",
    "forecast_history",
    "implemented_stages",
    "missing_stages",
    "regime_multiplier",
    "run_pipeline",
]
