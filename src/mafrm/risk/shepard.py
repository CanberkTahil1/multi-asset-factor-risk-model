"""Shepard's second-order risk. SPEC.md 6.4. Shepard (2009), arXiv:0908.2455.

ONE FORMULA, TWO READINGS, AND THE UNIT TRAVELS WITH THE NUMBER
----------------------------------------------------------------

::

    Sigma^2_SO   = E(w' Omega w) . [1 - N/T]^-2       asset-level covariance   (Eq. 13)
    Sigma^2_SO,f = [1 - K/T]^-2 . w' X F X' w         factor model             (Eq. 32)
    T_eff        = 2 tau / ln 2                        EWMA half-life -> T      (Eq. 33)

Both equations write ``[1 - K/T]^-2`` against a **variance**. The realised
variance of a portfolio optimised on a sample covariance exceeds its forecast
variance by that factor; the realised *volatility* exceeds its forecast by the
square root, ``[1 - K/T]^-1``. MWO's own cross-check in SPEC.md 6.4 fixes the
reading: at ``T = 100, N = 50`` Shepard "predicts 2.0" against an observed bias
statistic of 2.25, and ``(1 - 0.5)^-1 = 2.0`` while ``(1 - 0.5)^-2 = 4.0`` -- the
published comparison is on volatility and uses the square root.

SPEC.md 6.4's own worked table then applies ``[1 - K/T]^-2 - 1`` directly to
volatility ("5.1% vol understatement" at ``K/T = 0.025``), which is the variance
multiplier read as a volatility one. That convention runs through the project's
earlier reports and is **kept, labelled**, rather than silently corrected: W4-P2b
found that a lambda amplitude quoted without its unit produced a contradiction
between two sessions, and the same number quoted in two units differs here by a
factor of two.

So :class:`SecondOrderRisk` never emits a bare figure. Every accessor that
returns an understatement takes a :class:`Unit`, ``format()`` on the object
refuses, and :meth:`SecondOrderRisk.table_convention` is the one place the
SPEC.md 6.4 convention lives, named for what it is.

WHICH ``T_eff`` THIS IS
-----------------------

W3-P3b (``experiments.md`` rows 109-111) measured ``2 tau / ln 2`` transferring
at 1.007x for an EWMA **second moment** and at 1.31x after correlation
normalisation, with the excess depending on the spectrum. Shepard's multiplier is
a variance-scaling correction, so the second-moment case is the one in force
here and the formula transfers; the normalised case is the eigenfactor stage's
business and it simulates its own estimator rather than reading a ``T`` off this
formula (SPEC.md 5.3.3). :mod:`mafrm.risk.second_order` measures both legs
rather than asserting either.

THIS IS THE ONE IMPLEMENTATION
------------------------------

Before W4-P3 the closed form was written out in four places -- the covariance
build, the eigenfactor report, the half-life report and the bias report -- and
two of them had already drifted in what they called the result. Every use site
now reads it from here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from mafrm.numerics import effective_sample_size

__all__ = [
    "SecondOrderRisk",
    "ShepardError",
    "Unit",
    "jensen_term",
    "second_order_risk",
]


class ShepardError(ValueError):
    """A second-order risk figure was asked for without a unit, or from bad inputs."""


class Unit(StrEnum):
    """The two things ``[1 - K/T]^-2`` can multiply. Every emitted figure names one."""

    #: The multiplier as Eq. 13 and Eq. 32 write it: on a portfolio VARIANCE.
    VARIANCE = "variance"
    #: The square root of the same: on a portfolio VOLATILITY, which is the unit
    #: of the bias statistic ``B`` and the unit MWO's "2.0 at T = 100" is in.
    VOLATILITY = "volatility"


@dataclass(frozen=True)
class SecondOrderRisk:
    """``[1 - p/T_eff]^-2`` for ``p`` parameters at ``T_eff`` effective observations.

    ``p`` is ``K`` for a factor model (Eq. 32) and ``N`` for an asset-level
    sample covariance (Eq. 13) -- one formula, two ranks, which is the comparison
    SPEC.md 6.4's table is built on and the content of hypothesis H4.
    """

    parameters: int
    effective_observations: float

    def __post_init__(self) -> None:
        if self.parameters < 1:
            raise ShepardError(
                f"SecondOrderRisk: parameters must be positive, got {self.parameters}"
            )
        if not self.effective_observations > 0.0:
            raise ShepardError(
                f"SecondOrderRisk: effective observations must be positive, got "
                f"{self.effective_observations}"
            )

    @property
    def ratio(self) -> float:
        """``p / T_eff``. The quantity everything here is a function of."""
        return self.parameters / self.effective_observations

    @property
    def variance_multiplier(self) -> float:
        """``[1 - p/T]^-2``. Eq. 13 / Eq. 32 as written. ``inf`` once ``p >= T``."""
        if self.ratio >= 1.0:
            return math.inf
        return float((1.0 - self.ratio) ** -2.0)

    @property
    def volatility_multiplier(self) -> float:
        """``[1 - p/T]^-1``, the square root -- the unit ``B`` is measured in."""
        if self.ratio >= 1.0:
            return math.inf
        return float((1.0 - self.ratio) ** -1.0)

    def multiplier(self, unit: Unit) -> float:
        """The realised-over-forecast ratio, in ``unit``."""
        if unit is Unit.VARIANCE:
            return self.variance_multiplier
        if unit is Unit.VOLATILITY:
            return self.volatility_multiplier
        raise ShepardError(f"SecondOrderRisk: unknown unit {unit!r}")  # pragma: no cover

    def understatement(self, unit: Unit) -> float:
        """``multiplier - 1``: the fraction by which the forecast is too small, in ``unit``."""
        return self.multiplier(unit) - 1.0

    @property
    def table_convention(self) -> float:
        """SPEC.md 6.4's worked-table figure: the VARIANCE multiplier minus one.

        The spec's table labels this "vol understatement". It is the number every
        report before W4-P3 quoted, and it is kept under this name so the
        superseded figures stay reproducible and recognisable -- not because it is
        a volatility understatement, which it is not. Use
        ``understatement(Unit.VOLATILITY)`` for that.
        """
        return self.variance_multiplier - 1.0

    def render(self, unit: Unit, *, decimals: int = 1) -> str:
        """``"5.1% (variance)"`` -- the unit is part of the string, always."""
        value = self.understatement(unit)
        if math.isinf(value):
            return f"inf ({unit.value}; p >= T_eff)"
        return f"{100.0 * value:.{decimals}f}% ({unit.value})"

    def __format__(self, spec: str) -> str:
        raise ShepardError(
            "SecondOrderRisk cannot be formatted without a unit. Use .render(Unit.VARIANCE) or "
            ".render(Unit.VOLATILITY), or .understatement(unit) -- the two differ by a factor of "
            "two and a bare figure is how the W4-P2b lambda-amplitude contradiction happened."
        )

    def __str__(self) -> str:
        return self.__format__("")

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, SecondOrderRisk):
            return NotImplemented
        return (self.parameters, self.effective_observations) == (
            other.parameters,
            other.effective_observations,
        )

    def __hash__(self) -> int:
        return hash((self.parameters, self.effective_observations))


def second_order_risk(
    parameters: int,
    *,
    halflife: float | None = None,
    effective_observations: float | None = None,
) -> SecondOrderRisk:
    """Shepard's multiplier for ``parameters`` at a half-life OR an effective size.

    Exactly one of the two must be given. ``halflife`` is converted with Eq. 33,
    ``T_eff = 2 tau / ln 2``, through :func:`mafrm.numerics.effective_sample_size`
    -- the asymptotic figure SPEC.md 6.4's table quotes. Pass
    ``effective_observations`` to use a realised Kish size instead, which is the
    honest denominator on a window short relative to its half-life.
    """
    if (halflife is None) == (effective_observations is None):
        raise ShepardError(
            "second_order_risk: give exactly one of halflife= or effective_observations="
        )
    if halflife is not None:
        effective_observations = effective_sample_size(float(halflife))
    assert effective_observations is not None
    return SecondOrderRisk(
        parameters=int(parameters), effective_observations=effective_observations
    )


def jensen_term(
    *, halflife: float | None = None, effective_observations: float | None = None
) -> SecondOrderRisk:
    """Shepard at ``p = 1``: ``[1 - 1/T]^-2 ~ 1 + 2/T``, the inverse-variance Jensen bias.

    With one asset there is nothing to optimise, and the whole of the
    second-order risk is ``E[sigma^2 / sigma_hat^2] - 1 ~ 2/T`` -- the bias of a
    forecast variance's reciprocal. It is also the only part of the estimation
    error a per-factor cross-sectional bias statistic can see, because SPEC.md
    5.4's ``B_t^F`` averages ``f_kt^2 / sigma_kt^2`` over the ``K`` fixed factor
    axes and never over an optimiser's direction. That makes this term the
    estimation-error side of the Shepard-VRA overlap; W4-P3's simulation
    (:mod:`mafrm.risk.second_order`) measures it as condition (E).
    """
    return second_order_risk(1, halflife=halflife, effective_observations=effective_observations)
