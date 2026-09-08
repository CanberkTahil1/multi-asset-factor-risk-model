"""``reports/specific_risk.md`` and ``reports/specific_residual_corr.md``. SPEC.md 5.5.

WHY THIS LIVES IN ``factors/`` AND NOT IN ``risk/``
---------------------------------------------------

Same reason as :mod:`mafrm.factors.vra_report` and
:mod:`mafrm.factors.eigenfactor_report`. The four legs live in
:mod:`mafrm.risk.specific`, which knows nothing about what its columns are;
*this* module reads Model A's residual panel, maps each asset onto its universe
sleeve and knows which two assets are exact linear combinations of the factor
set, so it cannot live under ``risk/`` without breaking CLAUDE.md invariant 10.

THERE IS NO CACHE HERE, AND THAT IS THE POINT
----------------------------------------------

SPEC.md 5.4's specific leg needs a forecast history for the same reason its
factor leg does -- each ``B_t^S`` is standardised by the forecast made at
``t-1`` -- so it runs the pipeline once per date. The factor leg's history costs
ninety minutes and is therefore cached, checkpointed and committed; **this one
costs about nine seconds per variant**, because SPEC.md 5.5 has no Monte Carlo
in it. So it is rebuilt every time and there is no staleness hazard to manage.

Worth stating rather than leaving as an absence: the expensive machinery in
:mod:`mafrm.history` was extracted precisely so that a caller that needs it can
have it, and a caller that does not should not carry it.

WHAT THE REPORT IS FOR
-----------------------

SPEC.md 5.5 at ``N = 13`` is mostly a **negative** result, and the report is
written to make that legible rather than to bury it in a table of forecasts:
four legs implemented as published, three of them degenerate here, and the two
places where a degeneracy reaches the production number rather than merely
idling. The forecasts are in it too, but they are not the finding.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from mafrm import config
from mafrm.factors import hybrid
from mafrm.risk import specific
from mafrm.risk.config import HORIZONS, Horizon, RiskConfig

__all__ = [
    "Inputs",
    "SpecificRun",
    "bucket_contamination",
    "inputs",
    "main",
    "newey_west_absorption",
    "render_residual_correlation",
    "render_specific_risk",
    "run",
]

_REPORTS: Final[Path] = Path(__file__).resolve().parents[3] / "reports"
_RISK_PATH: Final[Path] = _REPORTS / "specific_risk.md"
_CORRELATION_PATH: Final[Path] = _REPORTS / "specific_residual_corr.md"

#: The two linked pairs SPEC.md 5.5 names in its closing paragraph, by VENDOR
#: SYMBOL: *"SPY/IWM residuals are not independent, nor are LQD/HYG"*. They are
#: resolved against ``config/universe.yaml``'s tickers rather than written as
#: universe ids, so that renaming an id cannot silently drop a prediction the
#: specification made. Not parameters -- they are the spec's own worked examples,
#: and the report exists to check them.
_SPEC_LINKED_TICKERS: Final[tuple[tuple[str, str], ...]] = (("SPY", "IWM"), ("LQD", "HYG"))


@dataclass(frozen=True)
class Inputs:
    """Model A's residual panel, dressed for SPEC.md 5.5's signature."""

    residuals: pd.DataFrame
    exposures: pd.DataFrame
    buckets: pd.Series
    #: Assets excluded from the specific VRA's cross-section AND from stage
    #: (d)'s shrinkage target. The W4-P2 forward constraint as amended in W4-P1b;
    #: see :func:`inputs` for why the two exclusions are the same set and why
    #: neither is an exclusion from the model.
    excluded: tuple[str, ...]
    #: Sleeves in ``config/universe.yaml`` with no member in the residual panel.
    empty_buckets: tuple[str, ...]
    panel: hybrid.ResidualPanel


def inputs(settings: config.Config | None = None) -> Inputs:
    """Residuals, the ``N x K`` design, and one bucket label per asset.

    THE BUCKETS ARE THE SIX SLEEVES, UNMAPPED -- A RULING
        SPEC.md 5.5 names five classes for ``s_n``. ``config/universe.yaml``
        declares six sleeves and they are used **as they are**. Forcing this
        project's own documented partition onto a list written for a different
        universe would be the sixth time in this build that a published shape has
        been fitted to a panel it was not written for, and every previous time
        the finding was that the shape did not transfer. The sleeves are dated,
        frozen and carry a rationale per line; SPEC.md's five are an illustration.

    THE EXCLUSION IS THE W4-P2 FORWARD CONSTRAINT, DERIVED RATHER THAN LISTED
        ``hy_credit`` and ``commodity`` regress against the factor set at
        ``R^2 = 1.000`` because the proxy IS the factor. They are read from
        ``factors.macro.credit.asset_id`` and ``factors.macro.commodity.asset_id``
        -- the same derivation ``betas.diagnostics`` uses for
        ``spans_a_factor`` -- so renaming an asset cannot orphan the constraint.

    THE CONSTRAINT WAS AMENDED IN W4-P1b, AND THE AMENDMENT IS THE POINT
        As registered in W2-P2 it excluded them from **bias aggregation** only.
        W4-P1's measurement showed that is not sufficient: stage (d) shrinks
        toward a bucket mean, so a value that is a construction artefact has
        already propagated into every other member of its bucket **before** any
        average is taken. Excluding a corrupted value from a mean does not
        uncorrupt the values it corrupted.

        So the same set is now held out of stage (d)'s ``sigma_bar`` and
        ``sigma_delta`` as well. The justification is row 119's exactly: their
        residuals are ~0 by construction, so their specific volatility is a
        construction artefact rather than an estimate, and a shrinkage target
        built partly from one is a target built partly from a manufactured
        number. It **removes a manufactured number and substitutes no estimate**,
        which is why no result can reverse it and why the row is
        ``data-diagnostic``.

        They stay in the panel, are still bucketed, still shrunk and still
        reported -- labelled, as they are everywhere else.
    """
    loaded = settings or config.load()
    panel = hybrid.residual_panel(config=loaded)
    residuals = panel.residuals
    exposures = panel.exposure_matrix(pd.Timestamp(residuals.index[-1]))
    sleeves = {asset.id: asset.sleeve for asset in loaded.universe.assets}
    buckets = pd.Series([sleeves[str(name)] for name in residuals.columns], index=residuals.columns)
    scheme = loaded.model.factors.macro
    spans = {scheme.credit.asset_id, scheme.commodity.asset_id}
    present = set(buckets)
    return Inputs(
        residuals=residuals,
        exposures=exposures,
        buckets=buckets,
        excluded=tuple(sorted(name for name in residuals.columns if name in spans)),
        empty_buckets=tuple(sorted({asset.sleeve for asset in loaded.universe.assets} - present)),
        panel=panel,
    )


def _minimum_observations(risk: RiskConfig) -> int:
    """``lags + 1``. An ARITHMETIC floor, not a burn-in choice.

    The same rule, and the same reasoning, as
    :func:`mafrm.factors.vra_report._minimum_observations`: below ``lags + 1``
    SPEC.md 5.5(a)'s longest Bartlett lag has no pairs at all, which is the point
    where the stage has no answer rather than a poor one. SPEC.md 5.1.2 leaves
    the actual burn-in with the caller and no bound is published, so **nothing is
    filtered** -- the window each forecast was made on is carried instead.

    The factor leg's floor also carries a ``K + 1`` term for SPEC.md 5.3's Monte
    Carlo, which re-estimates a ``K x K`` matrix. Nothing in SPEC.md 5.5 inverts
    a matrix over the time dimension, so that term has no analogue here; leg
    (b)'s cross-sectional identification is a constraint on ``N``, not on ``T``,
    and :func:`mafrm.risk.specific.structural_fit` enforces it directly.
    """
    return risk.specific_newey_west_lags + 1


@dataclass(frozen=True)
class SpecificRun:
    """One horizon, built end to end with SPEC.md 5.4's specific leg applied."""

    horizon: str
    risk: RiskConfig
    build: specific.SpecificRiskBuild
    history: specific.SpecificForecastHistory
    #: The pre-VRA build, kept because the VRA is a level scaling and the report
    #: quotes both sides of it.
    pre_regime: specific.SpecificRiskBuild

    @property
    def frame(self) -> pd.DataFrame:
        """One row per asset: every leg, side by side, in the order SPEC.md fixes."""
        build = self.build
        return pd.DataFrame(
            {
                "bucket": list(build.shrinkage.buckets.labels),
                "sigma_TS": build.time_series.deviation,
                "nw_ratio": build.time_series.correction_ratio,
                "sigma_STR": build.structural.deviation,
                "gamma": build.blend.gamma,
                "Z": build.blend.z,
                "sigma_hat": build.blended,
                "v": build.shrinkage.v,
                "sigma_SH": build.shrinkage.deviation,
                "sigma_final": build.forecast,
            },
            index=list(build.columns),
        )


def run(horizon: Horizon, data: Inputs, settings: config.Config | None = None) -> SpecificRun:
    """SPEC.md 5.5's four legs plus SPEC.md 5.4's specific multiplier, at one horizon."""
    loaded = settings or config.load()
    risk = RiskConfig.load(horizon=horizon, config=loaded)
    # ONE set, used for two different exclusions, and that is deliberate: the
    # W4-P2 constraint as amended in W4-P1b keeps the identity assets out of the
    # bias cross-section AND out of stage (d)'s shrinkage target. Both are derived
    # from `data.excluded` here so the two cannot drift apart.
    include = [name for name in data.residuals.columns if name not in set(data.excluded)]
    history = specific.specific_forecast_history(
        data.residuals,
        data.exposures,
        data.buckets,
        risk,
        minimum_observations=_minimum_observations(risk),
        include=include,
        shrinkage_target=include,
    )
    return SpecificRun(
        horizon=horizon,
        risk=risk,
        build=specific.build_specific_risk(
            data.residuals,
            data.exposures,
            data.buckets,
            risk,
            regime_bias=history.bias,
            shrinkage_target=include,
        ),
        history=history,
        pre_regime=specific.build_specific_risk(
            data.residuals, data.exposures, data.buckets, risk, shrinkage_target=include
        ),
    )


# ---------------------------------------------------------------------------
# The two controls, both pre-registered in experiments.md before they were run
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Absorption:
    """Control C1: how much of SPEC.md 5.5(a)'s correction does SPEC.md 5.4 undo?"""

    horizon: str
    lambda_with_correction: float
    lambda_without_correction: float
    correction_ratio_low: float
    correction_ratio_high: float
    correction_ratio_median: float

    @property
    def ratio(self) -> float:
        """``lambda_S(lags=0) / lambda_S(lags=5)``."""
        return self.lambda_without_correction / self.lambda_with_correction

    @property
    def inside_the_registered_range(self) -> bool:
        """The pre-registered falsifier: outside the per-asset range refutes it."""
        return self.correction_ratio_low <= self.ratio <= self.correction_ratio_high


def newey_west_absorption(data: Inputs, run_at: SpecificRun) -> Absorption:
    """Rebuild the history with ``lags = 0`` and compare. A CONTROL, not a candidate.

    ``lags = 0`` is not a configuration this model could adopt -- the count comes
    from USE4 Table 4.1 through ``config/model.yaml`` and no result here would
    change it. It is the eigenfactor-off control of ``experiments.md`` rows
    116-117, applied to a different pair of stages: run the pipeline with one
    stage disabled to measure what the stage after it absorbed.
    """
    risk = run_at.risk
    disabled = dataclasses.replace(risk, specific_newey_west_lags=0)
    include = [name for name in data.residuals.columns if name not in set(data.excluded)]
    history = specific.specific_forecast_history(
        data.residuals,
        data.exposures,
        data.buckets,
        disabled,
        # The SAME burn-in as the live run, taken from the live config. Rebuilding
        # it from `disabled` would start the control one date earlier and compare
        # two multipliers fitted on different windows.
        minimum_observations=_minimum_observations(risk),
        include=include,
    )
    ratios = run_at.build.time_series.correction_ratio
    return Absorption(
        horizon=run_at.horizon,
        lambda_with_correction=run_at.build.regime.lambda_,  # type: ignore[union-attr]
        lambda_without_correction=history.multiplier(
            halflife=risk.volatility_regime_halflife
        ).lambda_,
        correction_ratio_low=float(ratios.min()),
        correction_ratio_high=float(ratios.max()),
        correction_ratio_median=float(np.median(ratios)),
    )


def bucket_contamination(data: Inputs, run_at: SpecificRun) -> pd.DataFrame:
    """Control C2, now a CHECK ON THE FIX rather than a measurement of a defect.

    C2 was registered and run in W4-P1 against a production path that let the
    identity assets into stage (d)'s bucket mean. It HELD: `gold` and `ig_credit`
    were being pulled down 4.6-4.8%. W4-P1b acted on that finding by holding the
    identity assets out of the target, and this comparison now carries a
    **stronger property** than it did when it was a diagnostic.

    The right-hand column rebuilds stage (d) with the identity assets removed
    from the universe **entirely**. The production path merely holds them out of
    the *target*, keeping them bucketed, shrunk and reported. For every retained
    asset those two must now agree **exactly** -- a target computed from a set is
    the same number whether the non-contributors are present or absent -- so this
    function has stopped measuring contamination and started asserting there is
    none. `tests/test_specific_risk.py` pins the equality, and a non-zero entry in
    `change_pct` means the exclusion is not doing what W4-P1b claims it does.
    """
    kept = [name for name in data.residuals.columns if name not in set(data.excluded)]
    lean = specific.build_specific_risk(
        data.residuals[kept],
        data.exposures.loc[kept],
        data.buckets.loc[kept],
        run_at.risk,
    )
    with_them = pd.Series(run_at.build.shrinkage.deviation, index=list(run_at.build.columns))
    without = pd.Series(lean.shrinkage.deviation, index=list(lean.columns))
    affected = [
        name
        for name in kept
        if data.buckets[name] in {data.buckets[other] for other in data.excluded}
    ]
    return pd.DataFrame(
        {
            "bucket": [data.buckets[name] for name in affected],
            "with_identity_assets": with_them[affected].to_numpy(),
            "without": without[affected].to_numpy(),
            "change_pct": (without[affected] / with_them[affected] - 1.0).to_numpy() * 100.0,
        },
        index=affected,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _table(frame: pd.DataFrame, *, index_name: str, floats: str = "{:.4f}") -> str:
    header = f"| {index_name} | " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "|---" * (len(frame.columns) + 1) + "|"
    rows = [
        f"| `{index}` | "
        + " | ".join(value if isinstance(value, str) else floats.format(value) for value in row)
        + " |"
        for index, row in zip(frame.index, frame.to_numpy(), strict=True)
    ]
    return "\n".join([header, rule, *rows])


def render_specific_risk(
    data: Inputs,
    runs: dict[str, SpecificRun],
    absorption: dict[str, Absorption],
    contamination: dict[str, pd.DataFrame],
) -> str:
    """``reports/specific_risk.md``."""
    reference = runs[HORIZONS[0]]
    counts = reference.build.shrinkage.buckets.counts()
    residuals = data.residuals
    lines: list[str] = []
    add = lines.append

    add("# Specific risk — SPEC.md 5.5 at N = 13")
    add("")
    add(
        f"Generated by `mafrm.factors.specific_report`. Model A's residuals, "
        f"**{residuals.shape[1]} assets x {residuals.shape[0]:,} dates**, "
        f"{residuals.index[0].date()}..{residuals.index[-1].date()}, basis points per day. "
        f"{data.panel.dates_dropped_by_completeness} date(s) dropped to make the panel "
        "complete-cases."
    )
    add("")
    add(
        "**Everything here ends strictly before `sample.holdout_start`.** The residual panel "
        "inherits the boundary from the exposure panel it is built from, which passes "
        "`Config.require_holdout_start()` into every leg of its construction."
    )
    add("")
    add("## The headline: four legs, three of them degenerate here")
    add("")
    add(
        "SPEC.md 5.5 is a large-`N` cross-sectional construction. Run faithfully at `N = 13` "
        "it becomes the **fifth published technique this project has measured out of its own "
        "regime**, after Marchenko-Pastur denoising, the PSD repair, the parabola inside the "
        "eigenfactor stage and the Bartlett correction — and the most structural of them, "
        "because it is a whole section rather than one stage's machinery. "
        "`reports/stage_k_dependence.md` carries it beside the other four."
    )
    add("")
    add("| Leg | SPEC.md | What it needs | What happens at `N = 13` |")
    add("|---|---|---|---|")
    add(
        "| (a) time series | 5.5(a) | Only a time dimension | **Works.** `K_0` at the "
        "volatility half-life, Bartlett lag terms at the autocorrelation half-life |"
    )
    add(
        f"| (b) structural | 5.5(b) | `N` well above `K` | **Barely identified.** "
        f"{residuals.shape[1]} observations against "
        f"{reference.build.structural.coefficients.size} parameters, "
        f"**{reference.build.structural.residual_degrees_of_freedom} residual d.o.f.** |"
    )
    add(
        "| (c) blend | 5.5(c) | Ragged histories to discriminate between | **Pinned at "
        "`gamma = 1` by ruling.** The panel is complete, and the fatness term is *inverted* "
        "here — see below |"
    )
    add(
        "| (d) shrinkage | 5.5(d) | Populated buckets of ESTIMATES | **Vacuous on 3 of 5.** A "
        "two-member bucket pins `v` at `1/(1+q)` regardless of data; and after W4-P1b holds the "
        "identity assets out of the target, `commodity`, `credit` and `inflation` all have "
        "one-contributor targets. Only `equity` and `government` are live |"
    )
    add("")

    add("## Ruling: `gamma_n` is pinned at 1, and the reason is an inversion")
    add("")
    add(
        "SPEC.md 5.5(c) specifies `gamma_n` qualitatively and names no constants; "
        "`config/model.yaml` has none. CLAUDE.md invariant 9 forbids inventing them, so the "
        "route taken was to show the weight **saturates**, making the constants irrelevant. "
        "It half worked."
    )
    add("")
    add("| Term | Saturates? | Evidence |")
    add("|---|---|---|")
    for horizon in HORIZONS:
        blend = runs[horizon].build.blend
        add(
            f"| missing observations ({horizon}) | **yes** | complete panel, "
            f"`min h_n` = {blend.minimum_observations:,} of {blend.periods:,}; any ramp "
            "ceiling at or below that returns 1 |"
        )
    for horizon in HORIZONS:
        blend = runs[horizon].build.blend
        worst = int(np.argmax(blend.z))
        add(
            f"| distribution fatness ({horizon}) | **NO** | `max Z` = "
            f"{blend.maximum_z:.4f} on `{runs[horizon].build.columns[worst]}` |"
        )
    add("")
    add(
        "The premise that complete histories saturate both terms conflates two different "
        "things: completeness is a statement about *how many* observations there are and "
        "fatness about *what they look like*, and a complete history of a fat-tailed series "
        "is still fat-tailed. Measured over every date on an expanding window rather than "
        "only at the last one, `Z` reaches **7.61** (short, `hy_credit`, 2020-04-06) and "
        "**5.57** (long, `hy_credit`, 2013-04-04); 4 of 13 assets breach `Z > 1` and 13 of 13 "
        "breach `Z > 0.2`."
    )
    add("")
    add(
        "**So the constants could not be dissolved, and the ruling is not that the term is "
        "out of regime — it is that the term is inverted here.** `gamma_n < 1` routes weight "
        "off leg (a) and onto leg (b). That is worth doing when the structural estimate is "
        "better identified; at `N = 13` it is the worse one, so the mechanism moves weight "
        "the wrong way *even implemented exactly as published*. The clinching detail is "
        "**where** it fires — March-April 2020 — which is precisely where leg (a) is "
        "fat-tailed and therefore precisely where leg (b) is fitted on the same stressed "
        "data. Constants would not have rescued it."
    )
    add("")
    add(
        "`Z` is **reported rather than discarded**: it correctly identifies where the "
        "time-series estimate is unreliable, which is real information about this panel "
        "whether or not a weight is computed from it. It is in the per-asset table below. "
        "The code path is kept and both diagnostics run live, so a larger-`N` module can turn "
        "the blend on where it belongs; that prediction is registered in `experiments.md` "
        "before the panel it concerns exists."
    )
    add("")

    add("## Weighting and buckets, stated because both are rulings")
    add("")
    add(
        "**Every cross-sectional mean here is EQUALLY weighted.** SPEC.md 5.4 and 5.5 ask for "
        "notional weighting and **no notionals exist**: no portfolio has been constructed at "
        "this point in the project, and `config/universe.yaml` is right not to invent any. "
        "Equal weighting is the *absence* of a weighting rather than a chosen one. A later "
        "session with real holdings that revisits this owes a `model-config` row at the "
        "moment of switching."
    )
    add("")
    add(
        "**The buckets are the six sleeves of `config/universe.yaml`, unmapped.** SPEC.md 5.5 "
        "names five classes; this project's own partition is dated, frozen and carries a "
        "rationale per line, and forcing it onto a list written for a different universe is "
        "the error this build has now found six times."
    )
    add("")
    add(
        "**Members and target contributors are different counts, and W4-P1b is why.** Every "
        "asset is bucketed, shrunk and reported. Which values are allowed to *build* the "
        "target `sigma_bar` and `sigma_delta` is a separate question, and the two identity "
        "assets are held out of it — see the amendment below."
    )
    add("")
    add("| Bucket | Members | Target contributors | Stage (d) |")
    add("|---|---|---|---|")
    targets = reference.build.shrinkage.buckets.target_counts()
    for label in reference.build.shrinkage.buckets.distinct:
        size = counts[label]
        contributors = targets[label]
        if contributors == 0:
            verdict = "**NO-OP** — no contributor left; each member is its own target"
        elif contributors == 1:
            verdict = "**NO-OP** — `sigma_bar = sigma_hat`, nothing to shrink toward"
        elif contributors == 2:
            verdict = (
                "**VACUOUS** — `d = sigma_delta` identically, so `v = 1/(1+q)` whatever the data"
            )
        else:
            verdict = "informative"
        add(f"| `{label}` | {size} | {contributors} | {verdict} |")
    for label in data.empty_buckets:
        add(f"| `{label}` | 0 | 0 | no member in the residual panel |")
    add("")
    add(
        "**The two-member result is exact, not empirical.** In a bucket of two, each member's "
        "distance from the mean is half the gap between them and the population dispersion is "
        f"*also* half the gap, so `d = sigma_delta` identically and `v = 1/(1+q) = "
        f"{1.0 / (1.0 + reference.risk.bayesian_shrinkage_q):.4f}` for **every** pair of "
        "values. Stage (d) at two members carries no information about the data at all: it is "
        "a fixed proportional pull toward the bucket mean. "
        "`tests/test_specific_risk.py` pins it."
    )
    add("")

    for horizon in HORIZONS:
        current = runs[horizon]
        add(f"## {horizon.title()} horizon")
        add("")
        for line in current.build.render():
            add(f"    {line}")
        add("")
        add(
            _table(
                current.frame,
                index_name="asset",
                floats="{:.4f}",
            )
        )
        add("")
        add(
            "`sigma_final = lambda_S x sigma_SH`. Units are basis points per day. "
            "`gamma` is pinned (see above), so `sigma_hat = sigma_TS` exactly and "
            "`sigma_STR` is computed but never reaches the forecast."
        )
        add("")

    add("## Where a degeneracy reaches the production number")
    add("")
    add(
        "Two of the three degeneracies above are idle. These two are not, and both were "
        "measured with the prediction registered in `experiments.md` before the run."
    )
    add("")
    add("### C1 — the specific VRA absorbs most of the Newey-West correction")
    add("")
    add(
        "SPEC.md 5.5(a)'s Bartlett term lowers `sigma^TS`; SPEC.md 5.4's multiplier is fitted "
        "to realised data and therefore raises the level back. Rebuilding the history with "
        "`lags = 0` — a **control**, not a candidate configuration — measures how much of the "
        "first stage the second one undoes."
    )
    add("")
    add(
        "| horizon | `lambda_S` (lags=5) | `lambda_S` (lags=0) | ratio | per-asset NW range | "
        "median | registered falsifier |"
    )
    add("|---|---|---|---|---|---|---|")
    for horizon in HORIZONS:
        item = absorption[horizon]
        add(
            f"| {horizon} | {item.lambda_with_correction:.6f} | "
            f"{item.lambda_without_correction:.6f} | **{item.ratio:.4f}** | "
            f"[{item.correction_ratio_low:.4f}, {item.correction_ratio_high:.4f}] | "
            f"{item.correction_ratio_median:.4f} | "
            f"{'**HOLDS**' if item.inside_the_registered_range else '**REFUTED**'} |"
        )
    add("")
    add(
        "**This is CLAUDE.md failure mode 7 on the production path — two corrections that "
        "each pass their own sanity check and then cancel.** Read the signs: without the "
        "Bartlett term the model *over*-forecasts and the VRA marks it down "
        f"(`lambda_S` = {absorption['short'].lambda_without_correction:.3f} short); with it "
        "the model *under*-forecasts and the VRA marks it up "
        f"({absorption['short'].lambda_with_correction:.3f}). At the long horizon the "
        f"absorption is near-exact — a ratio of {absorption['long'].ratio:.4f} against a "
        f"median correction of {absorption['long'].correction_ratio_median:.4f}, a gap of "
        f"{absorption['long'].ratio - absorption['long'].correction_ratio_median:+.4f}."
    )
    add("")
    add(
        "It does **not** follow that the Bartlett term is worthless: the two stages correct "
        "different objects, and the VRA is fitted rather than derived, so what is measured "
        "here is that the fitted stage has already priced in whatever level the stage before "
        "it left. That is the same relationship SPEC.md 5.4.2 recorded between the VRA and "
        "the eigenfactor stage, arriving a second time on a different pair."
    )
    add("")
    add("### C2 — the identity assets contaminated their bucket-mates, and no longer do")
    add("")
    add(
        f"`{'`, `'.join(data.excluded)}` regress against the factor set at `R^2 = 1.000` "
        "because the proxy **is** the factor, so their specific volatility is a construction "
        "artefact rather than an estimate. The W4-P2 forward constraint, registered in W2-P2, "
        "excluded them from **bias aggregation**. W4-P1 measured that this is **not "
        "sufficient**: stage (d) shrinks toward a bucket mean, so an artefact propagates into "
        "every other member of its bucket *before* any average is taken. **Excluding a "
        "corrupted value from a mean does not uncorrupt the values it corrupted.**"
    )
    add("")
    add(
        "**W4-P1b amended the constraint rather than patching the symptom.** The same two "
        "assets are now held out of `sigma_bar` and `sigma_delta` as well. The justification "
        "is `experiments.md` row 119's exactly — it removes a manufactured number and "
        "substitutes no estimate, so no result can reverse it. They stay in the panel, stay "
        "bucketed, stay shrunk and stay reported."
    )
    add("")
    add("What it was worth, at the short horizon, measured before the amendment:")
    add("")
    add("| asset | before W4-P1b | after | |")
    add("|---|---|---|---|")
    add("| `gold` | 65.4043 | **68.5084** | +4.75%, a construction artefact removed |")
    add("| `ig_credit` | 9.4896 | **9.9289** | +4.63% |")
    add("| `commodity` | 3.3206 | **0.2165** | **-93.5%** — the +1434% distortion is gone |")
    add("| `hy_credit` | 0.7025 | **0.2631** | -62.5% |")
    add("")
    add(
        "**C2 has changed job as a result, and now asserts rather than measures.** The table "
        "below rebuilds stage (d) with the identity assets removed from the universe "
        "*entirely*, against a production path that merely holds them out of the *target*. A "
        "target computed from a set is the same number whether the non-contributors are "
        "present or absent, so for every retained asset these must now agree **exactly** — a "
        "non-zero entry means the exclusion is not doing what W4-P1b claims. "
        "`tests/test_specific_risk.py` pins the same equality on synthetic input."
    )
    add("")
    for horizon in HORIZONS:
        add(f"**{horizon}**")
        add("")
        add(_table(contamination[horizon], index_name="asset", floats="{:.10f}"))
        add("")
    add(
        "**What is left, and it is not nothing.** The amendment costs stage (d) two of its "
        "five buckets: `commodity` and `credit` each had two members, one of which was an "
        "identity asset, so both now have one-contributor targets and are no-ops. Stage (d) "
        "is live on `equity` and `government` alone. That is the honest position — the "
        "alternative was a target built partly from a manufactured number — but it means "
        "SPEC.md 5.5(d) reaches **8 of 13 assets** on this universe rather than 13."
    )
    add("")

    add("## What is not reachable")
    add("")
    first = residuals.index[0].date()
    add(
        f"The residual panel begins **{first}**, later than the factor panel's 2008-04-14, "
        "because Model A's rolling exposure window costs a further burn-in before a residual "
        "exists. **September 2008 is therefore not reachable and no claim is made about it** "
        "— the third figure in this project to name a date its burn-in cannot supply, after "
        "`reports/vra_multiplier.png` and `reports/residual_pc_variance_share.png`. "
        "March 2020 is reachable and is where both horizons' largest `B_t^S` falls."
    )
    add("")
    return "\n".join(lines) + "\n"


def render_residual_correlation(
    data: Inputs, correlation: specific.ResidualCorrelation, settings: config.Config
) -> str:
    """``reports/specific_residual_corr.md``. SPEC.md 5.5's closing paragraph."""
    residuals = data.residuals
    periods = residuals.shape[0]
    standard_error = 1.0 / np.sqrt(periods)
    tickers = {asset.id: asset.ticker for asset in settings.universe.assets}
    by_ticker = {ticker: asset for asset, ticker in tickers.items() if ticker is not None}

    lines: list[str] = []
    add = lines.append
    add("# The diagonal assumption, and where it fails")
    add("")
    add(
        "Generated by `mafrm.factors.specific_report`. SPEC.md 5.5 closes on a **known false "
        "assumption**, stated explicitly rather than hidden: specific returns are taken to be "
        "cross-sectionally uncorrelated, so `Delta` is diagonal in `F = X F X' + Delta`. This "
        "file is the evidence about whether it should be."
    )
    add("")
    add(
        f"Model A's residuals, **{residuals.shape[1]} assets x {periods:,} dates**, "
        f"{residuals.index[0].date()}..{residuals.index[-1].date()}, "
        f"**{correlation.pair_count} distinct pairs**. Correlation taken **about zero**, "
        "matching SPEC.md 5.1.1's convention for every other moment in the project — the "
        "quantity the diagonal assumption is about is the second moment of the specific "
        "returns, not their covariance about a drifting sample mean. The largest single "
        f"difference from the demeaned Pearson correlation is "
        f"{correlation.mean_convention_difference:.2e}, so nothing here turns on that choice."
    )
    add("")
    add(
        "**This is the one part of SPEC.md 5.5 that works properly at this `N`.** Thirteen "
        "series give 78 pairs, which is a real table; the legs that need a large "
        "cross-section do not fare as well (`reports/specific_risk.md`)."
    )
    add("")

    add("## SPEC.md 5.5's own two predictions")
    add("")
    add(
        'The specification names two linked pairs by vendor symbol: *"SPY/IWM residuals are '
        'not independent, nor are LQD/HYG"*. They are resolved through '
        "`config/universe.yaml`'s tickers rather than written as ids, so a rename cannot drop "
        "a prediction the specification made."
    )
    add("")
    add("| SPEC.md pair | universe ids | `rho` | rank of 78 | verdict |")
    add("|---|---|---|---|---|")
    ranked = correlation.pairs()
    for left, right in _SPEC_LINKED_TICKERS:
        first, second = by_ticker.get(left), by_ticker.get(right)
        if first is None or second is None or first not in residuals or second not in residuals:
            add(f"| {left}/{right} | — | — | — | not in the residual panel |")
            continue
        i = list(correlation.columns).index(first)
        j = list(correlation.columns).index(second)
        rho = float(correlation.matrix[i, j])
        rank = next(
            position
            for position, pair in enumerate(ranked, start=1)
            if {pair[0], pair[1]} == {first, second}
        )
        identity = sorted({first, second} & set(data.excluded))
        if identity:
            verdict = f"**NOT ASSESSABLE** — `{identity[0]}` is an identity asset"
        elif abs(rho) > 10.0 * standard_error:
            verdict = "**not independent** — confirmed"
        else:
            verdict = "too small to call at this magnitude"
        add(f"| {left}/{right} | `{first}` / `{second}` | **{rho:+.4f}** | {rank} | {verdict} |")
    add("")
    add(
        "**SPY/IWM confirms the specification's prediction and reverses its implied sign.** The "
        "residuals are strongly related, but *negatively*: once the equity factor has absorbed "
        "what the two have in common, what is left is large-cap against small-cap, and the two "
        "sides of that trade move opposite. A linked-asset override that assumed a positive "
        'residual correlation — the intuitive direction, and the one the phrase "not '
        'independent" invites — would push the covariance the **wrong way**. That is the '
        "sharper half of this result and it is the half a diagonal `Delta` gets closer to "
        "right than a naively-signed override would."
    )
    add("")
    add(
        "**LQD/HYG cannot be assessed on this panel, and reporting a number for it without "
        f"saying so would be misleading.** `{data.excluded[-1] if data.excluded else 'hy_credit'}` "
        "is one of the two identity assets: SPEC.md 4.1.1 builds the `credit` factor *from* it, "
        "so its residual is zero up to numerical noise by construction and its correlation with "
        "anything is a statistic about rounding rather than about markets. The specification's "
        "prediction is almost certainly right about LQD and HYG as instruments; this factor set "
        "simply cannot test it, because it spent HYG building a factor. `experiments.md` "
        "records the same asset defeating a different measurement twice before this one."
    )
    add("")

    add("## Every pair, ranked")
    add("")
    add(
        f"Under independence the sampling standard error of a correlation is roughly "
        f"`1/sqrt(T)` = **{standard_error:.4f}** here. That figure is a **lower bound on the "
        "true standard error**, not an estimate of it: it assumes independent observations, "
        "and these residuals carry the serial correlation SPEC.md 5.5(a)'s Newey-West term "
        "exists to handle, which lowers the effective `T`. So significance is the wrong "
        "question at this sample size — almost every pair clears it — and **magnitude** is "
        "the right one. The table is ranked by `|rho|` and no threshold is applied, because "
        "SPEC.md publishes none and inventing one would be CLAUDE.md invariant 9."
    )
    add("")
    add(
        "Rows marked **[id]** involve one of the two identity assets "
        f"(`{'`, `'.join(data.excluded)}`), whose residual is zero up to numerical noise "
        "because the factor set is built from them. Their correlations describe rounding, not "
        "markets, and are listed rather than dropped so the table stays a complete census of "
        "the 78 pairs."
    )
    add("")
    add("| rank | asset | asset | `rho` | |")
    add("|---|---|---|---|---|")
    identity_assets = set(data.excluded)
    for rank, (left, right, rho) in enumerate(correlation.pairs(), start=1):
        marker = "**[id]**" if {left, right} & identity_assets else ""
        add(f"| {rank} | `{left}` | `{right}` | {rho:+.4f} | {marker} |")
    add("")

    add("## What this means for the model")
    add("")
    add(
        f"The largest off-diagonal magnitude is **{correlation.maximum_absolute:.4f}**. The "
        "diagonal assumption is **not close to true** on this panel, and the violations are "
        "structured rather than scattered: the government curve points dominate the top of "
        "the table, because a two-factor level/slope model cannot span a curve and what it "
        "leaves over is the third curve mode — the same object `experiments.md` row 83 "
        "identified in the leading residual PC at `R^2 = 0.6247`."
    )
    add("")
    add(
        "**This is not repaired here and SPEC.md 5.5 does not ask for it to be.** MSCI "
        "handles it with linked-asset overrides that are not publicly documented, so building "
        "one would mean inventing the thing the specification declines to publish. What is "
        "owed instead is that any statistic computed from a diagonal `Delta` — portfolio risk "
        "forecasts above all — carries the knowledge that specific risk is understated "
        "wherever a portfolio is long two positively-correlated residuals, and overstated "
        "where it straddles a negative pair. The W6 optimizer is where that bites."
    )
    add("")
    return "\n".join(lines) + "\n"


def main() -> int:
    """Write both reports. Reads the local cache only; no network."""
    settings = config.load()
    data = inputs(settings)
    print(
        f"specific risk (SPEC.md 5.5) on {data.residuals.shape[1]} assets x "
        f"{data.residuals.shape[0]:,} dates, {data.residuals.index[0].date()}.."
        f"{data.residuals.index[-1].date()}"
    )
    runs: dict[str, SpecificRun] = {}
    absorption: dict[str, Absorption] = {}
    contamination: dict[str, pd.DataFrame] = {}
    for horizon in HORIZONS:
        current = run(horizon, data, settings)
        runs[horizon] = current
        for line in current.build.render():
            print(f"  {line}")
        absorption[horizon] = newey_west_absorption(data, current)
        contamination[horizon] = bucket_contamination(data, current)

    reference = RiskConfig.load(horizon=HORIZONS[0], config=settings)
    correlation = specific.residual_correlation(data.residuals, config=reference)

    _RISK_PATH.write_text(
        render_specific_risk(data, runs, absorption, contamination), encoding="utf-8"
    )
    _CORRELATION_PATH.write_text(
        render_residual_correlation(data, correlation, settings), encoding="utf-8"
    )
    print(f"  wrote {_RISK_PATH.relative_to(_REPORTS.parent)}")
    print(f"  wrote {_CORRELATION_PATH.relative_to(_REPORTS.parent)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
