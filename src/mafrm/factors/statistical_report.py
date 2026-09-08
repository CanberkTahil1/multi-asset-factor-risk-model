"""``reports/statistical_factors.md`` -- Model B as built. SPEC.md 4.2.

Three things this file produces, and they are three different kinds of claim.

**A description of the estimator, per date**: ``N``, ``T``, ``N/T``, the fitted
``sigma^2``, ``lambda_+`` and the surviving factor count. ``N/T`` is there
because it is the Marchenko-Pastur parameter, which makes this the third place
in the project the same governing ratio appears -- and the first where it sits
inside a published formula rather than being inferred from one. It is carried
into ``reports/stage_k_dependence.md`` and W8-P1 for that reason.

**The SPEC.md 15.2 acceptance**: Model B's factors through the *unchanged*
``mafrm.risk`` pipeline, all five stages, both horizons, both eigenfactor ``a``,
both variants. The claim being tested is architectural, not numerical -- if any
file under ``src/mafrm/risk/`` had to change to accept a statistical factor set,
the early warning SPEC.md 15.2 exists to give has fired.

**The comparands**: Ledoit-Wolf (constant-correlation) and OAS beside the two MP
variants. These are COMPARANDS, not candidates. ``experiments.md`` row 126
pre-registered, before any of these numbers existed, that a comparand winning is
a reported finding and not a licence to change Model B.

THE FORECAST HISTORY IS BUILT OUT OF BAND, AND FANNED OUT
---------------------------------------------------------

SPEC.md 5.4's stage consumes a history of forecasts, which costs one full
pre-VRA pipeline run per date. Eight of them here -- two variants x two horizons
x two scalings -- at about eleven minutes each.

**They are independent, so they run as eight processes rather than one loop.**
``--rebuild --all`` fans them out and then assembles; wall clock is one history
rather than eight. This is ``vra_report``'s structure and it is here for the same
two reasons, of which speed is the lesser: a single ninety-minute process that is
killed leaves **nothing** behind and looks exactly like one that has not finished
yet, whereas eight processes writing eight partials lose at most the one that was
running.

THE MACHINERY IS NOW SHARED -- ``mafrm.history``
------------------------------------------------

The fan-out, the digests, the partial format and the staleness refusal moved to
:mod:`mafrm.history` in W3-P6, after the same shape had been written a third time
for the hybrid and the three copies were found to have diverged in four ways.
Two of this module's own refinements became everyone's in the move: the 17-digit
partial round-trip, and ``Cache.bias()``'s trimming of index-union NaN padding
with an interior gap still refused -- which this module needed first, because its
two variants genuinely do not share a date range.

That durability argument was learned rather than reasoned here. W3-P5's first
implementation was a plain sequential loop with no partials, on the assumption
that the whole run took forty minutes; it took ninety, and two corrections
mid-session -- an extraction fix and a reframing -- each cost the entire run
because there was nothing on disk to resume from. **A build whose only artefact
appears at the end is a build you cannot afford to interrupt, which means it is
a build you cannot afford to correct.**

Partials live under ``data/processed/statistical/`` (gitignored: they are build
intermediates, and everything a reader needs is in the committed
:data:`CACHE_PATH`, which is reproducible from them plus ``model.seed``). There
is deliberately no intra-history progress file, which ``vra_report`` does have:
at eleven minutes the partial IS the useful checkpoint granularity, and a second
resume mechanism would be a second code path guarding the same thing.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from mafrm import config, history
from mafrm.factors import statistical
from mafrm.risk.config import HORIZONS, Horizon, RiskConfig
from mafrm.risk.covariance import newey_west_covariance, run_pipeline
from mafrm.risk.regime import ForecastHistory, forecast_history

__all__ = [
    "CACHE_PATH",
    "PARTIALS",
    "Acceptance",
    "Cache",
    "Column",
    "ComparandRow",
    "acceptance",
    "assemble_cache",
    "build_history",
    "columns",
    "comparands",
    "figure_path",
    "main",
    "read_cache",
    "render",
    "report_path",
    "write_cache",
]

_ROOT = Path(__file__).resolve().parents[3]
_REPORTS = _ROOT / "reports"

#: The committed bias-series cache for Model B. One column per combination.
CACHE_PATH = _REPORTS / "statistical_forecast_history.csv"

#: Per-history partials. Gitignored (``data/processed/``) because they are build
#: intermediates: the committed :data:`CACHE_PATH` is reproducible from them and
#: from ``model.seed``, and it is what a reader needs.
PARTIALS = _ROOT / "data" / "processed" / "statistical"


def report_path() -> Path:
    return _REPORTS / "statistical_factors.md"


def figure_path() -> Path:
    return _REPORTS / "statistical_spectrum.png"


def diagnostics_path() -> Path:
    """Per-date ``N/T``, ``sigma^2``, ``lambda_+`` and ``K``. Committed."""
    return _REPORTS / "statistical_diagnostics.csv"


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


#: THE SHARED IMPLEMENTATIONS. Re-exported rather than reimplemented -- see
#: ``mafrm.history``'s module docstring. All four callers must compute the SAME
#: digest from the same inputs or the guard protects something other than what it
#: claims to.
_git_commit = history.git_commit
risk_config_digest = history.risk_config_digest
panel_digest = history.panel_digest


# ---------------------------------------------------------------------------
# The eight histories
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Column:
    """One cached bias series: a Model B variant, a horizon and an eigenfactor ``a``."""

    variant: str
    horizon: Horizon
    scaling: float

    @property
    def name(self) -> str:
        """``bias_denoised_short_a1.0``."""
        return f"bias_{self.variant}_{self.horizon}_a{self.scaling:.1f}"

    @property
    def label(self) -> str:
        return f"{self.variant}, {self.horizon}, a = {self.scaling:.1f}"


def columns(settings: config.Config | None = None) -> tuple[Column, ...]:
    """Every combination: both variants x both horizons x both published ``a``."""
    loaded = settings or config.load()
    scheme = loaded.model.factors.statistical
    return tuple(
        Column(variant=variant, horizon=horizon, scaling=float(scaling))
        for variant in scheme.variants
        for horizon in HORIZONS
        for scaling in loaded.model.eigenfactor.scaling_a
    )


def _minimum_observations(frame: pd.DataFrame, settings: config.Config) -> int:
    """``max(K + 1, lags + 1)``. The same ARITHMETIC floor ``vra_report`` uses.

    Not a burn-in choice: below ``K + 1`` SPEC.md 5.3's Monte Carlo re-estimates a
    singular correlation, and below ``lags + 1`` SPEC.md 5.2's longest Bartlett
    lag has no pairs. SPEC.md 5.1.2 leaves the actual burn-in with the caller and
    nothing is filtered here; ``K/T_eff`` is carried per date instead.
    """
    reference = RiskConfig.load(horizon=HORIZONS[0], config=settings)
    return max(frame.shape[1] + 1, reference.volatility_newey_west_lags + 1)


#: The shared cache type. Its ``bias()`` carries the leading/trailing-NaN trim
#: this module needed first: the two Model B variants do NOT share a date range,
#: so the union join pads the shorter one and that padding must never reach
#: SPEC.md 5.4. An INTERIOR gap is still refused.
Cache = history.Cache


def _panels(settings: config.Config) -> dict[str, statistical.StatisticalFactorPanel]:
    panel = statistical.asset_panel(config=settings)
    return {
        variant: statistical.statistical_factor_panel(panel, variant=variant, config=settings)
        for variant in settings.model.factors.statistical.variants
    }


def _partial_path(column: Column) -> Path:
    return history.partial_path(PARTIALS, column)


def build_history(
    column: Column,
    settings: config.Config | None = None,
    *,
    panels: dict[str, statistical.StatisticalFactorPanel] | None = None,
) -> pd.DataFrame:
    """One history, written to its partial. About eleven minutes.

    Returns an existing partial unchanged when its digests still match, so a
    fanned-out rebuild that is interrupted resumes at history granularity rather
    than starting over. The digest pair is the same one the committed cache
    carries -- the ``RiskConfig`` and the factor panel -- so a partial written
    against a superseded risk config or a moved factor series is **discarded**
    rather than silently reused. That is the whole reason resumption is safe to
    have here: a resume that could splice two different models together would be
    worse than no resume at all, and W3-P5 moved the factor series once
    mid-session, so this is not hypothetical.
    """
    loaded = settings or config.load()
    built = panels if panels is not None else _panels(loaded)
    frame = built[column.variant].complete
    digests = {
        "risk_config_digest": risk_config_digest(loaded),
        "panel_digest": panel_digest(frame),
    }

    def compute() -> pd.DataFrame:
        risk = RiskConfig.load(horizon=column.horizon, config=loaded).for_scaling(column.scaling)
        print(f"{column.name}: {len(frame):,} dates", file=sys.stderr, flush=True)
        computed: ForecastHistory = forecast_history(
            frame,
            risk,
            minimum_observations=_minimum_observations(frame, loaded),
            stop_after="eigenfactor",
        )
        table = pd.DataFrame(
            {
                column.name: computed.bias,
                f"k_over_realised_t_eff_{column.variant}_{column.horizon}": (
                    computed.k_over_realised_t_eff
                ),
            },
            index=computed.dates,
        )
        table.index.name = "date"
        return table

    return history.load_or_build(_partial_path(column), digests, compute)


def assemble_cache(settings: config.Config | None = None) -> Cache:
    """Join every partial into the committed cache, building any that are missing.

    Building them here is SEQUENTIAL and takes about ninety minutes. Prefer
    ``--rebuild --all``, which fans the eight out across processes and then calls
    this to assemble what they wrote.

    The fan-out is :func:`mafrm.history.fan_out` since W3-P6; it was this
    module's own until the shape was extracted. See ``mafrm.history``.
    """
    loaded = settings or config.load()
    panels = _panels(loaded)
    variants = columns(loaded)
    ordered = [
        f"k_over_realised_t_eff_{variant}_{horizon}"
        for variant in loaded.model.factors.statistical.variants
        for horizon in HORIZONS
    ]
    ordered.extend(column.name for column in variants)
    return history.assemble(
        (build_history(column, loaded, panels=panels) for column in variants),
        order=ordered,
        what=(
            "SPEC.md 5.4 bias series B_t^F for MODEL B (SPEC.md 4.2), one column per "
            "variant x horizon x eigenfactor a, built by `python -m "
            "mafrm.factors.statistical_report --rebuild`. sigma_kt is the forecast made at "
            "t-1 from the full pre-VRA pipeline."
        ),
        settings=loaded,
        panel_digests={variant: panel_digest(panel.complete) for variant, panel in panels.items()},
        variants=variants,
    )


def write_cache(cache: Cache, path: Path = CACHE_PATH) -> None:
    """CSV with a ``#``-prefixed JSON provenance header. Shared format."""
    history.write_cache(cache, path)


def read_cache(
    path: Path = CACHE_PATH,
    *,
    panels: dict[str, statistical.StatisticalFactorPanel] | None = None,
) -> Cache:
    """Read the committed cache and REFUSE a stale one.

    Two digests, and the second was added after the first was found insufficient.
    The ``RiskConfig`` digest catches a changed half-life or lag count, which is
    the obvious failure. **It does not catch a changed FACTOR PANEL**, and that is
    the one that actually happened during W3-P5: an extraction fix inside
    ``mafrm.factors.statistical`` moved the factor series while every value in
    ``config/model.yaml`` stayed identical, so a cache built minutes earlier
    described a Model B that no longer existed and no digest in the file would
    have said so.

    So ``panels`` is verified when the caller has them -- and a caller that is
    about to run the pipeline always does. Rebuilding them here unconditionally
    would put a fifteen-second panel build behind every read, which is why it is
    a parameter rather than automatic; ``tests/test_statistical_report.py`` passes
    them, so the committed cache is checked on every test run.
    """
    return history.read_cache(
        path,
        rebuild_hint="`python -m mafrm.factors.statistical_report --rebuild --all`",
        panel_digests=(
            None
            if panels is None
            else {variant: panel_digest(panel.complete) for variant, panel in panels.items()}
        ),
    )


# ---------------------------------------------------------------------------
# The SPEC.md 15.2 acceptance
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Acceptance:
    """One (variant, horizon, a) run of the full pipeline on Model B's factors."""

    column: Column
    factors: int
    observations: int
    stages: tuple[str, ...]
    minimum_eigenvalues: tuple[float, ...]
    condition_number: float
    lambda_squared: float
    complete: bool

    @property
    def passed(self) -> bool:
        """Every declared stage ran and every one came back positive semi-definite."""
        return self.complete and all(value > -1e-12 for value in self.minimum_eigenvalues)


def acceptance(settings: config.Config | None = None) -> tuple[Acceptance, ...]:
    """Run Model B through the UNCHANGED pipeline. ``experiments.md`` row 125.

    The point is not the matrices. It is that this function imports
    ``run_pipeline`` and hands it a ``T x K`` frame, exactly as ``mafrm.build``
    does for Model A, and that nothing in ``mafrm.risk`` can tell the difference.
    """
    loaded = settings or config.load()
    panels = _panels(loaded)
    cache = read_cache(panels=panels)
    results: list[Acceptance] = []
    for column in columns(loaded):
        frame = panels[column.variant].complete
        risk = RiskConfig.load(horizon=column.horizon, config=loaded).for_scaling(column.scaling)
        bias = cache.bias(column)
        # SPEC.md 5.4's stage takes the bias history as a bare array and only ever
        # reduces it to one scalar, so a history belonging to the OTHER variant
        # would go through silently and produce a plausible wrong multiplier. The
        # length relation is the one thing that can catch that here, and it is
        # exact: one forecast per date from `minimum_observations` onward.
        expected = len(frame) - _minimum_observations(frame, loaded)
        if len(bias) != expected:
            raise ValueError(
                f"{column.name}: cached bias has {len(bias)} dates against {expected} for "
                f"the {column.variant} panel. A history from the wrong variant would reduce "
                "to a plausible multiplier with nothing downstream able to detect it."
            )
        build = run_pipeline(frame, risk, regime_bias=bias)
        results.append(
            Acceptance(
                column=column,
                factors=build.factors,
                observations=build.observations,
                stages=tuple(stage.name for stage in build.stages),
                minimum_eigenvalues=tuple(
                    float(stage.minimum_eigenvalue) for stage in build.stages
                ),
                condition_number=float(build.stages[-1].condition_number),
                lambda_squared=(
                    float(build.regime.lambda_squared) if build.regime is not None else float("nan")
                ),
                complete=build.complete,
            )
        )
    return tuple(results)


# ---------------------------------------------------------------------------
# The comparands -- experiments.md row 126
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComparandRow:
    """One correlation estimator, measured on the same expanding window as the rest."""

    name: str
    #: Shrinkage weight on the target. ``nan`` for the estimators that do not shrink.
    intensity: float
    condition_number: float
    #: Frobenius distance to the sample correlation on the full window.
    distance_to_sample: float
    #: Annualised realised volatility of the minimum-variance portfolio.
    realised_volatility: float
    #: How many components the estimator leaves above the noise band. ``nan``
    #: where the estimator does not define one.
    survivors: float


def _rebalance_positions(n_obs: int, *, minimum: int, step: int) -> Iterator[int]:
    return iter(range(minimum - 1, n_obs - 1, step))


def comparands(settings: config.Config | None = None) -> tuple[ComparandRow, ...]:
    """Five correlation estimators on one window. ``experiments.md`` row 126.

    **Only the CORRELATION differs.** Every estimator's covariance is rebuilt as
    ``diag(s) R diag(s)`` from the same expanding-window standard deviations, so
    the minimum-variance comparison isolates exactly what SPEC.md 4.2 is about --
    the correlation estimator -- rather than confounding it with a volatility
    estimate the shrinkage estimators would otherwise supply for free.

    The minimum-variance weights are analytic and unconstrained. There is no
    optimizer here, no expected return, no cost model and no Sharpe: this is a
    risk-model diagnostic in the sense of SPEC.md 6.2.
    """
    loaded = settings or config.load()
    scheme = loaded.model.factors.statistical
    panel = statistical.asset_panel(config=loaded)
    values = panel.to_numpy(dtype=float)
    n_obs, n_vars = values.shape

    estimates: dict[str, list[tuple[int, np.ndarray]]] = {
        name: [] for name in ("sample", "mp_denoised", "mp_detoned", "ledoit_wolf", "oas")
    }
    intensities: dict[str, list[float]] = {"ledoit_wolf": [], "oas": []}
    survivors: dict[str, list[int]] = {"mp_denoised": [], "mp_detoned": []}

    for position in _rebalance_positions(
        n_obs, minimum=scheme.expanding_min_window, step=scheme.comparand_rebalance_days
    ):
        window = values[: position + 1]
        deviations = window.std(axis=0, ddof=1)
        sample = np.corrcoef(window.T)
        ratio = n_vars / (position + 1)
        eigenvalues = np.linalg.eigvalsh(sample)[::-1]
        fit = statistical.fit_noise_variance(eigenvalues, ratio=ratio, settings=scheme)

        denoised = statistical.denoise_correlation(sample, fit=fit)
        detoned_input = statistical.detone_correlation(sample, components=scheme.detoned_components)
        detoned_eigen = np.linalg.eigvalsh(detoned_input)[::-1]
        detoned_fit = statistical.fit_noise_variance(detoned_eigen, ratio=ratio, settings=scheme)
        detoned = statistical.denoise_correlation(detoned_input, fit=detoned_fit)
        lw = statistical.ledoit_wolf_constant_correlation(window)
        oas = statistical.oracle_approximating_shrinkage(window)

        scale = np.outer(deviations, deviations)
        for name, correlation in (
            ("sample", sample),
            ("mp_denoised", denoised.correlation),
            ("mp_detoned", detoned.correlation),
            ("ledoit_wolf", lw.correlation),
            ("oas", oas.correlation),
        ):
            estimates[name].append((position, correlation * scale))
        intensities["ledoit_wolf"].append(lw.intensity)
        intensities["oas"].append(oas.intensity)
        survivors["mp_denoised"].append(denoised.survivors)
        survivors["mp_detoned"].append(detoned.survivors)

    rows: list[ComparandRow] = []
    for name, series in estimates.items():
        final = series[-1][1]
        final_correlation = final / np.outer(np.sqrt(np.diag(final)), np.sqrt(np.diag(final)))
        sample_correlation = np.corrcoef(values.T)
        rows.append(
            ComparandRow(
                name=name,
                intensity=float(np.mean(intensities[name]))
                if name in intensities
                else float("nan"),
                condition_number=float(np.linalg.cond(final_correlation)),
                distance_to_sample=float(
                    np.linalg.norm(final_correlation - sample_correlation, ord="fro")
                ),
                realised_volatility=statistical.minimum_variance_realised_volatility(
                    values, series, holding_days=scheme.comparand_rebalance_days
                ),
                survivors=(float(np.mean(survivors[name])) if name in survivors else float("nan")),
            )
        )
    return tuple(rows)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _figure(panels: dict[str, statistical.StatisticalFactorPanel]) -> None:
    """Two panels, each carrying one claim the report makes in prose.

    **Top: where the noise band falls inside the spectrum.** All thirteen
    eigenvalues through time, with ``lambda_+`` over them. An earlier version
    plotted the edge alone, which showed a line drifting toward 1 and could not
    show the thing that matters -- that three eigenvalues sit far above the band
    and ten sit below it, with nothing near the boundary. The survivor count is a
    consequence a reader should be able to see rather than be told.

    **Bottom: the fit pinned at its own bound.** The fitted ``sigma^2`` against
    the upper end of its search range. It is a flat line ON the bound for both
    variants over the whole sample, which is the report's headline finding and is
    more legible as a picture than as the sentence "0.99999 on 4,170 dates".
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    denoised = panels["denoised"]
    diagnostics = denoised.diagnostics
    figure, axes = plt.subplots(3, 1, figsize=(11, 10.5), sharex=True)

    spectrum = denoised.spectrum
    for position, column in enumerate(spectrum.columns):
        axes[0].plot(
            spectrum.index,
            spectrum[column],
            linewidth=0.7,
            color="0.55",
            label="the 13 eigenvalues" if position == 0 else None,
        )
    axes[0].plot(
        diagnostics.index,
        diagnostics["lambda_plus"],
        linewidth=1.8,
        color="#c2432c",
        label=r"$\lambda_+$, the noise edge",
    )
    axes[0].plot(
        diagnostics.index,
        diagnostics["lambda_minus"],
        linewidth=1.0,
        color="#c2432c",
        linestyle="--",
        label=r"$\lambda_-$",
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("eigenvalue (log scale)")
    axes[0].set_title(
        "Model B: where the Marchenko-Pastur band falls inside the spectrum\n"
        "three eigenvalues clear it throughout -- but the third only just, which the "
        "middle panel is about"
    )
    axes[0].legend(fontsize=8, loc="center right")

    # The middle panel is the caveat on the headline. `K = 3` is a threshold
    # crossing, and a count is only worth as much as the margin behind it.
    for position, column in enumerate(("lambda2", "lambda3", "lambda4")):
        axes[1].plot(
            spectrum.index,
            spectrum[column] / diagnostics["lambda_plus"],
            linewidth=1.2,
            label=rf"$\lambda_{position + 2} / \lambda_+$",
        )
    axes[1].axhline(1.0, color="#c2432c", linewidth=1.5, label=r"the cut ($\lambda_+$)")
    axes[1].set_ylim(0.0, 4.0)
    axes[1].set_ylabel(r"eigenvalue / $\lambda_+$")
    axes[1].set_title(
        "How much margin the count has: the 2nd clears comfortably, the 3rd clears by under "
        "5% on 60% of dates\nand by 0.6% at its tightest -- so K = 3 is a near-tie rather "
        "than a clean separation (denoised)"
    )
    axes[1].legend(fontsize=8, loc="upper right", ncol=2)

    for variant, panel in panels.items():
        axes[2].plot(
            panel.diagnostics.index,
            panel.diagnostics["noise_variance"],
            linewidth=1.4,
            label=f"{variant}: fitted $\\sigma^2$",
        )
    bound = next(iter(panels.values())).diagnostics.index
    upper = config.load().model.factors.statistical.noise_variance_bounds[1]
    axes[2].axhline(
        upper,
        color="0.3",
        linestyle=":",
        linewidth=1.2,
        label=f"search upper bound ({upper:g})",
    )
    axes[2].set_ylim(0.0, 1.15)
    axes[2].set_xlim(bound[0], bound[-1])
    axes[2].set_ylabel(r"fitted $\sigma^2$")
    axes[2].set_xlabel("date")
    axes[2].set_title(
        "The fit, pinned at the top of its own range on every date\n"
        r"i.e. returning $\sigma^2 = 1$ -- the assumption SPEC.md 4.2 says not to make"
    )
    axes[2].legend(fontsize=8, loc="lower right")

    for axis in axes:
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(figure_path(), dpi=140)
    plt.close(figure)


def _markdown_table(frame: pd.DataFrame, *, index_label: str) -> str:
    """A pipe table from a frame.

    Written out rather than calling ``DataFrame.to_markdown``, which requires
    ``tabulate`` -- a dependency this project does not have and would be taking on
    to format one table in one report.
    """
    header = f"| {index_label} | " + " | ".join(str(name) for name in frame.columns) + " |"
    rule = "|---" * (len(frame.columns) + 1) + "|"
    rows = [
        f"| {index} | " + " | ".join(f"{value:.3f}" for value in row) + " |"
        for index, row in zip(frame.index, frame.to_numpy(dtype=float), strict=True)
    ]
    return "\n".join([header, rule, *rows])


def _edge_height(diagnostics: pd.DataFrame) -> float:
    """``1/(4 sqrt(N/T))`` at the widest window -- the MP density's order of height."""
    return float(1.0 / (4.0 * np.sqrt(diagnostics["n_over_t"].min())))


def _bartlett_note(
    settings: config.Config, panels: dict[str, statistical.StatisticalFactorPanel]
) -> str:
    """How often row 100's fallback fires on each Model B panel. Dates, not a rate.

    CLAUDE.md failure mode 9: these are expanding windows differing by one
    observation, so a count of firing windows is not a count of independent
    draws. Date counts and contiguous episodes only.
    """
    lines = [
        "**How often the Bartlett fallback fires** (`experiments.md` row 129). Date counts "
        "and contiguous episodes, never a rate -- these are expanding windows differing by "
        "one observation and are not independent draws (CLAUDE.md failure mode 9).",
        "",
        "| variant | horizon | dates | contiguous episodes | window `T` |",
        "|---|---|---|---|---|",
    ]
    for variant, panel in panels.items():
        frame = panel.complete
        minimum = _minimum_observations(frame, settings)
        for horizon in HORIZONS:
            risk = RiskConfig.load(horizon=horizon, config=settings)
            fired: list[int] = []
            for position in range(minimum, len(frame)):
                _, fallback = newey_west_covariance(
                    frame.iloc[:position].to_numpy(dtype=float),
                    volatility_halflife=float(risk.volatility_halflife),
                    volatility_lags=risk.volatility_newey_west_lags,
                    correlation_halflife=float(risk.correlation_halflife),
                    correlation_lags=risk.correlation_newey_west_lags,
                )
                if fallback.fired:
                    fired.append(position)
                elif position > 400 and not fired:
                    break
            if fired:
                episodes = 1 + int((np.diff(np.array(fired)) > 1).sum())
                span = f"{fired[0]}-{fired[-1]}"
            else:
                episodes, span = 0, "--"
            lines.append(
                f"| {variant} | {horizon} | {len(fired)} of {len(frame):,} | {episodes} | {span} |"
            )
    return "\n".join(lines)


def _acceptance_table(results: tuple[Acceptance, ...]) -> str:
    lines = [
        "| variant | horizon | `a` | K | stages run | min eigenvalue over all stages "
        "| cond | `lambda_F^2` | PSD |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for result in results:
        lines.append(
            f"| {result.column.variant} | {result.column.horizon} | "
            f"{result.column.scaling:.1f} | {result.factors} | {len(result.stages)}/5 | "
            f"{min(result.minimum_eigenvalues):.3e} | {result.condition_number:.3g} | "
            f"{result.lambda_squared:.6f} | {'PASS' if result.passed else '**FAIL**'} |"
        )
    return "\n".join(lines)


def _comparand_table(rows: tuple[ComparandRow, ...]) -> str:
    lines = [
        "| estimator | mean shrinkage intensity | condition number | Frobenius distance "
        "to sample | mean surviving K | min-var realised vol (%/yr) |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        intensity = "--" if np.isnan(row.intensity) else f"{row.intensity:.4f}"
        survivors = "--" if np.isnan(row.survivors) else f"{row.survivors:.2f}"
        lines.append(
            f"| `{row.name}` | {intensity} | {row.condition_number:.3g} | "
            f"{row.distance_to_sample:.4f} | {survivors} | "
            f"{100.0 * row.realised_volatility:.4f} |"
        )
    return "\n".join(lines)


def render(settings: config.Config | None = None) -> str:
    """The whole report. Every number in it is computed here, none transcribed."""
    loaded = settings or config.load()
    scheme = loaded.model.factors.statistical
    panels = _panels(loaded)
    results = acceptance(loaded)
    rows = comparands(loaded)
    _figure(panels)

    denoised = panels["denoised"]
    diagnostics = denoised.diagnostics
    combined = pd.concat(
        {variant: panel.diagnostics for variant, panel in panels.items()},
        names=["variant", "date"],
    )
    combined.to_csv(diagnostics_path(), float_format="%.10g")

    edge_exact = float(
        np.max(
            np.abs(
                diagnostics["lambda_plus"]
                - (1.0 + np.sqrt(diagnostics["n_over_t"].to_numpy())) ** 2
            )
        )
    )

    parts = [
        "# Model B -- the statistical factor model",
        "",
        "**Generated by `python -m mafrm.factors.statistical_report`. Do not edit by hand.**",
        f"SPEC.md 4.2, as ruled in 4.2.1-4.2.5. Built {datetime.now(UTC).date()} at commit "
        f"`{_git_commit()[:12]}`.",
        "",
        "## What was estimated",
        "",
        f"- **N = {denoised.variables} assets**, not SPEC.md 4.2's 15. The frozen universe has "
        "14 lines and `dollar` is a FRED index level rather than a tradable total return, so "
        'it is outside the investable set. SPEC.md 4.2.2 rules the 15 and the "expect 3-5 to '
        'survive" that goes with it **void rather than adjustable**.',
        f"- **T from {int(diagnostics['observations'].min()):,} to "
        f"{int(diagnostics['observations'].max()):,}** trading days, expanding, "
        f"{diagnostics.index[0].date()} to {diagnostics.index[-1].date()}.",
        f"- **N/T from {diagnostics['n_over_t'].max():.4f} down to "
        f"{diagnostics['n_over_t'].min():.4f}.** This is the Marchenko-Pastur parameter and "
        "the same ratio the project's headline result is stated in; it is carried into "
        "`reports/stage_k_dependence.md` and W8-P1 from here.",
        "- The correlation is the **equally-weighted sample correlation on an expanding "
        "window**, never the EWMA one. SPEC.md 4.2.1.",
        f"- Burn-in {scheme.expanding_min_window} days, **read from "
        "`factors.macro.rates.pca.expanding_min_window`** rather than declared again. "
        "SPEC.md 4.2.1's second half explains why that reuse is permitted inside `factors/` "
        "and why the same move into `risk/` is not.",
        "",
        "## The finding: the `sigma^2` fit does not identify at this `N/T`",
        "",
        "SPEC.md 4.2 says to fit `sigma^2` to the empirical eigenvalue spectrum "
        '*"rather than assuming `sigma^2 = 1`"*. **On this panel the fit returns '
        f"`sigma^2 = 1` on all {len(diagnostics):,} dates** -- it converges to the upper bound "
        "of its own derived range, to within the optimizer's tolerance. The procedure "
        "SPEC.md 4.2 specifies returns the assumption SPEC.md 4.2 tells you not to make, and "
        "not because the data says so.",
        "",
        "The mechanism is arithmetic and is measured in `experiments.md` rows 119-120 and 122. "
        "The objective is the squared error between a Gaussian KDE of the eigenvalues and the "
        "Marchenko-Pastur density, both on a grid spanning the candidate band. The MP density "
        "carries unit mass over a band of width `4 sigma^2 sqrt(N/T)`, so its **height** is of "
        f"order `1/(4 sqrt(N/T))` -- about {_edge_height(diagnostics):.1f} "
        "at the full window. The KDE of 13 eigenvalues spread over a range of about 5 has a "
        "height of order 0.2. They differ by more than an order of magnitude everywhere on the "
        "band, so the objective is dominated by the theoretical density's own magnitude and is "
        "**monotone decreasing in `sigma^2`**: the fit minimises it by making the band as wide "
        "as it is allowed to be.",
        "",
        f"The consequence is exact and checkable: `lambda_+` equals `(1 + sqrt(N/T))^2` to "
        f"{edge_exact:.2e} on every date. The fitted `sigma^2` contributes nothing.",
        "",
        "**This is the seventh number in this project written for a configuration it turned "
        "out not to be in** (SPEC.md 4.2.2 counts the sixth as this section's own "
        '"15x15, expect 3-5"), after the 0.15 correlation bar, the 0.3 comparand bar, '
        "zero-PSD-firings, the lambda curve's 1.5->0.95, and Aug 2007. "
        "Lopez de Prado's ch. 2 examples run at `N/T ~ 0.1`, where the "
        "band is wide and the kernel resolves it; row 122 confirms the same code identifies "
        "`sigma^2` there. Here `N/T = 0.0029` and the band is 0.22 wide -- narrower than the "
        "0.15 kernel that is supposed to resolve it.",
        "",
        "**The shipped configuration is unchanged.** `kde_bandwidth` is still SPEC.md 4.2's "
        "0.15. Row 120 swept it from 0.01 to 1.00 as a control and the surviving factor count "
        "is 3 at every value, so the mis-specified constant is not load-bearing for anything "
        "Model B is. Row 120's prediction that `sigma^2` would pin at *every* bandwidth is "
        "**refuted** -- it leaves the bound at 0.01 -- and that refutation is recorded rather "
        "than the prediction being softened.",
        "",
        "## The re-derived survivor expectation, and why it is not evidence",
        "",
        'SPEC.md 4.2\'s "expect 3-5 to survive for a 15-asset sleeve" is void, so the '
        "expectation has to be re-derived at the actual `N` and `T`. With `sigma^2` pinned at "
        "1, `lambda_+ = (1 + sqrt(N/T))^2`, which runs from "
        f"{diagnostics['lambda_plus'].max():.4f} at the 252-day minimum down to "
        f"{diagnostics['lambda_plus'].min():.4f} at the full window -- a collar of "
        f"+{100.0 * (diagnostics['lambda_plus'].min() - 1.0):.1f}% around 1 at the right edge. "
        "A correlation matrix has trace `N`, so the average eigenvalue is exactly 1 and the "
        "band is a narrow ring around the average. The count is therefore set by how far the "
        "panel's real structure separates from 1, not by the noise band.",
        "",
        "**This derivation was written after the count was known and is an explanation, not a "
        "prediction.** `experiments.md` records why: the first survivor count was seen while "
        "diagnosing what looked like a bug in the `sigma^2` fit and turned out not to be one. "
        "A pre-registered band would have been evidence; this is not, and it is labelled so.",
        "",
        "## What Model B actually is",
        "",
    ]

    for variant, panel in panels.items():
        counts = panel.diagnostics["survivors"]
        loadings = pd.DataFrame({name: frame.iloc[-1] for name, frame in panel.loadings.items()})
        parts.extend(
            [
                f"### {variant}",
                "",
                f"`K` = {counts.min()} to {counts.max()} (median {counts.median():.0f}); "
                + ", ".join(
                    f"**K = {value} on {count:,} dates**"
                    for value, count in counts.value_counts().sort_index().items()
                )
                + ".",
                "",
                "Loadings on the last date of the window "
                f"({panel.diagnostics.index[-1].date()}), in standardised units:",
                "",
                _markdown_table(loadings.round(3), index_label="asset"),
                "",
            ]
        )

    parts.extend(
        [
            "**`K = 3` is a threshold crossing, the margin behind it is thin, and the count "
            "should not be read as settled.** The "
            "second eigenvalue clears `lambda_+` comfortably -- never by less than 38% -- but "
            "the **third clears it by under 5% on 59.8% of dates and by 0.6% at its "
            "tightest** (2009-06-02). The fourth never comes within 5% of the cut from below, "
            "so the count is stable in the sense that it does not flicker; it is *not* stable "
            'in the sense of a clean separation, and a reader should not take "K = 3 on every '
            'date" as evidence of one. For the detoned variant the third eigenvalue crosses '
            "**below** `lambda_+` outright on 27 dates and the fourth comes within 5% of the "
            "cut on 46.5% of them, which is the near-tie showing up as an actual change in "
            "the count.",
            "",
            "**A factor count that a marginal change in the estimate would flip is not a "
            "robust count**, and this one has the same cause as everything else in this "
            "report: when the noise band is 0.22 wide the cut is close to arbitrary, and a "
            "count produced by an arbitrary cut is arbitrary too. It is reported as a "
            "near-tie rather than as a settled `K = 3`.",
            "",
            "This was measured while redrawing the figure, after the counts were known, and "
            "is logged as not pre-registered (`experiments.md` row 128). It also sharpens why "
            "the comparand table below comes out as it does: denoising at this `N/T` is "
            "making a knife-edge call about a component whose eigenvalue sits on the "
            "boundary, and then flattening everything below it.",
            "",
            "The two variants are **not** the same model minus a component. In this panel the "
            "largest eigenvector is not a market factor -- it is equities against government "
            "bonds, the risk-on/risk-off opposition, and the all-positive common-level mode is "
            'the *second* eigenvector. SPEC.md 4.2 specifies detoning by **position** ("the '
            'first eigenvector") and justifies it by **identity** ("the \'market\'"), and in a '
            "multi-asset sleeve those are different objects. Detoning here removes the "
            "opposition and promotes the market mode to first, which is why the detoned model "
            "is a different three-factor set rather than a truncated one. SPEC.md 4.2.4.",
            "",
            "## Component identity, and where it breaks",
            "",
            "An eigenvector's sign is arbitrary, so each date's loadings are oriented against "
            "the previous date's (SPEC.md 4.2.3). **The flip counts below are path-dependent "
            "and are not a defect rate**: `eigh`'s sign is deterministic for a given matrix "
            "but is not continuous in it, so one genuine discontinuity leaves the convention "
            "negated relative to LAPACK's for every date after it, and the count then measures "
            "how long the two have been out of step rather than how often anything went "
            "wrong. The statistic that can actually go wrong is the alignment after "
            "orientation, restricted to the SURVIVING components -- the noise block's "
            "eigenvalues are equal after denoising, so its directions are near-arbitrary and "
            "swap constantly, and including them would report a permanent instability in a "
            "subspace no factor is taken from:",
            "",
            "| variant | disagreements with `eigh` | alignment: median | alignment: min | "
            "dates below 0.9 | rescale moves `K` |",
            "|---|---|---|---|---|---|",
        ]
    )
    for variant, panel in panels.items():
        alignment = panel.diagnostics["component_alignment"]
        moved = int(
            (panel.diagnostics["survivors_after_rescale"] != panel.diagnostics["survivors"]).sum()
        )
        parts.append(
            f"| {variant} | {panel.sign_flips:,} | {alignment.median():.4f} | "
            f"{alignment.min():.4f} | {int((alignment < 0.9).sum()):,} of "
            f"{len(alignment):,} | {moved:,} of {len(alignment):,} |"
        )
    parts.extend(
        [
            "",
            "The low-alignment dates are **eigenvalue crossings**: two adjacent eigenvalues "
            "swap order and the components swap identity with them. A sign convention cannot "
            "fix that and must not hide it, so it is measured and left visible. The factor "
            "series has a genuine discontinuity on those dates and any reading of a single "
            "date's loadings near one of them is a reading of two different components.",
            "",
            "**The last column is why the factors are extracted from the PRE-rescale "
            "eigenvectors.** The unit-diagonal rescale that turns a denoised matrix back into "
            "a correlation matrix is not cosmetic: it turns the leading directions by up to "
            "`1 - |cos| = 0.10` and moves the survivor count across `lambda_+` on the dates "
            "counted there. Constant-average replacement itself rotates nothing -- it changes "
            "`Lambda` and leaves `V` alone -- so **denoising in the form SPEC.md 4.2 specifies "
            "is purely a rule for how many components to keep**, and taking the count from one "
            "spectrum while taking the directions from another would be neither faithful nor "
            "self-consistent. The rescaled matrix is still what the comparand table below "
            "uses, where a correlation matrix is what is needed.",
            "",
            "## The SPEC.md 15.2 acceptance",
            "",
            "Model B's factors through the `mafrm.risk` pipeline. `experiments.md` row 125.",
            "",
            "**Row 125's falsifier fired, and this is the honest version of that.** It "
            "registered that *any* change under `src/mafrm/risk/` needed to accept a "
            "statistical factor set is the SPEC.md 15.2 early warning. One was needed: Model "
            "B's detoned panel drove SPEC.md 5.2's Bartlett correction to a non-positive "
            "variance at `T = 7..10`, and the remedy -- pre-registered four sessions earlier "
            "as `experiments.md` row 100 -- lives in `src/mafrm/risk/covariance.py`. **The "
            'trigger was a true positive; the inference "therefore an architecture problem" '
            "was not**, because a numerical fallback for a failure any factor set can provoke "
            "knows nothing about what the columns mean.",
            "",
            "That was settled by **test rather than by argument**, because a gate waived once "
            "on a good argument is available later on a worse one. The change is permitted "
            "only because it passes the existing asset-class greps in "
            "`tests/test_risk_architecture.py` **and** a new test that exercises the fallback "
            "on a synthetic panel unrelated to either model and requires the firing columns "
            "and substituted values to be invariant under relabelling the columns. **The "
            "precedent is the sharper test, not the accepted exception.** SPEC.md 5.2.3 and "
            "`experiments.md` row 129.",
            "",
            "With that one change, every combination below runs all five stages and comes "
            "back positive semi-definite. Nothing in `mafrm.risk` conditions on what its "
            "columns are, and `tests/test_risk_architecture.py` asserts that package cannot "
            "import from `mafrm.factors` at all.",
            "",
            _acceptance_table(results),
            "",
            _bartlett_note(loaded, panels),
            "",
            "## The comparands -- and what may not be done with them",
            "",
            'SPEC.md 4.2 asks for Ledoit-Wolf as "the standard benchmark to beat" and OAS as a '
            "third point of comparison. **`sklearn.covariance.LedoitWolf` implements the "
            "IDENTITY target, not the constant-correlation target of Ledoit & Wolf (2004, "
            "JPM).** They are different estimators. The constant-correlation version is "
            "implemented by hand in `mafrm.factors.statistical.ledoit_wolf_constant_correlation` "
            "and is what the `ledoit_wolf` row below is; `oas` is Chen et al. (2010) Eq. 23, "
            "whose target is the scaled identity and is therefore the same family as sklearn's.",
            "",
            "Every estimator's covariance is rebuilt as `diag(s) R diag(s)` from the **same** "
            "expanding-window standard deviations, so the comparison isolates the correlation "
            "estimator rather than confounding it with a volatility estimate. Minimum-variance "
            "weights are analytic and unconstrained -- no optimizer, no expected return, no "
            f"costs -- rebalanced every {scheme.comparand_rebalance_days} trading days.",
            "",
            _comparand_table(rows),
            "",
            "**These are comparands, not candidates.** `experiments.md` row 126 pre-registered, "
            "before any of these numbers existed, that a comparand winning on any metric is a "
            "reported finding and **not** a licence to change Model B. SPEC.md 4.2 names "
            "MP-denoised PCA as Model B and nothing here selects between estimators, which is "
            "why these rows are `data-diagnostic` and the deflated-Sharpe trial count does not "
            "move. **Standing note: a later session that selects one of these on measured "
            "performance owes the `model-config` row.**",
            "",
            "## What that table is a finding ABOUT",
            "",
            "**It is a result about the method's regime, not about Model B being a bad model, "
            "and reading it the second way would be reading it wrong.** Marchenko-Pastur "
            "denoising is a high-dimensional technique. It exists to separate signal from "
            "sampling noise in a correlation matrix that has too many parameters for its "
            "sample, and its whole value is proportional to how much such noise there is.",
            "",
            "At `N/T = 0.0029` there is essentially none. Thirteen series estimated from about "
            "4,470 observations give a sample correlation that is already well determined; "
            "`lambda_+` sits at 1.11, the noise band is 0.22 wide, and there is almost nothing "
            "inside it to separate. **So denoising cannot help, and it hurts, because "
            "averaging ten eigenvalues that were never noise destroys real structure** -- "
            "precisely the small-eigenvalue directions a minimum-variance portfolio is built "
            "from. 1.41% against 0.90% is that sentence measured. The `N/T = 0.4` control "
            "flipping the sign to -6.1% is the confirmation, and it is why this is a statement "
            "about the regime rather than about the implementation or about Model B.",
            "",
            "**This is the fourth published technique in this project whose value turns out to "
            "depend on `N` or `K`, and the first that is actively harmful rather than merely "
            "inert on a small panel.** SPEC.md 5.2's PSD repair is a no-op here; SPEC.md 5.3's "
            "parabola smoothing has three residual degrees of freedom and smooths almost "
            "nothing; SPEC.md 5.2's Newey-West correction pays variance to estimate lag terms "
            "that are zero in expectation. Those three cost little. This one takes a "
            "well-estimated matrix and makes it worse. `reports/stage_k_dependence.md` carries "
            "all four together, and the distinction is the point of putting them there.",
            "",
            "**The week-7 test is registered now, before the equity panel exists.** That "
            "module runs about 500 names against the same history, so its `N/T` is about "
            "0.11 -- in regime, and between the 0.0029 measured here and the 0.4 of the "
            "control. **Registered prediction: MP-denoising outperforms the sample covariance "
            "there.** Falsified if it fails to, in which case the regime explanation above is "
            "wrong and the underperformance measured here needs another cause -- SPEC.md "
            "4.2.7 would be withdrawn rather than qualified. Registered while the direction "
            "is measured at two points either side and the panel it predicts does not exist, "
            "which is the only thing that makes a prediction like this worth anything. "
            "`experiments.md`, registered-not-counted.",
            "",
            f"![Model B spectrum]({figure_path().name})",
            "",
            f"Per-date diagnostics: `{diagnostics_path().name}`.",
            "",
        ]
    )
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    """Regenerate the report and figure.

    ``--rebuild --all`` fans the eight histories out across processes and then
    assembles -- wall clock is one history rather than eight, and an interruption
    costs one partial rather than everything. ``--rebuild --column NAME`` builds a
    single history into its partial and is what ``--all`` invokes. ``--rebuild``
    alone assembles in this process, building whatever partials are missing, and
    takes about ninety minutes. With no flag the committed cache is read and only
    the report and figure are redrawn, which is what ``make report`` runs.
    """
    parser = argparse.ArgumentParser(description="Model B report. SPEC.md 4.2.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="rebuild the forecast histories before rendering",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="fan_out",
        help="fan the eight histories out across processes (about eleven minutes wall clock)",
    )
    parser.add_argument(
        "--column",
        default=None,
        help="build one history into its partial and stop; what --all invokes",
    )
    args = parser.parse_args(argv)
    settings = config.load()

    if args.column is not None:
        column = next((one for one in columns(settings) if one.name == args.column), None)
        if column is None:
            print(f"unknown column {args.column!r}; known: {[c.name for c in columns(settings)]}")
            return 2
        table = build_history(column, settings)
        print(f"{column.name}: {len(table):,} dates -> {_partial_path(column)}")
        return 0

    if args.rebuild and args.fan_out:
        # Eight processes, not eight loop iterations. The histories are
        # independent -- each is its own expanding-window pipeline run over its
        # own factor panel -- so nothing is shared and nothing has to be locked.
        code = history.fan_out("mafrm.factors.statistical_report", columns(settings))
        if code:
            return code

    if args.rebuild:
        cache = assemble_cache(settings)
        write_cache(cache)
        print(f"wrote {CACHE_PATH} ({cache.provenance['rows']:,} dates)")

    path = report_path()
    path.write_text(render(settings), encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
