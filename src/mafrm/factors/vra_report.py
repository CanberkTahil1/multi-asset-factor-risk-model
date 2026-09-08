"""``reports/vra_multiplier.png``, ``reports/volatility_regime.md`` and the cache. SPEC.md 5.4.

WHY THIS LIVES IN ``factors/`` AND NOT IN ``risk/``
---------------------------------------------------

Same reason as :mod:`mafrm.factors.psd_repair_report` and
:mod:`mafrm.factors.eigenfactor_report`. The stage is in
:mod:`mafrm.risk.regime`, which knows nothing about what its columns are; *this*
module reads the macro factor panel, so it knows exactly what they are and
therefore cannot live under ``risk/`` without breaking CLAUDE.md invariant 10.

THE CACHE, AND WHY THERE IS ONE
--------------------------------

SPEC.md 5.4 standardises each factor return by **the forecast made at t-1**, so
the stage needs a history of forecasts, and building one means running the whole
pre-VRA pipeline once per date -- including SPEC.md 5.3's 2,000-trial Monte
Carlo. Six histories are needed (two horizons x two published scalings, plus two
with the eigenfactor stage off for the overlap control) and they take about
ninety minutes together. That is fine once and wrong inside ``make test`` or
``make report``'s inner loop, so the bias series are built out of band by
``--rebuild``, committed under ``reports/``, and read from there.

BUILT ONE VARIANT AT A TIME, AND THAT IS NOT A CONVENIENCE
-----------------------------------------------------------

``--rebuild`` builds a **single** variant and writes it to a partial file under
``data/processed/vra/`` (gitignored); ``--assemble`` joins the partials into the
committed CSV. Both are driven by ``--all``, which fans the expensive variants
out across processes.

The reason is that a ninety-minute single process is a **fragile** ninety
minutes: anything that kills it -- a terminated shell, a session ending, a
machine sleeping -- costs the whole run, and it costs it silently, because a
process that is no longer there looks exactly like a process that has not
finished. This happened once while W3-P4 was being written. Per-variant partials
make progress durable and the fan-out makes the wall-clock roughly the cost of
the slowest variant instead of the sum of all six. A partial whose digests still
match is **reused** rather than rebuilt, so an interrupted build resumes.

THE MACHINERY IS NOW SHARED -- ``mafrm.history``
------------------------------------------------

Everything in the two sections above -- the digests, the partial format, the
staleness refusal, the fan-out -- moved to :mod:`mafrm.history` in W3-P6, and
this module calls it rather than owning it. What stays here is what is genuinely
about Model A's VRA: which variants exist, what ``stop_after`` each uses, and
this module's mid-column checkpointing, which no other caller needs.

The extraction happened because the shape had been written three times and the
three copies had **diverged in four ways**, one of which was a digest computing a
different number from the same inputs. `experiments.md` records them. This module
was the source the other two were modelled on and it lost nothing in the move; the
one behaviour that changed is that ``Cache.bias()`` now trims index-union NaN
padding and refuses an interior gap, which is a no-op here because every variant
shares one panel.

**A committed artifact that later work reads is a staleness hazard**, which is
the failure mode already flagged for the hand-maintained
``reports/stage_k_dependence.md``. So the cache carries a provenance header --
UTC timestamp, git commit, the SHA-256 of ``config/model.yaml``, a digest of the
factor panel, and a digest of the RiskConfig fields that determine it -- and
``tests/test_regime.py`` fails when the recorded risk-config digest stops
matching the one ``config/model.yaml`` produces today.

The digest deliberately covers the **determining fields** rather than the whole
config file. A full-file hash would fire on a cost-parameter edit that cannot
change a single number in the cache, and a guard that cries wolf is a guard that
gets deleted. The file hash is recorded beside it as provenance rather than as
the test's trigger.

WHAT THE CHART SHOWS, AND WHAT IT CANNOT
-----------------------------------------

``lambda_F`` over time at both horizons. SPEC.md 5.4 names Aug 2007, Sep 2008 and
Mar 2020 as the episodes a working VRA must respond to. **Aug 2007 is not
reachable on this panel and no claim is made about it**: the complete
orthogonalized macro factor panel begins 2008-04-14, bound by ``credit``
(first valid 2008-04-14) and ``rates_slope`` (2008-04-02) through the
252-observation expanding-window orthogonalization burn-in off
``sample.start`` = 2007-04-01. That start is a documented consequence of W2-P2's
row 69. The exclusion is stated in the report rather than left to a reader who
would otherwise see a chart quietly beginning after the window it claims.

CLAUDE.md invariant 5: the panel stops strictly before ``sample.holdout_start``.
CLAUDE.md failure mode 9: the plotted multiplier is a ROLLING statistic whose
consecutive values share all but one observation, so no count of its points is
quoted as an ``n``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # No display in CI; must precede the pyplot import.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from mafrm import config, history
from mafrm.factors import macro
from mafrm.risk.config import HORIZONS, Horizon, RiskConfig
from mafrm.risk.regime import (
    ForecastHistory,
    ForecastRow,
    forecast_history,
    regime_multiplier,
)

__all__ = [
    "CACHE_PATH",
    "PARTIALS",
    "Cache",
    "Column",
    "assemble_cache",
    "build_cache",
    "build_variant",
    "columns",
    "figure_path",
    "main",
    "panel_digest",
    "read_cache",
    "render",
    "report_path",
    "risk_config_digest",
]

_REPORTS = Path(__file__).resolve().parents[3] / "reports"

#: The committed bias-series cache. Read by :func:`render` and by ``make model``.
CACHE_PATH = _REPORTS / "vra_forecast_history.csv"

#: Per-variant partials. Gitignored (``data/processed/``) because they are build
#: intermediates: everything a reader needs is in :data:`CACHE_PATH`, which is
#: reproducible from them and from ``model.seed``.
PARTIALS = Path(__file__).resolve().parents[3] / "data" / "processed" / "vra"

#: How many dates a variant computes before appending them to its progress file.
#: A durability/throughput trade and **not a model parameter**: it changes no
#: number, only how much work an interruption discards. One fsync per date would
#: dominate a run whose per-date cost is a fraction of a second.
_CHECKPOINT_EVERY: int = 100

#: SPEC.md 5.4's own list, minus the one this panel cannot reach. See the module
#: docstring for why Aug 2007 is excluded and what excludes it.
_EPISODES: tuple[tuple[str, str, str], ...] = (
    ("Sep 2008", "2008-09-01", "2008-11-30"),
    ("Mar 2020", "2020-02-15", "2020-05-15"),
)

#: The episode SPEC.md 5.4 names that this panel begins after. Carried as data so
#: the report states the exclusion rather than silently omitting it.
_UNREACHABLE_EPISODE = ("Aug 2007", "2007-08-01", "2007-09-30")


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


#: THE SHARED IMPLEMENTATIONS. Re-exported rather than reimplemented -- see
#: ``mafrm.history``'s module docstring for why this module no longer owns them.
#: Every caller must compute the SAME digest from the same inputs or the guard
#: protects something other than what it claims to.
risk_config_digest = history.risk_config_digest
panel_digest = history.panel_digest
_git_commit = history.git_commit
_read_partial = history.read_partial


def _model_yaml_sha256() -> str:
    path = Path(__file__).resolve().parents[3] / "config" / "model.yaml"
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# The six histories
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Column:
    """One cached bias series: a horizon, and whether the eigenfactor stage ran."""

    horizon: Horizon
    #: The published ``a`` this history was built at; ``None`` for the control.
    scaling: float | None

    @property
    def name(self) -> str:
        """The CSV column label. ``bias_short_a1.4``, ``bias_long_none``."""
        variant = "none" if self.scaling is None else f"a{self.scaling:.1f}"
        return f"bias_{self.horizon}_{variant}"

    @property
    def stop_after(self) -> str:
        """Where the pre-VRA pipeline stops for this history."""
        return "psd_repair" if self.scaling is None else "eigenfactor"

    @property
    def label(self) -> str:
        """For a chart legend and a table row."""
        if self.scaling is None:
            return f"{self.horizon}, eigenfactor OFF"
        return f"{self.horizon}, a = {self.scaling:.1f}"


def columns(settings: config.Config | None = None) -> tuple[Column, ...]:
    """Every history the cache holds. Both horizons x both scalings, plus the control."""
    loaded = settings or config.load()
    scalings = loaded.model.eigenfactor.scaling_a
    built: list[Column] = []
    for horizon in HORIZONS:
        built.extend(Column(horizon=horizon, scaling=float(a)) for a in scalings)
        built.append(Column(horizon=horizon, scaling=None))
    return tuple(built)


#: The shared cache type. Its ``bias()`` trims the index union's NaN padding,
#: which is a no-op here -- every variant shares Model A's single panel, so no
#: column is padded -- and is load-bearing for the callers whose variants have
#: different date ranges.
Cache = history.Cache


def _minimum_observations(frame: pd.DataFrame, settings: config.Config) -> int:
    """``max(K + 1, lags + 1)``. An ARITHMETIC floor, not a burn-in choice.

    Both terms are the point below which a stage has no answer at all rather
    than a poor one:

    * ``lags + 1`` -- SPEC.md 5.2's Bartlett sum at 5 lags needs more than 5
      observations or the longest lag has no pairs;
    * ``K + 1`` -- SPEC.md 5.3's Monte Carlo re-estimates a ``K x K`` correlation
      matrix from the simulated sample, which is singular by construction at
      ``T <= K``.

    SPEC.md 5.1.2 leaves the actual burn-in with the caller and no bound on
    ``K/T_eff`` is published, so **nothing is filtered**. The ratio is carried per
    date -- it starts near ``K/7`` and falls to its asymptote -- and W4 rules on
    it with a criterion stated before the bias statistics are read. Reporting an
    unhealthy ratio is what W3-P1 built the diagnostic for; suppressing rows on
    an unpublished threshold is what it was built to avoid.
    """
    reference = RiskConfig.load(horizon=HORIZONS[0], config=settings)
    return max(frame.shape[1] + 1, reference.volatility_newey_west_lags + 1)


def _partial_path(column: Column) -> Path:
    return history.partial_path(PARTIALS, column)


def _progress_path(column: Column) -> Path:
    return PARTIALS / f"{column.name}.progress.csv"


def build_variant(column: Column, settings: config.Config | None = None) -> pd.DataFrame:
    """One history, checkpointed as it goes and resumable from where it stopped.

    Three states, checked in order:

    * a **finished** partial whose digests match -- returned as is;
    * a **progress** file whose digests match -- resumed from the date after the
      last one it recorded;
    * neither -- built from the arithmetic minimum.

    The digest pair is the same one the committed cache carries, so a checkpoint
    written against a superseded risk config or a moved panel is discarded rather
    than silently continued into. That is the whole reason resumption is safe to
    have: a resume that could splice two different models together would be worse
    than no resume at all.
    """
    loaded = settings or config.load()
    frame = macro.macro_factor_panel(config=loaded).complete
    digests = {
        "risk_config_digest": risk_config_digest(loaded),
        "panel_digest": panel_digest(frame),
    }
    path = _partial_path(column)
    if path.exists():
        finished = _read_partial(path)
        if all(finished.attrs.get(key) == value for key, value in digests.items()):
            return finished

    minimum = _minimum_observations(frame, loaded)
    progress = _progress_path(column)
    done = pd.DataFrame()
    if progress.exists():
        recorded = _read_partial(progress)
        if all(recorded.attrs.get(key) == value for key, value in digests.items()):
            done = recorded
            minimum = int(recorded["observations"].iloc[-1]) + 1
            print(f"{column.name}: resuming at {minimum} of {len(frame)}")
        else:
            progress.unlink()

    base = RiskConfig.load(horizon=column.horizon, config=loaded)
    risk = base if column.scaling is None else base.for_scaling(column.scaling)

    progress.parent.mkdir(parents=True, exist_ok=True)
    if not progress.exists():
        header = json.dumps(digests, indent=2, sort_keys=True)
        progress.write_text(
            "".join(f"# {line}\n" for line in header.splitlines())
            + "date,bias,observations,k_over_realised_t_eff\n",
            encoding="utf-8",
        )

    if minimum < len(frame):
        pending: list[str] = []

        def record(row: ForecastRow) -> None:
            pending.append(
                # 17 significant digits: the shortest precision that round-trips
                # a float64 exactly. At 12 the resumed history differed from an
                # unbroken one at 5e-12 -- immaterial to any published number, and
                # still the wrong answer to "does resuming change the result?".
                # It costs a few bytes in a file that is deleted on completion.
                f"{row.date.date()},{row.bias:.17g},{row.observations},"
                f"{row.k_over_realised_t_eff:.17g}\n"
            )
            # Appended in batches: one fsync per date would dominate the run, and
            # losing at most `_CHECKPOINT_EVERY` dates to an interruption is the
            # point of the trade. The file is append-only, so a partial final
            # line is the worst an abrupt kill can leave, and the resume reads it
            # with pandas, which would fail loudly on one.
            if len(pending) >= _CHECKPOINT_EVERY:
                with progress.open("a", encoding="utf-8") as handle:
                    handle.write("".join(pending))
                pending.clear()

        computed: ForecastHistory = forecast_history(
            frame,
            risk,
            minimum_observations=minimum,
            stop_after=column.stop_after,
            on_row=record,
        )
        if pending:
            with progress.open("a", encoding="utf-8") as handle:
                handle.write("".join(pending))
        fresh = pd.DataFrame(
            {
                "bias": computed.bias,
                "observations": computed.observations,
                "k_over_realised_t_eff": computed.k_over_realised_t_eff,
            },
            index=computed.dates,
        )
        fresh.index.name = "date"
        done = fresh if done.empty else pd.concat([done, fresh])

    table = pd.DataFrame(
        {
            column.name: done["bias"].to_numpy(dtype=float),
            f"k_over_realised_t_eff_{column.horizon}": done["k_over_realised_t_eff"].to_numpy(
                dtype=float
            ),
        },
        index=pd.DatetimeIndex(done.index),
    )
    table.index.name = "date"
    table.attrs.update(digests)
    history.write_partial(path, table, digests)
    progress.unlink(missing_ok=True)
    return table


def assemble_cache(settings: config.Config | None = None) -> Cache:
    """Join every partial into the committed cache. Builds any that are missing."""
    loaded = settings or config.load()
    frame = macro.macro_factor_panel(config=loaded).complete
    variants = columns(loaded)
    ordered = [f"k_over_realised_t_eff_{horizon}" for horizon in HORIZONS]
    ordered.extend(column.name for column in variants)
    return history.assemble(
        (build_variant(column, loaded) for column in variants),
        order=ordered,
        what=(
            "SPEC.md 5.4 bias series B_t^F, one column per model variant, built by "
            "`python -m mafrm.factors.vra_report --rebuild --all`. sigma_kt is the "
            "forecast made at t-1 from the full pre-VRA pipeline. tests/test_regime.py "
            "fails if risk_config_digest stops matching config/model.yaml."
        ),
        settings=loaded,
        # A BARE STRING, not a map: every variant here shares Model A's single
        # panel, so there is one digest and not one per variant.
        panel_digests=panel_digest(frame),
        variants=variants,
        extra={
            "model_yaml_sha256": _model_yaml_sha256(),
            "panel_start": str(frame.index[0].date()),
            "panel_end": str(frame.index[-1].date()),
            "minimum_observations": _minimum_observations(frame, loaded),
        },
    )


def build_cache(settings: config.Config | None = None) -> Cache:
    """Every history, in one process. **About ninety minutes** -- prefer ``--all``."""
    return assemble_cache(settings)


def write_cache(cache: Cache, path: Path = CACHE_PATH) -> None:
    """CSV with a ``#``-prefixed JSON provenance header. Shared format."""
    history.write_cache(cache, path)


def read_cache(path: Path = CACHE_PATH) -> Cache:
    """Read the committed cache and REFUSE a stale one.

    The refusal is here as well as in ``tests/test_regime.py`` because the test
    guards the commit and this guards the read: ``make model`` and ``make report``
    would otherwise publish numbers from a superseded config with no sign that
    anything was wrong.

    **The panel digest is recorded but not checked on read here**, and that is a
    deliberate difference from the other callers rather than an oversight. This
    cache is read by ``make model`` on every run, and rebuilding Model A's panel
    to verify it would put a fifteen-second build behind a target that is
    supposed to be fast. ``tests/test_regime.py`` carries the check instead.
    """
    return history.read_cache(
        path,
        rebuild_hint="`python -m mafrm.factors.vra_report --rebuild --all`",
    )


# ---------------------------------------------------------------------------
# The chart
# ---------------------------------------------------------------------------


def figure_path() -> Path:
    return _REPORTS / "vra_multiplier.png"


def report_path() -> Path:
    return _REPORTS / "volatility_regime.md"


def draw(cache: Cache, settings: config.Config | None = None) -> Figure:
    """``lambda_F`` over time, both horizons, in TWO panels and for a reason.

    The first four forecast dates -- 2008-04-23 to 2008-04-28, where SPEC.md
    5.2's PSD repair fires against ``K/T_eff >= 0.67`` -- carry a forecast
    inflated by four orders of magnitude, so ``B_t^F`` there is ~1e-4 and the
    rolling multiplier starts near zero. On one linear panel that burn-in
    compresses the entire rest of the series into a line.

    **Truncating the x-axis to hide it is the move this project keeps catching
    itself at**, so neither panel does. The top panel is logarithmic and shows
    every date including the burn-in; the bottom is linear and scaled to the
    bulk, so the episodes are legible while the burn-in simply runs off it with
    the top panel directly above showing where it went. Nothing is excluded and
    no threshold is chosen.
    """
    loaded = settings or config.load()
    figure, (upper, lower) = plt.subplots(
        2, 1, figsize=(11.0, 7.0), sharex=True, height_ratios=(1.0, 1.6)
    )
    series: dict[str, pd.Series] = {}
    for horizon in HORIZONS:
        halflife = RiskConfig.load(horizon=horizon, config=loaded).volatility_regime_halflife
        column = Column(horizon=horizon, scaling=loaded.model.eigenfactor.scaling_a[0])
        series[horizon] = pd.Series(
            np.sqrt(_rolling(cache.bias(column), halflife=halflife)),
            index=cache.frame.index,
        )
        label = f"{horizon} (VRA half-life {halflife}d)"
        upper.plot(cache.frame.index, series[horizon], linewidth=0.9, label=label)
        lower.plot(cache.frame.index, series[horizon], linewidth=1.1, label=label)

    upper.set_yscale("log")
    upper.set_title(
        "SPEC.md 5.4 volatility regime multiplier, $\\lambda_F$ "
        f"(a = {loaded.model.eigenfactor.scaling_a[0]:.1f})"
    )
    upper.set_ylabel("$\\lambda_F$, log scale\n(every date, burn-in included)")
    upper.axhline(1.0, color="0.4", linewidth=0.8, linestyle="--")
    upper.annotate(
        "burn-in: SPEC.md 5.2's repair fires at $K/T_{eff}\\geq0.67$,\n"
        "the forecast is inflated ~$10^4$, and $B_t^F$ collapses.\n"
        "4 dates; worth $<10^{-9}$ on the reported $\\lambda_F^2$.",
        xy=(0.012, 0.10),
        xycoords="axes fraction",
        fontsize=7,
        color="0.25",
    )

    bulk = pd.concat(series.values())
    floor = float(bulk[bulk > 0.5].min())
    lower.set_ylim(0.9 * floor, 1.05 * float(bulk.max()))
    lower.axhline(1.0, color="0.4", linewidth=0.8, linestyle="--")
    lower.set_ylabel("$\\lambda_F$, linear\n(scaled to the bulk)")
    for name, start, end in _EPISODES:
        for axes in (upper, lower):
            axes.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="0.85", zorder=0)
        lower.annotate(
            name,
            xy=(pd.Timestamp(start), lower.get_ylim()[1]),
            xytext=(3, -11),
            textcoords="offset points",
            fontsize=8,
            color="0.3",
        )
    lower.set_xlabel(
        "ROLLING statistic: consecutive values share all but one observation "
        "(CLAUDE.md failure mode 9)"
    )
    lower.legend(loc="upper right", fontsize=8)
    figure.tight_layout()
    return figure


def _rolling(bias: np.ndarray, *, halflife: int) -> np.ndarray:
    from mafrm.risk.regime import rolling_multiplier

    return rolling_multiplier(bias, halflife=halflife)


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def render(cache: Cache, settings: config.Config | None = None) -> str:
    """``reports/volatility_regime.md``."""
    loaded = settings or config.load()
    lines: list[str] = [
        "# Volatility regime adjustment (SPEC.md 5.4)",
        "",
        "Generated by `python -m mafrm.factors.vra_report`. The bias series it reads are in",
        f"`{CACHE_PATH.name}`, built at commit `{cache.provenance['git_commit'][:12]}` on",
        f"{cache.provenance['generated_utc']}.",
        "",
        "```",
        "B_t^F      = sqrt( (1/K) sum_k (f_kt / sigma_kt)^2 )   sigma_kt made at t-1",
        "lambda_F^2 = sum_t w_t (B_t^F)^2                       exponential w at tau_VRA",
        "F         <- lambda_F^2 * F",
        "```",
        "",
        f"Panel: {cache.provenance['panel_start']} to {cache.provenance['panel_end']}, "
        f"{cache.provenance['rows']:,} forecast dates, minimum window "
        f"{cache.provenance['minimum_observations']} observations. Everything ends strictly "
        f"before `sample.holdout_start` = {cache.provenance['holdout_start']}.",
        "",
        "## The multiplier, over the whole history",
        "",
        "| Variant | VRA half-life | `lambda_F^2` | `lambda_F` | vol effect | max `B_t^F` |",
        "|---|---|---|---|---|---|",
    ]
    multipliers: dict[str, float] = {}
    for column in columns(loaded):
        halflife = RiskConfig.load(horizon=column.horizon, config=loaded).volatility_regime_halflife
        multiplier = regime_multiplier(cache.bias(column), halflife=halflife)
        multipliers[column.name] = multiplier.lambda_squared
        lines.append(
            f"| {column.label} | {halflife}d | {multiplier.lambda_squared:.6f} | "
            f"{multiplier.lambda_:.6f} | {100.0 * (multiplier.lambda_ - 1.0):+.2f}% | "
            f"{multiplier.maximum_bias:.3f} |"
        )

    lines.extend(
        [
            "",
            "**Nothing is clipped.** SPEC.md 6.1 clips standardized returns at +-4 before "
            "taking a standard deviation; SPEC.md 5.4 specifies no clip and none is applied "
            "(CLAUDE.md invariant 9). The maximum single-day `B_t^F` is in the table above so "
            "a reader can judge how much of the multiplier one day carries.",
            "",
            "## The overlap with the eigenfactor stage",
            "",
            "`lambda_F^2` with SPEC.md 5.3's stage on and with it off. The VRA corrects for "
            "REGIME and the eigenfactor stage corrects for SAMPLING ERROR, which are different "
            "objects -- but the VRA is fitted to realised data, so it absorbs any systematic "
            "level bias including the one the eigenfactor stage exists to remove. The "
            "difference below is the size of that absorption.",
            "",
            "| Horizon | `a` | on | off | off - on | as vol |",
            "|---|---|---|---|---|---|",
        ]
    )
    for horizon in HORIZONS:
        off = multipliers[Column(horizon=horizon, scaling=None).name]
        for scaling in loaded.model.eigenfactor.scaling_a:
            on = multipliers[Column(horizon=horizon, scaling=float(scaling)).name]
            lines.append(
                f"| {horizon} | {scaling:.1f} | {on:.6f} | {off:.6f} | {off - on:+.6f} | "
                f"{100.0 * (np.sqrt(off) / np.sqrt(on) - 1.0):+.3f}% |"
            )

    unreachable, start, end = _UNREACHABLE_EPISODE
    lines.extend(
        [
            "",
            "## Episodes",
            "",
            f"![lambda_F over time]({figure_path().name})",
            "",
            "SPEC.md 5.4 names three episodes a working VRA must respond to. Two are on this "
            "panel and are shaded in the figure:",
            "",
        ]
    )
    for name, start_date, end_date in _EPISODES:
        for horizon in HORIZONS:
            halflife = RiskConfig.load(horizon=horizon, config=loaded).volatility_regime_halflife
            column = Column(horizon=horizon, scaling=loaded.model.eigenfactor.scaling_a[0])
            rolling = pd.Series(
                np.sqrt(_rolling(cache.bias(column), halflife=halflife)),
                index=cache.frame.index,
            )
            window = rolling.loc[start_date:end_date]
            baseline = rolling.loc[: pd.Timestamp(start_date)]
            if window.empty or baseline.empty:
                continue
            lines.append(
                f"- **{name}, {horizon}**: `lambda_F` peaks at {window.max():.3f} "
                f"(from {baseline.iloc[-1]:.3f} entering the window), "
                f"{100.0 * (window.max() / baseline.iloc[-1] - 1.0):+.1f}%."
            )
    lines.extend(
        [
            "",
            f"**{unreachable} is NOT reachable on this panel and no claim is made about it.** "
            f"The complete orthogonalized macro factor panel begins "
            f"{cache.provenance['panel_start']}, bound by `credit` (first valid 2008-04-14) "
            "and `rates_slope` (2008-04-02) through the 252-observation expanding-window "
            "orthogonalization burn-in off `sample.start` = 2007-04-01 -- a documented "
            f"consequence of W2-P2's row 69. The {unreachable} window ({start} to {end}) sits "
            "entirely before the first date this model can forecast. Stating the exclusion is "
            "the point: a chart that quietly began after the window it claimed to cover would "
            "read as a pass.",
            "",
            "## Reading the chart",
            "",
            "`lambda_F` here is a **rolling statistic** -- element `t` is the exponentially "
            "weighted mean of `B^2` over every forecast date up to `t`, so consecutive values "
            "share all but one observation and the series is enormously autocorrelated "
            "(CLAUDE.md failure mode 9). Its wiggles are not independent readings and **no "
            "count of its points is quoted as an `n`** anywhere in this report.",
            "",
            "## The burn-in, which is in the figure rather than hidden",
            "",
            "At the first four forecast dates -- **2008-04-23 to 2008-04-28**, where `K/T_eff` "
            "runs from **0.857 down to 0.667** -- the forecast volatility is inflated by about "
            "**four orders of magnitude**, so `B_t^F` is ~1e-4 and the rolling multiplier starts "
            "near zero. Traced stage by stage, the EWMA, Newey-West and PSD-repair outputs are "
            "all sane; **the eigenfactor stage produces it**, and only on the dates where "
            "SPEC.md 5.2's repair has fired. The repair floors a near-zero eigenvalue of "
            "`rho_hat` at 1e-14, SPEC.md 5.3's Monte Carlo then measures an enormous `lambda(k)` "
            "for a direction with essentially no variance, and the parabola fit spreads that "
            "through all six `gamma(k)` -- which is why every factor inflates rather than one. "
            "Two corrections that each pass their own check, compounding in sequence: "
            "CLAUDE.md failure mode 7. SPEC.md 5.4.4 carries the full record.",
            "",
            "**It is worth at most 1.1e-9 on every `lambda_F^2` in the table above**, because a "
            "2008 date carries weight 1.5e-31 (short) or 6.4e-10 (long) in an exponentially "
            "weighted sum ending in 2024. It is **not** fixed here: the fix is a burn-in, there "
            "is no published bound on `K/T_eff`, and inventing one would be CLAUDE.md invariant "
            "9 (SPEC.md 5.4.1). What it gives W4 is a second, measured argument for the bound "
            "SPEC.md 5.1.2 pre-committed to -- with a located mechanism and a known reach.",
            "",
            "## What is NOT here",
            "",
            "SPEC.md 5.4's **specific-risk multiplier** -- a notional-weighted cross-sectional "
            "bias statistic across assets at the same `tau_VRA`. It needs specific returns, "
            "which SPEC.md 5.5 has not built. It is absent rather than stubbed. The shared "
            "half-life already sits at `volatility_regime_adjustment.halflife`, in its own "
            "top-level section precisely so the two legs cannot drift apart, so the session "
            "that writes SPEC.md 5.5 reads the same key and adds no parameter.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Regenerate the report and figure.

    ``--rebuild --variant NAME`` builds one history into its partial;
    ``--rebuild --all`` fans the variants out across processes and then
    assembles; ``--rebuild`` alone assembles in this process, building whatever
    is missing. With no flag at all the committed cache is read and only the
    report and figure are redrawn, which is what ``make report`` runs.
    """
    import sys

    arguments = sys.argv[1:] if argv is None else argv
    settings = config.load()

    if "--variant" in arguments:
        name = arguments[arguments.index("--variant") + 1]
        column = next((one for one in columns(settings) if one.name == name), None)
        if column is None:
            print(f"unknown variant {name!r}; known: {[c.name for c in columns(settings)]}")
            return 2
        table = build_variant(column, settings)
        print(f"{name}: {len(table):,} dates -> {_partial_path(column)}")
        return 0

    if "--rebuild" in arguments and "--all" in arguments:
        # Fanned out, and the reason is durability as much as speed: a single
        # ninety-minute process that is killed leaves nothing behind and looks
        # exactly like one that has not finished. See the module docstring.
        code = history.fan_out("mafrm.factors.vra_report", columns(settings), flag="--variant")
        if code:
            return code

    if "--rebuild" in arguments:
        cache = assemble_cache(settings)
        write_cache(cache)
        print(f"wrote {CACHE_PATH} ({cache.provenance['rows']:,} dates)")
    else:
        cache = read_cache()

    figure = draw(cache, settings)
    figure.savefig(figure_path(), dpi=160)
    plt.close(figure)
    report_path().write_text(render(cache, settings), encoding="utf-8")
    print(f"wrote {report_path()} and {figure_path()}")
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
