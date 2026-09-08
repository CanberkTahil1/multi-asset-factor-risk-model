"""Generates ``reports/universe_screen.md``, ``.png`` and ``.csv``. SPEC.md 15.3.

What the report says, in order: that the membership matrix is an approximate
point-in-time universe reconstructed from an incomplete source, with the gaps
counted per year; what the loader's per-ticker status says about the source's
coverage of departed names, as numbers and -- for anything still ``failed``
after the retry -- as names; and how many names the estimation-universe
screen drops per session, by cause. It also scores experiments.md rows
265-267 against the thresholds held in ``equity_universe.registrations`` and
prints each verdict, and records row 268's measurement, which was not
pre-registered and says so.

**The files are written only when the report runs on the snapshot's own
recorded inputs.** Raw pulls are keyed by date and Yahoo serves delisted
histories intermittently, so a later pull -- even a same-day one -- gives
different numbers; on such a pull the report prints its comparison, says the
verdicts are not the ones of record, and leaves ``reports/universe_screen.*``
untouched (W7-P1b). A fresh clone therefore cannot regenerate these files
byte for byte, and the README says so.

Reads the COMMITTED reference tables and the cache. The reconstruction is run
again from the cached Wikipedia tables and compared with the committed
interval file: if the two differ the report refuses, because a reference file
that drifted from its source without ``make data`` rewriting it is a
provenance failure, not something to render around. The screen makes the same
check on the matrix's third state against the bars.

CLAUDE.md invariant 5: the screen and every series drawn stop strictly before
``sample.holdout_start``; the bars are read through ``cache.read`` and the
matrix is cut on load. The reconstruction itself and the coverage table are
source snapshots and reach the snapshot date by construction (SPEC.md 15.3.1);
the report states their reach only as counts of rows per year and of tickers
per status.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # No display in CI; must precede the pyplot import.

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from mafrm import config as config_mod
from mafrm.data import cache, calendar, loaders, sp500, sp500_reference
from mafrm.factors import equity_universe

__all__ = ["UniverseReport", "Verdict", "build", "main", "render", "score"]


def _report_path(name: str) -> Path:
    return Path(__file__).resolve().parents[3] / "reports" / name


@dataclass(frozen=True)
class Verdict:
    row: int
    leg: str
    holds: bool
    detail: str


@dataclass(frozen=True)
class Coverage:
    """The loader's per-ticker status (ruling 6), summarised as counts and names."""

    requested: int
    ok: int
    empty: int
    failed: int
    failed_names: tuple[str, ...]
    #: Departed names: on the removed side of the changes table, not current.
    departed: int
    departed_ok: int
    departed_empty: int
    departed_failed: int


@dataclass(frozen=True)
class UniverseReport:
    reference: sp500_reference.ReferenceBuild
    committed_intervals: pd.DataFrame
    committed_matrix: pd.DataFrame
    screen: equity_universe.Screen
    coverage: Coverage
    #: Members with no valid bar in the CALENDAR MONTH of the first sample
    #: month end -- the lost attempt's monthly third state, for row 267.
    monthly_no_data_first: int
    first_month_end: pd.Timestamp
    verdicts: tuple[Verdict, ...]
    #: Row 268, NOT PRE-REGISTERED: the worst session for insufficient history.
    worst_insufficient: tuple[pd.Timestamp, int, tuple[str, ...]]
    sample_start: pd.Timestamp
    holdout_start: pd.Timestamp
    #: True when the comparisons ran on the snapshot's own recorded inputs
    #: (one-snapshot rule, W7-P1b); False when the cache has moved past them
    #: and the drift checks are reported as counts rather than refusals.
    on_snapshot_inputs: bool = True
    #: The interval drift check, when not on the snapshot's inputs.
    interval_drift: str | None = None
    #: Tickers whose committed third state disagrees with the bars the report
    #: ran on, when not on the snapshot's inputs.
    disagreeing_tickers: tuple[str, ...] = ()


def _interval_drift(rec: sp500.Reconstruction, committed: pd.DataFrame) -> str | None:
    left = rec.intervals.reset_index(drop=True)
    right = committed[left.columns].reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(left, right, check_dtype=False)
    except AssertionError as exc:
        return str(exc).splitlines()[0]
    return None


def _assert_committed_matches(rec: sp500.Reconstruction, committed: pd.DataFrame) -> None:
    drift = _interval_drift(rec, committed)
    if drift is not None:
        raise RuntimeError(
            "the committed interval table differs from the reconstruction of its own recorded "
            "inputs. The reference file has been hand-edited or its inputs replaced; rewrite it "
            "deliberately with `python -m mafrm.data.sp500_reference --resnapshot` and commit "
            f"the change with its reason.\n{drift}"
        )


def _coverage(inputs: sp500_reference.ReferenceInputs) -> Coverage:
    status = dict(inputs.coverage_entry.status)
    if not status:
        raise RuntimeError(
            "the cached sp500_coverage entry carries no per-ticker status; re-run `make data` "
            "with the W7-P1 loader (SPEC.md 15.3.1 ruling 6)"
        )
    current = set(inputs.constituents["ticker"].astype(str))
    departed = {str(t) for t in inputs.changes["removed"].dropna()} - current

    def count(names: set[str] | None, value: str) -> int:
        return sum(1 for t, s in status.items() if s == value and (names is None or t in names))

    return Coverage(
        requested=len(status),
        ok=count(None, loaders.OK),
        empty=count(None, loaders.EMPTY),
        failed=count(None, loaders.FAILED),
        failed_names=tuple(sorted(t for t, s in status.items() if s == loaders.FAILED)),
        departed=len(departed),
        departed_ok=count(departed, loaders.OK),
        departed_empty=count(departed, loaders.EMPTY),
        departed_failed=count(departed, loaders.FAILED),
    )


def _monthly_no_data(
    close: pd.DataFrame, volume: pd.DataFrame, matrix: pd.DataFrame, month_end: pd.Timestamp
) -> int:
    """Members at ``month_end`` with no valid bar anywhere in its calendar month."""
    valid = sp500_reference.valid_bars(close, volume)
    period = pd.Timestamp(month_end).to_period("M")
    in_month = valid[pd.DatetimeIndex(valid.index).to_period("M") == period].any(axis=0)
    states = matrix.loc[month_end].to_numpy()
    member_names = [
        str(t) for t, s in zip(matrix.columns, states, strict=True) if int(s) != sp500.NOT_MEMBER
    ]
    present = in_month.reindex(member_names).fillna(False).astype(bool)
    return int((~present).sum())


def score(
    cfg: config_mod.Config,
    *,
    matrix: pd.DataFrame,
    screen: equity_universe.Screen,
    coverage: Coverage,
    monthly_no_data_first: int,
    first_month_end: pd.Timestamp,
    sample_start: pd.Timestamp,
    holdout_start: pd.Timestamp,
) -> tuple[Verdict, ...]:
    """experiments.md rows 265-267 against the registered thresholds."""
    universe = cfg.model.equity_universe
    reg = universe.registrations
    out: list[Verdict] = []

    out.append(
        Verdict(
            265,
            f"(a) tickers still failed after the retry at most {reg.row_265_failed_after_retry_max}",
            coverage.failed <= reg.row_265_failed_after_retry_max,
            f"{coverage.failed} failed of {coverage.requested} requested"
            + (f": {', '.join(coverage.failed_names)}" if coverage.failed_names else ""),
        )
    )
    gap = coverage.empty - reg.row_265_empty_expected
    out.append(
        Verdict(
            265,
            f"(b) empty count within {reg.row_265_empty_tolerance} of the lost attempt's "
            f"{reg.row_265_empty_expected} names without bars",
            abs(gap) <= reg.row_265_empty_tolerance,
            f"{coverage.empty} empty, {coverage.ok} ok: difference {gap:+d}",
        )
    )

    window = matrix[(matrix.index >= sample_start) & (matrix.index < holdout_start)]
    members = (window != sp500.NOT_MEMBER).sum(axis=1)
    low, high = universe.membership_count_band
    lowest, highest = int(members.to_numpy().min()), int(members.to_numpy().max())
    out.append(
        Verdict(
            266,
            "member count inside the band on every session of the sample",
            bool(lowest >= low and highest <= high),
            f"min {lowest} ({pd.Timestamp(members.idxmin()).date()}), "
            f"max {highest} ({pd.Timestamp(members.idxmax()).date()}) "
            f"against [{low}, {high}], {len(members)} sessions",
        )
    )

    daily_first = int(screen.counts["member_no_data"].loc[first_month_end])
    excess = daily_first - monthly_no_data_first
    out.append(
        Verdict(
            267,
            f"daily minus monthly 'member, no data' at {first_month_end.date()} below "
            f"{reg.row_267_no_data_excess_max}",
            0 <= excess < reg.row_267_no_data_excess_max,
            f"daily {daily_first}, monthly {monthly_no_data_first}, excess {excess}",
        )
    )
    return tuple(out)


def build(cfg: config_mod.Config | None = None) -> UniverseReport:
    cfg = cfg or config_mod.load()
    universe = cfg.model.equity_universe
    sample_start = pd.Timestamp(cfg.model.sample.start)
    holdout_start = pd.Timestamp(cfg.require_holdout_start())

    manifest = cache.Manifest.load()
    snapshot_entries = sp500_reference.snapshot_input_entries(cfg, manifest)
    on_snapshot = snapshot_entries is not None
    inputs = sp500_reference.load_inputs(manifest, entries=snapshot_entries)
    reference = sp500_reference.build(cfg, inputs=inputs, write=False)
    directory = sp500_reference.reference_dir(cfg)
    committed_intervals = sp500.load_intervals(directory / universe.intervals_file)
    interval_drift: str | None = None
    if on_snapshot:
        _assert_committed_matches(reference.reconstruction, committed_intervals)
    else:
        interval_drift = _interval_drift(reference.reconstruction, committed_intervals)
    committed_matrix = equity_universe.load_membership(cfg)
    coverage = _coverage(reference.inputs)

    close, volume = equity_universe.load_bars(cfg, entry=inputs.prices_entry)
    result = equity_universe.screen(
        close,
        volume,
        committed_matrix,
        min_history_days=universe.min_history_days,
        min_dollar_adv=universe.min_dollar_adv,
        start=sample_start,
        end=holdout_start,
        strict=on_snapshot,
    )
    disagreeing: tuple[str, ...] = ()
    if result.state_disagreements:
        window = committed_matrix.reindex(index=result.member_no_data.index)
        committed_no_data = window == sp500.MEMBER_NO_DATA
        bars_no_data = result.member_no_data.reindex(columns=window.columns).fillna(False)
        differs = (committed_no_data != bars_no_data).any(axis=0)
        disagreeing = tuple(sorted(str(t) for t, flag in differs.items() if bool(flag)))
    month_ends = calendar.month_end_dates(pd.DatetimeIndex(result.counts.index))
    first_month_end = pd.Timestamp(month_ends[0])
    monthly_first = _monthly_no_data(close, volume, committed_matrix, first_month_end)

    worst_series = result.counts["insufficient_history"]
    worst_date = pd.Timestamp(worst_series.idxmax())
    worst_row = result.insufficient_history.loc[worst_date]
    worst_names = tuple(sorted(str(t) for t, flag in worst_row.items() if bool(flag)))

    verdicts = score(
        cfg,
        matrix=committed_matrix,
        screen=result,
        coverage=coverage,
        monthly_no_data_first=monthly_first,
        first_month_end=first_month_end,
        sample_start=sample_start,
        holdout_start=holdout_start,
    )
    return UniverseReport(
        reference=reference,
        committed_intervals=committed_intervals,
        committed_matrix=committed_matrix,
        screen=result,
        coverage=coverage,
        monthly_no_data_first=monthly_first,
        first_month_end=first_month_end,
        verdicts=verdicts,
        worst_insufficient=(worst_date, int(worst_series.max()), worst_names),
        sample_start=sample_start,
        holdout_start=holdout_start,
        on_snapshot_inputs=on_snapshot,
        interval_drift=interval_drift,
        disagreeing_tickers=disagreeing,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _gap_rows(gaps: sp500.GapCounts, *, from_year: int) -> list[str]:
    lines: list[str] = []
    early_years = [y for y in gaps.rows if y < from_year]
    if early_years:
        lines.append(
            f"| {min(early_years)}-{max(early_years)} | "
            f"{sum(gaps.rows[y] for y in early_years)} | "
            f"{sum(gaps.additions_without_row[y] for y in early_years)} | "
            f"{sum(gaps.inconsistencies[y] for y in early_years)} | "
            f"{sum(gaps.rename_links[y] for y in early_years)} |"
        )
    for year in sorted(y for y in gaps.rows if y >= from_year):
        lines.append(
            f"| {year} | {gaps.rows[year]} | {gaps.additions_without_row[year]} | "
            f"{gaps.inconsistencies[year]} | {gaps.rename_links[year]} |"
        )
    return lines


def render(report: UniverseReport, cfg: config_mod.Config) -> str:
    universe = cfg.model.equity_universe
    rec = report.reference.reconstruction
    prov = report.reference.provenance
    cov = report.coverage
    counts = report.screen.counts
    month_ends = calendar.month_end_dates(pd.DatetimeIndex(counts.index))
    monthly = counts.loc[month_ends]
    december = monthly[pd.DatetimeIndex(monthly.index).month == 12]
    window = report.committed_matrix[
        (report.committed_matrix.index >= report.sample_start)
        & (report.committed_matrix.index < report.holdout_start)
    ]
    members_window = (window != sp500.NOT_MEMBER).sum(axis=1).to_numpy()
    adv_line = (
        f"applied at {universe.min_dollar_adv:,.0f} USD/day"
        if universe.min_dollar_adv is not None
        else "**not applied: `equity_universe.min_dollar_adv` is null** -- no published "
        "threshold exists and none is invented (SPEC.md 15.3.1 ruling 1); the liquidity "
        "screen is the one S&P applied at every addition, and this project has no float "
        "data to re-apply it"
    )
    first, last = counts.iloc[0], counts.iloc[-1]
    verdict_lines = [
        f"| {v.row} | {v.leg} | {'**HOLDS**' if v.holds else '**REFUTED**'} | {v.detail} |"
        for v in report.verdicts
    ]
    worst_date, worst_n, worst_names = report.worst_insufficient
    failed_line = (
        f"**{cov.failed} still `failed` after the retry: {', '.join(cov.failed_names)}** -- "
        "listed by name and NOT folded into the coverage count (ruling 6)."
        if cov.failed_names
        else "**No ticker is still `failed` after the retry**, so every absence below is a "
        "provider answer, not a transport failure (ruling 6)."
    )
    departed_empty_fraction = cov.departed_empty / cov.departed if cov.departed else 0.0
    lines = [
        "# The equity estimation universe -- SPEC.md 15.3",
        "",
        "Generated by `python -m mafrm.factors.universe_report`. Rulings at SPEC.md 15.3.1;",
        "experiments.md rows 261-268.",
        "",
        "## What this is, and is not",
        "",
        "An **approximate point-in-time S&P 500 membership**, reconstructed by walking",
        "Wikipedia's changes table backwards from its current constituent list. It is a",
        "genuine attempt at survivorship control and **not a solution**: the changes table is",
        "incomplete, every addition or removal it omits is inherited by the walk, and the",
        "per-year table below counts what is known to be missing. **No selection procedure",
        "this project did not run is claimed.** Delisted names vanish from Yahoo and no free",
        "source restores them; the count of members without a bar, per session, is the",
        "residual survivorship bias stated as a number rather than as a caveat (ruling 5).",
        "",
        "## Provenance",
        "",
        f"- Constituents: {universe.constituents_url}",
        f"- Changes: {universe.changes_url}",
        f"- Snapshot {prov.snapshot.date()}, pulled {prov.pulled_at}; bars pull {prov.bars_pull}.",
        f"- Licence of the committed tables: {universe.licence} ({universe.licence_url}), as a",
        "  derived work of Wikipedia text, separate from the repository's code licence.",
        f"- Committed: `{universe.reference_dir}/{universe.intervals_file}` "
        f"({len(rec.intervals)} spells, {rec.intervals['ticker'].nunique()} tickers) and",
        f"  `{universe.reference_dir}/{universe.membership_file}` "
        f"({report.reference.matrix.shape[0]} trading sessions x {report.reference.matrix.shape[1]} tickers,",
        f"  {pd.Timestamp(report.reference.matrix.index.min()).date()} to "
        f"{pd.Timestamp(report.reference.matrix.index.max()).date()}, DAILY; the pipeline reads",
        "  month ends off it, ruling 4).",
        "- **One snapshot.** The committed files were written once and are not re-snapshotted "
        "during the build (ruling 3, W7-P1b); `make data` keeps them, and a rewrite is a "
        "deliberate `--resnapshot` committed with its reason. "
        + (
            "This report ran on the snapshot's own recorded inputs, so the drift checks were "
            "strict."
            if report.on_snapshot_inputs
            else "**The cache has moved past the snapshot**: this report ran on the latest pull, "
            f"the drift checks are reported rather than enforced -- interval drift: "
            f"{report.interval_drift or 'none'}; third-state disagreements: "
            f"{report.screen.state_disagreements} cell(s) over {len(report.disagreeing_tickers)} "
            f"ticker(s): {', '.join(report.disagreeing_tickers[:30]) or 'none'}"
            f"{' ...' if len(report.disagreeing_tickers) > 30 else ''}. A disagreement here is "
            "the vendor serving a delisted history on one pull and not on another; the "
            "committed matrix records what the snapshot pull returned."
        ),
        f"- The changes table reaches back to **{rec.reach.date()}**; membership before the",
        "  first year with a plausible row count is what the walk carries, not what the index held.",
        f"- Rename links applied: {len(rec.rename_links)} "
        f"({', '.join(f'{a}<-{b} {d.date()}' for a, b, d in rec.rename_links[:12])}"
        f"{' ...' if len(rec.rename_links) > 12 else ''}).",
        f"- Current members whose `Date added` did not parse: {len(rec.undated_members)}"
        f"{' (' + ', '.join(rec.undated_members) + ')' if rec.undated_members else ''}.",
        "",
        "## Known gaps, per year (SPEC.md 15.3.1 ruling 4)",
        "",
        "Counts of rows and names, never rates. *Additions without a row* counts CURRENT",
        "members whose `Date added` has no addition row -- a lower bound that cannot see a",
        "name added and since removed. *Inconsistencies* are rows the walk could not apply:",
        "an added ticker not in the running set, or a removed one already in it.",
        "",
        "| Year | Changes rows | Additions without a row | Inconsistencies | Rename links |",
        "|---|---|---|---|---|",
        *_gap_rows(report.reference.gaps, from_year=1997),
        "",
        "## Coverage of the bars (rulings 5 and 6)",
        "",
        "The loader records a status per requested ticker in `data/manifest.json`: `ok`",
        "(bars came back), `empty` (Yahoo answered and has no history -- its chart endpoint",
        "returned its explicit *no data found* payload), or `failed` (an exception, timeout",
        "or HTTP error, retried once). Only `empty` becomes *member, no data*.",
        "",
        f"- Tickers requested (current plus every added or removed name): **{cov.requested}**: "
        f"ok {cov.ok}, empty {cov.empty}, failed {cov.failed}.",
        f"- {failed_line}",
        f"- **Departed names** (on the removed side, not current): {cov.departed}; ok "
        f"{cov.departed_ok}, empty {cov.departed_empty} (**{departed_empty_fraction:.1%}** of "
        f"departed names have no Yahoo history), failed {cov.departed_failed}.",
        f"- Members with no valid bar on the session at {counts.index[0].date()}: "
        f"**{int(first['member_no_data'])}** of {int(first['members'])}; at "
        f"{counts.index[-1].date()}: {int(last['member_no_data'])} of {int(last['members'])}.",
        f"- The same count in the lost attempt's MONTHLY definition (no valid bar anywhere in "
        f"the calendar month) at {report.first_month_end.date()}: {report.monthly_no_data_first}; "
        f"daily on that session: {int(counts['member_no_data'].loc[report.first_month_end])} "
        "(row 267).",
        "",
        "## The screen",
        "",
        f"- History: **{universe.min_history_days} valid daily bars for the ticker in the "
        f"trailing {universe.min_history_days} sessions** (SPEC.md 15.3; ruling 2 -- history, "
        "not tenure in the index).",
        f"- Dollar ADV: {adv_line}.",
        f"- Window: every trading session from {counts.index[0].date()} to "
        f"{counts.index[-1].date()}, {len(counts)} sessions, strictly before the holdout "
        f"boundary {report.holdout_start.date()}.",
        "",
        "Members by cause at each December month end (the last traded session of the month;",
        "the full daily series is in `reports/universe_screen.csv`):",
        "",
        "| Date | Members | No data | Insufficient history | Below dollar ADV | Estimation universe |",
        "|---|---|---|---|---|---|",
        *(
            f"| {stamp.date()} | {r['members']} | {r['member_no_data']} | "
            f"{r['insufficient_history']} | {r['below_dollar_adv']} | {r['estimation_universe']} |"
            for stamp, r in zip(
                pd.DatetimeIndex(december.index), december.to_dict("records"), strict=True
            )
        ),
        "",
        f"Estimation universe over the window: min {int(counts['estimation_universe'].min())}, "
        f"max {int(counts['estimation_universe'].max())}. Members without data: min "
        f"{int(counts['member_no_data'].min())}, max {int(counts['member_no_data'].max())}.",
        "",
        f"**Row 268, NOT PRE-REGISTERED (measured):** insufficient history peaks at "
        f"**{worst_n}** on {worst_date.date()}: {', '.join(worst_names[:40])}"
        f"{' ...' if len(worst_names) > 40 else ''}. The lost attempt measured 11 at "
        "2020-02-28 under its monthly construction; this is the daily figure, and the names",
        "are listed so the reader can check each against its listing date rather than take",
        "a mechanism on trust.",
        "",
        "## The count band (experiments.md row 266)",
        "",
        f"Member count from the committed matrix on every session of "
        f"{report.sample_start.date()} to {report.holdout_start.date()}: min "
        f"{int(members_window.min())}, max {int(members_window.max())}, band "
        f"{list(universe.membership_count_band)}.",
        "",
        "## Registered rows, scored",
        "",
        "| Row | Leg | Verdict | Measured |",
        "|---|---|---|---|",
        *verdict_lines,
        "",
        "![screen](universe_screen.png)",
        "",
    ]
    return "\n".join(lines)


def _draw(report: UniverseReport) -> Figure:
    counts = report.screen.counts
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    top.stackplot(
        counts.index,
        counts["estimation_universe"],
        counts["below_dollar_adv"],
        counts["insufficient_history"],
        counts["member_no_data"],
        labels=[
            "estimation universe",
            "below dollar ADV",
            "insufficient history",
            "member, no data",
        ],
        colors=["#4c72b0", "#dd8452", "#55a868", "#c44e52"],
        alpha=0.85,
    )
    top.plot(
        counts.index, counts["members"], color="black", linewidth=0.8, label="members (Wikipedia)"
    )
    top.set_ylabel("names")
    top.set_title("S&P 500 estimation universe: members per session by cause (SPEC.md 15.3)")
    top.legend(loc="lower left", fontsize=8, ncol=3)
    bottom.plot(counts.index, counts["member_no_data"], color="#c44e52", label="member, no data")
    bottom.plot(
        counts.index, counts["insufficient_history"], color="#55a868", label="insufficient history"
    )
    bottom.set_ylabel("names dropped")
    bottom.set_title(
        "Names the screen drops, by cause -- the red series is the residual survivorship bias"
    )
    bottom.legend(loc="upper right", fontsize=8)
    for ax in (top, bottom):
        ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    return fig


def main() -> int:
    cfg = config_mod.load()
    report = build(cfg)
    for v in report.verdicts:
        print(f"row {v.row} {'HOLDS ' if v.holds else 'REFUTED'} {v.leg}: {v.detail}")
    worst_date, worst_n, _ = report.worst_insufficient
    print(f"row 268 (not pre-registered) max insufficient_history {worst_n} at {worst_date.date()}")
    if not report.on_snapshot_inputs:
        print(
            "the cache has moved past the snapshot (one-snapshot rule, SPEC.md 15.3.1 ruling 3): "
            "the verdicts above were computed on the latest pull and are NOT the rows' verdicts "
            f"of record; third-state disagreements {report.screen.state_disagreements} cell(s) "
            f"over {len(report.disagreeing_tickers)} ticker(s): "
            f"{', '.join(report.disagreeing_tickers) or 'none'}. reports/universe_screen.* are "
            "the snapshot's run and are left untouched; re-score deliberately with "
            "`python -m mafrm.data.sp500_reference --resnapshot`, committed with its reason."
        )
        return 0

    figure = _draw(report)
    png = _report_path("universe_screen.png")
    png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    csv = _report_path("universe_screen.csv")
    report.screen.counts.to_csv(csv, lineterminator="\n")
    md = _report_path("universe_screen.md")
    md.write_text(render(report, cfg), encoding="utf-8")
    print(f"wrote {md}, {png}, {csv}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
