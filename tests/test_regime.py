"""SPEC.md 5.4's volatility regime adjustment. W3-P4.

Five things are pinned here, and the middle three are the ones a later session
could break without noticing:

* the arithmetic, against fractions computed by hand;
* that ``sigma_kt`` is the forecast made at ``t-1`` -- tested by perturbing the
  return on date ``t`` and asserting the forecast for ``t`` does not move;
* that the stage changes the LEVEL of risk and not the correlation structure,
  which is SPEC.md 5.4's own claim and is testable rather than descriptive;
* that a caller who supplies no forecast history is refused rather than handed
  an un-adjusted matrix;
* that the cached forecast history in ``reports/`` still describes the config
  that is in ``config/model.yaml`` today.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mafrm.config import load
from mafrm.numerics import ewma_weights
from mafrm.risk import regime as regime_module
from mafrm.risk.checks import eigenvalue_floor
from mafrm.risk.config import HORIZONS, RiskConfig
from mafrm.risk.covariance import CovarianceError, correlation_from_covariance, run_pipeline
from mafrm.risk.regime import (
    RegimeError,
    cross_sectional_bias,
    forecast_history,
    regime_multiplier,
)
from mafrm.risk.synthetic import panel, specification


def _frame(values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        values,
        columns=[f"f{index}" for index in range(values.shape[1])],
        index=pd.bdate_range("2010-01-04", periods=values.shape[0]),
    )


# ---------------------------------------------------------------------------
# The arithmetic, by hand
# ---------------------------------------------------------------------------


def test_the_cross_sectional_bias_is_hand_computed() -> None:
    """HAND-COMPUTED. SPEC.md 5.4, first line.

    One date, two factors. ``f = [0.02, -0.01]`` against ``sigma = [0.01, 0.01]``
    standardises to ``b = [2, -1]``, so

        B = sqrt((2^2 + (-1)^2) / 2) = sqrt(5/2) = sqrt(2.5).

    The SIGN of the standardized return is gone by construction -- the statistic
    squares before it averages -- which is what makes it a volatility signal
    rather than a directional one.
    """
    returns = np.array([[0.02, -0.01]])
    deviations = np.array([[0.01, 0.01]])
    bias = cross_sectional_bias(returns, deviations)
    assert bias.shape == (1,)
    assert bias[0] == pytest.approx(np.sqrt(2.5))

    flipped = cross_sectional_bias(-returns, deviations)
    assert flipped[0] == pytest.approx(bias[0])


def test_the_multiplier_is_hand_computed() -> None:
    """HAND-COMPUTED. SPEC.md 5.4, second line.

    Two dates at ``halflife = 1``. Weights are ``[1/2, 1]`` oldest-first,
    normalised to ``[1/3, 2/3]``, so with ``B = [1, 2]``

        lambda_F^2 = (1/3)(1^2) + (2/3)(2^2) = 1/3 + 8/3 = 3,

    and ``lambda_F = sqrt(3)``. The most recent date carries twice the weight of
    the one before it, which is the whole point of the stage: it is a FAST
    correction and the newest observation must dominate.
    """
    multiplier = regime_multiplier(np.array([1.0, 2.0]), halflife=1)
    assert multiplier.lambda_squared == pytest.approx(3.0)
    assert multiplier.lambda_ == pytest.approx(np.sqrt(3.0))
    assert multiplier.observations == 2
    assert multiplier.maximum_bias == 2.0


def test_a_perfectly_calibrated_history_gives_exactly_one() -> None:
    """The null. ``B = 1`` at every date makes the stage the identity.

    True for ANY history length and ANY half-life, because the weights are
    normalised. That is what makes the multiplier a statement about calibration
    rather than about how much history the caller happened to supply.
    """
    for size in (1, 7, 500):
        for halflife in (42, 168):
            multiplier = regime_multiplier(np.ones(size), halflife=halflife)
            assert multiplier.lambda_squared == pytest.approx(1.0, abs=1e-15)


def test_the_multiplier_uses_the_configured_half_life_at_each_horizon() -> None:
    """42d short, 168d long, from ``volatility_regime_adjustment.halflife``.

    A rising bias series must give a LARGER multiplier at the short half-life,
    because the short one puts more weight on the recent, higher readings. Same
    series, same code, different config value -- so this fails if the half-life
    is ever read from the wrong key or hard-coded.
    """
    rising = np.linspace(0.5, 1.5, 400)
    short = RiskConfig.load(horizon="short")
    long = RiskConfig.load(horizon="long")
    assert (short.volatility_regime_halflife, long.volatility_regime_halflife) == (42, 168)

    fast = regime_multiplier(rising, halflife=short.volatility_regime_halflife)
    slow = regime_multiplier(rising, halflife=long.volatility_regime_halflife)
    assert fast.lambda_squared > slow.lambda_squared
    assert fast.realised_effective_sample_size < slow.realised_effective_sample_size


def test_the_weights_are_the_projects_own_and_are_normalised() -> None:
    """No second decay implementation. ``mafrm.numerics.ewma_weights`` or nothing."""
    bias = np.linspace(0.8, 1.4, 90)
    weights = ewma_weights(bias.size, 42.0)
    assert weights.sum() == pytest.approx(1.0)
    expected = float(weights @ np.square(bias))
    assert regime_multiplier(bias, halflife=42).lambda_squared == pytest.approx(expected, rel=1e-15)


# ---------------------------------------------------------------------------
# SPEC.md 5.4's own claim: the LEVEL of risk, not the correlation structure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("horizon", HORIZONS)
def test_the_stage_leaves_the_correlation_matrix_unchanged(horizon: str) -> None:
    """SPEC.md 5.4: *"it changes the level of risk, not the correlation structure"*.

    That sentence is a testable claim and this is the test. ``corr(cF) = corr(F)``
    for any ``c > 0``, so a stage that had started doing anything other than a
    scalar multiplication -- a per-factor multiplier, a shrinkage toward a target,
    a re-estimated correlation -- would move this and nothing else in the suite
    would notice.
    """
    frame = panel(specification(6))
    config = RiskConfig.load(horizon=horizon).for_scaling(1.4)  # type: ignore[arg-type]
    bias = np.linspace(0.7, 1.9, 300)

    before = run_pipeline(frame, config, stop_after="eigenfactor").matrix
    after = run_pipeline(frame, config, regime_bias=bias).matrix

    np.testing.assert_allclose(
        correlation_from_covariance(after), correlation_from_covariance(before), rtol=1e-13
    )
    # And the level DID move, so the invariance above is not vacuous.
    assert not np.allclose(after, before)


@pytest.mark.parametrize("horizon", HORIZONS)
def test_the_stage_is_the_recorded_multiplier_times_the_previous_matrix(horizon: str) -> None:
    """``F <- lambda_F^2 F``, exactly, with the multiplier the build reports."""
    frame = panel(specification(6))
    config = RiskConfig.load(horizon=horizon).for_scaling(1.0)  # type: ignore[arg-type]
    bias = np.linspace(0.7, 1.9, 300)

    before = run_pipeline(frame, config, stop_after="eigenfactor").matrix
    build = run_pipeline(frame, config, regime_bias=bias)
    assert build.regime is not None
    np.testing.assert_allclose(build.matrix, build.regime.lambda_squared * before, rtol=1e-14)
    assert build.complete
    assert build.stages[-1].name == "volatility_regime"


def test_psd_survives_the_stage_and_the_assertion_still_runs() -> None:
    """Invariant 4. A positive scalar cannot break PSD -- which is why it is asserted.

    A pipeline whose assertions sit only where failure is expected cannot tell
    you the stages before it were fine.
    """
    frame = panel(specification(6))
    config = RiskConfig.load(horizon="short").for_scaling(1.0)
    build = run_pipeline(frame, config, regime_bias=np.linspace(0.9, 2.5, 200))
    final = build.stages[-1]
    assert final.name == "volatility_regime"
    assert final.check is not None
    spectrum = np.linalg.eigvalsh(final.matrix)
    assert spectrum[0] >= eigenvalue_floor(
        len(spectrum), spectrum[-1], numerics=load().model.numerics
    )


# ---------------------------------------------------------------------------
# The lead-lag. This is the first thing in the project to consume a forecast.
# ---------------------------------------------------------------------------


def test_the_forecast_for_date_t_does_not_see_date_t() -> None:
    """A LEAK TEST, not a construction check.

    W4-P2 will require the same of the bias statistic, and the VRA is where the
    requirement first bites: a contemporaneous forecast leaking in would push
    ``B_t^F`` toward one, shrink ``lambda_F^2``, and silently corrupt every
    statistic later calibrated against it.

    So the return on one date is multiplied by 50 -- an unmissable shock -- and
    the forecast FOR that date must not move at all, while the forecast for the
    NEXT date must move a lot. Asserting only the first half would pass for a
    function that ignored the panel entirely.
    """
    rng = np.random.default_rng(load().model.seed)
    frame = _frame(rng.standard_normal((80, 3)))
    config = RiskConfig.load(horizon="short").for_scaling(1.0)

    shocked = frame.copy()
    target = 60
    shocked.iloc[target] = shocked.iloc[target] * 50.0

    # 6, not 3: Newey-West at 5 lags needs more than 5 observations, so the
    # arithmetic minimum for the whole pre-VRA pipeline is `lags + 1` and not `K`.
    base = forecast_history(frame, config, minimum_observations=6, stop_after="eigenfactor")
    after = forecast_history(shocked, config, minimum_observations=6, stop_after="eigenfactor")

    row = int(np.flatnonzero(base.dates == frame.index[target])[0])
    np.testing.assert_allclose(after.deviations[row], base.deviations[row], rtol=1e-14)
    assert np.max(after.deviations[row + 1] / base.deviations[row + 1]) > 1.5


def test_the_history_reports_the_window_each_forecast_saw() -> None:
    """Position, count and ``K/T_eff`` line up, and the ratio is CARRIED not filtered.

    SPEC.md 5.1.2 leaves the burn-in with the caller and no bound on ``K/T_eff``
    is published, so an unhealthy ratio is labelled rather than suppressed. The
    early rows here are unhealthy on purpose.
    """
    rng = np.random.default_rng(load().model.seed)
    frame = _frame(rng.standard_normal((60, 4)))
    config = RiskConfig.load(horizon="short").for_scaling(1.0)
    history = forecast_history(frame, config, minimum_observations=6, stop_after="eigenfactor")

    assert history.dates[0] == frame.index[6]
    assert history.dates[-1] == frame.index[-1]
    np.testing.assert_array_equal(history.observations, np.arange(6, 60))
    # Fewer observations means a smaller Kish window means a worse ratio, and it
    # is monotone because the window only grows.
    assert np.all(np.diff(history.k_over_realised_t_eff) < 0.0)
    assert history.k_over_realised_t_eff[0] > history.k_over_realised_t_eff[-1]
    assert history.bias.shape == (54,)


def test_the_control_history_stops_before_the_eigenfactor_stage() -> None:
    """The eigenfactor-off half of the overlap measurement, and it really differs.

    SPEC.md 5.3.2 puts the eigenfactor adjustment in correlation space, where it
    inflates the diagonal deliberately. So the two histories differ, which is what
    makes the ``lambda_F^2`` on/off comparison a measurement rather than zero by
    construction.
    """
    rng = np.random.default_rng(load().model.seed)
    frame = _frame(rng.standard_normal((60, 4)))
    config = RiskConfig.load(horizon="short").for_scaling(1.4)

    on = forecast_history(frame, config, minimum_observations=20, stop_after="eigenfactor")
    off = forecast_history(frame, config, minimum_observations=20, stop_after="psd_repair")

    assert on.stages[-1] == "eigenfactor"
    assert off.stages[-1] == "psd_repair"
    assert on.eigenfactor_variant == 1.4
    assert off.eigenfactor_variant is None

    # TOTAL variance is higher with the stage on, at every date. That is the
    # trace claim ``tests/test_eigenfactor.py`` already pins, restated here in
    # the units this stage consumes: trace(F) is the sum of the squared forecast
    # deviations, so a stage that raises the trace raises this sum.
    assert np.all(np.sum(on.deviations**2, axis=1) > np.sum(off.deviations**2, axis=1))

    # PER FACTOR it is NOT monotone and the test says so rather than asserting a
    # direction that does not hold. The adjustment inflates small eigenvalues and
    # deflates large ones, so a factor loading mostly on the dominant eigenfactor
    # can come out lower even while the trace rises. What matters for the overlap
    # measurement is only that the two histories differ at all -- if they did not,
    # the on/off comparison would return zero by construction rather than by
    # finding.
    assert not np.allclose(on.bias, off.bias)


def test_a_history_that_would_need_itself_is_refused() -> None:
    rng = np.random.default_rng(0)
    frame = _frame(rng.standard_normal((40, 3)))
    config = RiskConfig.load(horizon="short").for_scaling(1.0)
    with pytest.raises(RegimeError, match=r"would run SPEC\.md 5\.4's stage"):
        forecast_history(frame, config, minimum_observations=10, stop_after="volatility_regime")


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_the_stage_refuses_to_run_without_a_forecast_history() -> None:
    """No default. An un-adjusted matrix is PSD, symmetric and simply too small."""
    frame = panel(specification(6))
    config = RiskConfig.load(horizon="short").for_scaling(1.0)
    with pytest.raises(CovarianceError, match="no regime_bias"):
        run_pipeline(frame, config)


def test_a_non_positive_forecast_volatility_is_refused() -> None:
    """Refused, not floored. There is no published floor (CLAUDE.md invariant 9)."""
    with pytest.raises(RegimeError, match="zero or negative"):
        cross_sectional_bias(np.array([[0.01, 0.01]]), np.array([[0.01, 0.0]]))


def test_mismatched_shapes_are_refused() -> None:
    with pytest.raises(RegimeError, match="same T x K shape"):
        cross_sectional_bias(np.ones((5, 3)), np.ones((5, 2)))


def test_an_empty_history_is_refused() -> None:
    with pytest.raises(RegimeError, match="empty forecast history"):
        regime_multiplier(np.array([]), halflife=42)


def test_a_minimum_below_k_is_refused() -> None:
    rng = np.random.default_rng(0)
    frame = _frame(rng.standard_normal((40, 5)))
    config = RiskConfig.load(horizon="short").for_scaling(1.0)
    with pytest.raises(RegimeError, match="below K"):
        forecast_history(frame, config, minimum_observations=4, stop_after="eigenfactor")


# ---------------------------------------------------------------------------
# The cache, and its staleness guard
# ---------------------------------------------------------------------------

_CACHE = Path(__file__).resolve().parents[1] / "reports" / "vra_forecast_history.csv"


def _provenance() -> dict[str, object]:
    header: list[str] = []
    with _CACHE.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            header.append(line[1:].strip())
    parsed: dict[str, object] = json.loads("\n".join(header))
    return parsed


def test_the_cached_forecast_history_still_describes_the_current_config() -> None:
    """THE STALENESS GUARD. W3-P4.

    ``reports/vra_forecast_history.csv`` takes about ninety minutes to build --
    the eigenfactor Monte Carlo runs once per date -- so it is generated out of
    band and committed, and ``reports/volatility_regime.md`` and ``make model``
    read it. A committed artifact that later work reads is exactly the staleness
    failure mode already flagged for ``reports/stage_k_dependence.md``, and it is
    far cheaper to close here than to discover in week 8.

    The digest covers the fields that DETERMINE the artifact -- both horizons'
    RiskConfig, and the panel -- and not the whole of ``config/model.yaml``. A
    full-file hash would fire on a cost-parameter edit that cannot change a
    single number in the cache, and a guard that cries wolf is a guard that gets
    deleted. The file's own SHA-256 is recorded beside it as provenance.
    """
    from mafrm.factors.vra_report import risk_config_digest

    assert _CACHE.exists(), (
        f"the cached forecast history is not committed: {_CACHE}. Rebuild it with "
        "`python -m mafrm.factors.vra_report --rebuild` (about ninety minutes)."
    )
    provenance = _provenance()
    assert provenance["risk_config_digest"] == risk_config_digest(), (
        "reports/vra_forecast_history.csv was built against a DIFFERENT risk config than the "
        "one in config/model.yaml today. Every lambda_F^2 in reports/volatility_regime.md and "
        "every number experiments.md records from it describes the old one. Rebuild with "
        "`python -m mafrm.factors.vra_report --rebuild`."
    )
    for key in ("generated_utc", "git_commit", "model_yaml_sha256", "panel_digest", "rows"):
        assert key in provenance, f"the cache provenance header is missing {key!r}"


# ---------------------------------------------------------------------------
# The rolling multiplier the chart is drawn from
# ---------------------------------------------------------------------------


def test_the_rolling_multiplier_agrees_with_the_estimator_at_every_prefix() -> None:
    """One estimator, two evaluation paths, and the chart uses the fast one.

    A chart drawn from a second implementation of an estimator is a chart of the
    second implementation. The recursion is checked against the weighted sum at
    every prefix rather than at the end, because an error in the decay term shows
    up early and cancels late.
    """
    rng = np.random.default_rng(load().model.seed)
    bias = np.abs(rng.standard_normal(120)) + 0.3
    for halflife in (42, 168):
        rolling = regime_module.rolling_multiplier(bias, halflife=halflife)
        assert rolling.shape == bias.shape
        for cut in (1, 2, 5, 41, 90, 120):
            expected = regime_multiplier(bias[:cut], halflife=halflife).lambda_squared
            assert rolling[cut - 1] == pytest.approx(expected, rel=1e-12)


def test_the_rolling_multiplier_starts_at_the_first_observation() -> None:
    """With one date the weights are trivially ``[1]``, so lambda^2 is that date's B^2."""
    assert regime_module.rolling_multiplier(np.array([1.5]), halflife=42)[0] == pytest.approx(2.25)


def test_the_cache_round_trips_through_its_own_format(tmp_path: Path) -> None:
    """The provenance header must not eat the data, and vice versa.

    The header is ``#``-prefixed JSON in front of a CSV, which is a format that
    works until a value contains a newline in the wrong place. Round-tripping it
    is a cheap check that ``make model`` and ``make report`` are reading the
    numbers the rebuild wrote, on a temporary path so the committed cache is
    never touched.
    """
    from mafrm.factors import vra_report

    settings = load()
    columns = vra_report.columns(settings)
    assert [column.name for column in columns] == [
        "bias_short_a1.0",
        "bias_short_a1.4",
        "bias_short_none",
        "bias_long_a1.0",
        "bias_long_a1.4",
        "bias_long_none",
    ]
    # The control stops one stage earlier, and that is what makes it a control.
    assert {column.stop_after for column in columns if column.scaling is None} == {"psd_repair"}
    assert {column.stop_after for column in columns if column.scaling is not None} == {
        "eigenfactor"
    }

    index = pd.date_range("2010-01-04", periods=5, freq="B", name="date")
    values = np.linspace(0.8, 1.2, 5)
    frame = pd.DataFrame(
        {
            **{
                f"k_over_realised_t_eff_{horizon}": np.linspace(0.9, 0.1, 5) for horizon in HORIZONS
            },
            **{column.name: values for column in columns},
        },
        index=index,
    )
    provenance = {
        "risk_config_digest": vra_report.risk_config_digest(settings),
        "generated_utc": "2026-08-31T00:00:00+00:00",
        "git_commit": "0" * 40,
        "model_yaml_sha256": "0" * 64,
        "panel_digest": "0" * 64,
        "panel_start": "2010-01-04",
        "panel_end": "2010-01-08",
        "holdout_start": "2025-01-01",
        "minimum_observations": 7,
        "rows": 5,
    }
    path = tmp_path / "vra_forecast_history.csv"
    vra_report.write_cache(vra_report.Cache(provenance=provenance, frame=frame), path)
    back = vra_report.read_cache(path)

    np.testing.assert_allclose(back.frame.to_numpy(), frame.to_numpy(), rtol=1e-12)
    assert list(back.frame.columns) == list(frame.columns)
    assert back.provenance["rows"] == 5
    np.testing.assert_allclose(back.bias(columns[0]), values, rtol=1e-12)


def test_a_stale_cache_is_refused_on_read_and_not_only_in_a_test(tmp_path: Path) -> None:
    """The guard is in ``read_cache`` too, because the test guards the COMMIT.

    ``make model`` and ``make report`` would otherwise publish numbers from a
    superseded config with nothing to indicate anything was wrong.
    """
    from mafrm.factors import vra_report

    columns = vra_report.columns()
    index = pd.date_range("2010-01-04", periods=3, freq="B", name="date")
    frame = pd.DataFrame({column.name: np.ones(3) for column in columns}, index=index)
    path = tmp_path / "stale.csv"
    vra_report.write_cache(
        vra_report.Cache(provenance={"risk_config_digest": "not-the-current-one"}, frame=frame),
        path,
    )
    with pytest.raises(ValueError, match="is STALE"):
        vra_report.read_cache(path)


@pytest.mark.dataset
def test_an_interrupted_variant_resumes_to_a_BIT_IDENTICAL_result(tmp_path: Path) -> None:
    """Checkpointing must not change the answer, and "almost" is not good enough.

    Two things had to be right for this to hold and neither was by default: the
    checkpoint is written at 17 significant digits (12 left a 5e-12 difference)
    and read with ``float_precision="round_trip"`` (pandas' fast C parser is not
    round-trip exact, and left one ulp). Both are cheap; the alternative is a
    cache whose contents depend on whether the machine was interrupted while
    building it.

    Run on the eigenfactor-OFF control, which is the cheap one -- the checkpoint
    path does not know or care which variant it is carrying.
    """
    import mafrm.risk.regime as regime
    from mafrm.factors import vra_report

    settings = load()
    column = vra_report.Column(horizon="short", scaling=None)
    monkey = pytest.MonkeyPatch()
    monkey.setattr(vra_report, "PARTIALS", tmp_path)
    try:
        unbroken = vra_report.build_variant(column, settings).copy()
        vra_report._partial_path(column).unlink()

        # Interrupt after 850 dates, which is not a multiple of the flush size,
        # so the tail that was still pending is genuinely lost and recomputed.
        seen = {"rows": 0}
        original = regime.forecast_history

        def interrupting(frame, config, **kwargs):  # type: ignore[no-untyped-def]
            hook = kwargs.pop("on_row", None)

            def guard(row: regime.ForecastRow) -> None:
                seen["rows"] += 1
                if hook is not None:
                    hook(row)
                if seen["rows"] >= 850:
                    raise KeyboardInterrupt("simulated kill")

            return original(frame, config, on_row=guard, **kwargs)

        monkey.setattr(vra_report, "forecast_history", interrupting)
        with pytest.raises(KeyboardInterrupt):
            vra_report.build_variant(column, settings)
        assert vra_report._progress_path(column).exists()

        monkey.setattr(vra_report, "forecast_history", original)
        resumed = vra_report.build_variant(column, settings)
    finally:
        monkey.undo()

    assert resumed.index.equals(unbroken.index)
    assert np.array_equal(resumed.to_numpy(), unbroken.to_numpy()), (
        "a resumed build differs from an unbroken one; the checkpoint is lossy"
    )
    assert not vra_report._progress_path(column).exists()


@pytest.mark.dataset
def test_a_checkpoint_from_a_different_config_is_discarded(tmp_path: Path) -> None:
    """Resuming across a config change would splice two models into one series."""
    from mafrm.factors import vra_report

    column = vra_report.Column(horizon="short", scaling=None)
    monkey = pytest.MonkeyPatch()
    monkey.setattr(vra_report, "PARTIALS", tmp_path)
    try:
        progress = vra_report._progress_path(column)
        progress.parent.mkdir(parents=True, exist_ok=True)
        progress.write_text(
            '# {\n#   "panel_digest": "wrong",\n#   "risk_config_digest": "wrong"\n# }\n'
            "date,bias,observations,k_over_realised_t_eff\n"
            "2008-04-23,1.0,7,0.5\n",
            encoding="utf-8",
        )
        built = vra_report.build_variant(column, load())
    finally:
        monkey.undo()
    # Rebuilt from the arithmetic minimum -- position 7 in the panel -- rather
    # than resumed from the stale run's last recorded date.
    assert len(built) > 4000
    assert built.index[0] == pd.Timestamp("2008-04-23")
