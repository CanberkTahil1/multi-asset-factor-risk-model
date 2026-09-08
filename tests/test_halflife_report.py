"""SPEC.md 5.1's half-life sweep. W4-P2b.

The tests that matter here are the ones guarding the sweep's **classification**,
not its arithmetic. SPEC.md 5.1.3 rules the sweep ``data-diagnostic`` on the
ground that the shipped half-lives were fixed before it ran and no result may
change them, and that argument collapses the moment the shipped values stop being
points on the curve. :func:`test_the_shipped_half_lives_are_points_on_the_grid`
is that argument in executable form.
"""

from __future__ import annotations

import dataclasses
import itertools
import math

import numpy as np
import pandas as pd
import pytest

from mafrm import config, history
from mafrm.config import ConfigError, load
from mafrm.factors import bias_report, halflife_report
from mafrm.risk.config import HORIZONS, RiskConfig


def _settings() -> config.Config:
    return load()


# ---------------------------------------------------------------------------
# The grid, and the claim config/model.yaml makes about it
# ---------------------------------------------------------------------------


def test_the_grid_is_nine_integer_trading_months_spanning_spec_5_1s_endpoints() -> None:
    """HAND-COMPUTED. The grid's justification is a principle, so it is checkable.

    ``config/model.yaml`` justifies the nine points as *integer trading months,
    log-spaced, spanning SPEC.md 5.1's own endpoints 21 and 504, hitting both
    shipped values exactly*. An earlier draft justified them individually as
    constants the project already carries; that argument covers eight of the nine
    and fails on 336, and SPEC.md 5.1.3 records it as refused for exactly that
    reason. This asserts the rule that replaced it.
    """
    settings = _settings()
    grid = settings.model.validation.halflife_sensitivity.grid
    month = settings.model.data.trading_days_per_month

    assert grid == (21, 42, 63, 84, 126, 168, 252, 336, 504)
    assert [value // month for value in grid] == [1, 2, 3, 4, 6, 8, 12, 16, 24]
    assert all(value % month == 0 for value in grid)
    assert (grid[0], grid[-1]) == (21, 504)  # SPEC.md 5.1's own endpoints
    assert list(grid) == sorted(set(grid))

    ratios = [b / a for a, b in itertools.pairwise(grid)]
    assert [round(r, 3) for r in ratios] == [2.0, 1.5, 1.333, 1.5, 1.333, 1.5, 1.333, 1.5]
    assert (grid[-1] / grid[0]) ** (1 / 8) == pytest.approx(1.4877, abs=5e-5)


def test_the_shipped_half_lives_are_points_on_the_grid() -> None:
    """THE CLASSIFICATION DEPENDS ON THIS, which is why it is asserted and not assumed."""
    settings = _settings()
    grid = settings.model.validation.halflife_sensitivity.grid
    for horizon in HORIZONS:
        assert RiskConfig.load(horizon=horizon, config=settings).volatility_halflife in grid
    halflife_report.assert_shipped_on_grid(settings)


def test_a_grid_that_misses_a_shipped_value_is_refused() -> None:
    settings = _settings()
    validation_config = settings.model.validation
    narrowed = dataclasses.replace(
        validation_config,
        halflife_sensitivity=dataclasses.replace(
            validation_config.halflife_sensitivity, grid=(21, 42, 63, 126, 504)
        ),
    )
    broken = dataclasses.replace(
        settings, model=dataclasses.replace(settings.model, validation=narrowed)
    )
    with pytest.raises(ValueError, match="off the grid"):
        halflife_report.assert_shipped_on_grid(broken)


def test_a_grid_point_that_is_not_an_integer_trading_month_is_refused() -> None:
    settings = _settings()
    validation_config = settings.model.validation
    ragged = dataclasses.replace(
        validation_config,
        halflife_sensitivity=dataclasses.replace(
            validation_config.halflife_sensitivity, grid=(21, 84, 100, 252, 504)
        ),
    )
    broken = dataclasses.replace(
        settings, model=dataclasses.replace(settings.model, validation=ragged)
    )
    with pytest.raises(ValueError, match="integer trading months"):
        halflife_report.assert_shipped_on_grid(broken)


def test_the_parser_refuses_a_grid_that_is_not_strictly_increasing() -> None:
    """A repeat would build the same history twice under two names; a reversal
    would draw a curve that doubles back. Neither is visible on the chart."""
    from mafrm.config import _parse_validation_battery

    node = {
        "standardized_return_clip": 4.0,
        "rolling_window_months": 12,
        "normal_band_z": 1.96,
        "chi_square_level": 0.95,
        "random_portfolios": 100,
        "halflife_sensitivity": {"grid": [21, 84, 84, 252]},
    }
    with pytest.raises(ConfigError, match="strictly increasing"):
        _parse_validation_battery({"validation": node})


# ---------------------------------------------------------------------------
# Shepard -- the reason this sweep is a K/T experiment
# ---------------------------------------------------------------------------


def test_effective_sample_size_matches_the_configs_own_reference_table() -> None:
    """``T_eff = 2*tau/ln 2``. SPEC.md 6.4 Eq. 33, CLAUDE.md's parameter table."""
    reference = _settings().model.numerics.effective_sample_size.reference
    for halflife, expected in reference.items():
        assert round(halflife_report.effective_sample_size(halflife)) == expected
    # 2 * 84 / ln 2 = 168 / 0.6931472 = 242.3728. The config's reference table rounds
    # it to 242, which is the figure CLAUDE.md's parameter table quotes.
    assert halflife_report.effective_sample_size(84) == pytest.approx(242.3728, abs=1e-4)


def test_shepard_reproduces_spec_6_4s_published_table_at_the_shipped_half_life() -> None:
    """HAND-COMPUTED, and it is SPEC.md 6.4's own worked example.

    ``T_eff = 2*84/ln 2 = 242.3728``; ``K/T_eff = 6/242.3728 = 0.024755``;
    ``(1 - 0.024755)^-2 - 1 = 0.05141``. SPEC.md 6.4's table says **5.1%**.
    """
    t_eff = 2 * 84 / math.log(2)
    assert halflife_report.shepard_understatement(6, 84) == pytest.approx((1 - 6 / t_eff) ** -2 - 1)
    assert halflife_report.shepard_understatement(6, 84) == pytest.approx(0.0514, abs=5e-4)
    assert halflife_report.shepard_understatement(6, 252) == pytest.approx(0.0170, abs=5e-4)


def test_the_grid_moves_k_over_t_eff_by_twenty_four_fold() -> None:
    """The claim SPEC.md 5.1.3 makes, as arithmetic: ``K/T_eff`` is linear in ``1/tau``."""
    grid = _settings().model.validation.halflife_sensitivity.grid
    ratios = [6 / halflife_report.effective_sample_size(tau) for tau in grid]
    assert ratios[0] / ratios[-1] == pytest.approx(grid[-1] / grid[0])
    assert ratios[0] / ratios[-1] == pytest.approx(24.0)
    assert ratios[0] == pytest.approx(0.0990, abs=5e-4)
    assert ratios[-1] == pytest.approx(0.0041, abs=5e-4)
    assert halflife_report.shepard_understatement(6, 21) == pytest.approx(0.232, abs=1e-3)


# ---------------------------------------------------------------------------
# Nine histories and not eighteen
# ---------------------------------------------------------------------------


def test_the_cached_history_depends_on_the_half_life_alone() -> None:
    """SPEC.md 5.1.3's nine-not-eighteen argument, as an assertion.

    Also the guard against a later session adding a ``RiskConfig`` field that
    differs between horizons and silently acquiring a horizon-dependence this
    module would render straight past.
    """
    settings = _settings()
    halflife_report.assert_horizon_independent(settings)

    short, long_ = (
        dataclasses.asdict(RiskConfig.load(horizon=horizon, config=settings))
        for horizon in HORIZONS
    )
    differing = {key for key in short if short[key] != long_[key]}
    assert differing.intersection(halflife_report.COVARIANCE_FIELDS) == {"volatility_halflife"}


def test_a_horizon_dependent_covariance_field_is_refused() -> None:
    """If the horizons ever disagree on a field the pipeline reads, nine histories
    stop being enough and the module must say so rather than render."""
    settings = _settings()
    covariance = settings.model.covariance
    diverged = dataclasses.replace(
        settings,
        model=dataclasses.replace(
            settings.model,
            covariance=dataclasses.replace(
                covariance,
                factor_correlation_halflife=dataclasses.replace(
                    covariance.factor_correlation_halflife, long=252
                ),
            ),
        ),
    )
    with pytest.raises(ValueError, match="correlation_halflife"):
        halflife_report.assert_horizon_independent(diverged)


def test_risk_for_replaces_only_the_swept_dial() -> None:
    settings = _settings()
    base = RiskConfig.load(horizon="short", config=settings)
    swept = halflife_report.risk_for(halflife_report.GridPoint(21), settings)
    assert swept.volatility_halflife == 21
    assert swept.correlation_halflife == base.correlation_halflife == 504
    changed = {key for key in dataclasses.asdict(base) if getattr(base, key) != getattr(swept, key)}
    assert changed == {"volatility_halflife"}


# ---------------------------------------------------------------------------
# The three-part cache key -- the W4-P2b ruling's third condition
# ---------------------------------------------------------------------------


def test_the_partial_key_moves_when_any_of_its_three_parts_moves() -> None:
    """``history.risk_config_digest`` alone cannot separate grid points.

    It hashes both horizons' shipped ``RiskConfig``, so every point of this sweep
    presents the same one. A guard that returns the same key for two different
    models protects something other than what it claims to -- the defect
    ``mafrm.history`` was extracted to fix -- so the half-life is carried beside
    it rather than folded into it.
    """
    settings = _settings()
    frame = pd.DataFrame(
        np.arange(12, dtype=float).reshape(6, 2),
        index=pd.date_range("2020-01-01", periods=6, freq="B", name="date"),
        columns=["a", "b"],
    )
    data = _stub_inputs(frame)

    first = halflife_report.partial_digests(halflife_report.GridPoint(21), data, settings)
    same = halflife_report.partial_digests(halflife_report.GridPoint(21), data, settings)
    assert first == same
    assert set(first) == {"risk_config_digest", "panel_digest", "volatility_halflife"}

    # (1) the half-life
    other_point = halflife_report.partial_digests(halflife_report.GridPoint(504), data, settings)
    assert other_point["volatility_halflife"] != first["volatility_halflife"]
    assert other_point["risk_config_digest"] == first["risk_config_digest"]

    # (2) the panel
    moved = frame.copy()
    moved.iloc[0, 0] += 1.0
    other_panel = halflife_report.partial_digests(
        halflife_report.GridPoint(21), _stub_inputs(moved), settings
    )
    assert other_panel["panel_digest"] != first["panel_digest"]

    # (3) the shipped configuration
    numerics = settings.model.numerics
    retuned = dataclasses.replace(
        settings,
        model=dataclasses.replace(
            settings.model,
            numerics=dataclasses.replace(numerics, psd_reconstruction_roundings=3),
        ),
    )
    assert history.risk_config_digest(retuned) != first["risk_config_digest"]


def _stub_inputs(frame: pd.DataFrame) -> bias_report.Inputs:
    """Just enough of :class:`bias_report.Inputs` for the digest, which reads ``factors``."""
    return bias_report.Inputs(
        panel=None,  # type: ignore[arg-type]
        returns=frame,
        factors=frame,
        exposures=np.zeros((len(frame), frame.shape[1], frame.shape[1])),
        buckets=pd.Series(dtype=object),
        excluded=(),
    )


def test_grid_point_names_are_stable_and_unique() -> None:
    settings = _settings()
    points = halflife_report.grid_points(settings)
    assert [point.name for point in points] == [
        f"hl{value}" for value in settings.model.validation.halflife_sensitivity.grid
    ]
    assert len({point.name for point in points}) == len(points)


# ---------------------------------------------------------------------------
# The common scored window
# ---------------------------------------------------------------------------


def test_common_dates_is_the_intersection_and_refuses_an_empty_one() -> None:
    def window(dates: pd.DatetimeIndex) -> bias_report.ScoredWindow:
        return bias_report.ScoredWindow(
            mask=np.ones(len(dates), dtype=bool),
            dates=dates,
            dropped_by_repair=0,
            dropped_by_rank=0,
            dropped_by_regime=0,
            assets=13,
        )

    early = pd.date_range("2020-01-01", periods=10, freq="B")
    late = pd.date_range("2020-01-06", periods=10, freq="B")
    shared = halflife_report.common_dates({"a": window(early), "b": window(late)})
    assert list(shared) == list(early.intersection(late))
    assert shared[0] == late[0]

    with pytest.raises(ValueError, match="share no scored dates"):
        halflife_report.common_dates(
            {"a": window(early), "b": window(pd.date_range("2030-01-01", periods=3, freq="B"))}
        )


def test_restricting_a_window_records_what_it_cost_and_refuses_a_gap() -> None:
    """SPEC.md 5.1.3's intersected window. A hole inside it is a defect, not an artefact."""
    dates = pd.date_range("2020-01-01", periods=8, freq="B")
    window = bias_report.ScoredWindow(
        mask=np.ones(8, dtype=bool),
        dates=dates,
        dropped_by_repair=0,
        dropped_by_rank=0,
        dropped_by_regime=0,
        assets=13,
    )
    narrowed = bias_report.restrict_window(window, dates, dates[3:])
    assert narrowed.observations == 5
    assert narrowed.dropped_by_common == 3
    assert "outside the window common to every grid point" in narrowed.render()

    holed = dates.delete(4)
    with pytest.raises(ValueError, match="interior gap"):
        bias_report.restrict_window(window, dates, holed)
