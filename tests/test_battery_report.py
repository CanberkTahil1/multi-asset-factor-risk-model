"""The W4-P3 report plumbing that can be checked without the data cache.

The rendered reports need Model A's residual panel (``data/raw``, gitignored),
so what is pinned here is everything around them: that the config carries the
five rulings and resolves its pointers to the keys they name, that a dangling
pointer is refused, that the registered-comparison logic adjudicates the way
the registration says, and that the report's verdict function is exactly the
"inside the band widened by two standard errors" rule.
"""

from __future__ import annotations

import numpy as np
import pytest

from mafrm import config
from mafrm.config import ConfigError
from mafrm.factors import battery_report, second_order_report
from mafrm.risk import second_order
from mafrm.risk.config import RiskConfig


@pytest.fixture(scope="module")
def settings() -> config.Config:
    return config.load()


def test_the_battery_rulings_are_in_the_config(settings: config.Config) -> None:
    tests = settings.model.validation.battery
    assert tests.var_levels == (0.95, 0.99)
    assert tests.ljung_box_lags_months == (1, 3, 12)
    assert tests.basel.level == 0.99
    assert (
        tests.basel.window_days,
        tests.basel.green_max_breaches,
        tests.basel.yellow_max_breaches,
    ) == (250, 4, 9)


def test_pointers_resolve_to_the_keys_they_name(settings: config.Config) -> None:
    model = settings.model
    tests = model.validation.battery
    assert (
        config.resolve(model, tests.monte_carlo_trials_from) == model.eigenfactor.monte_carlo_trials
    )
    assert config.resolve(model, tests.t_statistic_threshold_from) == pytest.approx(2.0)
    assert (
        config.resolve(model, tests.rank_correlation_block_days_from)
        == model.data.trading_days_per_month
    )
    assert config.resolve(model, model.validation.second_order.monte_carlo_trials_from) == 2000


def test_battery_inputs_derive_everything_from_config(settings: config.Config) -> None:
    inputs = battery_report.BatteryInputs.from_config(settings, "short")
    days = settings.model.data.trading_days_per_month
    assert inputs.ljung_box_lags == (days, 3 * days, 12 * days)
    assert inputs.trials == settings.model.eigenfactor.monte_carlo_trials
    assert inputs.newey_west_lags == settings.model.covariance.volatility_newey_west_lags.short
    assert inputs.block == days
    assert inputs.t_window == settings.model.validation.rolling_window_months * days
    assert inputs.significance == pytest.approx(1.0 - settings.model.validation.chi_square_level)
    assert inputs.seed == settings.model.seed


def test_a_dangling_pointer_is_refused() -> None:
    config._check_pointer({"a": {"b": 1}}, "a.b", "test")
    with pytest.raises(ConfigError, match="not in config"):
        config._check_pointer({"a": {"b": 1}}, "a.c", "test")
    with pytest.raises(ConfigError, match="not in config"):
        config._check_pointer({"a": 1}, "a.b", "test")


def test_resolve_refuses_an_unknown_path(settings: config.Config) -> None:
    with pytest.raises(ConfigError):
        config.resolve(settings.model, "eigenfactor.no_such_key")


def test_within_is_the_band_widened_by_two_standard_errors() -> None:
    within = second_order_report._within
    assert within(0.50, 0.01, 0.40, 0.60) == "HOLDS"
    assert within(0.61, 0.01, 0.40, 0.60) == "HOLDS"  # inside by the 2 s.e. allowance
    assert within(0.63, 0.01, 0.40, 0.60) == "REFUTED"
    assert within(0.37, 0.01, 0.40, 0.60) == "REFUTED"


def test_compare_adjudicates_six_rows_in_registration_order(settings: config.Config) -> None:
    truth = np.diag([1.0, 2.0, 3.0])
    kwargs = {
        "volatility_halflife": 20.0,
        "correlation_halflife": 40.0,
        "observations": 120,
        "trials": 60,
        "seed": 1,
    }
    panel = second_order.simulate(truth, **kwargs)  # type: ignore[arg-type]
    result = second_order_report.HorizonDecomposition(
        horizon="short",
        risk=RiskConfig.load(horizon="short", config=settings),
        panel=panel,
        identity_control=panel,
        control_closed_form=panel.volatility_leg_closed_form,
    )
    comparisons = second_order_report.compare(result, settings, rows=(175, 176, 177, 178, 179, 180))
    assert [item.row for item in comparisons] == [175, 176, 177, 178, 179, 180]
    assert all(item.verdict in {"HOLDS", "REFUTED"} for item in comparisons)
    # The disjointness row is the interaction share against the registered 10%.
    share = settings.model.validation.second_order.interaction_share_falsifier
    assert comparisons[4].upper == share
    assert comparisons[4].measured == pytest.approx(abs(panel.interaction_share))
    # The Jensen row is a point comparison: lower == upper.
    assert comparisons[5].lower == comparisons[5].upper
