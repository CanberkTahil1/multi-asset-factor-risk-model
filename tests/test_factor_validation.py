"""W2-P3: external validation of Model A against published series. SPEC.md 6.5.

The properties that carry this module, each with a test that can actually fail:

1. **The Newey-West lag is a rule, not a number.** Newey & West (1994)'s
   deterministic plug-in bandwidth is computed on paper below at four sample
   sizes and the code has to agree exactly.
2. **Daily returns compound and yield changes sum.** The monthly aggregation is
   tested against a hand-built series whose monthly answer is known in closed
   form, and separately as an exact round trip.
3. **The alignment cannot silently lose a month.** AQR stamps the last business
   day and this project stamps the last traded day; the join has to notice a
   month it could not match rather than shrinking quietly, and it has to refuse
   an observation on or after ``holdout_start``.
4. **The mapping is the deliverable, so the parser defends it.** A factor cannot
   be both mapped and listed as having no analogue, a sign prediction cannot ride
   on a comparand that is not gated on its sign, and the equity factor cannot be
   validated against the series it is taken from.
5. **The placebo can fail.** A gate that cannot fail is worse than no gate, so
   the failing case is constructed rather than assumed unreachable.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from mafrm.config import Comparand, ConfigError, load
from mafrm.factors import macro, validation
from mafrm.factors.validation import PlaceboRow

# ---------------------------------------------------------------------------
# 1. The Newey-West plug-in rule
# ---------------------------------------------------------------------------
#
# lag = floor(4 * (T/100)^(2/9)), computed on paper:
#
#   T = 100  ->  4 * 1^(2/9)            = 4.000  -> 4
#   T = 213  ->  4 * 2.13^(2/9)
#                ln 2.13 = 0.7561, x 2/9 = 0.16803, exp = 1.18297
#                4 x 1.18297           = 4.7319  -> 4
#   T = 4146 ->  4 * 41.46^(2/9)
#                ln 41.46 = 3.72479, x 2/9 = 0.82773, exp = 2.28829
#                4 x 2.28829           = 9.1532  -> 9
#   T = 36   ->  4 * 0.36^(2/9)
#                ln 0.36 = -1.02165, x 2/9 = -0.22703, exp = 0.79789
#                4 x 0.79789           = 3.1916  -> 3


@pytest.mark.parametrize(("observations", "expected"), [(100, 4), (213, 4), (4146, 9), (36, 3)])
def test_newey_west_plug_in_matches_the_hand_computed_rule(
    observations: int, expected: int
) -> None:
    assert validation.newey_west_plug_in_lags(observations) == expected


def test_the_monthly_regressions_and_the_daily_pipeline_get_different_lags() -> None:
    """The reason the rule is a rule. Reusing the daily count would be wrong."""
    monthly = validation.newey_west_plug_in_lags(213)
    daily = validation.newey_west_plug_in_lags(4146)
    assert monthly != daily
    # And it is not the covariance pipeline's daily parameter either.
    assert monthly != load().model.covariance.volatility_newey_west_lags.short


def test_a_non_positive_sample_raises_rather_than_returning_a_lag() -> None:
    with pytest.raises(validation.ValidationError, match="must be positive"):
        validation.newey_west_plug_in_lags(0)


# ---------------------------------------------------------------------------
# 2. Aggregation
# ---------------------------------------------------------------------------


def _panel(frame: pd.DataFrame, units: dict[str, str]) -> macro.FactorPanel:
    """A minimal FactorPanel carrying a hand-built daily frame."""
    return macro.FactorPanel(
        raw=frame,
        orthogonal=frame,
        units=units,
        pca=None,  # type: ignore[arg-type]  # unused by the aggregation
        start=pd.Timestamp(frame.index[0]),
        end=pd.Timestamp(frame.index[-1]),
    )


def test_returns_compound_and_yield_changes_sum() -> None:
    """A hand-built month with a known answer in closed form.

    Two trading days in January carrying +10% and -10%:

      * a RETURN compounds:  1.10 x 0.90 - 1 = -0.01 exactly.
      * a yield CHANGE sums:  10 + (-10)     =  0    exactly.

    Averaging the return would give 0.0, which is the error this distinction
    exists to prevent -- and it is off by a full percentage point on one month.
    """
    index = pd.DatetimeIndex(["2020-01-02", "2020-01-03"])
    frame = pd.DataFrame({"ret": [0.10, -0.10], "bps": [10.0, -10.0]}, index=index)
    monthly = validation.monthly_factor_returns(
        _panel(frame, {"ret": "decimal daily excess return", "bps": "basis points of yield"})
    )
    january = pd.Period("2020-01", freq="M")
    assert monthly.frame.loc[january, "ret"] == pytest.approx(-0.01, abs=1e-15)
    assert monthly.frame.loc[january, "bps"] == pytest.approx(0.0, abs=1e-15)
    # The average would have been 0.0. It is not what a monthly return is.
    assert monthly.frame.loc[january, "ret"] != pytest.approx(0.0, abs=1e-6)


def test_only_a_boundary_month_is_dropped() -> None:
    """An interior gap is kept; a partial first month is not.

    ``late`` opens on the 3rd, after the panel calendar's 2nd, so January is a
    partial month for it and is dropped. February is complete for both. The
    interior hole on the 4th costs ``late`` nothing, which is the behaviour that
    keeps six good months in the real ``equity`` series.
    """
    index = pd.DatetimeIndex(["2020-01-02", "2020-01-03", "2020-02-03", "2020-02-04"])
    frame = pd.DataFrame(
        {"early": [0.01, 0.01, 0.01, 0.01], "late": [np.nan, 0.02, 0.02, 0.02]},
        index=index,
    )
    units = {"early": "decimal daily excess return", "late": "decimal daily excess return"}
    monthly = validation.monthly_factor_returns(_panel(frame, units))
    assert monthly.dropped["early"] == ()
    assert monthly.dropped["late"] == (pd.Period("2020-01", freq="M"),)
    assert monthly.observations == {"early": 2, "late": 1}


def test_the_month_end_stamp_is_a_traded_date_not_a_calendar_month_end() -> None:
    """CLAUDE.md failure mode 1. The stamp is what actually closed the month."""
    index = pd.DatetimeIndex(["2020-01-02", "2020-01-30"])
    frame = pd.DataFrame({"ret": [0.01, 0.01]}, index=index)
    monthly = validation.monthly_factor_returns(
        _panel(frame, {"ret": "decimal daily excess return"})
    )
    stamp = monthly.observed_month_end.loc[pd.Period("2020-01", freq="M"), "ret"]
    assert pd.Timestamp(stamp) == pd.Timestamp("2020-01-30")
    assert pd.Timestamp(stamp) != pd.Timestamp("2020-01-31")


def test_the_round_trip_is_exact_on_a_hand_built_series() -> None:
    index = pd.DatetimeIndex(["2020-01-02", "2020-01-03", "2020-02-03"])
    frame = pd.DataFrame({"ret": [0.03, -0.02, 0.01], "bps": [4.0, -1.0, 2.0]}, index=index)
    panel = _panel(frame, {"ret": "decimal daily excess return", "bps": "basis points of yield"})
    residual = validation.monthly_round_trip(panel, validation.monthly_factor_returns(panel))
    assert residual["ret"] == pytest.approx(0.0, abs=1e-15)
    assert residual["bps"] == pytest.approx(0.0, abs=1e-13)


# ---------------------------------------------------------------------------
# 3. Alignment
# ---------------------------------------------------------------------------


def _monthly(values: dict[str, float]) -> pd.Series:
    return pd.Series(values.values(), index=pd.PeriodIndex(list(values), freq="M"), dtype=float)


def test_the_join_refuses_an_observation_on_or_after_the_holdout_boundary() -> None:
    """The right edge is ``holdout_start``, always -- asserted, not inferred.

    The comparand here runs past the boundary, which is the real situation: all
    three AQR files end after 2025-01-01.
    """
    factor = _monthly({"2024-11": 1.0, "2024-12": 2.0, "2025-01": 3.0})
    comparand = pd.Series(
        [1.0, 2.0, 3.0],
        index=pd.DatetimeIndex(["2024-11-29", "2024-12-31", "2025-01-31"]),
    )
    joined = validation._join(factor, comparand, boundary=date(2025, 1, 1), label="t")
    assert len(joined) == 2
    assert pd.DatetimeIndex(joined["observed"]).max() < pd.Timestamp("2025-01-01")


def test_the_join_keys_on_the_month_and_keeps_the_published_date() -> None:
    """AQR's last business day and our last traded day need not be the same day."""
    factor = _monthly({"2024-11": 1.0, "2024-12": 2.0})
    comparand = pd.Series([10.0, 20.0], index=pd.DatetimeIndex(["2024-11-29", "2024-12-30"]))
    joined = validation._join(factor, comparand, boundary=date(2025, 1, 1), label="t")
    assert list(pd.DatetimeIndex(joined["observed"]).date) == [
        date(2024, 11, 29),
        date(2024, 12, 30),
    ]
    assert list(joined["factor"]) == [1.0, 2.0]


def test_two_observations_in_one_month_raise_rather_than_collapsing() -> None:
    factor = _monthly({"2024-12": 2.0})
    comparand = pd.Series([1.0, 2.0], index=pd.DatetimeIndex(["2024-12-30", "2024-12-31"]))
    with pytest.raises(validation.ValidationError, match="two observations in one month"):
        validation._join(factor, comparand, boundary=date(2025, 1, 1), label="t")


def test_a_comparand_that_never_reaches_the_window_raises() -> None:
    factor = _monthly({"2024-12": 2.0})
    comparand = pd.Series([1.0], index=pd.DatetimeIndex(["2030-01-31"]))
    with pytest.raises(validation.ValidationError, match="no observation before"):
        validation._join(factor, comparand, boundary=date(2025, 1, 1), label="t")


def test_no_shared_month_raises_rather_than_returning_an_empty_table() -> None:
    factor = _monthly({"2024-12": 2.0})
    comparand = pd.Series([1.0], index=pd.DatetimeIndex(["2023-06-30"]))
    with pytest.raises(validation.ValidationError, match="share no month"):
        validation._join(factor, comparand, boundary=date(2025, 1, 1), label="t")


# ---------------------------------------------------------------------------
# 4. The mapping, defended by the parser
# ---------------------------------------------------------------------------


def _raw_validation() -> dict:
    import copy

    import yaml

    from mafrm.config import config_dir

    raw = yaml.safe_load((config_dir() / "model.yaml").read_text(encoding="utf-8"))
    return copy.deepcopy(raw["factors"])


def _parse(node: dict) -> None:
    from mafrm.config import _parse_validation

    _parse_validation(node, "model.factors")


def test_the_shipped_mapping_parses() -> None:
    _parse(_raw_validation())


def test_a_factor_cannot_be_both_mapped_and_unmapped() -> None:
    node = _raw_validation()
    node["validation"]["unmapped"]["equity"] = "a later session tries to have it both ways"
    with pytest.raises(ConfigError, match="both mapped and listed as having no analogue"):
        _parse(node)


def test_a_sign_prediction_cannot_ride_on_a_placebo_comparand() -> None:
    """A falsifier the gate does not apply is worse than none."""
    node = _raw_validation()
    node["validation"]["comparands"]["equity"]["expected_beta_sign"] = 1
    with pytest.raises(ConfigError, match="expected_beta_sign is set on a"):
        _parse(node)


def test_a_sign_gate_without_a_sign_raises() -> None:
    node = _raw_validation()
    del node["validation"]["comparands"]["rates_level"]["expected_beta_sign"]
    with pytest.raises(ConfigError, match="expected_beta_sign is required"):
        _parse(node)


def test_the_sign_is_a_convention_not_a_magnitude() -> None:
    node = _raw_validation()
    node["validation"]["comparands"]["rates_level"]["expected_beta_sign"] = -7.9
    with pytest.raises(ConfigError, match="convention, not a magnitude"):
        _parse(node)


def test_an_unrecognised_gate_raises() -> None:
    node = _raw_validation()
    node["validation"]["comparands"]["equity"]["gate"] = "vibes"
    with pytest.raises(ConfigError, match="expected one of"):
        _parse(node)


def test_an_unmapped_factor_owes_a_reason() -> None:
    node = _raw_validation()
    node["validation"]["unmapped"]["credit"] = "   "
    with pytest.raises(ConfigError, match="owes a stated reason"):
        _parse(node)


def test_only_the_newey_west_1994_rule_is_accepted() -> None:
    node = _raw_validation()
    node["validation"]["newey_west_lag_rule"] = "covariance.volatility_newey_west_lags.short"
    with pytest.raises(ConfigError, match="is not implemented"):
        _parse(node)


def test_the_rolling_window_must_admit_a_fisher_standard_error() -> None:
    node = _raw_validation()
    node["validation"]["rolling_correlation_window_months"] = 3
    with pytest.raises(ConfigError, match="undefined at w <= 3"):
        _parse(node)


def test_the_equity_factor_is_never_validated_against_its_own_source() -> None:
    """SPEC.md 4.1.1 ruling 2. Regressing Mkt-RF on Mkt-RF returns R^2 = 1.

    The factor is Ken French ``Mkt-RF``, so no configured comparand may come
    from the Ken French table. Its comparands are AQR's equity-index market
    portfolio and, at daily frequency, our own SPY excess return.
    """
    settings = load()
    assert settings.model.factors.macro.equity.source == "ken_french"
    for item in settings.model.factors.validation.comparands:
        assert "french" not in item.dataset
        assert item.column != settings.model.factors.macro.equity.column
    assert settings.model.factors.validation.daily_anchor.comparand == "spy_excess"


def test_tsmom_is_recorded_as_applying_to_nothing_in_model_a() -> None:
    """SPEC.md 6.5.1. No factor in SPEC.md 4.1's set is a trend factor."""
    settings = load().model.factors.validation
    assert settings.tsmom_applicable is False
    for item in settings.comparands:
        assert item.dataset != "tsmom"


def test_every_model_a_factor_is_accounted_for_exactly_once() -> None:
    """Mapped or reasoned-about. A factor cannot silently fall out of the table."""
    settings = load().model.factors.validation
    covered = set(settings.mapped) | set(settings.unmapped)
    assert covered == {
        macro.EQUITY,
        macro.LEVEL,
        macro.SLOPE,
        macro.CREDIT,
        macro.COMMODITY,
        macro.DOLLAR,
    }


def test_the_fisher_standard_error_is_the_hand_computed_value() -> None:
    """1/sqrt(36-3) = 1/5.744562... = 0.174078..."""
    settings = load().model.factors.validation
    assert settings.rolling_correlation_window_months == 36
    assert settings.rolling_correlation_standard_error == pytest.approx(
        0.1740776559556978, rel=1e-12
    )


# ---------------------------------------------------------------------------
# 5. The placebo can fail
# ---------------------------------------------------------------------------


def test_the_placebo_passes_when_the_mapping_does_work() -> None:
    row = PlaceboRow(
        factor="commodity",
        own="commodity",
        r_squared={"commodity": 0.49, "equity": 0.06, "rates_level": 0.20},
        observations={"commodity": 201, "equity": 201, "rates_level": 201},
    )
    assert row.passed
    assert row.best_other == ("rates_level", 0.20)
    assert row.margin == pytest.approx(0.29)


def test_the_placebo_fails_when_another_comparand_explains_the_factor_as_well() -> None:
    """The gate has teeth: this is the decoration case it exists to catch."""
    row = PlaceboRow(
        factor="commodity",
        own="commodity",
        r_squared={"commodity": 0.19, "equity": 0.06, "rates_level": 0.20},
        observations={"commodity": 201, "equity": 201, "rates_level": 201},
    )
    assert not row.passed
    assert row.margin < 0


def test_a_tie_does_not_pass() -> None:
    row = PlaceboRow(
        factor="equity",
        own="equity",
        r_squared={"equity": 0.5, "commodity": 0.5},
        observations={"equity": 10, "commodity": 10},
    )
    assert not row.passed


# ---------------------------------------------------------------------------
# 6. No two comparands are the same series
# ---------------------------------------------------------------------------


def _pair(raw: float, standardized: float, months: int = 200) -> validation.ComparandPair:
    return validation.ComparandPair(
        left="a",
        right="b",
        months=months,
        max_abs_difference=raw,
        max_abs_standardized_difference=standardized,
        correlation=1.0,
    )


def test_a_literal_duplicate_is_the_same_series() -> None:
    assert _pair(0.0, 0.0).is_same_series


def test_an_affine_rescaling_is_also_the_same_series() -> None:
    """The realistic duplication mode is a unit change, not a byte-for-byte copy.

    A comparand at twice the scale would tautologise the placebo exactly as
    thoroughly, so the check standardizes both sides before comparing.
    """
    assert _pair(raw=12.5, standardized=0.0).is_same_series


def test_two_genuinely_different_series_are_not_flagged() -> None:
    assert not _pair(raw=0.26, standardized=7.26).is_same_series


def test_no_overlap_is_not_a_duplicate() -> None:
    assert not _pair(float("nan"), float("nan"), months=0).is_same_series


# ---------------------------------------------------------------------------
# The live table. `dataset`-marked: reads data/raw, absent on a clean clone.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def result() -> validation.ValidationResult:
    pytest.importorskip("pyarrow")
    try:
        return validation.build()
    except FileNotFoundError as exc:  # pragma: no cover - clean clone
        pytest.skip(f"data cache absent: {exc}")


@pytest.mark.dataset
def test_the_aggregation_round_trips_exactly_on_the_real_panel(
    result: validation.ValidationResult,
) -> None:
    for name, residual in result.round_trip.items():
        assert abs(residual) < 1e-9, f"{name} does not round trip"


@pytest.mark.dataset
def test_nothing_in_the_table_reaches_the_holdout(
    result: validation.ValidationResult,
) -> None:
    """CLAUDE.md invariant 5, asserted on every comparison the report prints."""
    boundary = pd.Timestamp(load().require_holdout_start())
    for item in result.fits:
        assert item.observed_end < boundary
    for anchor in result.anchors:
        assert anchor.end < boundary
    assert pd.Timestamp(result.panel.end) < boundary


@pytest.mark.dataset
def test_the_two_commodity_columns_are_one_series(
    result: validation.ValidationResult,
) -> None:
    """Measured, not assumed. It is why only one of them is configured."""
    assert result.duplicate_months > 1000
    assert result.duplicate_max_difference < 1e-9


@pytest.mark.dataset
def test_the_configured_comparands_are_pairwise_distinct(
    result: validation.ValidationResult,
) -> None:
    """The general rule, on the shipped mapping."""
    assert len(result.comparand_pairs) == 3
    for pair in result.comparand_pairs:
        assert not pair.is_same_series, pair.render()
        assert pair.months > 1000


@pytest.mark.dataset
def test_the_distinctness_assertion_fires_on_the_real_duplicate() -> None:
    """The near-miss, turned into a regression test.

    Had Century's `Commodities Market` been configured alongside the CLR column
    -- which an earlier draft of the mapping proposed, on the reasoning that a
    second AQR construction costs nothing -- `commodity` would have been scored
    against a second copy of its own series and the placebo would have passed by
    tautology. This is that configuration, and it must now raise.
    """
    settings = load().model.factors.validation
    century_commodity = Comparand(
        factor="commodity_duplicate",
        dataset="century_of_factor_premia",
        column="Commodities Market",
        gate="placebo",
        expected_beta_sign=None,
    )
    with pytest.raises(validation.ValidationError, match="same series"):
        validation.assert_comparands_are_distinct([*settings.comparands, century_commodity])


@pytest.mark.dataset
def test_the_three_registered_gates_hold(result: validation.ValidationResult) -> None:
    assert result.anchor_passed, "row 41 not reproduced on its own window"
    assert result.placebo_passed, [row.factor for row in result.placebo_rows if not row.passed]
    assert result.sign_gate_passed, "rates_level beta did not come back negative"


@pytest.mark.dataset
def test_the_monthly_sample_is_two_hundred_odd_months_not_four_thousand(
    result: validation.ValidationResult,
) -> None:
    """The frequency change is the thing this session could get quietly wrong."""
    for item in result.fits:
        assert 190 <= item.observations <= 215
        assert item.newey_west_lags == 4


@pytest.mark.dataset
def test_the_flagged_alphas_are_handed_on_rather_than_closed(
    result: validation.ValidationResult,
) -> None:
    """The published comparand does not explain them, and the record says so.

    Pinning the *direction* of the answer, not its size: if a later change made
    AQR's bond series account for the government carry-and-roll term premium,
    the W3-P5 control this session fixes would no longer be needed and that must
    surface as a failing test rather than as a stale paragraph.
    """
    assert len(result.confounds) == 4
    assert result.unexplained_confounds == result.confounds
    for item in result.confounds:
        assert abs(item.delta_r_squared) < 0.01


@pytest.mark.dataset
def test_the_report_states_what_it_could_not_validate(
    result: validation.ValidationResult,
) -> None:
    from mafrm.factors import validation_report

    text = validation_report.render(result, load())
    for factor in (macro.SLOPE, macro.CREDIT, macro.DOLLAR):
        assert f"`{factor}`" in text
    assert "no analogue" in text
    assert "TSMOM" in text
    assert "sign test" in text
    # The withdrawn gate is argued, not asserted.
    assert "0.3" in text and "SPEC.md 4.1.2 precedent" in text
    # Every comparison carries its window.
    for item in result.fits:
        assert item.window in text
