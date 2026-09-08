"""W6-P3's bands: the twelve points from the rules, the legs scored, no selection."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from mafrm import config
from mafrm.backtest import bands, grid

CFG = config.load()
GRID = CFG.model.optimizer.grid


def _fake_universe(assets: int = 13, thinnest_adv: float = 31.2e6) -> Any:
    """Only what ``band_specs`` reads: the asset count and the ADV medians."""
    names = [f"a{i}" for i in range(assets)]
    return SimpleNamespace(
        assets=tuple(names),
        adv_median=pd.Series([thinnest_adv + i * 1e6 for i in range(assets)], index=names),
    )


def test_the_twelve_band_points_come_from_the_rules_and_the_reference_is_none_of_them() -> None:
    specs = bands.band_specs(_fake_universe(), CFG)
    assert len(specs) == 12
    tags = [b.tag for b in specs]
    assert tags == [
        "spread_high",
        "te_0.5x",
        "te_2x",
        "book_A_low",
        "book_A_high",
        "gamma_trade_1.5",
        "gamma_trade_2",
        "gamma_trade_2.5",
        "gamma_trade_3",
        "bounds_2_over_N",
        "horizon_long",
        "spec_initialisation",
    ]
    by_tag = {b.tag: b for b in specs}
    assert by_tag["spread_high"].overrides == {"spread_end": "high"}
    assert by_tag["te_0.5x"].overrides == {"tracking_error_multiple": 0.5}
    assert by_tag["te_2x"].overrides == {"tracking_error_multiple": 2.0}
    # A_low = 0.001 x 13 x 31.2M; A_high = 0.10 x 13 x 31.2M -- W6-P1's rule, by hand.
    assert by_tag["book_A_low"].overrides["nav"] == pytest.approx(0.001 * 13 * 31.2e6, rel=1e-12)
    assert by_tag["book_A_high"].overrides["nav"] == pytest.approx(0.10 * 13 * 31.2e6, rel=1e-12)
    assert [by_tag[f"gamma_trade_{g:g}"].overrides["gamma_trade"] for g in (1.5, 2, 2.5, 3)] == [
        1.5,
        2.0,
        2.5,
        3.0,
    ]
    assert by_tag["bounds_2_over_N"].overrides["position_box"] == pytest.approx(2 / 13, rel=1e-12)
    assert by_tag["horizon_long"].horizon == "long" and by_tag["horizon_long"].overrides == {}
    assert by_tag["spec_initialisation"].overrides == {"risk_aversion": "spec_initialisation"}
    # Every band carries its registered prediction, written before the run.
    assert all(b.registered for b in specs)
    # No band restates the reference's own value on its dial.
    assert GRID.gamma_trade not in [b.overrides.get("gamma_trade") for b in specs]
    assert GRID.tracking_error_multiple not in [
        b.overrides.get("tracking_error_multiple") for b in specs
    ]
    assert GRID.spread_end not in [b.overrides.get("spread_end") for b in specs]


def _synthetic_result(tag: str, seed: int, **overrides: Any) -> grid.CellResult:
    cell = grid.CellSpec("eigenfactor_a1.0", "time_varying", "patient", tag=tag)
    inputs = dataclasses.replace(grid.synthetic_inputs(CFG, cell=cell, seed=seed), **overrides)
    return grid.run_cell(inputs, CFG)


@pytest.fixture(scope="module")
def synthetic_bands() -> bands.BandsResult:
    """A reduced band set on the synthetic cell, so the scoring and the report run offline."""
    reference = _synthetic_result("", 5)
    specs = (
        bands.Band(
            "te_0.5x", "TE multiple", "0.5x", "B > 1.10", overrides={"tracking_error_multiple": 0.5}
        ),
        bands.Band(
            "te_2x", "TE multiple", "2x", "B falls", overrides={"tracking_error_multiple": 2.0}
        ),
        bands.Band("gamma_trade_3", "gamma_trade", "3", "monotone", overrides={"gamma_trade": 3.0}),
        bands.Band(
            "bounds_2_over_N",
            "per-asset bound",
            "2/3",
            "closer to one",
            overrides={"position_box": 2 / 3},
        ),
        bands.Band(
            "spec_initialisation",
            "risk aversion",
            "spec_initialisation",
            "attainment < 0.5",
            overrides={"risk_aversion": "spec_initialisation"},
        ),
    )
    results = []
    for band in specs:
        kwargs = dict(band.overrides)
        multiple = kwargs.pop("tracking_error_multiple", None)
        if multiple is not None:
            kwargs["te_target"] = reference.inputs.te_target * float(multiple)
        results.append(_synthetic_result(band.tag, 5, **kwargs))
    out = bands.BandsResult(
        reference=reference,
        bands=specs,
        results=tuple(results),
        trials=50,
        trials_variance=1e-4,
    )
    return dataclasses.replace(out, checks=bands.registered_checks(out, CFG))


def test_the_legs_are_scored_against_the_reference_in_the_same_run(
    synthetic_bands: bands.BandsResult,
) -> None:
    out = synthetic_bands
    legs = {str(c["band"]): c for c in out.checks}
    assert {
        "te_multiples",
        "te_0.5x",
        "gamma_trade",
        "bounds_2_over_N",
        "spec_initialisation",
    } <= set(legs)
    # The TE ordering leg reads B at 0.5x, 1x (the reference) and 2x, in that order.
    assert "0.5x" in str(legs["te_multiples"]["observed"]) and "2x" in str(
        legs["te_multiples"]["observed"]
    )
    b_low = out.by_tag("te_0.5x").identity.bias_ratio
    assert legs["te_0.5x"]["holds"] == (b_low > 1.10)
    # The gamma_trade leg includes the reference's 1.0 as the first point of the sweep.
    assert str(legs["gamma_trade"]["observed"]).startswith("1: turnover")
    heavier = out.by_tag("gamma_trade_3")
    assert heavier.summary["turnover_annual"] <= out.reference.summary["turnover_annual"]
    bounded = out.by_tag("bounds_2_over_N")
    assert legs["bounds_2_over_N"]["holds"] == (
        abs(bounded.identity.bias_ratio - 1.0) < abs(out.reference.identity.bias_ratio - 1.0)
    )
    spec = out.by_tag("spec_initialisation")
    assert "tracking_error_binding" not in spec.rows.columns


def test_the_report_and_the_metrics_section_carry_every_point_and_no_selection(
    synthetic_bands: bands.BandsResult,
) -> None:
    text = bands.render(synthetic_bands, CFG)
    for band in synthetic_bands.bands:
        assert f"| {band.tag} |" in text
    assert "| reference |" in text
    assert "no point is selected" in text
    section = bands.metrics_section(synthetic_bands, CFG)
    assert section["band_points"] == len(synthetic_bands.results)
    assert section["trial_count_N"] == 50
    tags = [b["tag"] for b in section["bands"]]  # type: ignore[union-attr]
    assert tags == [b.tag for b in synthetic_bands.bands]
    assert "selected" not in {k for entry in section["bands"] for k in entry}  # type: ignore[union-attr]
    ident = section["reference"]["identity"]  # type: ignore[index]
    assert ident["risk_model_term"] + ident["cost_term"] == pytest.approx(
        ident["sharpe_paper"] - ident["sharpe_real"], abs=1e-9
    )


def test_a_band_with_a_missing_reference_book_size_rule_is_refused() -> None:
    universe = SimpleNamespace(assets=("a",), adv_median=pd.Series([1.0], index=["a"]))
    raw = dataclasses.replace(
        CFG.model.optimizer.verification,
        book_size=config.OptimizerBookSize(rule="fixed", nav_dollars=1.0e6),
    )
    optimizer = dataclasses.replace(CFG.model.optimizer, verification=raw)
    model = dataclasses.replace(CFG.model, optimizer=optimizer)
    settings = dataclasses.replace(CFG, model=model)
    with pytest.raises(bands.BandsError, match="A_low and A_high"):
        bands.band_specs(universe, settings)  # type: ignore[arg-type]


def test_write_metrics_section_leaves_the_grids_own_payload_alone(tmp_path: Any) -> None:
    path = tmp_path / "metrics.json"
    path.write_text('{"task": "W6-P2", "cells": [1, 2]}', encoding="utf-8")
    bands.write_metrics_section({"task": "W6-P3", "band_points": 12}, path)
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["task"] == "W6-P2" and payload["cells"] == [1, 2]
    assert payload["bands"]["band_points"] == 12
    # And the grid writer preserves the bands section in turn.
    grid.write_metrics({"task": "W6-P3", "cells": []}, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["bands"]["band_points"] == 12 and payload["cells"] == []
    assert np.isfinite(payload["bands"]["band_points"])


def test_the_chart_data_reads_the_committed_bands_section() -> None:
    from pathlib import Path

    metrics_path = Path(__file__).resolve().parents[1] / "results" / "metrics.json"
    data = bands.ChartData.from_metrics(metrics_path)
    assert len(data.labels) == 12 and data.trials >= 57
    for values, reference in data.series.values():
        assert len(values) == 12 and np.isfinite(reference)
    risk, _cost, _b = data.series.values()
    # The reference's identity is exact in the committed section.
    section = __import__("json").loads(metrics_path.read_text(encoding="utf-8"))["bands"]
    ident = section["reference"]["identity"]
    assert ident["risk_model_term"] + ident["cost_term"] == pytest.approx(
        ident["sharpe_paper"] - ident["sharpe_real"], abs=1e-8
    )
    assert risk[0][1] == pytest.approx(section["bands"][1]["identity"]["risk_model_term"])
