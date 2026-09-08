"""Ken French, AQR and Stooq parsers. Offline, against synthesised payloads.

The AQR fixtures are built here with the same banner heights the real workbooks
carry, because the banner height is the thing that breaks: it differs per file
and a parser that guesses reads prose as data.
"""

from __future__ import annotations

import io

import pandas as pd
import pytest

from mafrm.data import aqr, contracts, french, stooq

# ---------------------------------------------------------------------------
# Ken French
# ---------------------------------------------------------------------------


def french_table() -> pd.DataFrame:
    return pd.DataFrame(
        {"Mkt-RF": [0.09, 0.45], "SMB": [-0.25, -0.33], "HML": [-0.27, -0.06], "RF": [0.01, 0.01]},
        index=pd.DatetimeIndex(pd.to_datetime(["1926-07-01", "1926-07-02"]), name="Date"),
    )


def test_normalise_names_and_sorts_the_index() -> None:
    raw = french_table().iloc[::-1]
    tidy = french.normalise_daily_factors(raw, rf_column="RF")
    assert tidy.index.name == "date"
    assert list(tidy.index) == [pd.Timestamp("1926-07-01"), pd.Timestamp("1926-07-02")]


def test_a_missing_rf_column_raises() -> None:
    raw = french_table().drop(columns=["RF"])
    with pytest.raises(french.FrenchError, match="no 'RF' column"):
        french.normalise_daily_factors(raw, rf_column="RF")


def test_a_duplicated_date_raises() -> None:
    raw = pd.concat([french_table(), french_table().iloc[:1]])
    with pytest.raises(french.FrenchError, match="repeats"):
        french.normalise_daily_factors(raw, rf_column="RF")


def test_values_are_left_in_the_publishers_units() -> None:
    """A cache that silently rescales cannot be checked against the source file."""
    tidy = french.normalise_daily_factors(french_table(), rf_column="RF")
    assert float(tidy["RF"].iloc[0]) == pytest.approx(0.01)


# --- the units trap, hand-computed ---------------------------------------


def test_annualisation_is_a_units_change_and_nothing_more() -> None:
    """RF = 0.021 percent per trading day, 252 days -> 5.292 percent per annum."""
    daily = pd.Series([0.021], index=pd.DatetimeIndex(["2020-01-02"]), name="RF")
    annual = french.as_annual_percent(daily, trading_days_per_year=252)
    assert float(annual.iloc[0]) == pytest.approx(5.292, abs=1e-12)


def test_the_annualisation_round_trips_through_the_daily_accrual_exactly() -> None:
    """The identity the conversion exists to preserve.

    ``mafrm.data.gsw.constant_maturity_return`` charges the cash leg as
    ``rf_annual * delta / 100`` with ``delta = 1/252``. Feeding it the converted
    Ken French rate must return the fraction Ken French actually published for
    that day -- otherwise substituting this leg for DGS1MO changes the excess
    returns by 252x, which is far more than the choice of series does.
    """
    published_percent_per_day = 0.021
    daily = pd.Series(
        [published_percent_per_day], index=pd.DatetimeIndex(["2020-01-02"]), name="RF"
    )
    annual = french.as_annual_percent(daily, trading_days_per_year=252)

    accrued = float(annual.iloc[0]) * (1.0 / 252) / 100.0

    assert accrued == pytest.approx(published_percent_per_day / 100.0, rel=1e-15)


def test_a_nonsensical_trading_year_raises() -> None:
    daily = pd.Series([0.01], index=pd.DatetimeIndex(["2020-01-02"]), name="RF")
    with pytest.raises(french.FrenchError, match="must be positive"):
        french.as_annual_percent(daily, trading_days_per_year=0)


# ---------------------------------------------------------------------------
# AQR
# ---------------------------------------------------------------------------


def workbook(sheet: str, banner_rows: int, header: list[object], rows: list[list[object]]) -> bytes:
    """An .xlsx with ``banner_rows`` rows of prose above the header, like AQR's."""
    body: list[list[object]] = [[f"AQR banner line {i}"] for i in range(banner_rows)]
    body.append(header)
    body.extend(rows)
    width = max(len(r) for r in body)
    padded = [list(r) + [None] * (width - len(r)) for r in body]
    buffer = io.BytesIO()
    pd.DataFrame(padded).to_excel(buffer, sheet_name=sheet, header=False, index=False)
    return buffer.getvalue()


def spec(**overrides: object) -> aqr.WorkbookSpec:
    base = {
        "name": "fixture",
        "url": "https://example.invalid/f.xlsx",
        "sheet": "Sheet",
        "header_row": 3,
        "date_column": "Date",
        "required_columns": ("A",),
        "description": "fixture",
    }
    base.update(overrides)
    return aqr.WorkbookSpec(**base)  # type: ignore[arg-type]


def test_a_labelled_date_column_parses() -> None:
    """The *Century of Factor Premia* layout: banner, then a `Date` header."""
    payload = workbook("Sheet", 3, ["Date", "A"], [["07/30/1926", -0.0198], ["08/31/1926", 0.0523]])
    frame = aqr.parse_workbook(payload, spec())
    assert list(frame.columns) == ["A"]
    assert frame.index[0] == pd.Timestamp("1926-07-30")
    assert float(frame["A"].iloc[1]) == pytest.approx(0.0523)


def test_a_blank_date_header_is_taken_by_position() -> None:
    """The *TSMOM* and *Commodities* layout: AQR leaves the header cell empty."""
    payload = workbook("Sheet", 3, [None, "A"], [["1985-01-31", 0.043], ["1985-02-28", 0.037]])
    frame = aqr.parse_workbook(payload, spec(date_column=None))
    assert list(frame.columns) == ["A"]
    assert frame.index[0] == pd.Timestamp("1985-01-31")


def test_the_wrong_header_row_raises_instead_of_yielding_prose() -> None:
    """The failure a header sniffer would produce silently."""
    payload = workbook("Sheet", 3, ["Date", "A"], [["07/30/1926", -0.0198]])
    with pytest.raises(aqr.AqrError):
        aqr.parse_workbook(payload, spec(header_row=1))


def test_a_missing_named_date_column_raises_and_blames_the_banner() -> None:
    payload = workbook("Sheet", 3, ["Period", "A"], [["07/30/1926", -0.0198]])
    with pytest.raises(aqr.AqrError, match="banner height"):
        aqr.parse_workbook(payload, spec())


def test_a_missing_required_column_raises() -> None:
    payload = workbook("Sheet", 3, ["Date", "B"], [["07/30/1926", 1.0], ["08/31/1926", 2.0]])
    with pytest.raises(contracts.ContractError, match="missing column"):
        aqr.parse_workbook(payload, spec(required_columns=("A",)))


def test_a_missing_sheet_raises() -> None:
    payload = workbook("Sheet", 3, ["Date", "A"], [["07/30/1926", 1.0], ["08/31/1926", 2.0]])
    with pytest.raises(aqr.AqrError, match="cannot read sheet"):
        aqr.parse_workbook(payload, spec(sheet="Absent"))


def test_categorical_columns_survive_rather_than_becoming_all_nan() -> None:
    """*Commodities for the Long Run* carries two regime-label columns.

    A blanket ``to_numeric(errors="coerce")`` turns all 1,780 rows of each into
    NaN, which then trips the all-NaN contract and looks exactly like a renamed
    upstream field. It is neither -- the column was never numeric.
    """
    payload = workbook(
        "Sheet",
        3,
        ["Date", "A", "State of inflation"],
        [["1877-02-28", -0.08, "High"], ["1877-03-30", -0.04, "Low"]],
    )
    frame = aqr.parse_workbook(payload, spec(required_columns=("A", "State of inflation")))
    assert list(frame["State of inflation"]) == ["High", "Low"]
    assert float(frame["A"].iloc[0]) == pytest.approx(-0.08)


def test_trailing_footnote_rows_are_dropped() -> None:
    payload = workbook(
        "Sheet",
        3,
        ["Date", "A"],
        [["1877-02-28", -0.08], ["1877-03-30", -0.04], ["Source: AQR", None]],
    )
    frame = aqr.parse_workbook(payload, spec())
    assert len(frame) == 2


# ---------------------------------------------------------------------------
# Stooq
# ---------------------------------------------------------------------------

CHALLENGE = (
    '<!DOCTYPE html><html><head><meta charset="utf-8"></head><body>'
    "<noscript>This site requires JavaScript to verify your browser.</noscript>"
    '<script>const h=await crypto.subtle.digest("SHA-256",e);'
    'await fetch("/__verify",{method:"POST"})</script></body></html>'
)


def test_symbols_take_the_us_suffix() -> None:
    assert stooq.symbol_for("SPY", template="{ticker}.us") == "spy.us"


def test_a_daily_csv_export_parses() -> None:
    text = "Date,Open,High,Low,Close,Volume\n2020-01-03,100,101,99,100.5,1000\n"
    frame = stooq.parse_csv(text, symbol="spy.us")
    assert list(frame.columns) == list(stooq.COLUMNS)
    assert frame.index[0] == pd.Timestamp("2020-01-03")
    assert float(frame["Close"].iloc[0]) == pytest.approx(100.5)


def test_the_browser_challenge_is_named_rather_than_parsed() -> None:
    """Distinguishing "the site refused us" from "our parser is wrong"."""
    with pytest.raises(stooq.StooqUnavailable) as caught:
        stooq.parse_csv(CHALLENGE, symbol="spy.us")
    assert "access control" in str(caught.value)


def test_other_html_is_a_plain_parse_error() -> None:
    with pytest.raises(stooq.StooqError, match="returned HTML"):
        stooq.parse_csv(
            "<html><body>The page you requested does not exist</body></html>", symbol="x"
        )


def test_a_missing_column_raises() -> None:
    with pytest.raises(stooq.StooqError, match="missing"):
        stooq.parse_csv("Date,Open,High,Low,Close\n2020-01-03,1,1,1,1\n", symbol="x")


def test_close_comparison_is_reported_on_shared_dates_only() -> None:
    index = pd.DatetimeIndex(pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]))
    primary = pd.Series([100.0, 101.0, 102.0], index=index)
    secondary = pd.Series([100.0, 100.0], index=index[:2])

    difference = stooq.compare_closes(primary, secondary)

    assert len(difference) == 2
    assert float(difference.iloc[0]) == pytest.approx(0.0)
    assert float(difference.iloc[1]) == pytest.approx(0.01)


def test_comparing_disjoint_series_raises() -> None:
    a = pd.Series([1.0], index=pd.DatetimeIndex(["2020-01-02"]))
    b = pd.Series([1.0], index=pd.DatetimeIndex(["2021-01-04"]))
    with pytest.raises(stooq.StooqError, match="share no dates"):
        stooq.compare_closes(a, b)
