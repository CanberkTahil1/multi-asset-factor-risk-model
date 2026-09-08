"""SPEC.md 15.6.1's industry inputs, on fixtures.

Every expected value below was worked by hand from the fixtures before the
code ran (CLAUDE.md, definition of done). Nothing here fetches: the Siccodes
excerpt is a five-industry file in French's format, the submissions records
are hand-written in the endpoint's shape, and the SGML header is the public
EDGAR record of XOM's 10-K filed 2008-02-28.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from mafrm.data import sic

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def siccodes() -> pd.DataFrame:
    return sic.parse_siccodes(
        # 5 of 49 industries from Ken French's Siccodes49.txt; provenance and read
        # date in tests/fixtures/README.md (no header in the file itself -- the
        # parser under test rejects comment lines, which is the point of it).
        (FIXTURES / "siccodes_excerpt.txt").read_bytes(),
        industries=5,
        unassigned=5,
    )


# ---------------------------------------------------------------------------
# Siccodes
# ---------------------------------------------------------------------------


def test_siccodes_parse_by_hand(siccodes: pd.DataFrame) -> None:
    """Five blocks, eleven ranges (3 + 2 + 3 + 2 + 1); block 3 'Oil' holds 1300-1300,
    1310-1319, 2900-2912."""
    assert len(siccodes) == 11
    assert siccodes["industry"].tolist() == [1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5]
    oil = siccodes[siccodes["industry"] == 3]
    assert oil["abbrev"].unique().tolist() == ["Oil"]
    assert oil["name"].unique().tolist() == ["Petroleum and Natural Gas"]
    assert list(zip(oil["sic_low"], oil["sic_high"], strict=True)) == [
        (1300, 1300),
        (1310, 1319),
        (2900, 2912),
    ]
    assert siccodes.loc[siccodes["sic_low"] == 6200, "description"].iloc[0] == (
        "Security and commodity brokers, dealers, exchanges & services"
    )


def test_industry_of_by_hand(siccodes: pd.DataFrame) -> None:
    """2911 -> 3 (2900-2912); 0150 -> 1; 6798 -> 4; 4953 -> 5 (listed under Other);
    3571 -> 5 (in no range: French's rule); 2048 -> 1 (a single-code range)."""
    assert sic.industry_of(2911, siccodes, unassigned=5) == 3
    assert sic.industry_of(150, siccodes, unassigned=5) == 1
    assert sic.industry_of(6798, siccodes, unassigned=5) == 4
    assert sic.industry_of(4953, siccodes, unassigned=5) == 5
    assert sic.industry_of(3571, siccodes, unassigned=5) == 5
    assert sic.industry_of(2048, siccodes, unassigned=5) == 1


def test_map_keeps_a_missing_sic_missing(siccodes: pd.DataFrame) -> None:
    """Ruling 2: a missing code is NOT an 'Other' code."""
    sics = pd.Series([2911, pd.NA, 3571], index=["XOM", "GOOGL", "AAPL"], dtype="Int64")
    out = sic.map_sic_to_industry(sics, siccodes, unassigned=5)
    assert out["XOM"] == 3 and out["AAPL"] == 5
    assert pd.isna(out["GOOGL"])


def test_siccodes_rejects_a_wrong_industry_count_and_an_overlap() -> None:
    text = (FIXTURES / "siccodes_excerpt.txt").read_bytes()
    with pytest.raises(sic.SicError, match=r"expected industries 1\.\.49"):
        sic.parse_siccodes(text, industries=49, unassigned=49)
    overlapping = text + b" 6 Dup    Duplicate\r\n          2905-2910 Overlaps Oil\r\n"
    with pytest.raises(sic.SicError, match="overlap"):
        sic.parse_siccodes(overlapping, industries=6, unassigned=5)


# ---------------------------------------------------------------------------
# Submissions
# ---------------------------------------------------------------------------


def test_submissions_name_is_ten_digits() -> None:
    assert sic.submissions_name(34088) == "CIK0000034088.json"


def test_parse_submissions_by_hand() -> None:
    record = sic.parse_submissions((FIXTURES / "edgar_submissions_CIK0000034088.json").read_bytes())
    assert record.cik == 34088 and record.sic == 2911
    assert record.sic_description == "Petroleum Refining"
    assert len(record.filings) == 5
    assert record.older_pages == (
        ("CIK0000034088-submissions-001.json", date(2005, 10, 4), date(2019, 12, 16)),
        ("CIK0000034088-submissions-002.json", date(1994, 3, 4), date(2005, 10, 2)),
    )
    # Only the page whose span reaches sample.start (2007-04-01) is needed.
    assert sic.older_pages_needed(record, start=date(2007, 4, 1)) == (
        "CIK0000034088-submissions-001.json",
    )
    assert sic.older_pages_needed(record, start=date(2020, 1, 1)) == ()


def test_parse_submissions_without_a_sic() -> None:
    record = sic.parse_submissions((FIXTURES / "edgar_submissions_nosic.json").read_bytes())
    assert record.sic is None and record.filings.empty and record.older_pages == ()


def test_first_filing_by_hand() -> None:
    """Across recent + the older page: the 10-Ks on/after 2007-04-01 are 2022-02-23,
    2020-02-26 (recent; the 10-K/A of 2021 is not a 10-K) and 2009-02-27, 2008-02-28
    (older page; 2007-02-28 is before the start). Earliest: 2008-02-28."""
    record = sic.parse_submissions((FIXTURES / "edgar_submissions_CIK0000034088.json").read_bytes())
    page = sic.parse_filings_page(
        (FIXTURES / "edgar_submissions_CIK0000034088-001.json").read_bytes()
    )
    filings = pd.concat([record.filings, page], ignore_index=True)
    hit = sic.first_filing(filings, forms=("10-K",), start=date(2007, 4, 1))
    assert hit is not None
    assert hit["accession"] == "0001193125-08-041781"
    assert hit["filing_date"] == pd.Timestamp("2008-02-28")
    # Recent only: the 2020 10-K, not the 2021 amendment.
    recent = sic.first_filing(record.filings, forms=("10-K",), start=date(2007, 4, 1))
    assert recent is not None and recent["accession"] == "0000034088-20-000016"
    # Nothing on or after a start past the last 10-K.
    assert sic.first_filing(filings, forms=("10-K",), start=date(2023, 1, 1)) is None


# ---------------------------------------------------------------------------
# The SGML header
# ---------------------------------------------------------------------------


def test_parse_header_by_hand() -> None:
    header = sic.parse_header(
        (FIXTURES / "edgar_header_0001193125-08-041781.hdr.sgml").read_bytes(), cik=34088
    )
    assert header.accession == "0001193125-08-041781"
    assert header.form == "10-K"
    assert header.filing_date == date(2008, 2, 28)
    assert header.period == date(2007, 12, 31)
    assert header.sic == 2911 and header.filer_found


def test_parse_header_takes_the_sic_from_the_matching_filer_only() -> None:
    text = (FIXTURES / "edgar_header_0001193125-08-041781.hdr.sgml").read_text("latin-1")
    second = text.replace(
        "<CIK>0000034088", "<CIK>0000000999", 1
    )  # a different filer's block first
    two_filers = text.replace("</SEC-HEADER>", "") + second[second.index("<FILER>") :]
    header = sic.parse_header(two_filers, cik=999)
    assert header.sic == 2911 and header.filer_found
    other = sic.parse_header(two_filers, cik=123456)
    assert other.sic is None and not other.filer_found


# ---------------------------------------------------------------------------
# Per-ticker industries and the drift table
# ---------------------------------------------------------------------------


def test_industries_for_and_drift_by_hand(siccodes: pd.DataFrame) -> None:
    """XOM 2911 -> 3; AAPL 3571 -> 5 (Other); BRK-B has a CIK with no SIC -> no_sic;
    ZZZ has no CIK -> no_cik. First-10-K SICs: XOM 2911 (same), AAPL 2000 (-> 2,
    industry differs), BRK-B none."""
    mapping = pd.DataFrame(
        {"cik": pd.array([34088, 320193, 1067983, pd.NA], dtype="Int64")},
        index=pd.Index(["XOM", "AAPL", "BRK-B", "ZZZ"], name="ticker"),
    )
    current = pd.DataFrame(
        {
            "sic": pd.array([2911, 3571, pd.NA], dtype="Int64"),
            "sic_description": ["Petroleum Refining", "Electronic Computers", ""],
            "name": ["EXXON MOBIL CORP", "Apple Inc.", "BERKSHIRE HATHAWAY INC"],
            "status": ["ok", "ok", "no_sic"],
        },
        index=pd.Index([34088, 320193, 1067983], name="cik"),
    )
    industries = sic.industries_for(mapping, current, siccodes, unassigned=5)
    assert industries["status"].tolist() == ["ok", "ok", "no_sic", "no_cik"]
    assert industries["industry"].tolist()[:2] == [3, 5]
    assert pd.isna(industries.loc["BRK-B", "industry"]) and pd.isna(industries.loc["ZZZ", "cik"])

    first = pd.DataFrame(
        {
            "cik": [34088, 320193],
            "accession": ["0001193125-08-041781", "0001047469-07-009340"],
            "filing_date": pd.to_datetime(["2008-02-28", "2007-12-14"]),
            "sic": pd.array([2911, 2000], dtype="Int64"),
            "status": ["ok", "ok"],
        },
        index=pd.Index(["XOM", "AAPL"], name="ticker"),
    )
    drift = sic.drift_table(industries, first, siccodes, unassigned=5)
    assert drift.loc["XOM", "sic_differs"] is False or not bool(drift.loc["XOM", "sic_differs"])
    assert bool(drift.loc["AAPL", "sic_differs"]) and bool(drift.loc["AAPL", "industry_differs"])
    assert drift.loc["AAPL", "industry_first"] == 2 and drift.loc["AAPL", "industry_current"] == 5
    assert pd.isna(drift.loc["BRK-B", "sic_differs"]) and pd.isna(
        drift.loc["ZZZ", "industry_differs"]
    )


# ---------------------------------------------------------------------------
# Point-in-time industries (W7-P3b, SPEC.md 15.6.3)
# ---------------------------------------------------------------------------


def test_point_in_time_industries_by_hand(siccodes: pd.DataFrame) -> None:
    """Sessions 2008-01-01..2008-01-10 (8 business days). AMT (drifted) filed a 10-K
    on 2008-01-03 with SIC 2911 (-> 3 Oil) and on 2008-01-08 with SIC 6798 (-> 4 Fin);
    current industry 4. So: 01-01, 01-02 back-filled to 3; 01-03..01-07 -> 3;
    01-08..01-10 -> 4. XOM (not drifted, current 3) is 3 throughout. A filing whose
    status is not ok sets nothing."""
    sessions = pd.bdate_range("2008-01-01", "2008-01-10")
    industries = pd.DataFrame(
        {"industry": pd.array([4, 3], dtype="Int64")},
        index=pd.Index(["AMT", "XOM"], name="ticker"),
    )
    history = pd.DataFrame(
        {
            "ticker": ["AMT", "AMT", "AMT"],
            "sic": pd.array([2911, 6798, 100], dtype="Int64"),
            "status": ["ok", "ok", "no_sic_in_header"],
        },
        index=pd.DatetimeIndex(["2008-01-03", "2008-01-08", "2008-01-09"], name="filing_date"),
    )
    out = sic.point_in_time_industries(industries, history, sessions, siccodes, unassigned=5)
    assert out.shape == (8, 2)
    assert out["AMT"].tolist() == [3, 3, 3, 3, 3, 4, 4, 4]
    assert out["XOM"].tolist() == [3] * 8
    # No history at all: everyone keeps the current industry.
    empty = sic.point_in_time_industries(
        industries, history.iloc[0:0], sessions, siccodes, unassigned=5
    )
    assert empty["AMT"].tolist() == [4] * 8


def test_drifted_tickers_reads_the_industry_flag() -> None:
    drift = pd.DataFrame(
        {"industry_differs": pd.array([True, False, pd.NA], dtype="boolean")},
        index=pd.Index(["AMT", "XOM", "BRK-B"], name="ticker"),
    )
    assert sic.drifted_tickers(drift) == ("AMT",)
