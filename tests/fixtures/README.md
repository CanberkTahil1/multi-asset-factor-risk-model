# Test fixtures — provenance

Some fixtures below are excerpts of third-party files. **None is redistributed as
data**: each is the smallest sample that exercises a parser, and CLAUDE.md
invariant 8 governs the rest — the real files live in the gitignored cache and
are recorded in `data/manifest.json`, never committed.

Provenance is recorded here rather than in a header inside each file, because
these are *parser inputs*: `siccodes_excerpt.txt` is read by a strict
fixed-width parser (`mafrm.data.sic.parse_siccodes`) that rejects a comment line,
so a header in the file would break the very thing the fixture exists to test.

| Fixture | Source | Read | Note |
|---|---|---|---|
| `siccodes_excerpt.txt` | Ken French, `Siccodes49.txt` — [det_49_ind_port.html](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_49_ind_port.html) | 2026-09-05 (W7-P3) | 5 of 49 industries, reformatted to the same fixed-width shape |
| `edgar_company_tickers.json` | SEC EDGAR, `company_tickers.json` | 2026-09-05 (W7-P2a) | trimmed to the names under test |
| `edgar_frame_CY2020Q1I.json` | SEC EDGAR XBRL frames API | 2026-09-05 (W7-P2a) | one frame, trimmed |
| `edgar_submissions_CIK*.json`, `edgar_submissions_nosic.json` | SEC EDGAR submissions API | 2026-09-05 (W7-P2, W7-P3) | XOM and its predecessor CIK; the no-SIC case is hand-built |
| `edgar_header_*.hdr.sgml` | SEC EDGAR filing header | 2026-09-05 (W7-P3b) | one 10-K header |
| `edgar_xbrl_2020_QTR1.idx` | SEC EDGAR quarterly index | 2026-09-05 (W7-P2a) | truncated |
| `sp500_constituents.html`, `sp500_changes.html` | Wikipedia, the two S&P 500 articles | 2026-09-05 (W7-P1) | CC BY-SA 4.0, as `data/reference/` states |
| `covariance_golden.json` | generated — `python -m mafrm.risk.synthetic` | — | not third-party; see SPEC.md 11 on its per-platform digests |
| `golden_weights.json` | generated — the synthetic backtest fixture | — | not third-party |
