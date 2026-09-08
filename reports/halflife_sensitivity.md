# Half-life sensitivity, and what it says about `K/T`

Generated 2026-09-03 by `python -m mafrm.factors.halflife_report`. SPEC.md 5.1, registered in SPEC.md 5.1.3.

SPEC.md 5.1: *"Half-life selection is the main dial and there is no right answer ...
ship a sensitivity chart of bias statistic and realised min-var portfolio volatility
across half-lives from 21 to 504 days."* This is that chart, and it turned out to be
more than that.

## This is a `K/T` experiment

`T_eff = 2*tau/ln 2`, so sweeping the volatility half-life over 24x sweeps `K/T_eff`
over 24x -- 0.0041 to 0.0990 -- **on one panel, with `K`, `N`, the
asset class, the data source, the factor construction and the scored window all held
fixed.**
Shepard's predicted understatement -- variance multiplier minus one, SPEC.md 6.4's table convention; the volatility figure is about half -- runs 0.8% to 23.2%
across it.

SPEC.md 12's week-8 deliverable plots measured optimizer-portfolio bias against `K/T`
with Shepard's `[1 - K/T]^-2` overlaid. Until this session the second end of that range
depended entirely on W7's equity module arriving. It no longer does -- and a
within-panel range is arguably the cleaner evidence, because a macro-versus-equity
comparison moves `K`, `N`, the asset class and the input data at once while this moves
one number. W7 remains the wider range and the harder test; it is no longer the only
one.

## What is swept, and what is NOT being selected

The **factor volatility** half-life, alone. The correlation half-life is held at 504 at
both horizons, because CLAUDE.md's parameter table pins it identically across horizons
and marks that deliberate: it is not a dial in this model.

**The shipped values are unchanged: 84d at the short horizon and 252d at the long one**, the published USE4 constants, fixed before this
grid existed. They are points **on** the curve below, computed by the same code on the
same scored window, not a separate run beside it -- `assert_shipped_on_grid` refuses to
render otherwise. No result here may change which value ships; the moment one does,
`experiments.md` rows 165-173 become `model-config` and count toward the deflated
Sharpe trial count retroactively and in full. The sweep is reported across its whole
range rather than near the shipped points, so nobody can claim the published values
were confirmed by a curve that only examined its own neighbourhood.

## The scored window, common to all nine points

SPEC.md 6.2.4's *"the window is common to every variant"* extended one step. A curve of
`B` against `tau` scored on nine different date sets would be partly a curve of `B`
against **sample**.

- **Common window: 3,874 dates, 2009-05-07 to 2024-12-31.**
- Exact chi-square 95% interval at `T = 3,874`: [0.9777, 1.0223] (chi2, T=3874). Normal band (display convention, SPEC.md 6.1.1): [0.9685, 1.0315] (normal, T=3874).

| `tau` | own scored dates | own window starts | dropped to reach the common window |
|---|---|---|---|
| 21 | 3,874 | 2009-05-07 | 0 |
| 42 | 3,874 | 2009-05-07 | 0 |
| 63 | 3,874 | 2009-05-07 | 0 |
| 84 | 3,874 | 2009-05-07 | 0 |
| 126 | 3,874 | 2009-05-07 | 0 |
| 168 | 3,874 | 2009-05-07 | 0 |
| 252 | 3,874 | 2009-05-07 | 0 |
| 336 | 3,874 | 2009-05-07 | 0 |
| 504 | 3,874 | 2009-05-07 | 0 |

**CLAUDE.md failure mode 9.** Every `B` below is a full-sample statistic over the
3,874 common dates with its exact interval at that `T`; none is a rolling
window and no count of windows is quoted as an `n`. The nine histories are expanding
windows and consecutive dates share all but one observation, which is why the
uncertainty quoted is the chi-square interval on the scored sample and not a spread
across the grid.

## `short` horizon shape (SPEC.md 5.5 specific half-life 84d)

Pre-VRA, `a = 1.0` (SPEC.md 6.2.2). `B` is the median across each family's
members; min-var volatility is in bps/day and is SPEC.md 6.5's headline
model-comparison metric.

| `tau` | `K/T_eff` | Shepard `K` (var) | Shepard `N` (var) | fam1 | fam2 | fam3 | **fam4** | **fam4-fam2** | min-var vol | naive vol |
|---|---|---|---|---|---|---|---|---|---|---|
| 21 | 0.0990 | 23.2% | 62.1% | 1.0580 | 1.0301 | 1.0982 | **1.3285** | **+0.2984** | 5.195 | 4.728 |
| 42 | 0.0495 | 10.7% | 25.5% | 1.0410 | 1.0196 | 1.0599 | **1.3305** | **+0.3109** | 5.206 | 4.583 |
| 63 | 0.0330 | 6.9% | 16.0% | 1.0293 | 1.0141 | 1.0474 | **1.3316** | **+0.3175** | 5.203 | 4.562 |
| 84 **(shipped)** | 0.0248 | 5.1% | 11.7% | 1.0245 | 1.0101 | 1.0415 | **1.3321** | **+0.3220** | 5.196 | 4.570 |
| 126 | 0.0165 | 3.4% | 7.6% | 1.0194 | 1.0056 | 1.0338 | **1.3324** | **+0.3269** | 5.178 | 4.604 |
| 168 | 0.0124 | 2.5% | 5.6% | 1.0178 | 1.0027 | 1.0297 | **1.3323** | **+0.3295** | 5.162 | 4.639 |
| 252 | 0.0083 | 1.7% | 3.7% | 1.0143 | 1.0011 | 1.0266 | **1.3319** | **+0.3308** | 5.135 | 4.688 |
| 336 | 0.0062 | 1.2% | 2.7% | 1.0120 | 1.0002 | 1.0263 | **1.3319** | **+0.3317** | 5.118 | 4.723 |
| 504 | 0.0041 | 0.8% | 1.8% | 1.0095 | 1.0003 | 1.0277 | **1.3321** | **+0.3318** | 5.100 | 4.779 |

## `long` horizon shape (SPEC.md 5.5 specific half-life 252d)

Pre-VRA, `a = 1.0` (SPEC.md 6.2.2). `B` is the median across each family's
members; min-var volatility is in bps/day and is SPEC.md 6.5's headline
model-comparison metric.

| `tau` | `K/T_eff` | Shepard `K` (var) | Shepard `N` (var) | fam1 | fam2 | fam3 | **fam4** | **fam4-fam2** | min-var vol | naive vol |
|---|---|---|---|---|---|---|---|---|---|---|
| 21 | 0.0990 | 23.2% | 62.1% | 1.0392 | 1.0233 | 1.0982 | **1.3131** | **+0.2898** | 5.221 | 4.728 |
| 42 | 0.0495 | 10.7% | 25.5% | 1.0301 | 1.0132 | 1.0599 | **1.3144** | **+0.3012** | 5.228 | 4.583 |
| 63 | 0.0330 | 6.9% | 16.0% | 1.0213 | 1.0081 | 1.0474 | **1.3150** | **+0.3070** | 5.224 | 4.562 |
| 84 | 0.0248 | 5.1% | 11.7% | 1.0200 | 1.0049 | 1.0415 | **1.3152** | **+0.3103** | 5.217 | 4.570 |
| 126 | 0.0165 | 3.4% | 7.6% | 1.0182 | 1.0012 | 1.0338 | **1.3150** | **+0.3138** | 5.200 | 4.604 |
| 168 | 0.0124 | 2.5% | 5.6% | 1.0158 | 0.9991 | 1.0297 | **1.3145** | **+0.3154** | 5.184 | 4.639 |
| 252 **(shipped)** | 0.0083 | 1.7% | 3.7% | 1.0127 | 0.9987 | 1.0266 | **1.3135** | **+0.3148** | 5.157 | 4.688 |
| 336 | 0.0062 | 1.2% | 2.7% | 1.0116 | 0.9987 | 1.0263 | **1.3127** | **+0.3141** | 5.136 | 4.723 |
| 504 | 0.0041 | 0.8% | 1.8% | 1.0110 | 0.9994 | 1.0277 | **1.3119** | **+0.3125** | 5.109 | 4.779 |

