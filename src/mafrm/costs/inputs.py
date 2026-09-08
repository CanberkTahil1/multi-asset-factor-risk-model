"""Assembles SPEC.md 7.1's per-asset inputs from the cache. Reads no network.

Two entry points, split on purpose:

- :func:`spread_shapes` needs NO issuer level. It builds every asset's clipped
  monthly EDGE series through its cost ticker (the ETF itself, or the tradable
  proxy of ``costs.tradable_proxies`` for a synthetic curve point), takes the
  in-sample median as the calm baseline and reports the widening. This is what
  ``reports/cost_calibration.md`` tabulates before any level exists.
- :func:`build_cost_inputs` needs every level. It refuses, by asset name, while
  a row of ``costs.spread_level.file`` is a placeholder, and there is no
  default (SPEC.md 3.4.1; :mod:`mafrm.costs.spread_level`).

Everything is monthly, stamped on each asset's own last traded date of the
month (:mod:`mafrm.data.calendar`) and aligned on the union without filling,
and everything stops strictly before ``sample.holdout_start`` -- the reads go
through :func:`mafrm.data.cache.read`, which truncates there, and the alignment
passes the boundary again as its half-open upper bound.

``sigma_i`` is the trailing ``costs.volatility_window`` standard deviation of
the asset's own daily TOTAL return (adjusted for dividends by
:func:`mafrm.data.prices.total_return`, which never looks forward), while the
spread and the dollar volume come from RAW bars. That asymmetry is SPEC.md
3.4's: a spread and a traded volume are properties of the prices that actually
traded; a return is not.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from mafrm import config as config_mod
from mafrm.costs import adv as adv_mod
from mafrm.costs import spread as spread_mod
from mafrm.costs import spread_level
from mafrm.data import cache, calendar, prices

__all__ = ["AssetShape", "CostInputs", "build_cost_inputs", "load_levels", "spread_shapes"]

_BPS = 1e-4


@dataclass(frozen=True)
class AssetShape:
    """One asset's EDGE shape statistics, before any level is attached."""

    asset: str
    ticker: str
    #: The proxy's caveat, or ``None`` for an asset that trades as itself.
    proxy_caveat: str | None
    #: Clipped monthly production series, proportional, in-sample only.
    production: pd.Series
    #: Incidence of negative estimates before the clip -- the resolution diagnostic.
    negatives: spread_mod.NegativeEstimates
    #: In-sample median of ``production``, proportional.
    baseline: float

    @property
    def widening(self) -> pd.Series:
        return spread_level.widening(self.production, self.baseline)

    @property
    def months(self) -> int:
        return int(self.production.count())

    @property
    def baseline_bps(self) -> float:
        return self.baseline / _BPS

    @property
    def max_widening_bps(self) -> float:
        return float(self.widening.max()) / _BPS

    @property
    def widening_months(self) -> int:
        """Months where EDGE sat above its baseline -- where the shape is non-zero."""
        return int((self.widening.dropna() > 0.0).sum())


def _load(ticker: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = cache.Manifest.load()
    key = ticker.lower()
    bars = cache.read(manifest.latest(source="yfinance", name=f"{key}_prices"), manifest=manifest)
    actions = cache.read(
        manifest.latest(source="yfinance", name=f"{key}_actions"), manifest=manifest
    )
    return bars, actions


def _cost_assets(cfg: config_mod.Config) -> dict[str, str]:
    """Asset id -> cost ticker for every universe member that carries a cost."""
    out: dict[str, str] = {}
    for asset in cfg.universe.assets:
        try:
            out[asset.id] = cfg.cost_ticker(asset.id)
        except config_mod.ConfigError:
            continue
    return out


def spread_shapes(cfg: config_mod.Config | None = None) -> dict[str, AssetShape]:
    """Every cost-bearing asset's clipped EDGE series and calm baseline. No levels."""
    cfg = cfg or config_mod.load()
    costs = cfg.model.costs
    edge = costs.edge_spread
    holdout = cfg.require_holdout_start()

    bars_by_asset: dict[str, pd.DataFrame] = {}
    for asset_id, ticker in _cost_assets(cfg).items():
        bars_by_asset[asset_id], _ = _load(ticker)

    panel = spread_mod.monthly_spread_panel(
        bars_by_asset,
        window=edge.window,
        min_periods=edge.min_periods,
        signed=edge.signed_estimates,
        clip=edge.clip_negative_to_zero,
        end=holdout,
    )
    if not edge.clip_negative_to_zero:
        raise spread_level.SpreadLevelError(
            "costs.edge_spread.clip_negative_to_zero is false; the calm baseline is defined on "
            "the CLIPPED production series (config/model.yaml, costs.spread_level)"
        )

    shapes: dict[str, AssetShape] = {}
    for asset_id, ticker in _cost_assets(cfg).items():
        production = panel.production[asset_id]
        proxy = costs.proxy_for(asset_id)
        shapes[asset_id] = AssetShape(
            asset=asset_id,
            ticker=ticker,
            proxy_caveat=None if proxy is None else proxy.caveat,
            production=production,
            negatives=panel.negatives[asset_id],
            baseline=spread_level.calm_baseline(production),
        )
    return shapes


@dataclass(frozen=True)
class CostInputs:
    """SPEC.md 7.1's inputs, monthly, (date x asset id), proportional units.

    ``half_spread_low`` and ``half_spread_high`` coincide for every asset with a
    point level and differ only where the level file gives a band; the cost is
    then reported as a range. ``adv_dollars`` is in dollars because NAV is the
    caller's -- :meth:`adv_over_nav` divides by it.
    """

    half_spread_low: pd.DataFrame
    half_spread_high: pd.DataFrame
    daily_volatility: pd.DataFrame
    adv_dollars: pd.DataFrame
    shapes: Mapping[str, AssetShape]
    levels: spread_level.SpreadLevels

    @property
    def has_band(self) -> bool:
        """True where any asset's two ends differ. ``NaN`` on both sides is not a band."""
        differs = (self.half_spread_low != self.half_spread_high) & self.half_spread_low.notna()
        return bool(differs.any().any())

    def adv_over_nav(self, nav_dollars: float) -> pd.DataFrame:
        if not np.isfinite(nav_dollars) or nav_dollars <= 0.0:
            raise spread_level.SpreadLevelError(f"nav_dollars must be positive, got {nav_dollars}")
        return self.adv_dollars / nav_dollars


def load_levels(cfg: config_mod.Config | None = None) -> spread_level.SpreadLevels:
    """The level file, read under ``costs.spread_level`` -- the one place it is opened."""
    cfg = cfg or config_mod.load()
    level_cfg = cfg.model.costs.spread_level
    return spread_level.load_spread_levels(
        config_mod._REPO_ROOT / level_cfg.file,
        expected=_cost_assets(cfg),
        display_resolution_bps=level_cfg.display_resolution_bps,
    )


def build_cost_inputs(cfg: config_mod.Config | None = None) -> CostInputs:
    """Every input the cost model needs, or a refusal naming the missing level."""
    cfg = cfg or config_mod.load()
    costs = cfg.model.costs
    holdout = cfg.require_holdout_start()
    assets = _cost_assets(cfg)

    levels = load_levels(cfg)
    # Refuse BEFORE reading any bars: the whole point is that no cost input
    # exists for an asset without a level, and building the rest first would
    # invite a caller to use a partial panel.
    for asset_id in assets:
        levels.require(asset_id)

    shapes = spread_shapes(cfg)

    low: dict[str, pd.Series] = {}
    high: dict[str, pd.Series] = {}
    vol: dict[str, pd.Series] = {}
    depth: dict[str, pd.Series] = {}
    for asset_id, ticker in assets.items():
        entry = levels.require(asset_id)
        bounds = entry.levels_bps
        shape = shapes[asset_id]
        series_low = spread_level.spread_series(
            level_bps=bounds[0], production=shape.production, commission=costs.commission
        )
        series_high = spread_level.spread_series(
            level_bps=bounds[-1], production=shape.production, commission=costs.commission
        )
        low[asset_id] = series_low.half
        high[asset_id] = series_high.half

        bars, actions = _load(ticker)
        returns = prices.total_return(bars, actions)
        window = costs.volatility_window
        daily_vol = returns.rolling(window=window, min_periods=window).std()
        sampled_vol = calendar.monthly(daily_vol, label=ticker)
        assert isinstance(sampled_vol, pd.Series)
        vol[asset_id] = sampled_vol
        dollar = adv_mod.average_daily_volume(bars, window=costs.adv.window, label=ticker)
        sampled_adv = calendar.monthly(dollar, label=ticker)
        assert isinstance(sampled_adv, pd.Series)
        depth[asset_id] = sampled_adv

    return CostInputs(
        half_spread_low=calendar.align(low, how="union", end=holdout),
        half_spread_high=calendar.align(high, how="union", end=holdout),
        daily_volatility=calendar.align(vol, how="union", end=holdout),
        adv_dollars=calendar.align(depth, how="union", end=holdout),
        shapes=shapes,
        levels=levels,
    )
