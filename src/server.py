#!/usr/bin/env python3
"""
RGM MCP Server
==============
MCP server exposing analytical tools for Revenue Growth Management (RGM)
in the Alco-Bev / North-America space.

Data is expected as flat CSV or Parquet files exported from NIQ / Nielsen.
All heavy lifting uses pandas + numpy + scipy.
"""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastmcp import FastMCP

mcp = FastMCP("rgm-rgm-server")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(path: str) -> pd.DataFrame:
    """Load a CSV or Parquet file into a DataFrame."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if p.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    return pd.read_csv(p, low_memory=False)


def _require_cols(df: pd.DataFrame, cols: list[str], file_label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{file_label} is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def _df_to_json(df: pd.DataFrame, max_rows: int = 2000) -> str:
    """Serialise a DataFrame to JSON, capping rows to avoid huge payloads."""
    if len(df) > max_rows:
        df = df.head(max_rows)
    return df.to_json(orient="records", date_format="iso", indent=2)


# ===========================================================================
# TOOL 1 — Build Analytical Base Table
# ===========================================================================

@mcp.tool()
def build_analytical_base_table(
    sales_file: str,
    distribution_file: str,
    pricing_file: str,
    output_file: str,
    date_col: str = "period_end_date",
    sku_col: str = "upc",
    market_col: str = "market",
    channel_col: str = "channel",
    volume_col: str = "unit_sales",
    dollar_col: str = "dollar_sales",
    price_col: str = "avg_price_per_unit",
    any_promo_col: str = "any_promo_flag",
    tdp_col: str = "total_distribution_points",
    competitor_brand_col: str = "competitor_brand",
    comp_price_col: str = "comp_avg_price",
) -> str:
    """
    Clean and join NIQ-exported flat files (sales, distribution, pricing) into a
    single model-ready Analytical Base Table (ABT) suitable for elasticity
    modelling, promo scoring, and competitive indexing.

    Parameters
    ----------
    sales_file            : Path to NIQ weekly/monthly POS sales file (CSV or Parquet).
    distribution_file     : Path to NIQ distribution / TDP file.
    pricing_file          : Path to NIQ pricing file (own + competitor).
    output_file           : Destination path for the output ABT (CSV or Parquet).
    date_col              : Column name for the time period.
    sku_col               : Column identifying the SKU / UPC / product.
    market_col            : Column identifying the retail market / geography.
    channel_col           : Column identifying the trade channel.
    volume_col            : Column holding unit volume sold.
    dollar_col            : Column holding dollar sales.
    price_col             : Column holding own avg price per unit.
    any_promo_col         : Column flagging any promotional activity (0/1 or bool).
    tdp_col               : Column holding total distribution points.
    competitor_brand_col  : Column identifying the competitor brand (in pricing file).
    comp_price_col        : Column holding competitor avg price.

    Returns
    -------
    JSON summary: row count, column list, date range, and first 10 rows of the ABT.
    """
    # --- Load ---
    sales = _load(sales_file)
    dist = _load(distribution_file)
    pricing = _load(pricing_file)

    join_keys = [k for k in [date_col, sku_col, market_col, channel_col] if k in sales.columns and k in dist.columns]

    _require_cols(sales, [date_col, sku_col, volume_col, dollar_col], "sales_file")
    _require_cols(dist, [tdp_col], "distribution_file")
    _require_cols(pricing, [comp_price_col], "pricing_file")

    # --- Normalise dates ---
    for df in (sales, dist, pricing):
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # --- Merge ---
    abt = sales.merge(dist[list(set(join_keys + [tdp_col]))], on=join_keys, how="left", suffixes=("", "_dist"))

    price_join_keys = [k for k in join_keys if k in pricing.columns]
    if price_col not in abt.columns and price_col in pricing.columns:
        abt = abt.merge(pricing[list(set(price_join_keys + [price_col]))], on=price_join_keys, how="left", suffixes=("", "_price"))

    # Pivot competitor pricing to wide form: one column per competitor brand
    comp_keys = [k for k in price_join_keys if k != competitor_brand_col]
    if competitor_brand_col in pricing.columns and comp_price_col in pricing.columns:
        comp_wide = (
            pricing[comp_keys + [competitor_brand_col, comp_price_col]]
            .drop_duplicates()
            .pivot_table(index=comp_keys, columns=competitor_brand_col, values=comp_price_col, aggfunc="mean")
        )
        comp_wide.columns = [f"comp_price_{c}" for c in comp_wide.columns]
        comp_wide = comp_wide.reset_index()
        abt = abt.merge(comp_wide, on=comp_keys, how="left")

    # --- Derived columns ---
    if price_col in abt.columns and volume_col in abt.columns:
        abt["revenue"] = abt[dollar_col]
        abt["price_per_unit"] = abt[price_col]
        abt["log_volume"] = np.log1p(abt[volume_col].clip(lower=0))
        abt["log_price"] = np.log1p(abt[price_col].clip(lower=0))

    if tdp_col in abt.columns:
        abt["log_tdp"] = np.log1p(abt[tdp_col].clip(lower=0))

    # --- Clean ---
    abt = abt.drop_duplicates()
    abt = abt.sort_values([date_col] + ([market_col] if market_col in abt.columns else []))

    # --- Save ---
    out = Path(output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() in {".parquet", ".pq"}:
        abt.to_parquet(out, index=False)
    else:
        abt.to_csv(out, index=False)

    date_min = abt[date_col].min() if date_col in abt.columns else "n/a"
    date_max = abt[date_col].max() if date_col in abt.columns else "n/a"

    return json.dumps({
        "status": "success",
        "rows": len(abt),
        "columns": list(abt.columns),
        "date_range": {"from": str(date_min), "to": str(date_max)},
        "output_file": str(out),
        "preview": json.loads(_df_to_json(abt.head(10))),
    }, indent=2)


# ===========================================================================
# TOOL 2 — Calculate Price Elasticity
# ===========================================================================

@mcp.tool()
def calculate_price_elasticity(
    abt_file: str,
    group_by: list[str] | None = None,
    log_volume_col: str = "log_volume",
    log_price_col: str = "log_price",
    log_tdp_col: str = "log_tdp",
    promo_col: str = "any_promo_flag",
    cross_price_cols: list[str] | None = None,
    min_observations: int = 20,
) -> str:
    """
    Estimate own-price and cross-price elasticities from the ABT using an
    OLS log-log regression (ln Volume ~ ln Price + controls).

    The own-price elasticity coefficient is the slope on ln(Price). It is
    typically negative: e.g. -2.3 means a 1% price increase → 2.3% volume drop.

    Parameters
    ----------
    abt_file           : Path to the ABT produced by build_analytical_base_table.
    group_by           : List of columns to segment by (e.g. ["market","channel"]).
                         If None, a single aggregate model is fit.
    log_volume_col     : Log-transformed volume column.
    log_price_col      : Log-transformed own price column.
    log_tdp_col        : Log-transformed TDP column (distribution control).
    promo_col          : Promo flag column (0/1 control variable).
    cross_price_cols   : List of log-transformed competitor price columns.
    min_observations   : Minimum rows per segment to fit a model.

    Returns
    -------
    JSON: elasticity estimates per segment with R², p-values, and observation count.
    """
    from scipy import stats

    abt = _load(abt_file)
    _require_cols(abt, [log_volume_col, log_price_col], "abt_file")

    cross_cols = [c for c in (cross_price_cols or []) if c in abt.columns]
    control_cols = []
    if log_tdp_col in abt.columns:
        control_cols.append(log_tdp_col)
    if promo_col in abt.columns:
        abt[promo_col] = pd.to_numeric(abt[promo_col], errors="coerce").fillna(0)
        control_cols.append(promo_col)
    control_cols += cross_cols

    def _fit_segment(df: pd.DataFrame, label: str) -> dict[str, Any]:
        df = df[[log_volume_col, log_price_col] + control_cols].dropna()
        n = len(df)
        if n < min_observations:
            return {"segment": label, "status": f"insufficient_data (n={n})"}

        y = df[log_volume_col].values
        X_cols = [log_price_col] + control_cols
        X = np.column_stack([df[c].values for c in X_cols])
        X = np.column_stack([np.ones(n), X])  # intercept

        # OLS: (X'X)^-1 X'y
        try:
            coeffs, residuals_ss, rank, sv = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError as e:
            return {"segment": label, "status": f"linalg_error: {e}"}

        y_hat = X @ coeffs
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        # p-values via t-distribution
        dof = n - X.shape[1]
        mse = ss_res / max(dof, 1)
        try:
            XtX_inv = np.linalg.inv(X.T @ X)
            se = np.sqrt(np.diag(XtX_inv) * mse)
            t_stats = coeffs / se
            p_vals = 2 * stats.t.sf(np.abs(t_stats), df=dof)
        except np.linalg.LinAlgError:
            se = p_vals = [float("nan")] * len(coeffs)

        feature_names = ["intercept", log_price_col] + control_cols
        coeff_map = {
            name: {
                "estimate": round(float(coeffs[i]), 4),
                "p_value": round(float(p_vals[i]), 4) if not np.isnan(p_vals[i]) else None,
            }
            for i, name in enumerate(feature_names)
        }

        cross_elasticities = {
            c: coeff_map[c] for c in cross_cols if c in coeff_map
        }

        return {
            "segment": label,
            "n_observations": n,
            "own_price_elasticity": round(float(coeffs[1]), 4),
            "own_price_p_value": round(float(p_vals[1]), 4),
            "r_squared": round(r2, 4),
            "cross_price_elasticities": cross_elasticities,
            "all_coefficients": coeff_map,
            "interpretation": (
                f"A 1% increase in price is associated with a "
                f"{abs(round(float(coeffs[1]), 2))}% {'decrease' if coeffs[1] < 0 else 'increase'} in volume."
            ),
        }

    results = []
    if group_by:
        valid_group = [c for c in group_by if c in abt.columns]
        if valid_group:
            for keys, grp in abt.groupby(valid_group):
                label = " | ".join(f"{k}={v}" for k, v in zip(valid_group, (keys if isinstance(keys, tuple) else (keys,))))
                results.append(_fit_segment(grp, label))
        else:
            results.append(_fit_segment(abt, "aggregate"))
    else:
        results.append(_fit_segment(abt, "aggregate"))

    return json.dumps({"elasticity_results": results}, indent=2)


# ===========================================================================
# TOOL 3 — Score Promo Effectiveness
# ===========================================================================

@mcp.tool()
def score_promo_effectiveness(
    abt_file: str,
    date_col: str = "period_end_date",
    sku_col: str = "upc",
    market_col: str = "market",
    volume_col: str = "unit_sales",
    dollar_col: str = "dollar_sales",
    promo_col: str = "any_promo_flag",
    cost_col: str = "trade_spend",
    group_by: list[str] | None = None,
    baseline_window_weeks: int = 4,
) -> str:
    """
    Score the effectiveness of past promotional events.

    Methodology
    -----------
    - Baseline volume = rolling average of non-promoted weeks in the preceding window.
    - Incremental volume = promoted volume − baseline volume.
    - Promo lift % = incremental / baseline × 100.
    - Incremental revenue = incremental volume × avg price during promo.
    - Promo ROI = (incremental revenue − trade spend) / trade spend  [if cost_col present].

    Parameters
    ----------
    abt_file              : Path to the ABT.
    date_col              : Time period column.
    sku_col               : SKU / UPC column.
    market_col            : Market / geography column.
    volume_col            : Unit volume column.
    dollar_col            : Dollar sales column.
    promo_col             : Promo flag column (1 = promoted week).
    cost_col              : Trade spend / promo cost column (optional; used for ROI).
    group_by              : Segment columns for aggregated scoring. Defaults to [sku_col, market_col].
    baseline_window_weeks : Number of prior non-promo weeks to average for baseline.

    Returns
    -------
    JSON: per-segment promo event summary with lift, incremental volume, and ROI.
    """
    abt = _load(abt_file)
    _require_cols(abt, [date_col, volume_col, promo_col], "abt_file")

    abt[date_col] = pd.to_datetime(abt[date_col], errors="coerce")
    abt[promo_col] = pd.to_numeric(abt[promo_col], errors="coerce").fillna(0).astype(int)

    dims = group_by if group_by else [c for c in [sku_col, market_col] if c in abt.columns]
    abt_sorted = abt.sort_values([*dims, date_col] if dims else [date_col])

    records = []

    def _score_group(df: pd.DataFrame, label: str) -> None:
        df = df.sort_values(date_col).reset_index(drop=True)
        non_promo_vol = df.loc[df[promo_col] == 0, volume_col]

        # Rolling baseline: mean of last N non-promo weeks
        baseline_mean = non_promo_vol.rolling(baseline_window_weeks, min_periods=1).mean().iloc[-1] if len(non_promo_vol) > 0 else float("nan")

        promo_weeks = df[df[promo_col] == 1]
        if promo_weeks.empty:
            return

        total_promo_vol = promo_weeks[volume_col].sum()
        n_promo_weeks = len(promo_weeks)
        avg_promo_vol = total_promo_vol / n_promo_weeks if n_promo_weeks else 0

        incremental_vol = (avg_promo_vol - baseline_mean) * n_promo_weeks
        lift_pct = (avg_promo_vol / baseline_mean - 1) * 100 if baseline_mean and baseline_mean > 0 else float("nan")

        avg_price = df.loc[df[promo_col] == 1, dollar_col].sum() / total_promo_vol if dollar_col in df.columns and total_promo_vol > 0 else float("nan")
        incremental_revenue = incremental_vol * avg_price if not np.isnan(avg_price) else float("nan")

        roi = float("nan")
        if cost_col in df.columns:
            total_spend = promo_weeks[cost_col].sum()
            if total_spend > 0 and not np.isnan(incremental_revenue):
                roi = (incremental_revenue - total_spend) / total_spend

        records.append({
            "segment": label,
            "n_promo_weeks": n_promo_weeks,
            "baseline_volume_per_week": round(float(baseline_mean), 2) if not np.isnan(baseline_mean) else None,
            "avg_promo_volume_per_week": round(float(avg_promo_vol), 2),
            "incremental_volume": round(float(incremental_vol), 2),
            "promo_lift_pct": round(float(lift_pct), 2) if not np.isnan(lift_pct) else None,
            "incremental_revenue": round(float(incremental_revenue), 2) if not np.isnan(incremental_revenue) else None,
            "promo_roi": round(float(roi), 4) if not np.isnan(roi) else None,
            "verdict": (
                "effective" if (not np.isnan(lift_pct) and lift_pct > 10) else
                "marginal" if (not np.isnan(lift_pct) and lift_pct > 0) else
                "ineffective"
            ),
        })

    if dims:
        for keys, grp in abt_sorted.groupby(dims):
            label = " | ".join(str(v) for v in (keys if isinstance(keys, tuple) else (keys,)))
            _score_group(grp, label)
    else:
        _score_group(abt_sorted, "aggregate")

    return json.dumps({"promo_scores": records, "total_segments": len(records)}, indent=2)


# ===========================================================================
# TOOL 4 — Recommend Dynamic Pricing
# ===========================================================================

@mcp.tool()
def recommend_dynamic_pricing(
    elasticity_json: str,
    current_prices: dict[str, float],
    margin_targets: dict[str, float],
    cogs: dict[str, float],
    max_price_change_pct: float = 10.0,
    price_step_pct: float = 0.5,
) -> str:
    """
    Recommend optimal price points per segment / SKU given price elasticities
    and margin targets.

    Uses a simple grid-search over ±max_price_change_pct in price_step_pct increments
    to find the price that maximises revenue while meeting the margin floor.

    Parameters
    ----------
    elasticity_json      : JSON string from calculate_price_elasticity (or path to file).
    current_prices       : Dict mapping segment label → current price per unit (USD).
    margin_targets       : Dict mapping segment label → minimum gross margin % (0–100).
    cogs                 : Dict mapping segment label → cost of goods sold per unit (USD).
    max_price_change_pct : Maximum allowed price move in either direction (%).
    price_step_pct       : Granularity of the price grid search (%).

    Returns
    -------
    JSON: per-segment recommended price, expected volume change, and projected revenue uplift.
    """
    # Accept either a JSON string or a file path
    if elasticity_json.strip().startswith("{") or elasticity_json.strip().startswith("["):
        elast_data = json.loads(elasticity_json)
    else:
        with open(elasticity_json) as f:
            elast_data = json.load(f)

    results_raw = elast_data.get("elasticity_results", [elast_data])

    recommendations = []
    steps = np.arange(-max_price_change_pct, max_price_change_pct + price_step_pct, price_step_pct)

    for rec in results_raw:
        segment = rec.get("segment", "unknown")
        elasticity = rec.get("own_price_elasticity")
        if elasticity is None or rec.get("status"):
            recommendations.append({"segment": segment, "status": "skipped — no valid elasticity"})
            continue

        p0 = current_prices.get(segment, current_prices.get("default"))
        cogs_val = cogs.get(segment, cogs.get("default"))
        margin_target = margin_targets.get(segment, margin_targets.get("default", 0))

        if p0 is None or cogs_val is None:
            recommendations.append({"segment": segment, "status": "skipped — missing price or COGS"})
            continue

        best = {"delta_pct": 0.0, "revenue_index": 1.0, "margin_pct": (p0 - cogs_val) / p0 * 100}

        for delta in steps:
            p1 = p0 * (1 + delta / 100)
            if p1 <= cogs_val:
                continue
            margin_pct = (p1 - cogs_val) / p1 * 100
            if margin_pct < margin_target:
                continue
            # Volume index relative to current: (p1/p0)^elasticity
            vol_index = (p1 / p0) ** elasticity
            rev_index = (p1 / p0) * vol_index  # price × volume relative change
            if rev_index > best["revenue_index"]:
                best = {"delta_pct": round(delta, 2), "revenue_index": round(rev_index, 4), "margin_pct": round(margin_pct, 2)}

        recommended_price = round(p0 * (1 + best["delta_pct"] / 100), 4)
        vol_change_pct = round(((recommended_price / p0) ** elasticity - 1) * 100, 2)

        recommendations.append({
            "segment": segment,
            "current_price": round(p0, 4),
            "recommended_price": recommended_price,
            "price_change_pct": best["delta_pct"],
            "expected_volume_change_pct": vol_change_pct,
            "expected_revenue_index": best["revenue_index"],
            "projected_gross_margin_pct": best["margin_pct"],
            "rationale": (
                f"At elasticity {elasticity}, raising price by {best['delta_pct']}% yields "
                f"a {round((best['revenue_index']-1)*100, 2)}% revenue uplift while "
                f"maintaining a {best['margin_pct']}% gross margin."
            ),
        })

    return json.dumps({"pricing_recommendations": recommendations}, indent=2)


# ===========================================================================
# TOOL 5 — Optimise Promo Calendar
# ===========================================================================

@mcp.tool()
def optimize_promo_calendar(
    promo_scores_json: str,
    budget: float,
    weeks_available: list[str],
    sku_list: list[str] | None = None,
    market_list: list[str] | None = None,
    avg_cost_per_promo_event: float | None = None,
    min_roi_threshold: float = 0.0,
    max_events_per_sku: int = 4,
) -> str:
    """
    Recommend an optimised promotional calendar by greedily selecting the
    highest-ROI promo events within budget and capacity constraints.

    Parameters
    ----------
    promo_scores_json         : JSON string (or path) from score_promo_effectiveness.
    budget                    : Total trade-spend budget available (USD).
    weeks_available           : List of ISO week strings to schedule into (e.g. ["2025-W01", ...]).
    sku_list                  : Restrict optimisation to these SKUs (optional).
    market_list               : Restrict to these markets (optional).
    avg_cost_per_promo_event  : Cost per event if trade-spend data is absent in scores.
    min_roi_threshold         : Exclude events with ROI below this value.
    max_events_per_sku        : Maximum number of promos per SKU across the calendar.

    Returns
    -------
    JSON: recommended schedule (week, SKU, market, expected lift, estimated spend),
    total spend, total expected incremental revenue, and calendar-level ROI.
    """
    if promo_scores_json.strip().startswith("{") or promo_scores_json.strip().startswith("["):
        data = json.loads(promo_scores_json)
    else:
        with open(promo_scores_json) as f:
            data = json.load(f)

    scores = data.get("promo_scores", [data]) if isinstance(data, dict) else data

    # Filter
    if sku_list:
        scores = [s for s in scores if any(sku in s.get("segment", "") for sku in sku_list)]
    if market_list:
        scores = [s for s in scores if any(mkt in s.get("segment", "") for mkt in market_list)]

    # Remove non-effective / below threshold
    candidates = [
        s for s in scores
        if s.get("promo_roi") is not None and s["promo_roi"] >= min_roi_threshold
    ]
    # Sort by ROI descending
    candidates.sort(key=lambda x: x["promo_roi"], reverse=True)

    default_event_cost = avg_cost_per_promo_event or (budget / max(len(candidates), 1))

    schedule = []
    remaining_budget = budget
    sku_event_count: dict[str, int] = {}
    total_incremental_revenue = 0.0
    total_spend = 0.0

    week_pool = list(weeks_available)

    for candidate in candidates:
        if not week_pool:
            break
        if remaining_budget <= 0:
            break

        segment = candidate["segment"]
        event_cost = default_event_cost
        if event_cost > remaining_budget:
            continue

        # Count events per SKU
        sku_event_count[segment] = sku_event_count.get(segment, 0)
        if sku_event_count[segment] >= max_events_per_sku:
            continue

        week = week_pool.pop(0)
        inc_rev = candidate.get("incremental_revenue") or (
            candidate.get("incremental_volume", 0) * 1.0  # fallback — 1 USD/unit if no rev data
        )

        schedule.append({
            "week": week,
            "segment": segment,
            "expected_lift_pct": candidate.get("promo_lift_pct"),
            "expected_incremental_volume": candidate.get("incremental_volume"),
            "expected_incremental_revenue": round(float(inc_rev), 2),
            "estimated_trade_spend": round(event_cost, 2),
            "expected_roi": candidate["promo_roi"],
        })

        sku_event_count[segment] += 1
        remaining_budget -= event_cost
        total_incremental_revenue += inc_rev
        total_spend += event_cost

    calendar_roi = (total_incremental_revenue - total_spend) / total_spend if total_spend > 0 else float("nan")

    return json.dumps({
        "promo_calendar": schedule,
        "total_events_scheduled": len(schedule),
        "total_estimated_spend": round(total_spend, 2),
        "total_expected_incremental_revenue": round(total_incremental_revenue, 2),
        "calendar_roi": round(calendar_roi, 4) if not np.isnan(calendar_roi) else None,
        "remaining_budget": round(remaining_budget, 2),
    }, indent=2)


# ===========================================================================
# TOOL 6 — Compute Competitive Price Index
# ===========================================================================

@mcp.tool()
def compute_competitive_price_index(
    abt_file: str,
    own_brand: str,
    brand_col: str = "brand",
    category_col: str = "category",
    pack_size_col: str = "pack_size",
    market_col: str = "market",
    date_col: str = "period_end_date",
    price_col: str = "avg_price_per_unit",
    volume_col: str = "unit_sales",
    group_by: list[str] | None = None,
) -> str:
    """
    Compute a Competitive Price Index (CPI) for the own brand versus all other
    brands in the dataset.

    CPI = (own brand avg price) / (weighted avg competitor price) × 100

    A CPI > 100 means the own brand is priced above the competitive set.
    A CPI < 100 means it is priced below.

    Parameters
    ----------
    abt_file      : Path to the ABT (must contain brand and price columns).
    own_brand     : The brand name to treat as the focal brand.
    brand_col     : Column identifying brand.
    category_col  : Column identifying category.
    pack_size_col : Column identifying pack size / volume format.
    market_col    : Column identifying market / retailer geography.
    date_col      : Time period column.
    price_col     : Price per unit column.
    volume_col    : Volume column (used for weighting competitor average).
    group_by      : Extra dimensions to slice CPI by (e.g. ["channel"]).

    Returns
    -------
    JSON: CPI per segment with own price, weighted competitor price, price gap (USD),
    and a positioning summary.
    """
    abt = _load(abt_file)
    _require_cols(abt, [brand_col, price_col], "abt_file")

    abt[date_col] = pd.to_datetime(abt[date_col], errors="coerce") if date_col in abt.columns else abt[date_col]

    dims = [c for c in ([category_col, pack_size_col, market_col] + (group_by or [])) if c in abt.columns]

    own = abt[abt[brand_col] == own_brand].copy()
    comp = abt[abt[brand_col] != own_brand].copy()

    if own.empty:
        return json.dumps({"error": f"No rows found for brand='{own_brand}' in {brand_col} column."})

    # Weighted avg competitor price per segment
    if volume_col in comp.columns:
        comp["_weighted_price"] = comp[price_col] * comp[volume_col]
        comp_agg = comp.groupby(dims).apply(
            lambda g: g["_weighted_price"].sum() / g[volume_col].sum() if g[volume_col].sum() > 0 else np.nan,
            include_groups=False
        ).reset_index(name="comp_wavg_price")
    else:
        comp_agg = comp.groupby(dims)[price_col].mean().reset_index()
        comp_agg.rename(columns={price_col: "comp_wavg_price"}, inplace=True)

    own_agg = own.groupby(dims)[price_col].mean().reset_index()
    own_agg.rename(columns={price_col: "own_avg_price"}, inplace=True)

    merged = own_agg.merge(comp_agg, on=dims, how="left")
    merged["cpi"] = (merged["own_avg_price"] / merged["comp_wavg_price"] * 100).round(2)
    merged["price_gap_usd"] = (merged["own_avg_price"] - merged["comp_wavg_price"]).round(4)
    merged["positioning"] = merged["cpi"].apply(
        lambda x: "premium" if x > 110 else ("at-par" if 90 <= x <= 110 else "value") if not np.isnan(x) else "unknown"
    )

    result_df = merged.sort_values(dims)
    summary = json.loads(_df_to_json(result_df))

    return json.dumps({
        "own_brand": own_brand,
        "n_segments": len(result_df),
        "avg_cpi": round(float(merged["cpi"].mean()), 2) if not merged["cpi"].isna().all() else None,
        "segments": summary,
    }, indent=2)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    mcp.run(transport="stdio")
