"""Statistical forecasting: baseline/promo decomposition, Holt-Winters baseline, promo overlay."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

DEFAULT_ELASTICITY = 2.5  # lift per 100% discount depth when no promo history exists
MAX_ELASTICITY = 6.0


def _series(g: pd.DataFrame) -> pd.DataFrame:
    g = g.set_index("Week_End_Date").sort_index()
    idx = pd.date_range(g.index.min(), g.index.max(), freq="W-SUN")
    return g.reindex(idx)


def decompose_baseline(g: pd.DataFrame) -> pd.Series:
    """Promo-free baseline: promo weeks and non-promo outliers are replaced by interpolation."""
    g = _series(g)
    units = g["Units_Sold_POS"].astype(float)
    promo = g["Promo_Flag"].fillna(0).astype(bool)
    base = units.mask(promo)
    # Non-promo outlier scrub (e.g. scan errors): clip to 2.5x centred rolling median
    med = base.rolling(9, center=True, min_periods=3).median()
    base = base.where(~(base > 2.5 * med), np.nan)
    return base.interpolate(limit_direction="both").clip(lower=0)


def promo_lift_table(pos: pd.DataFrame) -> pd.DataFrame:
    """Per SKU/retailer: promo lift = (promo units - base units) / base units, plus lift per unit of depth."""
    rows = []
    for (sku, retailer), g in pos.groupby(["SKU_ID", "Retailer"]):
        base = decompose_baseline(g)
        s = _series(g)
        promo = s["Promo_Flag"].fillna(0).astype(bool)
        lift, depth = np.nan, np.nan
        if promo.sum() > 0 and base[promo].sum() > 0:
            lift = s.loc[promo, "Units_Sold_POS"].sum() / base[promo].sum() - 1
            depth = s.loc[promo, "Promo_Depth"].mean()
        rows.append({
            "SKU_ID": sku, "Retailer": retailer, "Promo_Weeks": int(promo.sum()),
            "Avg_Promo_Depth": depth, "Promo_Lift": lift,
            "Elasticity": lift / depth if depth and depth > 0 and lift == lift else np.nan,
        })
    out = pd.DataFrame(rows)
    sku_median = out.groupby("SKU_ID")["Elasticity"].transform("median")
    out["Elasticity"] = out["Elasticity"].fillna(sku_median).fillna(DEFAULT_ELASTICITY).clip(0, MAX_ELASTICITY)
    return out


def _holt_winters(y: pd.Series, horizon: int) -> np.ndarray:
    y = y.asfreq("W-SUN").interpolate()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            seasonal = "add" if len(y) >= 104 else None
            model = ExponentialSmoothing(
                y, trend="add", damped_trend=True, seasonal=seasonal,
                seasonal_periods=52 if seasonal else None, initialization_method="estimated",
            ).fit()
            fc = model.forecast(horizon).to_numpy()
        if np.isfinite(fc).all():
            return np.clip(fc, 0, None)
    except Exception:
        pass
    return np.full(horizon, float(y.tail(8).mean()))


def fit_baselines(pos: pd.DataFrame, horizon: int = 12) -> pd.DataFrame:
    """Baseline (promo-free) forecast per SKU/retailer/week."""
    rows = []
    for (sku, retailer), g in pos.groupby(["SKU_ID", "Retailer"]):
        base = decompose_baseline(g)
        fc = _holt_winters(base, horizon)
        dates = pd.date_range(base.index[-1] + pd.Timedelta(days=7), periods=horizon, freq="W-SUN")
        rows.append(pd.DataFrame({
            "SKU_ID": sku, "Retailer": retailer, "Week_End_Date": dates,
            "Week_No": np.arange(1, horizon + 1), "Baseline_Units": fc,
        }))
    return pd.concat(rows, ignore_index=True)


def apply_promos(baseline: pd.DataFrame, lift: pd.DataFrame, promo_cal: pd.DataFrame,
                 lift_adj: float = 0.0) -> pd.DataFrame:
    """Overlay scheduled promos. lift_adj scales the promo uplift (e.g. +0.2 => lift x 1.2)."""
    df = baseline.merge(lift[["SKU_ID", "Retailer", "Elasticity"]], on=["SKU_ID", "Retailer"], how="left")
    cal = promo_cal[["SKU_ID", "Retailer", "Week_End_Date", "Promo_Depth_Pct"]].copy() if len(promo_cal) else \
        pd.DataFrame(columns=["SKU_ID", "Retailer", "Week_End_Date", "Promo_Depth_Pct"])
    cal = cal.groupby(["SKU_ID", "Retailer", "Week_End_Date"], as_index=False)["Promo_Depth_Pct"].max()
    df = df.merge(cal, on=["SKU_ID", "Retailer", "Week_End_Date"], how="left")
    df["Promo_Depth"] = df["Promo_Depth_Pct"].fillna(0) / 100
    df["Elasticity"] = df["Elasticity"].fillna(DEFAULT_ELASTICITY)
    df["Promo_Lift_Factor"] = df["Elasticity"] * df["Promo_Depth"] * (1 + lift_adj)
    df["Promo_Uplift_Units"] = df["Baseline_Units"] * df["Promo_Lift_Factor"]
    df["Forecast_Units"] = df["Baseline_Units"] + df["Promo_Uplift_Units"]
    return df.drop(columns=["Promo_Depth_Pct"])


def weekly_actuals(pos: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    keys = ["Week_End_Date"] + (by or [])
    return pos.groupby(keys, as_index=False)["Units_Sold_POS"].sum()
