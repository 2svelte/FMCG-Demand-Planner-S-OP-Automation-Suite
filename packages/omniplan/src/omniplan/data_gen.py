"""Synthetic FMCG personal-care POS, master data and promo calendar generator."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

RETAILERS = {"Chemist Warehouse": 0.45, "Priceline": 0.25, "Coles": 0.30}

# id: (description, category, brand, price, unit cost, weekly units (all retailers),
#      seasonality amplitude, season phase, promo elasticity, cover weeks, lead time, MOQ, order multiple)
_SKUS = {
    "PC-1001": ("Hydrating Body Wash 500ml", "Personal Care", "AquaPure", 8.99, 3.90, 900, 0.10, 0.0, 3.0, 8, 4, 480, 48),
    "PC-1002": ("Daily Moisturiser SPF50 75ml", "Personal Care", "DermaDay", 16.99, 6.50, 400, 0.30, 1.2, 2.5, 1.5, 6, 240, 24),
    "PC-1003": ("Foaming Hand Wash 250ml", "Personal Care", "AquaPure", 4.99, 1.60, 1200, 0.05, 0.5, 2.0, 26, 3, 1200, 2400),
    "PC-1004": ("Intimate Wellness Gel 100ml", "Sexual Wellness", "Velvet", 14.99, 4.20, 300, 0.08, 2.0, 3.0, 7, 7, 100, 7),
    "PC-1005": ("Ultra-Thin Condoms 12pk", "Sexual Wellness", "Velvet", 12.99, 3.80, 650, 0.12, 3.0, 2.5, 3.0, 6, 600, 60),
    "PC-1006": ("Silky Personal Lubricant 50ml", "Sexual Wellness", "Velvet", 11.99, 3.30, 280, 0.08, 4.0, 3.0, 9, 6, None, 24),
    "PC-1007": ("Charcoal Face Wash 150ml", "Personal Care", "DermaDay", 9.99, 3.10, 520, 0.10, 0.8, 3.0, 8, 5, 360, 36),
    "PC-1008": ("Natural Deodorant Roll-On 50ml", "Personal Care", "AquaPure", 7.49, 2.40, 800, 0.15, 1.0, 2.0, 6, 5, 480, 48),
    "PC-1009": ("Sensitive Shampoo 400ml", "Personal Care", "DermaDay", 10.99, 4.00, 600, 0.05, 0.3, 2.5, 7, None, 360, 36),
    "PC-1010": ("SPF30 Sunscreen Lotion 200ml", "Personal Care", "SunSafe", 13.99, 5.00, 700, 0.45, 1.2, 2.5, 2.0, 8, 480, 48),
    "PC-1011": ("Lip Balm SPF15 Twin Pack", "Personal Care", "SunSafe", 5.99, 1.80, 950, 0.20, 1.0, 2.0, 30, 4, 1000, 100),
    "PC-1012": ("Hair Removal Cream 100ml", "Personal Care", "Velvet", 8.49, 2.90, 350, 0.18, 1.4, 2.5, 8, 5, 240, 24),
}

DEFAULT_END_DATE = date(2026, 10, 25)  # a Sunday: retailers report week-ending Sunday
HISTORY_WEEKS = 104
HORIZON_WEEKS = 12
SPIKE_SKU, SPIKE_RETAILER = "PC-1002", "Priceline"


def generate(seed: int = 42, end_date: date = DEFAULT_END_DATE):
    """Return (pos, master, promo_calendar) DataFrames."""
    rng = np.random.default_rng(seed)
    weeks = pd.date_range(end=pd.Timestamp(end_date), periods=HISTORY_WEEKS, freq="W-SUN")
    future_weeks = pd.date_range(weeks[-1] + pd.Timedelta(days=7), periods=HORIZON_WEEKS, freq="W-SUN")
    t = np.arange(HISTORY_WEEKS)
    woy = weeks.isocalendar().week.to_numpy().astype(float)

    pos_rows, promo_rows = [], []
    for sku, (desc, cat, _brand, price, _cost, wk, amp, phase, elast, cover, lt, _moq, _mult) in _SKUS.items():
        season = 1 + amp * np.sin(2 * np.pi * woy / 52 + phase)
        trend = 1 + 0.0012 * t
        for retailer, share in RETAILERS.items():
            promo = np.zeros(HISTORY_WEEKS, dtype=bool)
            depth = np.zeros(HISTORY_WEEKS)
            for start in rng.choice(np.arange(2, HISTORY_WEEKS - 6, 4), size=rng.integers(6, 9), replace=False):
                length = int(rng.integers(1, 3))
                promo[start:start + length] = True
                depth[start:start + length] = rng.choice([0.15, 0.20, 0.25, 0.30, 0.35])
            if sku == SPIKE_SKU and retailer == SPIKE_RETAILER:
                promo[-6:-2] = False
                depth[-6:-2] = 0.0

            base = wk * share * season * trend * rng.lognormal(0, 0.07, HISTORY_WEEKS)
            lift = 1 + elast * depth * rng.lognormal(0, 0.1, HISTORY_WEEKS)
            units = np.round(base * np.where(promo, lift, 1.0))
            if sku == SPIKE_SKU and retailer == SPIKE_RETAILER:
                units[-4] = round(units[-5] * 4.5)  # keying/scan error to be caught by hygiene audit

            inv = np.round(wk * share * cover * rng.lognormal(0, 0.05, HISTORY_WEEKS))
            pos_rows.append(pd.DataFrame({
                "Week_End_Date": weeks.strftime("%Y-%m-%d"),
                "Retailer": retailer,
                "SKU_ID": sku,
                "SKU_Description": desc,
                "Category": cat,
                "Units_Sold_POS": units.astype(int),
                "Base_Price": price,
                "Promo_Price": np.round(price * (1 - depth), 2),
                "Promo_Flag": promo.astype(int),
                "Inventory_On_Hand": inv.astype(int),
                "Supplier_Lead_Time_Weeks": lt if lt is not None else 4,
            }))

            n_promos = int(rng.choice([0, 1, 1, 2]))
            for start in rng.choice(np.arange(0, HORIZON_WEEKS - 1), size=n_promos, replace=False):
                promo_rows.append({
                    "Retailer": retailer,
                    "SKU_ID": sku,
                    "Week_End_Date": future_weeks[start].strftime("%Y-%m-%d"),
                    "Promo_Depth_Pct": float(rng.choice([20, 25, 30, 35])),
                })

    pos = pd.concat(pos_rows, ignore_index=True)
    pos = pos.sort_values(["Week_End_Date", "Retailer", "SKU_ID"]).reset_index(drop=True)

    master = pd.DataFrame([
        {
            "SKU_ID": sku,
            "SKU_Description": v[0],
            "Category": v[1],
            "Brand": v[2],
            "Status": "Discontinued" if sku == "PC-1011" else "Active",
            "Unit_Cost": v[4],
            "MOQ": v[11],
            "Order_Multiple": v[12],
            "Lead_Time_Weeks": v[10],
            "Safety_Stock_Weeks": 2,
            "Budget_Growth_Pct": round(float(rng.uniform(2, 10)), 1),
        }
        for sku, v in _SKUS.items()
    ])
    # PC-1009 lead time column intentionally left empty (None above) for the hygiene audit
    master["Lead_Time_Weeks"] = master["Lead_Time_Weeks"].astype("Float64")
    master["MOQ"] = master["MOQ"].astype("Float64")

    promo_calendar = pd.DataFrame(promo_rows, columns=["Retailer", "SKU_ID", "Week_End_Date", "Promo_Depth_Pct"])
    return pos, master, promo_calendar


def write(out_dir: Path, seed: int = 42) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pos, master, promo = generate(seed)
    pos.to_csv(out_dir / "pos_weekly_data.csv", index=False)
    master.to_csv(out_dir / "master_sku_data.csv", index=False)
    promo.to_csv(out_dir / "promo_calendar.csv", index=False)
