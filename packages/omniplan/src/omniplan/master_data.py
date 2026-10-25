"""Rule-based master data hygiene audit."""
from __future__ import annotations

import numpy as np
import pandas as pd

COLUMNS = ["SKU_ID", "SKU_Description", "Issue", "Severity", "Detail", "Suggested_Action"]
SWING_UP = 2.0     # >200% week-on-week increase
SWING_DOWN = -2 / 3  # symmetric decrease (a 200% swing in the other direction)


def _issue(sku, desc, issue, severity, detail, action):
    return {"SKU_ID": sku, "SKU_Description": desc, "Issue": issue, "Severity": severity,
            "Detail": detail, "Suggested_Action": action}


def audit(master: pd.DataFrame, pos: pd.DataFrame, recent_weeks: int = 8, swing_window: int = 13) -> pd.DataFrame:
    issues = []
    last = pos["Week_End_Date"].max()
    recent = pos[pos["Week_End_Date"] > last - pd.Timedelta(weeks=recent_weeks)]
    recent_units = recent.groupby("SKU_ID")["Units_Sold_POS"].sum()
    avg_weekly = pos[pos["Week_End_Date"] > last - pd.Timedelta(weeks=13)].groupby("SKU_ID")["Units_Sold_POS"].sum() / 13

    for r in master.itertuples():
        sku, desc = r.SKU_ID, r.SKU_Description
        if str(r.Status).strip().lower() == "discontinued" and recent_units.get(sku, 0) > 0:
            issues.append(_issue(sku, desc, "Discontinued SKU with active demand", "High",
                                 f"{int(recent_units[sku]):,} units sold in last {recent_weeks} weeks",
                                 "Confirm lifecycle status; reactivate SKU or plan run-out / sell-through."))
        if pd.isna(r.MOQ) or r.MOQ <= 0:
            issues.append(_issue(sku, desc, "Missing MOQ", "High", "MOQ is blank or zero",
                                 "Obtain MOQ from supplier and update ERP before next replenishment run."))
        if pd.isna(r.Lead_Time_Weeks) or r.Lead_Time_Weeks <= 0:
            issues.append(_issue(sku, desc, "Missing lead time", "High", "Lead time is blank or zero",
                                 "Confirm supplier lead time; planning currently uses a default."))
        mult = r.Order_Multiple
        if pd.isna(mult) or mult <= 0:
            issues.append(_issue(sku, desc, "Missing order multiple", "Medium", "Order multiple is blank or zero",
                                 "Set the pack/pallet order multiple."))
        else:
            if not pd.isna(r.MOQ) and r.MOQ > 0 and r.MOQ % mult != 0:
                issues.append(_issue(sku, desc, "Abnormal order multiple", "Medium",
                                     f"MOQ {r.MOQ:,.0f} is not a multiple of order multiple {mult:,.0f}",
                                     "Align MOQ and order multiple with the supplier."))
            wk = avg_weekly.get(sku, np.nan)
            if wk == wk and wk > 0 and mult > 4 * wk:
                issues.append(_issue(sku, desc, "Abnormal order multiple", "Medium",
                                     f"Order multiple {mult:,.0f} is {mult / wk:.1f} weeks of demand",
                                     "Review multiple; it forces large over-orders."))

    desc_map = master.set_index("SKU_ID")["SKU_Description"].to_dict()
    window = pos[pos["Week_End_Date"] > last - pd.Timedelta(weeks=swing_window)]
    for (sku, retailer), g in window.groupby(["SKU_ID", "Retailer"]):
        g = g.sort_values("Week_End_Date")
        prev_units = g["Units_Sold_POS"].shift()
        pct = g["Units_Sold_POS"] / prev_units.replace(0, np.nan) - 1
        clean = (g["Promo_Flag"] == 0) & (g["Promo_Flag"].shift() == 0)  # promo swings are expected
        for i in g.index[clean & ((pct > SWING_UP) | (pct < SWING_DOWN))]:
            issues.append(_issue(
                sku, desc_map.get(sku, g["SKU_Description"].iloc[0]), "Abnormal POS swing (non-promo)", "Medium",
                f"{retailer}, w/e {g.loc[i, 'Week_End_Date']:%Y-%m-%d}: {pct[i] * 100:+.0f}% WoW "
                f"({int(prev_units[i]):,} -> {int(g.loc[i, 'Units_Sold_POS']):,})",
                "Validate POS record; scrub baseline manually if it is a data error."))
    return pd.DataFrame(issues, columns=COLUMNS)
