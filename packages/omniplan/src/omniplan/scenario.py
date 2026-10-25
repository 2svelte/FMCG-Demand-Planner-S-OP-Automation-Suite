"""S&OP risk/opportunity simulator and consensus-vs-budget view."""
from __future__ import annotations

import numpy as np
import pandas as pd


def effective_lead_times(master: pd.DataFrame, pos: pd.DataFrame, default_weeks: int = 4) -> pd.Series:
    """Master lead time, falling back to the latest POS-file value, then a default."""
    pos_lt = pos.sort_values("Week_End_Date").groupby("SKU_ID")["Supplier_Lead_Time_Weeks"].last()
    lt = master.set_index("SKU_ID")["Lead_Time_Weeks"].astype(float)
    return lt.fillna(pos_lt).fillna(default_weeks)


def project_inventory(plan_fc: pd.DataFrame, scenario_fc: pd.DataFrame, pos: pd.DataFrame,
                      master: pd.DataFrame, delay_weeks: int = 0, cover_days: int = 120,
                      default_lead_time: int = 4):
    """Project weekly SKU inventory.

    Supply (receipts) follows the *plan* forecast (what supply was committed against), shifted by
    the lead-time delay. Demand follows the *scenario* forecast, so promo-lift and delay shocks
    show up as stockouts or excess.
    Returns (weekly_projection, sku_summary).
    """
    delay_weeks = int(delay_weeks)
    last_week = pos["Week_End_Date"].max()
    start_inv = pos[pos["Week_End_Date"] == last_week].groupby("SKU_ID")["Inventory_On_Hand"].sum()
    plan = plan_fc.groupby(["SKU_ID", "Week_No", "Week_End_Date"], as_index=False)["Forecast_Units"].sum()
    scen = scenario_fc.groupby(["SKU_ID", "Week_No", "Week_End_Date"], as_index=False)["Forecast_Units"].sum()
    m = master.set_index("SKU_ID")
    lead = effective_lead_times(master, pos, default_lead_time)

    weekly_rows, summary_rows = [], []
    for sku, d in scen.groupby("SKU_ID"):
        d = d.sort_values("Week_No")
        demand = d["Forecast_Units"].to_numpy()
        planned = plan[plan["SKU_ID"] == sku].sort_values("Week_No")["Forecast_Units"].to_numpy()
        receipts = np.zeros_like(planned)
        if delay_weeks < len(planned):
            receipts[delay_weeks:] = planned[: len(planned) - delay_weeks]
        start = float(start_inv.get(sku, 0.0))
        end_inv = start + np.cumsum(receipts - demand)
        avg_wk = max(float(demand.mean()), 1e-9)
        ss_weeks = m["Safety_Stock_Weeks"].get(sku, np.nan)
        safety = (2.0 if pd.isna(ss_weeks) else float(ss_weeks)) * avg_wk
        cost = m["Unit_Cost"].get(sku, np.nan)
        cost = 0.0 if pd.isna(cost) else float(cost)

        below = end_inv < safety
        status = "Stockout" if (end_inv < 0).any() else ("Below Safety Stock" if below.any() else "OK")
        first_risk = int(d["Week_No"].to_numpy()[below.argmax()]) if below.any() else None
        cover_end = end_inv[-1] / (avg_wk / 7)
        excess_units = max(0.0, end_inv[-1] - cover_days / 7 * avg_wk)
        shortfall = max(0.0, safety - end_inv.min())
        buy = 0.0
        if shortfall > 0:
            mult = m["Order_Multiple"].get(sku, np.nan)
            moq = m["MOQ"].get(sku, np.nan)
            mult = 1.0 if pd.isna(mult) or mult <= 0 else float(mult)
            buy = float(np.ceil(shortfall / mult) * mult)
            if not pd.isna(moq):
                buy = max(buy, float(moq))

        for i, row in enumerate(d.itertuples()):
            weekly_rows.append({
                "SKU_ID": sku, "Week_No": row.Week_No, "Week_End_Date": row.Week_End_Date,
                "Demand": demand[i], "Receipts": receipts[i], "Projected_Inventory": end_inv[i],
                "Safety_Stock": safety, "Below_Safety_Stock": bool(below[i]),
            })
        summary_rows.append({
            "SKU_ID": sku, "SKU_Description": m["SKU_Description"].get(sku, ""),
            "Status": m["Status"].get(sku, ""), "Lead_Time_Weeks": lead.get(sku, default_lead_time) + delay_weeks,
            "Unit_Cost": cost, "Start_Inventory": start, "Total_Demand": demand.sum(),
            "Avg_Weekly_Demand": avg_wk, "Safety_Stock": safety, "Min_Projected_Inventory": end_inv.min(),
            "First_Risk_Week": first_risk, "Risk_Status": status, "Ending_Inventory": end_inv[-1],
            "Forward_Cover_Days": cover_end, "Excess_Units": excess_units,
            "Capital_At_Risk": excess_units * cost, "Recommended_Buy_Units": buy,
        })
    return pd.DataFrame(weekly_rows), pd.DataFrame(summary_rows)


def consensus_vs_budget(scenario_fc: pd.DataFrame, pos: pd.DataFrame, master: pd.DataFrame,
                        overrides: dict[str, float] | None = None) -> pd.DataFrame:
    """Per-SKU 12-week statistical forecast, commercial consensus (override %) and budget.

    Budget = same weeks last year x (1 + Budget_Growth_Pct).
    """
    overrides = overrides or {}
    fc = scenario_fc.groupby(["SKU_ID", "Week_End_Date"], as_index=False)["Forecast_Units"].sum()
    fc["LY_Date"] = fc["Week_End_Date"] - pd.Timedelta(days=364)
    ly = pos.groupby(["SKU_ID", "Week_End_Date"], as_index=False)["Units_Sold_POS"].sum() \
        .rename(columns={"Week_End_Date": "LY_Date", "Units_Sold_POS": "LY_Units"})
    fc = fc.merge(ly, on=["SKU_ID", "LY_Date"], how="left")
    agg = fc.groupby("SKU_ID", as_index=False).agg(Statistical_Forecast=("Forecast_Units", "sum"), LY_Units=("LY_Units", "sum"))
    m = master.set_index("SKU_ID")
    agg["SKU_Description"] = agg["SKU_ID"].map(m["SKU_Description"])
    agg["Unit_Cost"] = agg["SKU_ID"].map(m["Unit_Cost"]).fillna(0)
    growth = agg["SKU_ID"].map(m["Budget_Growth_Pct"]).fillna(0) / 100
    agg["Budget_Units"] = agg["LY_Units"] * (1 + growth)
    agg["Override_Pct"] = agg["SKU_ID"].map(overrides).fillna(0.0)
    agg["Consensus_Forecast"] = agg["Statistical_Forecast"] * (1 + agg["Override_Pct"] / 100)
    agg["Variance_vs_Budget_Units"] = agg["Consensus_Forecast"] - agg["Budget_Units"]
    agg["Variance_vs_Budget_Pct"] = (agg["Variance_vs_Budget_Units"] / agg["Budget_Units"].replace(0, np.nan)) * 100
    agg["Variance_Value"] = agg["Variance_vs_Budget_Units"] * agg["Unit_Cost"]
    agg["Commentary"] = agg.apply(_commentary, axis=1)
    cols = ["SKU_ID", "SKU_Description", "Statistical_Forecast", "Override_Pct", "Consensus_Forecast",
            "Budget_Units", "Variance_vs_Budget_Units", "Variance_vs_Budget_Pct", "Variance_Value",
            "Unit_Cost", "Commentary"]
    return agg[cols]


def _commentary(r) -> str:
    pct = r["Variance_vs_Budget_Pct"]
    if pd.isna(pct):
        return "No prior-year base for budget comparison."
    if abs(pct) < 5:
        return "On budget (within +/-5%)."
    direction = "above" if pct > 0 else "below"
    action = "confirm supply capacity and promo funding" if pct > 0 else "review promo plan / pipeline for recovery"
    note = f" Includes {r['Override_Pct']:+.0f}% commercial override." if r["Override_Pct"] else ""
    return f"{abs(pct):.0f}% {direction} budget; {action}.{note}"
