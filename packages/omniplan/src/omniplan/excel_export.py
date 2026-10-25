"""Excel reporting pack with live formulas and variance heat-maps."""
from __future__ import annotations

from io import BytesIO

import pandas as pd
from openpyxl import Workbook
from openpyxl.formatting.rule import ColorScaleRule, FormulaRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter as L

HEADER_FILL = PatternFill("solid", fgColor="1F3864")
HEADER_FONT = Font(bold=True, color="FFFFFF")
INPUT_FONT = Font(color="0000FF")  # blue = editable assumption
RED, AMBER, GREEN = "F8696B", "FFEB84", "63BE7B"


def _header(ws, row: int, labels: list[str]) -> None:
    for i, label in enumerate(labels, 1):
        c = ws.cell(row=row, column=i, value=label)
        c.fill, c.font = HEADER_FILL, HEADER_FONT
        c.alignment = Alignment(horizontal="center", wrap_text=True)
    ws.freeze_panes = ws.cell(row=row + 1, column=3)


def _widths(ws, widths: dict[str, float]) -> None:
    for col, w in widths.items():
        ws.column_dimensions[col].width = w


def build_workbook(scenario_fc: pd.DataFrame, weekly_inv: pd.DataFrame, summary: pd.DataFrame,
                   consensus: pd.DataFrame, audit: pd.DataFrame, pos: pd.DataFrame, master: pd.DataFrame,
                   lift_adj: float, delay_weeks: int, cover_days: int) -> bytes:
    wb = Workbook()
    horizon = int(scenario_fc["Week_No"].max())
    week_dates = sorted(scenario_fc["Week_End_Date"].unique())

    # ---- Assumptions
    wa = wb.active
    wa.title = "Assumptions"
    wa["A1"], wa["A1"].font = "OmniPlan - Scenario Assumptions (blue cells drive formulas)", Font(bold=True, size=13)
    for r, (label, val) in enumerate([
        ("Promo lift adjustment (%)", lift_adj * 100), ("Lead-time delay (weeks)", delay_weeks),
        ("Max forward cover (days)", cover_days)], start=3):
        wa.cell(row=r, column=1, value=label)
        c = wa.cell(row=r, column=2, value=val)
        c.font = INPUT_FONT
    _widths(wa, {"A": 34, "B": 14})
    cover_ref = "Assumptions!$B$5"

    # ---- Forward forecast by SKU x retailer
    ws = wb.create_sheet("Forward Forecast")
    labels = ["SKU_ID", "Description", "Retailer", "Unit Cost"] + [f"Wk{i} {pd.Timestamp(d):%d-%b}" for i, d in enumerate(week_dates, 1)] \
        + ["Total Units", "Forecast $ (cost)", "Budget Units", "Variance Units", "Variance %"]
    _header(ws, 1, labels)
    cost = master.set_index("SKU_ID")["Unit_Cost"].fillna(0)
    growth = master.set_index("SKU_ID")["Budget_Growth_Pct"].fillna(0) / 100
    ly = pos.assign(LY=pos["Week_End_Date"] + pd.Timedelta(days=364))
    ly = ly[ly["LY"].isin(week_dates)].groupby(["SKU_ID", "Retailer"])["Units_Sold_POS"].sum()
    wide = scenario_fc.pivot_table(index=["SKU_ID", "Retailer"], columns="Week_No", values="Forecast_Units", aggfunc="sum")
    desc = master.set_index("SKU_ID")["SKU_Description"]
    first_wk, last_wk = 5, 4 + horizon
    tot_c, val_c, bud_c, var_c, pct_c = last_wk + 1, last_wk + 2, last_wk + 3, last_wk + 4, last_wk + 5
    for r, ((sku, retailer), row) in enumerate(wide.iterrows(), start=2):
        ws.cell(r, 1, sku); ws.cell(r, 2, desc.get(sku, "")); ws.cell(r, 3, retailer)
        ws.cell(r, 4, float(cost.get(sku, 0))).number_format = "$#,##0.00"
        for w in range(1, horizon + 1):
            ws.cell(r, 4 + w, round(float(row.get(w, 0)), 1)).number_format = "#,##0"
        ws.cell(r, tot_c, f"=SUM({L(first_wk)}{r}:{L(last_wk)}{r})").number_format = "#,##0"
        ws.cell(r, val_c, f"={L(tot_c)}{r}*D{r}").number_format = "$#,##0"
        budget = float(ly.get((sku, retailer), 0)) * (1 + float(growth.get(sku, 0)))
        ws.cell(r, bud_c, round(budget, 1)).font = INPUT_FONT
        ws.cell(r, bud_c).number_format = "#,##0"
        ws.cell(r, var_c, f"={L(tot_c)}{r}-{L(bud_c)}{r}").number_format = "#,##0;[Red]-#,##0"
        ws.cell(r, pct_c, f"=IFERROR({L(var_c)}{r}/{L(bud_c)}{r},0)").number_format = "0.0%"
    last_r = 1 + len(wide)
    t = last_r + 1
    ws.cell(t, 1, "TOTAL").font = Font(bold=True)
    for c in list(range(first_wk, last_wk + 1)) + [tot_c, val_c, bud_c, var_c]:
        cell = ws.cell(t, c, f"=SUM({L(c)}2:{L(c)}{last_r})")
        cell.font, cell.number_format = Font(bold=True), "#,##0" if c != val_c else "$#,##0"
    ws.cell(t, pct_c, f"=IFERROR({L(var_c)}{t}/{L(bud_c)}{t},0)").number_format = "0.0%"
    ws.conditional_formatting.add(f"{L(pct_c)}2:{L(pct_c)}{last_r}",
                                  ColorScaleRule(start_type="num", start_value=-0.3, start_color=RED,
                                                 mid_type="num", mid_value=0, mid_color="FFFFFF",
                                                 end_type="num", end_value=0.3, end_color=GREEN))
    ws.conditional_formatting.add(f"{L(first_wk)}2:{L(last_wk)}{last_r}",
                                  ColorScaleRule(start_type="min", start_color="FFFFFF", end_type="max", end_color="5B9BD5"))
    _widths(ws, {"A": 10, "B": 32, "C": 18})

    # ---- Inventory projection with formulas
    wi = wb.create_sheet("Inventory Projection")
    dem0, rec0, inv0 = 6, 6 + horizon, 6 + 2 * horizon
    labels = ["SKU_ID", "Description", "Start Inv", "Safety Stock", "Unit Cost"] \
        + [f"Demand Wk{i}" for i in range(1, horizon + 1)] + [f"Receipts Wk{i}" for i in range(1, horizon + 1)] \
        + [f"End Inv Wk{i}" for i in range(1, horizon + 1)] \
        + ["Min Inv", "Status", "Avg Wkly Demand", "Excess Units (> cover)", "Capital At Risk $"]
    _header(wi, 1, labels)
    dm = weekly_inv.pivot(index="SKU_ID", columns="Week_No", values="Demand")
    rc = weekly_inv.pivot(index="SKU_ID", columns="Week_No", values="Receipts")
    s = summary.set_index("SKU_ID")
    min_c = inv0 + horizon
    for r, sku in enumerate(s.index, start=2):
        wi.cell(r, 1, sku); wi.cell(r, 2, s.loc[sku, "SKU_Description"])
        wi.cell(r, 3, round(float(s.loc[sku, "Start_Inventory"]), 0)).font = INPUT_FONT
        wi.cell(r, 4, round(float(s.loc[sku, "Safety_Stock"]), 0)).font = INPUT_FONT
        wi.cell(r, 5, float(s.loc[sku, "Unit_Cost"])).number_format = "$#,##0.00"
        for w in range(1, horizon + 1):
            wi.cell(r, dem0 + w - 1, round(float(dm.loc[sku, w]), 1))
            wi.cell(r, rec0 + w - 1, round(float(rc.loc[sku, w]), 1))
            prev = f"$C{r}" if w == 1 else f"{L(inv0 + w - 2)}{r}"
            wi.cell(r, inv0 + w - 1, f"={prev}+{L(rec0 + w - 1)}{r}-{L(dem0 + w - 1)}{r}").number_format = "#,##0"
        inv_rng = f"{L(inv0)}{r}:{L(inv0 + horizon - 1)}{r}"
        wi.cell(r, min_c, f"=MIN({inv_rng})").number_format = "#,##0"
        wi.cell(r, min_c + 1, f'=IF({L(min_c)}{r}<0,"STOCKOUT",IF({L(min_c)}{r}<D{r},"BELOW SS","OK"))')
        wi.cell(r, min_c + 2, f"=AVERAGE({L(dem0)}{r}:{L(dem0 + horizon - 1)}{r})").number_format = "#,##0"
        wi.cell(r, min_c + 3, f"=MAX(0,{L(inv0 + horizon - 1)}{r}-{cover_ref}/7*{L(min_c + 2)}{r})").number_format = "#,##0"
        wi.cell(r, min_c + 4, f"={L(min_c + 3)}{r}*E{r}").number_format = "$#,##0"
    last_i = 1 + len(s)
    wi.cell(last_i + 1, 1, "TOTAL CAPITAL AT RISK").font = Font(bold=True)
    wi.cell(last_i + 1, min_c + 4, f"=SUM({L(min_c + 4)}2:{L(min_c + 4)}{last_i})").number_format = "$#,##0"
    inv_block = f"{L(inv0)}2:{L(inv0 + horizon - 1)}{last_i}"
    wi.conditional_formatting.add(inv_block, FormulaRule(formula=[f"{L(inv0)}2<0"], fill=PatternFill("solid", bgColor=RED)))
    wi.conditional_formatting.add(inv_block, FormulaRule(formula=[f"{L(inv0)}2<$D2"], fill=PatternFill("solid", bgColor=AMBER)))
    status_rng = f"{L(min_c + 1)}2:{L(min_c + 1)}{last_i}"
    wi.conditional_formatting.add(status_rng, FormulaRule(formula=[f'{L(min_c + 1)}2="STOCKOUT"'], fill=PatternFill("solid", bgColor=RED)))
    wi.conditional_formatting.add(status_rng, FormulaRule(formula=[f'{L(min_c + 1)}2="BELOW SS"'], fill=PatternFill("solid", bgColor=AMBER)))
    _widths(wi, {"A": 10, "B": 32})

    # ---- S&OP consensus vs budget
    wc = wb.create_sheet("S&OP Consensus")
    _header(wc, 1, ["SKU_ID", "Description", "Statistical Fcst", "Override %", "Consensus Fcst", "Budget Units",
                    "Variance Units", "Variance %", "Variance $ (cost)", "Unit Cost", "Commentary"])
    for r, row in enumerate(consensus.itertuples(), start=2):
        wc.cell(r, 1, row.SKU_ID); wc.cell(r, 2, row.SKU_Description)
        wc.cell(r, 3, round(row.Statistical_Forecast, 0)).number_format = "#,##0"
        c = wc.cell(r, 4, row.Override_Pct / 100); c.font, c.number_format = INPUT_FONT, "0%"
        wc.cell(r, 5, f"=C{r}*(1+D{r})").number_format = "#,##0"
        wc.cell(r, 6, round(row.Budget_Units, 0)).number_format = "#,##0"
        wc.cell(r, 7, f"=E{r}-F{r}").number_format = "#,##0;[Red]-#,##0"
        wc.cell(r, 8, f"=IFERROR(G{r}/F{r},0)").number_format = "0.0%"
        wc.cell(r, 10, row.Unit_Cost).number_format = "$#,##0.00"
        wc.cell(r, 9, f"=G{r}*J{r}").number_format = "$#,##0;[Red]-$#,##0"
        wc.cell(r, 11, row.Commentary)
    last_c = 1 + len(consensus)
    wc.cell(last_c + 1, 1, "TOTAL").font = Font(bold=True)
    for col in (3, 5, 6, 7, 9):
        wc.cell(last_c + 1, col, f"=SUM({L(col)}2:{L(col)}{last_c})").number_format = "#,##0"
    wc.cell(last_c + 1, 8, f"=IFERROR(G{last_c + 1}/F{last_c + 1},0)").number_format = "0.0%"
    wc.conditional_formatting.add(f"H2:H{last_c}", ColorScaleRule(
        start_type="num", start_value=-0.3, start_color=RED, mid_type="num", mid_value=0, mid_color="FFFFFF",
        end_type="num", end_value=0.3, end_color=GREEN))
    _widths(wc, {"A": 10, "B": 32, "K": 70})

    # ---- Master data
    wm = wb.create_sheet("Master Data Issues")
    cols = list(audit.columns) or ["SKU_ID", "SKU_Description", "Issue", "Severity", "Detail", "Suggested_Action"]
    _header(wm, 1, cols)
    for r, row in enumerate(audit.itertuples(index=False), start=2):
        for c, v in enumerate(row, 1):
            wm.cell(r, c, v)
    _widths(wm, {"A": 10, "B": 32, "C": 38, "D": 10, "E": 60, "F": 60})

    buf = BytesIO()
    wb.save(buf)
    return buf.getvalue()
