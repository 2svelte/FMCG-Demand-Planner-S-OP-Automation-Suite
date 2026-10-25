"""OmniPlan - FMCG / Personal Care demand planner & S&OP automation suite (Streamlit)."""
import pandas as pd
import streamlit as st

from omniplan import forecast as fc_mod
from omniplan import llm, master_data, scenario
from omniplan.config import get_settings
from omniplan.excel_export import build_workbook
from omniplan.loaders import DataValidationError, load_master, load_pos, load_promo_calendar

st.set_page_config(page_title="OmniPlan | Demand Planner Automation Suite", page_icon="📦", layout="wide")
settings = get_settings()


@st.cache_data(show_spinner="Loading data...")
def _load(pos_bytes, master_bytes, promo_bytes):
    pos = load_pos(pos_bytes, settings) if pos_bytes is not None else load_pos(settings=settings)
    master = load_master(master_bytes, settings) if master_bytes is not None else load_master(settings=settings)
    promo = load_promo_calendar(promo_bytes, settings) if promo_bytes is not None else load_promo_calendar(settings=settings)
    return pos, master, promo


@st.cache_data(show_spinner="Fitting Holt-Winters baselines for every SKU x retailer...")
def _fit(pos: pd.DataFrame, horizon: int):
    return fc_mod.fit_baselines(pos, horizon), fc_mod.promo_lift_table(pos)


# ---------------------------------------------------------------- sidebar
st.sidebar.title("📦 OmniPlan")
up_pos = st.sidebar.file_uploader("Upload Weekly Retailer POS Dump", type=["csv"])
up_master = st.sidebar.file_uploader("Upload SKU Master Data (optional)", type=["csv"])
up_promo = st.sidebar.file_uploader("Upload Promo Calendar (optional)", type=["csv"])
st.sidebar.caption("Defaults to the bundled files in `data/`.")

st.sidebar.header("S&OP scenario")
lift_adj_pct = st.sidebar.slider("Promo lift adjustment (%)", -30, 30, 0, 5,
                                 help="Scales the historical promo uplift assumed for scheduled promos.")
delay_weeks = st.sidebar.slider("Supplier lead-time delay (weeks)", 0, 3, 0)
cover_days = st.sidebar.number_input("Max forward cover (days)", 30, 365, settings.max_cover_days, 10)
horizon = settings.horizon_weeks

try:
    pos, master, promo_cal = _load(
        up_pos.getvalue() if up_pos else None,
        up_master.getvalue() if up_master else None,
        up_promo.getvalue() if up_promo else None,
    )
except (DataValidationError, FileNotFoundError) as exc:
    st.error(f"Could not load data: {exc}")
    st.stop()
except Exception as exc:
    st.error(f"Uploaded file could not be read: {exc}")
    st.stop()

# POS uploads without a matching master get a minimal one, so the app still runs
if up_pos and not up_master:
    skus = pos.drop_duplicates("SKU_ID")
    extra = skus.loc[~skus["SKU_ID"].isin(master["SKU_ID"]), ["SKU_ID", "SKU_Description", "Category"]]
    if not extra.empty:
        st.warning("Some uploaded SKUs are not in the bundled master data; they are added with blank attributes "
                   "(see Master Data tab).")
        master = pd.concat([master, extra], ignore_index=True)
        master["Status"] = master["Status"].fillna("Active")
        master["Safety_Stock_Weeks"] = master["Safety_Stock_Weeks"].fillna(2)

baseline, lift_tbl = _fit(pos, horizon)
plan_fc = fc_mod.apply_promos(baseline, lift_tbl, promo_cal, 0.0)
scen_fc = fc_mod.apply_promos(baseline, lift_tbl, promo_cal, lift_adj_pct / 100)
weekly_inv, summary = scenario.project_inventory(plan_fc, scen_fc, pos, master, delay_weeks, cover_days,
                                                 settings.default_lead_time_weeks)
audit = master_data.audit(master, pos)

st.title("Demand Planner Automation Suite")
st.caption(f"POS history to {pos['Week_End_Date'].max():%d %b %Y} | {pos['SKU_ID'].nunique()} SKUs | "
           f"{pos['Retailer'].nunique()} retailers | {horizon}-week forward horizon")

tab_sop, tab_fc, tab_scn, tab_md, tab_ai, tab_x = st.tabs(
    ["Executive S&OP", "Forecast & Promo Lift", "Risk & Opportunity", "Master Data Hygiene", "AI Brief", "Excel Export"])

# ---------------------------------------------------------------- Executive S&OP
with tab_sop:
    st.subheader("Consensus forecast vs budget")
    st.caption("Edit **Override %** to layer commercial intelligence over the statistical forecast. "
               "Budget = last year's same weeks x (1 + budget growth %).")
    base_cons = scenario.consensus_vs_budget(scen_fc, pos, master)
    edited = st.data_editor(
        base_cons[["SKU_ID", "SKU_Description", "Override_Pct"]], hide_index=True, key="overrides",
        disabled=["SKU_ID", "SKU_Description"], use_container_width=True,
        column_config={"Override_Pct": st.column_config.NumberColumn("Override %", min_value=-100.0, max_value=300.0, step=1.0)})
    consensus = scenario.consensus_vs_budget(scen_fc, pos, master, dict(zip(edited["SKU_ID"], edited["Override_Pct"])))
    tot_c, tot_b = consensus["Consensus_Forecast"].sum(), consensus["Budget_Units"].sum()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Consensus units (12 wk)", f"{tot_c:,.0f}")
    k2.metric("Budget units", f"{tot_b:,.0f}")
    k3.metric("Variance vs budget", f"{(tot_c / tot_b - 1) * 100:+.1f}%" if tot_b else "n/a")
    k4.metric("Variance value (at cost)", f"${consensus['Variance_Value'].sum():+,.0f}")
    st.dataframe(
        consensus.style.format({"Statistical_Forecast": "{:,.0f}", "Override_Pct": "{:+.0f}%", "Consensus_Forecast": "{:,.0f}",
                                "Budget_Units": "{:,.0f}", "Variance_vs_Budget_Units": "{:+,.0f}",
                                "Variance_vs_Budget_Pct": "{:+.1f}%", "Variance_Value": "${:+,.0f}", "Unit_Cost": "${:.2f}"})
        .background_gradient(subset=["Variance_vs_Budget_Pct"], cmap="RdYlGn", vmin=-30, vmax=30),
        hide_index=True, use_container_width=True)

# ---------------------------------------------------------------- Forecast
with tab_fc:
    c1, c2 = st.columns(2)
    sel_sku = c1.selectbox("SKU", sorted(pos["SKU_ID"].unique()),
                           format_func=lambda s: f"{s} - {master.set_index('SKU_ID')['SKU_Description'].get(s, '')}")
    retailers = ["All retailers"] + sorted(pos["Retailer"].unique())
    sel_ret = c2.selectbox("Retailer", retailers)

    hist = pos[pos["SKU_ID"] == sel_sku]
    fcs = scen_fc[scen_fc["SKU_ID"] == sel_sku]
    if sel_ret != "All retailers":
        hist, fcs = hist[hist["Retailer"] == sel_ret], fcs[fcs["Retailer"] == sel_ret]
    h = hist.groupby("Week_End_Date")["Units_Sold_POS"].sum().tail(52).rename("Actual POS")
    f = fcs.groupby("Week_End_Date")[["Baseline_Units", "Forecast_Units"]].sum() \
        .rename(columns={"Baseline_Units": "Baseline forecast", "Forecast_Units": "Forecast incl. promos"})
    st.line_chart(pd.concat([h, f], axis=1))
    st.markdown("**Promotional lift** = (promo units - base units) / base units, vs. an interpolated promo-free baseline")
    lt = lift_tbl[lift_tbl["SKU_ID"] == sel_sku]
    st.dataframe(lt.style.format({"Avg_Promo_Depth": "{:.0%}", "Promo_Lift": "{:+.0%}", "Elasticity": "{:.2f}"}),
                 hide_index=True, use_container_width=True)
    st.markdown("**12-week forward forecast by SKU x retailer**")
    wide = scen_fc.pivot_table(index=["SKU_ID", "Retailer"], columns="Week_End_Date", values="Forecast_Units", aggfunc="sum")
    wide.columns = [f"{c:%d-%b}" for c in wide.columns]
    wide["Total"] = wide.sum(axis=1)
    st.dataframe(wide.style.format("{:,.0f}").background_gradient(axis=None, cmap="Blues"), use_container_width=True)

# ---------------------------------------------------------------- Scenario
with tab_scn:
    n_risk = int((summary["Risk_Status"] != "OK").sum())
    n_out = int((summary["Risk_Status"] == "Stockout").sum())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("SKUs at risk", n_risk, help="Projected inventory below safety stock in at least one week")
    m2.metric("Projected stockouts", n_out)
    m3.metric("Capital at risk (excess)", f"${summary['Capital_At_Risk'].sum():,.0f}",
              help=f"Stock value above {cover_days} days of forward cover at end of horizon")
    m4.metric("Scenario", f"{lift_adj_pct:+d}% lift / +{delay_weeks}w delay")
    st.dataframe(
        summary[["SKU_ID", "SKU_Description", "Status", "Risk_Status", "First_Risk_Week", "Start_Inventory", "Safety_Stock",
                 "Min_Projected_Inventory", "Ending_Inventory", "Forward_Cover_Days", "Lead_Time_Weeks",
                 "Recommended_Buy_Units", "Excess_Units", "Capital_At_Risk"]].style.format({
            "Start_Inventory": "{:,.0f}", "Safety_Stock": "{:,.0f}", "Min_Projected_Inventory": "{:,.0f}",
            "Ending_Inventory": "{:,.0f}", "Forward_Cover_Days": "{:,.0f}", "Lead_Time_Weeks": "{:.0f}",
            "Recommended_Buy_Units": "{:,.0f}", "Excess_Units": "{:,.0f}", "Capital_At_Risk": "${:,.0f}"}, na_rep="-")
        .map(lambda v: "background-color:#f8696b" if v == "Stockout" else
             ("background-color:#ffeb84" if v == "Below Safety Stock" else ""), subset=["Risk_Status"]),
        hide_index=True, use_container_width=True)
    st.markdown("**Projected inventory by week** (rows = SKU; red = below zero, amber = below safety stock)")
    heat = weekly_inv.pivot(index="SKU_ID", columns="Week_End_Date", values="Projected_Inventory")
    heat.columns = [f"{c:%d-%b}" for c in heat.columns]
    ss = weekly_inv.groupby("SKU_ID")["Safety_Stock"].first()

    def _color(row):
        return ["background-color:#f8696b" if v < 0 else "background-color:#ffeb84" if v < ss[row.name] else ""
                for v in row]
    st.dataframe(heat.style.apply(_color, axis=1).format("{:,.0f}"), use_container_width=True)

# ---------------------------------------------------------------- Master data
with tab_md:
    if audit.empty:
        st.success("No master data issues found.")
    else:
        st.warning(f"{len(audit)} issue(s) across {audit['SKU_ID'].nunique()} SKU(s)")
        types = st.multiselect("Filter issue type", sorted(audit["Issue"].unique()), default=sorted(audit["Issue"].unique()))
        st.dataframe(audit[audit["Issue"].isin(types)], hide_index=True, use_container_width=True)
        if st.button("Draft master data fix memo"):
            with st.spinner("Writing memo..."):
                memo, src = llm.generate_master_data_memo(audit, settings)
            st.caption(f"Source: {src}")
            st.markdown(memo)

# ---------------------------------------------------------------- AI brief
with tab_ai:
    ctx = llm.build_context(scen_fc, summary, consensus, audit, lift_adj_pct / 100, delay_weeks)
    with st.expander("Prompt sent to the LLM"):
        st.code(llm.build_prompt(ctx), language="text")
    if not settings.llm_enabled:
        st.info("No `OPENAI_API_KEY` found in `.env` - a deterministic template brief will be generated instead.")
    if st.button("Generate S&OP Weekly Consensus Brief", type="primary"):
        with st.spinner("Drafting brief..."):
            st.session_state["brief"] = llm.generate_brief(ctx, settings)
    if "brief" in st.session_state:
        text, src = st.session_state["brief"]
        st.caption(f"Source: {src}")
        st.markdown(text)
        st.download_button("Download brief (.md)", text, file_name="sop_executive_brief.md")

# ---------------------------------------------------------------- Export
with tab_x:
    st.markdown("Excel reporting pack: forward forecast, inventory projection (live formulas), S&OP consensus and master data issues. "
                "Blue cells are inputs; totals, variances and risk flags recalculate in Excel.")
    xlsx = build_workbook(scen_fc, weekly_inv, summary, consensus, audit, pos, master,
                          lift_adj_pct / 100, delay_weeks, cover_days)
    st.download_button("Download Excel workbook", xlsx, file_name="omniplan_sop_pack.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")
