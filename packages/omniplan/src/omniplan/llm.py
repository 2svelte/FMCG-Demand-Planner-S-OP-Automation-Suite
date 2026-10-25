"""LLM narrative agent: S&OP executive brief and master-data memo (with offline fallback)."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import Settings, get_settings

SYSTEM_PROMPT = (
    "You are the Lead Demand Planner for a Personal Care / FMCG brand selling through Australian "
    "retailers (Chemist Warehouse, Priceline, Coles). Be concise, quantitative and action-oriented. "
    "Use only the facts supplied; never invent numbers."
)


@dataclass
class BriefContext:
    total_demand: float
    horizon_weeks: int
    promo_lift_adj_pct: float
    lead_time_delay_weeks: int
    stockout_skus: list[str]
    excess_dollars: float
    budget_variance_pct: float
    top_risks: list[str]
    top_variances: list[str]
    master_data_issue_count: int
    master_data_highlights: list[str]


def build_context(scenario_fc: pd.DataFrame, summary: pd.DataFrame, consensus: pd.DataFrame,
                  audit: pd.DataFrame, lift_adj: float, delay_weeks: int) -> BriefContext:
    at_risk = summary[summary["Risk_Status"] != "OK"].sort_values(["Risk_Status", "Min_Projected_Inventory"], ascending=[False, True])
    top_risks = [
        f"{r.SKU_ID} {r.SKU_Description}: {r.Risk_Status}, first risk wk {r.First_Risk_Week}, "
        f"suggested buy {r.Recommended_Buy_Units:,.0f} units"
        for r in at_risk.head(3).itertuples()
    ]
    var = consensus.reindex(consensus["Variance_Value"].abs().sort_values(ascending=False).index).head(3)
    top_var = [f"{r.SKU_ID} {r.SKU_Description}: {r.Variance_vs_Budget_Pct:+.0f}% vs budget (${r.Variance_Value:+,.0f})"
               for r in var.itertuples() if pd.notna(r.Variance_vs_Budget_Pct)]
    total_cons, total_budget = consensus["Consensus_Forecast"].sum(), consensus["Budget_Units"].sum()
    highlights = (audit.groupby("Issue").size().sort_values(ascending=False).head(4)
                  .apply(lambda n: f"{n}").to_dict() if len(audit) else {})
    return BriefContext(
        total_demand=float(scenario_fc["Forecast_Units"].sum()),
        horizon_weeks=int(scenario_fc["Week_No"].max()),
        promo_lift_adj_pct=lift_adj * 100,
        lead_time_delay_weeks=delay_weeks,
        stockout_skus=at_risk["SKU_ID"].tolist(),
        excess_dollars=float(summary["Capital_At_Risk"].sum()),
        budget_variance_pct=float((total_cons / total_budget - 1) * 100) if total_budget else 0.0,
        top_risks=top_risks,
        top_variances=top_var,
        master_data_issue_count=len(audit),
        master_data_highlights=[f"{k}: {v}" for k, v in highlights.items()],
    )


def build_prompt(c: BriefContext) -> str:
    return f"""
Review this week's POS data and forward forecast:
- Total {c.horizon_weeks}-Week Demand: {c.total_demand:,.0f} units
- Consensus vs budget: {c.budget_variance_pct:+.1f}%
- Stockout / below-safety-stock SKUs: {', '.join(c.stockout_skus) or 'none'}
- Top supply risks: {'; '.join(c.top_risks) or 'none'}
- Overstock / capital at risk (>120 days cover): ${c.excess_dollars:,.0f}
- Promo uplift assumption adjustment: {c.promo_lift_adj_pct:+.0f}%
- Supplier lead-time delay simulated: +{c.lead_time_delay_weeks} weeks
- Largest budget variances: {'; '.join(c.top_variances) or 'none'}
- Master data issues: {c.master_data_issue_count} ({'; '.join(c.master_data_highlights) or 'none'})

Draft a concise, 3-paragraph S&OP Executive Brief for the cross-functional Commercial and Supply Chain team:
1. Category demand trends and promotional performance.
2. Supply risks (potential stockouts) and recommended immediate purchase actions.
3. Inventory opportunities and master data adjustments.
""".strip()


def template_brief(c: BriefContext) -> str:
    """Deterministic fallback used when no API key is configured."""
    risk = ("; ".join(c.top_risks)) or "no SKUs breach safety stock under the current scenario"
    var = ("; ".join(c.top_variances)) or "no material budget variances"
    return (
        f"**Demand & promotions.** The {c.horizon_weeks}-week unconstrained forecast is {c.total_demand:,.0f} units, "
        f"{c.budget_variance_pct:+.1f}% versus budget, with promo uplift assumptions adjusted by {c.promo_lift_adj_pct:+.0f}%. "
        f"Largest budget variances: {var}.\n\n"
        f"**Supply risks & actions.** With a +{c.lead_time_delay_weeks}-week supplier delay, "
        f"{len(c.stockout_skus)} SKU(s) are at risk ({', '.join(c.stockout_skus) or 'none'}). Priorities: {risk}. "
        f"Raise purchase orders for the suggested quantities immediately and expedite where cover is under lead time.\n\n"
        f"**Inventory & master data.** ${c.excess_dollars:,.0f} of stock sits beyond the forward-cover limit; "
        f"hold replenishment and redirect volume to promotions where possible. "
        f"{c.master_data_issue_count} master data issue(s) were found ({'; '.join(c.master_data_highlights) or 'none'}) "
        f"and should be fixed before the next ERP replenishment run."
    )


def _complete(system: str, user: str, settings: Settings) -> str:
    from openai import OpenAI

    kwargs = {"api_key": settings.llm_api_key,
              "base_url": settings.llm_base_url or "https://api.openai.com/v1"}
    resp = OpenAI(**kwargs).chat.completions.create(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return (resp.choices[0].message.content or "").strip()


def generate_brief(c: BriefContext, settings: Settings | None = None) -> tuple[str, str]:
    """Return (text, source) where source is 'llm', or a note explaining the fallback."""
    s = settings or get_settings()
    if not s.llm_enabled:
        return template_brief(c), "template (set OPENAI_API_KEY in .env to enable the LLM)"
    try:
        return _complete(SYSTEM_PROMPT, build_prompt(c), s), f"llm ({s.llm_model})"
    except Exception as exc:  # network/auth/quota: never break the app
        return template_brief(c), f"template (LLM call failed: {type(exc).__name__})"


def generate_master_data_memo(audit: pd.DataFrame, settings: Settings | None = None) -> tuple[str, str]:
    s = settings or get_settings()
    if audit.empty:
        return "No master data issues detected.", "rules"
    table = audit.head(30).to_csv(index=False)
    fallback = "\n".join(f"- **{r.SKU_ID}** {r.Issue}: {r.Detail}. {r.Suggested_Action}" for r in audit.head(15).itertuples())
    if not s.llm_enabled:
        return fallback, "template (set OPENAI_API_KEY in .env to enable the LLM)"
    prompt = ("Below are master data exceptions found by a rules engine. Write a short memo for the supply planning "
              "and master data team: group by issue type, rank by planning impact, and give clear owner-ready fixes.\n\n" + table)
    try:
        return _complete(SYSTEM_PROMPT, prompt, s), f"llm ({s.llm_model})"
    except Exception as exc:
        return fallback, f"template (LLM call failed: {type(exc).__name__})"
