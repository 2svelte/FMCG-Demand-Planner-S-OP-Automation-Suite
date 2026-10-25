# OmniPlan: FMCG Demand Planner & S&OP Automation Suite

Streamlit app that pairs deterministic statistical forecasting with an LLM "S&OP narrative agent" for a
simulated Australian personal-care / FMCG category (Chemist Warehouse, Priceline, Coles).

| Capability | Where |
|---|---|
| 12-week Holt-Winters baseline by SKU x retailer, baseline vs promo-lift decomposition | `omniplan/forecast.py` |
| Risk & opportunity scenarios (promo-lift slider, lead-time delay, stockout + capital-at-risk) | `omniplan/scenario.py` |
| Consensus vs budget with commentary (editable overrides) | `omniplan/scenario.py`, *Executive S&OP* tab |
| Master data hygiene audit (discontinued-with-demand, missing MOQ/lead time, odd multiples, >200% WoW swings) | `omniplan/master_data.py` |
| LLM executive brief + master data memo (template fallback with no API key) | `omniplan/llm.py` |
| Excel pack with live formulas and heat-maps | `omniplan/excel_export.py` |

## Layout

```
apps/dashboard/app.py        Streamlit UI
packages/omniplan/           Core library (src/omniplan) + tests
data/                        Bundled pos_weekly_data.csv, master_sku_data.csv, promo_calendar.csv
scripts/generate_data.py     Regenerates the synthetic data (seeded)
.env.example                 Copy to .env
```

## Run

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt pytest
copy .env.example .env        # optionally add OPENAI_API_KEY
streamlit run apps/dashboard/app.py
pytest packages/omniplan/tests
```

The app loads the bundled CSVs by default; upload your own POS dump (and optionally master data / promo
calendar) from the sidebar. Required columns are validated in `omniplan/loaders.py`.

## Modelling notes

- **Baseline**: promo weeks are removed and interpolated, non-promo outliers (>2.5x rolling median) are scrubbed, then
  damped-trend additive Holt-Winters (52-week season when >=104 weeks of history) is fitted per SKU x retailer.
- **Promo lift** = (promo units - base units) / base units, converted to an elasticity per unit of discount depth and
  applied to the forward promo calendar. The sidebar slider scales this lift.
- **Inventory**: receipts follow the *planned* forecast, shifted by the lead-time delay; demand follows the scenario.
  Stockout alert = projected inventory below safety stock; capital at risk = value above `OMNIPLAN_MAX_COVER_DAYS` (120) of forward cover.
- **Budget** = same weeks last year x (1 + `Budget_Growth_Pct`).
- **LLM**: any OpenAI-compatible endpoint via `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_BASE_URL`.
