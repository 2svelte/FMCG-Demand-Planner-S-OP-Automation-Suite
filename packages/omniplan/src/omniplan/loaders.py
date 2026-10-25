"""Loading and validating POS / master / promo-calendar data."""
from __future__ import annotations

import pandas as pd

from .config import Settings, get_settings

POS_REQUIRED = [
    "Week_End_Date", "Retailer", "SKU_ID", "SKU_Description", "Category", "Units_Sold_POS",
    "Base_Price", "Promo_Price", "Promo_Flag", "Inventory_On_Hand", "Supplier_Lead_Time_Weeks",
]
MASTER_REQUIRED = [
    "SKU_ID", "SKU_Description", "Category", "Brand", "Status", "Unit_Cost", "MOQ",
    "Order_Multiple", "Lead_Time_Weeks", "Safety_Stock_Weeks", "Budget_Growth_Pct",
]
PROMO_REQUIRED = ["Retailer", "SKU_ID", "Week_End_Date", "Promo_Depth_Pct"]


class DataValidationError(ValueError):
    pass


def _check(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(f"{name} is missing required columns: {', '.join(missing)}")


def clean_pos(df: pd.DataFrame) -> pd.DataFrame:
    _check(df, POS_REQUIRED, "POS file")
    out = df.copy()
    out["Week_End_Date"] = pd.to_datetime(out["Week_End_Date"], errors="coerce")
    if out["Week_End_Date"].isna().any():
        raise DataValidationError("POS file has unparseable Week_End_Date values")
    for col in ["Units_Sold_POS", "Base_Price", "Promo_Price", "Promo_Flag", "Inventory_On_Hand", "Supplier_Lead_Time_Weeks"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["Units_Sold_POS"] = out["Units_Sold_POS"].fillna(0).clip(lower=0)
    out["Promo_Flag"] = out["Promo_Flag"].fillna(0).astype(int)
    out["Promo_Depth"] = (1 - out["Promo_Price"] / out["Base_Price"]).where(out["Promo_Flag"] == 1, 0.0).fillna(0.0).clip(0, 0.9)
    return out.sort_values(["SKU_ID", "Retailer", "Week_End_Date"]).reset_index(drop=True)


def clean_master(df: pd.DataFrame) -> pd.DataFrame:
    _check(df, MASTER_REQUIRED, "Master data file")
    out = df.copy()
    for col in ["Unit_Cost", "MOQ", "Order_Multiple", "Lead_Time_Weeks", "Safety_Stock_Weeks", "Budget_Growth_Pct"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def clean_promo(df: pd.DataFrame) -> pd.DataFrame:
    _check(df, PROMO_REQUIRED, "Promo calendar")
    out = df.copy()
    out["Week_End_Date"] = pd.to_datetime(out["Week_End_Date"], errors="coerce")
    out["Promo_Depth_Pct"] = pd.to_numeric(out["Promo_Depth_Pct"], errors="coerce").fillna(0)
    return out.dropna(subset=["Week_End_Date"])


def load_pos(source=None, settings: Settings | None = None) -> pd.DataFrame:
    s = settings or get_settings()
    return clean_pos(pd.read_csv(source if source is not None else s.pos_path))


def load_master(source=None, settings: Settings | None = None) -> pd.DataFrame:
    s = settings or get_settings()
    return clean_master(pd.read_csv(source if source is not None else s.master_path))


def load_promo_calendar(source=None, settings: Settings | None = None) -> pd.DataFrame:
    s = settings or get_settings()
    path = source if source is not None else s.promo_calendar_path
    try:
        return clean_promo(pd.read_csv(path))
    except FileNotFoundError:
        return pd.DataFrame(columns=PROMO_REQUIRED)
