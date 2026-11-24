"""Stage 3 - Data loading and cleaning.

Problems handled (real record books have all of them):
  * rows entered twice                    -> remove duplicates
  * impossible values (negative sales)    -> treat as missing
  * missing days / missing numbers        -> fill sensibly, and count them
  * typing mistakes (e.g. 5 typed as 50)  -> cap extreme outliers (not delete)
Every fix is counted and written to a cleaning report.

Run:  python -m src.data_preprocessing
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.feature_engineering import get_season, is_festive_array, is_payday_array

REQUIRED_COLUMNS = [
    "product_id", "product_name", "category", "date", "units_sold",
    "current_inventory", "unit_price", "promo_flag", "discount_pct",
    "supplier_lead_time_days",
]
NUMERIC_COLUMNS = ["units_sold", "current_inventory", "unit_price", "promo_flag",
                   "discount_pct", "supplier_lead_time_days"]


def load_raw_data(path=config.RAW_FILE):
    """Read the raw CSV and check that the needed columns exist."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\nCreate it first with:  python -m src.generate_data")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"The dataset is missing required columns: {missing}")
    return df


def clean_data(df):
    """Return (clean_dataframe, report_dictionary)."""
    report = {"rows_in": int(len(df))}
    df = df.copy()

    # 1. Correct data types
    for c in NUMERIC_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    bad_dates = int(df["date"].isna().sum())
    df = df.dropna(subset=["date", "product_id"])
    report["invalid_dates_removed"] = bad_dates

    # 2. Duplicates (same product on the same day)
    before = len(df)
    df = df.drop_duplicates(subset=["product_id", "date"], keep="first")
    report["duplicates_removed"] = int(before - len(df))

    # 3. Impossible values become "missing"
    neg_sales = int((df["units_sold"] < 0).sum())
    df.loc[df["units_sold"] < 0, "units_sold"] = np.nan
    df.loc[df["current_inventory"] < 0, "current_inventory"] = np.nan
    df.loc[df["unit_price"] <= 0, "unit_price"] = np.nan
    report["negative_sales_set_to_missing"] = neg_sales

    # 4. Complete the calendar and fill gaps, one product at a time
    parts = []
    missing_days = 0
    for pid, g in df.sort_values("date").groupby("product_id"):
        g = g.set_index("date")
        full_range = pd.date_range(g.index.min(), g.index.max(), freq="D")
        missing_days += len(full_range) - len(g)
        g = g.reindex(full_range)
        g.index.name = "date"
        for c in ["product_id", "product_name", "category"]:
            g[c] = g[c].ffill().bfill()
        for c in ["unit_price", "supplier_lead_time_days", "current_inventory"]:
            g[c] = g[c].ffill().bfill()
        for c in ["promo_flag", "discount_pct"]:
            g[c] = g[c].fillna(0)
        parts.append(g.reset_index())
    df = pd.concat(parts, ignore_index=True)
    report["missing_days_added"] = int(missing_days)

    # 5. Missing sales -> the product's median daily sales
    n_missing_sales = int(df["units_sold"].isna().sum())
    df["units_sold"] = df["units_sold"].fillna(df.groupby("product_id")["units_sold"].transform("median"))
    report["missing_sales_filled_with_median"] = n_missing_sales

    # 6. Outliers: cap at Q3 + 3*IQR for each product (capping keeps real spikes, but tames typos)
    def cap(s):
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        return s.clip(upper=q3 + 3 * (q3 - q1))
    capped = df.groupby("product_id")["units_sold"].transform(cap)
    report["outliers_capped"] = int((capped < df["units_sold"]).sum())
    df["units_sold"] = capped.round().astype(int)
    df["current_inventory"] = df["current_inventory"].round().astype(int)

    # 7. Rebuild calendar and "previous" columns so they are consistent after cleaning
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["season"] = df["month"].map(get_season)
    df["is_festive"] = is_festive_array(df["date"])
    df["is_payday_period"] = is_payday_array(df["date"])
    df = df.sort_values(["product_id", "date"]).reset_index(drop=True)
    by = df.groupby("product_id")
    df["prev_day_demand"] = by["units_sold"].shift(1).fillna(0).astype(int)
    df["prev_inventory"] = by["current_inventory"].shift(1).fillna(df["current_inventory"]).astype(int)
    for c in ["previous_reorder_qty", "reorder_qty_placed", "stockout_flag"]:
        df[c] = df[c].fillna(0).astype(int) if c in df.columns else 0

    report["rows_out"] = int(len(df))
    return df, report


def main():
    raw = load_raw_data()
    clean, report = clean_data(raw)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    clean.to_csv(config.CLEAN_FILE, index=False)
    config.CLEANING_REPORT_FILE.write_text(json.dumps(report, indent=2))
    print("=== Cleaning report ===")
    for k, v in report.items():
        print(f"{k:35s} {v:>8,}")
    print(f"\nSaved clean data to {config.CLEAN_FILE}")


if __name__ == "__main__":
    main()
