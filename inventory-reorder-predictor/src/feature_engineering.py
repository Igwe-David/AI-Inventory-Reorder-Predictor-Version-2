"""Stage 4 - Feature engineering.

Turns the cleaned daily table into one row per product per day with
  * FEATURES  - only information a shop owner knows at the END of day t
  * TARGET    - total units sold over the next 7 days (t+1 ... t+7)

Run:  python -m src.feature_engineering
"""
import numpy as np
import pandas as pd

from src import config

CATEGORICAL_FEATURES = ["category", "season"]
NUMERIC_FEATURES = [
    "unit_price",
    "promo_planned",          # 1 if a promotion is planned within the next 7 days
    "planned_discount_pct",   # biggest planned discount in the next 7 days
    "sales_last_day",         # units sold today
    "sales_last_7d",          # units sold in the last 7 days
    "avg_daily_demand_30d",   # average units per day over the last 30 days
    "demand_trend",           # recent daily pace divided by 30-day pace
    "day_of_week",            # 0 = Monday ... 6 = Sunday
    "month",
    "is_festive_window",      # 1 if a festive day falls in the next 7 days
    "payday_in_window",       # 1 if a month-end salary day falls in the next 7 days
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "demand_next_7d"

# Easter Sunday dates (the date moves every year)
EASTER = {2024: "2024-03-31", 2025: "2025-04-20", 2026: "2026-04-05",
          2027: "2027-03-28", 2028: "2028-04-16"}


# ------------------------------------------------------------------ calendar
def get_season(month):
    """Simple Owerri-style seasons: harmattan (Dec-Feb), dry (Nov, Mar), rainy (Apr-Oct)."""
    if month in (12, 1, 2):
        return "harmattan"
    if month in (11, 3):
        return "dry"
    return "rainy"


def is_festive_array(dates):
    """1 on Christmas/New Year days (10 Dec - 5 Jan) and within 5 days of Easter."""
    d = pd.DatetimeIndex(dates)
    flag = ((d.month == 12) & (d.day >= 10)) | ((d.month == 1) & (d.day <= 5))
    flag = np.asarray(flag)
    for year, easter in EASTER.items():
        e = pd.Timestamp(easter)
        flag = flag | np.asarray((d >= e - pd.Timedelta(days=5)) & (d <= e + pd.Timedelta(days=5)))
    return flag.astype(int)


def is_payday_array(dates):
    """1 during the month-end salary period (25th to 2nd of next month)."""
    d = pd.DatetimeIndex(dates)
    return np.asarray((d.day >= 25) | (d.day <= 2)).astype(int)


def window_calendar_features(decision_date, horizon=config.HORIZON_DAYS):
    """Calendar facts about the NEXT `horizon` days (known in advance)."""
    start = pd.Timestamp(decision_date) + pd.Timedelta(days=1)
    days = pd.date_range(start, periods=horizon)
    return {
        "is_festive_window": int(is_festive_array(days).max()),
        "payday_in_window": int(is_payday_array(days).max()),
    }


# ------------------------------------------------------------ training table
def _future_window(series, horizon, how):
    """Sum or max of the NEXT `horizon` values (t+1 ... t+horizon) for each row t."""
    reversed_shifted = series.shift(-1).iloc[::-1]
    roll = reversed_shifted.rolling(horizon, min_periods=horizon)
    result = roll.sum() if how == "sum" else roll.max()
    return result.iloc[::-1]


def build_features(clean_df, horizon=config.HORIZON_DAYS):
    """Create the model-ready table from the cleaned data."""
    df = clean_df.sort_values(["product_id", "date"]).reset_index(drop=True)
    parts = []
    for _, g in df.groupby("product_id", sort=False):
        g = g.copy()
        sold = g["units_sold"].astype(float)
        # ---- features: only the past and today ----
        g["sales_last_day"] = sold
        g["sales_last_7d"] = sold.rolling(7, min_periods=7).sum()
        g["avg_daily_demand_30d"] = sold.rolling(30, min_periods=30).mean()
        # ---- planned promotions are known in advance by the shop ----
        g["promo_planned"] = _future_window(g["promo_flag"].astype(float), horizon, "max")
        g["planned_discount_pct"] = _future_window(g["discount_pct"].astype(float), horizon, "max")
        # ---- target: what REALLY sold over the following 7 days ----
        g[TARGET] = _future_window(sold, horizon, "sum")
        parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    out["demand_trend"] = (out["sales_last_7d"] / 7) / out["avg_daily_demand_30d"].clip(lower=0.1)

    cal = pd.DataFrame([{"date": d, **window_calendar_features(d, horizon)}
                        for d in out["date"].drop_duplicates()])
    out = out.merge(cal, on="date", how="left")
    out = out.dropna(subset=NUMERIC_FEATURES + [TARGET]).reset_index(drop=True)
    return out[["product_id", "product_name", "date"] + FEATURE_COLUMNS + [TARGET]]


# ------------------------------------------------------------ app / user input
def features_from_inputs(category, unit_price, sales_last_7d, avg_daily_demand_30d,
                         decision_date, sales_last_day=None, promo_planned=0,
                         planned_discount_pct=0):
    """Build the one-row feature table for a product a user typed into the app."""
    d = pd.Timestamp(decision_date)
    if sales_last_day is None:
        sales_last_day = sales_last_7d / 7
    row = {
        "category": category,
        "season": get_season(d.month),
        "unit_price": float(unit_price),
        "promo_planned": int(bool(promo_planned)),
        "planned_discount_pct": float(planned_discount_pct) if promo_planned else 0.0,
        "sales_last_day": float(sales_last_day),
        "sales_last_7d": float(sales_last_7d),
        "avg_daily_demand_30d": float(avg_daily_demand_30d),
        "demand_trend": (sales_last_7d / 7) / max(avg_daily_demand_30d, 0.1),
        "day_of_week": d.dayofweek,
        "month": d.month,
        **window_calendar_features(d),
    }
    return pd.DataFrame([row])[FEATURE_COLUMNS]


def main():
    clean = pd.read_csv(config.CLEAN_FILE, parse_dates=["date"])
    feats = build_features(clean)
    feats.to_csv(config.FEATURE_FILE, index=False)
    print(f"Feature table: {len(feats):,} rows x {len(feats.columns)} columns")
    print(f"Saved to {config.FEATURE_FILE}")
    print("\nExample rows:")
    print(feats.head(3).T.to_string())


if __name__ == "__main__":
    main()
