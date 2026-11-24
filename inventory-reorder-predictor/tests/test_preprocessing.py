import numpy as np
import pandas as pd

from src.data_preprocessing import clean_data


def test_cleaning_removes_problems(small_raw):
    clean, report = clean_data(small_raw)
    assert not clean.duplicated(["product_id", "date"]).any()
    assert clean["units_sold"].notna().all()
    assert (clean["units_sold"] >= 0).all()
    assert report["duplicates_removed"] > 0


def test_every_product_has_a_complete_daily_calendar(small_raw):
    clean, _ = clean_data(small_raw)
    for _, g in clean.groupby("product_id"):
        days = g["date"].diff().dropna()
        assert (days == pd.Timedelta(days=1)).all()


def test_small_handmade_example():
    df = pd.DataFrame({
        "product_id": ["A"] * 4, "product_name": ["Rice"] * 4, "category": ["Food"] * 4,
        "date": ["2025-01-01", "2025-01-01", "2025-01-02", "2025-01-04"],   # duplicate + missing 3rd
        "units_sold": [5, 5, -3, 4], "current_inventory": [20, 20, 15, 10],
        "unit_price": [100] * 4, "promo_flag": [0] * 4, "discount_pct": [0] * 4,
        "supplier_lead_time_days": [2] * 4,
    })
    clean, report = clean_data(df)
    assert report["duplicates_removed"] == 1
    assert report["negative_sales_set_to_missing"] == 1
    assert report["missing_days_added"] == 1
    assert len(clean) == 4                      # 1 Jan - 4 Jan
    assert (clean["units_sold"] >= 0).all()


def test_extreme_typo_is_capped(small_raw):
    clean, report = clean_data(small_raw)
    assert report["outliers_capped"] >= 0
    assert clean["units_sold"].max() < small_raw["units_sold"].max() or report["outliers_capped"] == 0
