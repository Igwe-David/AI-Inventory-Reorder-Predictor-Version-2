import numpy as np

from src.data_preprocessing import clean_data
from src.feature_engineering import (FEATURE_COLUMNS, TARGET, build_features,
                                     features_from_inputs, window_calendar_features)


def test_features_have_no_missing_values(small_raw):
    clean, _ = clean_data(small_raw)
    feats = build_features(clean)
    assert feats[FEATURE_COLUMNS + [TARGET]].notna().all().all()


def test_no_data_leakage_target_is_future_and_features_are_past(small_raw):
    clean, _ = clean_data(small_raw)
    feats = build_features(clean)
    row = feats.iloc[100]
    hist = clean[clean["product_id"] == row["product_id"]].sort_values("date").reset_index(drop=True)
    i = hist.index[hist["date"] == row["date"]][0]
    assert row["sales_last_7d"] == hist["units_sold"].iloc[i - 6:i + 1].sum()        # past + today
    assert row[TARGET] == hist["units_sold"].iloc[i + 1:i + 8].sum()                 # next 7 days only
    assert np.isclose(row["avg_daily_demand_30d"], hist["units_sold"].iloc[i - 29:i + 1].mean())


def test_calendar_window_flags():
    assert window_calendar_features("2026-12-05")["is_festive_window"] == 1   # 10 Dec falls in window
    assert window_calendar_features("2026-06-10")["is_festive_window"] == 0
    assert window_calendar_features("2026-06-20")["payday_in_window"] == 1    # 25th falls in window
    assert window_calendar_features("2026-06-05")["payday_in_window"] == 0


def test_features_from_user_input_shape():
    row = features_from_inputs("Beverages", 1800, 63, 8.6, "2026-06-10")
    assert list(row.columns) == FEATURE_COLUMNS
    assert row.shape == (1, len(FEATURE_COLUMNS))
    assert row["season"].iloc[0] == "rainy"
    assert np.isclose(row["demand_trend"].iloc[0], 9 / 8.6)
