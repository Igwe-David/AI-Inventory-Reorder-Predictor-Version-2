from src.prediction import recommend


def test_recommendation_has_all_fields(model_bundle, valid_inputs):
    r = recommend(valid_inputs, model_bundle)
    for key in ["status", "recommended_quantity", "predicted_demand_7d", "coverage_demand",
                "reorder_point", "reason", "forecast_explanation", "summary_text"]:
        assert key in r
    assert r["predicted_demand_7d"] >= 0
    assert r["recommended_quantity"] >= 0


def test_forecast_follows_the_inputs_not_hard_coded(model_bundle, valid_inputs):
    low = recommend({**valid_inputs, "avg_daily_demand_30d": 2, "sales_last_7d": 14}, model_bundle)
    high = recommend({**valid_inputs, "avg_daily_demand_30d": 20, "sales_last_7d": 140}, model_bundle)
    assert high["predicted_demand_7d"] > low["predicted_demand_7d"]


def test_low_stock_triggers_reorder_and_high_stock_does_not(model_bundle, valid_inputs):
    low = recommend({**valid_inputs, "current_stock": 3}, model_bundle)
    high = recommend({**valid_inputs, "current_stock": 5000}, model_bundle)
    assert low["status"] in ("REORDER", "URGENT REORDER") and low["recommended_quantity"] > 0
    assert high["status"] == "NO REORDER NEEDED" and high["recommended_quantity"] == 0
