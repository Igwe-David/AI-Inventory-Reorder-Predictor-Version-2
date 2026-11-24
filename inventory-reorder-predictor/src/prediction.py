"""Stage 7 - Prediction pipeline used by the app.

user inputs -> validation -> features -> ML demand forecast -> inventory rules -> recommendation
"""
import math

import joblib
import pandas as pd

from src import config
from src.explain import explain_forecast
from src.feature_engineering import features_from_inputs
from src.inventory_logic import calculate_reorder, suggest_safety_stock


def load_model(path=config.MODEL_FILE):
    """Load the saved model (the app never retrains it)."""
    try:
        return joblib.load(path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Trained model not found at {path}.\nTrain it first with:  python -m src.train_model")


def _num(inputs, key, label, minimum=0.0, strictly_above=False, maximum=None, default=None):
    value = inputs.get(key, default)
    if value is None:
        raise ValueError(f"{label} is required")
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a number")
    if math.isnan(v) or math.isinf(v):
        raise ValueError(f"{label} must be a real number")
    if (v <= minimum) if strictly_above else (v < minimum):
        raise ValueError(f"{label} must be {'greater than' if strictly_above else 'at least'} {minimum:g}")
    if maximum is not None and v > maximum:
        raise ValueError(f"{label} must be at most {maximum:g}")
    return v


def validate_inputs(inputs):
    """Check what the user typed. Returns (clean_inputs, warnings). Raises ValueError if invalid."""
    clean = {}
    clean["product_name"] = str(inputs.get("product_name", "")).strip() or "Unnamed product"
    category = inputs.get("category")
    if category not in config.CATEGORIES:
        raise ValueError(f"Category must be one of: {', '.join(config.CATEGORIES)}")
    clean["category"] = category
    clean["current_stock"] = _num(inputs, "current_stock", "Current stock")
    clean["quantity_on_order"] = _num(inputs, "quantity_on_order", "Quantity on order", default=0)
    clean["sales_last_7d"] = _num(inputs, "sales_last_7d", "Sales in the last 7 days")
    clean["avg_daily_demand_30d"] = _num(inputs, "avg_daily_demand_30d", "Average daily demand")
    clean["lead_time_days"] = _num(inputs, "lead_time_days", "Supplier lead time", 0, True, maximum=60)
    clean["unit_price"] = _num(inputs, "unit_price", "Unit price", 0, True)
    clean["planned_discount_pct"] = _num(inputs, "planned_discount_pct", "Planned discount", 0, maximum=90, default=0)
    clean["promo_planned"] = bool(inputs.get("promo_planned", False))
    last_day = inputs.get("sales_last_day")
    clean["sales_last_day"] = None if last_day in (None, "") else _num(inputs, "sales_last_day", "Sales today")
    ss = inputs.get("safety_stock")
    clean["safety_stock"] = None if ss in (None, "") else _num(inputs, "safety_stock", "Safety stock")
    try:
        clean["decision_date"] = pd.Timestamp(inputs.get("decision_date"))
        if pd.isna(clean["decision_date"]):
            raise ValueError
    except Exception:
        raise ValueError("Date is not valid")

    warnings = []
    pace = clean["sales_last_7d"] / 7
    if clean["avg_daily_demand_30d"] == 0 and clean["sales_last_7d"] > 0:
        warnings.append("Average daily demand is 0 but there were sales last week - please double-check.")
    elif clean["avg_daily_demand_30d"] > 0 and pace > 5 * clean["avg_daily_demand_30d"]:
        warnings.append("Last week's sales are more than 5x your 30-day average - please double-check the numbers.")
    if clean["sales_last_7d"] == 0 and clean["avg_daily_demand_30d"] > 0:
        warnings.append("No sales last week although the 30-day average is above 0 - was the product out of stock?")
    return clean, warnings


def predict_weekly_demand(bundle, feature_row):
    """ML part: forecast of units sold over the next 7 days (never negative)."""
    return float(max(0.0, bundle["model"].predict(feature_row)[0]))


def recommend(inputs, bundle=None):
    """Full pipeline: returns the forecast, the decision and the explanations."""
    bundle = bundle or load_model()
    x, warnings = validate_inputs(inputs)

    row = features_from_inputs(
        x["category"], x["unit_price"], x["sales_last_7d"], x["avg_daily_demand_30d"],
        x["decision_date"], x["sales_last_day"], x["promo_planned"], x["planned_discount_pct"])
    weekly = predict_weekly_demand(bundle, row)

    rmse = bundle["test_rmse_by_category"].get(x["category"], bundle["test_rmse"])
    suggested = suggest_safety_stock(rmse, x["lead_time_days"])
    safety = suggested if x["safety_stock"] is None else x["safety_stock"]

    result = calculate_reorder(x["current_stock"], weekly, x["lead_time_days"], safety,
                               config.REVIEW_DAYS, x["quantity_on_order"])
    result.update({
        "product_name": x["product_name"],
        "category": x["category"],
        "unit_price": x["unit_price"],
        "suggested_safety_stock": suggested,
        "safety_stock_source": "suggested by the system" if x["safety_stock"] is None else "entered by you",
        "estimated_order_cost": result["recommended_quantity"] * x["unit_price"],
        "forecast_explanation": explain_forecast(bundle, row),
        "warnings": warnings,
        "model_name": bundle["model_name"],
    })
    result["summary_text"] = (
        f"Product: {x['product_name']}\n"
        f"Current Stock: {x['current_stock']:.0f} units\n"
        f"Predicted Demand: {result['coverage_demand']:.0f} units "
        f"(next {result['coverage_days']:.0f} days)\n"
        f"Reorder Status: {result['status']}\n"
        f"Recommended Quantity: {result['recommended_quantity']} units")
    return result
