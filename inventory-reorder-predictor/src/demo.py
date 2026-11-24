"""Stage 10 - Step-by-step demonstration scenario.

Run:  python -m src.demo

HYPOTHETICAL SCENARIO: a small mini-mart owner in Owerri checks "Bottled water (pack)".
 * Sales history  = the last rows of our SYNTHETIC dataset (not real shop data).
 * Stock, lead time, safety stock = assumed values for the demonstration.
"""
import pandas as pd

from src import config
from src.data_preprocessing import clean_data, load_raw_data
from src.explain import load_global_importance
from src.feature_engineering import features_from_inputs
from src.prediction import load_model, recommend

PRODUCT = "Bottled water (pack)"


def heading(text):
    print("\n" + "=" * 70 + f"\n{text}\n" + "=" * 70)


def main():
    heading("HYPOTHETICAL SCENARIO (synthetic data - not a real business)")
    print("Mini-mart owner checks 'Bottled water (pack)'.")
    print("Assumed: 18 packs on the shelf, supplier delivers in 2 days, safety stock 10 packs.")

    heading("STEP 1 - Historical data (raw, synthetic)")
    raw = load_raw_data()
    shown = raw[raw["product_name"] == PRODUCT].tail(5)
    print(shown[["date", "units_sold", "current_inventory", "unit_price", "promo_flag"]].to_string(index=False))

    heading("STEP 2 - Data preprocessing")
    clean, report = clean_data(raw)
    for k in ["duplicates_removed", "missing_days_added", "missing_sales_filled_with_median", "outliers_capped"]:
        print(f"{k:35s} {report[k]:>6}")

    heading("STEP 3 - Inputs the owner types into the app")
    hist = clean[clean["product_name"] == PRODUCT].sort_values("date")
    empty_days = int((hist["current_inventory"].tail(7) == 0).sum())
    if empty_days:
        print(f"NOTE: the shelf was empty on {empty_days} of the last 7 days, so recent sales\n"
              "understate true demand (customers could not buy). This is a known limitation.\n")
    last_date = hist["date"].max()
    inputs = dict(
        product_name=PRODUCT, category="Beverages",
        current_stock=18, quantity_on_order=0,
        sales_last_7d=float(hist["units_sold"].tail(7).sum()),
        avg_daily_demand_30d=float(hist["units_sold"].tail(30).mean()),
        sales_last_day=float(hist["units_sold"].iloc[-1]),
        lead_time_days=2, safety_stock=10,
        unit_price=float(hist["unit_price"].iloc[-1]),
        promo_planned=False, decision_date=last_date)
    for k, v in inputs.items():
        print(f"{k:22s} {v}")

    heading("STEP 4 - Features built for the model")
    row = features_from_inputs(inputs["category"], inputs["unit_price"], inputs["sales_last_7d"],
                               inputs["avg_daily_demand_30d"], last_date, inputs["sales_last_day"])
    print(row.T.round(2).to_string(header=False))

    bundle = load_model()
    result = recommend(inputs, bundle)
    heading(f"STEP 5 - ML prediction ({result['model_name']})")
    print(f"Predicted demand, next 7 days : {result['predicted_demand_7d']:.0f} units "
          f"({result['daily_rate']:.1f} per day)")

    heading("STEP 6 - Inventory calculation (business rules, no ML)")
    print(f"Demand while waiting for delivery : {result['daily_rate']:.1f} x {result['lead_time_days']:.0f} days = {result['lead_time_demand']:.1f}")
    print(f"Safety stock                      : {result['safety_stock']:.0f}")
    print(f"Reorder point                     : {result['reorder_point']:.1f}")
    print(f"Order-up-to level                 : {result['daily_rate']:.1f} x {result['coverage_days']:.0f} days + {result['safety_stock']:.0f} = {result['order_up_to_level']:.1f}")
    print(f"Stock position (on hand + ordered): {result['current_stock'] + result['quantity_on_order']:.0f}")

    heading("STEP 7 - Decision and recommended quantity")
    print(result["summary_text"])
    print("\nWhy:", result["reason"])
    print("\nForecast explanation:")
    for line in result["forecast_explanation"]:
        print(" -", line)
    imp = load_global_importance()
    if imp is not None:
        print("\nMost influential inputs overall (permutation importance):")
        print(imp.head(4).to_string(index=False))


if __name__ == "__main__":
    main()
