"""Stage 8 - Streamlit web app.   Run:  streamlit run app/app.py"""
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))          # lets "from src import ..." work from this folder

import pandas as pd
import streamlit as st

from src import config
from src.explain import load_global_importance
from src.prediction import load_model, recommend

st.set_page_config(page_title="Inventory Reorder Assistant", page_icon="📦", layout="wide")

st.markdown("""
<style>
@media (max-width: 900px) {
    .block-container {
        padding: 2rem 1.5rem 3rem;
    }

    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap;
        column-gap: 1rem;
        row-gap: 1rem;
    }

    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
    [data-testid="stHorizontalBlock"] > [data-testid="column"] {
        flex: 0 1 calc(50% - 0.5rem) !important;
        min-width: calc(50% - 0.5rem) !important;
    }

    [data-testid="stTabs"] [role="tablist"] {
        overflow-x: auto;
        flex-wrap: nowrap;
    }

    [data-testid="stTabs"] button[role="tab"] {
        flex: 0 0 auto;
    }
}

@media (max-width: 600px) {
    .block-container {
        padding: 1.25rem 1rem 2rem;
    }

    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
    [data-testid="stHorizontalBlock"] > [data-testid="column"] {
        flex-basis: 100% !important;
        min-width: 100% !important;
    }

    h1 {
        font-size: clamp(1.75rem, 8vw, 2.5rem);
    }

    [data-testid="stMetricValue"] {
        font-size: clamp(1.25rem, 6vw, 2rem);
    }

    pre {
        overflow-wrap: anywhere;
        white-space: pre-wrap;
    }
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_bundle():
    return load_model()      # loaded from disk once - the app never retrains


st.title("📦 Inventory Reorder Assistant")
st.caption("Decision-support prototype. The model was trained on SYNTHETIC data created for "
           "prototype development - treat results as guidance, not as certainty.")

try:
    bundle = get_bundle()
except FileNotFoundError as err:
    st.error(str(err))
    st.stop()

CATALOG = {p["name"]: p for p in config.PRODUCT_CATALOG}
OTHER = "Other product (type the name)"

tab_check, tab_perf, tab_how = st.tabs(["Reorder check", "Model performance", "How it works"])

# =============================================================== Reorder check
with tab_check:
    choice = st.selectbox("Product", list(CATALOG) + [OTHER])
    preset = CATALOG.get(choice)
    key = choice  # changing product resets the defaults below

    with st.form("reorder_form"):
        c1, c2 = st.columns(2)
        name = choice if preset else c1.text_input("Product name", "My product")
        category = c2.selectbox("Category", config.CATEGORIES,
                                index=config.CATEGORIES.index(preset["category"]) if preset else 0,
                                key=f"cat_{key}")
        a, b, c = st.columns(3)
        stock = a.number_input("Current stock (units)", 0, 1_000_000, 35, key=f"stock_{key}")
        on_order = b.number_input("Already ordered, not yet arrived (units)", 0, 1_000_000, 0, key=f"oo_{key}")
        price = c.number_input("Unit price (₦)", 1, 100_000_000,
                               int(preset["price"]) if preset else 1000, key=f"price_{key}")

        d1, d2, d3 = st.columns(3)
        default_avg = float(preset["base_demand"]) if preset else 5.0
        sales7 = d1.number_input("Units sold in the last 7 days", 0.0, 1_000_000.0,
                                 float(round(default_avg * 7)), 1.0, key=f"s7_{key}")
        avg30 = d2.number_input("Average units sold per day (last 30 days)", 0.0, 100_000.0,
                                default_avg, 0.1, key=f"a30_{key}")
        today_sales = d3.number_input("Units sold today (optional)", 0.0, 100_000.0,
                                      float(round(default_avg)), 1.0, key=f"st_{key}")

        e1, e2, e3 = st.columns(3)
        lead = e1.number_input("Supplier delivery time (days)", 1, 60,
                               int(preset["lead_time"]) if preset else 3, key=f"lead_{key}")
        use_suggested = e2.checkbox("Let the system suggest the safety stock", value=True)
        my_safety = e3.number_input("My safety stock (used if box is unticked)", 0, 1_000_000, 10, key=f"ss_{key}")

        f1, f2, f3 = st.columns(3)
        when = f1.date_input("Today's date", date.today())
        promo = f2.checkbox("A promotion is planned in the next 7 days")
        discount = f3.slider("Planned discount (%)", 0, 50, 10)

        submitted = st.form_submit_button("Check stock", type="primary")

    if submitted:
        inputs = dict(product_name=name, category=category, current_stock=stock,
                      quantity_on_order=on_order, unit_price=price, sales_last_7d=sales7,
                      avg_daily_demand_30d=avg30, sales_last_day=today_sales,
                      lead_time_days=lead, safety_stock=None if use_suggested else my_safety,
                      decision_date=when, promo_planned=promo, planned_discount_pct=discount)
        try:
            r = recommend(inputs, bundle)
        except ValueError as err:
            st.error(f"Please check your entries: {err}")
        else:
            for w in r["warnings"]:
                st.warning(w)
            qty = r["recommended_quantity"]
            if r["status"] == "URGENT REORDER":
                st.error(f"🔴 URGENT REORDER - order about **{qty} units** of {name} today")
            elif r["status"] == "REORDER":
                st.warning(f"🟠 REORDER - order about **{qty} units** of {name}")
            else:
                st.success("🟢 NO REORDER NEEDED right now")

            m = st.columns(5)
            m[0].metric("Current stock", f"{r['current_stock']:.0f}")
            m[1].metric(f"Predicted demand (next {r['coverage_days']:.0f} days)", f"{r['coverage_demand']:.0f}")
            m[2].metric("Reorder point", f"{r['reorder_point']:.0f}")
            m[3].metric("Recommended quantity", qty)
            m[4].metric("Estimated cost", f"₦{r['estimated_order_cost']:,.0f}")

            st.code(r["summary_text"], language=None)

            with st.expander("Why this recommendation?", expanded=True):
                st.markdown("**Stock decision:** " + r["reason"])
                st.markdown("**Demand forecast:**")
                for line in r["forecast_explanation"]:
                    st.markdown(f"- {line}")

            with st.expander("The numbers behind it"):
                days = "n/a" if r["days_of_cover"] is None else f"{r['days_of_cover']:.1f}"
                st.table(pd.DataFrame({
                    "Item": ["Predicted demand, next 7 days (ML)", "Expected sales per day",
                             f"Expected sales during delivery wait ({r['lead_time_days']:.0f} days)",
                             f"Safety stock ({r['safety_stock_source']})", "Reorder point",
                             f"Order-up-to level ({r['coverage_days']:.0f} days of cover + safety stock)",
                             "Days your current stock will last"],
                    "Value": [f"{r['predicted_demand_7d']:.0f}", f"{r['daily_rate']:.1f}",
                              f"{r['lead_time_demand']:.0f}", f"{r['safety_stock']:.0f}",
                              f"{r['reorder_point']:.0f}", f"{r['order_up_to_level']:.0f}", days]}))
                st.caption(f"Forecast model: {r['model_name']}. Suggested safety stock for this "
                           f"product type: {r['suggested_safety_stock']} units.")

# =========================================================== Model performance
with tab_perf:
    if config.COMPARISON_FILE.exists():
        st.subheader("Model comparison (test period, units per 7 days)")
        st.dataframe(pd.read_csv(config.COMPARISON_FILE).round(3), hide_index=True)
        st.caption("MAE = average error in units. RMSE punishes big misses more. "
                   "R² = share of the ups and downs the model explains. WAPE = total error as % of total sales.")
        cols = st.columns(3)
        for col, f in zip(cols, ["model_comparison", "actual_vs_predicted", "feature_importance"]):
            p = config.MODELS_DIR / f"{f}.png"
            if p.exists():
                col.image(str(p))
        st.caption("Feature importance shows what the model leans on - association, not proof of cause.")
    else:
        st.info("Train the model first to see its performance.")

# ================================================================= How it works
with tab_how:
    st.markdown("""
**Two separate parts**

1. **Machine learning (demand forecast):** learns from past sales how many units will sell in the next 7 days.
2. **Inventory rules (business logic):** turns that forecast into a decision.

```
reorder point   = expected sales while waiting for delivery + safety stock
order-up-to     = expected sales over (delivery time + 7 days) + safety stock
quantity        = order-up-to level - stock you have - stock already ordered
```
* **URGENT** - your shelf stock is likely to run out *before* the delivery arrives.
* **REORDER** - stock has reached the reorder point.
* **NO REORDER NEEDED** - you have enough for now.

**Limits:** trained on synthetic data; cannot foresee sudden events, price shocks or supplier failures;
sales on days when the shelf was empty understate real demand. It reduces the risk of stock-outs - it cannot eliminate it.
""")
