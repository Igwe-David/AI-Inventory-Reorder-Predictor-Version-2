"""Stage 7 - Explainability: WHY did the model predict this demand?

Two views:
  * Global  - which inputs matter most overall (permutation importance, saved in training)
  * Local   - "what-if" tests for ONE prediction: switch a factor off and see how much
              the forecast changes.

IMPORTANT: these show what the model has LEARNED from past data (association).
They do not prove that a factor CAUSES sales to change.
"""
import pandas as pd

from src import config


def load_global_importance():
    if not config.IMPORTANCE_FILE.exists():
        return None
    return pd.read_csv(config.IMPORTANCE_FILE)


def _predict(model, row):
    return float(max(0.0, model.predict(row)[0]))


def explain_forecast(bundle, row):
    """Return a list of plain-English sentences explaining one forecast."""
    model = bundle["model"]
    r = row.iloc[0]
    pred = _predict(model, row)
    avg30 = float(r["avg_daily_demand_30d"])
    last7_rate = float(r["sales_last_7d"]) / 7
    notes = [f"Starting point: your 30-day average ({avg30:.1f} units/day) points to about "
             f"{avg30 * 7:.0f} units per week. The model forecasts {pred:.0f} units for the next 7 days."]
    threshold = max(1.0, 0.03 * max(pred, 1.0))

    def effect(change):
        alt = row.copy()
        change(alt)
        return pred - _predict(model, alt)

    def promo_off(a):
        a["promo_planned"] = 0
        a["planned_discount_pct"] = 0.0
    if r["promo_planned"] == 1:
        d = effect(promo_off)
        if abs(d) >= threshold:
            notes.append(f"The planned promotion ({r['planned_discount_pct']:.0f}% off) "
                         f"{'adds' if d > 0 else 'removes'} about {abs(d):.0f} units.")

    if r["is_festive_window"] == 1:
        d = effect(lambda a: a.__setitem__("is_festive_window", 0))
        if abs(d) >= threshold:
            notes.append(f"A festive period (Christmas/New Year/Easter) in the coming week "
                         f"{'adds' if d > 0 else 'removes'} about {abs(d):.0f} units.")

    if r["payday_in_window"] == 1:
        d = effect(lambda a: a.__setitem__("payday_in_window", 0))
        if abs(d) >= threshold:
            notes.append(f"Month-end salary spending {'adds' if d > 0 else 'removes'} "
                         f"about {abs(d):.0f} units.")

    def neutral_recent(a):
        a["sales_last_7d"] = avg30 * 7
        a["sales_last_day"] = avg30
        a["demand_trend"] = 1.0
    d = effect(neutral_recent)
    if abs(d) >= threshold:
        direction = "higher" if last7_rate > avg30 else "lower"
        notes.append(f"Sales over the last 7 days ({last7_rate:.1f}/day) are {direction} than your "
                     f"30-day average, which moves the forecast {'up' if d > 0 else 'down'} by "
                     f"about {abs(d):.0f} units.")

    if len(notes) == 1:
        notes.append("No special factor (promotion, festive period, payday or a change in recent "
                     "sales) changes the forecast much - it follows your normal selling pattern.")
    notes.append("These are patterns the model learned from past data. They indicate association, "
                 "not proven cause and effect.")
    return notes
