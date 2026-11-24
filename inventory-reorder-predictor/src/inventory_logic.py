"""Inventory decision layer - BUSINESS RULES, no machine learning here.

The ML model only predicts demand. This file turns that prediction into a decision.

Formulas (daily_rate = predicted 7-day demand / 7):
  lead_time_demand = daily_rate x lead_time_days          what will sell while we wait for delivery
  safety_stock     = buffer against forecast error / late suppliers
  reorder_point    = lead_time_demand + safety_stock      order when stock falls to this level
  order_up_to      = daily_rate x (lead_time + review_days) + safety_stock
  quantity         = order_up_to - (stock on hand + stock already on order)

Status:
  URGENT REORDER     stock on hand <= lead_time_demand  (likely to run out BEFORE delivery)
  REORDER            stock position <= reorder_point
  NO REORDER NEEDED  otherwise
"""
import math

from src import config


def suggest_safety_stock(forecast_rmse_7d, lead_time_days, z=config.SERVICE_Z,
                         horizon=config.HORIZON_DAYS):
    """Suggested buffer = z x (typical daily forecast error) x sqrt(lead time).

    forecast_rmse_7d is the model's typical 7-day error measured on test data.
    """
    if forecast_rmse_7d < 0 or lead_time_days <= 0:
        raise ValueError("forecast error must be >= 0 and lead time must be > 0")
    daily_error = forecast_rmse_7d / math.sqrt(horizon)
    return math.ceil(z * daily_error * math.sqrt(lead_time_days))


def _check(name, value, minimum=0.0, strictly_above=False):
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number")
    if math.isnan(v) or math.isinf(v):
        raise ValueError(f"{name} must be a real number")
    if (v <= minimum) if strictly_above else (v < minimum):
        raise ValueError(f"{name} must be {'greater than' if strictly_above else 'at least'} {minimum:g}")
    return v


def calculate_reorder(current_stock, predicted_demand_next_7d, lead_time_days,
                      safety_stock, review_days=config.REVIEW_DAYS,
                      quantity_on_order=0, horizon=config.HORIZON_DAYS):
    """Turn a demand forecast into a reorder decision. Raises ValueError on bad input."""
    current_stock = _check("Current stock", current_stock)
    demand = _check("Predicted demand", predicted_demand_next_7d)
    lead = _check("Lead time (days)", lead_time_days, 0, strictly_above=True)
    safety = _check("Safety stock", safety_stock)
    review = _check("Review period (days)", review_days, 0, strictly_above=True)
    on_order = _check("Quantity on order", quantity_on_order)

    daily_rate = demand / horizon
    lead_time_demand = daily_rate * lead
    reorder_point = lead_time_demand + safety
    coverage_days = lead + review
    coverage_demand = daily_rate * coverage_days
    order_up_to = coverage_demand + safety
    position = current_stock + on_order

    if position <= reorder_point:
        quantity = max(1, math.ceil(order_up_to - position))
        urgent = current_stock <= lead_time_demand
        status = "URGENT REORDER" if urgent else "REORDER"
    else:
        quantity = 0
        status = "NO REORDER NEEDED"

    days_of_cover = current_stock / daily_rate if daily_rate > 0 else None

    # Plain-English reason
    if status == "NO REORDER NEEDED":
        reason = (f"Your stock position ({position:.0f} units) is above the reorder point "
                  f"({reorder_point:.0f} units = {lead_time_demand:.0f} expected to sell while waiting "
                  f"for delivery + {safety:.0f} safety stock).")
    elif status == "REORDER":
        reason = (f"Your stock position ({position:.0f} units) has fallen to the reorder point "
                  f"({reorder_point:.0f} units). Ordering now covers the next {coverage_days:.0f} days "
                  f"of expected sales plus the safety stock.")
    else:
        reason = (f"Only {current_stock:.0f} units are on the shelf but about {lead_time_demand:.0f} are "
                  f"expected to sell before a delivery arrives in {lead:.0f} day(s). "
                  f"You are likely to run out before new stock arrives.")

    return {
        "status": status,
        "recommended_quantity": int(quantity),
        "current_stock": current_stock,
        "quantity_on_order": on_order,
        "predicted_demand_7d": demand,
        "daily_rate": daily_rate,
        "lead_time_days": lead,
        "review_days": review,
        "lead_time_demand": lead_time_demand,
        "safety_stock": safety,
        "reorder_point": reorder_point,
        "coverage_days": coverage_days,
        "coverage_demand": coverage_demand,
        "order_up_to_level": order_up_to,
        "days_of_cover": days_of_cover,
        "reason": reason,
    }
