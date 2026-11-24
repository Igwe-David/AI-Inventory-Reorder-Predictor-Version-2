import pytest

from src.inventory_logic import calculate_reorder, suggest_safety_stock

# Example used below: forecast 70 units/week -> 10/day, lead time 2 days, safety stock 5, review 7 days
# lead_time_demand = 20, reorder_point = 25, order_up_to = 10 * (2+7) + 5 = 95


def test_no_reorder_when_stock_is_high():
    r = calculate_reorder(35, 70, 2, 5)
    assert r["status"] == "NO REORDER NEEDED"
    assert r["recommended_quantity"] == 0
    assert r["reorder_point"] == pytest.approx(25)


def test_reorder_when_stock_at_reorder_point():
    r = calculate_reorder(24, 70, 2, 5)
    assert r["status"] == "REORDER"
    assert r["recommended_quantity"] == 71          # 95 - 24


def test_urgent_when_stock_will_not_last_until_delivery():
    r = calculate_reorder(12, 70, 2, 5)
    assert r["status"] == "URGENT REORDER"
    assert r["recommended_quantity"] == 83          # 95 - 12


def test_stock_already_on_order_prevents_double_ordering():
    r = calculate_reorder(12, 70, 2, 5, quantity_on_order=90)
    assert r["status"] == "NO REORDER NEEDED"


def test_zero_demand_forecast_does_not_crash():
    r = calculate_reorder(0, 0, 2, 5)
    assert r["status"] in ("REORDER", "URGENT REORDER")
    assert r["days_of_cover"] is None


def test_safety_stock_grows_with_error_and_lead_time():
    assert suggest_safety_stock(10, 4) > suggest_safety_stock(5, 4)
    assert suggest_safety_stock(10, 8) > suggest_safety_stock(10, 2)
