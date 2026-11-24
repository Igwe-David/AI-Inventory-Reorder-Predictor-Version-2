import pytest

from src.inventory_logic import calculate_reorder
from src.prediction import recommend, validate_inputs


@pytest.mark.parametrize("change, message", [
    ({"current_stock": -5}, "Current stock"),
    ({"lead_time_days": 0}, "lead time"),
    ({"lead_time_days": "abc"}, "lead time"),
    ({"unit_price": 0}, "Unit price"),
    ({"sales_last_7d": -1}, "last 7 days"),
    ({"category": "Furniture"}, "Category"),
    ({"planned_discount_pct": 150}, "discount"),
    ({"decision_date": "not-a-date"}, "Date"),
    ({"avg_daily_demand_30d": float("nan")}, "Average daily demand"),
])
def test_bad_inputs_are_rejected_with_clear_message(valid_inputs, change, message):
    with pytest.raises(ValueError, match=f"(?i){message}"):
        validate_inputs({**valid_inputs, **change})


def test_missing_required_value(valid_inputs):
    del valid_inputs["current_stock"]
    with pytest.raises(ValueError):
        validate_inputs(valid_inputs)


def test_suspicious_numbers_give_a_warning_not_an_error(valid_inputs):
    _, warnings = validate_inputs({**valid_inputs, "sales_last_7d": 700, "avg_daily_demand_30d": 2})
    assert warnings


def test_inventory_logic_rejects_negative_stock():
    with pytest.raises(ValueError):
        calculate_reorder(-1, 70, 2, 5)


def test_recommend_rejects_bad_input(model_bundle, valid_inputs):
    with pytest.raises(ValueError):
        recommend({**valid_inputs, "current_stock": -1}, model_bundle)
