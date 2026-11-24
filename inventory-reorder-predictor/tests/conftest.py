import pytest

from src import config
from src.generate_data import generate_dataset


@pytest.fixture(scope="session")
def small_raw():
    """A small, messy, synthetic dataset (4 products, 150 days) built in memory."""
    return generate_dataset("2025-01-01", "2025-05-30", seed=1, messy=True,
                            products=config.PRODUCT_CATALOG[:4])


@pytest.fixture(scope="session")
def model_bundle():
    if not config.MODEL_FILE.exists():
        pytest.skip("Model not trained yet - run: python -m src.train_model")
    from src.prediction import load_model
    return load_model()


@pytest.fixture
def valid_inputs():
    return dict(product_name="Bottled water (pack)", category="Beverages",
                current_stock=35, sales_last_7d=63, avg_daily_demand_30d=8.6,
                lead_time_days=2, unit_price=1800, safety_stock=10,
                decision_date="2026-06-10", promo_planned=False)
