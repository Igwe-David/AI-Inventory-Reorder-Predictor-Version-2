"""Central settings. Change values here, not inside other files."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---- Folders and files ----
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

RAW_FILE = RAW_DIR / "synthetic_sales_inventory.csv"
CLEAN_FILE = PROCESSED_DIR / "clean_sales_inventory.csv"
FEATURE_FILE = PROCESSED_DIR / "model_features.csv"
CLEANING_REPORT_FILE = PROCESSED_DIR / "cleaning_report.json"

MODEL_FILE = MODELS_DIR / "best_model.joblib"
COMPARISON_FILE = MODELS_DIR / "model_comparison.csv"
METRICS_FILE = MODELS_DIR / "metrics.json"
IMPORTANCE_FILE = MODELS_DIR / "feature_importance.csv"

# ---- Forecasting and inventory settings ----
HORIZON_DAYS = 7      # the ML model predicts total demand for the next 7 days
REVIEW_DAYS = 7       # how often the owner reviews stock / places orders
SERVICE_Z = 1.65      # z-score for ~95% service level (used to suggest safety stock)
RANDOM_SEED = 42

# ---- Synthetic data period ----
DATA_START = "2024-01-01"
DATA_END = "2026-10-31"

CATEGORIES = ["Beverages", "Food", "Household", "Toiletries", "Pharmacy"]
SEASONS = ["harmattan", "dry", "rainy"]

# Example product catalogue used by the data generator and the app.
# SYNTHETIC ASSUMPTIONS: prices (NGN), daily demand and lead times are illustrative
# values chosen for prototype development. They are NOT measured market data.
_FLAT = {"harmattan": 1.0, "dry": 1.0, "rainy": 1.0}
PRODUCT_CATALOG = [
    dict(id="P001", name="Bottled water (pack)",      category="Beverages", price=1800,  base_demand=9.0, lead_time=2, seasonal={"harmattan": 1.0,  "dry": 1.2, "rainy": 0.9}),
    dict(id="P002", name="Soft drink (crate)",        category="Beverages", price=6500,  base_demand=5.0, lead_time=2, seasonal={"harmattan": 1.1,  "dry": 1.2, "rainy": 0.9}),
    dict(id="P003", name="Malt drink (carton)",       category="Beverages", price=12000, base_demand=3.0, lead_time=4, seasonal=_FLAT),
    dict(id="P004", name="Noodles (carton)",          category="Food",      price=9000,  base_demand=5.0, lead_time=3, seasonal=_FLAT),
    dict(id="P005", name="Rice (50kg bag)",           category="Food",      price=78000, base_demand=1.5, lead_time=5, seasonal=_FLAT),
    dict(id="P006", name="Vegetable oil (5L)",        category="Food",      price=14000, base_demand=4.0, lead_time=4, seasonal=_FLAT),
    dict(id="P007", name="Garri (paint bucket)",      category="Food",      price=2500,  base_demand=8.0, lead_time=2, seasonal=_FLAT),
    dict(id="P008", name="Sugar (carton)",            category="Food",      price=36000, base_demand=2.0, lead_time=5, seasonal=_FLAT),
    dict(id="P009", name="Evaporated milk (carton)",  category="Food",      price=28000, base_demand=3.0, lead_time=4, seasonal=_FLAT),
    dict(id="P010", name="Tomato paste (carton)",     category="Food",      price=24000, base_demand=2.5, lead_time=4, seasonal=_FLAT),
    dict(id="P011", name="Detergent powder (carton)", category="Household", price=22000, base_demand=3.0, lead_time=5, seasonal=_FLAT),
    dict(id="P012", name="Toilet tissue (pack)",      category="Household", price=2200,  base_demand=6.0, lead_time=3, seasonal=_FLAT),
    dict(id="P013", name="Mosquito coil (pack)",      category="Household", price=1200,  base_demand=4.0, lead_time=3, seasonal={"harmattan": 0.9,  "dry": 0.9, "rainy": 1.35}),
    dict(id="P014", name="Bathing soap (carton)",     category="Toiletries", price=18000, base_demand=3.0, lead_time=4, seasonal=_FLAT),
    dict(id="P015", name="Toothpaste (carton)",       category="Toiletries", price=15000, base_demand=2.0, lead_time=4, seasonal=_FLAT),
    dict(id="P016", name="Paracetamol tablets (pack)", category="Pharmacy", price=800,   base_demand=7.0, lead_time=3, seasonal={"harmattan": 1.1,  "dry": 1.0, "rainy": 1.0}),
    dict(id="P017", name="Vitamin C tablets (bottle)", category="Pharmacy", price=1500,  base_demand=3.0, lead_time=3, seasonal={"harmattan": 1.15, "dry": 1.0, "rainy": 1.0}),
    dict(id="P018", name="ORS sachets (box)",         category="Pharmacy",  price=2500,  base_demand=2.0, lead_time=4, seasonal={"harmattan": 0.9,  "dry": 0.9, "rainy": 1.3}),
    dict(id="P019", name="Hand sanitizer (bottle)",   category="Pharmacy",  price=1200,  base_demand=2.5, lead_time=3, seasonal=_FLAT),
    dict(id="P020", name="Adhesive plasters (box)",   category="Pharmacy",  price=1800,  base_demand=1.5, lead_time=4, seasonal=_FLAT),
]
