# AI Inventory Reorder Prediction System (SIWES project)

A decision-support tool that helps small and medium businesses decide **when to reorder** a product and
**how much to order**. A machine-learning model forecasts demand; business rules turn the forecast into a decision.

> **Data notice:** the system is trained on **synthetic data created for prototype development**
> (see `docs/DATA_DICTIONARY.md`). It is not data from any real Owerri business and does not represent Owerri
> businesses in general. Results are guidance, not guarantees; the system cannot eliminate stock-outs.

## How it works

```
Historical data -> preprocessing -> feature engineering -> train & compare models -> saved model
                                                                                        |
User inputs (app) -> validation -> features -> ML demand forecast -> inventory rules -> decision + quantity
```

| Part | Type | Job |
|---|---|---|
| Demand forecaster (`train_model.py`, `prediction.py`) | **Machine learning** | Predicts units sold over the next 7 days |
| Inventory logic (`inventory_logic.py`) | **Business rules, no ML** | Reorder point, safety stock, status, quantity |

```
daily_rate       = predicted 7-day demand / 7
lead_time_demand = daily_rate x lead_time_days
reorder_point    = lead_time_demand + safety_stock
order_up_to      = daily_rate x (lead_time_days + 7) + safety_stock
quantity         = order_up_to - (stock on hand + stock already on order)

URGENT REORDER     stock on hand <= lead_time_demand
REORDER            stock position <= reorder_point
NO REORDER NEEDED  otherwise
```
Safety stock = the buffer against forecast error and late suppliers. The app lets you type your own, or suggests
`1.65 x (typical daily forecast error) x sqrt(lead time)` (about 95% service level).

## Project structure

```
inventory-reorder-predictor/
├── data/raw/            synthetic_sales_inventory.csv  (created by generate_data.py)
├── data/processed/      clean data, feature table, cleaning report (created by the pipeline)
├── notebooks/           01_data_exploration.ipynb (charts for your report)
├── src/
│   ├── config.py              all settings and the product catalogue
│   ├── generate_data.py       synthetic dataset generator
│   ├── data_preprocessing.py  load, validate, clean (duplicates, gaps, outliers)
│   ├── feature_engineering.py calendar/lag/rolling features and the 7-day target
│   ├── train_model.py         compare 3 models + baseline, select, evaluate, save
│   ├── inventory_logic.py     reorder point, safety stock, status, quantity
│   ├── prediction.py          input validation + full pipeline used by the app
│   ├── explain.py             "why" explanations and feature importance
│   └── demo.py                step-by-step demonstration scenario
├── models/              best_model.joblib, comparison table, metrics, charts (created by training)
├── app/app.py           Streamlit web application
├── tests/               pytest tests
├── docs/                data dictionary, demo scenario, SIWES report structure
├── requirements.txt
└── README.md
```

## Setup on Windows

Install Python 3.10+ from python.org (tick **"Add Python to PATH"**). Open **Command Prompt** in the project folder.

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
(PowerShell: if activation is blocked, run `Set-ExecutionPolicy -Scope Process Bypass` first, or just use Command Prompt.)

## Run the pipeline (always from the project folder)

| Step | Command | Expected output |
|---|---|---|
| 1. Create the dataset | `python -m src.generate_data` | "Saved 20,680 rows ... Stock-out days: ~7%" |
| 2. Clean it | `python -m src.data_preprocessing` | Cleaning report (duplicates, gaps, outliers fixed) |
| 3. Build features | `python -m src.feature_engineering` | "Feature table: 19,980 rows x 17 columns" |
| 4. Train, compare, save | `python -m src.train_model` | Comparison table, selected model, "Saved model to ...best_model.joblib" |
| 5. Demonstration | `python -m src.demo` | 7 labelled steps ending in a recommendation |
| 6. Tests | `pytest` | "34 passed" |
| 7. Start the app | `streamlit run app/app.py` | Browser opens at http://localhost:8501; the layout adapts to tablet and mobile widths |

The app **loads** `models/best_model.joblib`; it never retrains. Retrain only when you have new data.

## Results from the reference run (seed 42)

Test period starts 2026-04-08 (the most recent 20% of dates). Units are per 7-day forecast.

| Model | CV MAE | Test MAE | Test RMSE | Test R² |
|---|---|---|---|---|
| Naive baseline (last 7 days) | - | 7.61 | 10.53 | 0.51 |
| Linear Regression | 7.34 | 5.92 | 8.23 | 0.70 |
| Random Forest | 6.49 | 5.96 | 8.04 | 0.71 |
| **Gradient Boosting (selected)** | **6.02** | **5.62** | **7.72** | **0.74** |

Gradient Boosting was chosen because it had the lowest cross-validated error on the training period.
The three ML models are fairly close on the test period, so the extra complexity buys a modest gain;
all of them clearly beat the naive baseline. Your numbers may differ slightly on another machine - check `models/model_comparison.csv`.

## Common errors

| Message | Cause and fix |
|---|---|
| `No module named 'src'` | You are not in the project folder. `cd` into `inventory-reorder-predictor`, use `python -m src.xxx` |
| `Dataset not found ... generate_data` | Run step 1 first |
| `Feature file missing` / `Trained model not found` | Run the earlier steps in order |
| `'streamlit' is not recognized` | Activate the venv (`venv\Scripts\activate`) and run `pip install -r requirements.txt` |
| `sparse_output` error in training | Old scikit-learn: `pip install --upgrade scikit-learn` |
| Port already in use | `streamlit run app/app.py --server.port 8502` |
| Warning about model/scikit-learn version | Retrain with `python -m src.train_model` on your own machine |

## Limitations

Synthetic data; real shops will be noisier. Stock-outs hide true demand (recorded sales are lower than what
customers wanted). Demand shocks, price jumps, strikes, fuel scarcity and supplier failures are not predicted.
Seasonal effects are simplified. Pharmacy expiry dates and batches are not modelled. Accuracy is limited: the
reference model is typically about 23% off in total weekly units (WAPE). Decision-level simulation of stock-outs
is future work. See `docs/SIWES_REPORT.md` for the full report structure with *implemented* vs *future work*.
