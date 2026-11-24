# SIWES report structure - AI Inventory Reorder Prediction System

Legend: **[Implemented]** = built and tested in this project. **[Future work]** = proposed, not built.
Numbers come from the reference run (seed 42); re-check `models/model_comparison.csv` after retraining.

## 1. Introduction
Small and medium businesses in Owerri, such as mini-marts, supermarkets, pharmacies and small wholesalers, must decide daily what to restock. This project builds a decision-support prototype that combines a machine-learning demand forecast with standard inventory rules. **[Implemented]** as a working Streamlit application.

## 2. Background of the problem
Many shops reorder by experience. Suppliers are often in other cities, so deliveries take days; demand changes with weekends, month-end salaries, the rainy/dry/harmattan seasons, promotions and festive periods such as Christmas. Mistakes are costly in either direction.

## 3. Statement of the problem
Businesses often do not know when to reorder or how much to buy. This causes stock-outs and lost sales, excess inventory, and inefficient use of capital.

## 4. Aim
To develop a prototype AI/ML system that forecasts product demand and recommends when and how much to reorder.

## 5. Objectives
1. Build a documented synthetic dataset of daily sales and inventory. **[Implemented]**
2. Clean the data and engineer features without data leakage. **[Implemented]**
3. Train and compare Linear Regression, Random Forest and Gradient Boosting demand models against a naive baseline. **[Implemented]**
4. Convert forecasts into reorder decisions using reorder point, safety stock and order-up-to rules. **[Implemented]**
5. Provide a Streamlit interface for non-programmers. **[Implemented]**
6. Test the system and document its limitations. **[Implemented]**
7. Validate the system on real shop data. **[Future work]**

## 6. Significance
Shows how low-cost AI can support small-business inventory decisions, can reduce stock-outs and idle capital if validated on real data, and runs on an ordinary laptop with free software.

## 7. Scope
In scope: demand forecasting (7-day), reorder decision, quantity, explanation, single-product entry in a web app. Out of scope: multi-shop management, live POS integration, perishables/expiry tracking, supplier price negotiation, automatic ordering.

## 8. System requirements
Windows laptop, Python 3.10+, about 2 GB RAM, no internet or paid service needed at run time. Libraries: pandas, NumPy, scikit-learn, matplotlib, joblib, Streamlit, pytest.

## 9. Data collection
**Synthetic data created for prototype development** (`src/generate_data.py`): 20 products, 5 categories, 2024-01-01 to 2026-10-31, about 20,700 product-days. A simulated rule-of-thumb owner reorders, so stock-outs occur on about 7% of days. Messy records were injected deliberately. It does **not** represent real Owerri businesses. **[Implemented]** Collecting real records with a shop owner's permission is **[Future work]**.

## 10. Data preprocessing
`src/data_preprocessing.py` **[Implemented]**: type conversion, duplicate removal (62 rows), impossible values treated as missing (10), complete daily calendar (82 missing days added), missing sales filled with product median (174), outlier capping at Q3 + 3 x IQR per product (71 values), categorical encoding (one-hot) and scaling for Linear Regression inside the model pipeline.

## 11. Feature engineering
`src/feature_engineering.py` **[Implemented]**. Only information known at the end of day t is used: units sold today, in the last 7 days, 30-day average, trend (recent pace / 30-day pace), day of week, month, season, price, planned promotion and discount in the coming week, festive and payday days in the coming week. Target = units sold on days t+1 to t+7. Supplier lead time and current stock are deliberately **not** demand features: they do not cause demand and are used by the inventory rules. Leakage controls: features look backwards only, a chronological split is used, a 7-day gap separates train and test, and a unit test verifies the target equals the next 7 days' sales.

## 12. Machine learning methodology
Supervised regression. Four candidates: naive baseline (next week = last week), Linear Regression (simple, transparent, cannot capture non-linear effects), Random Forest (non-linear, robust, harder to interpret), Gradient Boosting (often most accurate, more tuning and overfitting risk). Selection by 5-fold time-series cross-validated MAE on the training period; simplest model within 2% of the best wins. **[Implemented]**

## 13. Model training
`src/train_model.py` **[Implemented]**. Train: 15,840 rows (up to 2026-04-01). Test: 4,000 rows (from 2026-04-08). The selected model type is refitted on all data and saved with joblib; the app loads it and never retrains.

## 14. Model evaluation
**[Implemented]** MAE = average miss in units; MSE/RMSE = squared miss (RMSE punishes large misses); R² = share of variation explained; WAPE = total error as a percentage of total sales. R² is not used alone.

| Model | Test MAE | Test RMSE | Test R² | WAPE % |
|---|---|---|---|---|
| Naive baseline | 7.61 | 10.53 | 0.51 | 30.7 |
| Linear Regression | 5.92 | 8.23 | 0.70 | 23.9 |
| Random Forest | 5.96 | 8.04 | 0.71 | 24.0 |
| **Gradient Boosting** | **5.62** | **7.72** | **0.74** | **22.7** |

Gradient Boosting has the lowest error and cuts MAE by about 26% versus the naive baseline. The three ML models are close, so the gain from complexity is modest. Average weekly demand in the data is about 27 units, so a typical miss of about 5.6 units is still sizeable: the forecast is guidance, not certainty. Decision-level evaluation (simulating stock-outs under the full policy) is **[Future work]**.

Explainability: permutation importance shows the forecast relies mainly on the 30-day average demand, then price, planned discount, category and the last 7 days' sales. This is association, not causation; for example price partly acts as a product identifier because products have different price levels.

## 15. System architecture
Data layer (CSV files) -> preprocessing -> feature engineering -> ML models (saved `.joblib`) -> prediction module -> inventory rules -> Streamlit interface. The ML component and the business-rule component are separate modules (`train_model.py`/`prediction.py` vs `inventory_logic.py`). **[Implemented]**

## 16. Application development
`app/app.py` **[Implemented]**: product selection, stock, sales and lead-time inputs, optional safety stock, date and promotion; outputs predicted demand, status (No reorder / Reorder / Urgent), quantity, estimated cost, reasons and the numbers behind them. Inputs are validated with clear messages. 34 pytest tests cover loading, preprocessing, features, prediction, reorder logic and invalid input.

## 17. Results
Reference demonstration (hypothetical scenario on synthetic data, see `docs/DEMO_SCENARIO.md`): stock 18, lead time 2 days, safety stock 10 -> forecast about 40 units for 7 days -> reorder point 21.4 -> status REORDER, order 44 units. All values are computed by the system.

## 18. Limitations
Synthetic data and simulated owner behaviour; stock-outs hide true demand (the demo shows this); no prediction of sudden events, price shocks or supplier failures; simplified seasons; one-shop view; no expiry handling for pharmacies; moderate accuracy (about 23% WAPE); the system reduces but cannot eliminate stock-outs.

## 19. Conclusion
The prototype shows that a simple, explainable ML forecast plus standard inventory rules can produce practical reorder recommendations on a normal laptop. Its value for real businesses must be confirmed with real data.

## 20. Recommendations / future work
All **[Future work]**: collect real sales and stock records (with permission); estimate true demand around stock-outs; simulate decision outcomes (stock-outs avoided, capital tied up); per-product models; multi-shop support; expiry-aware pharmacy logic; mobile or WhatsApp alerts; scheduled retraining; supplier reliability tracking.
