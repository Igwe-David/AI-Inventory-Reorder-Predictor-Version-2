"""Stage 5 - Train, compare and save demand-forecasting models.

Run:  python -m src.train_model

Models compared (all predict units sold over the next 7 days):
  0. Naive baseline   - "next week = last week"         (benchmark, no learning)
  1. Linear Regression
  2. Random Forest Regressor
  3. Gradient Boosting Regressor
Selection rule: lowest cross-validated MAE on the TRAINING period; if several models
are within 2% of the best, the simplest one wins. The test period is only used for
the final, honest report.
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config
from src.feature_engineering import (CATEGORICAL_FEATURES, FEATURE_COLUMNS,
                                     NUMERIC_FEATURES, TARGET, build_features)

SIMPLICITY_ORDER = ["Linear Regression", "Random Forest", "Gradient Boosting"]


def build_models():
    def prep(scale):
        return ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("num", StandardScaler() if scale else "passthrough", NUMERIC_FEATURES),
        ])
    return {
        "Linear Regression": Pipeline([("prep", prep(True)), ("model", LinearRegression())]),
        "Random Forest": Pipeline([("prep", prep(False)), ("model", RandomForestRegressor(
            n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=config.RANDOM_SEED))]),
        "Gradient Boosting": Pipeline([("prep", prep(False)), ("model", GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=3, subsample=0.8,
            random_state=config.RANDOM_SEED))]),
    }


def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": r2_score(y_true, y_pred),
        "WAPE_%": 100 * np.abs(y_true - y_pred).sum() / y_true.sum(),
    }


def main():
    config.MODELS_DIR.mkdir(exist_ok=True)
    if not config.FEATURE_FILE.exists():
        raise SystemExit("Feature file missing. Run the earlier stages first "
                         "(generate_data, data_preprocessing, feature_engineering).")
    data = pd.read_csv(config.FEATURE_FILE, parse_dates=["date"])

    # ---- Time-ordered split: learn from the past, test on the most recent 20% ----
    dates = np.sort(data["date"].unique())
    cutoff = pd.Timestamp(dates[int(len(dates) * 0.8)])
    gap = pd.Timedelta(days=config.HORIZON_DAYS)   # target windows overlap by up to 7 days
    train = data[data["date"] < cutoff - gap].sort_values("date")
    test = data[data["date"] >= cutoff]
    X_train, y_train = train[FEATURE_COLUMNS], train[TARGET]
    X_test, y_test = test[FEATURE_COLUMNS], test[TARGET]
    print(f"Train rows: {len(train):,} (up to {(cutoff - gap).date()}) | "
          f"Test rows: {len(test):,} (from {cutoff.date()})")

    n_products = data["product_id"].nunique()
    cv = TimeSeriesSplit(n_splits=5, gap=config.HORIZON_DAYS * n_products)

    rows, fitted, test_pred = [], {}, {}
    baseline = X_test["sales_last_7d"].to_numpy()            # naive: next 7 days = last 7 days
    rows.append({"Model": "Naive baseline (last 7 days)", "CV_MAE": np.nan, **regression_metrics(y_test, baseline)})
    test_pred["Naive baseline (last 7 days)"] = baseline

    for name, pipe in build_models().items():
        cv_mae = -cross_val_score(pipe, X_train, y_train, cv=cv,
                                  scoring="neg_mean_absolute_error").mean()
        pipe.fit(X_train, y_train)
        pred = np.maximum(0, pipe.predict(X_test))
        fitted[name], test_pred[name] = pipe, pred
        rows.append({"Model": name, "CV_MAE": cv_mae, **regression_metrics(y_test, pred)})
        print(f"  trained {name}")

    table = pd.DataFrame(rows).set_index("Model")
    table.round(3).to_csv(config.COMPARISON_FILE)

    # ---- Selection by cross-validated MAE, preferring simpler models ----
    cv_scores = table.loc[SIMPLICITY_ORDER, "CV_MAE"]
    near_best = cv_scores[cv_scores <= cv_scores.min() * 1.02].index
    best_name = [m for m in SIMPLICITY_ORDER if m in near_best][0]

    print("\n=== Model comparison on the TEST period (units per 7 days) ===")
    print(table.round(3).to_string())
    print(f"\nSelected model: {best_name}  "
          f"(lowest CV error, simplest among models within 2% of the best)")

    # ---- Error by category (used to suggest safety stock) ----
    pred_best = test_pred[best_name]
    by_cat = {}
    for cat in config.CATEGORIES:
        m = (test["category"] == cat).to_numpy()
        if m.sum() > 0:
            by_cat[cat] = float(np.sqrt(mean_squared_error(y_test[m], pred_best[m])))

    # ---- Explainability: permutation importance (test period) ----
    imp = permutation_importance(fitted[best_name], X_test, y_test, n_repeats=5,
                                 scoring="neg_mean_absolute_error",
                                 random_state=config.RANDOM_SEED, n_jobs=-1)
    importance = (pd.DataFrame({"feature": FEATURE_COLUMNS, "mae_increase": imp.importances_mean,
                                "std": imp.importances_std})
                  .sort_values("mae_increase", ascending=False))
    importance.round(4).to_csv(config.IMPORTANCE_FILE, index=False)

    # ---- Charts ----
    fig, ax = plt.subplots(figsize=(7, 4))
    table["MAE"].plot.barh(ax=ax, color=["#bbbbbb"] + ["#2b7bba"] * 3)
    ax.set_xlabel("MAE on test period (units per 7 days) - lower is better")
    ax.set_title("Model comparison"); ax.invert_yaxis(); fig.tight_layout()
    fig.savefig(config.MODELS_DIR / "model_comparison.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_test, pred_best, s=5, alpha=0.3)
    top = max(y_test.max(), pred_best.max())
    ax.plot([0, top], [0, top], "r--", label="perfect prediction")
    ax.set_xlabel("Actual demand, next 7 days"); ax.set_ylabel("Predicted demand")
    ax.set_title(f"{best_name}: actual vs predicted (test period)"); ax.legend()
    fig.tight_layout(); fig.savefig(config.MODELS_DIR / "actual_vs_predicted.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    importance.head(10).iloc[::-1].plot.barh(x="feature", y="mae_increase", ax=ax, legend=False)
    ax.set_xlabel("Increase in error when the feature is scrambled")
    ax.set_title("Which inputs the forecast relies on"); fig.tight_layout()
    fig.savefig(config.MODELS_DIR / "feature_importance.png", dpi=150); plt.close(fig)

    # ---- Final model: refit the chosen type on ALL data, then save ----
    final = clone(build_models()[best_name]).fit(data[FEATURE_COLUMNS], data[TARGET])
    bundle = {
        "model": final,
        "model_name": best_name,
        "feature_columns": FEATURE_COLUMNS,
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred_best))),
        "test_rmse_by_category": by_cat,
        "data_note": "Trained on SYNTHETIC data created for prototype development.",
    }
    joblib.dump(bundle, config.MODEL_FILE)
    config.METRICS_FILE.write_text(json.dumps({
        "selected_model": best_name,
        "test_metrics": table.loc[best_name, ["MAE", "MSE", "RMSE", "R2", "WAPE_%"]].round(4).to_dict(),
        "baseline_test_metrics": table.iloc[0][["MAE", "MSE", "RMSE", "R2", "WAPE_%"]].round(4).to_dict(),
        "train_rows": len(train), "test_rows": len(test), "test_starts": str(cutoff.date()),
    }, indent=2))
    print(f"\nSaved model to {config.MODEL_FILE}")
    print("\nTop features (permutation importance):")
    print(importance.head(6).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
