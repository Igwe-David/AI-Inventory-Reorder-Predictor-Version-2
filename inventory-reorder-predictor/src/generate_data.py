"""Stage 2 - SYNTHETIC data generator.

*** The data produced here is SYNTHETIC: created for prototype development. ***
*** It does NOT come from any real Owerri business.                         ***

How it works: for each of 20 products we simulate daily customer demand
(weekday pattern, seasons, festive and month-end effects, promotions, slow
trends, random noise, rare unpredictable spikes) and a shop owner who reorders
by rule of thumb. Stock levels therefore rise, fall and sometimes hit zero
(stock-outs), so sales are cut off on those days - as in real shops.
Finally a few "messy" records are injected (duplicates, gaps, typos) to
imitate manual record keeping and to give the preprocessing stage real work.

Run:  python -m src.generate_data
"""
import numpy as np
import pandas as pd

from src import config
from src.feature_engineering import get_season, is_festive_array, is_payday_array

WEEKDAY = np.array([0.90, 0.90, 0.95, 1.00, 1.15, 1.35, 0.85])  # Mon..Sun
FESTIVE_MULT = {"Beverages": 1.7, "Food": 1.5, "Household": 1.2,
                "Toiletries": 1.15, "Pharmacy": 1.1}
PAYDAY_MULT = 1.12

COLUMNS = [
    "product_id", "product_name", "category", "date", "units_sold",
    "current_inventory", "unit_price", "promo_flag", "discount_pct",
    "day_of_week", "month", "season", "is_festive", "is_payday_period",
    "supplier_lead_time_days", "prev_day_demand", "prev_inventory",
    "previous_reorder_qty", "reorder_qty_placed", "stockout_flag",
]


def simulate_product(p, dates, rng):
    n = len(dates)
    base = p["base_demand"]
    seasons = np.array([get_season(m) for m in dates.month])
    season_mult = np.array([p["seasonal"][s] for s in seasons])
    festive = is_festive_array(dates)
    payday = is_payday_array(dates)
    trend = np.clip(1 + np.cumsum(rng.normal(0, 0.003, n)), 0.75, 1.35)

    # Promotions: short discount campaigns
    promo = np.zeros(n, dtype=int)
    disc = np.zeros(n, dtype=int)
    t = 0
    while t < n:
        if rng.random() < 0.025:
            length = int(rng.integers(3, 8))
            d = int(rng.choice([5, 10, 15, 20]))
            promo[t:t + length] = 1
            disc[t:t + length] = d
            t += length + 3
        else:
            t += 1

    # Slow price inflation in small steps
    list_price = np.empty(n)
    cur = float(p["price"])
    for i in range(n):
        if rng.random() < 0.006:
            cur *= 1 + rng.uniform(0.03, 0.08)
        list_price[i] = round(cur, -1)
    price_effect = (list_price / p["price"]) ** -0.3

    mean = (base * trend * WEEKDAY[dates.dayofweek] * season_mult
            * np.where(festive == 1, FESTIVE_MULT[p["category"]], 1.0)
            * np.where(payday == 1, PAYDAY_MULT, 1.0)
            * (1 + 0.03 * disc) * price_effect)
    r = 10.0  # smaller r = noisier demand
    demand = rng.negative_binomial(r, r / (r + mean))
    spikes = rng.random(n) < 0.01  # rare unpredictable surges
    demand = np.where(spikes, (demand * rng.uniform(1.8, 3.0, n)).astype(int), demand)

    # Inventory simulation with a rule-of-thumb owner
    stock = int(base * 10)
    pending = []
    quoted = p["lead_time"]
    sold = np.zeros(n, int); inv = np.zeros(n, int)
    ordered = np.zeros(n, int); stockout = np.zeros(n, int); lead_q = np.zeros(n, int)
    for i in range(n):
        if rng.random() < 0.004:
            quoted = int(np.clip(quoted + rng.choice([-1, 1]), 1, 8))
        lead_q[i] = quoted
        stock += sum(q for d, q in pending if d <= i)
        pending = [(d, q) for d, q in pending if d > i]
        stockout[i] = int(demand[i] > stock)
        sold[i] = min(demand[i], stock)
        stock -= sold[i]
        recent = max(sold[max(0, i - 13):i + 1].mean(), 0.3 * base)
        if not pending and stock <= recent * rng.uniform(2.5, 6.0):
            qty = int(np.ceil(recent * rng.uniform(8, 16)))
            delay = int(rng.choice([-1, 0, 0, 0, 0, 1, 2, 3]))  # supplier is sometimes late
            pending.append((i + max(1, quoted + delay), qty))
            ordered[i] = qty
        inv[i] = stock

    df = pd.DataFrame({
        "product_id": p["id"], "product_name": p["name"], "category": p["category"],
        "date": dates, "units_sold": sold, "current_inventory": inv,
        "unit_price": list_price, "promo_flag": promo, "discount_pct": disc,
        "day_of_week": dates.dayofweek, "month": dates.month, "season": seasons,
        "is_festive": festive, "is_payday_period": payday,
        "supplier_lead_time_days": lead_q,
        "reorder_qty_placed": ordered, "stockout_flag": stockout,
    })
    df["prev_day_demand"] = df["units_sold"].shift(1).fillna(0).astype(int)
    df["prev_inventory"] = df["current_inventory"].shift(1).bfill().astype(int)
    last_order = df["reorder_qty_placed"].where(df["reorder_qty_placed"] > 0).ffill().shift(1)
    df["previous_reorder_qty"] = last_order.fillna(0).astype(int)
    return df[COLUMNS]


def inject_messiness(df, rng):
    """Imitate manual record keeping: gaps, typos, duplicates."""
    df = df.copy()
    n = len(df)
    df.loc[rng.choice(df.index, int(0.004 * n), replace=False), "units_sold"] = np.nan
    df.loc[rng.choice(df.index, int(0.003 * n), replace=False), "current_inventory"] = np.nan
    valid = df.index[df["units_sold"].notna()]
    typo = rng.choice(valid, int(0.0015 * n), replace=False)
    df.loc[typo, "units_sold"] = df.loc[typo, "units_sold"] * 10        # extra zero typed
    neg = rng.choice(valid, int(0.0005 * n), replace=False)
    df.loc[neg, "units_sold"] = -1                                      # impossible value
    df = df.drop(rng.choice(df.index, int(0.004 * n), replace=False))   # missing days
    dups = df.sample(int(0.003 * n), random_state=int(rng.integers(1_000_000)))
    df = pd.concat([df, dups])                                          # entered twice
    return df.sort_values(["product_id", "date"], kind="stable").reset_index(drop=True)


def generate_dataset(start=config.DATA_START, end=config.DATA_END,
                     seed=config.RANDOM_SEED, messy=True, products=None):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    frames = [simulate_product(p, dates, rng) for p in (products or config.PRODUCT_CATALOG)]
    df = pd.concat(frames, ignore_index=True)
    return inject_messiness(df, rng) if messy else df


def main():
    df = generate_dataset()
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.RAW_FILE, index=False)
    print("SYNTHETIC dataset created for prototype development (not real shop data).")
    print(f"Saved {len(df):,} rows to {config.RAW_FILE}")
    print(f"Products: {df['product_id'].nunique()} | "
          f"Dates: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Stock-out days: {df['stockout_flag'].mean():.1%} of rows")
    print("\nAverage units sold per day, by category:")
    print(df.groupby("category")["units_sold"].mean().round(2).to_string())


if __name__ == "__main__":
    main()
