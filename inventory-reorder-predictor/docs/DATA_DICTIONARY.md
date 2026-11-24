# Data dictionary - `data/raw/synthetic_sales_inventory.csv`

> **SYNTHETIC DATA created for prototype development.** It was produced by `src/generate_data.py`.
> It is not collected from any business in Owerri and must not be described as such.
> Prices, demand levels and lead times are illustrative assumptions.

One row = one product on one day. 20 products, 2024-01-01 to 2026-10-31 (about 20,700 rows after cleaning).

| Column | Meaning | Why it is included |
|---|---|---|
| product_id, product_name | Product identity | Needed to build each product's own history |
| category | Beverages, Food, Household, Toiletries, Pharmacy | Categories sell differently |
| date | Day of the record | All time features, and the time-ordered train/test split |
| units_sold | Units sold that day | The history that the forecast learns from |
| current_inventory | Units on the shelf at end of day | Used by the inventory rules; also reveals stock-outs |
| unit_price | List price in NGN | Prices move demand; also used for order cost |
| promo_flag, discount_pct | Promotion running that day and its discount | Promotions lift demand; shops plan them in advance |
| day_of_week | 0 = Monday ... 6 = Sunday | Weekend peaks |
| month, season | Month; harmattan (Dec-Feb), dry (Nov, Mar), rainy (Apr-Oct) | Seasonal products (e.g. mosquito coils, drinks) |
| is_festive | Christmas/New Year (10 Dec-5 Jan) and +-5 days around Easter | Festive rush |
| is_payday_period | 25th to 2nd of the next month | Month-end salary spending |
| supplier_lead_time_days | Days the supplier normally needs to deliver | Drives the reorder point (not used by the demand model: lead time does not cause demand) |
| prev_day_demand | Units sold the day before | Recent sales history |
| prev_inventory | Stock at the end of the previous day | Context / exploration |
| previous_reorder_qty | Quantity of the most recent earlier order | Context / exploration of ordering habits |
| reorder_qty_placed | Quantity ordered on this day (0 = no order) | Shows how the simulated owner orders |
| stockout_flag | 1 if demand was higher than stock (sales cut off) | Shows where `units_sold` understates true demand |

## Realism features of the simulation
Weekday pattern, seasons, festive and month-end effects, promotions, slow demand drift,
small price rises (inflation), noisy overdispersed counts, rare unpredictable spikes,
supplier delays, and a rule-of-thumb owner whose reorders cause occasional stock-outs (about 7% of days).
Messy records were injected on purpose (duplicates, gaps, missing values, typing mistakes, impossible values)
so that the preprocessing stage has real work. Noise was NOT tuned to make the scores look good.

## Using real data instead
Create a CSV with the same required columns (`product_id, product_name, category, date, units_sold,
current_inventory, unit_price, promo_flag, discount_pct, supplier_lead_time_days`), save it as
`data/raw/synthetic_sales_inventory.csv` (or change `RAW_FILE` in `src/config.py`), then re-run the pipeline.
Get the shop owner's permission and never record customer names or phone numbers.
