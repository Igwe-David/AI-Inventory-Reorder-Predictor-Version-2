# Demonstration scenario (hypothetical)

> **Hypothetical scenario.** A mini-mart owner in Owerri checks "Bottled water (pack)".
> The sales history is the last rows of our **synthetic** dataset; the stock, lead time and safety stock
> are values assumed for the demonstration. None of this is data from a real business.

Run it yourself: `python -m src.demo` (after the pipeline has been run).

| Stage | What happens | Result in this run |
|---|---|---|
| 1. Historical data | Read the last days of bottled-water sales and stock | Shelf was empty on 4 of the last 7 days (stock-out) |
| 2. Preprocessing | Duplicates removed, gaps filled, typos capped | 62 duplicates, 82 missing days, 174 missing sales, 71 outliers fixed (whole dataset) |
| 3. Owner's inputs | Stock 18, lead time 2 days, safety stock 10, 29 units sold in last 7 days, 30-day average 6.3/day | |
| 4. Features | Calendar and sales features built for the model | payday falls in the coming week, no promotion |
| 5. ML prediction | Gradient Boosting forecasts the next 7 days | about 40 units (5.7 per day) |
| 6. Inventory calculation | Lead-time demand 5.7 x 2 = 11.4; reorder point = 11.4 + 10 = 21.4; order-up-to = 5.7 x 9 + 10 = 61.4 | stock position 18 <= 21.4 |
| 7. Decision | **REORDER**, because 18 is not below 11.4, so it is not urgent | order 61.4 - 18 = **44 units** |

Output shown to the user:

```
Product: Bottled water (pack)
Current Stock: 18 units
Predicted Demand: 51 units (next 9 days)
Reorder Status: REORDER
Recommended Quantity: 44 units
```

**Teaching point:** the recent sales (4.1/day) were low *because the shelf was empty*, not because customers stopped
buying. The model cannot know this, so it slightly under-forecasts. The safety stock and the owner's judgement
protect against this; it is one of the documented limitations.
