"""Create a realistic SYNTHETIC sales log for a small Owerri provisions shop.

Run:  python -m src.generate_data

Replace data/sales_log.csv with real records from a shop to train on real data
(same columns). Product prices below are illustrative, not official.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from src.features import is_festive

ROOT = Path(__file__).resolve().parent.parent
rng = np.random.default_rng(42)

# name, category, average units sold per day, unit price (NGN), usual lead time (days)
PRODUCTS = [
    ("Indomie noodles (carton)",   "Groceries",  5.0,  9000, 3),
    ("Rice 50kg bag",              "Groceries",  1.5, 78000, 5),
    ("Vegetable oil 5L",           "Groceries",  4.0, 14000, 4),
    ("Garri (paint bucket)",       "Groceries",  8.0,  2500, 2),
    ("Spaghetti (carton)",         "Groceries",  3.0, 11000, 3),
    ("Sugar (carton)",             "Groceries",  2.0, 36000, 5),
    ("Bottled water (pack)",       "Beverages", 10.0,  1800, 2),
    ("Soft drink (crate)",         "Beverages",  5.0,  6500, 2),
    ("Malt drink (carton)",        "Beverages",  3.0, 12000, 4),
    ("Evaporated milk (carton)",   "Provisions", 3.0, 28000, 4),
    ("Tomato paste (carton)",      "Provisions", 2.5, 24000, 4),
    ("Detergent powder (carton)",  "Household",  3.0, 22000, 5),
    ("Toilet tissue (pack)",       "Household",  6.0,  2200, 3),
    ("Bathing soap (carton)",      "Toiletries", 3.0, 18000, 4),
    ("Toothpaste (carton)",        "Toiletries", 2.0, 15000, 4),
]
FESTIVE_BOOST = {"Groceries": 1.6, "Beverages": 1.8, "Provisions": 1.5,
                 "Household": 1.2, "Toiletries": 1.15}
WEEKDAY = np.array([0.90, 0.90, 0.95, 1.00, 1.15, 1.35, 0.85])  # Mon..Sun


def main():
    dates = pd.date_range("2024-01-01", "2026-09-30")
    n = len(dates)
    festive = is_festive(dates)
    rows = []
    for name, cat, base, price, lead in PRODUCTS:
        trend = np.clip(1 + np.cumsum(rng.normal(0, 0.004, n)), 0.7, 1.4)
        promo = np.where(rng.random(n) < 0.03, 1.4, 1.0)
        boost = np.where(festive == 1, FESTIVE_BOOST[cat], 1.0)
        mean = base * trend * WEEKDAY[dates.dayofweek] * boost * promo
        sold = rng.poisson(mean)
        # Snapshot of stock on hand (simplification: random level each day)
        stock = (base * rng.uniform(0, 3.0, n) * (lead + 4)).astype(int)
        # Suppliers are sometimes late or early
        lead_day = np.maximum(1, lead + rng.choice([-1, 0, 0, 0, 1, 2], n))
        rows.append(pd.DataFrame({
            "date": dates, "product": name, "category": cat,
            "unit_price": price, "lead_time_days": lead_day,
            "units_sold": sold, "stock_on_hand": stock,
        }))
    log = pd.concat(rows, ignore_index=True)
    (ROOT / "data").mkdir(exist_ok=True)
    log.to_csv(ROOT / "data" / "sales_log.csv", index=False)
    print(f"Saved {len(log):,} rows to data/sales_log.csv")

    sample = pd.DataFrame({
        "product": ["Indomie noodles (carton)", "Garri (paint bucket)", "Bottled water (pack)",
                    "Rice 50kg bag", "Toilet tissue (pack)"],
        "category": ["Groceries", "Groceries", "Beverages", "Groceries", "Household"],
        "current_stock": [12, 60, 25, 20, 40],
        "avg_sales_7d": [5.2, 8.0, 10.5, 1.4, 6.0],
        "avg_sales_30d": [4.9, 7.5, 9.8, 1.5, 5.8],
        "lead_time_days": [3, 2, 2, 5, 3],
        "unit_price": [9000, 2500, 1800, 78000, 2200],
    })
    sample.to_csv(ROOT / "data" / "sample_inventory.csv", index=False)
    print("Saved data/sample_inventory.csv (for the batch upload demo)")


if __name__ == "__main__":
    main()
