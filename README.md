# AI Inventory Reorder Predictor (SIWES Project)

Helps small shops in Owerri, Imo State decide **when** to restock a product and **how much** to order.

## 1. The problem and the solution
Many small shop owners restock by feeling. They run out of fast-selling items (lost sales,
unhappy customers) or over-buy slow items (cash locked up). Suppliers are often in Aba, Onitsha
or Port Harcourt, so delivery takes days and demand jumps around weekends and Christmas.

This system uses two machine-learning models:
1. **Classifier** - will this product run out before the new delivery arrives? (Yes/No + probability)
2. **Regressor** - if yes, how many units should be ordered?

## 2. Data needed (one row per product per day)
| Column | Meaning |
|---|---|
| date | day of record |
| product, category | what is sold |
| units_sold | units sold that day |
| stock_on_hand | units in the shop at end of day |
| lead_time_days | days the supplier took to deliver |
| unit_price | cost per unit (NGN) |

## 3. Getting a dataset
- **Real data (best):** ask a shop owner (with permission) for exercise-book records, POS exports,
  WhatsApp order lists or supplier invoices. Type 3-6 months of daily sales for 5-20 products into
  `data/sales_log.csv` using the columns above. Do not record customer names or phone numbers.
- **Synthetic data (included):** `src/generate_data.py` simulates 15 typical products with weekday
  patterns, the December rush, promotions, random supplier delays and noise. State clearly in your
  report that the data is simulated, then retrain on real data when you can.

## 4. Algorithm
**Random Forest** (classifier + regressor): works well on small tables, needs no feature scaling,
handles non-linear patterns, resists overfitting, trains in seconds on a laptop, and gives
feature importance you can explain. It is compared against a simple rule-of-thumb formula.

## 5. Data preparation (`src/features.py`)
1. Sort each product's sales by date.
2. Build features from the past: 7-day and 30-day average sales, sales trend (7d/30d),
   days of cover (stock / daily sales), day of week, month, festive-season flag, lead time, price, category.
3. Build answers from the future: total sales in the next (lead time + 2) days gives
   `reorder_needed`; sales in the next (lead time + 14) days, plus a 10% buffer, minus stock gives `reorder_qty`.
4. One-hot encode category. Split by time (train on the past, test on the latest 20%, with a
   20-day gap) so the test reflects real use.

## 6-7. Training and evaluation
`python -m src.train_model` reports accuracy, precision, recall, F1, ROC-AUC, a confusion matrix,
MAE/RMSE/R2 for quantity, and charts saved in `models/`. **Recall matters most**: a missed reorder
means a stock-out.

## 8-11. Using the system
`streamlit run app.py` opens a web page: enter product details, get a YES/No decision, stockout
probability, suggested quantity and estimated cost. A CSV tab handles many products at once.
The sidebar "caution level" trades fewer stock-outs against less cash tied up.

## Setup (Windows / Mac / Linux, Python 3.9+)
```
python -m venv venv
venv\Scripts\activate          # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
python -m src.generate_data    # creates data/sales_log.csv
python -m src.train_model      # trains and saves models/
streamlit run app.py           # opens the app in your browser
```

## Project structure
```
inventory_reorder_predictor/
  app.py               user interface (Streamlit)
  requirements.txt
  data/                sales_log.csv, sample_inventory.csv
  models/              trained models, metrics.json, charts
  src/
    generate_data.py   synthetic data generator
    features.py        feature engineering + labels (shared)
    train_model.py     training and evaluation
    predict.py         prediction helper used by the app
```

## Limitations (mention these in your report)
- Synthetic data gives optimistic scores; real shop data will be noisier.
- Stock levels in the simulation are random snapshots, not a full inventory simulation.
- Does not model price changes, fuel scarcity, strikes or supplier stock-outs.
- Perishables and very new products with little history need extra care.
- Treat output as decision support; the owner makes the final call.

## Ideas to extend
Add per-product safety stock, a WhatsApp/SMS alert, a database (SQLite) for daily entries,
Prophet/XGBoost comparison, or deploy on Streamlit Community Cloud.
