"""Feature engineering shared by training and prediction.

Keeping this in ONE file guarantees the model sees the same kind of
features when it is trained and when it is used in the app.
"""
import numpy as np
import pandas as pd

# ---- Business rules used to create the "answers" (labels) ----
SAFETY_DAYS = 2     # extra days of cover we want while waiting for delivery
CYCLE_DAYS = 14     # how many days of sales one order should cover
BUFFER = 1.10       # 10% extra to protect against sudden demand

CATEGORICAL = ["category"]
NUMERIC = [
    "current_stock", "avg_sales_7d", "avg_sales_30d", "sales_trend",
    "lead_time_days", "unit_price", "days_of_cover",
    "day_of_week", "month", "is_festive",
]
FEATURES = CATEGORICAL + NUMERIC


def is_festive(dates):
    """1 during the Christmas / New Year rush (10 Dec - 5 Jan), else 0."""
    d = pd.DatetimeIndex(dates)
    flag = ((d.month == 12) & (d.day >= 10)) | ((d.month == 1) & (d.day <= 5))
    return np.asarray(flag).astype(int)


def add_derived_features(df):
    df = df.copy()
    df["sales_trend"] = df["avg_sales_7d"] / df["avg_sales_30d"].clip(lower=0.1)
    df["days_of_cover"] = df["current_stock"] / df["avg_sales_7d"].clip(lower=0.1)
    return df


def build_training_table(log):
    """Turn a daily sales log into one training row per product per day.

    Needed columns: date, product, category, unit_price, lead_time_days,
    units_sold, stock_on_hand
    """
    log = log.copy()
    log["date"] = pd.to_datetime(log["date"])
    log = log.sort_values(["product", "date"]).reset_index(drop=True)

    frames = []
    for _, d in log.groupby("product", sort=False):
        d = d.reset_index(drop=True)
        n = len(d)
        sold = d["units_sold"].astype(float)

        # Features: what the owner could know at the end of today
        d["avg_sales_7d"] = sold.rolling(7).mean()
        d["avg_sales_30d"] = sold.rolling(30).mean()

        # Labels: what ACTUALLY happened in the days after today
        cs = np.concatenate([[0.0], sold.cumsum().to_numpy()])
        i = np.arange(n)
        lead = d["lead_time_days"].to_numpy().astype(int)

        def future_sum(h):
            end = i + h + 1
            out = np.full(n, np.nan)
            ok = end <= n
            out[ok] = cs[end[ok]] - cs[i[ok] + 1]   # sales on days i+1 .. i+h
            return out

        d["demand_lead"] = future_sum(lead + SAFETY_DAYS)
        d["demand_cycle"] = future_sum(lead + CYCLE_DAYS)
        frames.append(d)

    t = pd.concat(frames, ignore_index=True)
    t = t.rename(columns={"stock_on_hand": "current_stock"})
    t["day_of_week"] = t["date"].dt.dayofweek
    t["month"] = t["date"].dt.month
    t["is_festive"] = is_festive(t["date"])
    t = add_derived_features(t)
    t = t.dropna(subset=["avg_sales_7d", "avg_sales_30d", "demand_lead", "demand_cycle"])

    # Label 1: will stock run out before the new delivery (+ safety days)?
    t["reorder_needed"] = (t["current_stock"] < t["demand_lead"]).astype(int)
    # Label 2: how many units to buy to cover the next cycle (+ buffer)
    t["reorder_qty"] = np.ceil(
        np.maximum(0, BUFFER * t["demand_cycle"] - t["current_stock"])
    )
    return t.reset_index(drop=True)


def make_input_features(df, today):
    """Features for new products entered by a user (no history needed)."""
    today = pd.Timestamp(today)
    out = pd.DataFrame({
        "category": df["category"].astype(str),
        "current_stock": df["current_stock"].astype(float),
        "avg_sales_7d": df["avg_sales_7d"].astype(float),
        "avg_sales_30d": df["avg_sales_30d"].astype(float),
        "lead_time_days": df["lead_time_days"].astype(float),
        "unit_price": df["unit_price"].astype(float),
    })
    out["day_of_week"] = today.dayofweek
    out["month"] = today.month
    out["is_festive"] = int(is_festive([today])[0])
    out = add_derived_features(out)
    return out[FEATURES]
