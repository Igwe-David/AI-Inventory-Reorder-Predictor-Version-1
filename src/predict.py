"""Load the trained models and make predictions for new inventory data."""
from datetime import date
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.features import SAFETY_DAYS, make_input_features

ROOT = Path(__file__).resolve().parent.parent
CATEGORIES = ["Groceries", "Beverages", "Provisions", "Household", "Toiletries"]


def load_models():
    clf = joblib.load(ROOT / "models" / "reorder_classifier.joblib")
    reg = joblib.load(ROOT / "models" / "reorder_regressor.joblib")
    return clf, reg


def predict_df(df, threshold=0.4, today=None, models=None):
    """Predict for many products. Returns df with extra result columns.

    threshold: reorder if probability of running out >= this value.
    A lower value is more cautious (fewer stock-outs, more orders).
    """
    clf, reg = models or load_models()
    today = today or date.today()
    X = make_input_features(df, today)

    prob = clf.predict_proba(X)[:, 1]
    reorder = prob >= threshold
    qty = np.where(reorder, np.ceil(np.maximum(1, reg.predict(X))), 0).astype(int)

    rate = X["avg_sales_7d"].clip(lower=0.1)
    out = df.copy()
    out["days_of_stock_left"] = (X["current_stock"] / rate).round(1)
    out["stockout_probability"] = (prob * 100).round(0)
    out["reorder_now"] = np.where(reorder, "YES", "No")
    out["suggested_quantity"] = qty
    out["estimated_cost_ngn"] = (qty * X["unit_price"]).round(0)
    out["risk"] = pd.cut(prob, [-0.01, threshold, 0.75, 1.01],
                         labels=["Low", "Medium", "High"]).astype(str)
    # Days until the stock hits the "must reorder" line
    line = rate * (X["lead_time_days"] + SAFETY_DAYS)
    out["days_until_reorder_point"] = np.maximum(0, (X["current_stock"] - line) / rate).round(1)
    return out
