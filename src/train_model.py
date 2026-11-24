"""Train and evaluate the two models.

Run:  python -m src.train_model

Model 1 (classifier): Should we reorder?      -> Random Forest Classifier
Model 2 (regressor):  How many units to order? -> Random Forest Regressor
"""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.features import (BUFFER, CATEGORICAL, CYCLE_DAYS, FEATURES,
                          SAFETY_DAYS, build_training_table)

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)


def make_pipeline(model):
    prep = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
        remainder="passthrough",
    )
    return Pipeline([("prep", prep), ("model", model)])


def main():
    csv = ROOT / "data" / "sales_log.csv"
    if not csv.exists():
        from src.generate_data import main as generate
        generate()
    table = build_training_table(pd.read_csv(csv))
    print(f"Training table: {len(table):,} rows, "
          f"{table['reorder_needed'].mean():.1%} need a reorder")

    # ---- Time-based split: train on the past, test on the most recent 20% ----
    dates = np.sort(table["date"].unique())
    cutoff = pd.Timestamp(dates[int(len(dates) * 0.8)])
    gap = pd.Timedelta(days=20)  # labels look ~20 days ahead, so leave a gap
    train = table[table["date"] < cutoff - gap]
    test = table[table["date"] >= cutoff]
    print(f"Train: {len(train):,} rows | Test: {len(test):,} rows (from {cutoff.date()})")

    # ---- Model 1: reorder or not ----
    clf = make_pipeline(RandomForestClassifier(
        n_estimators=200, min_samples_leaf=3, class_weight="balanced",
        random_state=42, n_jobs=-1))
    clf.fit(train[FEATURES], train["reorder_needed"])
    proba = clf.predict_proba(test[FEATURES])[:, 1]
    pred = (proba >= 0.5).astype(int)

    # Simple rule-of-thumb baseline owners often use: reorder if stock will not
    # last the lead time + safety days at the last-7-day selling rate.
    base_pred = (test["current_stock"] <= test["avg_sales_7d"] * (test["lead_time_days"] + SAFETY_DAYS)).astype(int)

    def cls_metrics(y, p):
        return {"accuracy": accuracy_score(y, p), "precision": precision_score(y, p),
                "recall": recall_score(y, p), "f1": f1_score(y, p)}

    y = test["reorder_needed"]
    cls_results = {"random_forest": cls_metrics(y, pred),
                   "rule_of_thumb_baseline": cls_metrics(y, base_pred)}
    cls_results["random_forest"]["roc_auc"] = roc_auc_score(y, proba)
    cm = confusion_matrix(y, pred)

    # ---- Model 2: reorder quantity (only learned from rows that needed a reorder) ----
    tr_need = train[train["reorder_needed"] == 1]
    te_need = test[test["reorder_needed"] == 1]
    reg = make_pipeline(RandomForestRegressor(
        n_estimators=200, min_samples_leaf=3, random_state=42, n_jobs=-1))
    reg.fit(tr_need[FEATURES], tr_need["reorder_qty"])
    qty_pred = np.maximum(0, reg.predict(te_need[FEATURES]))
    base_qty = np.maximum(0, np.ceil(
        BUFFER * te_need["avg_sales_7d"] * (te_need["lead_time_days"] + CYCLE_DAYS)
        - te_need["current_stock"]))

    def reg_metrics(y_true, p):
        return {"MAE_units": mean_absolute_error(y_true, p),
                "RMSE_units": float(np.sqrt(mean_squared_error(y_true, p))),
                "R2": r2_score(y_true, p)}

    reg_results = {"random_forest": reg_metrics(te_need["reorder_qty"], qty_pred),
                   "formula_baseline": reg_metrics(te_need["reorder_qty"], base_qty)}

    metrics = {"classification": cls_results, "regression": reg_results,
               "confusion_matrix": cm.tolist(),
               "rows": {"train": len(train), "test": len(test)}}
    (MODELS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    joblib.dump(clf, MODELS / "reorder_classifier.joblib")
    joblib.dump(reg, MODELS / "reorder_regressor.joblib")

    # ---- Charts for your report ----
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for (r, c), v in np.ndenumerate(cm):
        ax.text(c, r, v, ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1], ["No reorder", "Reorder"])
    ax.set_yticks([0, 1], ["No reorder", "Reorder"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion matrix")
    fig.tight_layout(); fig.savefig(MODELS / "confusion_matrix.png", dpi=150); plt.close(fig)

    names = [n.split("__")[-1] for n in clf.named_steps["prep"].get_feature_names_out()]
    imp = pd.Series(clf.named_steps["model"].feature_importances_, index=names).sort_values()
    fig, ax = plt.subplots(figsize=(6, 4))
    imp.tail(10).plot.barh(ax=ax)
    ax.set_title("What the model relies on most"); fig.tight_layout()
    fig.savefig(MODELS / "feature_importance.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(te_need["reorder_qty"], qty_pred, s=6, alpha=0.4)
    top = max(te_need["reorder_qty"].max(), qty_pred.max())
    ax.plot([0, top], [0, top], "r--")
    ax.set_xlabel("Actual quantity needed"); ax.set_ylabel("Predicted quantity")
    ax.set_title("Reorder quantity: actual vs predicted"); fig.tight_layout()
    fig.savefig(MODELS / "quantity_actual_vs_predicted.png", dpi=150); plt.close(fig)

    print("\n=== Should we reorder? (test set) ===")
    print(pd.DataFrame(cls_results).T.round(3))
    print("Confusion matrix [[TN FP],[FN TP]]:\n", cm)
    print("\n=== How many to order? (test set) ===")
    print(pd.DataFrame(reg_results).T.round(3))
    print("\nModels and charts saved in the models/ folder.")


if __name__ == "__main__":
    main()
