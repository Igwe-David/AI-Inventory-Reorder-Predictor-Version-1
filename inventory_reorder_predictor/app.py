import json
from pathlib import Path
import pandas as pd
import streamlit as st
from src.predict import CATEGORIES, load_models, predict_df

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Inventory Reorder Predictor", page_icon="📦", layout="wide")


@st.cache_resource
def get_models():
    return load_models()


st.title("📦 AI Inventory Reorder Predictor")
st.caption("Helps small businesses in Owerri know WHEN to restock and HOW MUCH to order.")

if not (ROOT / "models" / "reorder_classifier.joblib").exists():
    st.error("Model not found. Run `python -m src.generate_data` then `python -m src.train_model` first.")
    st.stop()
models = get_models()

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Caution level", 0.2, 0.8, 0.4, 0.05,
                          help="Lower = safer (orders earlier, fewer stock-outs). "
                               "Higher = orders later (less cash tied up).")
    st.write("Reorder is advised when the chance of running out before the "
             f"delivery arrives is **{threshold:.0%}** or more.")

tab1, tab2, tab3 = st.tabs(["Single product", "Many products (CSV)", "Model performance"])

# ---------- Tab 1: one product ----------
with tab1:
    with st.form("single"):
        c1, c2 = st.columns(2)
        name = c1.text_input("Product name", "Indomie noodles (carton)")
        category = c2.selectbox("Category", CATEGORIES)
        c3, c4, c5 = st.columns(3)
        stock = c3.number_input("Current stock (units)", 0, 100000, 12)
        s7 = c4.number_input("Avg. units sold per day (last 7 days)", 0.0, 10000.0, 5.2, 0.1)
        s30 = c5.number_input("Avg. units sold per day (last 30 days)", 0.0, 10000.0, 4.9, 0.1)
        c6, c7 = st.columns(2)
        lead = c6.number_input("Supplier delivery time (days)", 1, 60, 3)
        price = c7.number_input("Cost per unit (₦)", 0, 100000000, 9000, 100)
        go = st.form_submit_button("Predict", type="primary")

    if go:
        row = pd.DataFrame([{"product": name, "category": category, "current_stock": stock,
                             "avg_sales_7d": s7, "avg_sales_30d": s30,
                             "lead_time_days": lead, "unit_price": price}])
        r = predict_df(row, threshold, models=models).iloc[0]
        if r["reorder_now"] == "YES":
            st.error(f"🔴 REORDER NOW — order about **{int(r['suggested_quantity'])} units** of {name}")
        else:
            st.success(f"🟢 No need to reorder yet. Check again in about "
                       f"{r['days_until_reorder_point']:.0f} day(s).")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Days of stock left", r["days_of_stock_left"])
        m2.metric("Chance of running out", f"{r['stockout_probability']:.0f}%")
        m3.metric("Suggested quantity", int(r["suggested_quantity"]))
        m4.metric("Estimated cost", f"₦{r['estimated_cost_ngn']:,.0f}")
        st.caption(f"Risk level: {r['risk']}. Delivery takes {lead} day(s), so your stock must "
                   "last until then.")

# ---------- Tab 2: many products ----------
with tab2:
    st.write("Upload a CSV with columns: `product, category, current_stock, avg_sales_7d, "
             "avg_sales_30d, lead_time_days, unit_price`.")
    sample = ROOT / "data" / "sample_inventory.csv"
    if sample.exists():
        st.download_button("Download sample CSV", sample.read_bytes(), "sample_inventory.csv")
    up = st.file_uploader("Your inventory file", type="csv")
    if up:
        df = pd.read_csv(up)
        need = {"product", "category", "current_stock", "avg_sales_7d",
                "avg_sales_30d", "lead_time_days", "unit_price"}
        if not need.issubset(df.columns):
            st.error(f"Missing columns: {', '.join(sorted(need - set(df.columns)))}")
        else:
            res = predict_df(df, threshold, models=models).sort_values(
                "stockout_probability", ascending=False)
            st.dataframe(res, use_container_width=True, hide_index=True)
            st.download_button("Download results", res.to_csv(index=False).encode(),
                               "reorder_results.csv")

# ---------- Tab 3: performance ----------
with tab3:
    mp = ROOT / "models" / "metrics.json"
    if mp.exists():
        m = json.loads(mp.read_text())
        st.subheader("Should we reorder? (classification)")
        st.dataframe(pd.DataFrame(m["classification"]).T.round(3))
        st.subheader("How many units? (regression)")
        st.dataframe(pd.DataFrame(m["regression"]).T.round(3))
        a, b, c = st.columns(3)
        for col, f in zip((a, b, c), ("confusion_matrix", "feature_importance",
                                      "quantity_actual_vs_predicted")):
            p = ROOT / "models" / f"{f}.png"
            if p.exists():
                col.image(str(p))
