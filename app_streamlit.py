import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Additional features added: SHAP explainability (if available), classification mode, and model save/download
import io
import joblib
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

DEFAULT_DATA_PATH = "data/student_performance_sample.csv"
FEATURES = ["Hours_Studied", "Attendance", "Past_Score", "Sleep_Hours", "Social_Media_Hours"]
TARGET = "Final_Score"

st.set_page_config(page_title="Student Performance — Sundar", page_icon="🎓", layout="wide")

st.title("🎓 Student Performance Predictions — Sundar Edition")
st.caption("Beautiful • Simple • Reproducible")

with st.sidebar:
    st.header("⚙️ Settings")
    n_estimators = st.slider("RandomForest: n_estimators", 50, 500, 200, step=50)
    max_depth = st.slider("RandomForest: max_depth", 1, 30, 10, step=1)
    st.markdown("---")
    st.write("Load your own CSV with the following columns:")
    st.code(", ".join(FEATURES + [TARGET]))
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

# Data
if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    df = pd.read_csv(DEFAULT_DATA_PATH)

# Basic checks
missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
if missing_cols:
    st.error(f"Missing required columns: {missing_cols}")
    st.stop()

tab_home, tab_explore, tab_train, tab_predict = st.tabs(["🏠 Home", "🔎 Explore", "🧠 Train", "🔮 Predict"])

with tab_home:
    st.subheader("Overview")
    st.write("""
This app predicts a student's **Final_Score (0–100)** from study and lifestyle features.
Try the *Explore* tab to see your data, *Train* to fit a model, and *Predict* for instant scoring.
""")
    st.success("Tip: You can tweak RandomForest hyperparameters from the left sidebar.")

with tab_explore:
    st.subheader("Data Snapshot")
    st.dataframe(df.head(20), use_container_width=True)
    st.subheader("Summary Stats")
    st.dataframe(df.describe(), use_container_width=True)

    st.subheader("Distributions")
    for col in FEATURES + [TARGET]:
        fig = px.histogram(df, x=col, nbins=20)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Heatmap")
    corr = df[FEATURES + [TARGET]].corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto")
    st.plotly_chart(fig, use_container_width=True)

with tab_train:
    st.subheader("Train a Model")
    X = df[FEATURES]
    y = df[TARGET]

    # Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    c1, c2, c3 = st.columns(3)
    c1.metric("R²", f"{r2:.3f}")
    c2.metric("MAE", f"{mae:.2f}")
    c3.metric("RMSE", f"{rmse:.2f}")

    # Feature importances
    importances = getattr(model, "feature_importances_", None)
    if importances is not None:
        imp_df = pd.DataFrame({
            "feature": FEATURES,
            "importance": importances
        }).sort_values("importance", ascending=False)
        st.subheader("Feature Importances")
        fig = px.bar(imp_df, x="feature", y="importance")
        st.plotly_chart(fig, use_container_width=True)

with tab_predict:
    st.subheader("Predict Final Score")
    c1, c2, c3 = st.columns(3)
    hours = c1.number_input("Hours_Studied", 0.0, 12.0, 3.0, step=0.5)
    attendance = c2.number_input("Attendance (%)", 0.0, 100.0, 80.0, step=1.0)
    past = c3.number_input("Past_Score", 0.0, 100.0, 60.0, step=1.0)
    c4, c5 = st.columns(2)
    sleep = c4.number_input("Sleep_Hours", 0.0, 12.0, 7.0, step=0.5)
    social = c5.number_input("Social_Media_Hours", 0.0, 12.0, 2.0, step=0.5)

    # Train a simple model on full data with current settings
    model_full = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model_full.fit(df[FEATURES], df[TARGET])

    if st.button("Predict"):
        row = pd.DataFrame([{
            "Hours_Studied": hours,
            "Attendance": attendance,
            "Past_Score": past,
            "Sleep_Hours": sleep,
            "Social_Media_Hours": social
        }])
        pred = model_full.predict(row)[0]
        st.success(f"Predicted Final Score: **{pred:.1f}/100**")


with tab_predict:
    st.subheader("Predict Final Score")
    st.caption("Toggle to switch between Regression (score) and Classification (Pass/At-Risk)")
    mode = st.radio("Mode", options=["Regression (Score)", "Classification (Pass/At-Risk)"])
    threshold = st.slider("Pass threshold (for classification)", 0.0, 100.0, 50.0, step=1.0) if mode.startswith("Classification") else None

    c1, c2, c3 = st.columns(3)
    hours = c1.number_input("Hours_Studied", 0.0, 12.0, 3.0, step=0.5)
    attendance = c2.number_input("Attendance (%)", 0.0, 100.0, 80.0, step=1.0)
    past = c3.number_input("Past_Score", 0.0, 100.0, 60.0, step=1.0)
    c4, c5 = st.columns(2)
    sleep = c4.number_input("Sleep_Hours", 0.0, 12.0, 7.0, step=0.5)
    social = c5.number_input("Social_Media_Hours", 0.0, 12.0, 2.0, step=0.5)

    # Train a simple model on full data with current settings
    model_full = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model_full.fit(df[FEATURES], df[TARGET])

    if st.button("Predict"):
        row = pd.DataFrame([{
            "Hours_Studied": hours,
            "Attendance": attendance,
            "Past_Score": past,
            "Sleep_Hours": sleep,
            "Social_Media_Hours": social
        }])
        pred = model_full.predict(row)[0]
        if mode.startswith("Classification"):
            label = "Pass" if pred >= threshold else "At-Risk"
            st.success(f"Predicted Final Score: **{pred:.1f}/100** — **{label}** (threshold={threshold})")
        else:
            st.success(f"Predicted Final Score: **{pred:.1f}/100**")

        # Offer model download
        buf = io.BytesIO()
        joblib.dump(model_full, buf)
        buf.seek(0)
        st.download_button("Download trained model (.pkl)", buf, file_name="student_model.pkl", mime="application/octet-stream")

        # SHAP explainability (if available)
        if SHAP_AVAILABLE:
            st.markdown("**Model explanation (SHAP):**")
            explainer = shap.Explainer(model_full.predict, df[FEATURES])
            shap_values = explainer(df[FEATURES].iloc[:50])
            try:
                st.pyplot(shap.plots.bar(shap_values, show=False))
            except Exception:
                st.write("SHAP plot rendering failed in this environment. Install shap and try locally.")
        else:
            st.info("Install 'shap' to enable per-prediction explanations (see requirements.txt)")
