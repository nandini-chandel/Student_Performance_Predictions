import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# =============================
# Project Title
# =============================
st.set_page_config(page_title="Student Performance Predictions", layout="centered")
st.title("🎓 Student Performance Predictions")

# =============================
# Dataset Path
# =============================
DATA_FILE = "student-mat.csv"
MODEL_FILE = "model.pkl"

if not os.path.exists(DATA_FILE):
    st.error("❌ Dataset file 'student-mat.csv' not found! Please upload it to your repo.")
    st.stop()
else:
    df = pd.read_csv(DATA_FILE, sep=";")

# Remove romantic column if present
if "romantic" in df.columns:
    df = df.drop(columns=["romantic"])

# =============================
# Train or Load Model
# =============================
if not os.path.exists(MODEL_FILE):
    st.write("⚙️ Training model... (first run only)")

    X = df.drop("G3", axis=1)
    y = df["G3"]

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ])

    model_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_pipeline.fit(X_train, y_train)

    joblib.dump(model_pipeline, MODEL_FILE)
    model = model_pipeline
else:
    model = joblib.load(MODEL_FILE)

# =============================
# Sidebar Dropdown
# =============================
mode = st.sidebar.selectbox("Choose Mode", ["EDA", "Prediction"])

# =============================
# EDA Mode
# =============================
if mode == "EDA":
    st.subheader("📊 Exploratory Data Analysis")

    st.markdown("### Dataset Preview")
    st.dataframe(df.head())

    st.markdown("### Dataset Info")
    st.write(df.describe())

    st.markdown("### Distribution of Final Grade (G3)")
    fig, ax = plt.subplots()
    sns.histplot(df["G3"], kde=True, bins=10, ax=ax)
    st.pyplot(fig)

    st.markdown("### Correlation Heatmap (Numeric Features Only)")
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# =============================
# Prediction Mode
# =============================
elif mode == "Prediction":
    st.subheader("📈 Predict Final Grade (G3)")

    with st.form("input_form"):
        age = st.number_input("Age", min_value=10, max_value=25, value=16)
        Medu = st.selectbox("Mother's education (0-4)", [0,1,2,3,4], index=2)
        Fedu = st.selectbox("Father's education (0-4)", [0,1,2,3,4], index=2)
        traveltime = st.selectbox("Home to school travel time", [1,2,3,4], index=0)
        studytime = st.selectbox("Weekly study time", [1,2,3,4], index=1)
        failures = st.number_input("Number of past class failures", min_value=0, max_value=10, value=0)
        famrel = st.selectbox("Family relationship quality (1-5)", [1,2,3,4,5], index=3)
        freetime = st.selectbox("Free time after school (1-5)", [1,2,3,4,5], index=2)
        goout = st.selectbox("Going out with friends (1-5)", [1,2,3,4,5], index=2)
        Dalc = st.selectbox("Workday alcohol consumption (1-5)", [1,2,3,4,5], index=1)
        Walc = st.selectbox("Weekend alcohol consumption (1-5)", [1,2,3,4,5], index=1)
        health = st.selectbox("Current health status (1-5)", [1,2,3,4,5], index=2)
        absences = st.number_input("Number of school absences", min_value=0, max_value=100, value=3)
        G1 = st.number_input("First period grade (0-20)", min_value=0, max_value=20, value=10)
        G2 = st.number_input("Second period grade (0-20)", min_value=0, max_value=20, value=10)

        school = st.selectbox("School", ['GP', 'MS'])
        sex = st.selectbox("Sex", ['F', 'M'])
        address = st.selectbox("Address", ['U', 'R'])
        famsize = st.selectbox("Family size", ['GT3', 'LE3'])
        Pstatus = st.selectbox("Parent cohabitation status", ['A', 'T'])
        Mjob = st.selectbox("Mother's job", ['at_home', 'health', 'other', 'services', 'teacher'])
        Fjob = st.selectbox("Father's job", ['teacher', 'other', 'services', 'health', 'at_home'])
        reason = st.selectbox("Reason to choose this school", ['course', 'other', 'home', 'reputation'])
        guardian = st.selectbox("Guardian", ['mother', 'father', 'other'])
        schoolsup = st.selectbox("Extra educational support", ['yes', 'no'])
        famsup = st.selectbox("Family educational support", ['no', 'yes'])
        paid = st.selectbox("Extra paid classes", ['no', 'yes'])
        activities = st.selectbox("Extra-curricular activities", ['no', 'yes'])
        nursery = st.selectbox("Attended nursery", ['yes', 'no'])
        higher = st.selectbox("Wants higher education", ['yes', 'no'])
        internet = st.selectbox("Internet access at home", ['no', 'yes'])

        submitted = st.form_submit_button("Predict")

    if submitted:
        data = [[
            age, Medu, Fedu, traveltime, studytime, failures, famrel,
            freetime, goout, Dalc, Walc, health, absences, G1, G2,
            school, sex, address, famsize, Pstatus, Mjob, Fjob, reason,
            guardian, schoolsup, famsup, paid, activities, nursery,
            higher, internet
        ]]

        columns = [
            'age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures',
            'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health',
            'absences', 'G1', 'G2', 'school', 'sex', 'address', 'famsize',
            'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup',
            'famsup', 'paid', 'activities', 'nursery', 'higher',
            'internet'
        ]

        input_df = pd.DataFrame(data, columns=columns)
        pred = model.predict(input_df)[0]

        st.metric(label="Predicted final grade (G3 out of 20)", value=f"{pred:.2f}")
        st.write("Note: This prediction is approximate and model-dependent.")
