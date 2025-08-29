# Student Performance Predictions — Sundar Edition

<p align="center">
  <img src="assets/banner.svg" alt="Banner" width="80%"/>
</p>

<p align="center">
  <a href="https://img.shields.io/badge/build-passing-brightgreen"><img src="https://img.shields.io/badge/build-passing-brightgreen" alt="Build Status"></a>
  <a href="https://img.shields.io/badge/license-MIT-blue"><img src="https://img.shields.io/badge/license-MIT-blue" alt="License"></a>
  <a href="#"><img src="https://img.shields.io/badge/PRs-welcome-%23e91e63" alt="PRs Welcome"></a>
</p>

Predict student performance with a clean, modern ML stack and a **beautiful Streamlit app**.  
This edition includes an E2E pipeline, visual analytics, explainable feature importances, and ready-to-deploy CI.

---

## ✨ Features

- 📊 **Interactive dashboard** (Streamlit) with data upload + live predictions
- 🧠 **Scikit‑learn models** (RandomForest by default) + train/evaluate utilities
- 🔍 **Visual insights**: distributions, correlations, and feature importances
- 🧪 **Reproducible pipeline** with `requirements.txt` and a tidy structure
- 🔁 **CI** via GitHub Actions (lint + smoke test)
- 🧾 **Notebook** for quick EDA & model training
- 🔐 **MIT License**

---

## 🚀 Quickstart

```bash
# 1) Clone this repo
git clone <your-fork-url>
cd student-performance-sundar

# 2) Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3) Install dependencies
pip install -r requirements.txt

# 4) Run the app
streamlit run app_streamlit.py
```

The app ships with a small sample dataset in `data/student_performance_sample.csv` so it runs **out of the box**.

---

## 🖥️ App Overview

- **Home**: Project summary + how to use
- **Explore**: Data preview, summary stats, and visualizations
- **Train**: Train a RandomForest model and view metrics
- **Predict**: Enter student attributes and get an instant prediction

---

## 📁 Project Structure

```
student-performance-sundar/
├─ app_streamlit.py
├─ requirements.txt
├─ README.md
├─ LICENSE
├─ assets/
│  └─ banner.svg
├─ data/
│  └─ student_performance_sample.csv
├─ notebooks/
│  └─ EDA_and_Model.ipynb
├─ src/
│  ├─ train.py
│  └─ utils.py
└─ .github/
   └─ workflows/
      └─ ci.yml
```

---

## 🧪 Model & Metrics

Default model: **RandomForestRegressor** to predict continuous **Final_Score** (0–100).  
You can toggle hyperparameters in the app and retrain.

---

## 🧠 Explainability

We provide feature importances from the model out of the box.  
For deeper explainability (e.g., SHAP), add `shap` to requirements and integrate in `app_streamlit.py` (hooks included).

---

## 🛣️ Roadmap

- [ ] Add SHAP-based local explanations
- [ ] Add classification mode (e.g., Pass/At-Risk)
- [ ] Model registry + experiment tracking (Weights & Biases)

---

## 🤝 Contributing

PRs welcome! Please open an issue for any bug/feature request.

---

## 📜 License

This project is licensed under the **MIT License**.
