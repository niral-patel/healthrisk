# 🏥 Health Risk Predictor

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/Scikit--Learn-1.3-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0-green)
![Flask](https://img.shields.io/badge/Flask-3.0-black?logo=flask)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-teal?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.33-red?logo=streamlit)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📌 Problem Statement

Diabetes is one of the most prevalent and costly chronic diseases globally, affecting over 422 million people worldwide. Early identification of high-risk individuals enables timely medical intervention and significantly reduces long-term health complications.

This project builds a **production-grade machine learning system** that predicts diabetes risk from behavioral and clinical survey data — enabling healthcare providers and individuals to make informed, data-driven decisions.

---

## 🎯 Real-World Use Case

- **Healthcare providers** can screen high-risk patients before costly clinical tests
- **Insurance companies** can assess population health risk profiles
- **Public health analysts** can identify behavioral factors driving diabetes prevalence
- **Individuals** can self-assess their risk based on lifestyle factors

---

## 📊 Dataset

| Property | Details |
|---|---|
| Source | [CDC Diabetes Health Indicators — Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) |
| Rows | ~253,000 |
| Features | 22 behavioral & clinical features |
| Target | `Diabetes_binary` (0 = No Diabetes, 1 = Diabetes) |
| Class Imbalance | 6.2 : 1 (No Diabetes : Diabetes) |
| Missing Values | None |

### Feature Categories

| Type | Features |
|---|---|
| **Continuous** | BMI, MentHlth, PhysHlth |
| **Categorical (Ordered)** | GenHlth, Age, Education, Income |
| **Binary (0/1)** | HighBP, HighChol, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, DiffWalk, Sex |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Data Pipeline                      │
│  CDC Dataset → EDA → Feature Engineering →          │
│  Class Balancing (SMOTE + Class Weights)            │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│                 Model Training                       │
│  Logistic Regression (weighted) │ XGBoost (weighted)│
│  Random Forest (balanced)       │ LightGBM (balanced)│
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│              Deployment Layer                        │
│     Flask API  │  FastAPI  │  Streamlit UI           │
└─────────────────────────────────────────────────────┘
```

---

## 🔬 ML Pipeline

### 1. Exploratory Data Analysis
- Confirmed 100% data completeness (zero missing values)
- Identified 6.2:1 class imbalance requiring treatment
- Correlation heatmap to identify top features impacting diabetes risk
- Separated features by data type for appropriate visualization

### 2. Class Imbalance Strategy

| Model | Strategy |
|---|---|
| Logistic Regression | SMOTE + StandardScaler |
| XGBoost | `scale_pos_weight = 6.27` |
| Random Forest | `class_weight = 'balanced'` |
| LightGBM | `is_unbalance = True` |

### 3. Models Trained

| Model | Handling Imbalance |
|---|---|
| Logistic Regression | SMOTE oversampling |
| XGBoost | Class weight ratio |
| Random Forest | Built-in balanced mode |
| LightGBM | Built-in unbalanced mode |

### 4. Evaluation Metrics

Given the class imbalance, accuracy alone is misleading. The following metrics were prioritized:

- **ROC-AUC** — primary metric for imbalanced classification
- **Precision / Recall** — trade-off for medical screening context
- **F1 Score** — harmonic mean for balanced evaluation
- **Confusion Matrix** — false negative analysis (critical in healthcare)

---

## 📁 Project Structure

```
healthrisk/
├── artifacts/                        # Saved models and plots
│   └── diabetes_class_distribution.png
├── notebook/
│   ├── 1_EDA_diabetes.ipynb          # Full EDA notebook
│   └── 2_modeling_diabetes.ipynb     # Modeling notebook
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/
│   │   ├── train_pipeline.py
│   │   └── predict_pipeline.py
│   ├── utils.py
│   ├── logger.py
│   └── exception.py
├── templates/
│   ├── index.html
│   └── result.html
├── app.py                            # Flask API
├── fast_api.py                       # FastAPI endpoint
├── streamlit_app.py                  # Streamlit UI
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/niral-patel/healthrisk.git
cd healthrisk
```

### 2. Create Environment & Install Dependencies
```bash
conda create -p ./env python=3.11 -y
conda activate ./env
pip install -r requirements.txt
```

### 3. Run Flask App
```bash
python app.py
# Visit http://localhost:5000
```

### 4. Run FastAPI
```bash
uvicorn fast_api:app --reload
# Visit http://localhost:8000/docs
```

### 5. Run Streamlit UI
```bash
streamlit run streamlit_app.py
# Visit http://localhost:8501
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| ML | Scikit-learn, XGBoost, LightGBM |
| Imbalance Handling | Imbalanced-learn (SMOTE) |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| APIs | Flask, FastAPI |
| UI | Streamlit |
| Tracking | MLflow |
| Version Control | Git, GitHub |

---

## 📈 Resume Bullet Points

- Built an end-to-end diabetes risk prediction system on 253K CDC survey records using Logistic Regression, XGBoost, Random Forest, and LightGBM with a 6.2:1 class imbalance handled via SMOTE and class weighting strategies
- Deployed prediction pipeline via Flask REST API, FastAPI endpoint, and interactive Streamlit UI for multi-channel accessibility
- Implemented production ML architecture with modular data ingestion, transformation, and model training components following software engineering best practices

---

## 🙋 Author

**Niral Patel**
M.S. Data Science | AI/ML Engineer
[GitHub](https://github.com/niral-patel)
