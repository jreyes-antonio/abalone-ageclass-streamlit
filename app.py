import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier

# =============================
# CONFIG
# =============================
APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "abalone" / "abalone.data"

cols = [
    "Sex", "Length", "Diameter", "Height",
    "Whole_weight", "Shucked_weight",
    "Viscera_weight", "Shell_weight", "Rings"
]

st.set_page_config(page_title="Abalone Classifier", layout="wide")
st.title("🦪 Clasificación de Edad de Abalones")

# =============================
# DATA
# =============================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, header=None, names=cols)
    df = df[df["Height"] > 0].copy()

    q1, q2 = df["Rings"].quantile([1/3, 2/3]).values
    df["AgeClass"] = pd.cut(
        df["Rings"],
        bins=[-np.inf, q1, q2, np.inf],
        labels=["Joven", "Adulto", "Viejo"]
    )

    X = df.drop(columns=["Rings", "AgeClass"])
    y = df["AgeClass"]
    return X, y

X, y = load_data()

# =============================
# MODEL (same as notebook)
# =============================
@st.cache_resource
def train_model(X, y):
    num_cols = X.select_dtypes(include=np.number).columns
    cat_cols = ["Sex"]

    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    model = Pipeline([
        ("prep", pre),
        ("rf", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model.fit(Xtr, ytr)
    return model, Xte, yte

model, X_test, y_test = train_model(X, y)

# =============================
# RESULTS
# =============================
st.subheader("📊 Resultados del clasificador")

y_pred = model.predict(X_test)

fig, ax = plt.subplots()
ConfusionMatrixDisplay(
    confusion_matrix(y_test, y_pred),
    display_labels=["Joven", "Adulto", "Viejo"]
).plot(ax=ax)
st.pyplot(fig)

st.text(classification_report(y_test, y_pred))

# =============================
# MANUAL PREDICTION
# =============================
st.subheader("🔮 Probar un dato")

with st.form("predict"):
    sex = st.selectbox("Sex", ["M", "F", "I"])
    length = st.number_input("Length", 0.0, 1.0, 0.55)
    diameter = st.number_input("Diameter", 0.0, 1.0, 0.43)
    height = st.number_input("Height", 0.01, 1.0, 0.15)
    whole = st.number_input("Whole_weight", 0.0, 3.0, 0.85)
    shucked = st.number_input("Shucked_weight", 0.0, 2.0, 0.36)
    viscera = st.number_input("Viscera_weight", 0.0, 1.0, 0.18)
    shell = st.number_input("Shell_weight", 0.0, 2.0, 0.25)

    submit = st.form_submit_button("Predecir")

if submit:
    inp = pd.DataFrame([{
        "Sex": sex,
        "Length": length,
        "Diameter": diameter,
        "Height": height,
        "Whole_weight": whole,
        "Shucked_weight": shucked,
        "Viscera_weight": viscera,
        "Shell_weight": shell
    }])

    pred = model.predict(inp)[0]
    st.success(f"Predicción: **{pred}**")
