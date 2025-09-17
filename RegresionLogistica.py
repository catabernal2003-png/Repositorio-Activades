# RegresionLogistica.py
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib  

CSV_PATH = "data_abandono.csv"  # Ajusta en caso de que el CSV este en otra ruta

def load_and_prepare(df=None):
    """
    Carga el dataset (si no se provee) y devuelve X, y.
    Espera columnas: Promedio, Asistencia, HorasEstudio, Carrera, Abandona
    """
    if df is None:
        df = pd.read_csv(CSV_PATH)

    # Normalizar nombres de columnas por si acaso
    df = df.rename(columns=lambda s: s.strip())

    # Verifica columnas necesarias
    expected = {"Promedio", "Asistencia", "HorasEstudio", "Carrera", "Abandona"}
    if not expected.issubset(set(df.columns)):
        raise ValueError(f"Faltan columnas. Se requieren: {expected}. Columnas encontradas: {set(df.columns)}")

    X = df[["Promedio", "Asistencia", "HorasEstudio", "Carrera"]].copy()
    y = df["Abandona"].astype(int).copy()  # 0/1

    return X, y

def build_pipeline():
    """
    Construye un pipeline que aplica OneHot a 'Carrera' y StandardScaler a numéricos,
    y luego entrena un LogisticRegression (binario).
    """
    numeric_features = ["Promedio", "Asistencia", "HorasEstudio"]
    categorical_features = ["Carrera"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), categorical_features),
        ],
        remainder="drop",
    )

    clf = LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=500)

    pipe = Pipeline(steps=[("pre", preprocessor), ("clf", clf)])
    return pipe

def train_and_evaluate(test_size=0.2, random_state=42, persist_model=False, model_path="logistic_model.joblib"):
    """
    Entrena el modelo y devuelve: conf_matrix, accuracy, report_dict, report_text
    Si persist_model=True guarda el pipeline con joblib.
    """
    X, y = load_and_prepare()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    report_text = classification_report(y_test, y_pred)

    if persist_model:
        joblib.dump(pipe, model_path)

    return conf_matrix, accuracy, report, report_text, pipe

def plot_confusion_matrix(conf_matrix):
    """
    Devuelve BytesIO con la imagen PNG de la matriz (heatmap).
    """
    plt.figure(figsize=(6,5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Matriz de Confusión")
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=120)
    plt.close()
    buf.seek(0)
    return buf

def predict_single(promedio, asistencia, horas_estudio, carrera, model=None, model_path="logistic_model.joblib"):
    """
    Predice para una única instancia:
    - promedio: float
    - asistencia: float
    - horas_estudio: float
    - carrera: string (categorical)
    Devuelve diccionario: {'probabilidad': p, 'clase': 0/1, 'threshold': 0.5}
    """
    # Construir dataframe de entrada
    X_new = pd.DataFrame([{
        "Promedio": float(promedio),
        "Asistencia": float(asistencia),
        "HorasEstudio": float(horas_estudio),
        "Carrera": str(carrera)
    }])

    # Cargar modelo si no se pasa
    if model is None:
        try:
            model = joblib.load(model_path)
        except Exception:
            # Entrena un modelo rápido si no existe persistido (esto requiere data.csv)
            _, _, _, _, model = train_and_evaluate(persist_model=False)

    prob = model.predict_proba(X_new)[0][1]  # probabilidad de clase '1' (abandona)
    clase = int(prob >= 0.5)  # umbral por defecto 0.5

    return {"probabilidad": float(prob), "clase": int(clase), "threshold": 0.5}
