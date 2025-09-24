import pandas as pd
from sklearn.model_selection import train_test_split
from  sklearn.naive_bayesimport MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

RANDOM_STATE = 42
MODEL_PATH = "spam_nb_model.pkl"
CONF_IMG = "static/confusion_spam.png"

def load_data():
    # === Datos de ejemplo ===
    data = {
        "freq_gratis":     [3,0,1,0,2,0,1,4],
        "freq_promocion":  [1,0,0,0,2,1,0,3],
        "freq_urgente":    [0,0,1,0,1,0,0,2],
        "tiene_link":      [1,0,1,0,1,0,1,1],
        "remitente_conocido":[0,1,0,1,0,1,0,0],
        "num_adjuntos":    [0,1,0,0,2,0,0,1],
        "spam":            [1,0,1,0,1,0,1,1]
    }
    df = pd.DataFrame(data)
    X = df.drop("spam", axis=1)
    y = df["spam"]
    return X, y

def evaluate():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = round(accuracy_score(y_test, y_pred), 4)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    # Guardar matriz de confusión
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No spam","Spam"],
                yticklabels=["No spam","Spam"])
    plt.xlabel("Predicho"); plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(CONF_IMG)
    plt.close()

    joblib.dump(model, MODEL_PATH)
    return {"accuracy": acc, "report": report, "conf_matrix": cm.tolist()}

def predict_label(features, threshold=0.5):
    model = joblib.load(MODEL_PATH)
    prob_spam = model.predict_proba([features])[0][1]
    label = "Sí" if prob_spam >= threshold else "No"
    return label, round(prob_spam, 4)
