from flask import Flask, render_template, request, send_file
import io
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import RegresionLineal
import RegresionLogistica as rl
from flask import Flask, request, render_template
from SpamClassifier import evaluate, predict_label


app = Flask(__name__)

# ------------------------
# Rutas principales
# ------------------------
@app.route("/")
def home():
    return render_template("Home.html")

@app.route("/inicio")
def inicio():
    return render_template("Home.html")

@app.route("/casos")
def casos_exito():
    return render_template("CasosExito.html")

@app.route("/concepto")
def concepto_rl():
    return render_template("ConceptRL.html")

# ------------------------
# Regresión Lineal
# ------------------------
@app.route("/pruebaRL")
def prueba_rl():
    X, R, y = RegresionLineal.get_training_data()
    data_preview = list(zip(X, R, y))
    return render_template("PruebaRL.html", data_preview=data_preview, resultado=None)

@app.route("/precio_vivienda", methods=["POST"])
def precio_vivienda():
    metros = float(request.form["metros"])
    habitaciones = int(request.form["habitaciones"])
    resultado = RegresionLineal.predict_price(metros, habitaciones)
    X, R, y = RegresionLineal.get_training_data()
    data_preview = list(zip(X, R, y))
    return render_template("PruebaRL.html",
                           data_preview=data_preview,
                           resultado=resultado,
                           metros=metros,
                           habitaciones=habitaciones)

@app.route("/regresion/plot.png")
def plot_regresion():
    X, R, y = RegresionLineal.get_training_data()
    X = np.array(X)
    R = np.array(R)
    y = np.array(y)
    
    rooms_fixed = int(round(R.mean()))
    xs = np.linspace(X.min(), X.max(), 200)
    X_pred = np.column_stack((xs, np.full(xs.shape, rooms_fixed)))
    ys = RegresionLineal.model.predict(X_pred)

    fig, ax = plt.subplots(figsize=(6,4))
    sizes = 40 + (R - R.min()) * 30
    ax.scatter(X, y, s=sizes, alpha=0.8, label="Datos (tamaño vs precio)")
    ax.plot(xs, ys, linewidth=2, label=f"Predicción (habitaciones={rooms_fixed})", color="red")
    ax.set_xlabel("Metros²")
    ax.set_ylabel("Precio")
    ax.set_title("Datos de entrenamiento y línea de regresión")
    ax.grid(True)
    ax.legend()

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype="image/png")

# ------------------------
# Regresión Logística
# ------------------------
@app.route("/pruebaLogistica", methods=["GET", "POST"])
def prueba_logistica():
    conf_matrix, accuracy, report, report_text, model = rl.train_and_evaluate()

    prediction = None
    prob = None
    if request.method == "POST":
        promedio = float(request.form["Promedio"])
        asistencia = float(request.form["Asistencia"])
        horas = float(request.form["HorasEstudio"])
        carrera = request.form["Carrera"]

        result = rl.predict_single(promedio, asistencia, horas, carrera, model=model)
        prediction = "Sí" if result["clase"] == 1 else "No"
        prob = f"{result['probabilidad']:.4f}"

    cm_path = "/regresion_logistica/plot.png"
    return render_template("RegresionLogistica.html",
                           conf_matrix=conf_matrix,
                           accuracy=accuracy,
                           report_text=report_text,
                           cm_path=cm_path,
                           prediction=prediction,
                           prob=prob)

@app.route("/regresion_logistica/plot.png")
def plot_logistica():
    conf_matrix, _, _, _, _ = rl.train_and_evaluate()
    buf = rl.plot_confusion_matrix(conf_matrix)
    return send_file(buf, mimetype="image/png")

# Ruta conceptos de logística
@app.route("/regresion-logistica/conceptos")
def conceptos_logistica():
    return render_template("ConceptLogistica.html")

# Entrenamiento
@app.route("/tipos-clasificacion/conceptos")
def clasific_conceptos():
    # Página con el enlace/iframe a tu mapa MindMeister
    return render_template("clasific_basicos.html")

# --- Optimización: Entrenar el modelo de Spam una sola vez al iniciar ---
# Se llama a evaluate() para entrenar el modelo y generar la imagen de la matriz.
# Las métricas se guardan en una variable para reutilizarlas en cada petición.
print("Entrenando el modelo de clasificación de Spam al iniciar la app...")
spam_metrics = evaluate()
print("Modelo de Spam entrenado y listo.")

@app.route("/tipos-clasificacion/caso", methods=["GET", "POST"])
def clasific_caso():
    prediction = None
    if request.method == "POST":
        # Captura las 6 variables en el orden exacto
        vars_order = ["freq_gratis","freq_promocion","freq_urgente",
                      "tiene_link","remitente_conocido","num_adjuntos"]
        features = [float(request.form[v]) for v in vars_order]
        threshold = float(request.form.get("threshold", 0.5))
        label, prob = predict_label(features, threshold)
        prediction = {"label": label, "prob": prob, "threshold": threshold}

    return render_template("clasific_caso.html",
                           metrics=spam_metrics,
                           prediction=prediction)

# ------------------------
# Algoritmos de Clasificación
# ------------------------
@app.route("/conceptos-clasificacion")
def conceptos_clasificacion():
    return render_template("conceptos_clasificacion.html")

@app.route("/caso-practico-clasificacion")
def caso_practico_clasificacion():
   
    spam_metrics = {"accuracy": 0.95}
    prediction = None
    if request.method == "POST":
        vars_order = ["freq_gratis","freq_promocion","freq_urgente",
                      "tiene_link","remitente_conocido","num_adjuntos"]
        features = [float(request.form[v]) for v in vars_order]
        threshold = float(request.form.get("threshold", 0.5))
        label, prob = predict_label(features, threshold)
        prediction = {"label": label, "prob": prob, "threshold": threshold}
    return render_template("caso_practico_clasificacion.html",
                           metrics=spam_metrics,
                           prediction=prediction)

# ------------------------
# Main
# ------------------------
if __name__ == "__main__":
    app.run(debug=True)
