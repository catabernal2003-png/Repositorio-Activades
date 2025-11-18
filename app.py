from flask import Flask, render_template, request, send_file
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import RegresionLineal
import RegresionLogistica as rl
from SpamClassifier import evaluate, predict_label


app = Flask(__name__)

# --- Optimización: Entrenar los modelos una sola vez al iniciar ---

# 1. Modelo de Clasificación de Spam
print("Entrenando el modelo de clasificación de Spam...")
spam_metrics = evaluate()
print("-> Modelo de Spam entrenado y listo.")

# 2. Modelo de Regresión Logística
print("Entrenando el modelo de Regresión Logística...")
{% extends "base.html" %}

{% block title %}Error en la Aplicación{% endblock %}

{% block content %}
<div class="container mt-5">
  <div class="alert alert-danger text-center" role="alert">
    <h4 class="alert-heading">¡Ha ocurrido un error!</h4>
    <p>{{ message }}</p>
    <hr>
    <p class="mb-0">Por favor, verifica las instrucciones y vuelve a intentarlo.</p>
  </div>
  <div class="text-center mt-4">
    <a href="{{ url_for('inicio') }}" class="btn btn-primary">Volver al Inicio</a>
  </div>
</div>
{% endblock %}
try:
    log_conf_matrix, log_accuracy, log_report, log_report_text, log_model = rl.train_and_evaluate()
    if log_model is None:
        print("-> ADVERTENCIA: No se encontró 'data_abandono.csv'. El módulo de Regresión Logística estará deshabilitado.")
    else:
        print("-> Modelo de Regresión Logística entrenado y listo.")
except Exception as e:
    print(f"-> ERROR al entrenar el modelo de Regresión Logística: {e}")
    log_model = None



# ------------------------
# Rutas principales
# ------------------------
@app.route("/")
@app.route("/inicio")
def inicio():
    return render_template("Home.html")

@app.route("/casos-exito")
def casos_exito():
    return render_template("CasosExito.html")

@app.route("/concepto-rl")
def concepto_rl():
    return render_template("ConceptRL.html")

# ------------------------
# Regresión Lineal
# ------------------------
@app.route("/prueba-rl")
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
@app.route("/prueba-logistica", methods=["GET", "POST"])
def prueba_logistica():
    # Si el modelo no se pudo cargar (porque falta el CSV), muestra una página de error.
    if log_model is None:
        error_msg = "El modelo de Regresión Logística no pudo ser cargado. Asegúrate de que el archivo 'data_abandono.csv' exista en la carpeta principal del proyecto."
        return render_template("error.html", message=error_msg), 500

    prediction = None
    prob = None
    if request.method == "POST":
        promedio = float(request.form["Promedio"])
        asistencia = float(request.form["Asistencia"])
        horas = float(request.form["HorasEstudio"])
        carrera = request.form["Carrera"]
        result = rl.predict_single(promedio, asistencia, horas, carrera, model=log_model)
        prediction = "Sí" if result["clase"] == 1 else "No"
        prob = f"{result['probabilidad']:.4f}"

    cm_path = "/regresion_logistica/plot.png"
    return render_template("RegresionLogistica.html",
                           conf_matrix=log_conf_matrix,
                           accuracy=log_accuracy,
                           report_text=log_report_text,
                           cm_path=cm_path,
                           prediction=prediction,
                           prob=prob)

@app.route("/regresion_logistica/plot.png")
def plot_logistica():
    # Usa la matriz de confusión ya calculada al inicio
    buf = rl.plot_confusion_matrix(log_conf_matrix)
    return send_file(buf, mimetype="image/png")

@app.route("/regresion-logistica/conceptos")
def conceptos_logistica():
    return render_template("ConceptLogistica.html")

# ------------------------
# Algoritmos de Clasificación
# ------------------------
@app.route("/caso-practico-clasificacion", methods=["GET", "POST"])
def caso_practico_clasificacion():
    prediction = None
    if request.method == "POST":
        # Captura las 6 variables en el orden exacto
        vars_order = ["freq_gratis","freq_promocion","freq_urgente",
                      "tiene_link","remitente_conocido","num_adjuntos"]
        features = [float(request.form[v]) for v in vars_order]
        threshold = float(request.form.get("threshold", 0.5))
        label, prob = predict_label(features, threshold)
        prediction = {"label": label, "prob": prob, "threshold": threshold}
    return render_template("caso_practico_clasificacion.html",
                           metrics=spam_metrics,
                           prediction=prediction)

@app.route("/conceptos-clasificacion")
def conceptos_clasificacion():
    return render_template("conceptos_clasificacion.html")


# ------------------------
# Main
# ------------------------
if __name__ == "__main__":
    app.run(debug=True)
