from flask import Flask, render_template, request, send_file
import io
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import RegresionLineal
import RegresionLogistica

app = Flask(__name__)

# Ruta principal
@app.route("/")
def home():
    return render_template("Home.html")

# Ruta para 'inicio' (necesaria para los templates)
@app.route("/inicio")
def inicio():
    return render_template("Home.html")

# Ruta casos de éxito
@app.route("/casos")
def casos_exito():
    return render_template("CasosExito.html")

# Ruta conceptos
@app.route("/concepto")
def concepto_rl():
    return render_template("ConceptRL.html")

# Ruta ejercicio práctico Regresion Lineal
@app.route("/pruebaRL")
def prueba_rl():
    X, R, y = RegresionLineal.get_training_data()
    data_preview = list(zip(X, R, y))
    return render_template("PruebaRL.html", data_preview=data_preview, resultado=None)

# Ruta ejercicio práctico Regresion Logistica
@app.route("/pruebaLogistica")
def prueba_logistica():
    conf_matrix, accuracy, report, report_text = RegresionLogistica.train_and_evaluate()
    return render_template("RegresionLogistica.html",
                           conf_matrix=conf_matrix,
                           accuracy=accuracy,
                           report=report,
                           report_text=report_text)

# Agrega esta ruta adicional para 'precio_vivienda'
@app.route("/precio_vivienda", methods=["POST"])
def precio_vivienda():
    metros = float(request.form["metros"])
    habitaciones = int(request.form["habitaciones"])
    resultado = RegresionLineal.predict_price(metros, habitaciones)
    X, R, y = RegresionLineal.get_training_data()
    data_preview = list(zip(X, R, y))
    return render_template("PruebaRL.html", data_preview=data_preview, resultado=resultado, metros=metros, habitaciones=habitaciones)

# Agrega esta ruta adicional para el gráfico de regresión logística'
@app.route("/regresion_logistica/plot.png")
def plot_logistica():
    conf_matrix, _, _, _ = RegresionLogistica.train_and_evaluate()
    buf = RegresionLogistica.plot_confusion_matrix(conf_matrix)
    return send_file(buf, mimetype="image/png")

# Ruta para el gráfico
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

if __name__ == "__main__":
    app.run(debug=True)