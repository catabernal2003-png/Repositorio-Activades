from flask import Flask, render_template, request
import RegresionLineal

app = Flask(__name__)

# Ruta principal -> Home.html
@app.route("/")
def home():
    return render_template("Home.html")

# Ruta -> CasosExito.html
@app.route("/casos")
def casos_exito():
    return render_template("CasosExito.html")

# Ruta -> RegresionLineal.html
@app.route("/regresion")
def regresion_lineal():
    return render_template("RegresionLineal.html")

# Ruta -> ConceptRL.html
@app.route("/concepto")
def concepto_rl():
    return render_template("ConceptRL.html")

# Ruta -> PruebaRL.html (formulario de predicción)
@app.route("/prueba")
def prueba_rl():
    return render_template("PruebaRL.html")

# Ruta -> recibe datos del formulario y calcula precio vivienda
@app.route("/PV", methods=["POST"])
def precio_vivienda():
    metros = int(request.form["metros"])
    habitaciones = int(request.form["habitaciones"])
    resultado = RegresionLineal.predict_price(metros, habitaciones)
    return render_template("PruebaRL.html", resultado=resultado)

if __name__ == "__main__":
    app.run(debug=True)
