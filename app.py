from flask import Flask
from flask import render_template
app = Flask(__name__)

@app.route("/")
def home():
    myname = "Supervisado"
    return f"""
        <h1>Casos de Uso Machine Learning {myname}!</h1>
        <a href='/index' class='button-link'>Ir a la investigación</a>
        <link rel="stylesheet" href="/static/style.css">
    """


@app.route("/index")
def index():
     myname= "Flask"
     return render_template("index.html",name=myname)