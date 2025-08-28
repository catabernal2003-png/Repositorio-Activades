from flask import Flask
from flask import render_template
app = Flask(__name__)

@app.route("/")
def home():
    return f"Hello Flask"

@app.route("/index")
def index():
    Myname="flask"
    return render_template ("index.html")






