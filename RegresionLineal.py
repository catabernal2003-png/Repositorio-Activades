import pandas as pd
from sklearn.linear_model import LinearRegression

# Datos de ejemplo para entrenar el modelo
data = {
    "Metros2": [50, 80, 60, 100, 120, 70, 90],
    "Habitaciones": [2, 3, 2, 4, 4, 3, 3],
    "Precio": [150000, 250000, 180000, 320000, 400000, 210000, 280000]
}

df = pd.DataFrame(data)

X = df[["Metros2", "Habitaciones"]]
y = df["Precio"]

model = LinearRegression()
model.fit(X, y)

def predict_price(metros, habitaciones):
    """Recibe m² y habitaciones, retorna el precio estimado."""
    result = model.predict([[metros, habitaciones]])[0]
    return result
