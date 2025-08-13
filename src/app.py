from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from pickle import load
import os

app = Flask(__name__)

# Cargar el modelo 
#model = load(open("/workspaces/Web_ML_Flask/models/xg_boost_diabetes_v1.sav", "rb"))

model_path = os.path.join(os.path.dirname(__file__), "models", "xg_boost_diabetes_v1.sav")
model = load(open(model_path, "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        # Obtener valores del formulario y convertirlos a float
        features = [
            float(request.form["Pregnancies"]),
            float(request.form["Glucose"]),
            float(request.form["BloodPressure"]),
            # float(request.form["SkinThickness"]),
            # float(request.form["Insulin"]),
            float(request.form["BMI"]),
            float(request.form["DiabetesPedigreeFunction"]),
            float(request.form["Age"])
        ]
        
        # Convertir a formato para el modelo
        data = np.array([features])
        
        # Hacer predicción
        prediction = model.predict(data)[0]

     
        prediction = "Tiene diabetes" if prediction == 1 else "No tiene diabetes"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)