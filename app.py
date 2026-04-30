from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import os
import requests

app = Flask(__name__)

# 🌐 URL de tu modelo en hosting (CAMBIA ESTO)
MODEL_URL = "http://vj5.294.mytemp.website/best.pt"
MODEL_PATH = "best.pt"

# 📥 descargar modelo si no existe
def descargar_modelo():
    if not os.path.exists(MODEL_PATH):
        print("📥 Descargando modelo desde hosting...")
        r = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print("✅ Modelo descargado correctamente")

# 🧠 cargar modelo después de asegurar descarga
descargar_modelo()
model = YOLO(MODEL_PATH)


def riesgo_ph(ph_class):
    if ph_class == "8":
        return 3
    elif ph_class == "7_8":
        return 2
    elif ph_class == "6_7":
        return 1
    elif ph_class == "5_6":
        return 0
    else:
        return 0


@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    results = model(frame)

    cls = str(int(results[0].boxes.cls[0]))

    temp = float(request.form.get("temperatura", 36.5))

    riesgo = riesgo_ph(cls)

    score = 0

    if temp > 38:
        score += 2
    elif temp > 37.5:
        score += 1

    score += riesgo

    if score >= 3:
        resultado = "Infección probable 🚨"
    elif score == 2:
        resultado = "Riesgo medio ⚠️"
    else:
        resultado = "Normal 👍"

    return jsonify({
        "resultado": resultado,
        "score": score,
        "ph_clase": cls,
        "temperatura": temp
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
