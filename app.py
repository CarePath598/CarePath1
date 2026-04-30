from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
import os
import requests

app = Flask(__name__)
CORS(app)

# 🔥 URL del modelo
MODEL_URL = "http://vj5.294.mytemp.website/best.pt"
MODEL_PATH = "best.pt"


# 🧠 descargar modelo si no existe
def descargar_modelo():
    if not os.path.exists(MODEL_PATH):
        print("⬇ Descargando modelo YOLO...")
        r = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print("✅ Modelo descargado correctamente")


# 🚀 asegurar que el modelo exista ANTES de cargar YOLO
descargar_modelo()

# 🧠 cargar modelo UNA SOLA VEZ
model = YOLO(MODEL_PATH)


@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    # convertir a OpenCV
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # 🧠 YOLO inference
    results = model(frame)

    # validar detección (evita crash si no detecta nada)
    if len(results[0].boxes) == 0:
        return jsonify({
            "error": "No se detectaron objetos"
        })

    cls = int(results[0].boxes.cls[0])

    temp = float(request.form.get("temperatura", 36.5))

    return jsonify({
        "clase": cls,
        "temperatura": temp
    })


if __name__ == "__main__":
    app.run()
