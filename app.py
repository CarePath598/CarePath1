from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

# 🧠 cargar modelo UNA SOLA VEZ
model = YOLO("best.pt")


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
